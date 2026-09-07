# train_llm/train_hydravit.py
# Orchestrator for HydraViT training; exports run_training_from_cfg for train_llm.py.
from typing import Dict
import os
import random
from datetime import datetime
import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from common_helpers import (
    pick_best_attn_impl,
)
from model_implementations import (
    load_tokenizer_with_fallback,
    load_model,
)
from .train_dataset import load_train_eval_datasets_from_cfg
from .ema_utils import (
    EMAModel,
    PerplexityCallback,
    KHeadsLoggingCallback,
    EMASaveBothNoSwapCallback,
    resume_ema_shadow_from_checkpoint,
)
from .trainer_hydravit import HydraTrainer
from .arguments import build_training_arguments
from common_train_helpers import (
    setup_training_output_dir_and_config,
    start_timer,
    stop_timer,
)

logger = hf_logging.get_logger(__name__)


def _weights_tied(model) -> bool:
    try:
        inp = model.get_input_embeddings()
        outp = model.get_output_embeddings()
        return inp.weight.data_ptr() == outp.weight.data_ptr()
    except Exception:
        return False


def run_training_from_cfg(cfg: Dict):
    """
    Orchestrates training for HydraViT.
    Notes for future extensions (train_matformer.py already follows this pattern):
    - import shared components from train_dataset.py, ema_utils.py, arguments.py
    - implement or import a method-specific Trainer subclass to select and pass
      dynamic widths (e.g., current_num_heads, current_embed_ratio, current_ffn_ratio)
    - call a run_training_from_cfg(cfg) orchestrator similar to this one
    """
    final_output_dir = setup_training_output_dir_and_config(cfg)
    start_timer(final_output_dir)
    try:
        tr = cfg["training"]
        model_save_subfolder = tr.get("model_save_subfolder", "model")
        model_save_dir = os.path.join(final_output_dir, model_save_subfolder)

        implementation = tr.get("implementation", "hydra_gemma_from_flex")
        model_name_or_path = tr["model_name_or_path"]
        tokenizer_name_or_path = tr.get("tokenizer_name_or_path")
        seq_len = int(tr["seq_len"])
        num_samples = int(tr["num_samples"])
        per_device_train_batch_size = int(tr["per_device_train_batch_size"])
        gradient_accumulation_steps = int(tr["gradient_accumulation_steps"])
        learning_rate = float(tr["learning_rate"])
        weight_decay = float(tr.get("weight_decay", 0.0))
        lr_scheduler_type = tr.get("lr_scheduler_type", "cosine")
        warmup_ratio = tr.get("warmup_ratio", None)
        warmup_steps = tr.get("warmup_steps", None)
        max_steps = int(tr["max_steps"])
        logging_steps = int(tr["logging_steps"])
        save_steps = int(tr["save_steps"])
        save_total_limit = int(tr["save_total_limit"])
        fp16 = bool(tr.get("fp16", False))
        bf16 = bool(tr.get("bf16", False)) and not fp16
        deepspeed = tr.get("deepspeed_config") or tr.get("deepspeed")
        heads = tr.get("heads") or []
        head_weights = tr.get("head_weights")
        seed = int(tr.get("seed", 42))
        attn_implementation = tr.get("attn_implementation", "auto")
        gradient_checkpointing = bool(tr.get("gradient_checkpointing", False))
        report_to = tr.get("report_to", ["tensorboard"])

        configured_logging_dir_value = tr.get("logging_dir")
        default_logging_path_if_none = os.path.join(final_output_dir, "logging")
        if configured_logging_dir_value is None:
            logging_dir = default_logging_path_if_none
        else:
            logging_dir = tr.get("logging_dir", default_logging_path_if_none)

        exclude_wd_norms_embeddings = bool(tr.get("exclude_wd_norms_embeddings", True))

        # loss config
        loss_cfg = tr.get("loss", {}) or {}
        loss_type = loss_cfg.get("type", "ce_plus_z")
        z_loss_alpha = float(loss_cfg.get("z_loss_alpha", 0.0))

        # KD teacher config
        teacher_cfg = tr.get("teacher", {}) or {}
        teacher_enabled = bool(teacher_cfg.get("enabled", False))
        teacher_compile = bool(teacher_cfg.get("compile", True))
        teacher_model_path = teacher_cfg.get(
            "model_name_or_path", "google/gemma-3-270m"
        )
        teacher_impl = (
            teacher_cfg.get("implementation", "auto_model") or "auto_model"
        ).lower()
        teacher_attn_impl = teacher_cfg.get("attn_implementation", "auto")
        teacher_dtype_str = teacher_cfg.get("dtype", "auto")
        kd_temperature = float(teacher_cfg.get("temperature", 2.0))
        kd_weight = float(teacher_cfg.get("weight", 0.0))
        kd_every_n_steps = int(teacher_cfg.get("kd_every_n_steps", 1))

        # EMA
        ema_cfg = tr.get("ema", {}) or {}
        ema_enabled = bool(ema_cfg.get("enabled", False))
        ema_decay = float(ema_cfg.get("decay", 0.9999))

        # Grad clipping
        max_grad_norm_enabled = bool(tr.get("max_grad_norm_enabled", True))
        max_grad_norm = float(tr.get("max_grad_norm", 1.0))

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        set_seed(seed)

        tok = load_tokenizer_with_fallback(
            model_name_or_path, tokenizer_name_or_path, use_fast=True
        )
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        amp_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
        device_hint = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Load student model
        model = load_model(
            model_name_or_path,
            implementation=implementation,
            dtype=amp_dtype,
            attn_implementation=attn_implementation,
            device=device_hint,
        )
        model.config.use_cache = False
        model.config.eos_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.pad_token_id

        print(type(model), getattr(model.config, "_attn_implementation", None))

        if gradient_checkpointing:
            logger.info("Enabling gradient checkpointing.")
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                if hasattr(model, "model"):
                    setattr(model.model, "gradient_checkpointing", True)
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        else:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                if hasattr(model, "model"):
                    setattr(model.model, "gradient_checkpointing", False)

        # Resize token embeddings if vocab sizes differ (student)
        model_vocab_now = model.get_output_embeddings().weight.size(0)
        if model_vocab_now != len(tok):
            logger.warning(
                f"Student vocab mismatch (model={model_vocab_now}, tok={len(tok)}). "
                f"Resizing student embeddings/head to {len(tok)} (pad_to_multiple_of=64)."
            )
            model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)

        # Assert weight tying for student
        if not _weights_tied(model):
            try:
                model.tie_weights()
            except Exception:
                pass
        assert _weights_tied(
            model
        ), "Weight tying failed: input embeddings and lm_head must be tied (student)."

        # Prepare TEACHER (frozen)
        teacher_model = None
        teacher_full_heads = None
        if teacher_enabled:
            logger.info("Loading teacher model for KD.")
            try:
                teacher_model = load_model(
                    teacher_model_path,
                    implementation=teacher_impl,
                    dtype=amp_dtype,
                    attn_implementation=teacher_attn_impl,
                    device=device_hint,
                )
            except TypeError:
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    teacher_model_path,
                    torch_dtype=(amp_dtype or torch.float32),
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            try:
                resolved = pick_best_attn_impl(teacher_attn_impl, device=device_hint)
                teacher_model.set_attn_implementation(resolved)
            except Exception:
                pass

            teacher_model.eval()
            if teacher_compile:
                try:
                    teacher_model = torch.compile(
                        teacher_model, mode="max-autotune", dynamic=True
                    )
                    logger.info("Teacher Model compiled with torch.compile.")
                except Exception as e:
                    logger.warning(f"torch.compile failed for teacher model: {e}")
            else:
                logger.info("Teacher Model compilation disabled.")

            for p in teacher_model.parameters():
                p.requires_grad = False

            # Resize TEACHER to tokenizer vocab if needed
            teacher_vocab_now = teacher_model.get_output_embeddings().weight.size(0)
            target_vocab = len(tok)
            if teacher_vocab_now != target_vocab:
                logger.warning(
                    f"Teacher vocab mismatch (model={teacher_vocab_now}, tok={target_vocab}). "
                    f"Resizing teacher embeddings/head to {target_vocab} (pad_to_multiple_of=64)."
                )
                teacher_model.resize_token_embeddings(
                    target_vocab, pad_to_multiple_of=64
                )

            # Tie teacher
            try:
                if not _weights_tied(teacher_model):
                    teacher_model.tie_weights()
            except Exception:
                pass

            try:
                teacher_full_heads = int(teacher_model.config.num_attention_heads)
            except Exception:
                teacher_full_heads = None

            logger.info(f"Teacher loaded. Full heads = {teacher_full_heads}")

        # ------------------------- Dataset Loading and Preparation ------------------------- #
        logger.info("Loading training/eval datasets...")
        train_dataset, eval_dataset = load_train_eval_datasets_from_cfg(
            cfg,
            tokenizer_name_or_path=(tokenizer_name_or_path or model_name_or_path),
        )

        collator = DataCollatorForLanguageModeling(tok, mlm=False)

        model_heads = int(getattr(model.config, "num_attention_heads", 1))
        if not heads:
            heads = list(range(1, model_heads + 1))
        else:
            heads = [max(1, min(int(h), model_heads)) for h in heads]

        if head_weights is not None:
            if len(head_weights) != len(heads):
                raise ValueError("--head_weights must match the length of --heads")
            s = sum(head_weights)
            if s <= 0:
                raise ValueError("--head_weights must sum to a positive value")
            head_weights = [w / s for w in head_weights]

        if bf16 and not torch.cuda.is_available():
            logger.warning("bf16 requested but CUDA not available; running in fp32.")
            bf16 = False
        if fp16 and not torch.cuda.is_available():
            logger.warning("fp16 requested but CUDA not available; running in fp32.")
            fp16 = False

        # Build TrainingArguments robustly
        ta_kwargs = dict(
            output_dir=model_save_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=(
                None
                if warmup_steps is not None
                else (warmup_ratio if warmup_ratio is not None else 0.01)
            ),
            warmup_steps=warmup_steps if warmup_steps is not None else 0,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to if report_to else ["none"],
            logging_dir=logging_dir,
            deepspeed=deepspeed,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            fp16=fp16,
            bf16=bf16,
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=int(tr.get("dataloader_num_workers", 0)),
            dataloader_persistent_workers=(
                bool(tr.get("dataloader_persistent_workers", False))
                if int(tr.get("dataloader_num_workers", 0)) > 0
                else False
            ),
            dataloader_prefetch_factor=(
                int(tr.get("dataloader_prefetch_factor", 2))
                if int(tr.get("dataloader_num_workers", 0)) > 0
                else None
            ),
            max_grad_norm=(max_grad_norm if max_grad_norm_enabled else 0.0),
            _eval_enabled=bool(tr.get("evaluation", {}).get("enabled", False)),
            _eval_steps=int(tr.get("evaluation", {}).get("eval_steps", 1000)),
        )

        targs = build_training_arguments(ta_kwargs)

        trainer = HydraTrainer(
            model=model,
            args=targs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            heads=heads,
            head_weights=head_weights,
            loss_type=loss_type,
            z_loss_alpha=z_loss_alpha,
            teacher_model=teacher_model,
            teacher_full_heads=teacher_full_heads,
            kd_temperature=kd_temperature,
            kd_weight=kd_weight,
            tokenizer=tok,
            exclude_wd_norms_embeddings=exclude_wd_norms_embeddings,
            kd_every_n_steps=kd_every_n_steps,
        )
        ema = None
        # EMA: save BOTH raw (as before) and EMA (to ema_model/)
        if ema_enabled:
            logger.info(f"Enabling EMA (save raw+ema) with decay={ema_decay}")
            ema = EMAModel(model.parameters(), decay=ema_decay)
            # trainer.add_callback(EMASaveBothCallback(ema))
            trainer.add_callback(EMASaveBothNoSwapCallback(ema))
            # from train_llm.ema_verify import VerifyBothRawAndEMASavedCallback
            # trainer.add_callback(VerifyBothRawAndEMASavedCallback(ema))
        # Perplexity and per-k logging callbacks
        if eval_dataset is not None:
            trainer.add_callback(PerplexityCallback())

        # Always add per-k heads logging
        trainer.add_callback(KHeadsLoggingCallback(trainer))

        if trainer.is_world_process_zero():
            os.makedirs(model_save_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        last_ckpt = get_last_checkpoint(model_save_dir)

        # ---- Restore EMA shadow when resuming ----
        if ema_enabled and ema is not None and last_ckpt is not None:
            logger.info(
                f"[EMA] Attempting to resume EMA shadow from checkpoint matching: {last_ckpt}"
            )
            resume_ema_shadow_from_checkpoint(
                model=model,
                ema=ema,
                model_checkpoint_dir=last_ckpt,
                ema_dirname="ema_model",
                strict=False,
            )

        if last_ckpt is not None:
            logger.info(f"Resuming from checkpoint {last_ckpt}")
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            logger.info("Starting training from scratch")
            trainer.train()

        # RAW final save (same folder as before)
        trainer.save_model(model_save_dir)
        trainer.save_state()
        tok.save_pretrained(model_save_dir)

        logger.info(
            f"Training complete. Model and tokenizer saved to {model_save_dir}."
        )
    finally:
        stop_timer(final_output_dir)
