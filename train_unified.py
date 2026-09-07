from __future__ import annotations

import argparse
import os
import yaml

from common import (
    setup_environment,
    warn_fallbacks,
    validate_config,
    maybe_print_config,
    add_distributed_args,
    set_dist_env_from_args,
    str_to_bool,
    _deep_merge_with_fallbacks,
    REQUIRED,
)
from train_unified.train_unified_hydravit import run_unified_training_from_cfg


def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "train": {
            "output_dir": REQUIRED,
            "model_save_subfolder": "model",
            "implementation": "hydra_gemma_from_flex",
            "model_name_or_path": REQUIRED,
            "tokenizer_name_or_path": None,
            "multimodal": "auto",  # auto|true|false
            "seq_len": 2048,
            "text_model_lr": 3.0e-4,
            "projection_lr": None,
            "vision_encoder_lr": None,
            "weight_decay": 0.1,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.01,
            "warmup_steps": None,
            "warmup": {
                "text_model": {"steps": None, "ratio": None},
                "projection": {"steps": None, "ratio": None},
                "vision_encoder": {"steps": None, "ratio": None},
            },
            "max_steps": 20000,
            "max_grad_norm_enabled": True,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size": 2,
            "exclude_wd_norms_embeddings": False,
            "dtype": "bf16",
            "deepspeed": "train_unified/configs/ds_projector_stage1.json",
            "attn_implementation": "auto",
            "gradient_checkpointing": False,
            "report_to": ["tensorboard"],
            "seed": 42,
            "heads": [1, 2, 3, 4],
            "head_weights": None,
            "logging_steps": 50,
            "save_steps": 5000,
            "save_total_limit": 20,
            "dataloader_num_workers": 0,
            "dataloader_persistent_workers": False,
            "dataloader_prefetch_factor": None,
            "train_model_parts": ["projection"],
            "loss": {
                "type": "ce_plus_z",
                "z_loss_alpha": 1.0e-5,
            },
            "teacher": {
                "enabled": False,
                "model_name_or_path": REQUIRED,
                "implementation": "auto_model",
                "attn_implementation": "auto",
                "dtype": "auto",
                "temperature": 2.0,
                "weight": 0.1,
                "kd_every_n_steps": 1,
                "compile": False,
                "device_map": None,
                "trust_remote_code": True,
            },
            "ema": {
                "enabled": False,
                "decay": 0.999,
            },
            # NEW: single combined dataset
            "dataset": {
                "parquet_path": REQUIRED,
                "split": "train",
                # Root directory that contains "images/" (your image_path is relative to images/)
                "images_root": REQUIRED,
                "max_samples_per_rank": None,  # optional debug cap; cycles if None
                # logging controls
                "log": {
                    "enabled": False,
                    "every_n_batches": 0,
                    "max_prints": 2,
                    "rank0_only": True,
                    "print_raw_samples": True,
                    "print_full_decoded": True,
                    "print_loss_decoded": True,
                    "max_text_chars": 400,
                },
            },
        },
        # you also have a "data:" section in your yaml; we leave it untouched/optional
    }


def merge_overrides(cfg, args):
    tr = cfg.setdefault("train", {})

    if args.output_dir is not None:
        tr["output_dir"] = args.output_dir
    if args.model_save_subfolder is not None:
        tr["model_save_subfolder"] = args.model_save_subfolder
    if args.implementation is not None:
        tr["implementation"] = args.implementation
    if args.model_name_or_path is not None:
        tr["model_name_or_path"] = args.model_name_or_path
    if args.tokenizer_name_or_path is not None:
        tr["tokenizer_name_or_path"] = args.tokenizer_name_or_path
    if args.multimodal is not None:
        tr["multimodal"] = args.multimodal

    if args.seq_len is not None:
        tr["seq_len"] = args.seq_len
    if args.text_model_lr is not None:
        tr["text_model_lr"] = args.text_model_lr
    if args.projection_lr is not None:
        tr["projection_lr"] = args.projection_lr
    if args.vision_encoder_lr is not None:
        tr["vision_encoder_lr"] = args.vision_encoder_lr
    if args.weight_decay is not None:
        tr["weight_decay"] = args.weight_decay
    if args.lr_scheduler_type is not None:
        tr["lr_scheduler_type"] = args.lr_scheduler_type

    if args.warmup_ratio is not None:
        tr["warmup_ratio"] = args.warmup_ratio
    if args.warmup_steps is not None:
        tr["warmup_steps"] = args.warmup_steps

    w = tr.setdefault("warmup", {})
    w.setdefault("text_model", {})
    w.setdefault("projection", {})
    w.setdefault("vision_encoder", {})
    if args.text_model_warmup_steps is not None:
        w["text_model"]["steps"] = args.text_model_warmup_steps
    if args.text_model_warmup_ratio is not None:
        w["text_model"]["ratio"] = args.text_model_warmup_ratio
    if args.projection_warmup_steps is not None:
        w["projection"]["steps"] = args.projection_warmup_steps
    if args.projection_warmup_ratio is not None:
        w["projection"]["ratio"] = args.projection_warmup_ratio
    if args.vision_encoder_warmup_steps is not None:
        w["vision_encoder"]["steps"] = args.vision_encoder_warmup_steps
    if args.vision_encoder_warmup_ratio is not None:
        w["vision_encoder"]["ratio"] = args.vision_encoder_warmup_ratio

    if args.max_steps is not None:
        tr["max_steps"] = args.max_steps
    if args.max_grad_norm_enabled is not None:
        tr["max_grad_norm_enabled"] = str_to_bool(args.max_grad_norm_enabled)
    if args.max_grad_norm is not None:
        tr["max_grad_norm"] = args.max_grad_norm
    if args.gradient_accumulation_steps is not None:
        tr["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.per_device_train_batch_size is not None:
        tr["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.exclude_wd_norms_embeddings is not None:
        tr["exclude_wd_norms_embeddings"] = str_to_bool(
            args.exclude_wd_norms_embeddings
        )

    if args.dtype is not None:
        tr["dtype"] = args.dtype
    if args.deepspeed is not None:
        tr["deepspeed"] = args.deepspeed
    if args.attn_implementation is not None:
        tr["attn_implementation"] = args.attn_implementation
    if args.gradient_checkpointing is not None:
        tr["gradient_checkpointing"] = str_to_bool(args.gradient_checkpointing)

    if args.report_to is not None:
        tr["report_to"] = args.report_to
    if args.seed is not None:
        tr["seed"] = args.seed

    if args.heads is not None:
        tr["heads"] = args.heads
    if args.head_weights is not None:
        tr["head_weights"] = args.head_weights
    if args.logging_steps is not None:
        tr["logging_steps"] = args.logging_steps
    if args.save_steps is not None:
        tr["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        tr["save_total_limit"] = args.save_total_limit

    if args.dataloader_num_workers is not None:
        tr["dataloader_num_workers"] = args.dataloader_num_workers
    if args.dataloader_persistent_workers is not None:
        tr["dataloader_persistent_workers"] = str_to_bool(
            args.dataloader_persistent_workers
        )
    if args.dataloader_prefetch_factor is not None:
        tr["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor

    if args.train_model_parts is not None:
        tr["train_model_parts"] = args.train_model_parts

    loss_cfg = tr.setdefault("loss", {})
    if args.loss_type is not None:
        loss_cfg["type"] = args.loss_type
    if args.z_loss_alpha is not None:
        loss_cfg["z_loss_alpha"] = args.z_loss_alpha

    tch = tr.setdefault("teacher", {})
    if args.teacher_enabled is not None:
        tch["enabled"] = str_to_bool(args.teacher_enabled)
    if args.teacher_model_name_or_path is not None:
        tch["model_name_or_path"] = args.teacher_model_name_or_path
    if args.teacher_implementation is not None:
        tch["implementation"] = args.teacher_implementation
    if args.teacher_attn_implementation is not None:
        tch["attn_implementation"] = args.teacher_attn_implementation
    if args.teacher_dtype is not None:
        tch["dtype"] = args.teacher_dtype
    if args.teacher_temperature is not None:
        tch["temperature"] = float(args.teacher_temperature)
    if args.teacher_weight is not None:
        tch["weight"] = float(args.teacher_weight)
    if args.teacher_kd_every_n_steps is not None:
        tch["kd_every_n_steps"] = int(args.teacher_kd_every_n_steps)
    if args.teacher_compile is not None:
        tch["compile"] = str_to_bool(args.teacher_compile)
    if args.teacher_device_map is not None:
        tch["device_map"] = args.teacher_device_map
    if args.teacher_trust_remote_code is not None:
        tch["trust_remote_code"] = str_to_bool(args.teacher_trust_remote_code)

    ema_cfg = tr.setdefault("ema", {})
    if args.ema_enabled is not None:
        ema_cfg["enabled"] = str_to_bool(args.ema_enabled)
    if args.ema_decay is not None:
        ema_cfg["decay"] = float(args.ema_decay)

    # dataset: config-file only for now (no CLI flags)
    return cfg


def main():
    setup_environment()

    # Your existing NCCL environment knobs
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    parser = argparse.ArgumentParser(
        description="Unified training: combined Parquet dataset with synchronized modality mixing."
    )
    add_distributed_args(parser)

    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")

    # Core
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--model-save-subfolder", type=str)
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--tokenizer-name-or-path", type=str)
    parser.add_argument("--multimodal", type=str, help="auto|true|false")

    # Training basics
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--text-model-lr", type=float)
    parser.add_argument("--projection-lr", type=float)
    parser.add_argument("--vision-encoder-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-scheduler-type", type=str)

    # Global warmup
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--warmup-steps", type=int)

    # Per-part warmup controls
    parser.add_argument("--text-model-warmup-ratio", type=float)
    parser.add_argument("--text-model-warmup-steps", type=int)
    parser.add_argument("--projection-warmup-ratio", type=float)
    parser.add_argument("--projection-warmup-steps", type=int)
    parser.add_argument("--vision-encoder-warmup-ratio", type=float)
    parser.add_argument("--vision-encoder-warmup-steps", type=int)

    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-grad-norm-enabled", type=str)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--exclude-wd-norms-embeddings", type=str)
    parser.add_argument("--dtype", type=str, help="auto|bf16|fp16|fp32")
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--attn-implementation", type=str)
    parser.add_argument("--gradient-checkpointing", type=str)
    parser.add_argument("--report-to", type=str, nargs="+", action="extend")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--heads", type=int, nargs="+", action="extend")
    parser.add_argument("--head-weights", type=float, nargs="+", action="extend")
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)

    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--dataloader-persistent-workers", type=str)
    parser.add_argument("--dataloader-prefetch-factor", type=int)

    parser.add_argument(
        "--train-model-parts",
        type=str,
        nargs="+",
        action="extend",
        help="Any subset of: vision_encoder projection text_model",
    )

    # Loss
    parser.add_argument("--loss-type", type=str, help="ce|ce_plus_z|z_only")
    parser.add_argument("--z-loss-alpha", type=float)

    # Teacher/KD
    parser.add_argument("--teacher-enabled", type=str)
    parser.add_argument("--teacher-model-name-or-path", type=str)
    parser.add_argument("--teacher-implementation", type=str)
    parser.add_argument("--teacher-attn-implementation", type=str)
    parser.add_argument("--teacher-dtype", type=str)
    parser.add_argument("--teacher-temperature", type=float)
    parser.add_argument("--teacher-weight", type=float)
    parser.add_argument("--teacher-kd-every-n-steps", type=int)
    parser.add_argument("--teacher-compile", type=str)
    parser.add_argument("--teacher-device-map", type=str)
    parser.add_argument("--teacher-trust-remote-code", type=str)

    # EMA
    parser.add_argument("--ema-enabled", type=str, help="true|false")
    parser.add_argument("--ema-decay", type=float, help="e.g. 0.999")

    args = parser.parse_args()
    set_dist_env_from_args(args)

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(
        initial_defaults, user_cfg
    )
    final_cfg = merge_overrides(initial_merged_cfg, args)

    validate_config(final_cfg, "train-unified")
    warn_fallbacks(
        final_cfg,
        initial_defaults,
        initial_fallback_paths,
        ("project", "train"),
        "train-unified",
    )
    maybe_print_config(args.print_config, final_cfg, "train-unified")

    run_unified_training_from_cfg(final_cfg)

    os.environ.pop("NCCL_IB_DISABLE", None)
    os.environ.pop("NCCL_P2P_DISABLE", None)
    os.environ.pop("NCCL_SOCKET_IFNAME", None)


if __name__ == "__main__":
    main()
