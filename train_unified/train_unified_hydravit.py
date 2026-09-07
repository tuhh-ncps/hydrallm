from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import os
import random
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from model_implementations import load_model, load_tokenizer_with_fallback
from common_helpers import parse_dtype, pick_default_device, str_to_bool
from common_train_helpers import (
    build_training_arguments,
    setup_training_output_dir_and_config,
    start_timer,
    stop_timer,
    param_part,
)
from common_train_helpers import load_teacher_model, weights_tied
from .trainer_unified_hydravit import UnifiedHydraTrainer, EMAConfig
from .combined_dataset import (
    CombinedDataCollator,
    SynchronizedModalityBatchMixerIterable,
)


logger = hf_logging.get_logger(__name__)


# =============================================================================
# Path Resolution Utilities
# =============================================================================


def _infer_dataset_root_from_parquet_path(parquet_path: str) -> str:
    """
    Infer the dataset root directory from a parquet path.

    Handles multiple path formats:
      - /path/to/dataset_root
      - /path/to/dataset_root/parquet
      - /path/to/dataset_root/parquet/shard_id=00000/part-00000.parquet

    Returns the dataset_root directory path (defaults to parent of current directory).
    """
    p = os.path.abspath(str(parquet_path))

    # Navigate up from file to directory
    cur = os.path.dirname(p) if os.path.isfile(p) else p

    # Case: current dir is "parquet" → parent is dataset root
    if os.path.basename(cur.rstrip("/")) == "parquet":
        return os.path.dirname(cur)

    # Case: "parquet" appears somewhere in path → use everything before it
    parts = cur.split(os.sep)
    if "parquet" in parts:
        idx = parts.index("parquet")
        if idx > 0:
            return os.sep.join(parts[:idx])

    # Case: directory contains a "parquet" subdirectory
    if os.path.isdir(os.path.join(cur, "parquet")):
        return cur

    # Fallback: return parent directory
    return os.path.dirname(cur)


def _find_parquet_files(parquet_path: str) -> List[str]:
    """
    Resolve parquet files from various input formats.

    Accepts:
      - Single .parquet file path
      - Directory containing parquet files (searched recursively)
      - Dataset root containing a parquet/ subdirectory

    Returns a sorted list of absolute paths to .parquet files.
    """
    import glob

    p = os.path.abspath(str(parquet_path))

    if os.path.isfile(p):
        if not p.lower().endswith(".parquet"):
            raise FileNotFoundError(f"parquet_path is a file but not .parquet: {p}")
        return [p]

    if os.path.isdir(p):
        # Search the directory itself, then fall back to a parquet/ subdirectory
        for search_root in [p, os.path.join(p, "parquet")]:
            if os.path.isdir(search_root):
                files = sorted(
                    glob.glob(
                        os.path.join(search_root, "**", "*.parquet"), recursive=True
                    )
                )
                if files:
                    return files

        raise FileNotFoundError(f"No .parquet files found under: {p}")

    raise FileNotFoundError(f"parquet_path does not exist: {p}")


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def _dist_info_from_env() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size) from environment variables."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _rank_device() -> str:
    """Determine the appropriate torch device string for the current rank."""
    rank, local_rank, world_size = _dist_info_from_env()

    logger.info(
        f"[DIST] rank={rank} local_rank={local_rank} world_size={world_size} "
        f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}"
    )

    base = pick_default_device()
    return f"cuda:{local_rank}" if base == "cuda" else base


# =============================================================================
# Model Configuration Utilities
# =============================================================================


_PART_ALIASES: Dict[str, str] = {
    "vision": "vision_encoder",
    "vision_tower": "vision_encoder",
    "vision_encoder": "vision_encoder",
    "proj": "projection",
    "projector": "projection",
    "projection": "projection",
    "llm": "text_model",
    "text": "text_model",
    "language_model": "text_model",
    "text_model": "text_model",
}

_VALID_PARTS = frozenset({"vision_encoder", "projection", "text_model"})


def _normalize_train_parts(parts: List[str]) -> List[str]:
    """
    Normalize and validate trainable model part names.

    Accepts various aliases (e.g. "vision", "proj", "llm") and maps them to
    canonical names: vision_encoder, projection, text_model.
    """
    normalized = []
    for part in parts or []:
        canonical = _PART_ALIASES.get(str(part).strip().lower())
        if canonical is None:
            raise ValueError(
                f"Unknown train_model_parts entry: '{part}'. Allowed: {sorted(_VALID_PARTS)}"
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _set_trainable_params(model, *, train_parts: List[str]) -> Dict[str, int]:
    """
    Freeze/unfreeze parameters based on which model parts should be trained.

    Returns parameter counts (in elements, not bytes) per component.
    """
    train_parts = _normalize_train_parts(train_parts)
    if not train_parts:
        raise ValueError(
            "train.train_model_parts is empty — nothing would be trainable."
        )

    counts = {
        "vision_encoder": 0,
        "projection": 0,
        "text_model": 0,
        "total": 0,
        "trainable": 0,
    }

    for name, param in model.named_parameters():
        part = param_part(name)
        counts["total"] += param.numel()

        param.requires_grad = part in train_parts
        if param.requires_grad:
            counts[part] += param.numel()
            counts["trainable"] += param.numel()

    logger.info(
        "Trainable params (M): "
        + ", ".join(f"{k}={counts[k] / 1e6:.2f}" for k in _VALID_PARTS)
        + f" | trainable={counts['trainable'] / 1e6:.2f} / total={counts['total'] / 1e6:.2f}"
        + f" | parts={train_parts}"
    )
    return counts


# =============================================================================
# Configuration Parsing Helpers
# =============================================================================


def _read_part_warmup(
    train_cfg: Dict, part: str
) -> Tuple[Optional[int], Optional[float]]:
    """Extract per-component warmup (steps, ratio) from the warmup config block."""
    part_cfg = (train_cfg.get("warmup") or {}).get(part) or {}
    steps = part_cfg.get("steps")
    ratio = part_cfg.get("ratio")
    return (
        int(steps) if steps is not None else None,
        float(ratio) if ratio is not None else None,
    )


def _parse_multimodal_flag(train_cfg: Dict, has_mm_data: bool) -> bool:
    """
    Resolve the multimodal training flag from config.

    'auto' (default) enables multimodal iff the dataset contains multimodal rows.
    """
    mm = train_cfg.get("multimodal", "auto")

    if isinstance(mm, bool):
        return mm
    if mm is None:
        return has_mm_data

    key = str(mm).strip().lower()
    if key == "auto":
        return has_mm_data
    try:
        return bool(str_to_bool(key))
    except ValueError as e:
        raise ValueError("train.multimodal must be: auto|true|false") from e


def _row_is_multimodal(row: Dict[str, Any]) -> bool:
    """A row is multimodal if it has a non-empty image_path."""
    image_path = row.get("image_path")
    return isinstance(image_path, str) and image_path.strip() != ""


# =============================================================================
# Dataset Logging
# =============================================================================


class _DatasetLogConfig:
    """Parsed dataset-logging settings from YAML config."""

    def __init__(self, config: Dict[str, Any] | None):
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.every_n_batches: int = int(cfg.get("every_n_batches", 0) or 0)
        self.max_prints: int = int(cfg.get("max_prints", 2) or 2)
        self.rank0_only: bool = bool(cfg.get("rank0_only", True))
        self.print_raw_samples: bool = bool(cfg.get("print_raw_samples", True))
        self.print_full_decoded: bool = bool(cfg.get("print_full_decoded", True))
        self.print_loss_decoded: bool = bool(cfg.get("print_loss_decoded", True))
        self.max_text_chars: int = int(cfg.get("max_text_chars", 400) or 400)


class ConfigurableLogMixer(SynchronizedModalityBatchMixerIterable):
    """
    Dataset mixer with YAML-driven batch logging.

    Extends the base mixer to optionally print raw samples, tokenized shapes,
    decoded input_ids, and loss-contributing tokens for debugging.
    """

    def __init__(
        self,
        *args,
        log_cfg: _DatasetLogConfig,
        tokenizer_for_log: Any = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._log_cfg = log_cfg
        self._prints_done = 0
        self._tok = tokenizer_for_log or getattr(self.collator, "tokenizer", None)

    # -- Logging control ------------------------------------------------------

    def _should_log(self, step: int) -> bool:
        cfg = self._log_cfg
        if not cfg.enabled or self._prints_done >= cfg.max_prints:
            return False
        # If every_n_batches > 0, log on those steps; otherwise log first max_prints
        n = cfg.every_n_batches
        return (step % n == 0) if n > 0 else True

    def _truncate(self, text: str) -> str:
        text = text or ""
        m = self._log_cfg.max_text_chars
        return text if len(text) <= m else text[: m - 3] + "..."

    # -- Batch logging --------------------------------------------------------

    def _log_batch(
        self,
        *,
        step: int,
        modality: str,
        raw_rows: List[Dict[str, Any]],
        batch: Dict[str, torch.Tensor],
    ):
        rank, _, _ = _dist_info_from_env()
        if self._log_cfg.rank0_only and rank != 0:
            return
        if not raw_rows:
            return

        row = raw_rows[0]
        print(
            f"\n[DATA-LOG] batch={step} rank={rank} modality={modality} uid={row.get('uid')}",
            flush=True,
        )

        if self._log_cfg.print_raw_samples:
            self._log_raw_sample(row)

        self._log_tokenized_batch(batch)
        self._prints_done += 1

    def _log_raw_sample(self, row: Dict[str, Any]):
        print(
            f"  raw:\n"
            f"    dataset={row.get('dataset')} subset={row.get('subset')} task={row.get('task')}\n"
            f"    image_path={row.get('image_path')}\n"
            f"    prompt={self._truncate(str(row.get('prompt', '')))!r}\n"
            f"    target={self._truncate(str(row.get('target', '')))!r}",
            flush=True,
        )

    def _log_tokenized_batch(self, batch: Dict[str, torch.Tensor]):
        try:
            input_ids = batch.get("input_ids")
            labels = batch.get("labels")
            attention_mask = batch.get("attention_mask")

            # Shapes
            for name, tensor in [
                ("input_ids", input_ids),
                ("attention_mask", attention_mask),
                ("labels", labels),
            ]:
                if isinstance(tensor, torch.Tensor):
                    extra = f" dtype={tensor.dtype}" if name == "input_ids" else ""
                    print(
                        f"  padded.{name}.shape={tuple(tensor.shape)}{extra}",
                        flush=True,
                    )

            # Full decoded input
            if (
                self._log_cfg.print_full_decoded
                and self._tok
                and isinstance(input_ids, torch.Tensor)
            ):
                decoded = self._tok.decode(
                    input_ids[0].detach().cpu().tolist(), skip_special_tokens=False
                )
                print(
                    f"  padded.decode(input_ids[0])={self._truncate(decoded)!r}",
                    flush=True,
                )

            # Decode only the tokens that contribute to the loss (labels != -100)
            if (
                self._log_cfg.print_loss_decoded
                and self._tok
                and isinstance(labels, torch.Tensor)
                and isinstance(input_ids, torch.Tensor)
            ):
                loss_mask = (labels[0] != -100).detach().cpu()
                loss_ids = input_ids[0].detach().cpu()[loss_mask].tolist()
                decoded = self._tok.decode(loss_ids, skip_special_tokens=False)
                print(
                    f"  loss.decode(labels!=-100)={self._truncate(decoded)!r}",
                    flush=True,
                )

        except Exception as e:
            print(
                f"  [DATA-LOG][WARN] decode failed: {type(e).__name__}: {e}", flush=True
            )


# =============================================================================
# Deterministic Seeding
# =============================================================================


def _seed_everything(seed: int):
    """Set seeds across all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


# =============================================================================
# Gradient Checkpointing Helpers
# =============================================================================


# def _configure_gradient_checkpointing(model, enabled: bool):
#     """Enable or disable gradient checkpointing on the model."""
#     if enabled:
#         try:
#             model.gradient_checkpointing_enable()
#         except Exception:
#             pass
#         try:
#             if hasattr(model, "enable_input_require_grads"):
#                 model.enable_input_require_grads()
#         except Exception:
#             pass
#     else:
#         try:
#             if hasattr(model, "gradient_checkpointing_disable"):
#                 model.gradient_checkpointing_disable()
#         except Exception:
#             pass


def _configure_gradient_checkpointing(model, enabled: bool):
    """Enable or disable gradient checkpointing on the model."""
    if not enabled:
        try:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        except Exception:
            pass
        return

    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning(f"gradient_checkpointing_enable failed: {e}")

    # Fix Bug 2: enable_input_require_grads must be called on the module
    # whose forward() receives the first input to checkpoint() — i.e. the
    # language model (FlexCoreGemma3TextModel), NOT the top-level wrapper.
    # Without this, gradient checkpointing silently drops gradients through
    # the projector when embed_tokens is frozen.
    inner_lm = None
    try:
        # Multimodal path: model.model.language_model
        inner_lm = model.model.language_model
    except AttributeError:
        pass
    if inner_lm is None:
        try:
            # Text-only path: model.model
            inner_lm = model.model
        except AttributeError:
            pass

    if inner_lm is not None and hasattr(inner_lm, "enable_input_require_grads"):
        try:
            inner_lm.enable_input_require_grads()
            logger.info(
                f"[GC] enable_input_require_grads called on {type(inner_lm).__name__}"
            )
        except Exception as e:
            logger.warning(f"[GC] enable_input_require_grads failed: {e}")
    elif hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as e:
            logger.warning(f"[GC] enable_input_require_grads (top-level) failed: {e}")


def _resize_embeddings_if_needed(model, tokenizer):
    """Resize token embeddings if vocab size doesn't match the tokenizer."""
    try:
        if model.get_output_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    except Exception:
        pass


# =============================================================================
# Main Training Entry Point
# =============================================================================


def run_unified_training_from_cfg(cfg: Dict):
    """
    Run unified multimodal/text training from a configuration dictionary.

    Pipeline:
      1. Load combined Parquet dataset (local, non-streaming)
      2. Split into text-only and multimodal views by image_path presence
      3. Synchronize modality selection across ranks each batch (prevents DDP hangs)
      4. Train with Hydra multi-head attention, optional KD from a teacher
      5. Save model, tokenizer, and processor
    """
    final_output_dir = setup_training_output_dir_and_config(cfg, section="train")
    start_timer(final_output_dir)
    try:
        _run_training(cfg, final_output_dir)
    finally:
        stop_timer(final_output_dir)


def _run_training(cfg: Dict, final_output_dir: str):
    """Core training logic, separated for clean error handling in the caller."""
    tr = cfg["train"]

    # -------------------------------------------------------------------------
    # 1. Parse all configuration
    # -------------------------------------------------------------------------

    # Output paths
    model_save_dir = os.path.join(
        final_output_dir, tr.get("model_save_subfolder", "model")
    )

    # Model identity
    implementation = tr.get("implementation", "hydra_gemma_from_flex")
    model_name_or_path = tr["model_name_or_path"]
    tokenizer_name_or_path = tr.get("tokenizer_name_or_path")

    # Sequence / architecture
    seq_len = int(tr.get("seq_len", 2048))
    attn_implementation = tr.get("attn_implementation", "auto")
    gradient_checkpointing = bool(tr.get("gradient_checkpointing", False))

    # Per-component learning rates
    text_model_lr = float(tr.get("text_model_lr") or tr.get("learning_rate") or 3e-4)
    projection_lr = float(tr.get("projection_lr") or text_model_lr)
    vision_encoder_lr = float(tr.get("vision_encoder_lr") or text_model_lr)

    # Optimizer
    weight_decay = float(tr.get("weight_decay", 0.0))
    lr_scheduler_type = tr.get("lr_scheduler_type", "cosine")
    exclude_wd_norms_embeddings = bool(tr.get("exclude_wd_norms_embeddings", False))
    max_grad_norm_enabled = bool(tr.get("max_grad_norm_enabled", True))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))

    # Global warmup
    warmup_steps_global = tr.get("warmup_steps")
    warmup_ratio_global = tr.get("warmup_ratio")
    warmup_steps_global = (
        int(warmup_steps_global) if warmup_steps_global is not None else None
    )
    warmup_ratio_global = (
        float(warmup_ratio_global) if warmup_ratio_global is not None else None
    )

    # Per-component warmup
    text_warmup_steps, text_warmup_ratio = _read_part_warmup(tr, "text_model")
    proj_warmup_steps, proj_warmup_ratio = _read_part_warmup(tr, "projection")
    vis_warmup_steps, vis_warmup_ratio = _read_part_warmup(tr, "vision_encoder")

    # Training schedule
    max_steps = int(tr["max_steps"])
    logging_steps = int(tr.get("logging_steps", 50))
    save_steps = int(tr.get("save_steps", 5000))
    save_total_limit = int(tr.get("save_total_limit", 20))
    grad_accum_steps = int(tr.get("gradient_accumulation_steps", 1))
    per_device_batch_size = int(tr.get("per_device_train_batch_size", 1))

    # Infrastructure
    deepspeed = tr.get("deepspeed_config") or tr.get("deepspeed")
    report_to = tr.get("report_to", ["tensorboard"])
    seed = int(tr.get("seed", 42))

    # Hydra heads
    heads: List[int] = tr.get("heads") or []
    head_weights = tr.get("head_weights")
    train_parts = _normalize_train_parts(tr.get("train_model_parts") or ["projection"])

    # Loss
    loss_cfg = tr.get("loss") or {}
    loss_type = str(loss_cfg.get("type", "ce_plus_z")).lower()
    z_loss_alpha = float(loss_cfg.get("z_loss_alpha", 0.0))

    # Knowledge distillation
    teacher_cfg = tr.get("teacher") or {}
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    kd_temperature = float(teacher_cfg.get("temperature", 2.0))
    kd_weight = float(teacher_cfg.get("weight", 0.0))
    kd_every_n_steps = int(teacher_cfg.get("kd_every_n_steps", 1))

    # EMA
    ema_config = EMAConfig.from_cfg(tr.get("ema"))

    # -------------------------------------------------------------------------
    # 2. Seeding & device
    # -------------------------------------------------------------------------
    _seed_everything(seed)
    device = _rank_device()
    amp_dtype = parse_dtype(tr.get("dtype", "auto"), device)

    # -------------------------------------------------------------------------
    # 3. Tokenizer
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer_with_fallback(
        model_name_or_path, tokenizer_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # 4. Dataset
    # -------------------------------------------------------------------------
    ds_cfg = tr.get("dataset") or {}
    parquet_path = ds_cfg["parquet_path"]
    ds_split = ds_cfg.get("split", "train")
    images_root = ds_cfg.get("images_root")
    max_samples_per_rank = ds_cfg.get("max_samples_per_rank")

    parquet_files = _find_parquet_files(parquet_path)
    if images_root is None:
        images_root = _infer_dataset_root_from_parquet_path(parquet_files[0])

    ds = load_dataset(
        "parquet", data_files={ds_split: parquet_files}, split=ds_split, streaming=False
    )
    text_ds = ds.filter(lambda r: not _row_is_multimodal(r))
    mm_ds = ds.filter(lambda r: _row_is_multimodal(r))
    text_len, mm_len = len(text_ds), len(mm_ds)

    rank, _, world_size = _dist_info_from_env()
    if rank == 0:
        logger.info(
            f"[dataset] files={len(parquet_files)} split={ds_split} "
            f"text={text_len} mm={mm_len} world_size={world_size} "
            f"images_root={images_root}"
        )

    multimodal = _parse_multimodal_flag(tr, has_mm_data=(mm_len > 0))
    if mm_len > 0 and not multimodal:
        raise ValueError("Dataset has multimodal rows but train.multimodal=false.")
    if multimodal and not images_root:
        raise ValueError("images_root must be set for multimodal training.")

    # -------------------------------------------------------------------------
    # 5. Processor
    # -------------------------------------------------------------------------
    processor = None
    if multimodal:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=True
        )
        try:
            if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                processor.tokenizer = tokenizer
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 6. Student model
    # -------------------------------------------------------------------------
    model = load_model(
        model_name_or_path,
        implementation=implementation,
        dtype=amp_dtype,
        attn_implementation=attn_implementation,
        device=device,
        multimodal=multimodal,
    )
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    _configure_gradient_checkpointing(model, gradient_checkpointing)
    _set_trainable_params(model, train_parts=train_parts)
    _resize_embeddings_if_needed(model, tokenizer)
    if not weights_tied(model):
        try:
            model.tie_weights()
        except Exception:
            pass
    assert weights_tied(
        model
    ), "Weight tying failed: input embeddings and lm_head must be tied (student)."

    # -------------------------------------------------------------------------
    # 7. Teacher model (optional KD)
    # -------------------------------------------------------------------------
    teacher_model = None
    teacher_full_heads = None
    if teacher_enabled and kd_weight > 0.0:
        teacher_model, teacher_full_heads = load_teacher_model(
            teacher_cfg, device, multimodal=multimodal, tokenizer=tokenizer
        )

    # -------------------------------------------------------------------------
    # 8. Dataset mixer
    # -------------------------------------------------------------------------

    # Mixing weights proportional to subset sizes
    if mm_len <= 0:
        text_weight, mm_weight = 1.0, 0.0
    elif text_len <= 0:
        text_weight, mm_weight = 0.0, 1.0

    else:
        mm_weight = mm_len / (text_len + mm_len)
        text_weight = 1.0 - mm_weight
    # text_weight_cfg = ds_cfg.get("text_weight", None)
    # mm_weight_cfg = ds_cfg.get("mm_weight", None)

    # if text_weight_cfg is not None or mm_weight_cfg is not None:
    #     text_weight = float(text_weight_cfg or 0.0)
    #     mm_weight = float(mm_weight_cfg or 0.0)

    #     if text_weight + mm_weight <= 0:
    #         raise ValueError("text_weight + mm_weight must be > 0")

    # else:
    #     # Default: proportional to dataset sizes
    #     if mm_len <= 0:
    #         text_weight, mm_weight = 1.0, 0.0
    #     elif text_len <= 0:
    #         text_weight, mm_weight = 0.0, 1.0
    #     else:
    #         mm_weight = mm_len / (text_len + mm_len)
    #         text_weight = 1.0 - mm_weight

    if rank == 0:
        logger.info(f"[dataset] mix weights: text={text_weight:.4f} mm={mm_weight:.4f}")

    collator = CombinedDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=seq_len,
        pad_to_multiple_of=64,
        images_root=str(images_root),
    )
    log_cfg = _DatasetLogConfig(ds_cfg.get("log") or {})
    train_dataset = ConfigurableLogMixer(
        text_dataset=text_ds,
        mm_dataset=mm_ds,
        collator=collator,
        batch_size=per_device_batch_size,
        seed=seed,
        text_weight=text_weight,
        mm_weight=mm_weight,
        sync_across_ranks=True,  # Prevents DDP/DeepSpeed hangs
        shard_by_rank=True,  # Each rank sees a disjoint subset
        max_samples_per_rank=(
            int(max_samples_per_rank) if max_samples_per_rank is not None else None
        ),
        debug=None,
        tokenizer_for_log=tokenizer,
        log_cfg=log_cfg,
    )

    # -------------------------------------------------------------------------
    # 9. Hydra head configuration
    # -------------------------------------------------------------------------
    if multimodal:
        model_heads = int(
            getattr(
                getattr(model.config, "text_config", None),
                "num_attention_heads",
                1,
            )
        )
    else:
        model_heads = int(getattr(model.config, "num_attention_heads", 1))

    if not heads:
        heads = list(range(1, model_heads + 1))
    else:
        heads = [max(1, min(int(h), model_heads)) for h in heads]

    if head_weights is not None:
        if len(head_weights) != len(heads):
            raise ValueError("head_weights must match length of heads")
        total_weight = sum(head_weights)
        if total_weight <= 0:
            raise ValueError("head_weights must sum to a positive value")
        head_weights = [w / total_weight for w in head_weights]

    # -------------------------------------------------------------------------
    # 10. Build TrainingArguments and Trainer
    # -------------------------------------------------------------------------
    training_args = build_training_arguments(
        dict(
            output_dir=model_save_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=text_model_lr,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=0.0,  # Per-component warmup is handled by the trainer
            warmup_steps=0,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to or ["none"],
            deepspeed=deepspeed,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            fp16=(amp_dtype == torch.float16),
            bf16=(amp_dtype == torch.bfloat16),
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=int(tr.get("dataloader_num_workers", 0)),
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            max_grad_norm=max_grad_norm if max_grad_norm_enabled else 0.0,
            _eval_enabled=False,
        )
    )

    trainer = UnifiedHydraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=None,
        tokenizer=tokenizer,
        processor=processor,
        # Hydra heads
        heads=heads,
        head_weights=head_weights,
        # Per-component learning rates
        text_model_lr=text_model_lr,
        projection_lr=projection_lr,
        vision_encoder_lr=vision_encoder_lr,
        # Loss
        loss_type=loss_type,
        z_loss_alpha=z_loss_alpha,
        # Knowledge distillation
        teacher_model=teacher_model,
        teacher_full_heads=teacher_full_heads,
        kd_temperature=kd_temperature,
        kd_weight=kd_weight,
        kd_every_n_steps=kd_every_n_steps,
        teacher_device_map=teacher_cfg.get("device_map"),
        # Optimizer
        exclude_wd_norms_embeddings=exclude_wd_norms_embeddings,
        # Warmup (global)
        global_warmup_steps=warmup_steps_global,
        global_warmup_ratio=warmup_ratio_global,
        # Warmup (per-component)
        text_model_warmup_steps=text_warmup_steps,
        text_model_warmup_ratio=text_warmup_ratio,
        projection_warmup_steps=proj_warmup_steps,
        projection_warmup_ratio=proj_warmup_ratio,
        vision_encoder_warmup_steps=vis_warmup_steps,
        vision_encoder_warmup_ratio=vis_warmup_ratio,
        # EMA
        ema_config=ema_config,
        # Output
        run_output_dir=final_output_dir,
    )

    # -------------------------------------------------------------------------
    # 11. Train (with optional checkpoint resumption)
    # -------------------------------------------------------------------------
    os.makedirs(model_save_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(model_save_dir)
    if last_checkpoint is not None:
        trainer.load_token_counts_from_checkpoint(last_checkpoint)
        trainer.load_ema_from_checkpoint(last_checkpoint)
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()

    # -------------------------------------------------------------------------
    # 12. Save final model, tokenizer, processor, and EMA model
    # -------------------------------------------------------------------------
    trainer.save_model(model_save_dir)
    trainer.save_state()

    # Save EMA model (separate directory with EMA weights baked in)
    ema_dir = trainer.save_ema_model(model_save_dir)
    if ema_dir and rank == 0:
        logger.info(f"[EMA] Final EMA model saved to {ema_dir}")

    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    if trainer.is_world_process_zero():
        trainer.write_run_metadata(final_output_dir)
        trainer.write_run_metadata(model_save_dir)

    logger.info(f"Training complete. Model saved to {model_save_dir}")
