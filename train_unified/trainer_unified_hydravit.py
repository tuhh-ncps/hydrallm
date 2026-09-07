"""
train_unified/trainer_unified_hydravit.py

Unified Hydra trainer with:
  - Hydra head sampling (current_num_heads)
  - Optional knowledge distillation from a teacher model
  - Optional EMA of trainable parameters
  - Per-component learning rates and per-component warmup schedules
  - Datasets that yield already-collated batches (batch_size=None in DataLoader)
  - Per-head windowed loss logging
  - Token counting with run_metadata.json persistence and checkpoint resume
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert
from transformers import Trainer, TrainerCallback
from transformers.utils import logging as hf_logging

from common_train_helpers import (
    compute_kd_loss,
    compute_z_loss,
    param_part,
    EMATracker,
    EMA_STATE_FILENAME,
    EMA_MODEL_SUBDIR,
)

logger = hf_logging.get_logger(__name__)


# =============================================================================
# JSON I/O Helpers
# =============================================================================


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_json_atomic(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# =============================================================================
# Processor Saving
# =============================================================================


def _save_processor_artifacts(processor: Any, output_dir: str):
    """Save processor and its sub-components (image_processor, feature_extractor)."""
    if processor is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in [
        ("processor", processor),
        ("image_processor", getattr(processor, "image_processor", None)),
        ("feature_extractor", getattr(processor, "feature_extractor", None)),
    ]:
        if obj is None:
            continue
        try:
            if hasattr(obj, "save_pretrained"):
                obj.save_pretrained(output_dir)
        except Exception as e:
            logger.warning(f"Failed to save {name} to {output_dir}: {e}")


# =============================================================================
# Batch Size Inference
# =============================================================================


def _infer_batch_size(batch: Any) -> int:
    """Infer batch size from an already-collated batch dict."""
    if not isinstance(batch, dict):
        return 0
    for k in (
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "token_type_ids",
    ):
        v = batch.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    return 0


# =============================================================================
# Run Metadata Manager
# =============================================================================


class RunMetadataManager:
    """
    Manages run_metadata.json: token counts, persistence across checkpoints,
    and resume from a prior run.
    """

    def __init__(self):
        self.total_input_tokens_all_gpus: int = 0
        self.total_loss_tokens_all_gpus: int = 0

    @staticmethod
    def _path(directory: str) -> str:
        return os.path.join(directory, "run_metadata.json")

    def write(self, directory: str):
        """Merge token counters into directory/run_metadata.json."""
        path = self._path(directory)
        meta = _read_json(path)
        meta["tokens"] = {
            "total_input_tokens_all_gpus": self.total_input_tokens_all_gpus,
            "total_loss_tokens_all_gpus": self.total_loss_tokens_all_gpus,
        }
        _write_json_atomic(path, meta)

    def load_from_checkpoint(self, checkpoint_dir: str):
        """Restore token counters from a checkpoint's run_metadata.json."""
        path = self._path(checkpoint_dir)
        meta = _read_json(path)
        tokens = meta.get("tokens") or {}
        if not isinstance(tokens, dict):
            tokens = {}
        self.total_input_tokens_all_gpus = int(
            tokens.get("total_input_tokens_all_gpus", 0) or 0
        )
        self.total_loss_tokens_all_gpus = int(
            tokens.get("total_loss_tokens_all_gpus", 0) or 0
        )
        logger.info(
            f"[tokens] Resumed: input={self.total_input_tokens_all_gpus}, "
            f"loss={self.total_loss_tokens_all_gpus}"
        )

    def accumulate_from_batch(self, inputs: Dict[str, Any], model: torch.nn.Module):
        """
        Count tokens in a micro-batch, all-reduce across ranks, and accumulate
        into the global running totals.
        """
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        dev = _resolve_tensor_device(labels, attention_mask, model)

        local_input = (
            attention_mask.to(dtype=torch.long).sum()
            if isinstance(attention_mask, torch.Tensor)
            else torch.zeros((), device=dev, dtype=torch.long)
        )
        local_loss = (
            (labels != -100).to(dtype=torch.long).sum()
            if isinstance(labels, torch.Tensor)
            else torch.zeros((), device=dev, dtype=torch.long)
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_input, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_loss, op=torch.distributed.ReduceOp.SUM)

        self.total_input_tokens_all_gpus += int(local_input.item())
        self.total_loss_tokens_all_gpus += int(local_loss.item())


def _resolve_tensor_device(
    labels: Any, attention_mask: Any, model: torch.nn.Module
) -> torch.device:
    """Find a device from available tensors or model parameters (CPU fallback)."""
    if isinstance(labels, torch.Tensor):
        return labels.device
    if isinstance(attention_mask, torch.Tensor):
        return attention_mask.device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


# =============================================================================
# Loss Components & Tracking
# =============================================================================


@dataclass
class LossComponents:
    """Individual loss components for a single forward pass."""

    ce: torch.Tensor
    z: torch.Tensor
    kd: torch.Tensor
    total: torch.Tensor


class LossTracker:
    """
    Tracks loss statistics at two granularities:

    1. **Microbatch accumulation** — accumulated across gradient-accumulation
       micro-steps, averaged and promoted to the window at each optimizer step.
    2. **Window accumulation** — accumulated across optimizer steps, drained
       into log dicts at each logging step.

    Also tracks per-head (per-k) statistics for Hydra analysis.
    """

    _KEYS = ("ce", "z", "kd", "total")

    def __init__(self):
        self._reset_micro()
        self._reset_window()
        self.per_k_stats: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "ce": 0.0, "z": 0.0, "kd": 0.0, "total": 0.0}
        )
        self.last_k: Optional[int] = None

    # -- Micro-batch level ----------------------------------------------------

    def _reset_micro(self):
        self._micro_count: int = 0
        self._micro_sums: Dict[str, float] = {k: 0.0 for k in self._KEYS}

    def record_microbatch(self, losses: LossComponents):
        """Record losses from a single forward pass (micro-batch)."""
        self._micro_count += 1
        for key in self._KEYS:
            val = getattr(losses, key)
            self._micro_sums[key] += (
                float(val.detach().item())
                if isinstance(val, torch.Tensor)
                else float(val)
            )

    # -- Optimizer-step level -------------------------------------------------

    def _reset_window(self):
        self._win_count: int = 0
        self._win_sums: Dict[str, float] = {k: 0.0 for k in self._KEYS}

    def finalize_optimizer_step(self, k: Optional[int]):
        """
        Average micro-batch losses and push to window + per-k stats.
        Call exactly once per optimizer step (when gradients synchronize).
        Resets micro-batch accumulators.
        """
        if self._micro_count <= 0:
            return

        denom = float(self._micro_count)
        averages = {key: self._micro_sums[key] / denom for key in self._KEYS}

        # Push to window
        self._win_count += 1
        for key in self._KEYS:
            self._win_sums[key] += averages[key]

        # Push to per-k
        if k is not None:
            sw = self.per_k_stats[int(k)]
            sw["count"] += 1
            for key in self._KEYS:
                sw[key] += averages[key]
            self.last_k = int(k)

        self._reset_micro()

    # -- Logging-step level ---------------------------------------------------

    def drain_to_logs(self) -> Dict[str, float]:
        """
        Drain windowed + per-k stats into a log dict.
        Resets all window/per-k counters. Returns empty dict if nothing accumulated.
        """
        logs: Dict[str, float] = {}

        # Global window
        if self._win_count > 0:
            denom = float(self._win_count)
            for key in self._KEYS:
                logs[
                    f"unified/{key}_loss" if key != "total" else "unified/loss_total"
                ] = (self._win_sums[key] / denom)
            self._reset_window()

        # Per-k window
        if self.per_k_stats:
            if self.last_k is not None:
                logs["unified/last_k"] = float(self.last_k)
            for k_val in sorted(self.per_k_stats.keys()):
                sw = self.per_k_stats[k_val]
                c = int(sw["count"])
                if c <= 0:
                    continue
                d = float(c)
                logs[f"k{k_val}/loss_total"] = sw["total"] / d
                logs[f"k{k_val}/ce_loss"] = sw["ce"] / d
                logs[f"k{k_val}/z_loss"] = sw["z"] / d
                logs[f"k{k_val}/kd_loss"] = sw["kd"] / d
            self.per_k_stats.clear()
            self.last_k = None

        return logs


# =============================================================================
# Head Sampler
# =============================================================================


class HeadSampler:
    """Samples Hydra head counts, optionally with non-uniform weights."""

    def __init__(self, heads: List[int], weights: Optional[List[float]] = None):
        self.heads = heads or []
        self.weights = weights
        if weights is not None and len(weights) != len(self.heads):
            raise ValueError(
                f"head_weights length ({len(weights)}) must match "
                f"heads length ({len(self.heads)})"
            )

    def sample(self) -> Optional[int]:
        if not self.heads:
            return None
        if self.weights is None:
            return random.choice(self.heads)
        return random.choices(self.heads, weights=self.weights, k=1)[0]


# =============================================================================
# Knowledge Distillation Helper
# =============================================================================


class KDHelper:
    """Encapsulates teacher model management and KD loss computation."""

    def __init__(
        self,
        teacher_model: Optional[torch.nn.Module],
        teacher_full_heads: Optional[int],
        temperature: float,
        weight: float,
        every_n_steps: int,
        device_map: Optional[str],
    ):
        self.teacher = teacher_model
        self.full_heads = teacher_full_heads
        self.temperature = temperature
        self.weight = weight
        self.every_n_steps = max(1, every_n_steps)
        self.device_map = device_map
        self._vocab_warned_ref: list[bool] = [False]

    @property
    def enabled(self) -> bool:
        return self.teacher is not None and self.weight > 0.0

    def should_run_this_step(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        return ((global_step + 1) % self.every_n_steps) == 0

    def ensure_on_device(self, device: torch.device):
        """Move teacher to target device if needed (skipped when device_map is set)."""
        if self.teacher is None or self.device_map:
            return
        try:
            current = next(self.teacher.parameters()).device
            if current != device:
                self.teacher.to(device)
        except StopIteration:
            pass
        except Exception as e:
            logger.warning(f"[KD] Failed moving teacher to {device}: {e}")

    def compute(
        self,
        student_logits: torch.Tensor,
        inputs: Dict[str, Any],
        labels: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Run teacher forward pass and compute KD loss."""
        self.ensure_on_device(device)

        with torch.no_grad():
            t_kwargs: Dict[str, Any] = {
                "input_ids": inputs.get("input_ids"),
                "pixel_values": inputs.get("pixel_values"),
                "attention_mask": inputs.get("attention_mask"),
                "token_type_ids": inputs.get("token_type_ids"),
                "use_cache": False,
                "labels": None,
            }
            if self.full_heads is not None:
                t_kwargs["current_num_heads"] = int(self.full_heads)

            teacher_out = self.teacher(**t_kwargs)
            teacher_logits = (
                teacher_out["logits"]
                if isinstance(teacher_out, dict)
                else teacher_out.logits
            )

        return compute_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            kd_weight=self.weight,
            kd_temperature=self.temperature,
            _vocab_warned_ref=self._vocab_warned_ref,
        )


# =============================================================================
# EMA Configuration
# =============================================================================


@dataclass
class EMAConfig:
    """
    Parsed EMA configuration.  Constructed from the ``train.ema`` YAML block::

        ema:
          enabled: false
          decay: 0.98666667   # ≈ 75-step window
    """

    enabled: bool = False
    decay: float = 0.999

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any] | None) -> "EMAConfig":
        """Build from the ``train.ema`` dict (tolerates None / missing keys)."""
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            decay=float(cfg.get("decay", 0.999)),
        )


# =============================================================================
# Per-Component Warmup Configuration
# =============================================================================


@dataclass
class WarmupConfig:
    """
    Warmup schedule configuration with per-component overrides.

    Resolution order for each component:
      1. Component-specific steps/ratio (e.g. text_model_steps)
      2. Global steps/ratio
      3. TrainingArguments warmup_steps/warmup_ratio
      4. 0 (no warmup)
    """

    global_steps: Optional[int] = None
    global_ratio: Optional[float] = None
    text_model_steps: Optional[int] = None
    text_model_ratio: Optional[float] = None
    projection_steps: Optional[int] = None
    projection_ratio: Optional[float] = None
    vision_encoder_steps: Optional[int] = None
    vision_encoder_ratio: Optional[float] = None

    def resolve_steps(self, part: str, num_training_steps: int, args: Any) -> int:
        """Resolve warmup steps for a given component following the fallback chain."""

        def clamp(x: int) -> int:
            return max(0, min(int(x), num_training_steps))

        # 1. Per-component override
        part_steps = getattr(self, f"{part}_steps", None)
        part_ratio = getattr(self, f"{part}_ratio", None)
        if part_steps is not None:
            return clamp(part_steps)
        if part_ratio is not None:
            return clamp(int(part_ratio * num_training_steps))

        # 2. Global override
        if self.global_steps is not None:
            return clamp(self.global_steps)
        if self.global_ratio is not None:
            return clamp(int(self.global_ratio * num_training_steps))

        # 3. TrainingArguments fallback
        arg_steps = getattr(args, "warmup_steps", 0) or 0
        if arg_steps:
            return clamp(int(arg_steps))
        arg_ratio = getattr(args, "warmup_ratio", 0.0) or 0.0
        if arg_ratio:
            return clamp(int(float(arg_ratio) * num_training_steps))

        return 0


# =============================================================================
# Per-Component Optimizer Builder
# =============================================================================


def _is_no_decay_param(
    name: str, p: torch.nn.Parameter, *, exclude_embeddings: bool
) -> bool:
    """Determine if a parameter should be excluded from weight decay."""
    n = name.lower()
    if n.endswith(".bias"):
        return True
    if "norm" in n or "layernorm" in n or "rmsnorm" in n:
        return True
    if p.ndim == 1:
        return True
    if exclude_embeddings and ("embed" in n or "embedding" in n):
        return True
    return False


class PerComponentOptimizerBuilder:
    """Builds an AdamW optimizer with per-component LR groups and weight decay policies."""

    def __init__(
        self,
        text_model_lr: float,
        projection_lr: float,
        vision_encoder_lr: float,
        exclude_wd_norms_embeddings: bool,
    ):
        self._lr_map: Dict[str, float] = {
            "text_model": text_model_lr,
            "projection": projection_lr,
            "vision_encoder": vision_encoder_lr,
        }
        self._exclude_wd = exclude_wd_norms_embeddings

    def build(
        self, model: torch.nn.Module, args: Any
    ) -> tuple[torch.optim.Optimizer, list[dict]]:
        """
        Build optimizer and return (optimizer, group_meta).
        group_meta[i] = {"part": str} aligns with optimizer.param_groups[i].
        """
        optimizer_kwargs: Dict[str, Any] = {
            "lr": args.learning_rate,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        try:
            if (
                getattr(args, "optim", None) == "adamw_torch_fused"
                and torch.cuda.is_available()
                and "fused" in inspect.signature(torch.optim.AdamW).parameters
            ):
                optimizer_kwargs["fused"] = True
        except Exception:
            pass

        groups: Dict[tuple, Dict[str, Any]] = {}
        seen: set[int] = set()

        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))

            part = param_part(name)
            no_decay = _is_no_decay_param(name, p, exclude_embeddings=self._exclude_wd)
            key = (part, "no_decay" if no_decay else "decay")

            if key not in groups:
                groups[key] = {
                    "params": [],
                    "lr": self._lr_map.get(part, self._lr_map["text_model"]),
                    "weight_decay": (0.0 if no_decay else float(args.weight_decay)),
                    "part": part,
                }
            groups[key]["params"].append(p)

        param_groups = [g for g in groups.values() if g["params"]]
        if not param_groups:
            raise RuntimeError("No trainable parameters found when creating optimizer.")

        group_meta = [{"part": g["part"]} for g in param_groups]
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        return optimizer, group_meta


# =============================================================================
# Per-Component LR Scheduler
# =============================================================================


def build_per_component_scheduler(
    optimizer: torch.optim.Optimizer,
    group_meta: list[dict],
    warmup_cfg: WarmupConfig,
    num_training_steps: int,
    args: Any,
) -> LambdaLR:
    """Build a LambdaLR scheduler with per-component warmup durations."""
    scheduler_type = str(getattr(args, "lr_scheduler_type", "linear")).lower()
    lr_kwargs = getattr(args, "lr_scheduler_kwargs", None) or {}
    num_cycles = float(lr_kwargs.get("num_cycles", 1.0))
    power = float(lr_kwargs.get("power", 1.0))

    def _make_lambda(part: str):
        warmup_steps = warmup_cfg.resolve_steps(part, num_training_steps, args)

        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Decay phase
            denom = float(max(1, num_training_steps - warmup_steps))
            progress = min(max(float(current_step - warmup_steps) / denom, 0.0), 1.0)

            if scheduler_type in ("constant", "constant_with_warmup"):
                return 1.0
            if scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            if scheduler_type == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if scheduler_type == "cosine_with_restarts":
                if num_cycles <= 0:
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                return 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0)))
            if scheduler_type == "polynomial":
                return max(0.0, (1.0 - progress) ** power)
            # Fallback: linear
            return max(0.0, 1.0 - progress)

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[_make_lambda(m.get("part", "text_model")) for m in group_meta],
    )


# =============================================================================
# Trainer Callbacks
# =============================================================================


class SaveProcessorCallback(TrainerCallback):
    """Save processor artifacts alongside every checkpoint."""

    def __init__(self, processor: Any):
        super().__init__()
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _save_processor_artifacts(self.processor, ckpt_dir)
        return control


class TokenCountMetadataCallback(TrainerCallback):
    """
    Write run_metadata.json (with token counts) on every checkpoint save
    into both the checkpoint directory and the run root directory.
    """

    def __init__(self, trainer: UnifiedHydraTrainer):
        super().__init__()
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if not self.trainer.is_world_process_zero():
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer._metadata.write(ckpt_dir)

        if self.trainer.run_output_dir:
            self.trainer._metadata.write(self.trainer.run_output_dir)
        return control


class EMACallback(TrainerCallback):
    """
    Integrates :class:`EMATracker` with the HF Trainer lifecycle:

    * **on_step_end** — updates shadow parameters (fires after optimizer.step).
    * **on_save** — persists ``ema_state.pt`` inside every checkpoint directory.
    * **on_train_end** — logs final EMA statistics.

    This callback is *only* added when ``ema.enabled=true`` in config.
    """

    def __init__(self, ema: EMATracker):
        super().__init__()
        self.ema = ema

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update EMA shadows after every optimizer step."""
        if model is not None:
            self.ema.update(model)
        return control

    def on_save(self, args, state, control, **kwargs):
        """Persist compact EMA state alongside each checkpoint (rank-0 only)."""
        if not getattr(args, "should_save", False):
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.ema.save(ckpt_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"[EMA] Training finished. {self.ema}")
        return control


class SetTrainDatasetEpochCallback(TrainerCallback):
    """
    Calls train_dataset.set_epoch(epoch) at each epoch boundary so the dataset
    reshuffles deterministically per epoch.

    Uses an internal counter (not state.epoch) because state.epoch may not yet
    reflect the new epoch when on_epoch_begin fires.  Handles checkpoint resume
    by initialising the counter from state.epoch on the first call.
    """

    def __init__(self, train_dataset: Any):
        super().__init__()
        self.train_dataset = train_dataset
        self._epochs_started: int = 0
        self._initialised: bool = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        ds = self.train_dataset
        if ds is None or not hasattr(ds, "set_epoch"):
            return control

        # On first call, seed counter from state.epoch to handle checkpoint resume
        if not self._initialised:
            self._initialised = True
            if state.epoch is not None and state.epoch >= 1.0:
                self._epochs_started = int(state.epoch)

        epoch = self._epochs_started
        try:
            ds.set_epoch(epoch)
            logger.info(f"[dataset] set_epoch({epoch})")
        except Exception as e:
            logger.warning(f"[dataset] Failed to set_epoch({epoch}): {e}")

        self._epochs_started += 1
        return control


# =============================================================================
# LR Log Extraction
# =============================================================================


def _collect_lr_logs(optimizer: Any, group_meta: Optional[list]) -> Dict[str, float]:
    """Extract per-component learning rates from optimizer param groups."""
    logs: Dict[str, float] = {}
    if optimizer is None or not hasattr(optimizer, "param_groups"):
        return logs

    if not isinstance(group_meta, list) or len(group_meta) != len(
        optimizer.param_groups
    ):
        # Fallback: just log the first group
        if optimizer.param_groups:
            lr0 = optimizer.param_groups[0].get("lr")
            if lr0 is not None:
                logs["unified/lr_group0"] = float(lr0)
        return logs

    by_part: Dict[str, List[float]] = {
        "text_model": [],
        "projection": [],
        "vision_encoder": [],
    }
    for i, g in enumerate(optimizer.param_groups):
        part = group_meta[i].get("part")
        lr = g.get("lr")
        if part in by_part and lr is not None:
            by_part[part].append(float(lr))

    for part, lrs in by_part.items():
        if lrs:
            logs[f"unified/lr_{part}"] = sum(lrs) / len(lrs)

    return logs


# =============================================================================
# Unified Hydra Trainer
# =============================================================================


class UnifiedHydraTrainer(Trainer):
    """
    Trainer for unified multimodal/text training with Hydra multi-head attention.

    Key behaviours:
      - k (current_num_heads) is sampled once per gradient-accumulation window
        and held fixed across all micro-batches in that window.
      - Loss components (CE, z-loss, KD) are tracked per micro-batch, averaged
        per optimizer step, and reported per logging step.
      - Token counts are all-reduced across ranks and persisted to run_metadata.json.
      - EMA shadow parameters are updated once per optimizer step via callback.
    """

    def __init__(
        self,
        *args,
        # Hydra heads
        heads: Optional[List[int]] = None,
        head_weights: Optional[List[float]] = None,
        processor: Any = None,
        # Per-component learning rates
        text_model_lr: float = 3e-4,
        projection_lr: Optional[float] = None,
        vision_encoder_lr: Optional[float] = None,
        # Loss
        loss_type: str = "ce_plus_z",
        z_loss_alpha: float = 0.0,
        # Knowledge distillation
        teacher_model: Optional[torch.nn.Module] = None,
        teacher_full_heads: Optional[int] = None,
        kd_temperature: float = 2.0,
        kd_weight: float = 0.0,
        kd_every_n_steps: int = 1,
        teacher_device_map: Optional[str] = None,
        # Optimizer
        exclude_wd_norms_embeddings: bool = False,
        # Warmup (global)
        global_warmup_steps: Optional[int] = None,
        global_warmup_ratio: Optional[float] = None,
        # Warmup (per-component)
        text_model_warmup_steps: Optional[int] = None,
        text_model_warmup_ratio: Optional[float] = None,
        projection_warmup_steps: Optional[int] = None,
        projection_warmup_ratio: Optional[float] = None,
        vision_encoder_warmup_steps: Optional[int] = None,
        vision_encoder_warmup_ratio: Optional[float] = None,
        # EMA
        ema_config: Optional[EMAConfig] = None,
        # Output
        run_output_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ---- Delegates ----
        self._head_sampler = HeadSampler(heads or [], head_weights)
        self._loss_tracker = LossTracker()
        self._metadata = RunMetadataManager()
        self._kd = KDHelper(
            teacher_model=teacher_model,
            teacher_full_heads=teacher_full_heads,
            temperature=kd_temperature,
            weight=kd_weight,
            every_n_steps=kd_every_n_steps,
            device_map=teacher_device_map,
        )
        self._opt_builder = PerComponentOptimizerBuilder(
            text_model_lr=float(text_model_lr),
            projection_lr=float(
                projection_lr if projection_lr is not None else text_model_lr
            ),
            vision_encoder_lr=float(
                vision_encoder_lr if vision_encoder_lr is not None else text_model_lr
            ),
            exclude_wd_norms_embeddings=exclude_wd_norms_embeddings,
        )
        self._warmup_cfg = WarmupConfig(
            global_steps=_opt_int(global_warmup_steps),
            global_ratio=_opt_float(global_warmup_ratio),
            text_model_steps=_opt_int(text_model_warmup_steps),
            text_model_ratio=_opt_float(text_model_warmup_ratio),
            projection_steps=_opt_int(projection_warmup_steps),
            projection_ratio=_opt_float(projection_warmup_ratio),
            vision_encoder_steps=_opt_int(vision_encoder_warmup_steps),
            vision_encoder_ratio=_opt_float(vision_encoder_warmup_ratio),
        )

        # ---- EMA ----
        ema_config = ema_config or EMAConfig()
        self._ema: Optional[EMATracker] = None
        if ema_config.enabled:
            self._ema = EMATracker(
                self.model,
                decay=ema_config.decay,
                trainable_only=True,
            )
            self.add_callback(EMACallback(self._ema))
            logger.info(f"[EMA] Enabled: {self._ema}")

        # Simple scalar config
        self.processor = processor
        self.loss_type = (loss_type or "ce_plus_z").lower()
        self.z_loss_alpha = float(z_loss_alpha)
        self.run_output_dir = run_output_dir

        # Optimizer group metadata (populated by create_optimizer)
        self._optimizer_group_meta: Optional[list[dict]] = None

        # k is held fixed across an accumulation window; None means "needs sampling"
        self._step_k: Optional[int] = None

        # ---- Other callbacks ----
        if self.processor is not None:
            self.add_callback(SaveProcessorCallback(self.processor))
        self.add_callback(TokenCountMetadataCallback(self))
        self.add_callback(SetTrainDatasetEpochCallback(self.train_dataset))

    # ------------------------------------------------------------------
    # EMA public API
    # ------------------------------------------------------------------

    @property
    def ema(self) -> Optional[EMATracker]:
        """Access the EMA tracker (None when EMA is disabled)."""
        return self._ema

    def load_ema_from_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Restore EMA shadow parameters from a checkpoint.
        Call **before** ``trainer.train(resume_from_checkpoint=...)``.
        Returns True if state was loaded, False if no EMA state was found.
        """
        if self._ema is None:
            return False
        loaded = self._ema.load(checkpoint_dir)
        if loaded:
            logger.info(f"[EMA] Resumed from {checkpoint_dir}: {self._ema}")
        return loaded

    def save_ema_model(self, output_dir: str) -> Optional[str]:
        """
        Save a full model copy with EMA weights applied.

        Creates ``output_dir/ema_model/`` containing the model, tokenizer,
        and processor — ready for inference.  Training weights are restored
        after saving.

        Returns the EMA model directory path, or None if EMA is disabled.
        """
        if self._ema is None:
            return None

        ema_dir = os.path.join(output_dir, EMA_MODEL_SUBDIR)
        logger.info(f"[EMA] Saving EMA model to {ema_dir}")

        with self._ema.average_parameters(self.model):
            # Use Trainer's save logic (handles DeepSpeed, FSDP, etc.)
            super().save_model(ema_dir)
            _save_processor_artifacts(self.processor, ema_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(ema_dir)

        logger.info(f"[EMA] EMA model saved to {ema_dir}")
        return ema_dir

    # ------------------------------------------------------------------
    # Metadata public API
    # ------------------------------------------------------------------

    def write_run_metadata(self, directory: str):
        self._metadata.write(directory)

    def load_token_counts_from_checkpoint(self, checkpoint_dir: str):
        self._metadata.load_from_checkpoint(checkpoint_dir)

    # ------------------------------------------------------------------
    # DataLoader override
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        """
        The unified mixer yields *already-collated batches*.
        We disable auto-batching (batch_size=None) and skip HF's
        IterableDataset sharding since our dataset already shards by rank.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        nw = int(getattr(self.args, "dataloader_num_workers", 0) or 0)
        if nw != 0:
            logger.warning(
                f"[dataloader] Forcing dataloader_num_workers=0 (was {nw}) "
                "because the batch-yielding dataset must not be duplicated."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=None,
            collate_fn=default_convert,
            num_workers=0,
            pin_memory=bool(getattr(self.args, "dataloader_pin_memory", True)),
            persistent_workers=False,
        )

    # ------------------------------------------------------------------
    # DS/NCCL gather fix
    # ------------------------------------------------------------------

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """
        Override to ensure num_items_in_batch is a CUDA dense tensor,
        preventing a ValueError from NCCL gather on CPU tensors.
        """
        batch_samples = []
        num_items = 0
        for _ in range(num_batches):
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                break
            batch_samples.append(batch)
            num_items += _infer_batch_size(batch)

        num_items_in_batch = _make_cuda_long_tensor(num_items, device)

        try:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
        except Exception:
            pass

        return batch_samples, num_items_in_batch

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self.optimizer, self._optimizer_group_meta = self._opt_builder.build(
            self.model, self.args
        )
        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if optimizer is None:
            optimizer = self.optimizer or self.create_optimizer()

        meta = self._optimizer_group_meta
        if not isinstance(meta, list) or len(meta) != len(optimizer.param_groups):
            meta = [{"part": "text_model"} for _ in optimizer.param_groups]

        self.lr_scheduler = build_per_component_scheduler(
            optimizer=optimizer,
            group_meta=meta,
            warmup_cfg=self._warmup_cfg,
            num_training_steps=num_training_steps,
            args=self.args,
        )
        return self.lr_scheduler

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        # 1. Token counting (never break training)
        try:
            self._metadata.accumulate_from_batch(inputs, model)
        except Exception as e:
            logger.warning(f"[tokens] Failed to accumulate: {e}")

        # 2. Sample k once per accumulation window
        if self._step_k is None:
            self._step_k = self._head_sampler.sample()
        k = self._step_k

        # 3. Forward pass
        outputs = model(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=inputs.get("labels"),
            use_cache=False,
            current_num_heads=k,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = inputs.get("labels")

        # 4. Compute loss components
        losses = self._compute_all_losses(outputs, logits, labels, inputs, model)

        # 5. Track and finalize
        self._loss_tracker.record_microbatch(losses)
        if self._is_optimizer_step():
            self._loss_tracker.finalize_optimizer_step(k)
            self._step_k = None

        return (losses.total, outputs) if return_outputs else losses.total

    def _compute_all_losses(
        self,
        outputs: Any,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        inputs: Dict[str, Any],
        model: torch.nn.Module,
    ) -> LossComponents:
        """Compute CE, z-loss, and KD loss; return all components."""
        zero = logits.new_zeros(())

        # CE loss
        ce_loss = zero
        if self.loss_type in ("ce", "ce_plus_z"):
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        # Z loss
        z_loss = zero
        if self.loss_type in ("z_only", "zloss_only", "ce_plus_z", "z"):
            z_loss = compute_z_loss(logits, labels, self.z_loss_alpha)

        # KD loss
        kd_loss = zero
        if self._kd.should_run_this_step(int(self.state.global_step or 0)):
            device = _resolve_tensor_device(labels, inputs.get("attention_mask"), model)
            kd_loss = self._kd.compute(logits, inputs, labels, device)

        total = ce_loss + z_loss + kd_loss

        return LossComponents(ce=ce_loss, z=z_loss, kd=kd_loss, total=total)

    def _is_optimizer_step(self) -> bool:
        """True when gradients are synchronized (end of accumulation window)."""
        try:
            return bool(getattr(self.accelerator, "sync_gradients", True))
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, float], *args, **kwargs):
        logs = dict(logs)

        # Per-component LRs
        try:
            opt = self._resolve_optimizer_for_logging()
            logs.update(_collect_lr_logs(opt, self._optimizer_group_meta))
        except Exception:
            pass

        # Windowed loss stats
        try:
            logs.update(self._loss_tracker.drain_to_logs())
        except Exception:
            pass

        # EMA diagnostics
        if self._ema is not None:
            logs["ema/num_updates"] = float(self._ema.num_updates)

        return super().log(logs, *args, **kwargs)

    def _resolve_optimizer_for_logging(self) -> Any:
        """Find the optimizer, checking self.optimizer and scheduler.optimizer."""
        opt = getattr(self, "optimizer", None)
        if opt is not None:
            return opt
        sched = getattr(self, "lr_scheduler", None)
        if sched is not None:
            return getattr(sched, "optimizer", None)
        return None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        _save_processor_artifacts(self.processor, output_dir or self.args.output_dir)


# =============================================================================
# Module-level utilities
# =============================================================================


def _opt_int(x: Any) -> Optional[int]:
    return int(x) if x is not None else None


def _opt_float(x: Any) -> Optional[float]:
    return float(x) if x is not None else None


def _make_cuda_long_tensor(value: int, device: torch.device) -> torch.Tensor:
    """Create a long tensor for NCCL gather: on the given device, falling back to CUDA, then CPU."""
    try:
        return torch.tensor(int(value), device=device, dtype=torch.long)
    except Exception:
        if torch.cuda.is_available():
            return torch.tensor(
                int(value), device=torch.device("cuda"), dtype=torch.long
            )
        return torch.tensor(int(value), dtype=torch.long)
