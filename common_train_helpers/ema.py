"""
Exponential Moving Average (EMA) of model parameters.

Standalone, reusable module — no dependency on HF Trainer or DeepSpeed.
Integrates with any PyTorch training loop via update / apply / restore.

    shadow[name] = decay * shadow[name] + (1 - decay) * param

The effective averaging window is ≈ 1 / (1 - decay) optimizer steps.
Examples:
    decay=0.999      → ~1 000 step window
    decay=0.98666667 → ~75   step window
    decay=0.9        → ~10   step window
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

EMA_STATE_FILENAME = "ema_state.pt"
EMA_MODEL_SUBDIR = "ema_model"


class EMATracker:
    """
    Maintains exponential moving averages of model parameters.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be tracked.  If the model is wrapped
        (e.g. DeepSpeed engine, FSDP), the wrapper's ``.module`` attribute
        is used transparently.
    decay : float
        EMA coefficient in [0, 1].  Higher → slower averaging.
    trainable_only : bool
        If True (default), only track parameters with ``requires_grad=True``.

    Notes
    -----
    * Shadow parameters live on the **same device** as the source parameters.
    * With ZeRO-1/2 the full parameters are replicated on every rank, so
      calling ``update`` on every rank keeps shadows consistent.
    * ZeRO-3 shards parameters; EMA would need gather/scatter logic.
      This implementation does **not** support ZeRO-3.

    Typical usage::

        ema = EMATracker(model, decay=0.999)

        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update(model)               # after every optimizer step

        with ema.average_parameters(model): # temporarily swap in EMA weights
            evaluate(model)

        ema.save(checkpoint_dir)            # persist alongside checkpoint
        ema.load(checkpoint_dir)            # restore on resume
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        *,
        trainable_only: bool = True,
    ):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")

        self.decay = decay
        self._trainable_only = trainable_only
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._num_updates: int = 0

        for name, param in self._iter_tracked(model):
            self._shadow[name] = param.data.clone().detach()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        """Unwrap DeepSpeed / FSDP / DataParallel engines."""
        return getattr(model, "module", model)

    def _iter_tracked(self, model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield ``(name, param)`` for every tracked parameter."""
        for name, param in self._unwrap(model).named_parameters():
            if self._trainable_only and not param.requires_grad:
                continue
            yield name, param

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters from current model parameters.
        Call **once per optimizer step** (not per micro-batch).
        """
        self._num_updates += 1
        for name, param in self._iter_tracked(model):
            if name in self._shadow:
                self._shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """
        Copy EMA weights into the model for evaluation or saving.

        The current training weights are saved internally; call
        :meth:`restore` afterwards to return to training weights.
        """
        if self._backup:
            logger.warning(
                "[EMA] apply_shadow() called without prior restore(). "
                "Overwriting existing backup."
            )
        self._backup.clear()
        for name, param in self._iter_tracked(model):
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """
        Restore training weights after :meth:`apply_shadow`.
        No-op if ``apply_shadow`` was not called.
        """
        if not self._backup:
            return
        for name, param in self._iter_tracked(model):
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Context manager: applies EMA weights on entry, restores on exit.

        Usage::

            with ema.average_parameters(model):
                evaluate(model)
                model.save_pretrained(path)
        """
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialise EMA state for checkpointing (shadow params moved to CPU)."""
        return {
            "decay": self.decay,
            "num_updates": self._num_updates,
            "shadow": {k: v.cpu().clone() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore EMA state from a prior :meth:`state_dict`."""
        self.decay = float(state.get("decay", self.decay))
        self._num_updates = int(state.get("num_updates", 0))
        saved = state.get("shadow", {})

        loaded = skipped = 0
        for name in list(self._shadow.keys()):
            if name not in saved:
                skipped += 1
                continue
            src = saved[name]
            if src.shape != self._shadow[name].shape:
                logger.warning(
                    f"[EMA] Shape mismatch for '{name}': "
                    f"shadow={self._shadow[name].shape}, "
                    f"checkpoint={src.shape}. Keeping current shadow."
                )
                skipped += 1
                continue
            self._shadow[name].copy_(src.to(self._shadow[name].device))
            loaded += 1

        logger.info(
            f"[EMA] Loaded {loaded} shadow params, skipped {skipped}, "
            f"num_updates={self._num_updates}"
        )

    def save(self, directory: str, filename: str = EMA_STATE_FILENAME) -> str:
        """Save EMA state to ``directory/filename``.  Returns the file path."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save(self.state_dict(), path)
        logger.info(f"[EMA] Saved state to {path}")
        return path

    def load(self, directory: str, filename: str = EMA_STATE_FILENAME) -> bool:
        """
        Load EMA state from ``directory/filename`` if the file exists.
        Returns True on success, False if the file was not found.
        """
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            logger.info(f"[EMA] No state file at {path}, skipping load.")
            return False
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # older PyTorch without weights_only
            state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_updates(self) -> int:
        """Number of ``update()`` calls so far."""
        return self._num_updates

    @property
    def num_shadow_params(self) -> int:
        """Number of tracked parameter tensors."""
        return len(self._shadow)

    @property
    def shadow_numel(self) -> int:
        """Total number of scalar elements across all shadow parameters."""
        return sum(v.numel() for v in self._shadow.values())

    @property
    def effective_window(self) -> float:
        """
        Approximate averaging window in optimizer steps: ``1 / (1 - decay)``.

        Examples: decay=0.999 → 1000, decay=0.98667 → ~75, decay=0.9 → 10.
        """
        if self.decay >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - self.decay)

    def __repr__(self) -> str:
        return (
            f"EMATracker(decay={self.decay}, "
            f"params={self.num_shadow_params}, "
            f"numel={self.shadow_numel / 1e6:.2f}M, "
            f"window≈{self.effective_window:.0f}, "
            f"updates={self._num_updates})"
        )
