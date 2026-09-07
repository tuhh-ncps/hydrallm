from typing import Dict, Optional
import os
import importlib

import torch
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)


def str_to_bool(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Cannot parse boolean from {v}")


def parse_dtype(dtype_str: Optional[str], device: str):
    ds = (dtype_str or "auto").strip().lower()
    if ds == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        return torch.float32
    if ds in ("float32", "fp32"):
        return torch.float32
    if ds in ("bfloat16", "bf16"):
        return torch.bfloat16
    if ds in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def has_flash_attn2() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def pick_default_device() -> str:
    """Return the best available device as a string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_best_attn_impl(requested: Optional[str], device: Optional[str] = None) -> str:
    """
    Decide which attention implementation to use.
    - requested: if provided and valid, use it (with fallback if not available).
    - auto: prefer flash_attention_2 on CUDA if installed, else sdpa on GPU, else eager.
    """
    req = (requested or "auto").lower()
    dev = device or pick_default_device()

    def valid(x: str) -> bool:
        return x in ("flash_attention_2", "sdpa", "eager")

    if req != "auto":
        if valid(req):
            if req == "flash_attention_2" and not has_flash_attn2():
                logger.warning(
                    "Requested flash_attention_2 but FlashAttention-2 is not available; falling back to sdpa/eager."
                )
                return "sdpa" if dev.startswith("cuda") else "eager"
            return req
        logger.warning(
            f"Unknown attention implementation '{requested}', falling back to auto."
        )
    # auto
    if dev.startswith("cuda") and has_flash_attn2():
        return "flash_attention_2"
    if dev != "cpu":
        return "sdpa"
    return "eager"


def safe_int(x, minn=1) -> int:
    return max(minn, int(x))


def normalize_mlp_ratio(value, num_hidden_layers=None):
    """Normalize a scalar or per-layer FFN ratio and validate its shape."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("FFN granularity ratio list cannot be empty")
        ratios = [float(ratio) for ratio in value]
        value = ratios[0] if len(ratios) == 1 else ratios
    else:
        value = float(value)

    ratios = value if isinstance(value, list) else [value]
    if any(ratio <= 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("FFN granularity ratios must be greater than 0 and at most 1")
    if isinstance(value, list) and num_hidden_layers is not None:
        if len(value) != int(num_hidden_layers):
            raise ValueError(
                "Per-layer FFN granularity ratios must contain one value per "
                f"decoder layer (expected {num_hidden_layers}, got {len(value)})"
            )
    return value
