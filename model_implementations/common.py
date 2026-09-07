from typing import Dict, Optional
import os
import importlib
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import logging as hf_logging
from common_helpers import pick_best_attn_impl

logger = hf_logging.get_logger(__name__)


def load_tokenizer_with_fallback(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    use_fast: bool = True,
):
    tried = []

    def _try_load(name: str):
        tried.append(name)
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
            logger.info(f"Loaded tokenizer from: {name}")
            return tok
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from '{name}': {e}")
            return None

    if tokenizer_name_or_path:
        tok = _try_load(tokenizer_name_or_path)
        if tok is not None:
            return tok

    tok = _try_load(model_name_or_path)
    if tok is not None:
        return tok

    fallback = "google/gemma-3-270m"
    tok = _try_load(fallback)
    if tok is not None:
        return tok

    raise RuntimeError(f"Could not load tokenizer. Tried: {tried}")


def _load_config_for_autodetect(model_name_or_path: str, **model_kwargs):
    """
    Load a config with a safe subset of kwargs for AutoConfig.
    """
    allowed = {
        "revision", "subfolder", "trust_remote_code", "cache_dir",
        "token", "local_files_only", "force_download", "proxies", "use_auth_token"
    }
    cfg_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
    try:
        return AutoConfig.from_pretrained(model_name_or_path, **cfg_kwargs)
    except Exception as e:
        logger.warning(
            f"AutoConfig.from_pretrained failed for '{model_name_or_path}' ({e}); assuming text-only.")
        return None


def _is_multimodal_config(cfg) -> bool:
    """
    Decide if a config is multimodal. For Gemma3, the multimodal config is Gemma3Config,
    which has model_type='gemma3' and includes a vision_config sub-config.
    """
    if cfg is None:
        return False
    if getattr(cfg, "model_type", None) == "gemma3":
        return True  # Gemma3Config (multimodal)
    if hasattr(cfg, "vision_config"):
        return True  # any config that exposes a vision sub-config
    return False


def load_model(
    model_name_or_path: str,
    implementation: Optional[str] = "auto_model",
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    multimodal: Optional[bool] = None,
    **model_kwargs,
):
    """
    Load a model by specifying the implementation module filename.

    Auto-selects text-only vs multimodal variant:
      - If multimodal is None, we inspect the checkpoint config. If it has a vision_config
        or model_type == 'gemma3', we treat it as multimodal.
      - If multimodal is True, we call get_mm_model (if available), otherwise we fall back
        to flex_gemma.get_mm_model.
      - If multimodal is False, we call get_model (text-only) and keep the original fallback
        to auto_model.get_model.

    attn_implementation: "auto" | "flash_attention_2" | "sdpa" | "eager"
    device: optional hint for picking the best impl when requested="auto".
    Extra model_kwargs are forwarded to the underlying get_model/get_mm_model and then to from_pretrained.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)

    # Strip None values once here (avoid passing subfolder=None etc.)
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Map dtype -> torch_dtype unless explicitly provided
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    # Autodetect multimodality if not forced
    cfg = _load_config_for_autodetect(model_name_or_path, **model_kwargs)
    is_mm = _is_multimodal_config(
        cfg) if multimodal is None else bool(multimodal)

    # Import the requested implementation module
    module_path = f"model_implementations.{implementation}"
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        logger.warning(
            f"Failed to import '{implementation}' ({e}); falling back to AutoModelForCausalLM aka auto_model.")
        implementation = "auto_model"
        module_path = f"model_implementations.{implementation}"
        module = importlib.import_module(module_path)

    # Pick the entry point based on detected modality
    entry_fn_name = "get_mm_model" if is_mm else "get_model"
    entry_fn = getattr(module, entry_fn_name, None)

    # If we need multimodal but the chosen implementation has no get_mm_model, fall back to flex_gemma.
    if is_mm and entry_fn is None:
        if implementation != "flex_gemma":
            logger.warning(
                f"Implementation '{implementation}' has no '{entry_fn_name}'. "
                f"Falling back to 'flex_gemma.{entry_fn_name}'."
            )
            flex_module = importlib.import_module(
                "model_implementations.flex_gemma")
            entry_fn = getattr(flex_module, entry_fn_name, None)
        if entry_fn is None:
            raise RuntimeError(
                f"No multimodal loader '{entry_fn_name}' available in '{implementation}' or 'flex_gemma'."
            )

    # If we need text-only and the chosen implementation lacks get_model, fall back as before to auto_model
    if not is_mm and entry_fn is None:
        logger.warning(
            f"Failed to find '{entry_fn_name}' in '{implementation}'; "
            f"falling back to AutoModelForCausalLM aka auto_model."
        )
        implementation = "auto_model"
        module = importlib.import_module(
            f"model_implementations.{implementation}")
        entry_fn = getattr(module, "get_model")

    # Call the resolved factory
    try:
        model = entry_fn(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=resolved_impl,
            device=device,
            **model_kwargs,
        )
    except Exception as e:
        # Final fallback path:
        if is_mm:
            # If multimodal load failed, try flex_gemma.get_mm_model once
            try:
                logger.warning(
                    f"Primary multimodal load failed in '{implementation}': {e}. "
                    f"Retrying with 'flex_gemma.get_mm_model'."
                )
                flex_module = importlib.import_module(
                    "model_implementations.flex_gemma")
                model = getattr(flex_module, "get_mm_model")(
                    model_name_or_path,
                    dtype=dtype,
                    attn_implementation=resolved_impl,
                    device=device,
                    **model_kwargs,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load multimodal model: {e2}") from e2
        else:
            logger.warning(
                f"Primary text-only load failed in '{implementation}': {e}. "
                f"Falling back to AutoModelForCausalLM aka auto_model."
            )
            auto_module = importlib.import_module(
                "model_implementations.auto_model")
            model = getattr(auto_module, "get_model")(
                model_name_or_path,
                dtype=dtype,
                attn_implementation=resolved_impl,
                device=device,
                **model_kwargs,
            )

    # Move to device if requested and no device_map in kwargs
    if device:
        try:
            # Don't override an explicit device_map
            if "device_map" not in model_kwargs:
                model.to(device)
        except Exception:
            pass

    return model
