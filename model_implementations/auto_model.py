from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForSeq2SeqLM, AutoConfig, AutoModel
from common_helpers import pick_best_attn_impl
from typing import Optional
import torch


def get_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Loads a Causal LM with arbitrary extra kwargs forwarded to from_pretrained.
    Examples of extra kwargs:
      - revision, subfolder, trust_remote_code, device_map, max_memory,
        quantization_config, torch_dtype, low_cpu_mem_usage, rope_scaling, etc.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)

    # Map dtype -> torch_dtype unless explicitly provided
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    # Default to device_map="auto" if not overridden and no explicit device provided
    if "device_map" not in model_kwargs and device is None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )

    # If a specific device was requested and we didn't set a device_map, move the model
    if device is not None and "device_map" not in model_kwargs:
        try:
            model.to(device)
        except Exception:
            pass

    return model


def _from_pretrained_with_optional_attn(loader_cls, name, attn_impl, **kwargs):
    # Some models don't accept attn_implementation; retry without if needed
    try:
        return loader_cls.from_pretrained(name, attn_implementation=attn_impl, **kwargs)
    except TypeError:
        return loader_cls.from_pretrained(name, **kwargs)


def get_mm_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Multimodal "auto-style" loader that returns a model with .generate.

    Strategy:
      1) If config.model_type == "gemma3": load Gemma3ForConditionalGeneration (has generate).
      2) Else try AutoModelForCausalLM with trust_remote_code=True (many MM models register their
         own *ForConditionalGeneration classes under this auto head).
      3) Else try AutoModelForSeq2SeqLM (in case a model is encoder-decoder multimodal).
      4) Else use AutoModel as a last resort and error out if .generate is missing.

    Also:
      - Maps dtype -> torch_dtype if not provided.
      - Picks a good attention implementation for the target device.
      - Defaults to device_map="auto" when no explicit device is given.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)

    # Clean kwargs to avoid passing None
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Prefer remote code for better multimodal support (LLaVA, Qwen-VL, etc.)
    model_kwargs.setdefault("trust_remote_code", True)

    # Map dtype -> torch_dtype unless already set
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    # Default to device_map="auto" if no explicit device_map and no explicit device was requested
    if "device_map" not in model_kwargs and device is None:
        model_kwargs["device_map"] = "auto"

    # Inspect config to specialize Gemma3
    cfg = AutoConfig.from_pretrained(model_name_or_path, **{k: v for k, v in model_kwargs.items()
                                                            if k in ("revision", "subfolder", "trust_remote_code",
                                                                     "cache_dir", "token", "local_files_only",
                                                                     "force_download", "proxies", "use_auth_token")})
    # 1) Gemma3 multimodal path: use the conditional generation head with .generate [2]
    if getattr(cfg, "model_type", None) == "gemma3":
        try:
            from transformers import Gemma3ForConditionalGeneration
            model = _from_pretrained_with_optional_attn(
                Gemma3ForConditionalGeneration,
                model_name_or_path,
                resolved_impl,
                **model_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Gemma3ForConditionalGeneration for '{model_name_or_path}': {e}"
            ) from e
    else:
        # 2) Broad compatibility: many VLMs register their conditional generation class under this head
        try:
            model = _from_pretrained_with_optional_attn(
                AutoModelForCausalLM,
                model_name_or_path,
                resolved_impl,
                **model_kwargs,
            )
        except Exception:
            # 3) Try seq2seq LM (encoder-decoder multimodal families)
            try:
                model = _from_pretrained_with_optional_attn(
                    AutoModelForSeq2SeqLM,
                    model_name_or_path,
                    resolved_impl,
                    **model_kwargs,
                )
            except Exception:
                # 4) Last resort: base backbone; warn if it has no generate
                model = _from_pretrained_with_optional_attn(
                    AutoModel,
                    model_name_or_path,
                    resolved_impl,
                    **model_kwargs,
                )
                if not hasattr(model, "generate"):
                    raise RuntimeError(
                        "Loaded base model with AutoModel, but it does not implement `.generate`. "
                        "Please load a conditional-generation head (e.g., *ForConditionalGeneration) "
                        "or a task-specific Auto head that provides generate."
                    )

    # Move to device if explicitly requested and no device_map was set
    if device is not None and "device_map" not in model_kwargs:
        try:
            model.to(device)
        except Exception:
            pass

    return model
