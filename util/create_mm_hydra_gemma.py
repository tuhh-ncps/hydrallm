# util/create_mm_hydra_gemma.py
import argparse
import os
import sys
from typing import Dict, Tuple, Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
)

# Ensure we can import the local model implementations
# util directory is next to model_implementations
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from model_implementations.hydra_gemma_from_flex import HydraGemma3ForConditionalGeneration

try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Config, Gemma3TextConfig
    from transformers import Gemma3ForConditionalGeneration, AutoModelForCausalLM
except Exception as e:
    print("This script requires a recent transformers version with Gemma3 support.")
    raise


def find_proj_weight_key(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Try to find the projector's weight key in a Gemma3 multimodal state_dict.
    Expected candidates:
      - 'model.multi_modal_projector.mm_input_projection_weight' (our local)
      - 'model.multi_modal_projector.mm_input_projection.weight' (HF style)
      - other variants containing 'mm_input_projection'
    """
    # First, exact match to our local name (if loading a previously saved Flex/Hydra ckpt)
    if "model.multi_modal_projector.mm_input_projection_weight" in sd:
        return "model.multi_modal_projector.mm_input_projection_weight"

    # Common HF style key
    candidates = [
        k for k in sd.keys()
        if "multi_modal_projector" in k and "mm_input_projection" in k and k.endswith("weight")
    ]
    if candidates:
        # Prefer a deterministic choice
        candidates.sort()
        return candidates[0]

    # Fallback: search any key mentioning the projector that is 2D
    fallback = [
        k for k, v in sd.items()
        if "multi_modal_projector" in k and v.ndim == 2
    ]
    if fallback:
        fallback.sort()
        return fallback[0]

    return None


def find_soft_emb_norm_key(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Find the projector's soft embedding norm weight key.
    Expected: 'model.multi_modal_projector.mm_soft_emb_norm.weight'
    """
    key = "model.multi_modal_projector.mm_soft_emb_norm.weight"
    if key in sd:
        return key
    # Try any 'mm_soft_emb_norm.weight' occurrence
    candidates = [k for k in sd if k.endswith("mm_soft_emb_norm.weight")]
    if candidates:
        candidates.sort()
        return candidates[0]
    return None


def pad_or_slice_cols(t: torch.Tensor, out_cols: int) -> torch.Tensor:
    """
    Adjust the number of columns of a 2D tensor.
      - If t.shape[1] >= out_cols: slice to [:, :out_cols]
      - If t.shape[1] <  out_cols: right-pad with zeros to out_cols
    """
    assert t.ndim == 2, "Expected a 2D tensor"
    in_cols = t.shape[1]
    if in_cols == out_cols:
        return t
    if in_cols > out_cols:
        return t[:, :out_cols].contiguous()
    # pad
    pad_cols = out_cols - in_cols
    pad = torch.zeros(t.shape[0], pad_cols, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1).contiguous()


def main():
    parser = argparse.ArgumentParser(description="Build a hybrid Gemma3 multimodal checkpoint on CPU.")
    parser.add_argument("--text-model", type=str, default="google/gemma-3-270m",
                        help="Text-only Gemma3 backbone to use (e.g., google/gemma-3-270m).")
    parser.add_argument("--mm-source", type=str, default="google/gemma-3-4b-pt",
                        help="Multimodal Gemma3 source (provides vision tower + projector).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder to save the hybrid model.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load only from local cache.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.set_grad_enabled(False)

    print("Loading configs on CPU...")
    # Load text (small) config: Gemma3TextConfig
    text_cfg = AutoConfig.from_pretrained(
        args.text_model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if not isinstance(text_cfg, Gemma3TextConfig):
        raise ValueError(f"Expected a Gemma3TextConfig from {args.text_model}, got {type(text_cfg)}")

    # Load multimodal (large) config: Gemma3Config
    mm_cfg = AutoConfig.from_pretrained(
        args.mm_source,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if not isinstance(mm_cfg, Gemma3Config):
        raise ValueError(f"Expected a Gemma3Config from {args.mm_source}, got {type(mm_cfg)}")

    # Build a new combined config: keep vision + mm settings from MM source; replace text_config with small text config
    new_cfg_dict = mm_cfg.to_dict()
    new_cfg = Gemma3Config(**new_cfg_dict)
    new_cfg.text_config = Gemma3TextConfig(**text_cfg.to_dict())

    print("Instantiating target Hydra model (CPU)...")
    # Create our target model (uninitialized) on CPU
    target_model = HydraGemma3ForConditionalGeneration(new_cfg).to("cpu")

    # Load source models on CPU
    print("Loading source models on CPU...")
    # Small text LM
    text_src = AutoModelForCausalLM.from_pretrained(
        args.text_model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
    ).to("cpu")

    # Multimodal source (provides vision + projector)
    mm_src = Gemma3ForConditionalGeneration.from_pretrained(
        args.mm_source,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
    ).to("cpu")

    text_hidden = new_cfg.text_config.hidden_size
    vision_hidden = new_cfg.vision_config.hidden_size

    print(f"- Target text hidden size:   {text_hidden}")
    print(f"- Target vision hidden size: {vision_hidden}")

    # Build a state_dict for the target from the two sources
    print("Composing hybrid state_dict...")
    new_sd: Dict[str, torch.Tensor] = {}

    # 1) Text weights from text_src -> target_model
    # text_src keys: 'model.*' for the text backbone and 'lm_head.weight'
    text_sd = text_src.state_dict()
    for k, v in text_sd.items():
        if k.startswith("model."):
            # Map 'model.' -> 'model.language_model.'
            mapped = "model.language_model." + k[len("model."):]
            new_sd[mapped] = v
        elif k == "lm_head.weight":
            new_sd["lm_head.weight"] = v
        # else ignore other keys

    # 2) Vision tower and projector from mm_src -> target_model
    mm_sd = mm_src.state_dict()

    # Vision tower keys typically start with 'model.vision_tower.'
    for k, v in mm_sd.items():
        if k.startswith("model.vision_tower."):
            new_sd[k] = v

    # Projector soft emb norm (if present)
    soft_norm_key_src = find_soft_emb_norm_key(mm_sd)
    if soft_norm_key_src is not None:
        new_sd["model.multi_modal_projector.mm_soft_emb_norm.weight"] = mm_sd[soft_norm_key_src]
    else:
        print("Warning: projector soft emb norm weight not found in multimodal source; leaving default init.")

    # Projector input projection weight
    proj_w_key_src = find_proj_weight_key(mm_sd)
    if proj_w_key_src is None:
        raise RuntimeError("Could not find projector weight ('mm_input_projection_*') in the multimodal source state_dict.")
    proj_w_src = mm_sd[proj_w_key_src]  # shape [vision_hidden_src, text_hidden_src]
    if proj_w_src.ndim != 2:
        raise RuntimeError(f"Projector weight is not 2D: key={proj_w_key_src}, shape={tuple(proj_w_src.shape)}")

    # Validate first dim (vision hidden) and adjust columns to match target text hidden
    if proj_w_src.shape[0] != vision_hidden:
        print(f"Warning: vision hidden mismatch in projector: src {proj_w_src.shape[0]} vs target {vision_hidden}. "
              f"We will slice/pad rows to match.")
        # Conservative approach: slice or pad rows as well (rarely needed)
        if proj_w_src.shape[0] > vision_hidden:
            proj_w_src = proj_w_src[:vision_hidden, :]
        else:
            pad_rows = vision_hidden - proj_w_src.shape[0]
            proj_w_src = torch.cat(
                [proj_w_src, torch.zeros(pad_rows, proj_w_src.shape[1], dtype=proj_w_src.dtype)],
                dim=0,
            ).contiguous()

    proj_w_adj = pad_or_slice_cols(proj_w_src, out_cols=text_hidden)
    new_sd["model.multi_modal_projector.mm_input_projection_weight"] = proj_w_adj

    # Load composed weights into target model (strict=False to ignore irrelevant/missing keys)
    print("Loading composed weights into target model (strict=False)...")
    missing, unexpected = target_model.load_state_dict(new_sd, strict=False)
    if missing:
        print("Missing keys in target (will remain at init):")
        for k in missing:
            print("  -", k)
    if unexpected:
        print("Unexpected keys (ignored):")
        for k in unexpected:
            print("  -", k)

    # Save the hybrid model and config
    print(f"Saving model to: {args.output}")
    target_model.save_pretrained(args.output)

    # Also save a tokenizer for convenience (from the text model)
    try:
        tok = AutoTokenizer.from_pretrained(
            args.text_model,
            local_files_only=args.local_files_only,
            use_fast=True,
            trust_remote_code=True,
        )
        tok.save_pretrained(args.output)
        print("Saved tokenizer from text model.")
    except Exception as e:
        print(f"Warning: failed to save tokenizer from {args.text_model}: {e}")
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            args.mm_source,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        image_processor.save_pretrained(args.output)
        print("Saved image processor from multimodal source.")
    except Exception as e:
        print(f"Warning: failed to save image processor from {args.mm_source}: {e}")
        print("You can manually add it later (e.g., copy preprocessor_config.json from the multimodal source).")
    print("Done. You can now load the model via hydra_gemma_from_flex.get_mm_model using the output folder.")


if __name__ == "__main__":
    main()