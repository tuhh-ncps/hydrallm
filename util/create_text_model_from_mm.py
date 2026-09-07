# util/create_text_model_from_mm.py
"""
Remove the vision encoder and multimodal projector from a hybrid Gemma3
checkpoint OR a standard Hugging Face Gemma3 checkpoint, leaving only the text
backbone and lm_head.

Supports:
  1. Local hybrid checkpoints (built by create_mm_hydra_gemma.py)
  2. Hugging Face model IDs (e.g., "google/gemma-3-4b-pt")

Usage:
    # From local hybrid checkpoint
    python util/create_text_model_from_mm.py \
        --input  outputs/amrum/trained_models/align_projection_from_text_finetune_20260215_102329/model \
        --output outputs/amrum/trained_models/align_projection_from_text_finetune_20260215_102329/model/txt_only \
        --keep-tokenizer

    # From Hugging Face model ID
    python util/create_text_model_from_mm.py \
        --input  google/gemma-3-4b-pt \
        --output /path/to/text_only_checkpoint \
        --keep-tokenizer
"""
import argparse
import json
import os
import shutil
import sys
from typing import Dict, Optional
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ---------------------------------------------------------------------------
# Key classification helpers
# ---------------------------------------------------------------------------
VISION_PREFIXES = (
    "model.vision_tower.",
    "vision_tower.",
    "vision_model.",  # Common in standard HF checkpoints
    "model.vision_model.",
)
PROJECTOR_PREFIXES = (
    "model.multi_modal_projector.",
    "multi_modal_projector.",
)
LANGUAGE_PREFIX = "model.language_model."


def is_vision_key(key: str) -> bool:
    return any(key.startswith(p) for p in VISION_PREFIXES)


def is_projector_key(key: str) -> bool:
    return any(key.startswith(p) for p in PROJECTOR_PREFIXES)


def is_removable(key: str) -> bool:
    return is_vision_key(key) or is_projector_key(key)


def remap_text_key(key: str) -> str:
    """
    Remap a hybrid-checkpoint key to the text-only namespace.
    In the hybrid checkpoint the text weights live under
    ``model.language_model.*``.  A standalone Gemma3 text model expects
    them under ``model.*``.

    If the key is already in the standard format (e.g. from a pure HF download),
    we ensure it doesn't get double-prefixed.
    """
    if key.startswith(LANGUAGE_PREFIX):
        return "model." + key[len(LANGUAGE_PREFIX) :]

    # If it's already a standard text model key (e.g. "model.layers..."),
    # return as is.
    return key


# ---------------------------------------------------------------------------
# State-dict manipulation
# ---------------------------------------------------------------------------


def strip_and_remap(sd: Dict[str, torch.Tensor]):
    """
    Return a new state-dict with:
      - all vision-tower and projector keys removed
      - language-model keys remapped to text-only namespace
    """
    new_sd: Dict[str, torch.Tensor] = {}
    removed = []
    kept = []

    for key, value in sd.items():
        if is_removable(key):
            removed.append(key)
            continue

        new_key = remap_text_key(key)
        new_sd[new_key] = value
        kept.append((key, new_key))

    return new_sd, removed, kept


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------


def build_text_only_config(input_dir: str) -> dict:
    """
    Read the config.json and extract just the text_config portion,
    promoting it to a top-level Gemma3TextConfig.
    """
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Check if it's a hybrid config with nested text_config
    text_cfg = cfg.get("text_config", None)

    if text_cfg is None:
        # Assume it's already a text-only config (standard HF model)
        # We just need to clean up any potential vision fields that might exist
        # even in a "text" model if it was saved weirdly, though usually they don't.
        text_cfg = dict(cfg)
        print("Detected standard text config structure.")
    else:
        print("Detected hybrid config structure (nested text_config).")

    # Ensure model_type is correct for standalone loading
    # Gemma3 text-only usually expects "gemma" or "gemma3" depending on the specific HF version
    # We preserve the original model_type if it looks correct, or force it if needed.
    if "model_type" not in text_cfg:
        text_cfg["model_type"] = "gemma"

    # Remove any leftover vision / multimodal fields
    unwanted_keys = [
        "vision_config",
        "mm_tokens_per_image",
        "image_token_index",
        "boi_token_index",
        "eoi_token_index",
        "multi_modal_projector",
        "vision_tower",
        "num_image_tokens",
    ]

    for unwanted in unwanted_keys:
        text_cfg.pop(unwanted, None)

    # If we extracted from nested, we might need to promote some top-level args
    # that were only in the parent config (like vocab_size) if they weren't in text_cfg
    # But usually text_cfg is complete.

    return text_cfg


# ---------------------------------------------------------------------------
# Input Resolution (Local vs Hugging Face)
# ---------------------------------------------------------------------------


def resolve_input_path(input_arg: str) -> str:
    """
    If input_arg is a local directory, return it.
    If input_arg is a Hugging Face model ID, download it to the Hugging Face
    cache (via snapshot_download) and return the local path.
    """
    if os.path.isdir(input_arg):
        return input_arg

    print(
        f"Input '{input_arg}' is not a local directory. Attempting to load from Hugging Face..."
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "To load models from Hugging Face IDs, please install huggingface_hub: "
            "pip install huggingface_hub"
        )

    # snapshot_download stores the repo in the standard HF cache and returns
    # the local path; ignore_patterns skips non-essential docs.
    print(f"Downloading model '{input_arg}' to cache...")
    local_path = snapshot_download(
        repo_id=input_arg,
        repo_type="model",
        ignore_patterns=["*.md", "*.txt"],  # Ignore readme/etc if desired
    )

    print(f"Successfully downloaded to: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# File-level checkpoint loading helpers
# ---------------------------------------------------------------------------


def load_state_dict_from_dir(input_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load a state dict from a HuggingFace-style save directory.
    Supports safetensors and pytorch bins.
    """
    # Try safetensors first
    try:
        from safetensors.torch import load_file as st_load

        st_available = True
    except ImportError:
        st_available = False
        st_load = None

    single_st = os.path.join(input_dir, "model.safetensors")
    single_pt = os.path.join(input_dir, "pytorch_model.bin")

    # --- single shard safetensors ---
    if os.path.isfile(single_st) and st_available:
        print(f"Loading single safetensors shard: {single_st}")
        return st_load(single_st, device="cpu")

    # --- sharded safetensors ---
    index_st = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.isfile(index_st) and st_available:
        print("Loading sharded safetensors …")
        with open(index_st, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        sd: Dict[str, torch.Tensor] = {}
        for sf in shard_files:
            path = os.path.join(input_dir, sf)
            print(f"  {sf}")
            sd.update(st_load(path, device="cpu"))
        return sd

    # --- single shard pytorch ---
    if os.path.isfile(single_pt):
        print(f"Loading single pytorch shard: {single_pt}")
        return torch.load(single_pt, map_location="cpu", weights_only=True)

    # --- sharded pytorch ---
    index_pt = os.path.join(input_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(index_pt):
        print("Loading sharded pytorch bin …")
        with open(index_pt, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        sd = {}
        for sf in shard_files:
            path = os.path.join(input_dir, sf)
            print(f"  {sf}")
            sd.update(torch.load(path, map_location="cpu", weights_only=True))
        return sd

    raise FileNotFoundError(
        f"Could not find any model weights in {input_dir}. "
        "Expected model.safetensors, pytorch_model.bin, or sharded equivalents."
    )


def save_state_dict(sd: Dict[str, torch.Tensor], output_dir: str):
    """
    Save a state dict. Prefers safetensors if available.
    """
    try:
        from safetensors.torch import save_file as st_save

        out_path = os.path.join(output_dir, "model.safetensors")
        print(f"Saving state dict → {out_path}")
        st_save(sd, out_path)
    except ImportError:
        out_path = os.path.join(output_dir, "pytorch_model.bin")
        print(f"safetensors not available; saving → {out_path}")
        torch.save(sd, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Strip vision encoder & projector from a hybrid Gemma3 MM checkpoint or HF model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the hybrid checkpoint directory OR a Hugging Face model ID (e.g., 'google/gemma-3-4b-pt').",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the text-only checkpoint.",
    )
    parser.add_argument(
        "--keep-tokenizer",
        action="store_true",
        help="Copy tokenizer files from the input to the output directory.",
    )
    parser.add_argument(
        "--keep-image-processor",
        action="store_true",
        help="Copy image processor files to output. Usually NOT wanted for text-only models.",
    )

    args = parser.parse_args()

    # Resolve input (Local vs HF ID)
    resolved_input_dir = resolve_input_path(args.input)

    if os.path.abspath(resolved_input_dir) == os.path.abspath(args.output):
        print("Error: --input and --output must be different directories.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    torch.set_grad_enabled(False)

    # ---- Load state dict ----
    print(f"Processing checkpoint from: {resolved_input_dir}")
    sd = load_state_dict_from_dir(resolved_input_dir)
    total_params_before = sum(v.numel() for v in sd.values())

    # ---- Strip & remap ----
    new_sd, removed, kept = strip_and_remap(sd)
    total_params_after = sum(v.numel() for v in new_sd.values())

    print(
        f"\nRemoved {len(removed)} keys ({total_params_before - total_params_after:,} parameters):"
    )
    for k in removed:
        print(f"  - {k}")

    print(f"\nKept {len(kept)} keys ({total_params_after:,} parameters).")

    # Show remapped keys
    remapped = [(old, new) for old, new in kept if old != new]
    if remapped:
        print(f"Remapped {len(remapped)} keys:")
        for old, new in remapped[:20]:
            print(f"  {old}  →  {new}")
        if len(remapped) > 20:
            print(f"  … and {len(remapped) - 20} more")

    # ---- Save state dict ----
    save_state_dict(new_sd, args.output)

    # ---- Build & save text-only config ----
    text_cfg = build_text_only_config(resolved_input_dir)
    config_out = os.path.join(args.output, "config.json")
    with open(config_out, "w") as f:
        json.dump(text_cfg, f, indent=2)
    print(f"Saved text-only config → {config_out}")

    # ---- Optionally copy tokenizer files ----
    TOKENIZER_FILES = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    if args.keep_tokenizer:
        for fname in TOKENIZER_FILES:
            src = os.path.join(resolved_input_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(args.output, fname)
                shutil.copy2(src, dst)
                print(f"Copied {fname}")

    # ---- Optionally copy image processor files ----
    IMAGE_PROCESSOR_FILES = [
        "preprocessor_config.json",
    ]
    if args.keep_image_processor:
        for fname in IMAGE_PROCESSOR_FILES:
            src = os.path.join(resolved_input_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(args.output, fname)
                shutil.copy2(src, dst)
                print(f"Copied {fname}")

    # ---- Clean up any stale index files in output ----
    # Since we are saving a single file (non-sharded) in save_state_dict,
    # we should ensure no index files are left behind if they were copied from input
    for stale in [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]:
        stale_path = os.path.join(args.output, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)
            print(f"Removed stale index file: {stale}")

    print(f"\nDone. Text-only checkpoint saved to: {args.output}")
    print(
        f"  Parameters: {total_params_before:,} → {total_params_after:,} "
        f"(removed {total_params_before - total_params_after:,})"
    )


if __name__ == "__main__":
    main()
