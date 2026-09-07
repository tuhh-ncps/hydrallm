"""
CLIP Caption Benchmark – Caption Generation Stage
=================================================
Given a directory of images, runs a HydraGemma3 (or any supported) model to produce
one caption per image, then saves the captions to a JSON file for downstream scoring.

Usage example:
    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/benchmarks/clip/4b-pt/captions.json \
        --model_path "google/gemma-3-4b-pt" \
        --implementation auto_model \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2 \
        --caption_prompt 

    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/benchmarks/clip/4b-pt/captions.json \
        --model_path "google/gemma-3-4b-pt" \
        --implementation hydra_gemma_from_flex \
        --current_num_heads 1 \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2 \
        --caption_prompt 

    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/benchmarks/clip/qwen-3vl-4b/captions.json \
        --model_path "Qwen/Qwen3-VL-4B-Instruct" \
        --implementation auto_model \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2

    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/benchmarks/clip/full_from_text_finetune_pre_aligned_FULL_20260303_210106/captions_h1.json \
        --model_path outputs/amrum/trained_models/full_from_text_finetune_pre_aligned_FULL_20260303_210106/model \
        --implementation hydra_gemma_from_flex \
        --current_num_heads 1 \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2

        
    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/amrum/trained_models/joint_from_text_finetune_pre_aligned_FULL_20260226_092253/captions_h1.json \
        --model_path outputs/amrum/trained_models/joint_from_text_finetune_pre_aligned_FULL_20260226_092253/model \
        --implementation hydra_gemma_from_flex \
        --current_num_heads 1 \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2

    python benchmarks/clip/run_hydra_gemma3.py \
        --image_dir data/clip_images \
        --output_path outputs/amrum/trained_models/default_unified_dataset_20260208_211718/captions_h1.json \
        --model_path outputs/amrum/trained_models/default_unified_dataset_20260208_211718/model \
        --implementation hydra_gemma_from_flex \
        --current_num_heads 1 \
        --dtype bfloat16 \
        --attn_implementation flash_attention_2
"""

from __future__ import annotations

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

# ---------------------------------------------------------------------------
# Path setup for running this file directly from the benchmark directory.
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../benchmarks/clip
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # repo root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from model_implementations import load_model
from common_helpers import normalize_mlp_ratio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clip_bench.generate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_image_paths(image_dir: str) -> List[Path]:
    """
    Recursively collect all image files under *image_dir*.
    Returns a sorted list of Path objects.
    """
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(
            f"No supported image files found under '{image_dir}'. "
            f"Supported extensions: {SUPPORTED_IMAGE_EXTENSIONS}"
        )
    return paths


def load_image_safe(path: Path) -> Optional[Image.Image]:
    """Load a PIL Image, returning None on failure (logged as a warning)."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Could not load image %s: %s", path, exc)
        return None


def build_prompt(caption_prompt: str) -> str:
    """
    Wraps the caption instruction in the <start_of_image> tag expected by
    Gemma3 vision models.  The image placeholder must come *before* the text
    instruction so the processor can replace it with the actual patch tokens.
    """
    return f"<start_of_image>{caption_prompt.strip()}\n"


def load_yaml_config(path: str) -> dict:
    """Load a YAML file, returning an empty dict if the file is missing."""
    import yaml

    if not os.path.exists(path):
        logger.warning("Config file not found at %s; using CLI / defaults only.", path)
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_caption(
    model,
    tokenizer,
    processor,
    image: Image.Image,
    prompt: str,
    current_num_heads: Optional[int],
    mlp_ratio,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    force_fp16_pixel_values: bool,
    device: torch.device,
) -> str:
    """
    Run one forward+generate pass and return the newly generated text.
    Returns an empty string on failure (logged as a warning at the call site).
    """
    model_dtype = next(model.parameters()).dtype

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    for k, v in list(inputs.items()):
        if not hasattr(v, "to"):
            continue
        if k == "pixel_values" and force_fp16_pixel_values:
            inputs[k] = v.to(device=device, dtype=model_dtype)
        else:
            inputs[k] = v.to(device=device)

    input_len = int(inputs["input_ids"].shape[-1])

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        min_new_tokens=1,
    )
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
    if current_num_heads is not None:
        gen_kwargs["current_num_heads"] = int(current_num_heads)
    if mlp_ratio is not None:
        gen_kwargs["mlp_ratio"] = mlp_ratio

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions for images using a HydraGemma3 model."
    )

    # ── I/O ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory containing images (searched recursively).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for the output JSON file (captions keyed by image path).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(THIS_DIR, "configs", "hydra_gemma3.yaml"),
        help="YAML config with generation defaults.",
    )

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--implementation",
        type=str,
        default="hydra_gemma_from_flex",
        help=(
            "Model implementation key "
            "(e.g. 'hydra_gemma_from_flex', 'flex_gemma', 'auto_model')."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--current_num_heads",
        type=int,
        default=None,
        help="Hydra head count override (None = full model).",
    )
    parser.add_argument(
        "--ffn_granularity_ratio",
        type=float,
        nargs="+",
        default=None,
        help="One global FFN ratio or one ratio per decoder layer.",
    )

    # ── Generation ───────────────────────────────────────────────────────
    parser.add_argument(
        "--caption_prompt",
        type=str,
        default=None,
        help="Override the caption instruction from the YAML config.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--force_fp16_pixel_values", action="store_true")

    # ── Misc ─────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of images evaluated (useful for quick smoke tests).",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Seed & device ────────────────────────────────────────────────────
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Config (YAML → CLI overrides) ────────────────────────────────────
    cfg = load_yaml_config(args.config_path)

    caption_prompt: str = args.caption_prompt or cfg.get(
        "caption_prompt", "Briefly describe the image!"
    )
    max_new_tokens: int = args.max_new_tokens or cfg.get("max_new_tokens", 128)
    temperature: float = (
        args.temperature
        if args.temperature is not None
        else cfg.get("temperature", 0.0)
    )
    top_p: float = args.top_p if args.top_p is not None else cfg.get("top_p", 0.95)
    top_k: Optional[int] = (
        args.top_k if args.top_k is not None else cfg.get("top_k", 64)
    )

    logger.info("Caption prompt : %r", caption_prompt)
    logger.info("max_new_tokens : %d", max_new_tokens)
    logger.info("temperature    : %g", temperature)

    # ── Discover images ──────────────────────────────────────────────────
    image_paths = collect_image_paths(args.image_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    logger.info("Found %d images under '%s'.", len(image_paths), args.image_dir)

    # ── Tokenizer / Processor ────────────────────────────────────────────
    tok_name = args.tokenizer_path or args.model_path
    logger.info("Loading tokenizer from %s …", tok_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading processor from %s …", args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ── Model ────────────────────────────────────────────────────────────
    logger.info(
        "Loading model from %s (implementation=%s, dtype=%s) …",
        args.model_path,
        args.implementation,
        args.dtype,
    )
    model = load_model(
        args.model_path,
        implementation=args.implementation,
        dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device=str(device),
        multimodal=True,
        trust_remote_code=True,
    )
    model.eval()
    model_config = getattr(model.config, "text_config", model.config)
    mlp_ratio = normalize_mlp_ratio(
        args.ffn_granularity_ratio,
        getattr(model_config, "num_hidden_layers", None),
    )
    logger.info("Model loaded.")

    # ── Build the prompt string (reused for every image) ─────────────────
    prompt = build_prompt(caption_prompt)
    if args.debug:
        logger.debug("Prompt template: %r", prompt)

    # ── Caption generation loop ──────────────────────────────────────────
    results: Dict[str, Any] = {}
    empty_count = 0

    for img_path in tqdm(image_paths, desc="Generating captions"):
        key = str(img_path)
        image = load_image_safe(img_path)
        if image is None:
            results[key] = {"caption": "", "image_path": key, "error": "load_failed"}
            empty_count += 1
            continue

        try:
            caption = generate_caption(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                image=image,
                prompt=prompt,
                current_num_heads=args.current_num_heads,
                mlp_ratio=mlp_ratio,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                force_fp16_pixel_values=args.force_fp16_pixel_values,
                device=device,
            )
        except Exception as exc:
            logger.warning("Generation failed for %s: %s", img_path, exc)
            caption = ""

        if caption == "":
            empty_count += 1

        results[key] = {
            "caption": caption,
            "image_path": key,
        }

        if args.debug:
            logger.debug("%-60s → %r", img_path.name, caption[:120])

    # ── Save captions ────────────────────────────────────────────────────
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "model_path": args.model_path,
            "implementation": args.implementation,
            "current_num_heads": args.current_num_heads,
            "ffn_granularity_ratio": mlp_ratio,
            "caption_prompt": caption_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "seed": args.seed,
            "num_images": len(image_paths),
            "num_empty": empty_count,
        },
        "captions": results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d captions to %s  (empty: %d)",
        len(results),
        out_path,
        empty_count,
    )


if __name__ == "__main__":
    main()
