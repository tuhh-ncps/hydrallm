"""
CLIP Caption Benchmark - Scoring Stage
=======================================
Loads the captions JSON produced by *run_hydra_gemma3.py*, encodes each
(image, caption) pair with a large CLIP model, computes the cosine-similarity
CLIP score, and writes results into a JSON.

The CLIP score reported here follows the definition used in:
  Hessel et al. (2021) "CLIPScore: A Reference-free Evaluation Metric for
  Image Captioning" - EMNLP 2021.
  Formula:  CLIPScore = w * max(0, cos_sim(image_emb, text_emb))  with w=2.5

Usage example:
    python benchmarks/clip/eval_clip_scores.py \
        --captions_path outputs/benchmarks/clip/4b-pt/captions.json \
        --output_path   outputs/benchmarks/clip/4b-pt/clip_scores.json \
        --clip_model    openai/clip-vit-large-patch14 \
        --clip_batch_size 1

    python benchmarks/clip/eval_clip_scores.py \
        --captions_path outputs/benchmarks/clip/full_from_text_finetune_pre_aligned_FULL_20260303_210106/captions_h4.json \
        --output_path   outputs/benchmarks/clip/full_from_text_finetune_pre_aligned_FULL_20260303_210106/clip_scores_h4.json \
        --clip_model    openai/clip-vit-large-patch14

    python benchmarks/clip/eval_clip_scores.py \
        --captions_path outputs/amrum/trained_models/default_unified_dataset_20260208_211718/captions_h1.json \
        --output_path   outputs/amrum/trained_models/default_unified_dataset_20260208_211718/clip_scores_h1.json \
        --clip_model    openai/clip-vit-large-patch14
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clip_bench.score")

# ---------------------------------------------------------------------------
# CLIPScore weight (Hessel et al. 2021)
# ---------------------------------------------------------------------------
CLIPSCORE_WEIGHT: float = 2.5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Given a flat list of scalars, return a comprehensive statistics dict:
      count, mean, std (population), variance (population),
      stderr (standard error of the mean), min, max, median, p25, p75.
    """
    if not values:
        nan = float("nan")
        return {
            "count": 0,
            "mean": nan,
            "std": nan,
            "variance": nan,
            "stderr": nan,
            "min": nan,
            "max": nan,
            "median": nan,
            "p25": nan,
            "p75": nan,
        }

    arr = np.array(values, dtype=np.float64)
    n = int(len(arr))
    mean = float(np.mean(arr))
    std_pop = float(np.std(arr, ddof=0))
    var_pop = float(np.var(arr, ddof=0))
    # Standard error of the mean: sample std / sqrt(n)
    stderr = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    return {
        "count": n,
        "mean": mean,
        "std": std_pop,
        "variance": var_pop,
        "stderr": stderr,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_safe(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.warning("Could not load image %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------
class CLIPScorer:
    """
    Thin wrapper around a HuggingFace CLIPModel that computes per-sample
    cosine similarities between an image embedding and a text embedding.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        logger.info("Loading CLIP model '%s' …", model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.model.to(device)
        logger.info("CLIP model loaded.")

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Return L2-normalised image embeddings, shape [N, D]."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats.float(), dim=-1)

    @torch.inference_mode()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Return L2-normalised text embeddings, shape [N, D]."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP text-encoder token limit
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats.float(), dim=-1)

    # ------------------------------------------------------------------
    def score_pairs(
        self,
        image_paths: List[str],
        captions: List[str],
    ) -> Tuple[List[float], List[float], List[bool]]:
        """
        Score a list of (image_path, caption) pairs in batches.

        Returns three parallel lists:
          raw_cosine_sims  - cosine similarity ∈ [-1, 1]
          clip_scores      - CLIPSCORE_WEIGHT * max(0, cos_sim)
          success_flags    - True if the pair was successfully scored
        """
        assert len(image_paths) == len(captions)
        n = len(image_paths)

        raw_cosine_sims: List[float] = []
        clip_scores: List[float] = []
        success_flags: List[bool] = []

        for start in tqdm(range(0, n, self.batch_size), desc="CLIP scoring"):
            end = min(start + self.batch_size, n)
            batch_paths = image_paths[start:end]
            batch_texts = captions[start:end]

            # Load images; keep track of which slots are valid
            batch_images: List[Optional[Image.Image]] = [
                load_image_safe(p) for p in batch_paths
            ]

            valid_imgs: List[Image.Image] = []
            valid_texts: List[str] = []
            valid_indices: List[int] = []  # indices relative to this batch

            for i, (img, txt) in enumerate(zip(batch_images, batch_texts)):
                if img is not None and txt and txt.strip():
                    valid_imgs.append(img)
                    valid_texts.append(txt.strip())
                    valid_indices.append(i)

            batch_raw = [float("nan")] * (end - start)
            batch_clip = [float("nan")] * (end - start)
            batch_ok = [False] * (end - start)

            if valid_imgs:
                try:
                    img_embs = self._encode_images(valid_imgs)  # [V, D]
                    txt_embs = self._encode_texts(valid_texts)  # [V, D]
                    # Dot product of L2-normalised vectors = cosine similarity
                    sims = (img_embs * txt_embs).sum(dim=-1).cpu().tolist()

                    for rel_i, sim in zip(valid_indices, sims):
                        cs = CLIPSCORE_WEIGHT * max(0.0, float(sim))
                        batch_raw[rel_i] = float(sim)
                        batch_clip[rel_i] = cs
                        batch_ok[rel_i] = True

                except Exception as exc:
                    logger.warning(
                        "CLIP scoring failed for batch [%d:%d]: %s",
                        start,
                        end,
                        exc,
                    )

            raw_cosine_sims.extend(batch_raw)
            clip_scores.extend(batch_clip)
            success_flags.extend(batch_ok)

        return raw_cosine_sims, clip_scores, success_flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP scores for generated captions."
    )

    # ── I/O ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--captions_path",
        type=str,
        required=True,
        help="Path to the captions JSON produced by run_hydra_gemma3.py.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Destination for the CLIP scores JSON.",
    )

    # ── CLIP model ───────────────────────────────────────────────────────
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP model identifier.",
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=64,
        help="Number of image–text pairs processed per CLIP forward pass.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for CLIP model weights (float32 recommended for accuracy).",
    )
    parser.add_argument("--device", type=str, default="cuda")

    # ── Filtering ────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip_empty_captions",
        action="store_true",
        default=True,
        help="Exclude samples with empty captions from aggregate statistics.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (for quick smoke tests).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    clip_dtype = dtype_map[args.dtype]

    # ── Load captions ────────────────────────────────────────────────────
    cap_path = Path(args.captions_path)
    if not cap_path.exists():
        raise FileNotFoundError(f"Captions file not found: {cap_path}")

    with open(cap_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both the wrapped {"meta": …, "captions": {…}} format and a
    # plain flat dict {image_path: {"caption": …}} for convenience.
    if "captions" in data and isinstance(data["captions"], dict):
        meta: Dict[str, Any] = data.get("meta", {})
        captions_dict: Dict[str, Any] = data["captions"]
    else:
        meta = {}
        captions_dict = data

    logger.info("Loaded %d caption entries.", len(captions_dict))

    # ── Build ordered lists ──────────────────────────────────────────────
    all_keys = list(captions_dict.keys())
    if args.max_samples is not None:
        all_keys = all_keys[: args.max_samples]

    image_paths: List[str] = []
    captions: List[str] = []

    for key in all_keys:
        entry = captions_dict[key]
        if isinstance(entry, dict):
            caption = entry.get("caption", "")
            img_path = entry.get("image_path", key)
        else:
            caption = str(entry)
            img_path = key
        image_paths.append(img_path)
        captions.append(caption)

    logger.info(
        "Scoring %d image-caption pairs with CLIP model '%s' …",
        len(image_paths),
        args.clip_model,
    )

    # ── CLIP scoring ─────────────────────────────────────────────────────
    scorer = CLIPScorer(
        model_name=args.clip_model,
        device=device,
        dtype=clip_dtype,
        batch_size=args.clip_batch_size,
    )

    raw_sims, clip_scores_list, success_flags = scorer.score_pairs(
        image_paths=image_paths,
        captions=captions,
    )

    # ── Collect per-sample records and accumulate valid values ────────────
    per_sample: List[Dict[str, Any]] = []
    valid_raw: List[float] = []
    valid_clip: List[float] = []
    failed_count = 0
    empty_caption_count = 0

    for key, img_path, caption, raw, cs, ok in zip(
        all_keys,
        image_paths,
        captions,
        raw_sims,
        clip_scores_list,
        success_flags,
    ):
        is_empty = not (caption and caption.strip())
        if is_empty:
            empty_caption_count += 1

        record: Dict[str, Any] = {
            "key": key,
            "image_path": img_path,
            "caption": caption,
            "cosine_similarity": raw if ok else None,
            "clip_score": cs if ok else None,
            "scored": ok,
            "empty_caption": is_empty,
        }
        per_sample.append(record)

        if ok:
            if not (args.skip_empty_captions and is_empty):
                valid_raw.append(raw)
                valid_clip.append(cs)
        else:
            failed_count += 1

    # ── Aggregate statistics ─────────────────────────────────────────────
    stats_clip = compute_statistics(valid_clip)
    stats_raw = compute_statistics(valid_raw)

    logger.info(
        "CLIPScore   - mean=%.4f  std=%.4f  stderr=%.4f  n=%d",
        stats_clip["mean"],
        stats_clip["std"],
        stats_clip["stderr"],
        stats_clip["count"],
    )
    logger.info(
        "Cosine sim  - mean=%.4f  std=%.4f  stderr=%.4f",
        stats_raw["mean"],
        stats_raw["std"],
        stats_raw["stderr"],
    )
    logger.info(
        "Failures: %d  |  Empty captions: %d",
        failed_count,
        empty_caption_count,
    )

    # ── Assemble output ──────────────────────────────────────────────────
    output = {
        "meta": {
            **meta,
            "clip_model": args.clip_model,
            "clip_batch_size": args.clip_batch_size,
            "clip_score_weight": CLIPSCORE_WEIGHT,
            "skip_empty_captions_in_stats": args.skip_empty_captions,
            "total_samples": len(per_sample),
            "scored_samples": sum(1 for s in per_sample if s["scored"]),
            "failed_samples": failed_count,
            "empty_caption_samples": empty_caption_count,
        },
        "aggregate": {
            "clip_score": stats_clip,
            "cosine_similarity": stats_raw,
        },
        "per_sample": per_sample,
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved CLIP score results to %s", out_path)


if __name__ == "__main__":
    main()
