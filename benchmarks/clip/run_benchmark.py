"""
CLIP Caption Benchmark – Pipeline Runner
=========================================
Runs the full two-stage pipeline (caption generation → CLIP scoring) from a
single Python entry point, with the same arguments as the individual scripts
plus a few pipeline-level conveniences.

This replaces the shell-script wrapper and works on every platform.

Usage example:
    python benchmarks/clip/run_benchmark.py \
        --model_path  google/gemma-3-1b-pt \
        --image_dir   data/clip_images \
        --output_dir  results/hydra_4heads \
        --num_heads   4 \
        --dtype       bfloat16 \
        --clip_model  openai/clip-vit-large-patch14

After it finishes you can compare runs with:
    python benchmarks/clip/print_results.py \
        --paths results/hydra_4heads/clip_scores.json \
                results/hydra_8heads/clip_scores.json \
        --labels "Hydra-4h" "Hydra-8h"
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clip_bench.pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_quotes(value: str) -> str:
    """
    Remove surrounding single or double quotes that the shell (or the user)
    may have left on a string argument.

    Example:  '"google/gemma-3-1b-pt"'  →  'google/gemma-3-1b-pt'
    """
    if len(value) >= 2:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
    return value


def _clean_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Strip accidental surrounding quotes from every string attribute in the
    parsed namespace.  This is a defensive measure: when users wrap values in
    extra quotes on the command line (e.g. --model_path "google/…") some
    shells pass the quotes through verbatim to the Python process.
    """
    for field, value in vars(args).items():
        if isinstance(value, str):
            setattr(args, field, _strip_quotes(value))
    return args


def _run(cmd: list[str], step: str) -> None:
    """Run *cmd* as a subprocess, streaming stdout/stderr. Raises on failure."""
    logger.info("=" * 60)
    logger.info("STEP: %s", step)
    logger.info("CMD : %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("Step '%s' failed with return code %d.", step, result.returncode)
        sys.exit(result.returncode)

    logger.info("Step '%s' finished successfully.", step)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full CLIP caption benchmark pipeline: "
            "caption generation → CLIP scoring → optional results printing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HF hub ID of the generative model.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory of images (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory for all outputs. "
            "captions.json and clip_scores.json will be written here."
        ),
    )

    # ── Generative model settings ─────────────────────────────────────────
    parser.add_argument(
        "--implementation",
        type=str,
        default="hydra_gemma_from_flex",
        help="Model implementation key passed to load_model().",
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
        help="Dtype for the generative model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the generative model.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Hydra current_num_heads override (omit for full model).",
    )
    parser.add_argument(
        "--ffn_granularity_ratio",
        type=float,
        nargs="+",
        default=None,
        help="One global FFN ratio or one ratio per decoder layer.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Override tokenizer path (defaults to --model_path).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(THIS_DIR, "configs", "hydra_gemma3.yaml"),
        help="YAML config for generation defaults.",
    )

    # ── Caption generation knobs ──────────────────────────────────────────
    parser.add_argument(
        "--caption_prompt",
        type=str,
        default=None,
        help="Override caption instruction (also overrides YAML).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--force_fp16_pixel_values",
        action="store_true",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap images for quick smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    # ── CLIP scoring settings ─────────────────────────────────────────────
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
        help="Batch size for CLIP encoding.",
    )
    parser.add_argument(
        "--clip_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for CLIP model weights.",
    )
    parser.add_argument(
        "--clip_device",
        type=str,
        default="cuda",
        help="Device for CLIP scoring (can differ from generative model device).",
    )
    parser.add_argument(
        "--skip_empty_captions",
        action="store_true",
        default=True,
        help="Exclude empty captions from aggregate CLIP statistics.",
    )
    parser.add_argument(
        "--max_score_samples",
        type=int,
        default=None,
        help="Score only the first N captions (for quick tests).",
    )

    # ── Pipeline control ──────────────────────────────────────────────────
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help=(
            "Skip the caption generation step. "
            "Useful if captions.json already exists and you only want to re-score."
        ),
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip the CLIP scoring step (generate captions only).",
    )
    parser.add_argument(
        "--print_results",
        action="store_true",
        help="Print the results table to stdout after scoring.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="If given, save a CSV summary to this path.",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Print per-run histogram of CLIPScores after scoring.",
    )

    args = parser.parse_args()
    # ── Defensive quote-stripping ─────────────────────────────────────────
    # Some shells or users wrap values in extra quotes; strip them so that
    # HuggingFace repo IDs and paths are not corrupted.
    args = _clean_args(args)
    return args


# ---------------------------------------------------------------------------
# Build sub-command argument lists
# ---------------------------------------------------------------------------


def build_generation_cmd(args: argparse.Namespace, captions_path: str) -> list[str]:
    cmd = [
        sys.executable,
        os.path.join(THIS_DIR, "run_hydra_gemma3.py"),
        "--model_path",
        args.model_path,
        "--image_dir",
        args.image_dir,
        "--output_path",
        captions_path,
        "--implementation",
        args.implementation,
        "--attn_implementation",
        args.attn_implementation,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--config_path",
        args.config_path,
        "--seed",
        str(args.seed),
    ]

    if args.num_heads is not None:
        cmd += ["--current_num_heads", str(args.num_heads)]
    if args.ffn_granularity_ratio is not None:
        cmd += ["--ffn_granularity_ratio"] + [
            str(ratio) for ratio in args.ffn_granularity_ratio
        ]
    if args.tokenizer_path is not None:
        cmd += ["--tokenizer_path", args.tokenizer_path]
    if args.caption_prompt is not None:
        cmd += ["--caption_prompt", args.caption_prompt]
    if args.max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(args.max_new_tokens)]
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.top_p is not None:
        cmd += ["--top_p", str(args.top_p)]
    if args.top_k is not None:
        cmd += ["--top_k", str(args.top_k)]
    if args.force_fp16_pixel_values:
        cmd.append("--force_fp16_pixel_values")
    if args.max_images is not None:
        cmd += ["--max_images", str(args.max_images)]
    if args.debug:
        cmd.append("--debug")

    return cmd


def build_scoring_cmd(
    args: argparse.Namespace,
    captions_path: str,
    scores_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        os.path.join(THIS_DIR, "eval_clip_scores.py"),
        "--captions_path",
        captions_path,
        "--output_path",
        scores_path,
        "--clip_model",
        args.clip_model,
        "--clip_batch_size",
        str(args.clip_batch_size),
        "--dtype",
        args.clip_dtype,
        "--device",
        args.clip_device,
    ]

    if args.skip_empty_captions:
        cmd.append("--skip_empty_captions")
    if args.max_score_samples is not None:
        cmd += ["--max_samples", str(args.max_score_samples)]

    return cmd


def build_print_cmd(
    args: argparse.Namespace,
    scores_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        os.path.join(THIS_DIR, "print_results.py"),
        "--paths",
        scores_path,
    ]

    if args.save_csv:
        cmd += ["--save_csv", args.save_csv]
    if args.distribution:
        cmd.append("--distribution")

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── Resolve output paths ──────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions_path = str(out_dir / "captions.json")
    scores_path = str(out_dir / "clip_scores.json")

    # ── Print run summary ─────────────────────────────────────────────────
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║        CLIP Caption Benchmark Pipeline       ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("  Generative model : %s", args.model_path)
    logger.info("  Implementation   : %s", args.implementation)
    logger.info("  Heads override   : %s", args.num_heads or "full")
    logger.info("  FFN ratio        : %s", args.ffn_granularity_ratio or "full")
    logger.info("  Image dir        : %s", args.image_dir)
    logger.info("  Output dir       : %s", out_dir)
    logger.info("  CLIP model       : %s", args.clip_model)
    logger.info("  Captions file    : %s", captions_path)
    logger.info("  Scores file      : %s", scores_path)

    # ── Stage 1: Caption generation ───────────────────────────────────────
    if not args.skip_generation:
        gen_cmd = build_generation_cmd(args, captions_path)
        _run(gen_cmd, "Caption Generation")
    else:
        logger.info("Skipping caption generation (--skip_generation set).")
        if not Path(captions_path).exists():
            logger.error(
                "Captions file '%s' does not exist and generation was skipped. "
                "Cannot proceed with scoring.",
                captions_path,
            )
            sys.exit(1)

    # ── Stage 2: CLIP scoring ─────────────────────────────────────────────
    if not args.skip_scoring:
        score_cmd = build_scoring_cmd(args, captions_path, scores_path)
        _run(score_cmd, "CLIP Scoring")
    else:
        logger.info("Skipping CLIP scoring (--skip_scoring set).")

    # ── Stage 3 (optional): Print results ────────────────────────────────
    if args.print_results and not args.skip_scoring:
        if Path(scores_path).exists():
            print_cmd = build_print_cmd(args, scores_path)
            _run(print_cmd, "Print Results")
        else:
            logger.warning(
                "Scores file '%s' not found; skipping results printing.",
                scores_path,
            )

    logger.info("Pipeline complete.")
    logger.info("  Captions : %s", captions_path)
    if not args.skip_scoring:
        logger.info("  Scores   : %s", scores_path)
    logger.info(
        "\nTo compare results later:\n" "  python %s --paths %s",
        os.path.join(THIS_DIR, "print_results.py"),
        scores_path,
    )


if __name__ == "__main__":
    main()
