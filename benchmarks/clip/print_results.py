"""
CLIP Caption Benchmark - Results Printer
=========================================
Pretty-prints a comparison table from one or more CLIP score JSON files.

Usage examples:

  # Single run
  python benchmarks/clip/print_results.py \
      --paths results/hydra_gemma3_4b/clip_scores.json

  # Compare multiple runs side-by-side
  python benchmarks/clip/print_results.py \
      --paths \
          results/hydra_4heads/clip_scores.json \
          results/hydra_8heads/clip_scores.json \
          results/baseline/clip_scores.json \
      --labels "Hydra-4h" "Hydra-8h" "Baseline"

  # Save a machine-readable comparison
  python benchmarks/clip/print_results.py \
      --paths results/*/clip_scores.json \
      --save_csv results/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional pretty-print with tabulate
# ---------------------------------------------------------------------------
try:
    from tabulate import tabulate as _tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ---------------------------------------------------------------------------
# Column definitions
# (header, dotted-key-path into the result dict, format string)
# dotted-key-path is relative to the root of the JSON:
#   e.g. "aggregate.clip_score.mean" → data["aggregate"]["clip_score"]["mean"]
# None dotted-key-path means the label column (filled in by build_rows).
# ---------------------------------------------------------------------------
COLUMNS: List[tuple] = [
    ("Model / Run", None, "{}"),
    ("N", "aggregate.clip_score.count", "{:d}"),
    ("CLIPScore ↑", "aggregate.clip_score.mean", "{:.4f}"),
    ("± stderr", "aggregate.clip_score.stderr", "{:.4f}"),
    ("std", "aggregate.clip_score.std", "{:.4f}"),
    ("variance", "aggregate.clip_score.variance", "{:.6f}"),
    ("min", "aggregate.clip_score.min", "{:.4f}"),
    ("max", "aggregate.clip_score.max", "{:.4f}"),
    ("median", "aggregate.clip_score.median", "{:.4f}"),
    ("p25", "aggregate.clip_score.p25", "{:.4f}"),
    ("p75", "aggregate.clip_score.p75", "{:.4f}"),
    ("CosSim mean", "aggregate.cosine_similarity.mean", "{:.4f}"),
    ("CosSim stderr", "aggregate.cosine_similarity.stderr", "{:.4f}"),
    ("CLIP model", "meta.clip_model", "{}"),
    ("Empty captions", "meta.empty_caption_samples", "{:d}"),
    ("Failed", "meta.failed_samples", "{:d}"),
    ("Gen model", "meta.model_path", "{}"),
    ("Impl", "meta.implementation", "{}"),
    ("Heads", "meta.current_num_heads", "{}"),
    ("FFN Ratio", "meta.ffn_granularity_ratio", "{}"),
    ("Caption prompt", "meta.caption_prompt", "{}"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_nested(d: Dict[str, Any], dotpath: str) -> Any:
    """Retrieve a value from a nested dict using a dotted key path."""
    cur: Any = d
    for part in dotpath.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part, None)
    return cur


def _fmt(value: Any, fmt_str: str) -> str:
    if value is None:
        return "—"
    try:
        if fmt_str == "{:d}":
            return fmt_str.format(int(value))
        return fmt_str.format(value)
    except (ValueError, TypeError):
        return str(value)


def load_result(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_rows(
    paths: List[str],
    labels: Optional[List[str]],
    active_columns: List[tuple],
) -> List[List[str]]:
    rows = []
    for i, path in enumerate(paths):
        try:
            data = load_result(path)
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc}", file=sys.stderr)
            continue

        label = (
            labels[i]
            if labels and i < len(labels)
            else f"{Path(path).parent.name}/{Path(path).stem}"
        )

        row: List[str] = []
        for header, dotpath, fmt_str in active_columns:
            if dotpath is None:
                row.append(label)
            else:
                value = _get_nested(data, dotpath)
                row.append(_fmt(value, fmt_str))
        rows.append(row)

    return rows


def print_table(rows: List[List[str]], headers: List[str]) -> None:
    if not rows:
        print("No results to display.")
        return

    if HAS_TABULATE:
        print(_tabulate(rows, headers=headers, tablefmt="orgtbl", stralign="right"))
        return

    # Fallback: manual column-aligned printing
    col_widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.rjust(w) for c, w in zip(cells, col_widths))

    sep = "  ".join("-" * w for w in col_widths)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def save_csv(rows: List[List[str]], headers: List[str], csv_path: str) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nComparison table saved to: {out}")


# ---------------------------------------------------------------------------
# Optional per-file distribution summary
# ---------------------------------------------------------------------------
def print_distribution_summary(path: str, label: str) -> None:
    """Print a histogram-style text summary of per-sample CLIP scores."""
    try:
        data = load_result(path)
    except Exception as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
        return

    per_sample = data.get("per_sample", [])
    clip_scores = [
        s["clip_score"]
        for s in per_sample
        if s.get("scored") and s.get("clip_score") is not None
    ]

    if not clip_scores:
        print(f"\n{label}: no valid scored samples found.")
        return

    n_bins = 10
    lo, hi = min(clip_scores), max(clip_scores)
    bin_width = (hi - lo) / n_bins if hi > lo else 1.0
    bins = [0] * n_bins

    for v in clip_scores:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        bins[idx] += 1

    bar_width = 30
    max_count = max(bins) or 1

    print(f"\nDistribution of CLIPScores – {label}")
    print(f"  Total scored : {len(clip_scores)}")
    print(f"  Range        : [{lo:.4f}, {hi:.4f}]")
    print(f"  {'Bin':>24}  {'Count':>6}  Bar")
    for i, count in enumerate(bins):
        left = lo + i * bin_width
        right = left + bin_width
        bar = "█" * int(bar_width * count / max_count)
        print(f"  [{left:8.4f}, {right:8.4f})  {count:6d}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    available_headers = [h for h, _, _ in COLUMNS]
    parser = argparse.ArgumentParser(
        description="Print a comparison table of CLIP caption benchmark results."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more clip_scores.json files to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional short labels for each file (same order as --paths).",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="If given, also save the comparison table as a CSV file.",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Print a per-file histogram of CLIPScore values.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Subset of column headers to display. " f"Available: {available_headers}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Filter columns if requested
    active_columns = COLUMNS
    if args.columns:
        col_set = set(args.columns)
        active_columns = [c for c in COLUMNS if c[0] in col_set]
        if not active_columns:
            print(
                "[ERROR] None of the requested column names matched. "
                f"Available: {[h for h, _, _ in COLUMNS]}",
                file=sys.stderr,
            )
            sys.exit(1)

    headers = [h for h, _, _ in active_columns]
    rows = build_rows(args.paths, args.labels, active_columns)

    print_table(rows, headers)

    if args.save_csv:
        save_csv(rows, headers, args.save_csv)

    if args.distribution:
        for i, path in enumerate(args.paths):
            label = (
                args.labels[i]
                if args.labels and i < len(args.labels)
                else Path(path).parent.name
            )
            print_distribution_summary(path, label)


if __name__ == "__main__":
    main()
