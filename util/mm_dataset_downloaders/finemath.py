#!/usr/bin/env python3
"""
Download HuggingFaceTB/finemath (finemath-4plus subset).
"""
import argparse
from pathlib import Path

from datasets import load_dataset

REPO_ID = "HuggingFaceTB/finemath"
CONFIG_NAME = "finemath-4plus"


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFaceTB/finemath (finemath-4plus)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/finemath/finemath-4plus"),
        help="Target directory for the finemath-4plus dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_NAME,
        help="finemath config/subset name (default: finemath-4plus)",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading HuggingFaceTB/finemath")
    print(f"  Config : {args.config}")
    print(f"  Output : {out}")

    dataset = load_dataset(
        REPO_ID,
        name=args.config,
        split="train",
        cache_dir=str(out / "_cache"),
    )

    dataset.save_to_disk(str(out))

    print(f"\n✅ finemath ({args.config}) download complete")
    print(f"Saved to : {out}")
    print(f"Rows     : {len(dataset):,}")
    print(f"Columns  : {dataset.column_names}")


if __name__ == "__main__":
    main()
