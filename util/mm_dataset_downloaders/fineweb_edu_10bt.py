#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets import load_dataset


REPO_ID = "HuggingFaceFW/fineweb-edu"
CONFIG_NAME = "sample-10BT"


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu (10BT sample)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/fineweb_edu/sample_10bt"),
        help="Target directory for FineWeb-Edu 10BT sample",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FineWeb-Edu (10BT sample)")
    print(f"Config: {CONFIG_NAME}")
    print(f"Output directory: {args.output_dir}")

    dataset = load_dataset(
        REPO_ID,
        name=CONFIG_NAME,
        split="train",
        cache_dir=args.output_dir,
    )

    dataset.save_to_disk(args.output_dir)

    print("\n✅ FineWeb-Edu 10BT sample download complete")
    print(f"Saved to: {args.output_dir}")
    print("Dataset columns:", dataset.column_names)


if __name__ == "__main__":
    main()
