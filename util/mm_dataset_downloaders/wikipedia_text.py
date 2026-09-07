#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets import load_dataset


REPO_ID = "wikimedia/wikipedia"


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikimedia Wikipedia (text-only) dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="20231101.en",
        help="Wikipedia config in the form <dump_date>.<language> (e.g. 20231101.en)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/wikipedia"),
        help="Root directory for the Wikipedia dataset",
    )

    args = parser.parse_args()

    dataset_dir = args.output_dir / args.config
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Wikimedia Wikipedia dataset")
    print(f"Config: {args.config}")
    print(f"Output directory: {dataset_dir}")

    dataset = load_dataset(
        REPO_ID,
        args.config,
        split="train",
        cache_dir=dataset_dir,
    )

    dataset.save_to_disk(dataset_dir)

    print("\n✅ Wikipedia dataset download complete")
    print(f"Saved to: {dataset_dir}")
    print("Example fields:")
    print(dataset.column_names)


if __name__ == "__main__":
    main()
