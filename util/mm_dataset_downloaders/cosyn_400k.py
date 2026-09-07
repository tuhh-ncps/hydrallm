#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets import load_dataset


REPO_ID = "allenai/CoSyn-400K"

SUBSETS = [
    "chart",
    "chemical",
    "circuit",
    "diagram",
    "document",
    "graphic",
    "math",
    "music",
    "nutrition",
    "table",
]


def download_subset(subset: str, output_dir: Path):
    subset_dir = output_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading subset: {subset}")
    print(f"Saving to: {subset_dir}")

    dataset = load_dataset(
        REPO_ID,
        name=subset,
        split=None,  # download all splits (train/validation if present)
        cache_dir=subset_dir,
    )

    # Persist to disk so it can be memory-mapped later
    dataset.save_to_disk(subset_dir)

    print(f"Finished subset: {subset}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all subsets of the CoSyn-400K dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/cosyn_400k"),
        help="Root directory for CoSyn-400K",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting CoSyn-400K download")
    print(f"Root directory: {args.output_dir}")
    print(f"Subsets: {', '.join(SUBSETS)}")

    for subset in SUBSETS:
        download_subset(subset, args.output_dir)

    print("\n✅ CoSyn-400K download complete")
    print(f"Dataset location: {args.output_dir}")


if __name__ == "__main__":
    main()
