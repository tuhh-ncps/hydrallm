#!/usr/bin/env python3

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm


REPO_ID = "allenai/pixmo-ask-model-anything"


def url_to_filename(url: str) -> str:
    """
    Generate a stable filename from a URL.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(url.split("?")[0])[1]
    if not ext or len(ext) > 5:
        ext = ".jpg"
    return f"{h}{ext}"


def download_image(url: str, images_dir: Path, timeout=20):
    filename = url_to_filename(url)
    out_path = images_dir / filename

    if out_path.exists():
        return True

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


def download_images_concurrently(urls, images_dir: Path, num_workers: int):
    images_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_image, url, images_dir): url for url in urls
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading images",
        ):
            if future.result():
                success += 1
            else:
                failed += 1

    print("\nImage download complete")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed:  {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PixMo-Ask-Model-Anything and fetch images from URLs"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/pixmo_ask_model_anything"),
        help="Root directory for the dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of concurrent image downloads",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = args.output_dir / "metadata"
    images_dir = args.output_dir / "images"

    print("Loading PixMo-Ask-Model-Anything metadata...")
    dataset = load_dataset(
        REPO_ID,
        split="train",
        cache_dir=metadata_dir,
    )

    dataset.save_to_disk(metadata_dir)
    print(f"Metadata saved to: {metadata_dir}")

    # Deduplicate URLs (many rows can share the same image)
    urls = sorted({row["image_url"] for row in dataset if row.get("image_url")})
    print(f"Found {len(urls):,} unique image URLs")

    download_images_concurrently(
        urls=urls,
        images_dir=images_dir,
        num_workers=args.num_workers,
    )

    print("\n✅ PixMo-Ask-Model-Anything dataset preparation complete")
    print(f"Dataset location: {args.output_dir}")


if __name__ == "__main__":
    main()
