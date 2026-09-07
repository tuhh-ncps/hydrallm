#!/usr/bin/env python3

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "fighter3005/coco-recaptioned"

FILES = [
    "coco_captions.json",
    "images.zip",
]


def download_files(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading COCO-Recaptioned files into: {output_dir}")

    local_paths = {}
    for filename in FILES:
        print(f"  - Downloading {filename}")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        local_paths[filename] = Path(local_path)

    return local_paths


def extract_images(images_zip: Path, output_dir: Path):
    images_dir = output_dir / "images"

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Images already extracted at: {images_dir}")
        return

    print(f"Extracting images to: {images_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(images_dir)

    print("Image extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the COCO-Recaptioned dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/coco_recaptioned"),
        help="Target directory for the dataset",
    )

    args = parser.parse_args()

    paths = download_files(args.output_dir)
    extract_images(paths["images.zip"], args.output_dir)

    print("\nCOCO-Recaptioned dataset is ready.")
    print(f"Dataset location: {args.output_dir}")
    print("Structure:")
    print("  ├── coco_captions.json")
    print("  └── images/")


if __name__ == "__main__":
    main()
