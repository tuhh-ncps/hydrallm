#!/usr/bin/env python3
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm

# If True, Pillow may load truncated images instead of failing.
# We keep it False so truncated files are treated as broken.
ImageFile.LOAD_TRUNCATED_IMAGES = False


DEFAULT_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".ppm",
}


def iter_image_files(root: Path, exts: Optional[set[str]] = None) -> Iterable[Path]:
    exts = exts or DEFAULT_EXTS
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def is_broken_image(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_broken, reason).
    Checks:
      - file size > 0
      - PIL verify()
      - PIL load() (decode)
    """
    try:
        st = path.stat()
        if st.st_size == 0:
            return True, "empty_file"

        # 1) verify() checks file integrity without decoding all pixels
        with Image.open(path) as im:
            im.verify()

        # 2) reopen and load() to catch decode errors verify may miss
        with Image.open(path) as im:
            im.load()

        return False, None

    except Exception as e:
        return True, f"{type(e).__name__}: {e}"


def move_preserve_structure(src: Path, root: Path, dst_root: Path) -> Path:
    rel = src.relative_to(root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser(
        description="Recursively scan a directory for broken images and optionally delete/move them."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing images (scanned recursively).",
    )
    ap.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Comma-separated list of file extensions to consider (e.g. .jpg,.png,.webp).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of threads used to check images.",
    )

    action = ap.add_mutually_exclusive_group()
    action.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print broken image paths (default behavior if no action is specified).",
    )
    action.add_argument(
        "--delete",
        action="store_true",
        help="Delete broken images.",
    )
    action.add_argument(
        "--move_to",
        type=Path,
        default=None,
        help="Move broken images into this directory, preserving subdirectory structure.",
    )

    ap.add_argument(
        "--print_reason",
        action="store_true",
        help="Also print the reason an image is considered broken.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of images to scan (0 = no limit).",
    )

    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if not exts:
        raise SystemExit("No extensions provided.")

    # Default to dryrun if user didn't pick delete/move_to/dryrun explicitly
    do_delete = bool(args.delete)
    do_move = args.move_to is not None
    do_dryrun = bool(args.dryrun) or (not do_delete and not do_move)

    if do_move:
        args.move_to.mkdir(parents=True, exist_ok=True)

    files = list(iter_image_files(root, exts=exts))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Scanning root: {root}")
    print(f"Extensions: {sorted(exts)}")
    print(f"Found {len(files):,} image files")
    if do_dryrun:
        print("Mode: DRY RUN (no files will be modified)")
    elif do_delete:
        print("Mode: DELETE (broken images will be removed)")
    elif do_move:
        print(f"Mode: MOVE (broken images will be moved to {args.move_to})")

    broken: List[Tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(is_broken_image, p): p for p in files}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Checking images",
            unit="img",
        ):
            p = futures[fut]
            try:
                is_broken, reason = fut.result()
            except Exception as e:
                is_broken, reason = True, f"FutureError: {type(e).__name__}: {e}"

            if is_broken:
                broken.append((p, reason or "unknown"))

    # Print broken paths
    if broken:
        print(f"\nBroken images: {len(broken):,}")
        for p, reason in broken:
            if args.print_reason:
                print(f"{p}\t{reason}")
            else:
                print(str(p))
    else:
        print("\nNo broken images detected.")

    # Apply action
    if do_dryrun:
        print("\nDry run complete. No files changed.")
        return

    changed = 0
    if do_delete:
        for p, _ in broken:
            try:
                p.unlink()
                changed += 1
            except Exception as e:
                print(f"[WARN] Failed to delete {p}: {e}")

        print(f"\nDeleted {changed:,} broken images.")

    elif do_move:
        assert args.move_to is not None
        for p, _ in broken:
            try:
                dst = move_preserve_structure(p, root, args.move_to)
                changed += 1
            except Exception as e:
                print(f"[WARN] Failed to move {p}: {e}")

        print(f"\nMoved {changed:,} broken images to: {args.move_to}")

    # Optional: remove empty directories after delete/move
    # (kept off by default; you can add a flag if you want)

    print("\nDone.")


if __name__ == "__main__":
    main()
