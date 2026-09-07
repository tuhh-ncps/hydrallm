#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from tqdm import tqdm


def _truncate(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _iter_unique_nonnull_strings(
    dset: ds.Dataset, column: str, batch_size: int
) -> Set[str]:
    uniques: Set[str] = set()
    scanner = dset.scanner(columns=[column], batch_size=batch_size)
    for batch in tqdm(
        scanner.to_batches(), desc=f"Collecting unique {column}", unit="batch"
    ):
        arr = batch.column(0)
        mask = pc.is_valid(arr)
        arr_valid = pc.filter(arr, mask)
        for v in arr_valid.to_pylist():
            if v:
                uniques.add(v)
    return uniques


def _count_null_nonnull(
    dset: ds.Dataset, column: str, batch_size: int
) -> Dict[str, int]:
    null_rows = 0
    nonnull_rows = 0
    scanner = dset.scanner(columns=[column], batch_size=batch_size)
    for batch in tqdm(
        scanner.to_batches(), desc=f"Counting null/non-null {column}", unit="batch"
    ):
        arr = batch.column(0)
        valid_mask = pc.is_valid(arr)
        valid_count = int(pc.sum(pc.cast(valid_mask, pa.int64())).as_py())
        nonnull_rows += valid_count
        null_rows += len(arr) - valid_count
    return {"null": null_rows, "nonnull": nonnull_rows}


def _get_first_n_for_dataset(
    dset: ds.Dataset,
    dataset_name: str,
    n: int,
    columns: List[str],
    batch_size: int,
) -> List[Dict]:
    """
    Collect first N rows for dataset == dataset_name, without relying on scanner(limit=...).
    Stops early once enough rows have been collected.
    """
    filt = ds.field("dataset") == dataset_name
    scanner = dset.scanner(columns=columns, filter=filt, batch_size=batch_size)

    out: List[Dict] = []
    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])
        out.extend(tbl.to_pylist())
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser(
        description="Check unified Parquet dataset: counts, image existence, and print first N samples per dataset."
    )
    ap.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Root directory of the unified dataset (contains parquet/ and images/).",
    )
    ap.add_argument(
        "--parquet_glob",
        type=str,
        default="parquet/**/part-*.parquet",
        help="Glob (relative to dataset_dir) for parquet files.",
    )
    ap.add_argument("--image_path_column", type=str, default="image_path")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_unique_images_check", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)
    ap.add_argument("--print_truncate", type=int, default=2048)
    args = ap.parse_args()

    dataset_dir: Path = args.dataset_dir
    parquet_paths = sorted(dataset_dir.glob(args.parquet_glob))
    if not parquet_paths:
        raise SystemExit(
            f"No parquet files found with glob: {dataset_dir / args.parquet_glob}"
        )

    print(f"Found {len(parquet_paths)} parquet files under: {dataset_dir}")

    dset = ds.dataset([str(p) for p in parquet_paths], format="parquet")

    total_rows = dset.count_rows()
    print(f"Total rows (samples): {total_rows:,}")
    print(f"Columns: {dset.schema.names}")

    image_col = args.image_path_column
    if image_col in dset.schema.names:
        counts = _count_null_nonnull(dset, image_col, args.batch_size)
        print(f"Rows with {image_col} present: {counts['nonnull']:,}")
        print(f"Rows with {image_col} NULL/absent (text-only): {counts['null']:,}")
    else:
        print(f"No '{image_col}' column; skipping image checks.")

    if "dataset" not in dset.schema.names:
        print("No 'dataset' column; cannot print per-dataset samples.")
        return

    dataset_names = sorted(
        _iter_unique_nonnull_strings(dset, "dataset", args.batch_size)
    )
    print(f"Datasets present: {len(dataset_names)}")
    for name in dataset_names:
        print(f"  - {name}")

    # Print first N samples per dataset
    n = int(args.print_samples)
    print_cols = [
        c
        for c in ["uid", "dataset", "subset", "task", image_col, "prompt", "target"]
        if c in dset.schema.names
    ]

    print(f"\nPrinting first {n} samples from each dataset (columns: {print_cols})")
    for name in dataset_names:
        rows = _get_first_n_for_dataset(
            dset,
            dataset_name=name,
            n=n,
            columns=print_cols,
            batch_size=args.batch_size,
        )
        print(f"\n=== {name} (showing {len(rows)}/{n}) ===")
        for i, r in enumerate(rows):
            r2 = dict(r)
            if "prompt" in r2:
                r2["prompt"] = _truncate(r2.get("prompt"), args.print_truncate)
            if "target" in r2:
                r2["target"] = _truncate(r2.get("target"), args.print_truncate)

            # Optional: also show whether the image exists for this row
            if image_col in r2 and r2[image_col]:
                img_path = dataset_dir / r2[image_col]
                r2["image_exists"] = img_path.exists()

            print(f"[{i}] {r2}")

    # Unique image existence check (deduped)
    if image_col in dset.schema.names:
        unique_paths = _iter_unique_nonnull_strings(dset, image_col, args.batch_size)
        print(f"\nUnique image paths referenced: {len(unique_paths):,}")

        unique_list = list(unique_paths)
        if args.max_unique_images_check and args.max_unique_images_check > 0:
            unique_list = unique_list[: args.max_unique_images_check]
            print(
                f"Limiting existence checks to first {len(unique_list):,} unique images"
            )

        missing = 0
        present = 0
        for rel in tqdm(unique_list, desc="Checking image files", unit="img"):
            p = dataset_dir / rel
            if p.exists():
                present += 1
            else:
                missing += 1

        print("\nImage file existence results (unique images checked):")
        print(f"  Present: {present:,}")
        print(f"  Missing: {missing:,}")


if __name__ == "__main__":
    main()
