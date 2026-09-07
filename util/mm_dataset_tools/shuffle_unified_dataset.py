#!/usr/bin/env python3
import argparse
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm


def make_order_key(seed: int, uid: str, group: Optional[str] = None) -> int:
    # 64-bit deterministic key
    s = f"{seed}|{uid}" if group is None else f"{seed}|{group}|{uid}"
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def shard_for(order_key: int, num_shards: int) -> int:
    return int(order_key % num_shards)


class GroupShardedWriter:
    """
    Writes parquet parts into:
      out_root/[group=...]/shard_id=XXXXX/part-YYYYYY.parquet

    Rows are buffered per (group, shard_id) to keep memory bounded.
    """

    def __init__(
        self,
        out_root: Path,
        compression: str,
        buffer_rows_per_shard: int,
        separate_groups: bool,
        schema: pa.Schema,
    ):
        self.out_root = out_root
        self.compression = compression
        self.buffer_rows_per_shard = buffer_rows_per_shard
        self.separate_groups = separate_groups
        # Explicit schema so output types match the input exactly
        # (e.g. uint64 order_key, large_string text columns).
        self.schema = schema

        self.out_root.mkdir(parents=True, exist_ok=True)

        self.buffers: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        self.part_counters: Dict[str, Dict[int, int]] = {}

    def _dir_for(self, group: str, shard_id: int) -> Path:
        base = self.out_root
        if self.separate_groups:
            base = base / f"group={group}"
        d = base / f"shard_id={shard_id:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_part_path(self, group: str, shard_id: int) -> Path:
        counters = self.part_counters.setdefault(group, {})
        part = counters.get(shard_id, 0)
        counters[shard_id] = part + 1
        return self._dir_for(group, shard_id) / f"part-{part:06d}.parquet"

    def _flush(self, group: str, shard_id: int):
        rows = self.buffers.get(group, {}).get(shard_id, [])
        if not rows:
            return
        rows.sort(key=lambda r: r["order_key"])
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(
            table, self._next_part_path(group, shard_id), compression=self.compression
        )
        self.buffers[group][shard_id] = []

    def write(self, group: str, shard_id: int, row: Dict[str, Any]):
        self.buffers.setdefault(group, {}).setdefault(shard_id, []).append(row)
        if len(self.buffers[group][shard_id]) >= self.buffer_rows_per_shard:
            self._flush(group, shard_id)

    def close(self):
        for group, shard_map in self.buffers.items():
            for shard_id in list(shard_map.keys()):
                self._flush(group, shard_id)


def main():
    ap = argparse.ArgumentParser(
        description="Shuffle unified dataset parquet in place (global or separate text/mm)."
    )
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--parquet_dirname", type=str, default="parquet")
    ap.add_argument("--parquet_glob", type=str, default="**/part-*.parquet")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num_shards", type=int, default=2048)
    ap.add_argument("--compression", type=str, default="zstd")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--buffer_rows_per_shard", type=int, default=2000)

    ap.add_argument("--mode", choices=["global", "separate"], default="global")
    ap.add_argument(
        "--text_first",
        action="store_true",
        help="Only with --mode separate. Writes to group=text and group=multimodal directories (so text can be loaded first by path ordering).",
    )

    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Write shuffled parquet here (no in-place swap).",
    )
    ap.add_argument(
        "--no_in_place",
        action="store_true",
        help="If set and --out_dir is not provided, do not swap into place; keep tmp dir.",
    )
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    parquet_root = dataset_dir / args.parquet_dirname
    if not parquet_root.exists():
        raise SystemExit(f"Parquet dir not found: {parquet_root}")

    parquet_paths = sorted(parquet_root.glob(args.parquet_glob))
    if not parquet_paths:
        raise SystemExit(
            f"No parquet files found under: {parquet_root / args.parquet_glob}"
        )

    dset = ds.dataset([str(p) for p in parquet_paths], format="parquet")

    required_cols = [
        "uid",
        "dataset",
        "subset",
        "task",
        "shard_id",
        "order_key",
        "image_path",
        "prompt",
        "target",
        "meta_json",
    ]
    for c in required_cols:
        if c not in dset.schema.names:
            raise SystemExit(
                f"Missing required column '{c}'. Found: {dset.schema.names}"
            )

    separate_groups = args.mode == "separate"

    # Output location
    if args.out_dir is not None:
        out_root = args.out_dir
        out_root.mkdir(parents=True, exist_ok=True)
        do_in_place = False
    else:
        out_root = (
            dataset_dir / f"{args.parquet_dirname}__tmp_shuffle_{int(time.time())}"
        )
        do_in_place = not args.no_in_place

    writer = GroupShardedWriter(
        out_root=out_root,
        compression=args.compression,
        buffer_rows_per_shard=args.buffer_rows_per_shard,
        separate_groups=separate_groups,
        schema=dset.schema,
    )

    # Scan all columns (not just required_cols) and write them back with the
    # original schema, so no column is dropped and types are preserved.
    scanner = dset.scanner(batch_size=args.batch_size)

    total = 0
    text_rows = 0
    mm_rows = 0

    for batch in tqdm(scanner.to_batches(), desc="Shuffling (streaming)", unit="batch"):
        tbl = pa.Table.from_batches([batch])
        rows = tbl.to_pylist()

        for r in rows:
            uid = r["uid"]
            is_text = r.get("image_path") is None

            if not separate_groups:
                group = "all"
                ok = make_order_key(args.seed, uid)
            else:
                group = "text" if is_text else "multimodal"
                ok = make_order_key(args.seed, uid, group=group)

            sid = shard_for(ok, args.num_shards)
            r["order_key"] = int(ok)
            r["shard_id"] = int(sid)

            writer.write(group=group, shard_id=sid, row=r)

            total += 1
            if separate_groups:
                if is_text:
                    text_rows += 1
                else:
                    mm_rows += 1

    writer.close()

    print("\nShuffle summary:")
    print(f"  Total rows processed: {total:,}")
    if separate_groups:
        print(f"  Text rows: {text_rows:,}")
        print(f"  Multimodal rows: {mm_rows:,}")
        if args.text_first:
            print(
                "  Output uses group=text/ and group=multimodal/ directories (load text first by sorting paths)."
            )

    # In-place swap
    if args.out_dir is None and do_in_place:
        backup = dataset_dir / f"{args.parquet_dirname}__backup_{int(time.time())}"
        print(f"\nIn-place swap:")
        print(f"  Moving current parquet -> {backup}")
        print(f"  Moving new parquet     -> {parquet_root}")

        shutil.move(str(parquet_root), str(backup))
        shutil.move(str(out_root), str(parquet_root))

        print(f"\nDone. Backup kept at: {backup}")
    else:
        print(f"\nDone. Shuffled parquet written to: {out_root}")
        if args.out_dir is None:
            print("Note: in-place swap was not performed.")


if __name__ == "__main__":
    main()
