#!/usr/bin/env python3
"""
Download HuggingFaceTB/stack-edu (all language subsets).

For each row the HF dataset only stores repo_name + path.
This script fetches the actual source code from GitHub raw content:
    https://raw.githubusercontent.com/<repo>/refs/heads/main/<path>

The resulting Arrow dataset (per language) contains a ``text`` column with
the fetched code, plus the original metadata columns.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import Dataset, load_dataset
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

REPO_ID = "HuggingFaceTB/stack-edu"

ALL_LANGUAGES = [
    "C",
    "CSharp",
    "Cpp",
    "Go",
    "Java",
    "JavaScript",
    "Markdown",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "SQL",
    "Shell",
    "Swift",
    "TypeScript",
]


# ── HTTP helpers ────────────────────────────────────────────────────────────


def _make_session(token: str | None = None) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=96, pool_maxsize=96)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    if token:
        sess.headers["Authorization"] = f"token {token}"
    sess.headers["User-Agent"] = "stack-edu-downloader/1.0"
    return sess


def _fetch_one(sess: requests.Session, repo_name: str, file_path: str) -> str | None:
    """Return file contents as str, or None on any failure."""
    file_path = file_path.lstrip("/")
    encoded_path = quote(file_path, safe="/")
    encoded_repo = quote(repo_name, safe="/")
    url = (
        f"https://raw.githubusercontent.com/"
        f"{encoded_repo}/refs/heads/main/{encoded_path}"
    )
    try:
        resp = sess.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None


# ── Batch fetcher (threaded) ───────────────────────────────────────────────


def fetch_batch(
    sess: requests.Session,
    repo_names: list[str],
    paths: list[str],
    max_workers: int,
) -> list[str | None]:
    """Fetch a batch of files concurrently. Returns list aligned with inputs."""
    results: list[str | None] = [None] * len(repo_names)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for i, (repo, path) in enumerate(zip(repo_names, paths)):
            future_to_idx[pool.submit(_fetch_one, sess, repo, path)] = i
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
    return results


# ── Per-language pipeline ──────────────────────────────────────────────────


def process_language(
    lang: str,
    output_dir: Path,
    cache_dir: Path,
    max_workers: int,
    batch_size: int,
    sess: requests.Session,
    max_rows: int | None = None,
):
    lang_dir = output_dir / lang
    done_marker = lang_dir / "_DONE"
    if done_marker.exists():
        print(f"  [{lang}] Already completed – skipping.")
        return

    lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{lang}] Loading metadata from HuggingFace …")
    ds = load_dataset(REPO_ID, lang, split="train", cache_dir=str(cache_dir / lang))
    total = len(ds)
    if max_rows is not None:
        total = min(total, max_rows)
    print(f"  [{lang}] {total:,} rows to process")

    # ── Checkpoint: which batches are already done ──
    checkpoint_file = lang_dir / "_batch_checkpoint.txt"
    done_batches: set[int] = set()
    if checkpoint_file.exists():
        done_batches = set(
            int(x) for x in checkpoint_file.read_text().strip().split("\n") if x
        )
        print(f"  [{lang}] Resuming – {len(done_batches)} batches already done")

    # We accumulate per-batch parquet files, then combine at the end.
    batch_dir = lang_dir / "_batches"
    batch_dir.mkdir(exist_ok=True)

    num_batches = (total + batch_size - 1) // batch_size
    fetched_total = 0
    failed_total = 0

    for b_idx in tqdm(range(num_batches), desc=f"  {lang}", unit="batch"):
        if b_idx in done_batches:
            continue

        start = b_idx * batch_size
        end = min(start + batch_size, total)
        batch_slice = ds.select(range(start, end))

        repo_names = batch_slice["repo_name"]
        paths = batch_slice["path"]

        texts = fetch_batch(sess, repo_names, paths, max_workers)

        # Build filtered rows (only keep successful fetches)
        rows = {
            "text": [],
            "language": [],
            "repo_name": [],
            "path": [],
        }
        ok = 0
        for i, txt in enumerate(texts):
            if txt is not None and len(txt.strip()) > 0:
                rows["text"].append(txt)
                rows["language"].append(lang)
                rows["repo_name"].append(repo_names[i])
                rows["path"].append(paths[i])
                ok += 1

        fetched_total += ok
        failed_total += (end - start) - ok

        if rows["text"]:
            batch_ds = Dataset.from_dict(rows)
            batch_ds.save_to_disk(str(batch_dir / f"batch_{b_idx:06d}"))

        # Checkpoint
        done_batches.add(b_idx)
        checkpoint_file.write_text("\n".join(str(x) for x in sorted(done_batches)))

    # ── Combine all batch datasets ──
    print(
        f"  [{lang}] Combining batches …  (fetched={fetched_total:,}  failed={failed_total:,})"
    )
    from datasets import concatenate_datasets, load_from_disk

    batch_paths = sorted(batch_dir.glob("batch_*"))
    if not batch_paths:
        print(f"  [{lang}] WARNING: no data fetched at all.")
        done_marker.write_text("empty")
        return

    parts = []
    for bp in batch_paths:
        try:
            parts.append(load_from_disk(str(bp)))
        except Exception as e:
            print(f"  [{lang}] WARNING: could not load {bp}: {e}")

    if parts:
        combined = concatenate_datasets(parts)
        combined.save_to_disk(str(lang_dir))
        print(f"  [{lang}] Saved {len(combined):,} rows to {lang_dir}")
    else:
        print(f"  [{lang}] WARNING: no valid batches.")

    done_marker.write_text(f"rows={len(combined) if parts else 0}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFaceTB/stack-edu with source code fetched from GitHub"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/stack_edu"),
        help="Root output directory (one subdirectory per language)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/stack_edu/_hf_cache"),
        help="HuggingFace cache directory for metadata downloads",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help=f"Which language subsets to download (default: all). Choices: {ALL_LANGUAGES}",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=48,
        help="Max concurrent HTTP fetches per batch (default: 48)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Rows per processing batch (default: 2000)",
    )
    parser.add_argument(
        "--max_rows_per_language",
        type=int,
        default=None,
        help="Cap the number of rows processed per language (default: all)",
    )
    parser.add_argument(
        "--github_token",
        type=str,
        default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var). "
        "Highly recommended for rate-limit (5000 req/hr vs 60 req/hr).",
    )
    args = parser.parse_args()

    languages = args.languages or ALL_LANGUAGES
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "⚠️  No GitHub token provided. Rate limit will be ~60 req/hr.\n"
            "   Set --github_token or GITHUB_TOKEN env var for 5000 req/hr.\n"
        )
    else:
        print("✅ GitHub token detected.\n")

    sess = _make_session(token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"\n{'=' * 64}")
        print(f"  Language: {lang}")
        print(f"{'=' * 64}")
        process_language(
            lang=lang,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            sess=sess,
            max_rows=args.max_rows_per_language,
        )

    print("\n✅ stack-edu download complete")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
