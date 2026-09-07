# train_llm/train_dataset.py
from typing import Optional, Iterator, Dict, List, Tuple, Union
import os
import json
import shutil
import time
import random
import itertools  # Added for fallback sharding
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    IterableDataset,
    Features,
    Sequence,
    Value,
)
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

# Common schema for packed blocks
FEATURES = Features({"input_ids": Sequence(Value("int64"))})


def set_hf_cache_env(cache_dir: Optional[str]):
    """
    Configure Hugging Face cache env variables if a cache_dir is provided.
    """
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE",
                          os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE",
                          os.path.join(cache_dir, "hub"))
    logger.info(
        f"HF cache configured: HF_HOME={os.environ.get('HF_HOME')} "
        f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}"
    )


def _skip_by_tokens(text_iter: Iterator[str], tokenizer, tokens_to_skip: int) -> Iterator[str]:
    """
    Deterministically skip the first 'tokens_to_skip' tokens across the stream of texts.
    """
    if tokens_to_skip <= 0:
        for t in text_iter:
            yield t
        return

    skipped = 0
    for t in text_iter:
        ids = tokenizer(t, add_special_tokens=True)["input_ids"]
        if not ids:
            continue
        # +1 for EOS when packing
        if skipped + len(ids) + 1 <= tokens_to_skip:
            skipped += len(ids) + 1
            continue
        # consume the remainder of the skip in this first kept example
        remain = tokens_to_skip - skipped
        ids = ids[remain:]
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        yield txt
        break

    # pass the remainder
    for t in text_iter:
        yield t


def _pack_text_to_blocks(
    text_iterator: Iterator[str],
    tokenizer,
    seq_len: int,
    max_blocks: Optional[int],
) -> Iterator[Dict[str, List[int]]]:
    """
    Consumes an iterator of text documents and yields packed, fixed-length token blocks.
    Documents are separated with the tokenizer's EOS token.
    """
    buffer: List[int] = []
    produced = 0
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        if tokenizer.pad_token_id is not None:
            eos_token_id = tokenizer.pad_token_id
        else:
            raise ValueError(
                "Tokenizer must have an EOS token for document separation."
            )

    for txt in text_iterator:
        ids = tokenizer(txt, add_special_tokens=True)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos_token_id)

        while len(buffer) >= seq_len:
            yield {"input_ids": buffer[:seq_len]}
            buffer = buffer[seq_len:]
            produced += 1
            if max_blocks is not None and max_blocks > 0 and produced >= max_blocks:
                return


def _stream_hf_text(
    *,
    hf_dataset: str,
    hf_name: Optional[str],
    hf_split: str,
    cache_dir: Optional[str],
    text_field: str,
    seed: int,
    buffer_size: int,
    shard_by_rank: bool = False,
    shard_idx: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> Iterator[str]:
    """
    Build a streaming HF dataset iterator (shuffled) and yield the text field as strings.
    """
    
    # --- RATE LIMIT HANDLING ---
    # Stagger processes to avoid hitting HF 429 Too Many Requests
    if shard_idx is not None and num_shards is not None and num_shards > 1:
        sleep_delay = (shard_idx * 1.5) + random.uniform(0.1, 1.0)
        time.sleep(sleep_delay)

    ds = None
    max_retries = 10
    base_wait = 2.0

    for attempt in range(max_retries):
        try:
            ds = load_dataset(
                hf_dataset,
                name=hf_name,
                split=hf_split,
                streaming=True,
                cache_dir=cache_dir,
            )
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg or "connection" in msg.lower():
                wait_time = base_wait * (1.5 ** attempt) + random.uniform(0, 2.0)
                logger.warning(
                    f"load_dataset rate limited (attempt {attempt+1}/{max_retries}). "
                    f"Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                raise e
    
    if ds is None:
        raise RuntimeError(f"Failed to load dataset {hf_dataset} after {max_retries} attempts.")

    _shard_idx = None
    _num_shards = None

    if shard_by_rank:
        _shard_idx = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        _num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    elif shard_idx is not None and num_shards is not None:
        _shard_idx = shard_idx
        _num_shards = num_shards

    # Apply sharding
    use_islice_fallback = False
    
    if _num_shards is not None and _num_shards > 1:
        _shard_idx = _shard_idx % _num_shards
        try:
            # Try native sharding (file-based, efficient)
            ds = ds.shard(num_shards=_num_shards, index=_shard_idx)
        except Exception as e:
            # Fallback to islice (iterator-based, 100% safe but less bandwidth efficient)
            logger.warning(
                f"Native sharding failed with error: {e}. "
                f"Falling back to itertools.islice for shard {_shard_idx}/{_num_shards}. "
                "This guarantees correctness (no duplicates) but may download more data."
            )
            use_islice_fallback = True

    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    def text_iter():
        iterator = iter(ds)
        
        # If native sharding failed, we mechanically skip examples
        if use_islice_fallback:
            # islice(iterable, start, stop, step) -> e.g. indices 0, 12, 24... for rank 0 of 12
            iterator = itertools.islice(iterator, _shard_idx, None, _num_shards)
            
        for ex in iterator:
            t = ex.get(text_field, None)
            if isinstance(t, str):
                yield t

    return text_iter()


def build_streaming_packed_dataset_from_hf(
    hf_dataset: str,
    hf_name: Optional[str],
    hf_split: str,
    cache_dir: Optional[str],
    text_field: str,
    tokenizer_name: str,
    seq_len: int,
    buffer_size: int,
    seed: int,
    num_samples: int,
    initial_skip_tokens: int,
    max_blocks: Optional[int],
    shard_by_rank: bool = True,
) -> IterableDataset:
    """
    Build a streaming IterableDataset of packed blocks from an HF streaming dataset.
    """
    set_hf_cache_env(cache_dir)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    features = FEATURES

    def streaming_block_gen(
        hf_dataset: str,
        hf_name: Optional[str],
        hf_split: str,
        cache_dir: Optional[str],
        text_field: str,
        tokenizer_name: str,
        seq_len: int,
        buffer_size: int,
        seed: int,
        num_samples: int,
        initial_skip_tokens: int,
        max_blocks: Optional[int],
        shard_by_rank: bool,
    ):
        tok_local = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tok_local.pad_token is None and tok_local.eos_token is not None:
            tok_local.pad_token = tok_local.eos_token

        text_src = _stream_hf_text(
            hf_dataset=hf_dataset,
            hf_name=hf_name,
            hf_split=hf_split,
            cache_dir=cache_dir,
            text_field=text_field,
            seed=seed,
            buffer_size=buffer_size,
            shard_by_rank=shard_by_rank,
        )

        text_src = _skip_by_tokens(
            text_src, tok_local, int(initial_skip_tokens or 0)
        )

        def limited_text_iter():
            count = 0
            for t in text_src:
                yield t
                count += 1
                if num_samples and count >= num_samples:
                    break

        yield from _pack_text_to_blocks(
            text_iterator=limited_text_iter(),
            tokenizer=tok_local,
            seq_len=seq_len,
            max_blocks=max_blocks,
        )

    return IterableDataset.from_generator(
        streaming_block_gen,
        features=features,
        gen_kwargs=dict(
            hf_dataset=hf_dataset,
            hf_name=hf_name,
            hf_split=hf_split,
            cache_dir=cache_dir,
            text_field=text_field,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            buffer_size=buffer_size,
            seed=seed,
            num_samples=int(num_samples or 0),
            initial_skip_tokens=int(initial_skip_tokens or 0),
            max_blocks=max_blocks,
            shard_by_rank=shard_by_rank,
        ),
    )


def _materialization_block_gen(
    hf_dataset: str,
    hf_name: Optional[str],
    hf_split: str,
    cache_dir: Optional[str],
    text_field: str,
    tokenizer_name: str,
    seq_len: int,
    buffer_size: int,
    seed: int,
    num_samples: int,
    initial_skip_tokens: int,
    max_blocks: Optional[int],
    shard_idx: Union[int, List[int]] = 0,
    num_shards: int = 1,
):
    """
    Top-level generator function used by Dataset.from_generator for materialization.
    """
    # Robust unpacking: if datasets passes a list (chunk), take the first element
    if isinstance(shard_idx, list):
        shard_idx = shard_idx[0]
    if isinstance(num_shards, list):
        num_shards = num_shards[0]

    tok_local = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok_local.pad_token is None and tok_local.eos_token is not None:
        tok_local.pad_token = tok_local.eos_token

    # Use explicit sharding (shard_idx / num_shards)
    text_src = _stream_hf_text(
        hf_dataset=hf_dataset,
        hf_name=hf_name,
        hf_split=hf_split,
        cache_dir=cache_dir,
        text_field=text_field,
        seed=seed,
        buffer_size=buffer_size,
        shard_by_rank=False,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )

    text_src = _skip_by_tokens(
        text_src, tok_local, int(initial_skip_tokens or 0)
    )

    def limited_text_iter():
        count = 0
        for t in text_src:
            yield t
            count += 1
            if num_samples and count >= num_samples:
                break

    yield from _pack_text_to_blocks(
        text_iterator=limited_text_iter(),
        tokenizer=tok_local,
        seq_len=seq_len,
        max_blocks=max_blocks,
    )


def materialize_packed_dataset_to_disk(
    *,
    out_dir: str,
    hf_dataset: str,
    hf_name: Optional[str],
    hf_split: str,
    cache_dir: Optional[str],
    text_field: str,
    tokenizer_name: str,
    seq_len: int,
    buffer_size: int,
    seed: int,
    num_samples: int,
    initial_skip_tokens: int,
    max_blocks: Optional[int],
    num_proc: int = 1,
) -> str:
    """
    Build the same packed blocks as streaming, but materialize to disk.
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    if os.path.exists(out_dir):
        logger.info(f"Found existing packed dataset at: {out_dir}.")
        return out_dir

    if rank != 0 and world > 1:
        logger.info(f"[rank {rank}] Waiting for dataset materialization at: {out_dir}")
        wait_sec = 0
        timeout = 3600
        while not os.path.exists(out_dir) and wait_sec < timeout:
            time.sleep(2)
            wait_sec += 2
        if not os.path.exists(out_dir):
            raise TimeoutError(f"[rank {rank}] Timed out waiting for dataset at {out_dir}")
        logger.info(f"[rank {rank}] Detected dataset at: {out_dir}")
        return out_dir

    tmp_dir = out_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    set_hf_cache_env(cache_dir)

    # Calculate per-process limits
    per_proc_samples = int(num_samples or 0) // max(1, num_proc)
    per_proc_blocks = (max_blocks // num_proc) if max_blocks else None

    # Base arguments (constants passed as scalars to be broadcasted)
    gen_kwargs = dict(
        hf_dataset=hf_dataset,
        hf_name=hf_name,
        hf_split=hf_split,
        cache_dir=cache_dir,
        text_field=text_field,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        buffer_size=buffer_size,
        seed=seed,
        num_samples=per_proc_samples,
        initial_skip_tokens=int(initial_skip_tokens or 0),
        max_blocks=per_proc_blocks,
        num_shards=num_proc,
    )

    if num_proc > 1:
        # Sharding argument: passed as a list of length num_proc.
        gen_kwargs['shard_idx'] = [i for i in range(num_proc)]
    else:
        gen_kwargs['shard_idx'] = 0

    ds: Dataset = Dataset.from_generator(
        _materialization_block_gen,
        features=FEATURES,
        gen_kwargs=gen_kwargs,
        num_proc=num_proc,
    )

    ds.save_to_disk(tmp_dir)

    meta = {
        "hf_dataset": hf_dataset,
        "hf_name": hf_name,
        "hf_split": hf_split,
        "text_field": text_field,
        "seq_len": seq_len,
        "tokenizer": tokenizer_name,
        "num_rows": len(ds),
        "seed": seed,
        "buffer_size": buffer_size,
        "num_samples": int(num_samples or 0),
        "initial_skip_tokens": int(initial_skip_tokens or 0),
        "max_blocks": max_blocks,
        "num_proc": num_proc,
    }
    with open(os.path.join(tmp_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    os.replace(tmp_dir, out_dir)
    logger.info(f"[rank {rank}] Materialized packed dataset: {out_dir} (rows={len(ds)})")
    return out_dir


def load_packed_dataset_from_disk(out_dir: str) -> Dataset:
    return load_from_disk(out_dir)


def load_train_eval_datasets_from_cfg(
    cfg: dict,
    tokenizer_name_or_path: str,
):
    """
    Produce train_dataset and optional eval_dataset according to cfg["training"]["dataset"].
    """
    tr = cfg["training"]
    ds_cfg = tr.get("dataset", {}) or {}
    src = (ds_cfg.get("source") or "hf_streaming").lower()

    if src == "pretokenized":
        src = "packed_disk"

    ds_cache_dir = ds_cfg.get("cache_dir") or cfg.get("data", {}).get("cache_dir")
    seq_len = int(tr["seq_len"])
    num_samples = int(tr.get("num_samples", 0) or 0)
    seed = int(tr.get("seed", 42))

    eval_cfg = tr.get("evaluation", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    eval_max_blocks = int(eval_cfg.get("eval_max_blocks", 512))
    eval_split_ratio = float(eval_cfg.get("eval_split_ratio", 0.01))

    if src == "hf_streaming":
        hf_dataset = ds_cfg.get("hf_dataset", "HuggingFaceFW/fineweb")
        use_full = bool(ds_cfg.get("use_full", False))
        hf_name = None if use_full else ds_cfg.get("hf_name", None)
        hf_split = ds_cfg.get("hf_split", "train")
        text_field = ds_cfg.get("text_field", "text")
        buffer_size = int(ds_cfg.get("streaming_buffer_size", 100000))
        initial_skip = int(ds_cfg.get("initial_random_skip_tokens", 0))
        max_blocks = ds_cfg.get("max_blocks", None)
        tokenizer_ref = tokenizer_name_or_path

        train_dataset = build_streaming_packed_dataset_from_hf(
            hf_dataset=hf_dataset,
            hf_name=hf_name,
            hf_split=hf_split,
            cache_dir=ds_cache_dir,
            text_field=text_field,
            tokenizer_name=tokenizer_ref,
            seq_len=seq_len,
            buffer_size=buffer_size,
            seed=seed,
            num_samples=num_samples,
            initial_skip_tokens=initial_skip,
            max_blocks=max_blocks,
            shard_by_rank=True,
        )

        if eval_enabled:
            eval_dataset = build_streaming_packed_dataset_from_hf(
                hf_dataset=hf_dataset,
                hf_name=hf_name,
                hf_split=hf_split,
                cache_dir=ds_cache_dir,
                text_field=text_field,
                tokenizer_name=tokenizer_ref,
                seq_len=seq_len,
                buffer_size=buffer_size,
                seed=seed + 1,
                num_samples=num_samples,
                initial_skip_tokens=max(0, initial_skip // 2),
                max_blocks=eval_max_blocks,
                shard_by_rank=True,
            )
        else:
            eval_dataset = None

        return train_dataset, eval_dataset

    if src == "packed_disk":
        out_dir = ds_cfg.get("data_dir")
        if not out_dir:
            raise ValueError(
                "training.dataset.data_dir must be set for source=packed_disk"
            )

        prepare = bool(ds_cfg.get("prepare_if_missing", False))
        if not os.path.exists(out_dir):
            if not prepare:
                raise FileNotFoundError(
                    f"Packed dataset not found: {out_dir}. "
                    f"Set training.dataset.prepare_if_missing=true to build it automatically."
                )

            hf_dataset = ds_cfg.get("hf_dataset", "HuggingFaceFW/fineweb")
            use_full = bool(ds_cfg.get("use_full", False))
            hf_name = None if use_full else ds_cfg.get("hf_name", None)
            hf_split = ds_cfg.get("hf_split", "train")
            text_field = ds_cfg.get("text_field", "text")
            buffer_size = int(ds_cfg.get("streaming_buffer_size", 100000))
            initial_skip = int(ds_cfg.get("initial_random_skip_tokens", 0))
            max_blocks = ds_cfg.get("max_blocks", None)
            tokenizer_ref = tokenizer_name_or_path
            # Check for config num_proc
            num_proc = int(ds_cfg.get("num_proc", 1))

            logger.info(f"Building packed dataset to: {out_dir} with num_proc={num_proc}")
            materialize_packed_dataset_to_disk(
                out_dir=out_dir,
                hf_dataset=hf_dataset,
                hf_name=hf_name,
                hf_split=hf_split,
                cache_dir=ds_cache_dir,
                text_field=text_field,
                tokenizer_name=tokenizer_ref,
                seq_len=seq_len,
                buffer_size=buffer_size,
                seed=seed,
                num_samples=num_samples,
                initial_skip_tokens=initial_skip,
                max_blocks=max_blocks,
                num_proc=num_proc,
            )

        ds = load_packed_dataset_from_disk(out_dir)

        if eval_enabled:
            split = ds.train_test_split(
                test_size=eval_split_ratio, seed=seed, shuffle=True
            )
            eval_dataset = split["test"]
            train_dataset = split["train"]
        else:
            eval_dataset = None
            train_dataset = ds

        return train_dataset, eval_dataset

    raise ValueError(f"Unknown dataset source: {src}")