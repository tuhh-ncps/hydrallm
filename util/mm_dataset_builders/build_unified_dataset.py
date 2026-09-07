#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

MAX_IMAGE_PIXELS = 42_000_000  # 42 MP
JPEG_QUALITY = 95


# ----------------------------
# Utilities
# ----------------------------


def rescale_if_needed(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale image so that width * height <= max_pixels. Aspect ratio preserved."""
    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / pixels)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_image_token(
    text: Optional[str], start_of_image_token: str
) -> Optional[str]:
    if text is None:
        return None
    return text.replace("<image>", start_of_image_token)


def ensure_starts_with_soi(
    prompt: Optional[str], soi: str, has_image: bool
) -> Optional[str]:
    if prompt is None:
        return None
    if not has_image:
        return prompt
    if prompt.startswith(soi):
        return prompt
    if prompt.strip() == "":
        return soi
    return f"{soi}\n{prompt}"


def guess_ext_from_bytes(img_bytes: bytes) -> str:
    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if img_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def url_to_local_filename_sha1(url: str) -> str:
    """
    Must match the earlier URL downloaders: sha1(url) + ext from URL (fallback .jpg).
    """
    h = sha1_text(url)
    ext = os.path.splitext(url.split("?")[0])[1]
    if not ext or len(ext) > 5:
        ext = ".jpg"
    return f"{h}{ext}"


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


# ----------------------------
# Image Store (content-addressed)
# ----------------------------


@dataclass
class ImageRef:
    image_id: str
    image_path: str  # relative to output_dir


class ImageStore:
    def __init__(self, root: Path, output_dir: Path):
        self.root = root
        self.output_dir = output_dir
        safe_mkdir(self.root)

    def put_bytes(
        self, img_bytes: bytes, preferred_ext: Optional[str] = None
    ) -> ImageRef:
        h = sha256_bytes(img_bytes)
        image_id = f"sha256:{h}"

        subdir = self.root / h[:2] / h[2:4]
        safe_mkdir(subdir)

        ext = preferred_ext or guess_ext_from_bytes(img_bytes)
        out_path = subdir / f"{h}{ext}"

        if not out_path.exists():
            out_path.write_bytes(img_bytes)

        rel = out_path.relative_to(self.output_dir).as_posix()
        return ImageRef(image_id=image_id, image_path=rel)

    def put_file(self, path: Path) -> ImageRef:
        """
        Decode image with PIL, validate, convert to RGB,
        downscale to <= 42MP if needed, and re-encode as JPEG (deduped by bytes).
        """
        try:
            with Image.open(path) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = rescale_if_needed(img, MAX_IMAGE_PIXELS)
                return self.put_pil_as_jpeg(img, quality=JPEG_QUALITY)
        except Exception as e:
            raise RuntimeError(f"Invalid or corrupted image: {path}") from e

    def put_pil_as_jpeg(self, img: Image.Image, quality: int = 95) -> ImageRef:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = rescale_if_needed(img, MAX_IMAGE_PIXELS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return self.put_bytes(buf.getvalue(), preferred_ext=".jpg")


# ----------------------------
# Parquet sharded writer
# ----------------------------


class ParquetShardedWriter:
    def __init__(
        self, out_parquet_root: Path, num_shards: int, compression: str = "zstd"
    ):
        self.out_parquet_root = out_parquet_root
        self.num_shards = num_shards
        self.compression = compression
        safe_mkdir(self.out_parquet_root)
        self.part_counters: Dict[int, int] = {}

        # order_key is uint64 (important)
        self.schema = pa.schema(
            [
                ("uid", pa.string()),
                ("dataset", pa.string()),
                ("subset", pa.string()),
                ("task", pa.string()),
                ("shard_id", pa.int32()),
                ("order_key", pa.uint64()),
                ("image_id", pa.string()),
                ("image_path", pa.string()),
                ("prompt", pa.large_string()),
                ("target", pa.large_string()),
                ("meta_json", pa.large_string()),
            ]
        )

    def _next_part_path(self, shard_id: int) -> Path:
        part = self.part_counters.get(shard_id, 0)
        self.part_counters[shard_id] = part + 1
        shard_dir = self.out_parquet_root / f"shard_id={shard_id:05d}"
        safe_mkdir(shard_dir)
        return shard_dir / f"part-{part:06d}.parquet"

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        by_shard: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            sid = int(r["shard_id"])
            by_shard.setdefault(sid, []).append(r)

        for sid, shard_rows in by_shard.items():
            shard_rows.sort(key=lambda x: x["order_key"])
            table = pa.Table.from_pylist(shard_rows, schema=self.schema)
            out_path = self._next_part_path(sid)
            pq.write_table(table, out_path, compression=self.compression)


# ----------------------------
# Deterministic uid/order/shard
# ----------------------------


def make_uid(dataset: str, subset: str, task: str, source_id: str) -> str:
    return sha1_text(f"{dataset}|{subset}|{task}|{source_id}")


def make_order_key(seed: int, uid: str) -> int:
    h = hashlib.sha1(f"{seed}|{uid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def shard_for(order_key: int, num_shards: int) -> int:
    return int(order_key % num_shards)


# ----------------------------
# CoSyn triple extraction
# ----------------------------


def extract_cosyn_triples(row: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Return list of (question, explanation/reasoning, answer) triples.
    Handles list-of-dicts and dict-of-lists (including 'questions' dict).
    """
    triples: List[Tuple[str, str, str]] = []

    qa_pairs = row.get("qa_pairs")
    questions = row.get("questions")

    # Case 1: list of dicts
    if isinstance(qa_pairs, list):
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            q = (qa.get("question") or "").strip()
            r = (qa.get("reasoning") or qa.get("explanation") or "").strip()
            a = (qa.get("answer") or "").strip()
            if q and a:
                triples.append((q, r, a))
        return triples

    def _from_dict_of_lists(d: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        qs = d.get("question") or d.get("questions") or []
        rs = d.get("reasoning")
        if rs is None:
            rs = d.get("explanation")
        if rs is None:
            rs = []
        ans = d.get("answer") or d.get("answers") or []

        out: List[Tuple[str, str, str]] = []
        n = min(len(qs), len(ans))
        for i in range(n):
            q = (qs[i] or "").strip()
            r = ((rs[i] if i < len(rs) else "") or "").strip()
            a = (ans[i] or "").strip()
            if q and a:
                out.append((q, r, a))
        return out

    # Case 2: dict of lists under qa_pairs
    if isinstance(qa_pairs, dict):
        return _from_dict_of_lists(qa_pairs)

    # Case 3: dict of lists under questions
    if isinstance(questions, dict):
        return _from_dict_of_lists(questions)

    return triples


# ----------------------------
# Dataset adapters (CANDIDATE generators)
# IMPORTANT: These do NOT stop at num_samples anymore.
# The main loop enforces "exactly N WRITTEN samples" per dataset.
# ----------------------------


# ----------------------------
# stack_edu (code, text-only)
# ----------------------------


def iter_stack_edu_candidates(cfg: Dict[str, Any]):
    """
    Text-only code dataset with per-language quotas (managed internally,
    like cosyn_400k).  Each language lives in ``<root>/<Language>/`` as an
    Arrow dataset produced by download_stack_edu.py.

    Expected dataset columns: text, language, repo_name, path
    """
    root = Path(cfg["path"])
    per_language: Dict[str, int] = cfg.get("per_language", {})

    if not per_language:
        warn("stack_edu: no per_language quotas configured – nothing to emit.")
        return

    for lang, quota in per_language.items():
        quota = int(quota)
        lang_dir = root / lang
        if not lang_dir.exists():
            warn(f"stack_edu/{lang}: directory not found ({lang_dir}), skipping.")
            continue

        try:
            ds = load_from_disk(str(lang_dir))
        except Exception as e:
            warn(f"stack_edu/{lang}: failed to load dataset: {e}")
            continue

        count = 0
        for idx, row in enumerate(ds):
            if count >= quota:
                break
            text = row.get("text")
            if not text or not text.strip():
                continue

            meta = {
                "language": lang,
                "repo_name": row.get("repo_name"),
                "path": row.get("path"),
                "row_index": idx,
            }
            source_id = f"{lang}|{idx}"
            count += 1
            yield (None, None, text, meta, "code_lm", lang, source_id)

        if count < quota:
            warn(
                f"stack_edu/{lang}: requested {quota}, only got {count}. "
                f"Dataset may be smaller or too many empty texts."
            )


# ----------------------------
# finemath (math text, text-only)
# ----------------------------


def iter_finemath_candidates(cfg: Dict[str, Any]):
    """
    Text-only math dataset.  Works exactly like hf_text but keeps extra
    finemath metadata (token_count, etc.).

    Candidate generator – main loop enforces num_samples.
    """
    root = Path(cfg["path"])
    text_field = cfg.get("text_field", "text")

    ds = load_from_disk(str(root))

    for idx, row in enumerate(ds):
        text = row.get(text_field)
        if not text or not text.strip():
            continue

        meta = {
            "row_index": idx,
            "token_count": row.get("token_count"),
            "char_count": row.get("char_count"),
        }
        source_id = str(idx)
        yield (None, None, text, meta, "math_lm", "", source_id)


def iter_pixmo_caption_candidates(cfg: Dict[str, Any], soi: str):
    root = Path(cfg["path"])
    ds = load_from_disk(str(root / "metadata"))
    img_dir = root / "images"
    for idx, row in enumerate(ds):
        url = row.get("image_url")
        cap = row.get("caption")
        if not url or not cap:
            continue
        local = img_dir / url_to_local_filename_sha1(url)
        if not local.exists():
            continue
        yield (
            local,
            soi,
            cap,
            {"image_url": url, "row_index": idx},
            "caption",
            "",
            str(idx),
        )


def iter_pixmo_qa_candidates(cfg: Dict[str, Any], soi: str):
    root = Path(cfg["path"])
    ds = load_from_disk(str(root / "metadata"))
    img_dir = root / "images"
    include_q = bool(cfg.get("prompt", {}).get("include_question", True))

    for idx, row in enumerate(ds):
        url = row.get("image_url")
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not url or not a:
            continue
        local = img_dir / url_to_local_filename_sha1(url)
        if not local.exists():
            continue
        prompt = soi if not include_q else f"{soi}\n{q}"
        yield (
            local,
            prompt,
            a,
            {"image_url": url, "row_index": idx},
            "qa",
            "",
            str(idx),
        )


def iter_pixmo_cap_qa_candidates(cfg: Dict[str, Any], soi: str):
    root = Path(cfg["path"])
    ds = load_from_disk(str(root / "metadata"))
    img_dir = root / "images"

    for idx, row in enumerate(ds):
        url = row.get("image_url")
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or row.get("ans") or "").strip()
        if not url or not a:
            continue
        local = img_dir / url_to_local_filename_sha1(url)
        if not local.exists():
            continue
        yield (
            local,
            f"{soi}\n{q}",
            a,
            {"image_url": url, "row_index": idx},
            "qa",
            "",
            str(idx),
        )


def iter_llava_pretrain_candidates(cfg: Dict[str, Any], soi: str):
    root = Path(cfg["path"])
    images_dir = root / "images"
    json_file = root / cfg.get("json_file", "blip_laion_cc_sbu_558k.json")
    data = json.loads(json_file.read_text(encoding="utf-8"))

    for idx, row in enumerate(data):
        rel_img = row.get("image")
        conv = row.get("conversations") or []
        if not rel_img or len(conv) < 2:
            continue
        local = images_dir / rel_img
        if not local.exists():
            continue

        prompt = normalize_image_token(conv[0].get("value", ""), soi)
        target = normalize_image_token(conv[1].get("value", ""), soi)
        prompt = ensure_starts_with_soi(prompt, soi, has_image=True)

        llava_id = row.get("id", idx)
        meta = {"row_index": idx, "llava_id": llava_id, "image_rel": rel_img}
        yield (local, prompt, target, meta, "conversation", "", str(llava_id))


def iter_coco_recaptioned_candidates(cfg: Dict[str, Any], soi: str):
    root = Path(cfg["path"])
    images_dir = root / "images" / "train2017"
    json_path = root / "coco_captions.json"

    image_field = cfg.get("image_field", "image")
    caption_field = cfg.get("caption_field", "caption")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "annotations" in data:
        rows = data["annotations"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported coco_captions.json structure: {type(data)}")

    for idx, row in enumerate(rows):
        img_name = row.get(image_field) or row.get("file_name") or row.get("image_file")
        cap = row.get(caption_field) or row.get("text") or ""
        if not img_name or not cap:
            continue
        local = images_dir / str(img_name).lstrip("/")
        if not local.exists():
            continue
        yield (
            local,
            soi,
            cap,
            {"row_index": idx, "image": img_name},
            "caption",
            "",
            str(idx),
        )


def iter_cosyn_400k(cfg: Dict[str, Any], soi: str):
    """
    EXACT quotas semantics are handled here (as in your working version).

    NOTE: If an image fails to decode later (very unlikely for CoSyn),
    that would reduce the final count because the generator itself enforces
    quotas. If you want full "replace on decode-fail" for CoSyn too, we can
    refactor this similarly to other datasets, but that requires tracking quotas
    in main.
    """
    root = Path(cfg["path"])
    split = cfg.get("split", "train")
    tasks = cfg.get("tasks", {})

    vqa_cfg = tasks.get("vqa", {})
    code_cfg = tasks.get("code_generation", {})

    vqa_enabled = bool(vqa_cfg.get("enabled", True))
    code_enabled = bool(code_cfg.get("enabled", True))

    vqa_per_subset = vqa_cfg.get("per_subset") or cfg.get("per_subset") or {}
    code_per_subset = code_cfg.get("per_subset") or cfg.get("code_per_subset") or {}

    if code_enabled and not code_per_subset:
        warn(
            "cosyn_400k: code_generation enabled but no per-subset quota provided; emitting 0 code samples."
        )
        code_per_subset = {}

    code_prompts = code_cfg.get("prompts", ["Provide the code to create this figure."])
    target_format = vqa_cfg.get("target_format", "reasoning_answer")

    subset_names = sorted(set(vqa_per_subset.keys()) | set(code_per_subset.keys()))
    if not subset_names:
        warn("cosyn_400k: no subsets configured.")
        return

    for subset_name in subset_names:
        subset_path = root / subset_name
        dd = load_from_disk(str(subset_path))
        ds = dd[split] if isinstance(dd, dict) or hasattr(dd, "keys") else dd

        vqa_target = int(vqa_per_subset.get(subset_name, 0)) if vqa_enabled else 0
        code_target = int(code_per_subset.get(subset_name, 0)) if code_enabled else 0

        vqa_count = 0
        code_count = 0
        scanned_images = 0
        used_images = 0

        for idx, row in enumerate(ds):
            scanned_images += 1
            img = row.get("image")
            if img is None:
                continue

            cosyn_id = row.get("id", idx)
            used_images += 1

            if (vqa_count >= vqa_target) and (code_count >= code_target):
                break

            if vqa_enabled and vqa_count < vqa_target:
                triples = extract_cosyn_triples(row)
                remaining = vqa_target - vqa_count
                if remaining > 0 and triples:
                    for q_idx, (q, r, a) in enumerate(triples[:remaining]):
                        prompt = f"{soi}\n{q}"
                        if target_format == "reasoning_answer":
                            # include explanation/reasoning
                            if r:
                                target = f"{r}\n\nAnswer: {a}".strip()
                            else:
                                target = f"Answer: {a}".strip()
                        else:
                            target = a

                        meta = {
                            "subset": subset_name,
                            "row_index": idx,
                            "qa_index": q_idx,
                            "cosyn_id": cosyn_id,
                        }
                        source_id = f"{cosyn_id}|qa|{q_idx}"
                        vqa_count += 1
                        yield (img, prompt, target, meta, "vqa", subset_name, source_id)
                        if vqa_count >= vqa_target:
                            break

            if code_enabled and code_count < code_target:
                code = row.get("code") or row.get("program") or row.get("source_code")
                if code:
                    prompt = f"{soi}\n{{CODE_PROMPT_CHOICE}}"
                    target = str(code)
                    meta = {
                        "subset": subset_name,
                        "row_index": idx,
                        "cosyn_id": cosyn_id,
                        "_code_prompt_choices": code_prompts,
                    }
                    source_id = f"{cosyn_id}|code"
                    code_count += 1
                    yield (img, prompt, target, meta, "code", subset_name, source_id)

        if vqa_enabled and vqa_count < vqa_target:
            warn(
                f"cosyn_400k/{subset_name} VQA: requested {vqa_target}, got {vqa_count} "
                f"(scanned_images={scanned_images}, used_images={used_images})."
            )
        if code_enabled and code_count < code_target:
            warn(
                f"cosyn_400k/{subset_name} CODE: requested {code_target}, got {code_count} "
                f"(scanned_images={scanned_images}, used_images={used_images})."
            )


def iter_hf_text_candidates(cfg: Dict[str, Any]):
    """
    Text-only: no prompt; model predicts the entire text.
    Candidate generator; main enforces exact num_samples written.
    """
    root = Path(cfg["path"])
    ds = load_from_disk(str(root))
    for idx, row in enumerate(ds):
        text = row.get("text")
        if not text:
            continue
        meta = {
            "row_index": idx,
            "source_id": row.get("id"),
            "url": row.get("url"),
            "title": row.get("title"),
        }
        yield (None, None, text, meta, "lm", "", str(meta.get("source_id", idx)))


# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="YAML mix config")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 1337))
    out_dir = Path(cfg["output"]["output_dir"])
    num_shards = int(cfg["output"].get("num_shards", 1024))
    buffer_size = int(cfg["output"].get("buffer_size", 50000))
    compression = cfg["output"].get("compression", "zstd")

    soi = cfg.get("start_of_image_token", "<start_of_image>")

    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "manifest")
    (out_dir / "manifest" / "dataset_mix.yaml").write_text(
        args.config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    images_subdir = cfg.get("image_store", {}).get("dir", "images")
    image_store = ImageStore(out_dir / images_subdir, output_dir=out_dir)

    writer = ParquetShardedWriter(
        out_dir / "parquet", num_shards=num_shards, compression=compression
    )

    datasets_cfg: Dict[str, Dict[str, Any]] = cfg.get("datasets", {})

    total_rows_emitted = 0
    total_rows_written = 0
    skipped_missing_or_corrupt_images = 0

    buffer: List[Dict[str, Any]] = []

    def flush():
        nonlocal buffer, total_rows_written
        if not buffer:
            return
        buffer.sort(key=lambda r: r["order_key"])
        writer.write_rows(buffer)
        total_rows_written += len(buffer)
        buffer = []

    for dataset_name, dcfg in datasets_cfg.items():
        dtype = dcfg["type"]
        info(f"Processing {dataset_name} (type={dtype})")

        # For most datasets: num_samples means "exactly N WRITTEN rows".
        # For cosyn_400k: quotas are handled internally (per-subset per-task).
        want_written = dcfg.get("num_samples", None)
        if want_written is not None:
            want_written = int(want_written)

        if dtype == "pixmo_caption":
            gen = iter_pixmo_caption_candidates(dcfg, soi)
        elif dtype == "pixmo_qa":
            gen = iter_pixmo_qa_candidates(dcfg, soi)
        elif dtype == "pixmo_cap_qa":
            gen = iter_pixmo_cap_qa_candidates(dcfg, soi)
        elif dtype == "llava_pretrain":
            gen = iter_llava_pretrain_candidates(dcfg, soi)
        elif dtype == "coco_recaptioned":
            gen = iter_coco_recaptioned_candidates(dcfg, soi)
        elif dtype == "cosyn_400k":
            gen = iter_cosyn_400k(dcfg, soi)
            # cosyn handled by its own quotas; ignore num_samples if present
            want_written = None
        elif dtype == "hf_text":
            gen = iter_hf_text_candidates(dcfg)
        elif dtype == "stack_edu":
            gen = iter_stack_edu_candidates(dcfg)
            # Per-language quotas handled internally; ignore num_samples
            want_written = None
        elif dtype == "finemath":
            gen = iter_finemath_candidates(dcfg)
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        written_this_dataset = 0
        scanned_candidates = 0

        for image_obj_or_path, prompt, target, meta, task, subset, source_id in tqdm(
            gen, desc=dataset_name
        ):
            scanned_candidates += 1

            # Enforce exact written count for datasets that specify num_samples
            if want_written is not None and written_this_dataset >= want_written:
                break

            image_id = None
            image_path = None
            has_image = image_obj_or_path is not None

            if has_image:
                try:
                    if isinstance(image_obj_or_path, Path):
                        img_ref = image_store.put_file(image_obj_or_path)
                    elif isinstance(image_obj_or_path, Image.Image):
                        img_ref = image_store.put_pil_as_jpeg(image_obj_or_path)
                    else:
                        raise TypeError(
                            f"Unsupported image type: {type(image_obj_or_path)}"
                        )

                    image_id = img_ref.image_id
                    image_path = img_ref.image_path
                except Exception:
                    # IMPORTANT: do NOT increase written_this_dataset.
                    # We simply skip and continue consuming candidates until we write enough.
                    skipped_missing_or_corrupt_images += 1
                    continue

            # Normalize prompt/target tokens and formatting
            prompt = normalize_image_token(prompt, soi)
            target = normalize_image_token(target, soi)
            prompt = ensure_starts_with_soi(prompt, soi, has_image=has_image)

            # CoSyn code prompt deterministic choice
            if prompt and "{CODE_PROMPT_CHOICE}" in prompt:
                choices = meta.get("_code_prompt_choices") or [
                    "Provide the code to create this figure."
                ]
                uid_tmp = make_uid(dataset_name, subset or "", task, str(source_id))
                ok = make_order_key(seed, uid_tmp)
                choice = choices[ok % len(choices)]
                prompt = prompt.replace("{CODE_PROMPT_CHOICE}", choice)

            meta_clean = dict(meta) if isinstance(meta, dict) else {"meta": meta}
            meta_clean.pop("_code_prompt_choices", None)

            uid = make_uid(dataset_name, subset or "", task, str(source_id))
            order_key = make_order_key(seed, uid)
            shard_id = shard_for(order_key, num_shards)

            row = {
                "uid": uid,
                "dataset": dataset_name,
                "subset": subset or "",
                "task": task,
                "shard_id": int(shard_id),
                "order_key": int(order_key),
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,  # None for text-only
                "target": target,
                "meta_json": json.dumps(meta_clean, ensure_ascii=False),
            }

            buffer.append(row)
            total_rows_emitted += 1
            written_this_dataset += 1

            if len(buffer) >= buffer_size:
                flush()

        if want_written is not None and written_this_dataset < want_written:
            warn(
                f"{dataset_name}: requested {want_written} written samples, got {written_this_dataset} "
                f"(scanned_candidates={scanned_candidates}). Dataset exhausted / too many failures."
            )

    flush()

    build_info = {
        "seed": seed,
        "num_shards": num_shards,
        "buffer_size": buffer_size,
        "compression": compression,
        "total_rows_emitted": total_rows_emitted,
        "total_rows_written": total_rows_written,
        "skipped_missing_or_corrupt_images": skipped_missing_or_corrupt_images,
    }
    (out_dir / "manifest" / "build_info.json").write_text(
        json.dumps(build_info, indent=2), encoding="utf-8"
    )

    info("Done.")
    print(json.dumps(build_info, indent=2))


if __name__ == "__main__":
    main()
