# train_unified/combined_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import os
import random

import torch
from torch.utils.data import IterableDataset

from PIL import Image, ImageFile, UnidentifiedImageError

# Allow PIL to load truncated JPEG/PNG files instead of throwing
ImageFile.LOAD_TRUNCATED_IMAGES = True

START_IMAGE_TOKEN = "<start_of_image>"


# ---------------------------
# Text sanitization
# ---------------------------


def _sanitize_mm_text(text: str) -> str:
    """
    Ensure that multimodal text contains exactly ONE <start_of_image> token.
    Some processors/models require a 1:1 mapping between image placeholders and images.

    Strategy:
      - Remove all occurrences of the token
      - Prepend exactly one token
      - Keep the remaining text (stripped) after it
    """
    s = (text or "").strip()
    if not s:
        return START_IMAGE_TOKEN
    # remove all occurrences then prepend exactly one
    s_wo = s.replace(START_IMAGE_TOKEN, "").strip()
    if not s_wo:
        return START_IMAGE_TOKEN
    return f"{START_IMAGE_TOKEN}\n{s_wo}"


# ---------------------------
# Distributed helpers
# ---------------------------


def _dist_rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    # env fallback (useful before init)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _broadcast_int_from_rank0(x: int) -> int:
    """
    Broadcast an integer from rank0 to all ranks using torch.distributed.
    Uses CUDA tensor for NCCL.
    If not distributed, returns x unchanged.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return int(x)

    backend = None
    try:
        backend = torch.distributed.get_backend()
    except Exception:
        backend = None

    if backend == "nccl" and torch.cuda.is_available():
        dev = torch.device(f"cuda:{_local_rank()}")
    else:
        dev = torch.device("cpu")

    t = torch.tensor([int(x)], dtype=torch.long, device=dev)
    torch.distributed.broadcast(t, src=0)
    return int(t.item())


# ---------------------------
# Text / masking helpers
# ---------------------------


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _is_empty_text(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x.strip() == "")


def _effective_span_from_attention_mask(mask_1d: torch.Tensor) -> Tuple[int, int]:
    """
    Works for left- or right-padding:
      returns (start_index, effective_len) where mask==1.
    """
    if mask_1d.ndim != 1:
        mask_1d = mask_1d.view(-1)
    nz = (mask_1d != 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return 0, 0
    start = int(nz.min().item())
    eff_len = int(mask_1d.to(torch.long).sum().item())
    return start, eff_len


def _effective_span_from_pad_id(ids_1d: torch.Tensor, pad_id: int) -> Tuple[int, int]:
    """
    Fallback if attention_mask missing. Handles left or right padding by scanning non-pad.
    """
    if ids_1d.ndim != 1:
        ids_1d = ids_1d.view(-1)
    nz = (ids_1d != int(pad_id)).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return 0, 0
    start = int(nz.min().item())
    eff_len = int(nz.numel())
    return start, eff_len


def _common_prefix_len(a: torch.Tensor, b: torch.Tensor, max_len: int) -> int:
    """
    Length of common prefix between 1D tensors a and b up to max_len.
    """
    if max_len <= 0:
        return 0
    aa = a[:max_len]
    bb = b[:max_len]
    eq = aa == bb
    # find first mismatch
    if bool(eq.all().item()):
        return int(max_len)
    # argmin of eq==False
    mismatch = (~eq).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        return int(max_len)
    return int(mismatch.min().item())


def _truncate_for_log(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


class BadSampleError(RuntimeError):
    """
    Raised when a single dataset sample is unusable (corrupt image, decode failure, etc.)
    The mixer should catch this and resample WITHOUT changing the synchronized modality decision.
    """

    pass


# ---------------------------
# Rank-strided iterable view for HF map datasets
# ---------------------------


class RankStridedHFMapIterable(IterableDataset):
    """
    Iterable over a *map-style* HF dataset (datasets.Dataset), yielding
    rank-strided samples:
      rank r yields indices: r, r+world, r+2*world, ...
    Ranks are disjoint within a single pass; with cycle=True the index wraps
    around, so samples may repeat across ranks after the first pass.
    Optional max_samples caps the number of yielded examples per rank.
    If cycle=True (default), wraps around indefinitely; if False, stops at end.
    """

    def __init__(
        self,
        hf_dataset: Any,
        *,
        shard_by_rank: bool = True,
        max_samples: Optional[int] = None,
        cycle: bool = True,
    ):
        super().__init__()
        self.ds = hf_dataset
        self.shard_by_rank = bool(shard_by_rank)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.cycle = bool(cycle)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self.ds
        n = len(ds)
        if n <= 0:
            return

        rank, world = _dist_rank_world()
        start = rank if self.shard_by_rank else 0
        step = world if self.shard_by_rank else 1

        yielded = 0
        i = start

        while True:
            if self.max_samples is not None and yielded >= self.max_samples:
                break
            # non-cycling mode: stop when we've passed the end
            if not self.cycle and i >= n:
                break

            ex = ds[int(i % n) if self.cycle else int(i)]
            yielded += 1
            yield ex

            i += step

            if self.cycle and i >= n:
                i = i % n


# ---------------------------
# Combined collator (text + MM)
# ---------------------------


@dataclass
class CombinedDebugConfig:
    enabled: bool = False
    rank0_only: bool = True
    max_steps: int = 2
    every_n_steps: int = 0
    max_text_chars: int = 400


class CombinedDataCollator:
    """
    Collator for the combined Parquet schema.

    Input row schema (keys):
      uid, dataset, subset, task, image_id, image_path, prompt, target, meta

    Output batch dict must contain at least:
      input_ids, attention_mask, labels
    and optionally:
      pixel_values, token_type_ids
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        processor: Any,
        max_length: int,
        pad_to_multiple_of: Optional[int] = 64,
        images_root: str,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = int(max_length)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.images_root = images_root

        # prefer processor.tokenizer if present
        try:
            ptok = getattr(self.processor, "tokenizer", None)
            if ptok is not None:
                self.proc_tokenizer = ptok
            else:
                self.proc_tokenizer = tokenizer
        except Exception:
            self.proc_tokenizer = tokenizer

        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_pil_rgb(
        self, path: str, *, uid: Optional[str] = None, retries: int = 2
    ) -> Image.Image:
        """
        Robust image loader:
          - retries a couple of times
          - tolerates truncated images (ImageFile.LOAD_TRUNCATED_IMAGES=True)
          - raises BadSampleError on persistent failures so the batch mixer can resample
            without desynchronizing distributed training.
        """
        import time

        last_err: Optional[BaseException] = None
        for attempt in range(int(retries) + 1):
            try:
                with open(path, "rb") as f:
                    img = Image.open(f)
                    img.load()  # force decode now
                return img.convert("RGB")
            except (OSError, UnidentifiedImageError, ValueError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise BadSampleError(
                    f"Failed to decode image (uid={uid}) at '{path}': {type(e).__name__}: {e}"
                ) from e

        # unreachable
        raise BadSampleError(
            f"Failed to decode image at '{path}' (uid={uid}); last_err={last_err}"
        )

    def _stack(self, encoded_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not encoded_samples:
            raise ValueError("No encoded samples to stack.")
        keys = encoded_samples[0].keys()
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [e[k] for e in encoded_samples]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.cat(vals, dim=0)
            else:
                out[k] = vals
        return out

    def _resolve_image_path(self, rel: str) -> str:
        rel = (rel or "").strip()
        if not rel:
            return ""

        # Absolute path in dataset (allowed)
        if os.path.isabs(rel):
            return rel

        root = os.path.abspath(str(self.images_root))

        # If user passed .../images as root and rel starts with images/, strip it once.
        if os.path.basename(root.rstrip("/")) == "images":
            if rel.startswith("images/") or rel.startswith("images\\"):
                rel = rel.split("/", 1)[1] if "/" in rel else rel.split("\\", 1)[1]

        return os.path.join(root, rel)

    def _resolve_images_dir(self) -> str:
        """
        Your image_path is relative to .../images/.
        We accept images_root being either:
          - a directory that already ends with 'images'
          - a directory containing an 'images' subdir
        """
        root = self.images_root
        if os.path.basename(root.rstrip("/")) == "images":
            return root
        cand = os.path.join(root, "images")
        if os.path.isdir(cand):
            return cand
        # fall back to root itself (root is neither an 'images' dir nor a parent of one)
        return root

    def _row_is_mm(self, row: Dict[str, Any]) -> bool:
        ip = row.get("image_path", None)
        return isinstance(ip, str) and ip.strip() != ""

    def _build_texts(self, row: Dict[str, Any]) -> Tuple[str, str]:
        """
        Returns: (full_text, prompt_text_for_mask)
        Rules:
          - if prompt empty: full_text = target, prompt_text_for_mask=""
          - else: full_text = prompt + "\n" + target
        """
        prompt = _safe_str(row.get("prompt", ""))
        target = _safe_str(row.get("target", ""))
        if _is_empty_text(prompt):
            return target, ""
        # prompt is expected to contain <start_of_image> for mm rows (per your schema)
        prompt = prompt.strip()
        if target.strip():
            return f"{prompt}\n{target.strip()}", prompt
        return prompt, prompt

    def _encode_text_only(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Encode without images. Loss-on-target:
          - if prompt provided, mask prompt prefix
          - else loss over all non-pad tokens
        """
        encoded_samples: List[Dict[str, torch.Tensor]] = []

        pad_id = self.tokenizer.pad_token_id
        for row in rows:
            full_text, prompt_for_mask = self._build_texts(row)

            full_enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            prompt_enc = None
            if prompt_for_mask:
                prompt_enc = self.tokenizer(
                    prompt_for_mask,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )

            input_ids = full_enc["input_ids"]  # [1, L]
            attn_mask = full_enc.get("attention_mask", None)

            L = int(input_ids.size(1))
            if isinstance(attn_mask, torch.Tensor):
                full_start, full_eff_len = _effective_span_from_attention_mask(
                    attn_mask[0]
                )
            elif pad_id is not None:
                full_start, full_eff_len = _effective_span_from_pad_id(
                    input_ids[0], int(pad_id)
                )
            else:
                full_start, full_eff_len = 0, L

            labels = input_ids.clone()

            # mask prompt prefix if any
            if prompt_enc is not None:
                p_ids = prompt_enc["input_ids"]
                p_attn = prompt_enc.get("attention_mask", None)
                if isinstance(p_attn, torch.Tensor):
                    p_start, p_eff_len = _effective_span_from_attention_mask(p_attn[0])
                elif pad_id is not None:
                    p_start, p_eff_len = _effective_span_from_pad_id(
                        p_ids[0], int(pad_id)
                    )
                else:
                    p_start, p_eff_len = 0, L

                full_content = input_ids[0, full_start : full_start + full_eff_len]
                prompt_content = p_ids[0, p_start : p_start + p_eff_len]
                max_cmp = min(int(full_content.numel()), int(prompt_content.numel()))
                prefix_len = _common_prefix_len(prompt_content, full_content, max_cmp)
                if prefix_len > 0:
                    labels[:, full_start : full_start + prefix_len] = -100

            # mask padding
            if isinstance(attn_mask, torch.Tensor):
                labels[attn_mask == 0] = -100
            elif pad_id is not None:
                labels[input_ids == int(pad_id)] = -100

            full_enc["labels"] = labels
            encoded_samples.append(full_enc)

        return self._stack(encoded_samples)

    def _encode_multimodal(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Encode with images using processor.
        Loss-on-target:
          - mask prompt prefix
          - mask padding
          - mask image tokens via token_type_ids==1 when present

        IMPORTANT:
          - Raises BadSampleError for unusable samples so the mixer can resample
            without changing the synchronized modality decision.
        """
        if self.processor is None:
            raise ValueError("Multimodal encoding requires a non-None processor.")

        pad_id = getattr(self.proc_tokenizer, "pad_token_id", None)
        encoded_samples: List[Dict[str, torch.Tensor]] = []

        for row in rows:
            uid = row.get("uid", None)
            rel = _safe_str(row.get("image_path", "")).strip()
            img_path = self._resolve_image_path(rel)
            if not img_path or not os.path.exists(img_path):
                raise BadSampleError(f"Image not found for row uid={uid}: {img_path}")

            # Robust decode (raises BadSampleError on persistent failures)
            img = self._load_pil_rgb(
                img_path, uid=str(uid) if uid is not None else None, retries=2
            )

            full_text, prompt_for_mask = self._build_texts(row)

            # Ensure exactly one image token placeholder for 1 image.
            full_text = _sanitize_mm_text(full_text)
            if prompt_for_mask:
                prompt_for_mask = _sanitize_mm_text(prompt_for_mask)

            def _proc(images, text):
                return self.processor(
                    images=images,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )

            try:
                full_enc = _proc(img, full_text)
            except ValueError as e:
                # Defensive retry
                if "image tokens" in str(e).lower():
                    full_enc = _proc(img, _sanitize_mm_text(full_text))
                else:
                    raise

            prompt_enc = None
            if prompt_for_mask:
                try:
                    prompt_enc = _proc(img, prompt_for_mask)
                except ValueError as e:
                    if "image tokens" in str(e).lower():
                        prompt_enc = _proc(img, _sanitize_mm_text(prompt_for_mask))
                    else:
                        raise

            input_ids = full_enc["input_ids"]
            attn_mask = full_enc.get("attention_mask", None)
            token_type_ids = full_enc.get("token_type_ids", None)

            L = int(input_ids.size(1))
            if isinstance(attn_mask, torch.Tensor):
                full_start, full_eff_len = _effective_span_from_attention_mask(
                    attn_mask[0]
                )
            elif pad_id is not None:
                full_start, full_eff_len = _effective_span_from_pad_id(
                    input_ids[0], int(pad_id)
                )
            else:
                full_start, full_eff_len = 0, L

            labels = input_ids.clone()

            # mask prompt prefix if any
            if prompt_enc is not None:
                p_ids = prompt_enc["input_ids"]
                p_attn = prompt_enc.get("attention_mask", None)
                if isinstance(p_attn, torch.Tensor):
                    p_start, p_eff_len = _effective_span_from_attention_mask(p_attn[0])
                elif pad_id is not None:
                    p_start, p_eff_len = _effective_span_from_pad_id(
                        p_ids[0], int(pad_id)
                    )
                else:
                    p_start, p_eff_len = 0, L

                full_content = input_ids[0, full_start : full_start + full_eff_len]
                prompt_content = p_ids[0, p_start : p_start + p_eff_len]
                max_cmp = min(int(full_content.numel()), int(prompt_content.numel()))
                prefix_len = _common_prefix_len(prompt_content, full_content, max_cmp)
                if prefix_len > 0:
                    labels[:, full_start : full_start + prefix_len] = -100

            # mask padding
            if isinstance(attn_mask, torch.Tensor):
                labels[attn_mask == 0] = -100
            elif pad_id is not None:
                labels[input_ids == int(pad_id)] = -100

            # mask image tokens
            if isinstance(token_type_ids, torch.Tensor):
                labels[token_type_ids == 1] = -100

            full_enc["labels"] = labels
            encoded_samples.append(full_enc)

        return self._stack(encoded_samples)

    def __call__(
        self, rows: List[Dict[str, Any]], *, modality: str
    ) -> Dict[str, torch.Tensor]:
        modality = (modality or "").strip().lower()
        if modality not in ("text", "mm"):
            raise ValueError(f"modality must be 'text' or 'mm', got {modality}")

        if modality == "text":
            return self._encode_text_only(rows)
        return self._encode_multimodal(rows)


# ---------------------------
# Synchronized modality mixer (yields already-collated batches)
# ---------------------------


class SynchronizedModalityBatchMixerIterable(IterableDataset):
    """
    Yields already-collated batches. Each batch is *homogeneous* modality (text OR mm),
    chosen per batch and synchronized across ranks.

    Epoch-aware:
      - __len__ defines batches-per-epoch so HF Trainer advances epoch > 1.0
      - __iter__ reshuffles text_ds / mm_ds deterministically per epoch
      - set_epoch() is called by a trainer callback at each epoch boundary
    """

    def __init__(
        self,
        *,
        text_dataset: Any,  # HF map dataset
        mm_dataset: Any,  # HF map dataset
        collator: CombinedDataCollator,
        batch_size: int,
        seed: int,
        text_weight: float = 0.5,
        mm_weight: float = 0.5,
        sync_across_ranks: bool = True,
        shard_by_rank: bool = True,
        max_samples_per_rank: Optional[int] = None,
        debug: Optional[CombinedDebugConfig] = None,
        tokenizer_for_log: Optional[Any] = None,
    ):
        super().__init__()
        self.text_ds = text_dataset
        self.mm_ds = mm_dataset
        self.collator = collator
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.text_weight = float(text_weight)
        self.mm_weight = float(mm_weight)
        self.sync_across_ranks = bool(sync_across_ranks)
        self.shard_by_rank = bool(shard_by_rank)
        self.max_samples_per_rank = (
            int(max_samples_per_rank) if max_samples_per_rank is not None else None
        )
        self.debug = debug or CombinedDebugConfig()
        self.tok_log = tokenizer_for_log or getattr(collator, "tokenizer", None)

        # Epoch state — updated by SetTrainDatasetEpochCallback before each __iter__
        self.epoch: int = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if (self.text_weight + self.mm_weight) <= 0:
            raise ValueError("text_weight + mm_weight must be > 0")

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Called by trainer callback before each epoch's __iter__."""
        self.epoch = int(epoch)

    # ------------------------------------------------------------------
    # Length (drives HF Trainer epoch accounting)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Number of *batches* this rank yields per epoch.
        'One epoch' = enough batches to consume (text + mm) samples once globally.

        HF Trainer uses this to compute:
          num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
          num_train_epochs           = ceil(max_steps / num_update_steps_per_epoch)
        """
        _, world = _dist_rank_world()
        total_samples = int(len(self.text_ds)) + int(len(self.mm_ds))
        global_batch = int(self.batch_size) * max(1, int(world))
        if global_batch <= 0 or total_samples <= 0:
            return 1
        return max(1, total_samples // global_batch)

    # ------------------------------------------------------------------
    # Modality picking (unchanged logic, kept for completeness)
    # ------------------------------------------------------------------

    def _pick_modality(self, rng: random.Random) -> str:
        # Weighted draw. The caller invokes this on rank0 only and syncs the
        # result to other ranks via _sync_modality (if sync is enabled).
        x = rng.random() * (self.text_weight + self.mm_weight)
        return "text" if x < self.text_weight else "mm"

    def _sync_modality(self, modality: str) -> str:
        if not self.sync_across_ranks:
            return modality
        rank, _ = _dist_rank_world()
        if rank == 0:
            code = 0 if modality == "text" else 1
        else:
            code = 0
        code = _broadcast_int_from_rank0(code)
        return "text" if code == 0 else "mm"

    # ------------------------------------------------------------------
    # Debug logging helpers
    # ------------------------------------------------------------------

    def _should_log(self, step: int) -> bool:
        if not self.debug.enabled:
            return False
        if self.debug.every_n_steps and self.debug.every_n_steps > 0:
            if step % int(self.debug.every_n_steps) == 0:
                return True
        if step < int(self.debug.max_steps):
            return True
        return False

    def _log_batch(
        self,
        *,
        step: int,
        modality: str,
        raw_rows: List[Dict[str, Any]],
        batch: Dict[str, torch.Tensor],
    ):
        # rank gating
        rank, _ = _dist_rank_world()
        if self.debug.rank0_only and rank != 0:
            return

        if not raw_rows:
            return

        max_chars = int(self.debug.max_text_chars)
        row0 = raw_rows[0]
        uid = row0.get("uid", None)

        # -------- raw sample --------
        prompt = _truncate_for_log(_safe_str(row0.get("prompt", "")), max_chars)
        target = _truncate_for_log(_safe_str(row0.get("target", "")), max_chars)
        image_path = row0.get("image_path", None)

        print(
            f"\n[DATA-DEBUG] step={step} rank={rank} modality={modality} uid={uid}\n"
            f"  raw.dataset={row0.get('dataset', None)} subset={row0.get('subset', None)} task={row0.get('task', None)}\n"
            f"  raw.image_path={image_path}\n"
            f"  raw.prompt={repr(prompt)}\n"
            f"  raw.target={repr(target)}",
            flush=True,
        )

        # -------- padded sample --------
        try:
            input_ids = batch.get("input_ids", None)
            attn = batch.get("attention_mask", None)
            labels = batch.get("labels", None)
            if isinstance(input_ids, torch.Tensor):
                print(
                    f"  padded.input_ids.shape={tuple(input_ids.shape)} dtype={input_ids.dtype}",
                    flush=True,
                )
            if isinstance(attn, torch.Tensor):
                print(f"  padded.attention_mask.shape={tuple(attn.shape)}", flush=True)
            if isinstance(labels, torch.Tensor):
                print(f"  padded.labels.shape={tuple(labels.shape)}", flush=True)

            if self.tok_log is not None and isinstance(input_ids, torch.Tensor):
                ids0 = input_ids[0].detach().cpu().tolist()
                # decode full (includes pads and special tokens)
                text_full = self.tok_log.decode(ids0, skip_special_tokens=False)
                text_full = _truncate_for_log(text_full, max_chars)
                print(f"  padded.decode(input_ids[0])={repr(text_full)}", flush=True)

            # -------- target on which loss is computed --------
            if (
                self.tok_log is not None
                and isinstance(labels, torch.Tensor)
                and isinstance(input_ids, torch.Tensor)
            ):
                mask = (labels[0] != -100).detach().cpu()
                ids_target = input_ids[0].detach().cpu()[mask].tolist()
                text_target = self.tok_log.decode(ids_target, skip_special_tokens=False)
                text_target = _truncate_for_log(text_target, max_chars)
                print(
                    f"  loss_target.decode(labels!=-100)={repr(text_target)}",
                    flush=True,
                )

        except Exception as e:
            print(
                f"  [DATA-DEBUG][WARN] failed to log padded/target decode: {type(e).__name__}: {e}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Main iterator — reshuffles per epoch, yields finite batches
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rank, world = _dist_rank_world()
        epoch = int(self.epoch)

        # ---- Deterministic per-epoch reshuffle (same across all ranks) ----
        try:
            text_ds_shuffled = self.text_ds.shuffle(
                seed=int(self.seed + 1000 * epoch + 17)
            )
        except Exception:
            text_ds_shuffled = self.text_ds

        try:
            mm_ds_shuffled = self.mm_ds.shuffle(seed=int(self.seed + 1000 * epoch + 29))
        except Exception:
            mm_ds_shuffled = self.mm_ds

        # Per-epoch modality RNG (rank0 chooses; deterministic per epoch)
        rng = random.Random(int(self.seed + 1337 + 9999 * epoch))

        # Rank-strided iterables (cycle=True so retries always have data)
        text_it = iter(
            RankStridedHFMapIterable(
                text_ds_shuffled,
                shard_by_rank=self.shard_by_rank,
                max_samples=self.max_samples_per_rank,
                cycle=True,
            )
        )
        mm_it = iter(
            RankStridedHFMapIterable(
                mm_ds_shuffled,
                shard_by_rank=self.shard_by_rank,
                max_samples=self.max_samples_per_rank,
                cycle=True,
            )
        )

        steps_per_epoch = len(self)
        bad_mm_samples = 0
        bad_text_samples = 0

        for step_in_epoch in range(steps_per_epoch):
            # Global step for logging (monotonically increasing across epochs)
            global_like_step = epoch * steps_per_epoch + step_in_epoch

            # 1) Pick modality ONCE per batch (rank0 picks, then broadcast)
            modality = self._pick_modality(rng) if rank == 0 else "text"
            modality = self._sync_modality(modality)
            it = mm_it if modality == "mm" else text_it

            # 2) For this fixed modality, keep trying until we yield a valid batch
            #    (do NOT re-broadcast modality inside retries)
            while True:
                rows: List[Dict[str, Any]] = []

                # Build batch (skip obviously invalid rows)
                while len(rows) < self.batch_size:
                    try:
                        row = next(it)
                    except StopIteration:
                        # Only possible if max_samples_per_rank exhausted
                        return

                    if not isinstance(row, dict):
                        continue

                    if modality == "mm":
                        ip = row.get("image_path", None)
                        if not (isinstance(ip, str) and ip.strip()):
                            continue
                        # quick existence check only (decode happens in collator)
                        try:
                            full_path = self.collator._resolve_image_path(ip.strip())
                            if not full_path or not os.path.exists(full_path):
                                continue
                        except Exception:
                            continue

                    rows.append(row)

                # Collate; on BadSampleError, resample without changing modality
                try:
                    batch = self.collator(rows, modality=modality)
                except BadSampleError as e:
                    if modality == "mm":
                        bad_mm_samples += 1
                        if (bad_mm_samples <= 10) or (bad_mm_samples % 200 == 0):
                            print(
                                f"[BATCH][SKIP] rank={rank}/{world} epoch={epoch} "
                                f"step={global_like_step} modality=mm "
                                f"bad_mm_samples={bad_mm_samples} err={e}",
                                flush=True,
                            )
                    else:
                        bad_text_samples += 1
                        if (bad_text_samples <= 10) or (bad_text_samples % 200 == 0):
                            print(
                                f"[BATCH][SKIP] rank={rank}/{world} epoch={epoch} "
                                f"step={global_like_step} modality=text "
                                f"bad_text_samples={bad_text_samples} err={e}",
                                flush=True,
                            )
                    # retry same modality decision
                    continue
                except Exception as e:
                    # Unknown failure: warn and retry, but keep modality decision fixed.
                    print(
                        f"[BATCH][WARN] rank={rank}/{world} epoch={epoch} "
                        f"step={global_like_step} modality={modality} "
                        f"collation_failed={type(e).__name__}: {e}",
                        flush=True,
                    )
                    continue

                # Debug logging
                if self._should_log(global_like_step):
                    self._log_batch(
                        step=global_like_step,
                        modality=modality,
                        raw_rows=rows,
                        batch=batch,
                    )

                yield batch
                break  # next step_in_epoch => new modality decision # next batch => new synchronized modality decision
