from datetime import datetime
import json
import os
import time
import torch
import yaml
from typing import Any, Dict, Optional, Tuple
from common_helpers import pick_best_attn_impl, parse_dtype
from model_implementations import load_model
from transformers.utils import logging as hf_logging
from transformers import TrainingArguments

logger = hf_logging.get_logger(__name__)

# =============================================================================
# Distributed helpers
# =============================================================================


def _get_rank() -> int:
    """
    Return global rank.
    Works both before and after torch.distributed initialization
    because DeepSpeed sets RANK in the environment before launching subprocesses.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_distributed_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _barrier():
    """
    Call a distributed barrier if initialized, otherwise no-op.
    Safe to call at any point in the program.
    """
    if _is_distributed_initialized():
        torch.distributed.barrier()


def _get_synchronized_timestamp(base_dir: str, timeout_seconds: int = 300) -> str:
    """
    Get a run timestamp that is identical on all ranks,
    even when called BEFORE torch.distributed is initialized.

    Strategy (filesystem-based, works on any shared storage):
      - Rank 0 generates the timestamp and atomically writes it to a sentinel
        file inside base_dir.
      - All other ranks poll for that file and read the timestamp from it.
      - The sentinel file is intentionally left on disk — it will be
        overwritten by the next run. Deleting it after writing is NOT safe
        because other ranks may not have read it yet (they may still be busy
        loading models/datasets and arrive here much later).

    Parameters
    ----------
    base_dir : str
        A directory accessible from all ranks (shared filesystem).
        Created by rank 0 if it does not exist yet.
    timeout_seconds : int
        How long non-rank-0 processes wait for rank 0 to write the file.

    Returns
    -------
    str
        Timestamp string of the form ``YYYYMMDD_HHMMSS``.
    """
    rank = _get_rank()
    sentinel = os.path.join(base_dir, ".run_timestamp_sync")

    if rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Write atomically so other ranks never read a partial file.
        tmp = sentinel + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(ts)
        os.replace(tmp, sentinel)
        logger.info(f"[dist] Rank 0 wrote run timestamp: {ts}")
        return ts
    else:
        # Poll until rank 0 writes the file.
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if os.path.exists(sentinel):
                try:
                    with open(sentinel, "r", encoding="utf-8") as f:
                        ts = f.read().strip()
                    if ts:
                        logger.info(f"[dist] Rank {rank} read run timestamp: {ts}")
                        return ts
                except OSError:
                    pass  # file might still be in-flight; retry
            time.sleep(0.2)

        raise RuntimeError(
            f"[dist] Rank {rank} timed out ({timeout_seconds}s) waiting for "
            f"timestamp sentinel at '{sentinel}'.\n"
            f"  Rank 0 should have written this file. Possible causes:\n"
            f"  1. Rank 0 crashed before reaching setup_training_output_dir_and_config.\n"
            f"  2. The path is not on a shared filesystem (check NFS/GPFS mount).\n"
            f"  3. Rank 0 is still busy loading models/datasets — increase timeout_seconds."
        )


# =============================================================================
# Training Setup / Utils
# =============================================================================


def setup_training_output_dir_and_config(cfg: Dict, section: str = "training") -> str:
    """
    1. Gets base output dir from the given config section.
    2. Creates a timestamped version of it. The timestamp is generated on
       rank 0 and distributed to all ranks via a shared filesystem sentinel
       file — this works even before torch.distributed is initialized.
    3. Saves the full resolved config to that directory (rank 0 only).
    4. Returns the path to the final timestamped directory.

    Parameters
    ----------
    cfg : Dict
        Full configuration dictionary.
    section : str
        Name of the section to read ``output_dir`` and optional
        ``timestamp_override`` from.
    """
    tr_cfg = cfg.get(section, {})
    base_output_dir = tr_cfg.get("output_dir")
    if base_output_dir is None:
        raise ValueError(f"Config section '{section}' must define an 'output_dir' key.")

    timestamp_override = tr_cfg.get("timestamp_override")
    if timestamp_override:
        # Even for an override we synchronize via the same mechanism so that
        # a misconfigured multi-rank job (different configs per rank) is caught
        # early.
        ts = _get_synchronized_timestamp(
            base_dir=os.path.dirname(os.path.abspath(base_output_dir)) or ".",
        )
        # Rank 0 set the file content to the override; all ranks use it.
        # (Simpler: just use the override directly — all ranks share the same
        #  config file, so override is always identical.)
        timestamp = str(timestamp_override)
    else:
        timestamp = _get_synchronized_timestamp(
            base_dir=os.path.dirname(os.path.abspath(base_output_dir)) or ".",
        )

    final_output_dir = f"{base_output_dir}_{timestamp}"

    # All ranks create the directory.
    os.makedirs(final_output_dir, exist_ok=True)

    # Only rank 0 writes the config file.
    if _get_rank() == 0:
        config_save_path = os.path.join(final_output_dir, "training_config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info(
            f"Saved full training config to {config_save_path} "
            f"(section='{section}', base_output_dir='{base_output_dir}')"
        )

    # Barrier: ensure all ranks see the directory and config before proceeding.
    _barrier()

    return final_output_dir


def start_timer(output_dir: str):
    """Saves the start time to a metadata file. Rank 0 only."""
    if _get_rank() != 0:
        return
    metadata = {"start_time": datetime.now().isoformat()}
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def stop_timer(output_dir: str):
    """Updates the metadata file with end time and duration. Rank 0 only."""
    if _get_rank() != 0:
        return
    path = os.path.join(output_dir, "run_metadata.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    start_time = datetime.fromisoformat(metadata["start_time"])
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    metadata["end_time"] = end_time.isoformat()
    metadata["duration_seconds"] = duration_seconds
    # Human-readable duration
    days, remainder = divmod(duration_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    metadata["duration_human"] = (
        f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_training_arguments(kwargs: Dict) -> TrainingArguments:
    """
    Create TrainingArguments robustly (compatible with slight API differences).
    Handles eval_strategy/evaluation_strategy naming across versions.
    """
    allowed = set(getattr(TrainingArguments, "__dataclass_fields__", {}).keys())

    eval_enabled = kwargs.pop("_eval_enabled", False)
    eval_steps = kwargs.pop("_eval_steps", None)
    if eval_enabled:
        if "eval_strategy" in allowed:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in allowed:
            kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in allowed and eval_steps is not None:
            kwargs["eval_steps"] = eval_steps
    else:
        if "eval_strategy" in allowed:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in allowed:
            kwargs["evaluation_strategy"] = "no"

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


# =============================================================================
# Model
# =============================================================================


def weights_tied(model) -> bool:
    try:
        inp = model.get_input_embeddings()
        outp = model.get_output_embeddings()
        return inp.weight.data_ptr() == outp.weight.data_ptr()
    except Exception:
        return False


def param_part(name: str) -> str:
    n = name.lower()
    if "vision_tower" in n:
        return "vision_encoder"
    if "multi_modal_projector" in n:
        return "projection"
    if ("language_model" in n) or (".lm_head" in n) or n.startswith("lm_head"):
        return "text_model"
    return "text_model"


# =============================================================================
# Teacher Model Loading (Knowledge Distillation)
# =============================================================================


def _infer_teacher_full_heads(teacher_model) -> int | None:
    """Extract the number of attention heads from a teacher model's config."""
    # Try text_config first (multimodal models nest it there)
    for config_source in [
        getattr(teacher_model.config, "text_config", None),
        teacher_model.config,
    ]:
        try:
            if config_source and hasattr(config_source, "num_attention_heads"):
                return int(config_source.num_attention_heads)
        except Exception:
            continue
    return None


def load_teacher_model(
    teacher_cfg: Dict,
    device: str,
    multimodal: bool | None = None,
    tokenizer=None,  # REQUIRED for vocab resize parity (train_llm)
) -> Tuple[Any, int | None]:
    """
    Load and freeze the teacher model for knowledge distillation.
    Matches legacy behavior while keeping new features.
    Returns (teacher_model, teacher_full_heads).
    """
    model_path = teacher_cfg.get("model_name_or_path")
    if not model_path:
        raise ValueError(
            "train.teacher.model_name_or_path must be set when teacher.enabled=true"
        )

    impl = (teacher_cfg.get("implementation") or "auto_model").lower()
    attn_impl = teacher_cfg.get("attn_implementation", "auto")
    dtype = parse_dtype(teacher_cfg.get("dtype", "auto"), device=device)
    trust_remote = bool(teacher_cfg.get("trust_remote_code", True))

    # Multi-GPU safety: disable per-rank device_map
    device_map = teacher_cfg.get("device_map")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and device_map:
        logger.warning(
            f"[KD] Ignoring teacher.device_map={device_map} in multi-GPU "
            f"(world_size={world_size})"
        )
        device_map = None

    logger.info(
        f"[KD] Loading teacher: {model_path} "
        f"(impl={impl}, dtype={dtype}, attn={attn_impl})"
    )

    # ---- Load teacher (NO fallback) ----
    teacher_model = load_model(
        model_path,
        implementation=impl,
        dtype=dtype,
        attn_implementation=attn_impl,
        device=device,
        multimodal=multimodal,
        device_map=device_map,
        trust_remote_code=trust_remote,
    )

    # Disable cache for KD
    try:
        teacher_model.config.use_cache = False
    except Exception:
        pass

    # Explicitly resolve best attention implementation
    try:
        resolved_attn = pick_best_attn_impl(attn_impl, device=device)
        teacher_model.set_attn_implementation(resolved_attn)
    except Exception:
        pass

    # Freeze + eval
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # ---- Token embedding / LM head resize ----
    if tokenizer is not None:
        try:
            teacher_vocab_now = teacher_model.get_output_embeddings().weight.size(0)
            target_vocab = len(tokenizer)
            if teacher_vocab_now != target_vocab:
                logger.warning(
                    f"[KD] Teacher vocab mismatch "
                    f"(model={teacher_vocab_now}, tok={target_vocab}). "
                    f"Resizing teacher embeddings/head to {target_vocab} "
                    f"(pad_to_multiple_of=64)."
                )
                teacher_model.resize_token_embeddings(
                    target_vocab, pad_to_multiple_of=64
                )
        except Exception as e:
            logger.warning(f"[KD] Teacher vocab resize failed: {e}")

    try:
        if not weights_tied(teacher_model):
            teacher_model.tie_weights()
    except Exception:
        pass
    assert weights_tied(
        teacher_model
    ), "Weight tying failed: input embeddings and lm_head must be tied (teacher)."

    teacher_full_heads = _infer_teacher_full_heads(teacher_model)
    logger.info(f"[KD] Teacher loaded. full_heads={teacher_full_heads}")

    # ---- Optional torch.compile ----
    if bool(teacher_cfg.get("compile", False)):
        try:
            teacher_model = torch.compile(
                teacher_model, mode="max-autotune", dynamic=True
            )
            logger.info("[KD] Teacher compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"[KD] torch.compile failed for teacher: {e}")

    return teacher_model, teacher_full_heads
