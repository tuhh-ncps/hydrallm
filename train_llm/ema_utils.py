# train_llm/ema_utils.py
from transformers import TrainerCallback
from typing import Dict, Optional, Tuple
import os
import re
import json
import torch
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

_EMA_LOG_OVERRIDE = None
_EMA_LOG_LEVEL_OVERRIDE = None


def configure_ema_logging(logging=None, log_level=None):
    global _EMA_LOG_OVERRIDE, _EMA_LOG_LEVEL_OVERRIDE
    if logging is not None:
        _EMA_LOG_OVERRIDE = bool(logging)
    if log_level is not None:
        _EMA_LOG_LEVEL_OVERRIDE = str(log_level).strip().upper()


# ---------------------------
# Rank-aware print logging helpers
# ---------------------------


def _dist_info() -> Tuple[int, int]:
    """
    Best-effort rank/world_size detection.
    Works for torch.distributed and also for env-based launchers.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
    except Exception:
        pass

    def _to_int(x, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    rank = _to_int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")), 0)
    world = _to_int(os.environ.get("WORLD_SIZE", "1"), 1)
    return rank, world


def _rank_prefix() -> str:
    r, w = _dist_info()
    return f"[rank {r}/{w}]"


def _ema_log_enabled() -> bool:
    if _EMA_LOG_OVERRIDE is not None:
        return _EMA_LOG_OVERRIDE
    v = os.environ.get("EMA_LOG", "0")
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _ema_log_level() -> str:
    # DEBUG/INFO/WARNING
    if _EMA_LOG_LEVEL_OVERRIDE is not None:
        return _EMA_LOG_LEVEL_OVERRIDE
    return str(os.environ.get("EMA_LOG_LEVEL", "INFO")).strip().upper()


_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30}


def _should_log(level: str) -> bool:
    if not _ema_log_enabled():
        return False
    lvl = _LEVEL_ORDER.get(level.upper(), 20)
    cur = _LEVEL_ORDER.get(_ema_log_level(), 20)
    return lvl >= cur


def _log(level: str, msg: str):
    """
    Print-based logging (rank-prefixed), so it is visible even if HF logger verbosity is WARNING.
    """
    if not _should_log(level):
        return
    print(f"{_rank_prefix()} [EMA:{level.upper()}] {msg}", flush=True)


def _log_info(msg: str):
    _log("INFO", msg)


def _log_warning(msg: str):
    _log("WARNING", msg)


def _log_debug(msg: str):
    _log("DEBUG", msg)


# ---------------------------
# Try to import EMAModel from accelerate; provide a device-aware fallback otherwise.
# ---------------------------

try:
    from accelerate import EMAModel  # Preferred path

    _log_info(
        "Using accelerate.EMAModel (import path: 'accelerate.EMAModel')."
    )
except Exception:
    try:
        from accelerate.utils import EMAModel  # Older path

        _log_info(
            "Using accelerate.EMAModel (import path: 'accelerate.utils.EMAModel')."
        )
    except Exception:
        _log_warning(
            "accelerate.EMAModel not available; using local fallback EMAModel."
        )

        class EMAModel:
            """
            Device-aware fallback EMA for when accelerate.EMAModel is unavailable.
            - Keeps a shadow copy per parameter on that parameter's current device/dtype.
            - Safe for DDP/DeepSpeed Stage 1/2. For ZeRO-3 with offload, prefer accelerate.EMAModel.
            """

            def __init__(self, parameters, decay: float = 0.9999):
                self.decay = float(decay)
                self.shadow: Dict[torch.nn.Parameter, torch.Tensor] = {}
                self.collected: Dict[torch.nn.Parameter, torch.Tensor] = {}
                seen = set()
                self.params = []
                for p in parameters:
                    if not isinstance(p, torch.nn.Parameter):
                        continue
                    if not p.requires_grad:
                        continue
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    self.params.append(p)
                    self.shadow[p] = (
                        p.detach().clone().to(device=p.device, dtype=p.dtype)
                    )

                try:
                    devices = {str(p.device) for p in self.params}
                    dtypes = {str(p.dtype) for p in self.params}
                    _log_info(
                        f"Initialized fallback EMA: decay={self.decay} "
                        f"| tracked_params={len(self.params)} "
                        f"| devices={sorted(devices)[:8]}{'...' if len(devices) > 8 else ''} "
                        f"| dtypes={sorted(dtypes)[:8]}{'...' if len(dtypes) > 8 else ''}"
                    )
                except Exception:
                    _log_info(
                        f"Initialized fallback EMA: decay={self.decay} | tracked_params={len(self.params)}"
                    )

            @torch.no_grad()
            def _ensure_device_match(
                self, p: torch.nn.Parameter, t: torch.Tensor
            ) -> torch.Tensor:
                if t.device != p.device or t.dtype != p.dtype:
                    return t.to(device=p.device, dtype=p.dtype, non_blocking=True)
                return t

            @torch.no_grad()
            def update(self):
                d = self.decay
                for p in self.params:
                    sh = self._ensure_device_match(p, self.shadow[p])
                    sh.mul_(d).add_(p.detach(), alpha=1.0 - d)
                    self.shadow[p] = sh

            @torch.no_grad()
            def store(self):
                self.collected = {
                    p: p.detach().clone().to(device=p.device, dtype=p.dtype)
                    for p in self.params
                }

            @torch.no_grad()
            def copy_to(self):
                for p in self.params:
                    sh = self._ensure_device_match(p, self.shadow[p])
                    p.data.copy_(sh.data)

            @torch.no_grad()
            def restore(self):
                for p in self.params:
                    if p in self.collected:
                        col = self._ensure_device_match(p, self.collected[p])
                        p.data.copy_(col.data)


def _unwrap_model_for_saving(model):
    """
    Unwrap common wrappers (DDP, DeepSpeedEngine, etc.) to get the underlying nn.Module.
    """
    m = model
    depth = 0
    while hasattr(m, "module"):
        try:
            m = m.module
            depth += 1
        except Exception:
            break
    if depth > 0:
        _log_debug(f"Unwrapped model through {depth} '.module' levels for saving.")
    return m


def _ema_shadow_by_param(ema) -> Dict[torch.nn.Parameter, torch.Tensor]:
    """
    Return a mapping {Parameter -> ema_tensor} for supported EMA implementations.
    Supports:
      - this repo's fallback EMAModel: has `ema.shadow: Dict[Parameter, Tensor]`
      - accelerate.EMAModel (most versions): has `shadow_params` + `params`/`parameters`
    Raises if it cannot extract EMA weights without swapping.
    """
    shadow = getattr(ema, "shadow", None)
    if isinstance(shadow, dict) and shadow:
        _log_debug(f"Using fallback EMA shadow dict (entries={len(shadow)}).")
        return shadow

    shadow_params = getattr(ema, "shadow_params", None)
    if shadow_params is None:
        raise RuntimeError(
            "EMA object does not expose a usable shadow map (no `shadow` dict and no `shadow_params`). "
            "Cannot save/restore EMA without swapping weights."
        )

    params = (
        getattr(ema, "params", None)
        or getattr(ema, "parameters", None)
        or getattr(ema, "_parameters", None)
    )
    if params is None:
        raise RuntimeError(
            "accelerate.EMAModel detected (`shadow_params` exists) but no matching parameter list found "
            "(`params`/`parameters`). Cannot save/restore EMA without swapping weights."
        )

    params = list(params)
    shadow_params = list(shadow_params)
    if len(params) != len(shadow_params):
        raise RuntimeError(
            f"EMA params length mismatch: len(params)={len(params)} vs len(shadow_params)={len(shadow_params)}. "
            "Cannot save/restore EMA without swapping weights."
        )

    _log_debug(f"Using accelerate EMAModel shadow list (entries={len(params)}).")
    return {p: s for p, s in zip(params, shadow_params)}


def build_ema_state_dict_no_swap(
    model: torch.nn.Module,
    ema,
    *,
    keep_fp32: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a state_dict where each parameter tensor is replaced by its EMA shadow tensor.
    No swapping of live model weights is performed.
    Notes:
      - Returns CPU tensors (HF save_pretrained ultimately writes CPU tensors anyway).
      - Handles tied weights correctly by using named_parameters(remove_duplicate=False) when available.
    """
    sd = model.state_dict()

    try:
        name_to_param = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        name_to_param = dict(model.named_parameters())

    shadow_by_param = _ema_shadow_by_param(ema)

    replaced = 0
    missing = 0
    for name, tensor in list(sd.items()):
        p = name_to_param.get(name, None)
        if p is None:
            continue
        if p not in shadow_by_param:
            missing += 1
            continue
        ema_t = shadow_by_param[p].detach()
        if not keep_fp32:
            ema_t = ema_t.to(dtype=tensor.dtype)
        sd[name] = ema_t.to(device="cpu")
        replaced += 1

    _log_info(
        f"build_ema_state_dict_no_swap: replaced={replaced} "
        f"| tracked={len(shadow_by_param)} | keep_fp32={keep_fp32} | state_dict_keys={len(sd)} "
        f"| name_to_param={len(name_to_param)} | not_tracked_but_named={missing}"
    )
    return sd


# ---------------------------
# EMA RESUME / RESTORE LOGIC
# ---------------------------


def _parse_checkpoint_step(path: str) -> Optional[int]:
    """
    Extract global step from a Trainer checkpoint folder name like .../checkpoint-1234.
    """
    base = os.path.basename(path.rstrip("/"))
    m = re.match(r"checkpoint-(\d+)$", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _corresponding_ema_checkpoint_dir(
    model_checkpoint_dir: str,
    ema_dirname: str = "ema_model",
) -> str:
    """
    Map:
      .../model/checkpoint-N  ->  .../ema_model/checkpoint-N
    Assumes your layout:
      final_output_dir/
        model/
          checkpoint-N/
        ema_model/
          checkpoint-N/
    """
    model_ckpt_dir = model_checkpoint_dir.rstrip("/")
    model_dir = os.path.dirname(model_ckpt_dir)
    run_dir = os.path.dirname(model_dir)
    ema_root = os.path.join(run_dir, ema_dirname)
    ema_dir = os.path.join(ema_root, os.path.basename(model_ckpt_dir))
    _log_debug(
        f"Corresponding EMA checkpoint dir: model_ckpt={model_ckpt_dir} -> ema_ckpt={ema_dir}"
    )
    return ema_dir


def _load_state_dict_from_pretrained_dir(
    pretrained_dir: str,
) -> Dict[str, torch.Tensor]:
    """
    Load a (non-sharded) HF save_pretrained model file from a directory.
    Supports:
      - model.safetensors
      - pytorch_model.bin
    Returns CPU tensors.
    """
    st_path = os.path.join(pretrained_dir, "model.safetensors")
    bin_path = os.path.join(pretrained_dir, "pytorch_model.bin")

    if os.path.isfile(st_path):
        _log_info(f"Loading EMA state_dict from safetensors: {st_path}")
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError(f"Found {st_path} but safetensors is not available: {e}")
        sd = safe_load_file(st_path, device="cpu")
        return dict(sd)

    if os.path.isfile(bin_path):
        _log_info(f"Loading EMA state_dict from torch bin: {bin_path}")
        sd = torch.load(bin_path, map_location="cpu")
        if (
            isinstance(sd, dict)
            and "state_dict" in sd
            and isinstance(sd["state_dict"], dict)
        ):
            return sd["state_dict"]
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected format in {bin_path}: {type(sd)}")
        return sd

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in: {pretrained_dir}"
    )


@torch.no_grad()
def restore_ema_shadow_from_ema_state_dict(
    *,
    model: torch.nn.Module,
    ema,
    ema_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> Tuple[int, int]:
    """
    Restore EMA shadow tensors IN-PLACE from an EMA model state_dict.
    Returns:
      (num_restored, num_total_tracked)
    """
    base_model = _unwrap_model_for_saving(model)

    try:
        named_params = list(base_model.named_parameters(remove_duplicate=False))
    except TypeError:
        named_params = list(base_model.named_parameters())

    param_to_name: Dict[torch.nn.Parameter, str] = {}
    for n, p in named_params:
        if p not in param_to_name:
            param_to_name[p] = n

    shadow_by_param = _ema_shadow_by_param(ema)
    total = len(shadow_by_param)
    restored = 0
    missing_keys = 0
    unnamed = 0

    _log_info(
        f"Restoring EMA shadow from state_dict: ema_sd_keys={len(ema_state_dict)} "
        f"| tracked={total} | named_params={len(named_params)} | strict={strict}"
    )

    for p, shadow_t in shadow_by_param.items():
        name = param_to_name.get(p, None)
        if name is None:
            unnamed += 1
            if strict:
                raise KeyError(
                    "A tracked EMA parameter has no name in model.named_parameters()."
                )
            continue

        src = ema_state_dict.get(name, None)
        if src is None:
            missing_keys += 1
            if strict:
                raise KeyError(f"EMA state_dict is missing parameter key: {name}")
            continue

        try:
            tgt = shadow_t
            src_cast = src.to(device=tgt.device, dtype=tgt.dtype, non_blocking=True)
            tgt.data.copy_(src_cast)
            restored += 1
        except Exception as e:
            _log_debug(f"In-place shadow copy failed for '{name}': {e}")
            try:
                shadow_by_param[p] = src.to(device=p.device, dtype=p.dtype)
                restored += 1
            except Exception as e2:
                _log_debug(f"Shadow replace failed for '{name}': {e2}")
                if strict:
                    raise

    _log_info(
        f"Restore done: restored={restored}/{total} | missing_keys={missing_keys} | unnamed_tracked={unnamed}"
    )
    return restored, total


def resume_ema_shadow_from_checkpoint(
    *,
    model: torch.nn.Module,
    ema,
    model_checkpoint_dir: str,
    ema_dirname: str = "ema_model",
    strict: bool = False,
) -> bool:
    """
    Given a Trainer model checkpoint dir (e.g. .../model/checkpoint-50),
    load the corresponding EMA checkpoint (e.g. .../ema_model/checkpoint-50)
    and restore the EMA shadow copy.
    Returns True if restored, else False.
    """
    ema_ckpt_dir = _corresponding_ema_checkpoint_dir(
        model_checkpoint_dir=model_checkpoint_dir,
        ema_dirname=ema_dirname,
    )

    if not os.path.isdir(ema_ckpt_dir):
        _log_warning(f"No EMA checkpoint found for resume. Expected: {ema_ckpt_dir}")
        return False

    try:
        ema_sd = _load_state_dict_from_pretrained_dir(ema_ckpt_dir)
    except Exception as e:
        _log_warning(
            f"Failed to load EMA checkpoint state_dict from {ema_ckpt_dir}: {e}"
        )
        return False

    try:
        restored, total = restore_ema_shadow_from_ema_state_dict(
            model=model,
            ema=ema,
            ema_state_dict=ema_sd,
            strict=strict,
        )
        _log_info(
            f"Resumed EMA shadow from {ema_ckpt_dir}: {restored}/{total} tensors restored."
        )
        return restored > 0
    except Exception as e:
        _log_warning(f"Failed restoring EMA shadow from {ema_ckpt_dir}: {e}")
        return False


def _count_params(model: torch.nn.Module):
    total_elems = 0
    train_elems = 0
    total_tensors = 0
    train_tensors = 0
    for p in model.parameters():
        if not isinstance(p, torch.nn.Parameter):
            continue
        n = p.numel()
        total_elems += n
        total_tensors += 1
        if p.requires_grad:
            train_elems += n
            train_tensors += 1
    return {
        "total_tensors": total_tensors,
        "train_tensors": train_tensors,
        "total_elems": total_elems,
        "train_elems": train_elems,
    }


# ---------------------------
# EMA Callback
# ---------------------------


class EMASaveBothNoSwapCallback(TrainerCallback):
    """
    EMA callback that:
      - updates EMA during training
      - saves EMA weights to a separate folder (sibling of args.output_dir), WITHOUT swapping model weights.
    Raw checkpoints remain exactly as HF Trainer saves them under args.output_dir.
    EMA checkpoints are written under ../ema_model:
      - on_save:      ../ema_model/checkpoint-{global_step}
      - on_train_end: ../ema_model/
    """

    def __init__(
        self,
        ema: EMAModel,
        *,
        ema_dirname: str = "ema_model",
        keep_fp32: bool = True,
        safe_serialization: bool = True,
    ):
        self.ema = ema
        self.ema_dirname = str(ema_dirname)
        self.keep_fp32 = bool(keep_fp32)
        self.safe_serialization = bool(safe_serialization)

        # Logging counters (only affects logging frequency)
        self._update_calls = 0

        decay = getattr(self.ema, "decay", None)
        shadow = getattr(self.ema, "shadow", None)
        shadow_params = getattr(self.ema, "shadow_params", None)
        tracked = None
        if isinstance(shadow, dict):
            tracked = len(shadow)
        elif shadow_params is not None:
            try:
                tracked = len(list(shadow_params))
            except Exception:
                tracked = None

        _log_info(
            f"Callback initialized: ema_dirname='{self.ema_dirname}' "
            f"| keep_fp32={self.keep_fp32} | safe_serialization={self.safe_serialization} "
            f"| ema_class={type(self.ema).__name__} | decay={decay} | tracked={tracked} "
            f"| EMA_LOG_LEVEL={_ema_log_level()} | EMA_LOG={os.environ.get('EMA_LOG','0')}"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        # Helpful to confirm callback is actually active.
        model = kwargs.get("model", None)
        if model is not None:
            c = _count_params(model)
            _log_info(
                f"trainable params at on_train_begin: "
                f"train_tensors={c['train_tensors']}/{c['total_tensors']} | "
                f"train_elems={c['train_elems']:,}/{c['total_elems']:,}"
            )
        _log_info(
            f"on_train_begin: global_step={int(getattr(state, 'global_step', -1))} | output_dir={getattr(args, 'output_dir', None)}"
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # NOTE: Behavior unchanged: still updates on every step_end.
        try:
            self._update_calls += 1
            # log very sparsely by default at INFO, more at DEBUG
            if _should_log("DEBUG"):
                _log_debug(
                    f"on_step_end: update_call={self._update_calls} | global_step={int(getattr(state, 'global_step', -1))}"
                )
            else:
                # At INFO level, only log the first few updates + then every 500 updates
                if self._update_calls <= 5 or (self._update_calls % 500) == 0:
                    _log_info(
                        f"on_step_end: update_call={self._update_calls} | global_step={int(getattr(state, 'global_step', -1))}"
                    )
            self.ema.update()
        except Exception as e:
            _log_warning(f"update() failed in on_step_end: {e}")
        return control

    def _ema_root_dir(self, args) -> str:
        parent = os.path.dirname(args.output_dir.rstrip("/"))
        root = os.path.join(parent, self.ema_dirname)
        _log_debug(
            f"EMA root dir resolved: args.output_dir={args.output_dir} -> {root}"
        )
        return root

    def _ema_ckpt_dir(self, args, step: int) -> str:
        d = os.path.join(self._ema_root_dir(args), f"checkpoint-{step}")
        _log_debug(f"EMA checkpoint dir resolved: step={step} -> {d}")
        return d

    def _save_ema_snapshot_no_swap(self, *, save_dir: str, model, tokenizer=None):
        os.makedirs(save_dir, exist_ok=True)
        base_model = _unwrap_model_for_saving(model)

        _log_info(f"Building EMA state_dict (no-swap) for save_dir={save_dir}")
        ema_sd = build_ema_state_dict_no_swap(
            base_model,
            self.ema,
            keep_fp32=self.keep_fp32,
        )

        _log_info(
            f"Saving EMA snapshot via save_pretrained: dir={save_dir} "
            f"| safe_serialization={self.safe_serialization} | ema_sd_keys={len(ema_sd)}"
        )
        try:
            base_model.save_pretrained(
                save_dir,
                state_dict=ema_sd,
                safe_serialization=self.safe_serialization,
            )
        except TypeError:
            base_model.save_pretrained(save_dir, state_dict=ema_sd)

        try:
            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(save_dir)
                _log_debug(f"Saved tokenizer to {save_dir}")
        except Exception as e:
            _log_warning(f"Failed to save tokenizer to {save_dir}: {e}")

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            _log_debug("on_save: skipping (not world_process_zero).")
            return control

        model = kwargs.get("model", None)
        if model is None:
            _log_warning("on_save: no model provided in kwargs; skipping EMA save.")
            return control

        tokenizer = kwargs.get("tokenizer", None)
        step = int(state.global_step)
        save_dir = self._ema_ckpt_dir(args, step)

        _log_info(
            f"on_save triggered: global_step={step} | saving EMA checkpoint to {save_dir}"
        )

        try:
            self._save_ema_snapshot_no_swap(
                save_dir=save_dir, model=model, tokenizer=tokenizer
            )
            _log_info(f"Saved EMA (no-swap) checkpoint to: {save_dir}")
        except Exception as e:
            _log_warning(f"Failed saving EMA (no-swap) checkpoint to {save_dir}: {e}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            _log_debug("on_train_end: skipping (not world_process_zero).")
            return control

        model = kwargs.get("model", None)
        if model is None:
            _log_warning(
                "on_train_end: no model provided in kwargs; skipping EMA final save."
            )
            return control

        tokenizer = kwargs.get("tokenizer", None)
        save_dir = self._ema_root_dir(args)

        _log_info(f"on_train_end triggered: saving final EMA model to {save_dir}")

        try:
            self._save_ema_snapshot_no_swap(
                save_dir=save_dir, model=model, tokenizer=tokenizer
            )
            _log_info(f"Saved final EMA (no-swap) model to: {save_dir}")
        except Exception as e:
            _log_warning(f"Failed saving final EMA (no-swap) model to {save_dir}: {e}")

        return control


# ---------------------------
# Misc callbacks
# ---------------------------


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        try:
            if "eval_loss" in metrics and metrics["eval_loss"] is not None:
                import math

                metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
                _log_info(f"Eval perplexity: {metrics['eval_perplexity']:.4f}")
        except Exception:
            pass


class KHeadsLoggingCallback(TrainerCallback):
    """
    Logs per-k (subnetwork) running averages at each logging step.
    Injects TB scalars for:
      - k{K}/loss_avg
      - k{K}/ce_avg
      - k{K}/kd_avg
      - k{K}/z_avg
      - k{K}/grad_norm_last (best-effort attribution)
    """

    def __init__(self, hydra_trainer: "HydraTrainer"):
        self.trainer = hydra_trainer
        self._reentrant = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        try:
            if "grad_norm" in logs:
                last_k = getattr(self.trainer, "_last_k_for_log", None)
                if last_k is not None:
                    self.trainer.kheads_stats[last_k]["grad_norm_last"] = float(
                        logs["grad_norm"]
                    )
        except Exception:
            pass

        payload = {}
        try:
            stats = self.trainer.kheads_stats
            for k in sorted(stats.keys()):
                s = stats[k]
                cnt = max(1, int(s.get("count", 0)))
                payload[f"k{k}/count_total"] = float(s.get("count", 0.0))
                payload[f"k{k}/loss_avg"] = float(s.get("loss_sum", 0.0)) / cnt
                payload[f"k{k}/ce_avg"] = float(s.get("ce_sum", 0.0)) / cnt
                payload[f"k{k}/z_avg"] = float(s.get("z_sum", 0.0)) / cnt
                payload[f"k{k}/kd_avg"] = float(s.get("kd_sum", 0.0)) / cnt
                gn = s.get("grad_norm_last", None)
                if gn is not None:
                    payload[f"k{k}/grad_norm_last"] = float(gn)

            w_stats = self.trainer.kheads_window_stats
            for k in sorted(w_stats.keys()):
                sw = w_stats[k]
                if sw["count"] > 0:
                    payload[f"k{k}/count_step"] = float(sw.get("count", 0.0))
                    payload[f"k{k}/loss_step"] = (
                        float(sw.get("loss_sum", 0.0)) / sw["count"]
                    )
                    payload[f"k{k}/ce_step"] = (
                        float(sw.get("ce_sum", 0.0)) / sw["count"]
                    )
                    payload[f"k{k}/z_step"] = float(sw.get("z_sum", 0.0)) / sw["count"]
                    payload[f"k{k}/kd_step"] = (
                        float(sw.get("kd_sum", 0.0)) / sw["count"]
                    )

            self.trainer.kheads_window_stats.clear()
        except Exception as e:
            _log_warning(f"Failed to calculate per-k windowed metrics: {e}")

        if payload:
            if self._reentrant:
                return
            try:
                self._reentrant = True
                self.trainer.log(payload)
            finally:
                self._reentrant = False


class GranularityLoggingCallback(TrainerCallback):
    """
    Logs per-granularity running averages and windowed stats at each logging step for MatFormer.
    """

    def __init__(self, matformer_trainer: "MatFormerTrainer"):
        self.trainer = matformer_trainer
        self._reentrant = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}

        try:
            if "grad_norm" in logs:
                last_g = getattr(self.trainer, "_last_granularity_for_log", None)
                if last_g is not None:
                    self.trainer.granularity_stats[last_g]["grad_norm_last"] = float(
                        logs["grad_norm"]
                    )
        except Exception:
            pass

        payload = {}
        try:
            stats = self.trainer.granularity_stats
            for g in sorted(stats.keys()):
                s = stats[g]
                cnt = max(1, int(s.get("count", 0)))
                payload[f"g{g}/count_total"] = float(s.get("count", 0.0))
                payload[f"g{g}/loss_avg"] = float(s.get("loss_sum", 0.0)) / cnt
                payload[f"g{g}/ce_avg"] = float(s.get("ce_sum", 0.0)) / cnt
                payload[f"g{g}/z_avg"] = float(s.get("z_sum", 0.0)) / cnt
                payload[f"g{g}/kd_avg"] = float(s.get("kd_sum", 0.0)) / cnt
                gn = s.get("grad_norm_last", None)
                if gn is not None:
                    payload[f"g{g}/grad_norm_last"] = float(gn)

            w_stats = self.trainer.granularity_window_stats
            for g in sorted(w_stats.keys()):
                sw = w_stats[g]
                if sw["count"] > 0:
                    payload[f"g{g}/count_step"] = float(sw.get("count", 0.0))
                    payload[f"g{g}/loss_step"] = (
                        float(sw.get("loss_sum", 0.0)) / sw["count"]
                    )
                    payload[f"g{g}/ce_step"] = (
                        float(sw.get("ce_sum", 0.0)) / sw["count"]
                    )
                    payload[f"g{g}/z_step"] = float(sw.get("z_sum", 0.0)) / sw["count"]
                    payload[f"g{g}/kd_step"] = (
                        float(sw.get("kd_sum", 0.0)) / sw["count"]
                    )

            self.trainer.granularity_window_stats.clear()
        except Exception as e:
            _log_warning(f"Failed to calculate per-granularity windowed metrics: {e}")

        try:
            parts = []
            for g in sorted(self.trainer.granularity_stats.keys()):
                s = self.trainer.granularity_stats[g]
                cnt = max(1, int(s.get("count", 0)))
                avg = float(s.get("loss_sum", 0.0)) / cnt
                parts.append(f"g={g}: {avg:.4f}")
            if parts:
                _log_info("[per-granularity] " + " | ".join(parts))
        except Exception:
            pass

        if payload:
            if self._reentrant:
                return
            try:
                self._reentrant = True
                self.trainer.log(payload)
            finally:
                self._reentrant = False
