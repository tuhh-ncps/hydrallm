# train_llm/trainer_matformer.py
from typing import List, Optional, Union
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import logging as hf_logging
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

logger = hf_logging.get_logger(__name__)


class MatFormerTrainer(Trainer):
    """
    MatFormerTrainer (LLM-only pipeline):
    - Injects a global or per-layer `granularity` dynamically per step.
    - Supports z-loss and KD with a full-width teacher.
    - Keeps per-granularity running statistics for logging.
    - Mirrors HydraTrainer in all accumulation and KD logic.
    """

    def __init__(
        self,
        *args,
        ffn_granularity_ratios: Optional[List[float]] = None,
        ffn_granularity_weights: Optional[List[float]] = None,
        width_sampling_mode: str = "global",
        loss_type: str = "ce_plus_z",
        z_loss_alpha: float = 0.0,
        teacher_model: Optional[torch.nn.Module] = None,
        kd_temperature: float = 2.0,
        kd_weight: float = 0.0,
        exclude_wd_norms_embeddings: bool = True,
        kd_every_n_steps: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ffn_granularity_ratios = ffn_granularity_ratios or []
        self.ffn_granularity_weights = ffn_granularity_weights
        self.width_sampling_mode = (width_sampling_mode or "global").lower()
        if self.width_sampling_mode not in ("global", "per_layer"):
            raise ValueError(
                "width_sampling_mode must be either 'global' or 'per_layer'"
            )

        self.loss_type = (loss_type or "ce_plus_z").lower()
        self.z_loss_alpha = float(z_loss_alpha)

        self.teacher_model = teacher_model
        self.kd_temperature = float(kd_temperature)
        self.kd_weight = float(kd_weight)
        self._kd_vocab_warned = False

        self.exclude_wd_norms_embeddings = bool(exclude_wd_norms_embeddings)
        self.kd_every_n_steps = max(1, int(kd_every_n_steps))

        # ---- Per-granularity cumulative stats (per OPT step) ----
        self.granularity_stats = defaultdict(
            lambda: {
                "count": 0,
                "loss_sum": 0.0,
                "ce_sum": 0.0,
                "z_sum": 0.0,
                "kd_sum": 0.0,
                "grad_norm_last": None,
            }
        )

        # ---- Per-granularity windowed stats (reset every logging step by callback) ----
        self.granularity_window_stats = defaultdict(
            lambda: {
                "count": 0,
                "loss_sum": 0.0,
                "ce_sum": 0.0,
                "z_sum": 0.0,
                "kd_sum": 0.0,
            }
        )
        self._last_granularity_for_log: Optional[Union[int, str]] = None

        # ---- Accumulation-window control (fix g across microbatches) ----
        self._step_g: Optional[Union[int, List[int]]] = None

        # ---- Microbatch accumulators (averaged on optimizer step) ----
        self._micro_count = 0
        self._micro_loss_sum = 0.0
        self._micro_ce_sum = 0.0
        self._micro_z_sum = 0.0
        self._micro_kd_sum = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_optimizer_step(self) -> bool:
        """True only on the LAST microbatch of the grad-accum window."""
        try:
            return bool(getattr(self.accelerator, "sync_gradients", True))
        except Exception:
            return True

    def _pick_one_granularity(self) -> Optional[int]:
        if not self.ffn_granularity_ratios:
            return None
        indices = list(range(len(self.ffn_granularity_ratios)))
        if self.ffn_granularity_weights is None:
            return random.choice(indices)
        return random.choices(indices, weights=self.ffn_granularity_weights, k=1)[0]

    def _pick_granularity(self, model) -> Optional[Union[int, List[int]]]:
        """Sample one global index or one independent index per decoder layer."""
        if self.width_sampling_mode == "global" or not self.ffn_granularity_ratios:
            return self._pick_one_granularity()
        num_layers = int(getattr(model.config, "num_hidden_layers", 0))
        if num_layers <= 0:
            raise ValueError(
                "per_layer width sampling requires model.config.num_hidden_layers"
            )
        return [self._pick_one_granularity() for _ in range(num_layers)]

    def _effective_g(self, g: Optional[Union[int, List[int]]]) -> Union[int, str]:
        if isinstance(g, list):
            return "per_layer"
        if not self.ffn_granularity_ratios:
            return 0
        if g is None:
            return int(len(self.ffn_granularity_ratios) - 1)
        return int(g)

    def _z_loss(
        self, logits: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.z_loss_alpha <= 0.0:
            return logits.new_zeros(())
        dtype = logits.dtype
        if dtype == torch.float16:
            z = torch.logsumexp(logits.float(), dim=-1)
        else:
            z = torch.logsumexp(logits, dim=-1)
        if labels is not None:
            mask = (labels != -100).to(z.dtype)
            denom = mask.sum().clamp_min(1.0)
            z2 = (z.pow(2) * mask).sum() / denom
        else:
            z2 = z.pow(2).mean()
        return (self.z_loss_alpha * z2).to(dtype)

    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        KD loss — MUST match HydraTrainer._kd_loss exactly.

        Critical: do NOT upcast to .float(). With packed sequences
        (seq_len=3072, vocab=262208), each [B,T,V] tensor is ~1.6 GB
        in bf16 vs ~3.2 GB in fp32. The .float() upcast doubles peak
        VRAM and causes OOM on 24 GB GPUs.
        """
        if self.kd_weight <= 0.0:
            return student_logits.new_zeros(())

        V_s = student_logits.size(-1)
        V_t = teacher_logits.size(-1)
        if V_s != V_t:
            V = min(V_s, V_t)
            if not self._kd_vocab_warned:
                logger.warning(
                    f"[KD] Student/Teacher vocab mismatch (student={V_s}, teacher={V_t}). "
                    f"Slicing to common size {V}."
                )
                self._kd_vocab_warned = True
            student_logits = student_logits[..., :V]
            teacher_logits = teacher_logits[..., :V]

        # ── Identical to HydraTrainer: native dtype, mask AFTER softmax ──
        tau = self.kd_temperature
        s_logp = F.log_softmax(student_logits / tau, dim=-1)
        t_prob = F.softmax(teacher_logits / tau, dim=-1)

        if labels is not None:
            mask = labels != -100
            s_logp = s_logp[mask]
            t_prob = t_prob[mask]

        if s_logp.numel() == 0:
            return student_logits.new_zeros(())

        kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (tau * tau)
        return self.kd_weight * kl

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def get_decay_parameter_names(self, model):
        # Keep original behavior when flag is false
        if not self.exclude_wd_norms_embeddings:
            return super().get_decay_parameter_names(model)

        # Exclude norm layers and embeddings from weight decay
        forbidden_layer_types = list(ALL_LAYERNORM_LAYERS)
        forbidden_layer_types.append(nn.Embedding)

        decay_parameters = get_parameter_names(model, tuple(forbidden_layer_types))
        decay_parameters = [
            name for name in decay_parameters if not name.endswith(".bias")
        ]
        return decay_parameters

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._step_g is None:
            self._step_g = self._pick_granularity(model)
        granularity_idx = self._step_g
        eff_g = self._effective_g(granularity_idx)

        outputs = model(**inputs, granularity=granularity_idx, use_cache=False)
        logits = outputs.logits
        base_ce = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        labels = inputs.get("labels", None)
        z = self._z_loss(logits, labels)

        kd = logits.new_zeros(())
        if self.teacher_model is not None and self.kd_weight > 0.0:
            step_now = int(self.state.global_step or 0) + 1
            use_kd_now = (step_now % self.kd_every_n_steps) == 0
            if use_kd_now:
                with torch.no_grad():
                    dev = next(model.parameters()).device
                    if next(self.teacher_model.parameters()).device != dev:
                        self.teacher_model.to(dev)

                    # Teacher is plain AutoModelForCausalLM — no granularity arg
                    teacher_out = self.teacher_model(
                        input_ids=inputs.get("input_ids", None),
                        attention_mask=inputs.get("attention_mask", None),
                        use_cache=False,
                    )
                    teacher_logits = teacher_out.logits

                kd = self._kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                )

        lt = self.loss_type
        if lt == "ce":
            loss = base_ce + kd
        elif lt == "ce_plus_z":
            loss = base_ce + z + kd
        elif lt in ("z_only", "zloss_only"):
            loss = z + kd
        else:
            loss = base_ce + z + kd

        # ---- Accumulate MICRO stats (every microbatch) ----
        try:
            self._micro_count += 1
            self._micro_loss_sum += float(loss.detach().item())
            self._micro_ce_sum += float(base_ce.detach().item())
            self._micro_z_sum += float(z.detach().item())
            self._micro_kd_sum += float(kd.detach().item())
        except Exception:
            pass

        # ---- Finalize once per OPT step (average over microbatches) ----
        if self._is_optimizer_step():
            denom = max(1, int(self._micro_count))
            loss_avg = self._micro_loss_sum / denom
            ce_avg = self._micro_ce_sum / denom
            z_avg = self._micro_z_sum / denom
            kd_avg = self._micro_kd_sum / denom

            s = self.granularity_stats[eff_g]
            s["count"] += 1
            s["loss_sum"] += loss_avg
            s["ce_sum"] += ce_avg
            s["z_sum"] += z_avg
            s["kd_sum"] += kd_avg

            sw = self.granularity_window_stats[eff_g]
            sw["count"] += 1
            sw["loss_sum"] += loss_avg
            sw["ce_sum"] += ce_avg
            sw["z_sum"] += z_avg
            sw["kd_sum"] += kd_avg

            self._last_granularity_for_log = eff_g

            self._micro_count = 0
            self._micro_loss_sum = 0.0
            self._micro_ce_sum = 0.0
            self._micro_z_sum = 0.0
            self._micro_kd_sum = 0.0
            self._step_g = None

        return (loss, outputs) if return_outputs else loss
