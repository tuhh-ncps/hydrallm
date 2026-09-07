# train_llm/trainer_hydravit.py
from typing import List, Optional
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


class HydraTrainer(Trainer):
    """
    HydraTrainer (LLM-only pipeline):
    - Inject current_num_heads dynamically per step (HydraViT-style).
    - Supports z-loss and KD with a full-heads teacher.
    - Keeps per-k running statistics for logging.

    """

    def __init__(
        self,
        *args,
        heads: Optional[List[int]] = None,
        head_weights: Optional[List[float]] = None,
        loss_type: str = "ce_plus_z",
        z_loss_alpha: float = 0.0,
        teacher_model: Optional[torch.nn.Module] = None,
        teacher_full_heads: Optional[int] = None,
        kd_temperature: float = 2.0,
        kd_weight: float = 0.0,
        exclude_wd_norms_embeddings: bool = True,
        kd_every_n_steps: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heads = heads or []
        self.head_weights = head_weights

        self.loss_type = (loss_type or "ce_plus_z").lower()
        self.z_loss_alpha = float(z_loss_alpha)

        self.teacher_model = teacher_model
        self.teacher_full_heads = teacher_full_heads
        self.kd_temperature = float(kd_temperature)
        self.kd_weight = float(kd_weight)
        self._kd_vocab_warned = False  # warn once if slicing vocab

        # control WD exclusions and KD interval
        self.exclude_wd_norms_embeddings = bool(exclude_wd_norms_embeddings)
        self.kd_every_n_steps = max(1, int(kd_every_n_steps))

        # ---- Per-k cumulative stats (per OPT step) ----
        self.kheads_stats = defaultdict(
            lambda: {
                "count": 0,  # number of optimizer steps for this k
                "loss_sum": 0.0,  # sum of per-step averaged loss
                "ce_sum": 0.0,
                "z_sum": 0.0,
                "kd_sum": 0.0,
                "grad_norm_last": None,
            }
        )

        # ---- Per-k windowed stats (reset every logging step by callback) ----
        self.kheads_window_stats = defaultdict(
            lambda: {
                "count": 0,  # number of optimizer steps in current window
                "loss_sum": 0.0,
                "ce_sum": 0.0,
                "z_sum": 0.0,
                "kd_sum": 0.0,
            }
        )
        self._last_k_for_log: Optional[int] = None

        # ---- Accumulation-window control (fix k across microbatches) ----
        self._step_k: Optional[int] = None

        # ---- Microbatch accumulators (average on optimizer step) ----
        self._micro_count = 0
        self._micro_loss_sum = 0.0
        self._micro_ce_sum = 0.0
        self._micro_z_sum = 0.0
        self._micro_kd_sum = 0.0

    def _is_optimizer_step(self) -> bool:
        # True only on the LAST microbatch of grad-accum window
        try:
            return bool(getattr(self.accelerator, "sync_gradients", True))
        except Exception:
            return True

    def _pick_heads(self, model) -> Optional[int]:
        if not self.heads:
            return None
        if self.head_weights is None:
            return random.choice(self.heads)
        return random.choices(self.heads, weights=self.head_weights, k=1)[0]

    def _effective_k(self, model, k: Optional[int]) -> int:
        """
        Map k=None to "full heads" for stats keys.
        """
        if k is not None:
            return int(k)
        try:
            full_heads = int(getattr(model.config, "num_attention_heads", 0)) or 0
        except Exception:
            full_heads = 0
        return int(full_heads)

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
        if self.kd_weight <= 0.0:
            return student_logits.new_zeros(())

        # Align vocab sizes if they differ
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
        # # ------------------------------------------------------------------
        # # DETERMINISM CHECK
        # # ------------------------------------------------------------------
        # # Print batch signature for the first 10 steps to verify exact data match.
        # if self.is_world_process_zero() and self.state.global_step < 10:
        #     input_ids = inputs.get("input_ids")
        #     if input_ids is not None:
        #         # Calculate a checksum of the entire batch
        #         batch_sum = input_ids.sum().item()
        #         # Get the first 10 tokens of the first row
        #         start_tokens = input_ids[0, :10].tolist()

        #         print(
        #             f"\n>>> [STEP {self.state.global_step}] "
        #             f"Batch Checksum: {batch_sum} | "
        #             f"Start Tokens: {start_tokens}",
        #             flush=True
        #         )
        # # ------------------------------------------------------------------

        if self._step_k is None:
            self._step_k = self._pick_heads(model)
        k = self._step_k
        eff_k = self._effective_k(model, k)

        outputs = model(**inputs, current_num_heads=k, use_cache=False)
        logits = outputs.logits
        base_ce = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        labels = inputs.get("labels", None)
        z = self._z_loss(logits, labels)

        kd = logits.new_zeros(())
        if self.teacher_model is not None and self.kd_weight > 0.0:
            # NOTE: state.global_step increments per OPT step; this schedules KD per optimizer step.
            # It will still run on every microbatch in that window when KD is "on", since inputs differ.
            step_now = int(self.state.global_step or 0) + 1
            use_kd_now = (step_now % self.kd_every_n_steps) == 0
            if use_kd_now:
                with torch.no_grad():
                    dev = next(model.parameters()).device
                    if next(self.teacher_model.parameters()).device != dev:
                        self.teacher_model.to(dev)

                    t_kwargs = dict(use_cache=False)
                    if self.teacher_full_heads is not None:
                        t_kwargs["current_num_heads"] = int(self.teacher_full_heads)

                    teacher_out = self.teacher_model(
                        input_ids=inputs.get("input_ids", None),
                        attention_mask=inputs.get("attention_mask", None),
                        **t_kwargs,
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

            # 1) cumulative per-k stats (per optimizer step)
            s = self.kheads_stats[eff_k]
            s["count"] += 1
            s["loss_sum"] += loss_avg
            s["ce_sum"] += ce_avg
            s["z_sum"] += z_avg
            s["kd_sum"] += kd_avg

            # 2) windowed per-k stats (per optimizer step)
            sw = self.kheads_window_stats[eff_k]
            sw["count"] += 1
            sw["loss_sum"] += loss_avg
            sw["ce_sum"] += ce_avg
            sw["z_sum"] += z_avg
            sw["kd_sum"] += kd_avg

            self._last_k_for_log = eff_k

            # reset micro accumulators + k for next accumulation window
            self._micro_count = 0
            self._micro_loss_sum = 0.0
            self._micro_ce_sum = 0.0
            self._micro_z_sum = 0.0
            self._micro_kd_sum = 0.0
            self._step_k = None

        return (loss, outputs) if return_outputs else loss
