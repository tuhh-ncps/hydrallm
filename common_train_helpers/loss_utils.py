# common_train_helpers/loss_utils.py
"""Loss computation utilities (z-loss, KD loss) shared across trainers."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)


def compute_z_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    z_loss_alpha: float,
) -> torch.Tensor:
    """
    Compute the z-loss (log-partition regularizer).
    Returns a zero tensor if z_loss_alpha <= 0.
    """
    if z_loss_alpha <= 0.0:
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
    return (z_loss_alpha * z2).to(dtype)


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    kd_weight: float,
    kd_temperature: float,
    *,
    _vocab_warned_ref: Optional[list] = None,
) -> torch.Tensor:
    """
    Compute KL-divergence KD loss between student and teacher logits.

    Args:
        _vocab_warned_ref: Pass a single-element list like [False] to track
            whether the vocab-mismatch warning has been emitted. The element
            will be set to True after the first warning.

    Returns a zero tensor if kd_weight <= 0 or labels is None.
    """
    if kd_weight <= 0.0 or labels is None:
        return student_logits.new_zeros(())

    V_s = student_logits.size(-1)
    V_t = teacher_logits.size(-1)
    if V_s != V_t:
        V = min(V_s, V_t)
        if _vocab_warned_ref is not None and not _vocab_warned_ref[0]:
            logger.warning(
                f"[KD] Student/Teacher vocab mismatch (student={V_s}, teacher={V_t}). "
                f"Slicing to common size {V}."
            )
            _vocab_warned_ref[0] = True
        student_logits = student_logits[..., :V]
        teacher_logits = teacher_logits[..., :V]

    mask = labels != -100
    if mask.sum().item() == 0:
        return student_logits.new_zeros(())

    s = student_logits[mask]
    t = teacher_logits[mask]
    tau = kd_temperature
    s_logp = F.log_softmax(s.float() / tau, dim=-1)
    t_prob = F.softmax(t.float() / tau, dim=-1)
    kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (tau * tau)
    return (kd_weight * kl).to(student_logits.dtype)
