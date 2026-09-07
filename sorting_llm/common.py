# sorting_llm/common.py
from __future__ import annotations

"""
sorting_llm.common
Shared utilities for:
- loading models/tokenizers
- registering hooks for importance metrics
- computing per-layer importance scores
========================
CETT metric semantics
========================
CETT in the papers measures the (normalized) truncation error introduced by
dropping a *tail* set of small contributors. For reordering, we need *per-unit*
scores that produce a meaningful tail curve.
We implement CETT-style per-unit scores, computed at every token position t:
MLP neurons (hooked on down_proj):
  hidden_t is the down_proj input (shape [I])
  W_down[:, j] is the weight column for neuron j (shape [H])
  contrib_norm_{t,j} = || hidden_{t,j} * W_down[:, j] ||_2
                     = |hidden_{t,j}| * ||W_down[:, j]||_2
  cett:
      score_j = E_t[ contrib_norm_{t,j} ]                     (unnormalized)
  cett_normalized:
      score_j = E_t[ contrib_norm_{t,j} / ||FFN_out_t||_2 ]   (CETT-style ratio)
  cett_variance:
      score_j = Var_t[ contrib_norm_{t,j} / ||FFN_out_t||_2 ]
Attention heads (hooked on o_proj):
  x_{t,h} is the per-head attention output before o_proj (shape [D])
  W_o[:, h] is the corresponding slice of o_proj.weight (shape [H, D])
  head_contrib_norm_{t,h} = || W_o[:, h] @ x_{t,h} ||_2
  cett:
      score_h = E_t[ head_contrib_norm_{t,h} ]
  cett_normalized:
      score_h = E_t[ head_contrib_norm_{t,h} / ||Attn_out_t||_2 ]
  cett_variance:
      score_h = Var_t[ head_contrib_norm_{t,h} / ||Attn_out_t||_2 ]
Tail curve:
  If scores are computed using cett_normalized, then an upper bound on the
  expected CETT tail error when pruning the k least-important units is:
      CETT_ub(k) = sum_{i in k-smallest} score_i
  (triangle inequality + linearity of expectation)
This is what we report as the "CETT tail curve".
"""
import math
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union, Iterable
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from common_helpers import parse_dtype, has_flash_attn2

# ---------------------- Metric constants ---------------------- #
CETT_METRICS = frozenset({"cett", "cett_normalized", "cett_variance"})


# ---------------------- Dirs utilities ---------------------- #
def make_dirs_for_config(cfg: Dict, command: str):
    outputs_dir = cfg.get("project", {}).get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    if command == "sort-llm":
        save_dir = cfg.get("sorting", {}).get("save_dir")
        if save_dir:
            parent_dir = os.path.dirname(save_dir)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        cal_path = cfg.get("sorting", {}).get("calibration_file")
        if cal_path:
            os.makedirs(os.path.dirname(cal_path) or ".", exist_ok=True)
    elif command == "plot-importance":
        plot_dir = cfg.get("plot_importance", {}).get("out_dir")
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        cal_path = cfg.get("plot_importance", {}).get("calibration_file")
        if cal_path:
            os.makedirs(os.path.dirname(cal_path) or ".", exist_ok=True)


def setup_sort_output_dir_and_config(cfg: Dict) -> str:
    model_cfg = cfg.get("sorting", {})
    base_save_dir = model_cfg.get("save_dir")
    timestamp_override = model_cfg.get("timestamp_override")
    timestamp = timestamp_override or datetime.now().strftime("%Y%m%d_%H%M%S")
    final_save_dir = f"{base_save_dir}_{timestamp}"
    os.makedirs(final_save_dir, exist_ok=True)
    config_save_path = os.path.join(final_save_dir, "sort_config.yaml")
    import yaml

    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Saved full sorting config to {config_save_path}")
    return final_save_dir


# ---------------------- Attention backend helpers ---------------------- #
def pick_best_attn_impl_for_sorting(device: str) -> str:
    dev = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if dev.startswith("cuda") and has_flash_attn2():
        return "flash_attention_2"
    if dev != "cpu":
        return "sdpa"
    return "eager"


def _apply_attn_implementation(model, impl: str, dtype: torch.dtype, device: str):
    """
    Best-effort: if the model supports changing attention implementation, do it.
    """
    try:
        chosen = impl
        if chosen == "flash_attention_2" and dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            chosen = "sdpa" if device != "cpu" else "eager"
        model.set_attn_implementation(chosen)
        print(f"[attn] Using attention implementation: {chosen}")
    except Exception:
        pass


def load_model_and_tokenizer(
    model_id: str,
    device: str,
    dtype_str: str,
    device_map: Optional[str] = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = parse_dtype(dtype_str, device)
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="cpu"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=(device_map or "auto")
        )
    impl = pick_best_attn_impl_for_sorting(device)
    _apply_attn_implementation(model, impl, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def load_model_and_tokenizer_single_device(
    model_id: str,
    tokenizer_ref: Optional[str],
    device: str,
    dtype_str: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = parse_dtype(dtype_str or "auto", device or "cpu")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="cpu"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=None
        )
        if device.startswith("cuda") or device == "mps":
            model = model.to(device)
    impl = pick_best_attn_impl_for_sorting(device)
    _apply_attn_implementation(model, impl, dtype, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref or model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


# =====================================================================
# Generic statistics helpers (stored on the hooked module)
# =====================================================================
_ALL_STAT_ATTRS = (
    # Generic online stats:
    "m1_sum",
    "m2_sum",
    "n_sum",
    "calculated_importance",
    # CETT caches:
    "_gram_cache",
    "_cett_num_heads",
    "_cett_head_dim",
    "_cett_col_norms_cache",
)


def _clear_module_stats(module):
    """Clear accumulated statistics / caches from a module before (re-)hooking."""
    for attr in _ALL_STAT_ATTRS:
        if hasattr(module, attr):
            delattr(module, attr)


def _accumulate_sum_cpu(module, attr: str, vec: torch.Tensor):
    """
    Accumulate a 1D tensor into module.attr on CPU.
    """
    v = vec.detach().to(torch.float32).cpu()
    if not hasattr(module, attr):
        setattr(module, attr, torch.zeros_like(v))
    getattr(module, attr).add_(v)


def _accumulate_n(module, n: int):
    if not hasattr(module, "n_sum"):
        module.n_sum = torch.zeros((), dtype=torch.long)
    module.n_sum += torch.tensor(int(n), dtype=torch.long)


def _accumulate_moments_cpu(module, m1: torch.Tensor, m2: torch.Tensor, n: int):
    """
    Accumulate scalar moments:
      m1_sum += sum(x)
      m2_sum += sum(x^2)
      n_sum  += count
    all stored on CPU.
    """
    m1_cpu = m1.detach().to(torch.float32).cpu()
    m2_cpu = m2.detach().to(torch.float32).cpu()
    if not hasattr(module, "m1_sum"):
        module.m1_sum = torch.zeros_like(m1_cpu)
        module.m2_sum = torch.zeros_like(m2_cpu)
        module.n_sum = torch.zeros((), dtype=torch.long)
    module.m1_sum.add_(m1_cpu)
    module.m2_sum.add_(m2_cpu)
    module.n_sum += torch.tensor(int(n), dtype=torch.long)


# =====================================================================
# "Original" metrics (magnitude, variance_x_consumers)
# =====================================================================


def _update_online_variance(module, x_flat: torch.Tensor):
    """
    Updates m1_sum, m2_sum, n_sum on the module using flat input [N, Dim].
    """
    m2 = (x_flat * x_flat).sum(dim=0).cpu()
    m1 = x_flat.sum(dim=0).cpu()
    n = torch.tensor(x_flat.shape[0], dtype=torch.long)
    if not hasattr(module, "m2_sum"):
        module.m2_sum = torch.zeros_like(m2)
        module.m1_sum = torch.zeros_like(m1)
        module.n_sum = torch.zeros((), dtype=torch.long)
    module.m2_sum += m2
    module.m1_sum += m1
    module.n_sum += n


def _update_magnitude(module, x_flat: torch.Tensor):
    """
    Updates calculated_importance and n_sum on the module using sum of absolute
    values (L1). The final score is the *average* absolute value per token,
    computed during finalization.
    """
    imp = x_flat.abs().sum(dim=0).cpu()
    n = x_flat.shape[0]
    if not hasattr(module, "calculated_importance"):
        module.calculated_importance = torch.zeros_like(imp)
    module.calculated_importance += imp
    _accumulate_n(module, n)


# --------------- Head hooks (q_proj) — original metrics --------------- #


def head_hook_magnitude(module, ins, outs):
    """Accumulates magnitude (L2 norm over head_dim) per head, averaged over tokens at finalization."""
    try:
        cfg = module.config
        num_heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, "head_dim", None) or (outs.shape[-1] // num_heads))
    except Exception:
        num_heads = 4
        head_dim = outs.shape[-1] // num_heads
    x = outs.detach().to(torch.float32)  # [B, T, H*D]
    B, T, _ = x.shape
    x = x.view(B, T, num_heads, head_dim)
    vectors_norm = torch.linalg.vector_norm(x, ord=2, dim=-1)  # [B, T, H]
    imp = vectors_norm.sum(dim=(0, 1)).cpu()  # [H]
    n = B * T
    if not hasattr(module, "calculated_importance"):
        module.calculated_importance = torch.zeros(num_heads, device="cpu")
    module.calculated_importance += imp
    _accumulate_n(module, n)


def head_hook_variance(module, ins, outs):
    """Accumulates sufficient stats (m1, m2) per element; per-head aggregation happens at finalization."""
    try:
        cfg = module.config
        num_heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, "head_dim", None) or (outs.shape[-1] // num_heads))
    except Exception:
        num_heads = 4
        head_dim = outs.shape[-1] // num_heads
    x = outs.detach().to(torch.float32)
    x_flat = x.view(-1, num_heads * head_dim)
    _update_online_variance(module, x_flat)


# --------------- Linear hooks — original metrics --------------- #


def linear_hook_magnitude(module, ins, outs):
    """Generic hook for MLP neurons or Embeddings (L1 sum-abs), averaged over tokens at finalization."""
    x = outs.detach().to(torch.float32) # Change this to ins[0] for measuring the input to the down projection (change the element the hook is being applied to too!)
    x_flat = x.reshape(-1, x.shape[-1])
    _update_magnitude(module, x_flat)


def linear_hook_variance(module, ins, outs):
    """Generic hook for variance stats."""
    x = outs.detach().to(torch.float32)
    x_flat = x.reshape(-1, x.shape[-1])
    _update_online_variance(module, x_flat)


# =====================================================================
# CETT helpers — attention heads (hooked on o_proj)
# =====================================================================


def _compute_head_gram_matrix(
    o_proj_weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Compute Gram matrix G[h] = W_h^T @ W_h for each head, where
    W_h = o_proj.weight[:, h*D:(h+1)*D].
    Returns: [H, D, D] float32 on the same device as o_proj_weight.
    """
    W = o_proj_weight.to(torch.float32)  # [hidden, H*D]
    W_per_head = W.view(-1, num_heads, head_dim)  # [hidden, H, D]
    W_per_head = W_per_head.permute(1, 0, 2)  # [H, hidden, D]
    return torch.bmm(W_per_head.transpose(1, 2), W_per_head)  # [H, D, D]


def _per_head_norms_via_gram(x_heads: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    x_heads: [N, H, D]   per-head inputs to o_proj
    G:       [H, D, D]   Gram matrices W_h^T W_h
    Returns: [N, H]      ||W_h @ x_{n,h}||_2
    """
    xG = torch.einsum("nhd,hde->nhe", x_heads, G)  # [N, H, D]
    quad = (xG * x_heads).sum(dim=-1)  # [N, H] = v^T G v
    return quad.clamp(min=0).sqrt()


def _get_or_build_o_proj_gram(
    o_proj: nn.Linear, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Cache Gram matrix on the module to avoid recomputing every forward.
    """
    if not hasattr(o_proj, "_gram_cache"):
        with torch.no_grad():
            o_proj._gram_cache = _compute_head_gram_matrix(
                o_proj.weight.detach(), num_heads, head_dim
            )
    return o_proj._gram_cache


def _reshape_o_proj_input_to_heads(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    x: [B, T, H*D] -> [N, H, D]
    """
    x = x.detach().to(torch.float32)
    B, T, _ = x.shape
    N = B * T
    return x.reshape(N, num_heads, head_dim)


def attn_o_proj_hook_cett(o_proj, ins, outs):
    """
    cett for heads: mean per-token head contribution norm:
      score_h = E_t[ || W_h @ v_{t,h} ||_2 ]
    """
    x = ins[0]  # input to o_proj: [B, T, H*D]
    num_heads = int(o_proj._cett_num_heads)
    head_dim = int(o_proj._cett_head_dim)
    x_heads = _reshape_o_proj_input_to_heads(x, num_heads, head_dim)  # [N,H,D]
    N = x_heads.shape[0]
    G = _get_or_build_o_proj_gram(o_proj, num_heads, head_dim)  # [H,D,D]
    norms = _per_head_norms_via_gram(x_heads, G)  # [N,H]
    _accumulate_sum_cpu(o_proj, "calculated_importance", norms.sum(dim=0))
    _accumulate_n(o_proj, N)


def attn_o_proj_hook_cett_normalized(o_proj, ins, outs):
    """
    cett_normalized for heads: mean per-token ratio:
      score_h = E_t[ ||W_h @ v_{t,h}||_2 / ||AttnOut_t||_2 ]
    """
    x = ins[0]
    attn_out = outs
    num_heads = int(o_proj._cett_num_heads)
    head_dim = int(o_proj._cett_head_dim)
    x_heads = _reshape_o_proj_input_to_heads(x, num_heads, head_dim)  # [N,H,D]
    N = x_heads.shape[0]
    G = _get_or_build_o_proj_gram(o_proj, num_heads, head_dim)
    head_norms = _per_head_norms_via_gram(x_heads, G)  # [N,H]
    attn_out_f = attn_out.detach().to(torch.float32).reshape(N, -1)
    total_norms = torch.linalg.vector_norm(attn_out_f, ord=2, dim=-1).clamp(
        min=1e-8
    )  # [N]
    ratios = head_norms / total_norms.unsqueeze(-1)  # [N,H]
    _accumulate_sum_cpu(o_proj, "calculated_importance", ratios.sum(dim=0))
    _accumulate_n(o_proj, N)


def attn_o_proj_hook_cett_variance(o_proj, ins, outs):
    """
    cett_variance for heads: variance over tokens of the normalized CETT ratio:
      score_h = Var_t[ ||W_h @ v_{t,h}||_2 / ||AttnOut_t||_2 ]
    """
    x = ins[0]
    attn_out = outs
    num_heads = int(o_proj._cett_num_heads)
    head_dim = int(o_proj._cett_head_dim)
    x_heads = _reshape_o_proj_input_to_heads(x, num_heads, head_dim)  # [N,H,D]
    N = x_heads.shape[0]
    G = _get_or_build_o_proj_gram(o_proj, num_heads, head_dim)
    head_norms = _per_head_norms_via_gram(x_heads, G)  # [N,H]
    attn_out_f = attn_out.detach().to(torch.float32).reshape(N, -1)
    total_norms = torch.linalg.vector_norm(attn_out_f, ord=2, dim=-1).clamp(
        min=1e-8
    )  # [N]
    ratios = head_norms / total_norms.unsqueeze(-1)  # [N,H]
    m1 = ratios.sum(dim=0)
    m2 = (ratios * ratios).sum(dim=0)
    _accumulate_moments_cpu(o_proj, m1, m2, N)


# =====================================================================
# CETT helpers — MLP neurons (hooked on down_proj)
# =====================================================================


def _get_or_build_down_col_norms(down_proj: nn.Linear) -> torch.Tensor:
    """
    Cache ||W_down[:,j]||_2 for each neuron j.
    Stored on the same device as down_proj.weight, float32.
    """
    if not hasattr(down_proj, "_cett_col_norms_cache"):
        with torch.no_grad():
            W = down_proj.weight.detach().to(torch.float32)  # [hidden, I]
            down_proj._cett_col_norms_cache = torch.linalg.vector_norm(
                W, ord=2, dim=0
            )  # [I]
    return down_proj._cett_col_norms_cache


def _mlp_contrib_norms_from_down_proj_io(
    down_proj: nn.Linear, hidden_in: torch.Tensor
) -> torch.Tensor:
    """
    hidden_in: input to down_proj (i.e. post-activation intermediate), shape [..., I]
    returns contrib norms per neuron, shape [..., I]:
      |hidden| * ||W_col||_2
    """
    hidden_abs = hidden_in.detach().to(torch.float32).abs()
    col_norms = _get_or_build_down_col_norms(down_proj).to(hidden_abs.device)
    return hidden_abs * col_norms


def mlp_down_proj_hook_cett(down_proj, ins, outs):
    """
    cett for neurons: mean per-token neuron contribution norm:
      score_j = E_t[ || hidden_{t,j} * W_down[:,j] ||_2 ]
    """
    hidden_in = ins[0]  # [B,T,I]
    contrib = _mlp_contrib_norms_from_down_proj_io(down_proj, hidden_in)  # [B,T,I]
    flat = contrib.reshape(-1, contrib.shape[-1])  # [N,I]
    N = flat.shape[0]
    _accumulate_sum_cpu(down_proj, "calculated_importance", flat.sum(dim=0))
    _accumulate_n(down_proj, N)


def mlp_down_proj_hook_cett_normalized(down_proj, ins, outs):
    """
    cett_normalized for neurons: mean per-token ratio:
      score_j = E_t[ ||hidden_{t,j} * W_col_j||_2 / ||FFN_out_t||_2 ]
    """
    hidden_in = ins[0]
    ffn_out = outs  # [B,T,H]
    contrib = _mlp_contrib_norms_from_down_proj_io(down_proj, hidden_in)
    contrib_flat = contrib.reshape(-1, contrib.shape[-1])  # [N,I]
    N = contrib_flat.shape[0]
    out_f = ffn_out.detach().to(torch.float32).reshape(N, -1)
    out_norms = torch.linalg.vector_norm(out_f, ord=2, dim=-1).clamp(min=1e-8)  # [N]
    ratios = contrib_flat / out_norms.unsqueeze(-1)  # [N,I]
    _accumulate_sum_cpu(down_proj, "calculated_importance", ratios.sum(dim=0))
    _accumulate_n(down_proj, N)


def mlp_down_proj_hook_cett_variance(down_proj, ins, outs):
    """
    cett_variance for neurons: variance over tokens of normalized CETT ratio:
      score_j = Var_t[ ||hidden_{t,j} * W_col_j||_2 / ||FFN_out_t||_2 ]
    """
    hidden_in = ins[0]
    ffn_out = outs
    contrib = _mlp_contrib_norms_from_down_proj_io(down_proj, hidden_in)
    contrib_flat = contrib.reshape(-1, contrib.shape[-1])  # [N,I]
    N = contrib_flat.shape[0]
    out_f = ffn_out.detach().to(torch.float32).reshape(N, -1)
    out_norms = torch.linalg.vector_norm(out_f, ord=2, dim=-1).clamp(min=1e-8)  # [N]
    # Avoid materializing ratios twice
    ratios = contrib_flat / out_norms.unsqueeze(-1)  # [N,I]
    m1 = ratios.sum(dim=0)
    m2 = (ratios * ratios).sum(dim=0)
    _accumulate_moments_cpu(down_proj, m1, m2, N)


# =====================================================================
# Consumer Norm Calculators (used by variance_x_consumers)
# =====================================================================


def get_consumer_colnorms_generic(layers_list) -> torch.Tensor:
    """Sum of squared weights of all following consumers (on CPU)."""
    total = None
    for lin in layers_list:
        W = lin.weight.data.detach().to(torch.float32)
        norms = (W * W).sum(dim=0).cpu()
        total = norms if total is None else (total + norms)
    return total


def get_mlp_down_proj_norms(layer) -> torch.Tensor:
    return get_consumer_colnorms_generic([layer.mlp.down_proj])


def get_attention_o_proj_head_norms(layer, num_heads, head_dim) -> torch.Tensor:
    W = layer.self_attn.o_proj.weight.data.detach().to(torch.float32).cpu()
    W2 = W * W
    col_norms = W2.sum(dim=0)
    head_norms = col_norms.view(num_heads, head_dim).sum(dim=1)
    return head_norms


def get_embedding_global_consumers(layer) -> torch.Tensor:
    consumers = [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
    ]
    return get_consumer_colnorms_generic(consumers)


# =====================================================================
# Registration & finalization
# =====================================================================


def _get_head_geometry(layer) -> Tuple[int, int]:
    """Return (num_attention_heads, head_dim) for a decoder layer."""
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    if config is not None:
        num_heads = int(config.num_attention_heads)
        head_dim = int(
            getattr(config, "head_dim", None)
            or getattr(attn, "head_dim", None)
            or (attn.o_proj.in_features // num_heads)
        )
    else:
        head_dim = int(getattr(attn, "head_dim", 128))
        num_heads = attn.o_proj.in_features // head_dim
    return num_heads, head_dim


def register_importance_hooks(
    model,
    heads_metric: Optional[str],
    neurons_metric: Optional[str],
    embeddings_metric: Optional[str],
):
    """
    Attach forward hooks based on granular configuration.
    Supported metrics
    -----------------
    heads_metric:
      "magnitude" | "variance_x_consumers" | "cett" | "cett_normalized" | "cett_variance" | None
    neurons_metric:
      "magnitude" | "variance_x_consumers" | "cett" | "cett_normalized" | "cett_variance" | None
    embeddings_metric:
      "magnitude" | "variance_x_consumers" | None
    """
    print("Registering forward hooks...")
    text_model = (
        model.model
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model
    )
    for layer in text_model.layers:
        # ---- 1) Heads ----
        if heads_metric:
            if heads_metric in CETT_METRICS:
                # CETT variants hook on o_proj (needs per-head input to o_proj)
                o_proj = layer.self_attn.o_proj
                _clear_module_stats(o_proj)
                num_heads, head_dim = _get_head_geometry(layer)
                o_proj._cett_num_heads = num_heads
                o_proj._cett_head_dim = head_dim
                if heads_metric == "cett":
                    o_proj.register_forward_hook(attn_o_proj_hook_cett)
                elif heads_metric == "cett_normalized":
                    o_proj.register_forward_hook(attn_o_proj_hook_cett_normalized)
                elif heads_metric == "cett_variance":
                    o_proj.register_forward_hook(attn_o_proj_hook_cett_variance)
            else:
                # Non-CETT head metrics hook on q_proj
                layer.self_attn.q_proj.config = getattr(layer.self_attn, "config", None)
                _clear_module_stats(layer.self_attn.q_proj)
                if heads_metric == "magnitude":
                    layer.self_attn.q_proj.register_forward_hook(head_hook_magnitude)
                elif heads_metric == "variance_x_consumers":
                    layer.self_attn.q_proj.register_forward_hook(head_hook_variance)
                else:
                    raise ValueError(f"Unknown heads_metric: {heads_metric}")
        # ---- 2) Neurons ----
        if neurons_metric:
            if neurons_metric in CETT_METRICS:
                # IMPORTANT REFACTOR:
                # Hook on down_proj instead of layer.mlp to avoid recomputing gate/up.
                down_proj = layer.mlp.down_proj
                _clear_module_stats(down_proj)
                if neurons_metric == "cett":
                    down_proj.register_forward_hook(mlp_down_proj_hook_cett)
                elif neurons_metric == "cett_normalized":
                    down_proj.register_forward_hook(mlp_down_proj_hook_cett_normalized)
                elif neurons_metric == "cett_variance":
                    down_proj.register_forward_hook(mlp_down_proj_hook_cett_variance)
            else:
                # Hooks gate_proj output. Switching to down_proj input (as in the CETT
                # branch) would give better accuracy for the MLP under the magnitude
                # measurement; both branches must change simultaneously.
                gate = layer.mlp.gate_proj  # layer.mlp.gate_proj  # layer.mlp.up_proj
                _clear_module_stats(gate)
                if neurons_metric == "magnitude":
                    gate.register_forward_hook(linear_hook_magnitude)
                elif neurons_metric == "variance_x_consumers":
                    gate.register_forward_hook(linear_hook_variance)
                else:
                    raise ValueError(f"Unknown neurons_metric: {neurons_metric}")
        # ---- 3) Embeddings ----
        if embeddings_metric:
            ln = layer.input_layernorm
            _clear_module_stats(ln)
            if embeddings_metric == "magnitude":
                ln.register_forward_hook(linear_hook_magnitude)
            elif embeddings_metric == "variance_x_consumers":
                ln.register_forward_hook(linear_hook_variance)
            else:
                raise ValueError(f"Unknown embeddings_metric: {embeddings_metric}")
    print("Hooks registered.")


def _finalize_metric(module, metric_type, consumer_norms=None, head_info=None):
    """
    Convert accumulated stats into a final 1D importance vector.
    - magnitude:
        stored as calculated_importance (sum over tokens) + n_sum, returned as per-token mean
    - cett / cett_normalized:
        stored as calculated_importance (sum over tokens) + n_sum, returned as per-token mean
    - cett_variance:
        stored as m1_sum/m2_sum/n_sum over per-token normalized ratios, returned as variance
    - variance_x_consumers:
        stored as m1_sum/m2_sum/n_sum, returned as variance (optionally weighted by consumer norms)
    """
    if metric_type == "magnitude":
        n = max(int(getattr(module, "n_sum", torch.tensor(1)).item()), 1)
        return module.calculated_importance.to(torch.float32) / n
    if metric_type in ("cett", "cett_normalized"):
        n = max(int(getattr(module, "n_sum", torch.tensor(1)).item()), 1)
        return module.calculated_importance.to(torch.float32) / n
    if metric_type == "cett_variance":
        n = max(int(module.n_sum.item()), 1)
        mean = (module.m1_sum / n).to(torch.float32)
        var = (module.m2_sum / n - mean.pow(2)).clamp_min(0.0)
        return var
    if metric_type == "variance_x_consumers":
        n = max(int(module.n_sum.item()), 1)
        mean = (module.m1_sum / n).to(torch.float32)
        var = (module.m2_sum / n - mean.pow(2)).clamp_min(0.0)
        if head_info:
            num_heads, head_dim = head_info
            var_reshaped = var.view(num_heads, head_dim)
            head_var = var_reshaped.mean(dim=1)
            return head_var * consumer_norms if consumer_norms is not None else head_var
        return var * consumer_norms if consumer_norms is not None else var
    return None


def calculate_importance(
    model,
    tokenizer,
    text_data: str,
    max_chunk_len: int,
    heads_metric: Optional[str] = None,
    neurons_metric: Optional[str] = None,
    embeddings_metric: Optional[str] = None,
) -> Dict:
    """
    Run a forward pass over calibration text while hooks accumulate statistics,
    then finalize per-layer scores and sorted indices.
    """
    register_importance_hooks(model, heads_metric, neurons_metric, embeddings_metric)
    print("Tokenizing calibration data and running forward pass...")
    enc = tokenizer(text_data, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    seq_len = input_ids.size(1)
    text_model = (
        model.model
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model
    )
    fwd_device = next(text_model.parameters()).device
    with torch.no_grad():
        for i in tqdm(range(0, seq_len, max_chunk_len), desc="Processing data chunks"):
            end = min(i + max_chunk_len, seq_len)
            if end <= i:
                continue
            chunk = input_ids[:, i:end].to(fwd_device)
            _ = text_model(input_ids=chunk)
    print("Processing accumulated statistics...")
    importance_data = {
        "heads": {"scores": {}, "sorted_indices": {}},
        "neurons": {"scores": {}, "sorted_indices": {}},
        "embeddings": {"scores": {}, "sorted_indices": {}},
        "embeddings_global": {"scores": None, "sorted_indices": None},
    }
    hidden_size = int(model.config.hidden_size)
    global_emb_scores = torch.zeros(hidden_size, dtype=torch.float32, device="cpu")
    for i, layer in enumerate(text_model.layers):
        layer_key = f"layer_{i}"
        # --- HEADS ---
        if heads_metric:
            num_heads, head_dim = _get_head_geometry(layer)
            if heads_metric in CETT_METRICS:
                # Stats live on o_proj
                o = layer.self_attn.o_proj
                scores = _finalize_metric(o, heads_metric)
            else:
                # Stats live on q_proj
                q = layer.self_attn.q_proj
                cons = None
                if heads_metric == "variance_x_consumers":
                    cons = get_attention_o_proj_head_norms(layer, num_heads, head_dim)
                scores = _finalize_metric(
                    q,
                    heads_metric,
                    consumer_norms=cons,
                    head_info=(num_heads, head_dim),
                )
            importance_data["heads"]["scores"][layer_key] = scores
            importance_data["heads"]["sorted_indices"][layer_key] = torch.argsort(
                scores, descending=True
            )
        # --- NEURONS ---
        if neurons_metric:
            if neurons_metric in CETT_METRICS:
                # Stats live on down_proj (hooked directly)
                down = layer.mlp.down_proj
                scores = _finalize_metric(down, neurons_metric)
            else:
                gate = layer.mlp.gate_proj  # layer.mlp.gate_proj  # layer.mlp.up_proj
                cons = None
                if neurons_metric == "variance_x_consumers":
                    cons = get_mlp_down_proj_norms(layer)
                scores = _finalize_metric(gate, neurons_metric, consumer_norms=cons)
            importance_data["neurons"]["scores"][layer_key] = scores
            importance_data["neurons"]["sorted_indices"][layer_key] = torch.argsort(
                scores, descending=True
            )
        # --- EMBEDDINGS ---
        if embeddings_metric:
            ln = layer.input_layernorm
            cons = None
            if embeddings_metric == "variance_x_consumers":
                cons = get_embedding_global_consumers(layer)
            scores = _finalize_metric(ln, embeddings_metric, consumer_norms=cons)
            importance_data["embeddings"]["scores"][layer_key] = scores
            importance_data["embeddings"]["sorted_indices"][layer_key] = torch.argsort(
                scores, descending=True
            )
            global_emb_scores += scores
    if embeddings_metric:
        if global_emb_scores.abs().sum().item() == 0:
            print("[WARN] Global embedding scores are zero.")
        importance_data["embeddings_global"]["scores"] = global_emb_scores
        importance_data["embeddings_global"]["sorted_indices"] = torch.argsort(
            global_emb_scores, descending=True
        )
    return importance_data


# =====================================================================
# CETT tail curve helpers
# =====================================================================


def _sanitize_prune_fracs(fracs: Optional[Iterable[float]]) -> List[float]:
    if not fracs:
        return [0.25, 0.50, 0.75, 0.90]
    out = []
    for f in fracs:
        f = float(f)
        if f > 1.0:
            f = f / 100.0
        out.append(min(max(f, 0.0), 1.0))
    return sorted(set(out))


def cett_tail_cumsum(scores: torch.Tensor) -> torch.Tensor:
    """
    Given per-unit cett_normalized scores (mean ratios), return the cumulative sum
    of the *smallest* scores. This is the CETT tail error *upper bound curve*.
    curve[k-1] = sum of k-smallest scores
    """
    s = scores.detach().to(torch.float32).cpu().flatten()
    if s.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    s_sorted, _ = torch.sort(s, descending=False)
    return torch.cumsum(s_sorted, dim=0)


def cett_tail_curve_points(
    scores: torch.Tensor,
    prune_fracs: Optional[Iterable[float]] = None,
) -> Dict[float, float]:
    """
    Compute tail error upper bound at selected prune fractions.
    Returns: {prune_frac -> cett_error_ub}
    """
    prune_fracs = _sanitize_prune_fracs(prune_fracs)
    curve = cett_tail_cumsum(scores)
    n = int(curve.numel())
    out: Dict[float, float] = {}
    for frac in prune_fracs:
        k = int(math.floor(frac * n))
        if k <= 0:
            out[frac] = 0.0
        else:
            out[frac] = float(curve[k - 1].item())
    return out


def cett_max_prune_under_bound(
    scores: torch.Tensor, error_bound: float
) -> Tuple[float, int, float]:
    """
    Find the maximum prune fraction such that the tail error upper bound <= error_bound.
    Returns (frac, k, achieved_error).
    """
    curve = cett_tail_cumsum(scores)
    n = int(curve.numel())
    if n == 0:
        return 0.0, 0, 0.0
    mask = curve <= float(error_bound)
    k = int(mask.sum().item())  # number prunable
    frac = k / n
    achieved = float(curve[k - 1].item()) if k > 0 else 0.0
    return frac, k, achieved
