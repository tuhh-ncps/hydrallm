# sorting_llm/core.py
"""
Pure reorder pipeline:
- importance computation (via sorting_llm.common.calculate_importance)
- tail-curve reporting for CETT-normalized scores
- permutation utilities
- reorder_and_save_model
- run_sort_llm orchestrator

Attention head reordering semantics (as requested)
-------------------------------------------------
Let:
  num_heads = number of Q heads
  num_kv    = number of KV heads (= num_key_value_heads)

MHA (standard multi-head): num_kv == num_heads
  - Reorder heads individually by score.
  - Permute Q, K, V, and O consistently.

MQA (multi-query): num_kv == 1 and num_heads > 1
  - Reorder Q heads individually by score (matches original behavior).
  - Do NOT permute K/V (there is only one KV head).

GQA (grouped-query): 1 < num_kv < num_heads
  - Reorder KV groups by group importance.
  - Move Q heads as blocks with their KV group.
  - Do NOT reorder Q heads within a group.
  - Permute K/V groups and permute Q and O accordingly.
"""

import json
import os
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from .common import (
    CETT_METRICS,
    load_model_and_tokenizer,
    calculate_importance,
    setup_sort_output_dir_and_config,
    cett_tail_curve_points,
    cett_max_prune_under_bound,
)


# ---------------------- Text summary for importance ---------------------- #
def analyze_importance(importance_data: Dict, top_k: int = 3):
    print("\n--- Importance Analysis Summary ---")

    if importance_data["heads"]["scores"]:
        print("\n[Attention Head Importance]")
        for layer, indices in importance_data["heads"]["sorted_indices"].items():
            scores = importance_data["heads"]["scores"][layer]
            print(f"  {layer}:")
            print(f"    - Order (most->least): {indices.tolist()}")
            print(f"    - Scores: {[float(f'{s:.6f}') for s in scores]}")

    if importance_data["neurons"]["scores"]:
        print("\n[MLP Neuron Importance]")
        for layer, indices in importance_data["neurons"]["sorted_indices"].items():
            scores = importance_data["neurons"]["scores"][layer]
            print(f"  {layer}:")
            print(f"    - Top {top_k}: {indices[:top_k].tolist()}")
            print(
                f"    - Top {top_k} scores: {[float(f'{scores[i]:.6f}') for i in indices[:top_k]]}"
            )
            print(f"    - Bottom {top_k}: {indices[-top_k:].tolist()}")
            print(
                f"    - Bottom {top_k} scores: {[float(f'{scores[i]:.6f}') for i in indices[-top_k:]]}"
            )

    gi = importance_data.get("embeddings_global", {}).get("sorted_indices")
    gs = importance_data.get("embeddings_global", {}).get("scores")
    if gi is not None:
        print("\n[Global Embedding-dimension Importance]")
        print(f"  - Top {top_k} dims: {gi[:top_k].tolist()}")
        if gs is not None:
            print(
                f"  - Top {top_k} scores: {[float(f'{gs[i]:.6f}') for i in gi[:top_k]]}"
            )
        print(f"  - Bottom {top_k} dims: {gi[-top_k:].tolist()}")
        if gs is not None:
            print(
                f"  - Bottom {top_k} scores: {[float(f'{gs[i]:.6f}') for i in gi[-top_k:]]}"
            )

    print("\n--- Summary Complete ---\n")


# =====================================================================
# CETT tail curve reporting (for cett_normalized scores)
# =====================================================================


def _report_cett_tail_curve(
    *,
    scores_by_layer: Dict[str, torch.Tensor],
    prune_fracs: List[float],
    error_bound: Optional[float],
    title: str,
    per_layer: bool = True,
) -> Dict:
    if not scores_by_layer:
        return {}

    all_scores = torch.cat(
        [v.to(torch.float32).cpu().flatten() for v in scores_by_layer.values()]
    )

    global_points = cett_tail_curve_points(all_scores, prune_fracs=prune_fracs)
    global_max = None
    if error_bound is not None:
        frac, k, achieved = cett_max_prune_under_bound(
            all_scores, error_bound=float(error_bound)
        )
        global_max = {"prune_frac": frac, "k": k, "achieved_error": achieved}

    print(f"\n[CETT Tail Curve — {title}]")
    print("Pruned fraction -> tail error upper bound")
    for frac in prune_fracs:
        print(f"  {frac*100:6.1f}% -> {global_points.get(frac, 0.0):.6f}")

    if global_max is not None:
        print(
            f"Max prunable under error_bound={float(error_bound):.3f}: "
            f"{global_max['prune_frac']*100:.1f}% (k={global_max['k']}, err={global_max['achieved_error']:.6f})"
        )

    per_layer_points = {}
    per_layer_max = {}
    if per_layer:
        print("\nPer-layer tail error (upper bound):")
        for layer_key, s in scores_by_layer.items():
            pts = cett_tail_curve_points(s, prune_fracs=prune_fracs)
            per_layer_points[layer_key] = {str(k): float(v) for k, v in pts.items()}

            line = (
                "  "
                + layer_key
                + " | "
                + "  ".join(
                    [f"{frac*100:5.1f}%->{pts[frac]:.6f}" for frac in prune_fracs]
                )
            )
            print(line)

            if error_bound is not None:
                frac, k, achieved = cett_max_prune_under_bound(
                    s, error_bound=float(error_bound)
                )
                per_layer_max[layer_key] = {
                    "prune_frac": float(frac),
                    "k": int(k),
                    "achieved_error": float(achieved),
                }

    return {
        "title": title,
        "prune_fracs": [float(f) for f in prune_fracs],
        "error_bound": None if error_bound is None else float(error_bound),
        "global": {
            "points": {str(k): float(v) for k, v in global_points.items()},
            "max_prune_under_bound": global_max,
        },
        "per_layer": {
            "points": per_layer_points,
            "max_prune_under_bound": per_layer_max if error_bound is not None else None,
        },
    }


# =====================================================================
# Permutation helpers
# =====================================================================


def permute_linear_columns_(linear: nn.Linear, perm: torch.LongTensor):
    w = linear.weight.data
    if w.size(1) != perm.numel():
        raise ValueError(f"perm length ({perm.numel()}) != in_features ({w.size(1)})")
    linear.weight.data = w.index_select(dim=1, index=perm.to(w.device))


def permute_linear_rows_(linear: nn.Linear, perm: torch.LongTensor):
    w = linear.weight.data
    if w.size(0) != perm.numel():
        raise ValueError(f"perm length ({perm.numel()}) != out_features ({w.size(0)})")
    linear.weight.data = w.index_select(dim=0, index=perm.to(w.device))
    if linear.bias is not None:
        b = linear.bias.data
        linear.bias.data = b.index_select(dim=0, index=perm.to(b.device))


def permute_rmsnorm_(norm_module, perm: torch.LongTensor):
    if not hasattr(norm_module, "weight") or norm_module.weight is None:
        return
    w = norm_module.weight.data
    if w.numel() != perm.numel():
        raise ValueError(f"perm length ({perm.numel()}) != norm size ({w.numel()})")
    norm_module.weight.data = w.index_select(dim=0, index=perm.to(w.device))


def permute_embedding_columns_(embedding: nn.Embedding, perm: torch.LongTensor):
    w = embedding.weight.data
    if w.size(1) != perm.numel():
        raise ValueError(f"perm length ({perm.numel()}) != embedding dim ({w.size(1)})")
    embedding.weight.data = w.index_select(dim=1, index=perm.to(w.device))


# =====================================================================
# Attention head reordering (MQA/GQA/MHA semantics)
# =====================================================================


def _get_attn_layout(layer) -> Tuple[int, int, int, int]:
    attn = layer.self_attn
    cfg = getattr(attn, "config", None)

    hidden_size = int(attn.q_proj.in_features)
    num_heads = int(
        getattr(cfg, "num_attention_heads", None)
        or getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
    )
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(
        getattr(attn, "head_dim", None) or (attn.q_proj.out_features // num_heads)
    )
    return hidden_size, num_heads, num_kv, head_dim


def _permute_q_heads_(
    proj: nn.Linear,
    head_order: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
):
    W = proj.weight.data.view(num_heads, head_dim, hidden_size)
    Wp = W.index_select(dim=0, index=head_order.to(W.device))
    proj.weight.data.copy_(Wp.reshape(num_heads * head_dim, hidden_size))

    if proj.bias is not None:
        b = proj.bias.data.view(num_heads, head_dim)
        bp = b.index_select(dim=0, index=head_order.to(b.device))
        proj.bias.data.copy_(bp.reshape(num_heads * head_dim))


def _permute_kv_heads_(
    proj: nn.Linear,
    kv_order: torch.LongTensor,
    num_kv: int,
    head_dim: int,
    hidden_size: int,
):
    W = proj.weight.data.view(num_kv, head_dim, hidden_size)
    Wp = W.index_select(dim=0, index=kv_order.to(W.device))
    proj.weight.data.copy_(Wp.reshape(num_kv * head_dim, hidden_size))

    if proj.bias is not None:
        b = proj.bias.data.view(num_kv, head_dim)
        bp = b.index_select(dim=0, index=kv_order.to(b.device))
        proj.bias.data.copy_(bp.reshape(num_kv * head_dim))


def _permute_o_by_q_heads_(
    o_proj: nn.Linear,
    head_order: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
):
    W = o_proj.weight.data.view(hidden_size, num_heads, head_dim)
    Wp = W.index_select(dim=1, index=head_order.to(W.device))
    o_proj.weight.data.copy_(Wp.reshape(hidden_size, num_heads * head_dim))


def _reorder_attention_layer_(layer, head_scores: torch.Tensor):
    attn = layer.self_attn
    hidden_size, num_heads, num_kv, head_dim = _get_attn_layout(layer)

    scores = head_scores.detach().to(torch.float32).cpu()
    if scores.numel() != num_heads:
        print(
            f"[WARN] Head score length {scores.numel()} != num_heads {num_heads}. Skipping."
        )
        return

    # MHA: reorder Q,K,V,O by per-head importance
    if num_kv == num_heads:
        perm = torch.argsort(scores, descending=True).to(torch.long)
        _permute_q_heads_(attn.q_proj, perm, num_heads, head_dim, hidden_size)
        _permute_kv_heads_(attn.k_proj, perm, num_heads, head_dim, hidden_size)
        _permute_kv_heads_(attn.v_proj, perm, num_heads, head_dim, hidden_size)
        _permute_o_by_q_heads_(attn.o_proj, perm, num_heads, head_dim, hidden_size)
        return

    # MQA: reorder Q only (original behavior)
    if num_kv == 1:
        perm = torch.argsort(scores, descending=True).to(torch.long)
        _permute_q_heads_(attn.q_proj, perm, num_heads, head_dim, hidden_size)
        _permute_o_by_q_heads_(attn.o_proj, perm, num_heads, head_dim, hidden_size)
        return

    # GQA: reorder KV heads AND move corresponding Q head blocks; no within-block reorder
    if num_heads % num_kv != 0:
        print(
            f"[WARN] num_heads={num_heads} not divisible by num_kv={num_kv}. Skipping."
        )
        return

    q_per_kv = num_heads // num_kv
    s2 = scores.view(num_kv, q_per_kv)

    group_scores = s2.sum(dim=1)  # group importance
    group_perm = torch.argsort(group_scores, descending=True).to(torch.long)

    q_head_order = torch.cat(
        [
            torch.arange(int(g) * q_per_kv, (int(g) + 1) * q_per_kv, dtype=torch.long)
            for g in group_perm
        ],
        dim=0,
    )

    _permute_q_heads_(attn.q_proj, q_head_order, num_heads, head_dim, hidden_size)
    _permute_o_by_q_heads_(attn.o_proj, q_head_order, num_heads, head_dim, hidden_size)
    _permute_kv_heads_(attn.k_proj, group_perm, num_kv, head_dim, hidden_size)
    _permute_kv_heads_(attn.v_proj, group_perm, num_kv, head_dim, hidden_size)


# =====================================================================
# Reordering and saving
# =====================================================================


def reorder_and_save_model(
    model,
    tokenizer,
    importance_data: Dict,
    save_dir: str,
    model_save_subfolder: str = "model",
):
    model = model.to("cpu")
    model.eval()

    text_model = (
        model.model
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model
    )

    # 1) Global embedding permutation (optional)
    emb_perm = importance_data.get("embeddings_global", {}).get("sorted_indices", None)
    if emb_perm is not None:
        emb_perm = emb_perm.to(torch.long).cpu()
        print("[INFO] Applying Global Embedding Permutation.")

        emb = text_model.embed_tokens
        permute_embedding_columns_(emb, emb_perm)

        for layer in text_model.layers:
            permute_rmsnorm_(layer.input_layernorm, emb_perm)
            if hasattr(layer, "post_attention_layernorm"):
                permute_rmsnorm_(layer.post_attention_layernorm, emb_perm)
            if hasattr(layer, "pre_feedforward_layernorm"):
                permute_rmsnorm_(layer.pre_feedforward_layernorm, emb_perm)
            if hasattr(layer, "post_feedforward_layernorm"):
                permute_rmsnorm_(layer.post_feedforward_layernorm, emb_perm)

            permute_linear_columns_(layer.self_attn.q_proj, emb_perm)
            permute_linear_columns_(layer.self_attn.k_proj, emb_perm)
            permute_linear_columns_(layer.self_attn.v_proj, emb_perm)
            permute_linear_rows_(layer.self_attn.o_proj, emb_perm)

            permute_linear_columns_(layer.mlp.gate_proj, emb_perm)
            permute_linear_columns_(layer.mlp.up_proj, emb_perm)
            permute_linear_rows_(layer.mlp.down_proj, emb_perm)

        if hasattr(text_model, "norm"):
            permute_rmsnorm_(text_model.norm, emb_perm)

        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            try:
                tied = (
                    text_model.embed_tokens.weight.data.data_ptr()
                    == model.lm_head.weight.data.data_ptr()
                )
            except Exception:
                tied = False
            if not tied:
                permute_linear_columns_(model.lm_head, emb_perm)

    # 2) Layerwise attention head and MLP reordering
    with torch.no_grad():
        for i, layer in enumerate(text_model.layers):
            layer_key = f"layer_{i}"

            head_scores = importance_data["heads"]["scores"].get(layer_key)
            if head_scores is not None:
                _reorder_attention_layer_(layer, head_scores)

            neuron_order = importance_data["neurons"]["sorted_indices"].get(layer_key)
            if neuron_order is not None:
                neuron_order = neuron_order.detach().cpu().to(torch.long)
                permute_linear_rows_(layer.mlp.gate_proj, neuron_order)
                permute_linear_rows_(layer.mlp.up_proj, neuron_order)
                permute_linear_columns_(layer.mlp.down_proj, neuron_order)

    final_model_dir = os.path.join(save_dir, model_save_subfolder)
    os.makedirs(final_model_dir, exist_ok=True)

    model.save_pretrained(final_model_dir)
    model.config.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"[INFO] Reordered model saved to: {final_model_dir}")


# =====================================================================
# Orchestrator
# =====================================================================


def run_sort_llm(cfg: Dict):
    sort_cfg = cfg["sorting"]

    model_id = sort_cfg["model_id"]
    device = sort_cfg["device"]
    dtype = sort_cfg["dtype"]
    device_map = sort_cfg.get("device_map", "auto")
    model_save_subfolder = sort_cfg.get("model_save_subfolder", "model")

    final_save_dir = setup_sort_output_dir_and_config(cfg)

    calibration_path = sort_cfg["calibration_file"]
    max_chunk_len = int(sort_cfg["chunk_len"])
    top_k = int(sort_cfg.get("top_k", 3))
    postcheck_enabled = bool(sort_cfg.get("postcheck", True))

    heads_metric = sort_cfg.get("heads_metric", None)
    neurons_metric = sort_cfg.get("neurons_metric", None)
    embeddings_metric = sort_cfg.get("embeddings_metric", None)

    # Plot scales to be used by postcheck / single analysis
    plot_scales = sort_cfg.get("plot_scales", None)

    # CETT tail curve config
    tail_curve_enabled = bool(sort_cfg.get("cett_tail_curve", True))
    cett_error_bound = sort_cfg.get("cett_error_bound", 0.2)
    prune_fracs = sort_cfg.get("cett_tail_prune_fracs", [0.25, 0.50, 0.75, 0.90])
    prune_fracs = [float(x) for x in prune_fracs]

    print(f"\n[INFO] Importance metrics configuration:")
    print(f"  - heads_metric:      {heads_metric}")
    print(f"  - neurons_metric:    {neurons_metric}")
    print(f"  - embeddings_metric: {embeddings_metric}")

    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    print(f"Loading pretrained model: {model_id}...")
    model, tokenizer = load_model_and_tokenizer(model_id, device, dtype, device_map)

    with open(calibration_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    importance_data = calculate_importance(
        model,
        tokenizer,
        text_data,
        max_chunk_len=max_chunk_len,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
        embeddings_metric=embeddings_metric,
    )

    analyze_importance(importance_data, top_k=top_k)

    # CETT tail curve reporting (only for cett_normalized)
    tail_reports = {}
    if tail_curve_enabled:
        if neurons_metric == "cett_normalized":
            tail_reports["neurons"] = _report_cett_tail_curve(
                scores_by_layer=importance_data["neurons"]["scores"],
                prune_fracs=prune_fracs,
                error_bound=(
                    float(cett_error_bound) if cett_error_bound is not None else None
                ),
                title="MLP Neurons (cett_normalized)",
                per_layer=True,
            )

        if heads_metric == "cett_normalized":
            tail_reports["heads"] = _report_cett_tail_curve(
                scores_by_layer=importance_data["heads"]["scores"],
                prune_fracs=prune_fracs,
                error_bound=(
                    float(cett_error_bound) if cett_error_bound is not None else None
                ),
                title="Attention Heads (cett_normalized)",
                per_layer=True,
            )

        if tail_reports:
            tail_path = os.path.join(final_save_dir, "cett_tail_curve.json")
            with open(tail_path, "w", encoding="utf-8") as f:
                json.dump(tail_reports, f, indent=2)
            print(f"[INFO] Saved CETT tail curve report to: {tail_path}")

    # Save raw importance tensors
    imp_out_file = os.path.join(final_save_dir, "importance_data.pth")
    torch.save(importance_data, imp_out_file)
    print(f"\nSuccess! Importance data saved to: {imp_out_file}")

    # Reorder + save
    reorder_and_save_model(
        model,
        tokenizer,
        importance_data,
        save_dir=final_save_dir,
        model_save_subfolder=model_save_subfolder,
    )

    # Optional postcheck
    if postcheck_enabled:
        from .post_analyze import run_postcheck as _run_postcheck

        _run_postcheck(
            model_ref=model_id,
            reordered_dir=os.path.join(final_save_dir, model_save_subfolder),
            calibration_path=calibration_path,
            out_dir_original=os.path.join(final_save_dir, "postcheck_original"),
            out_dir_reordered=os.path.join(final_save_dir, "postcheck_reordered"),
            device=device,
            chunk_tokens=max_chunk_len,
            dtype_str=dtype,
            heads_metric=heads_metric,
            neurons_metric=neurons_metric,
            embeddings_metric=embeddings_metric,
            cett_tail_prune_fracs=prune_fracs,
            cett_error_bound=(
                float(cett_error_bound) if cett_error_bound is not None else None
            ),
            plot_scales=plot_scales,
        )
