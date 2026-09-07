# sorting_llm/post_analyze.py
from __future__ import annotations
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from .common import (
    load_model_and_tokenizer_single_device,
    calculate_importance,
    cett_tail_curve_points,
    cett_max_prune_under_bound,
)

# Optional plotting deps
try:
    import numpy as np  # type: ignore
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.colors as mcolors  # type: ignore

    _PLOTTING_AVAILABLE = True
except Exception:
    _PLOTTING_AVAILABLE = False
try:
    import seaborn as sns  # type: ignore

    _SEABORN_AVAILABLE = True
except Exception:
    _SEABORN_AVAILABLE = False
try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False
# ============================================================
# Per-plot font-size configuration
# ============================================================
# Each plot type has its own set of font-size knobs so you can
# tune them independently.  Plots where the fonts were too large
# (e.g. embedding bar / line) now start smaller.
# PLOT_OUTPUT_FORMAT = "pdf"
MAX_HEATMAP_WIDTH = 9.0  # 12.0
# --- Heads heatmap ---
# HEADS_HEATMAP_AXIS_LABEL = 20
# HEADS_HEATMAP_TICK_LABEL = 16
# HEADS_HEATMAP_COLORBAR_LBL = 18
# HEADS_HEATMAP_COLORBAR_TCK = 16
# HEADS_HEATMAP_ANNOT = 16  # base; auto-scaled in _annot_fontsize
HEADS_HEATMAP_AXIS_LABEL = 36
HEADS_HEATMAP_TICK_LABEL = 28
HEADS_HEATMAP_COLORBAR_LBL = 28
HEADS_HEATMAP_COLORBAR_TCK = 28
HEADS_HEATMAP_ANNOT = 20  # base; auto-scaled in _annot_fontsize
# --- Heads per-layer bar ---
HEADS_BAR_AXIS_LABEL = 16
HEADS_BAR_TICK_LABEL = 14
HEADS_BAR_XTICK = 14
# --- Neurons line / sorted / overlay ---
NEURONS_LINE_AXIS_LABEL = 14
NEURONS_LINE_TICK_LABEL = 12
NEURONS_LINE_LEGEND = 13
# --- Neurons group heatmap ---
# NEURONS_GRP_HEATMAP_AXIS_LABEL = 20
# NEURONS_GRP_HEATMAP_TICK_LABEL = 16
# NEURONS_GRP_HEATMAP_COLORBAR_LBL = 18
# NEURONS_GRP_HEATMAP_COLORBAR_TCK = 16
# NEURONS_GRP_HEATMAP_ANNOT = 16
NEURONS_GRP_HEATMAP_AXIS_LABEL = 40
NEURONS_GRP_HEATMAP_TICK_LABEL = 32
NEURONS_GRP_HEATMAP_COLORBAR_LBL = 36
NEURONS_GRP_HEATMAP_COLORBAR_TCK = 32
NEURONS_GRP_HEATMAP_ANNOT = 20
# --- Embeddings global line ---
EMB_LINE_AXIS_LABEL = 20
EMB_LINE_TICK_LABEL = 18
# --- Embeddings grouped bar ---
EMB_BAR_AXIS_LABEL = 16
EMB_BAR_TICK_LABEL = 14
EMB_BAR_XTICK = 14
# ============================================================
# Palette + style
# ============================================================
IBM_YELLOW = "#ffb000"
IBM_ORANGE = "#fe6100"
IBM_PINK = "#dc267f"
IBM_PURPLE = "#785ef0"
IBM_BLUE = "#648fff"
IBM_SEQ_LOW_TO_HIGH = [
    IBM_BLUE,
    IBM_ORANGE,
]  # [IBM_BLUE, IBM_PURPLE, IBM_PINK, IBM_ORANGE, IBM_YELLOW]
IBM_DISCRETE = [
    IBM_BLUE,
    IBM_PURPLE,
    IBM_PINK,
    IBM_ORANGE,
    IBM_YELLOW,
]


def _apply_matplotlib_style():
    if not _PLOTTING_AVAILABLE:
        return
    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
                "Times New Roman",
                "serif",
            ],
        }
    )


def _ibm_sequential_cmap():
    if not _PLOTTING_AVAILABLE:
        return None
    return mcolors.LinearSegmentedColormap.from_list(
        "ibm_custom_palette", IBM_SEQ_LOW_TO_HIGH, N=256
    )


def _despine_xy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.grid(False)


def _palette_colors_for_n(n: int, seed: int = 1337) -> List[Any]:
    out: List[Any] = []
    rng = random.Random(seed)
    for i in range(n):
        if i < len(IBM_DISCRETE):
            out.append(IBM_DISCRETE[i])
        else:
            out.append((rng.random(), rng.random(), rng.random()))
    return out


_apply_matplotlib_style()
# ============================================================
# Plot scale config helpers
# ============================================================


def _scale_cfg(plot_scales: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not isinstance(plot_scales, dict):
        return {}
    v = plot_scales.get(key, {}) or {}
    return v if isinstance(v, dict) else {}


def _maybe_float(x):
    """
    Convert config values to float if possible.
    Treat YAML-ish null sentinels as None even if user wrote them as strings:
      "None", "none", "null", "~", ""  -> None
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("none", "null", "~", ""):
            return None
        try:
            return float(s)
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def _maybe_bool(x, default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
    return bool(x)


# ============================================================
# Importance label helper
# ============================================================


def _importance_label_for_metric(metric: Optional[str]) -> str:
    """Return the colorbar / legend label based on the importance metric name."""
    if metric is None:
        return "Importance"
    m = metric.strip().lower()
    if m.startswith("cett"):
        return "CETT importance"
    if m.startswith("variance_x_consumer") or m.startswith("vxc"):
        return "VxC importance"
    if m.startswith("mag"):
        return "Mag importance"
    return "Importance"


# ============================================================
# IO
# ============================================================


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_config(
    out_dir: str, cfg: Dict[str, Any], filename: str = "postcheck_config.yaml"
):
    _ensure_dir(out_dir)
    if _YAML_AVAILABLE:
        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    else:
        _write_json(os.path.join(out_dir, filename.replace(".yaml", ".json")), cfg)


# ============================================================
# Model/layout helpers
# ============================================================


def _get_text_model(model):
    return (
        model.model
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model
    )


def _layer_key_to_int(layer_key: str) -> Optional[int]:
    try:
        if layer_key.startswith("layer_"):
            return int(layer_key.split("_", 1)[1])
        return int(layer_key)
    except Exception:
        return None


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


# ============================================================
# Identity checks
# ============================================================


def _is_identity_order_1d(sorted_indices: torch.Tensor) -> bool:
    n = int(sorted_indices.numel())
    return bool(torch.equal(sorted_indices.detach().cpu(), torch.arange(n)))


def _identity_checks(model, importance_data: Dict):
    text_model = _get_text_model(model)
    print("\nIdentity-order checks:")
    if importance_data["heads"]["sorted_indices"]:
        print("\n  [Heads]")
        for i, layer in enumerate(text_model.layers):
            lk = f"layer_{i}"
            order = importance_data["heads"]["sorted_indices"].get(lk)
            scores = importance_data["heads"]["scores"].get(lk)
            if order is None or scores is None:
                continue
            _, num_heads, num_kv, _ = _get_attn_layout(layer)
            # MHA or MQA => full identity check
            if num_kv == num_heads or num_kv == 1:
                ok = _is_identity_order_1d(order)
                print(f"    {lk} heads: identity={ok}  order={order.tolist()}")
                continue
            # GQA => group identity only
            if num_heads % num_kv != 0:
                print(f"    {lk} heads: [WARN] incompatible layout; skipping.")
                continue
            q_per_kv = num_heads // num_kv
            s2 = scores.detach().to(torch.float32).cpu().view(num_kv, q_per_kv)
            group_scores = s2.sum(dim=1)
            group_order = torch.argsort(group_scores, descending=True)
            ok_group = _is_identity_order_1d(group_order)
            print(
                f"    {lk} heads (GQA): group_identity={ok_group}  group_order={group_order.tolist()}"
            )
    if importance_data["neurons"]["sorted_indices"]:
        print("\n  [Neurons]")
        for lk, order in importance_data["neurons"]["sorted_indices"].items():
            ok = _is_identity_order_1d(order)
            print(
                f"    {lk} neurons: identity={ok}  top5={order[:5].tolist()}  last5={order[-5:].tolist()}"
            )


# ============================================================
# Tail curve reporting
# ============================================================


def _concat_scores(scores_by_layer: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not scores_by_layer:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(
        [v.detach().to(torch.float32).cpu().flatten() for v in scores_by_layer.values()]
    )


def _report_tail_curve(
    *,
    scores_by_layer: Dict[str, torch.Tensor],
    prune_fracs: Optional[Iterable[float]],
    error_bound: Optional[float],
    out_dir: str,
    title: str,
) -> Dict:
    _ensure_dir(out_dir)
    all_scores = _concat_scores(scores_by_layer)
    if all_scores.numel() == 0:
        return {}
    points = cett_tail_curve_points(all_scores, prune_fracs=prune_fracs)
    maxp = None
    if error_bound is not None:
        frac, k, achieved = cett_max_prune_under_bound(all_scores, float(error_bound))
        maxp = {
            "prune_frac": float(frac),
            "k": int(k),
            "achieved_error": float(achieved),
        }
    report = {
        "title": title,
        "error_bound": None if error_bound is None else float(error_bound),
        "global": {
            "points": {str(float(k)): float(v) for k, v in points.items()},
            "max_prune_under_bound": maxp,
        },
    }
    _write_json(os.path.join(out_dir, "cett_tail_curve.json"), report)
    return report


# ============================================================
# Heatmap helper (annotated)
# ============================================================


def _vmin_vmax(arr: np.ndarray) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    return vmin, vmax


def _heatmap_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    # fig_w = max(14.0, 0.75 * float(n_cols) + 7.0)
    fig_w = max(MAX_HEATMAP_WIDTH, 0.5 * float(n_cols) + 7.0)
    fig_h = max(8.0, 0.35 * float(n_rows) + 2.5)
    return fig_w, fig_h


def _annot_fontsize(n_rows: int, n_cols: int, base: int) -> int:
    """Compute heatmap annotation font size, using the given base and scaling down
    for large matrices."""
    fs = int(round(base - 0.12 * n_cols - 0.05 * n_rows))
    return max(4, min(base, fs))


def _luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _set_heatmap_annot_colors(ax, data: np.ndarray, cmap, vmin: float, vmax: float):
    if not hasattr(ax, "texts") or cmap is None:
        return
    denom = (vmax - vmin) if not math.isclose(vmax, vmin) else 1.0
    for txt in ax.texts:
        x, y = txt.get_position()
        j = int(round(x - 0.5))
        i = int(round(y - 0.5))
        if i < 0 or j < 0 or i >= data.shape[0] or j >= data.shape[1]:
            continue
        val = data[i, j]
        if not np.isfinite(val):
            continue
        t = float((val - vmin) / denom)
        t = max(0.0, min(1.0, t))
        rgba = cmap(t)
        lum = _luminance((rgba[0], rgba[1], rgba[2]))
        txt.set_color("black" if lum > 0.6 else "white")


def _draw_heatmap_with_annot(
    *,
    mat: np.ndarray,
    mask: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    x_ticks: List[int],
    y_ticks: List[int],
    out_path: str,
    fmt: str = ".3f",
    vmin_override: Optional[float] = None,
    vmax_override: Optional[float] = None,
    debug_name: Optional[str] = None,
    # per-plot font overrides (fall back to heads-heatmap globals)
    font_axis_label: Optional[int] = None,
    font_tick_label: Optional[int] = None,
    font_colorbar_label: Optional[int] = None,
    font_colorbar_tick: Optional[int] = None,
    font_annot_base: Optional[int] = None,
    colorbar_label: str = "Importance",
):
    fa_axis = font_axis_label or HEADS_HEATMAP_AXIS_LABEL
    fa_tick = font_tick_label or HEADS_HEATMAP_TICK_LABEL
    fa_cblbl = font_colorbar_label or HEADS_HEATMAP_COLORBAR_LBL
    fa_cbtck = font_colorbar_tick or HEADS_HEATMAP_COLORBAR_TCK
    fa_annot = font_annot_base or HEADS_HEATMAP_ANNOT
    cmap = _ibm_sequential_cmap()
    vmin_auto, vmax_auto = _vmin_vmax(mat if mask is None else mat[~mask])
    vmin = vmin_auto if vmin_override is None else float(vmin_override)
    vmax = vmax_auto if vmax_override is None else float(vmax_override)
    if vmax <= vmin:
        vmax = vmin + 1e-8
    if debug_name and (vmin_override is not None or vmax_override is not None):
        print(f"[plot-scales] {debug_name}: using vmin={vmin} vmax={vmax}")
    fig_w, fig_h = _heatmap_figsize(mat.shape[0], mat.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fs = _annot_fontsize(mat.shape[0], mat.shape[1], fa_annot)
    if _SEABORN_AVAILABLE:
        hm = sns.heatmap(
            mat,
            mask=mask,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt=fmt,
            annot_kws={"fontsize": fs},
            xticklabels=x_ticks,
            yticklabels=y_ticks,
            cbar=True,
            cbar_kws={"label": colorbar_label, "pad": 0.04},
        )
        cbar = hm.collections[0].colorbar
        cbar.set_label(colorbar_label, fontsize=fa_cblbl)
        cbar.ax.tick_params(labelsize=fa_cbtck)
        _set_heatmap_annot_colors(ax, mat, cmap, vmin, vmax)
    else:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels([str(i) for i in x_ticks])
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([str(i) for i in y_ticks])
        # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=fa_cblbl)
        cbar.ax.tick_params(labelsize=fa_cbtck)
    ax.set_xlabel(x_label, fontsize=fa_axis)
    ax.set_ylabel(y_label, fontsize=fa_axis)
    ax.tick_params(axis="both", which="major", labelsize=fa_tick)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(f"{out_path}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plots (current state only)
# ============================================================


def _plot_heads_heatmap(
    head_scores: Dict[str, torch.Tensor],
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
    heads_metric: Optional[str] = None,
):
    if not (_PLOTTING_AVAILABLE and head_scores):
        return
    items: List[Tuple[int, np.ndarray]] = []
    for lk, s in head_scores.items():
        li = _layer_key_to_int(lk)
        if li is None:
            continue
        items.append((li, s.detach().to(torch.float32).cpu().flatten().numpy()))
    items.sort(key=lambda x: x[0])
    if not items:
        return
    layer_ids = [i for i, _ in items]
    mat = np.stack([row for _, row in items], axis=0)
    sc = _scale_cfg(plot_scales, "heads_heatmap")
    _draw_heatmap_with_annot(
        mat=mat,
        mask=None,
        x_label="Head index",
        y_label="Layer",
        x_ticks=list(range(mat.shape[1])),
        y_ticks=layer_ids,
        out_path=os.path.join(out_dir, f"heads_importance_heatmap"),
        fmt=".3f",
        vmin_override=_maybe_float(sc.get("vmin")),
        vmax_override=_maybe_float(sc.get("vmax")),
        debug_name="heads_heatmap",
        font_axis_label=HEADS_HEATMAP_AXIS_LABEL,
        font_tick_label=HEADS_HEATMAP_TICK_LABEL,
        font_colorbar_label=HEADS_HEATMAP_COLORBAR_LBL,
        font_colorbar_tick=HEADS_HEATMAP_COLORBAR_TCK,
        font_annot_base=HEADS_HEATMAP_ANNOT,
        colorbar_label=_importance_label_for_metric(heads_metric),
    )


def _plot_heads_by_layer(
    head_scores: Dict[str, torch.Tensor],
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
):
    if not (_PLOTTING_AVAILABLE and head_scores):
        return
    sc = _scale_cfg(plot_scales, "heads_bar")
    ymin = _maybe_float(sc.get("ymin"))
    ymax = _maybe_float(sc.get("ymax"))
    for lk, s in head_scores.items():
        scores = s.detach().to(torch.float32).cpu().flatten().numpy()
        n = int(scores.size)
        if n == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n)
        colors = _palette_colors_for_n(n, seed=1337 + (_layer_key_to_int(lk) or 0))
        ax.bar(x, scores, color=colors, edgecolor="none")
        ax.set_xlabel("Head index", fontsize=HEADS_BAR_AXIS_LABEL)
        ax.set_ylabel("Importance", fontsize=HEADS_BAR_AXIS_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=HEADS_BAR_XTICK)
        ax.tick_params(axis="both", which="major", labelsize=HEADS_BAR_TICK_LABEL)
        if ymin is not None or ymax is not None:
            ax.set_ylim(
                ymin if ymin is not None else ax.get_ylim()[0],
                ymax if ymax is not None else ax.get_ylim()[1],
            )
        _despine_xy(ax)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{lk}_heads_importance.png"),
            dpi=220,
        )
        fig.savefig(
            os.path.join(out_dir, f"{lk}_heads_importance.pdf"),
            dpi=220,
        )
        plt.close(fig)


def _neuron_log_params(
    scores: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, float, float]:
    s = np.asarray(scores, dtype=np.float64).copy()
    s = np.clip(s, eps, None)
    ymin = float(np.min(s)) if s.size else eps
    ymax = float(np.max(s)) if s.size else 1.0
    if not np.isfinite(ymin) or ymin <= 0:
        ymin = eps
    if not np.isfinite(ymax) or ymax <= ymin:
        ymax = ymin * 1.0001
    return s, ymin, ymax


def _plot_neurons_line_and_sorted(
    neuron_scores: Dict[str, torch.Tensor],
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
):
    if not (_PLOTTING_AVAILABLE and neuron_scores):
        return
    sc = _scale_cfg(plot_scales, "neurons_line")
    ymin_cfg = _maybe_float(sc.get("ymin"))
    ymax_cfg = _maybe_float(sc.get("ymax"))
    use_log = _maybe_bool(sc.get("log"), default=True)
    for lk, s in neuron_scores.items():
        raw = s.detach().to(torch.float32).cpu().flatten().numpy()
        if raw.size == 0:
            continue
        raw_c, ymin_auto, ymax_auto = _neuron_log_params(raw)
        perm = torch.argsort(
            torch.tensor(raw, dtype=torch.float32), descending=True
        ).numpy()
        sorted_c = raw_c[perm]
        ymin = ymin_auto if ymin_cfg is None else ymin_cfg
        ymax = ymax_auto if ymax_cfg is None else ymax_cfg
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(raw_c, color=IBM_BLUE, linewidth=1.0)
        ax.set_xlabel("Neuron index", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.set_ylabel("Importance", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.tick_params(axis="both", which="major", labelsize=NEURONS_LINE_TICK_LABEL)
        if use_log:
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        _despine_xy(ax)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_importance.png"),
            dpi=220,
        )
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_importance.pdf"),
            dpi=220,
        )
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sorted_c, color=IBM_ORANGE, linewidth=1.0)
        ax.set_xlabel("Rank", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.set_ylabel("Importance", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.tick_params(axis="both", which="major", labelsize=NEURONS_LINE_TICK_LABEL)
        if use_log:
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        _despine_xy(ax)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_importance_sorted.png"),
            dpi=220,
        )
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_importance_sorted.pdf"),
            dpi=220,
        )
        plt.close(fig)


def _plot_neurons_original_vs_sorted(
    neuron_scores: Dict[str, torch.Tensor],
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
):
    if not (_PLOTTING_AVAILABLE and neuron_scores):
        return
    sc = _scale_cfg(plot_scales, "neurons_line")
    ymin_cfg = _maybe_float(sc.get("ymin"))
    ymax_cfg = _maybe_float(sc.get("ymax"))
    use_log = _maybe_bool(sc.get("log"), default=True)
    for lk, s in neuron_scores.items():
        raw = s.detach().to(torch.float32).cpu().flatten().numpy()
        if raw.size == 0:
            continue
        raw_c, ymin_auto, ymax_auto = _neuron_log_params(raw)
        perm = torch.argsort(
            torch.tensor(raw, dtype=torch.float32), descending=True
        ).numpy()
        sorted_c = raw_c[perm]
        ymin = ymin_auto if ymin_cfg is None else ymin_cfg
        ymax = ymax_auto if ymax_cfg is None else ymax_cfg
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(raw_c, color=IBM_BLUE, linewidth=1.0, label="Original order")
        ax.plot(sorted_c, color=IBM_ORANGE, linewidth=1.0, label="Sorted (desc)")
        ax.set_xlabel("Index / Rank", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.set_ylabel("Importance", fontsize=NEURONS_LINE_AXIS_LABEL)
        ax.tick_params(axis="both", which="major", labelsize=NEURONS_LINE_TICK_LABEL)
        if use_log:
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        _despine_xy(ax)
        ax.legend(frameon=False, fontsize=NEURONS_LINE_LEGEND)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_original_vs_sorted.png"),
            dpi=220,
        )
        fig.savefig(
            os.path.join(out_dir, f"{lk}_neurons_original_vs_sorted.pdf"),
            dpi=220,
        )
        plt.close(fig)


def _group_means_by_position(scores_1d: np.ndarray, num_groups: int) -> np.ndarray:
    N = int(scores_1d.size)
    if N == 0 or num_groups <= 0:
        return np.zeros((0,), dtype=np.float32)
    group_size = int(math.ceil(N / num_groups))
    out = np.zeros((num_groups,), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, N)
        out[g] = float(np.mean(scores_1d[start:end])) if start < end else 0.0
    return out


def _plot_neuron_group_heatmap(
    model,
    neuron_scores: Dict[str, torch.Tensor],
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
    neurons_metric: Optional[str] = None,
):
    if not (_PLOTTING_AVAILABLE and neuron_scores):
        return
    text_model = _get_text_model(model)
    layer_ids: List[int] = []
    rows: List[np.ndarray] = []
    max_groups = 0
    for i, layer in enumerate(text_model.layers):
        lk = f"layer_{i}"
        s = neuron_scores.get(lk)
        if s is None:
            continue
        scores = s.detach().to(torch.float32).cpu().flatten().numpy()
        if scores.size == 0:
            continue
        _, num_heads, _, _ = _get_attn_layout(layer)
        if num_heads <= 0:
            continue
        gmeans = _group_means_by_position(scores, num_heads)
        max_groups = max(max_groups, int(gmeans.size))
        layer_ids.append(i)
        rows.append(gmeans)
    if not rows or max_groups <= 0:
        return
    mat = np.full((len(rows), max_groups), np.nan, dtype=np.float32)
    for r, row in enumerate(rows):
        mat[r, : row.size] = row
    mask = np.isnan(mat)
    sc = _scale_cfg(plot_scales, "neurons_group_heatmap")
    _draw_heatmap_with_annot(
        mat=mat,
        mask=mask,
        x_label="Neuron-group index",
        y_label="Layer",
        x_ticks=list(range(max_groups)),
        y_ticks=layer_ids,
        out_path=os.path.join(out_dir, f"neurons_group_heatmap"),
        fmt=".3f",
        vmin_override=_maybe_float(sc.get("vmin")),
        vmax_override=_maybe_float(sc.get("vmax")),
        debug_name="neurons_group_heatmap",
        font_axis_label=NEURONS_GRP_HEATMAP_AXIS_LABEL,
        font_tick_label=NEURONS_GRP_HEATMAP_TICK_LABEL,
        font_colorbar_label=NEURONS_GRP_HEATMAP_COLORBAR_LBL,
        font_colorbar_tick=NEURONS_GRP_HEATMAP_COLORBAR_TCK,
        font_annot_base=NEURONS_GRP_HEATMAP_ANNOT,
        colorbar_label=_importance_label_for_metric(neurons_metric),
    )


def _plot_embeddings_global(
    importance_data: Dict, out_dir: str, plot_scales: Optional[Dict[str, Any]]
):
    if not _PLOTTING_AVAILABLE:
        return
    emb_global_scores = importance_data.get("embeddings_global", {}).get("scores", None)
    if emb_global_scores is None:
        return
    sc = _scale_cfg(plot_scales, "embeddings_line")
    ymin = _maybe_float(sc.get("ymin"))
    ymax = _maybe_float(sc.get("ymax"))
    use_log = _maybe_bool(sc.get("log"), default=False)
    scores = emb_global_scores.detach().to(torch.float32).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(scores, color=IBM_BLUE, linewidth=1.0)
    ax.set_xlabel("Embedding dim", fontsize=EMB_LINE_AXIS_LABEL)
    ax.set_ylabel("Importance", fontsize=EMB_LINE_AXIS_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=EMB_LINE_TICK_LABEL)
    if use_log:
        ax.set_yscale("log")
    if ymin is not None or ymax is not None:
        ax.set_ylim(
            ymin if ymin is not None else ax.get_ylim()[0],
            ymax if ymax is not None else ax.get_ylim()[1],
        )
    _despine_xy(ax)
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"embeddings_global_importance.png"),
        dpi=220,
    )
    fig.savefig(
        os.path.join(out_dir, f"embeddings_global_importance.pdf"),
        dpi=220,
    )
    plt.close(fig)


def _plot_embeddings_global_grouped_by_heads(
    model, importance_data: Dict, out_dir: str, plot_scales: Optional[Dict[str, Any]]
):
    if not _PLOTTING_AVAILABLE:
        return
    emb_global_scores = importance_data.get("embeddings_global", {}).get("scores", None)
    if emb_global_scores is None:
        return
    text_model = _get_text_model(model)
    if not hasattr(text_model, "layers") or not text_model.layers:
        return
    _, num_heads, _, _ = _get_attn_layout(text_model.layers[0])
    if num_heads <= 0:
        return
    sc = _scale_cfg(plot_scales, "embeddings_grouped_bar")
    ymin = _maybe_float(sc.get("ymin"))
    ymax = _maybe_float(sc.get("ymax"))
    scores = emb_global_scores.detach().to(torch.float32).cpu().flatten().numpy()
    group_means = _group_means_by_position(scores, num_heads)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(num_heads)
    colors = _palette_colors_for_n(num_heads, seed=15000)
    ax.bar(x, group_means, color=colors, edgecolor="none")
    ax.set_xlabel("Embedding-group index", fontsize=EMB_BAR_AXIS_LABEL)
    ax.set_ylabel("Importance", fontsize=EMB_BAR_AXIS_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_heads)], fontsize=EMB_BAR_XTICK)
    ax.tick_params(axis="both", which="major", labelsize=EMB_BAR_TICK_LABEL)
    if ymin is not None or ymax is not None:
        ax.set_ylim(
            ymin if ymin is not None else ax.get_ylim()[0],
            ymax if ymax is not None else ax.get_ylim()[1],
        )
    _despine_xy(ax)
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"embeddings_global_grouped_by_heads.png"),
        dpi=220,
    )
    fig.savefig(
        os.path.join(out_dir, f"embeddings_global_grouped_by_heads.pdf"),
        dpi=220,
    )
    plt.close(fig)


def _save_visualizations(
    model,
    importance_data: Dict,
    out_dir: str,
    plot_scales: Optional[Dict[str, Any]],
    heads_metric: Optional[str] = None,
    neurons_metric: Optional[str] = None,
):
    _ensure_dir(out_dir)
    if not _PLOTTING_AVAILABLE:
        print(
            "[WARN] Plotting not available (matplotlib/numpy missing). Skipping plots."
        )
        return
    head_scores = importance_data["heads"]["scores"]
    if head_scores:
        _plot_heads_heatmap(
            head_scores, out_dir, plot_scales, heads_metric=heads_metric
        )
        _plot_heads_by_layer(head_scores, out_dir, plot_scales)
    neuron_scores = importance_data["neurons"]["scores"]
    if neuron_scores:
        _plot_neurons_line_and_sorted(neuron_scores, out_dir, plot_scales)
        _plot_neurons_original_vs_sorted(neuron_scores, out_dir, plot_scales)
        _plot_neuron_group_heatmap(
            model, neuron_scores, out_dir, plot_scales, neurons_metric=neurons_metric
        )
    _plot_embeddings_global(importance_data, out_dir, plot_scales)
    _plot_embeddings_global_grouped_by_heads(
        model, importance_data, out_dir, plot_scales
    )


# ============================================================
# Main entry points
# ============================================================


def postcheck_analyze_target(
    model_ref: str,
    tokenizer_ref: Optional[str],
    cal_text: str,
    out_dir: str,
    device: str,
    chunk_tokens: int = 2048,
    dtype_str: Optional[str] = None,
    heads_metric: Optional[str] = None,
    neurons_metric: Optional[str] = None,
    embeddings_metric: Optional[str] = None,
    cett_tail_prune_fracs: Optional[Iterable[float]] = None,
    cett_error_bound: Optional[float] = None,
    postcheck_config: Optional[Dict[str, Any]] = None,
    plot_scales: Optional[Dict[str, Any]] = None,
) -> Dict:
    _ensure_dir(out_dir)
    resolved_device = (
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if postcheck_config is not None:
        cfg_to_save = dict(postcheck_config)
        cfg_to_save["plot_scales"] = plot_scales
        _write_config(out_dir, cfg_to_save, filename="postcheck_config.yaml")
    print(f"\nLoading model: {model_ref}  device: {resolved_device}")
    model, tokenizer = load_model_and_tokenizer_single_device(
        model_id=model_ref,
        tokenizer_ref=(tokenizer_ref or model_ref),
        device=resolved_device,
        dtype_str=(dtype_str or "auto"),
    )
    importance_data = calculate_importance(
        model=model,
        tokenizer=tokenizer,
        text_data=cal_text,
        max_chunk_len=chunk_tokens,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
        embeddings_metric=embeddings_metric,
    )
    _identity_checks(model, importance_data)
    _save_visualizations(
        model,
        importance_data,
        out_dir,
        plot_scales,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
    )
    tail_reports = {}
    if heads_metric == "cett_normalized":
        tail_reports["heads"] = _report_tail_curve(
            scores_by_layer=importance_data["heads"]["scores"],
            prune_fracs=cett_tail_prune_fracs,
            error_bound=cett_error_bound,
            out_dir=os.path.join(out_dir, "cett_tail_heads"),
            title="Heads (cett_normalized)",
        )
    if neurons_metric == "cett_normalized":
        tail_reports["neurons"] = _report_tail_curve(
            scores_by_layer=importance_data["neurons"]["scores"],
            prune_fracs=cett_tail_prune_fracs,
            error_bound=cett_error_bound,
            out_dir=os.path.join(out_dir, "cett_tail_neurons"),
            title="Neurons (cett_normalized)",
        )
    if tail_reports:
        _write_json(os.path.join(out_dir, "cett_tail_curve_all.json"), tail_reports)
    print(f"\nSaved outputs to: {out_dir}")
    return importance_data


def run_postcheck(
    model_ref: str,
    reordered_dir: str,
    calibration_path: str,
    out_dir_original: str,
    out_dir_reordered: str,
    device: str = "auto",
    chunk_tokens: int = 2048,
    dtype_str: Optional[str] = None,
    heads_metric: Optional[str] = None,
    neurons_metric: Optional[str] = None,
    embeddings_metric: Optional[str] = None,
    cett_tail_prune_fracs: Optional[Iterable[float]] = None,
    cett_error_bound: Optional[float] = None,
    plot_scales: Optional[Dict[str, Any]] = None,
):
    with open(calibration_path, "r", encoding="utf-8") as f:
        cal_text = f.read()
    base_cfg = {
        "calibration_path": calibration_path,
        "device": device,
        "chunk_tokens": int(chunk_tokens),
        "dtype_str": dtype_str,
        "heads_metric": heads_metric,
        "neurons_metric": neurons_metric,
        "embeddings_metric": embeddings_metric,
        "cett_tail_prune_fracs": (
            None if cett_tail_prune_fracs is None else list(cett_tail_prune_fracs)
        ),
        "cett_error_bound": cett_error_bound,
    }
    print("\n--- Post-check: Original Model ---")
    postcheck_analyze_target(
        model_ref=model_ref,
        tokenizer_ref=model_ref,
        cal_text=cal_text,
        out_dir=out_dir_original,
        device=device,
        chunk_tokens=chunk_tokens,
        dtype_str=dtype_str,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
        embeddings_metric=embeddings_metric,
        cett_tail_prune_fracs=cett_tail_prune_fracs,
        cett_error_bound=cett_error_bound,
        postcheck_config={**base_cfg, "target": "original", "model_ref": model_ref},
        plot_scales=plot_scales,
    )
    print("\n--- Post-check: Reordered Model ---")
    postcheck_analyze_target(
        model_ref=reordered_dir,
        tokenizer_ref=reordered_dir,
        cal_text=cal_text,
        out_dir=out_dir_reordered,
        device=device,
        chunk_tokens=chunk_tokens,
        dtype_str=dtype_str,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
        embeddings_metric=embeddings_metric,
        cett_tail_prune_fracs=cett_tail_prune_fracs,
        cett_error_bound=cett_error_bound,
        postcheck_config={
            **base_cfg,
            "target": "reordered",
            "model_ref": reordered_dir,
        },
        plot_scales=plot_scales,
    )


def run_single_model_analysis(
    model_path: str,
    out_dir: str,
    calibration_path: str,
    device: str,
    dtype_str: Optional[str],
    chunk_tokens: int,
    heads_metric: Optional[str],
    neurons_metric: Optional[str],
    embeddings_metric: Optional[str],
    cett_tail_prune_fracs: Optional[Iterable[float]] = None,
    cett_error_bound: Optional[float] = None,
    plot_scales: Optional[Dict[str, Any]] = None,
):
    with open(calibration_path, "r", encoding="utf-8") as f:
        cal_text = f.read()
    cfg = {
        "target": "single_model",
        "model_ref": model_path,
        "calibration_path": calibration_path,
        "device": device,
        "chunk_tokens": int(chunk_tokens),
        "dtype_str": dtype_str,
        "heads_metric": heads_metric,
        "neurons_metric": neurons_metric,
        "embeddings_metric": embeddings_metric,
        "cett_tail_prune_fracs": (
            None if cett_tail_prune_fracs is None else list(cett_tail_prune_fracs)
        ),
        "cett_error_bound": cett_error_bound,
    }
    postcheck_analyze_target(
        model_ref=model_path,
        tokenizer_ref=model_path,
        cal_text=cal_text,
        out_dir=out_dir,
        device=device,
        chunk_tokens=chunk_tokens,
        dtype_str=dtype_str,
        heads_metric=heads_metric,
        neurons_metric=neurons_metric,
        embeddings_metric=embeddings_metric,
        cett_tail_prune_fracs=cett_tail_prune_fracs,
        cett_error_bound=cett_error_bound,
        postcheck_config=cfg,
        plot_scales=plot_scales,
    )
