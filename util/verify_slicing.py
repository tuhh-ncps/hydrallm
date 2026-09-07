import torch
from types import MethodType
from transformers import AutoTokenizer
from model_implementations import load_model
from common_helpers import pick_best_attn_impl
from common_helpers import safe_int as _safe_int


def _format_first_last(vals, n=5):
    try:
        arr = [float(x) for x in vals.tolist()]
        first = [f"{x:.2f}" for x in arr[:n]]
        last = (
            [f"{x:.2f}" for x in arr[-n:]]
            if len(arr) >= n
            else [f"{x:.2f}" for x in arr]
        )
        return first, last
    except Exception:
        return [], []


def _wrap_forward_sliced(mod, label=""):
    """
    Monkeypatch a DynamicLinear instance's forward_sliced to record:
    - output shape
    - per-feature activation magnitude (sum over batch and time)
    """
    if not hasattr(mod, "forward_sliced"):
        return False
    if hasattr(mod, "_orig_forward_sliced"):
        # Already wrapped
        return True
    orig = mod.forward_sliced

    def new_forward_sliced(self, input, out_rows=None, in_cols=None):
        out = orig(input, out_rows=out_rows, in_cols=in_cols)
        try:
            dims = tuple(range(out.ndim - 1)) if out.ndim >= 2 else ()
            mag = (
                out.detach().abs().sum(dim=dims).to("cpu")
                if len(dims) > 0
                else out.detach().abs().to("cpu")
            )
        except Exception:
            mag = None
        self.slicing_info = {
            "label": label,
            "output_shape": tuple(out.shape),
            "out_rows": int(out_rows) if out_rows is not None else None,
            "in_cols": int(in_cols) if in_cols is not None else None,
            "activation_magnitude": mag,
        }
        return out

    # Bind to instance
    mod._orig_forward_sliced = orig
    mod.forward_sliced = MethodType(new_forward_sliced, mod)
    return True


def _wrap_dynamic_embedding_forward(emb_mod):
    """
    If using DynamicTextScaledWordEmbedding, its forward has signature (input_ids, out_cols=None).
    Wrap it to capture the sliced output width.
    """
    if hasattr(emb_mod, "_orig_forward_wrapped"):
        return True
    orig = getattr(emb_mod, "forward", None)
    if orig is None:
        return False

    def new_forward(self, input_ids, out_cols=None):
        out = orig(input_ids, out_cols=out_cols)  # uses dynamic out_cols
        try:
            self.slicing_info = {
                "output_shape": tuple(out.shape),
                "out_cols": int(out_cols) if out_cols is not None else None,
            }
        except Exception:
            pass
        return out

    emb_mod._orig_forward_wrapped = orig
    emb_mod.forward = MethodType(new_forward, emb_mod)
    return True


def _wrap_rmsnorm_forward(norm_mod, label=""):
    """
    Wrap DynamicRMSNorm or Gemma3RMSNorm to record input/output shape and (if present) in_cols argument.
    """
    if hasattr(norm_mod, "_orig_forward_wrapped"):
        return True
    orig = getattr(norm_mod, "forward", None)
    if orig is None:
        return False

    def new_forward(self, x, *args, **kwargs):
        out = orig(x, *args, **kwargs)
        in_cols = kwargs.get("in_cols", None)
        try:
            self.slicing_info = {
                "label": label,
                "input_shape": tuple(x.shape) if hasattr(x, "shape") else None,
                "output_shape": tuple(out.shape) if hasattr(out, "shape") else None,
                "in_cols": int(in_cols) if in_cols is not None else None,
            }
        except Exception:
            pass
        return out

    norm_mod._orig_forward_wrapped = orig
    norm_mod.forward = MethodType(new_forward, norm_mod)
    return True


def _human_count(n: int) -> str:
    if n >= 10**9:
        return f"{n/1e9:.2f}B"
    if n >= 10**6:
        return f"{n/1e6:.2f}M"
    if n >= 10**3:
        return f"{n/1e3:.2f}K"
    return str(n)


def _linear_active_count(mod, out_rows=None, in_cols=None):
    w = mod.weight
    rows = int(out_rows) if out_rows is not None else w.size(0)
    cols = int(in_cols) if in_cols is not None else w.size(1)
    cnt = rows * cols
    if getattr(mod, "bias", None) is not None:
        cnt += rows
    return int(cnt)


def _lm_head_and_embed_tied(model) -> bool:
    """
    Detect if lm_head.weight and embed_tokens.weight are tied (share storage).
    """
    try:
        W_e = model.model.embed_tokens.weight
        W_o = model.lm_head.weight
        return W_e.data_ptr() == W_o.data_ptr()
    except Exception:
        return False


def _compute_active_params(model, k_heads: int, dynamic_embeddings: bool) -> int:
    """
    Compute structurally-active parameters for the specified number of heads.
    - Counts the subset of weights actually indexed/used by the dynamic slicing.
    - If lm_head and embeddings are tied, counts that shared weight only once.
    Note: This is a structural count (unique parameters touched), not FLOPs.
    """
    cfg = model.config
    H_full = int(cfg.num_attention_heads)
    d = int(cfg.head_dim)
    h = int(cfg.hidden_size)
    ffn = int(cfg.intermediate_size)
    kv_heads = int(getattr(cfg, "num_key_value_heads", H_full))
    q_out = int(k_heads) * d
    kv_out = kv_heads * d
    ratio = float(k_heads) / float(H_full) if H_full > 0 else 1.0
    dyn_embed = _safe_int(h * ratio) if dynamic_embeddings else h
    dyn_ffn = _safe_int(ffn * ratio)

    total_active = 0

    # Embeddings and LM head (handle tying to avoid double counting)
    embed_active = 0
    lm_head_active = 0

    embed_weight = getattr(model.model, "embed_tokens", None)
    if embed_weight is not None and hasattr(embed_weight, "weight"):
        V, H = embed_weight.weight.shape
        embed_active = V * (dyn_embed if dynamic_embeddings else H)

    lm = getattr(model, "lm_head", None)
    if lm is not None and hasattr(lm, "weight"):
        out_feats, in_feats = lm.weight.shape
        if dynamic_embeddings:
            lm_head_active = out_feats * dyn_embed  # in_cols sliced, out_rows full
        else:
            lm_head_active = out_feats * in_feats
        # include bias if present
        if getattr(lm, "bias", None) is not None:
            lm_head_active += out_feats

    if _lm_head_and_embed_tied(model):
        # Shared weight: count unique parameters touched (columns are the same submatrix).
        total_active += max(embed_active, lm_head_active)
    else:
        total_active += embed_active + lm_head_active

    # Per-layer modules
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        # Attention linears (sliced by heads and dyn_embed when dynamic)
        total_active += _linear_active_count(
            attn.q_proj, out_rows=q_out, in_cols=dyn_embed if dynamic_embeddings else h
        )
        total_active += _linear_active_count(
            attn.k_proj, out_rows=kv_out, in_cols=dyn_embed if dynamic_embeddings else h
        )
        total_active += _linear_active_count(
            attn.v_proj, out_rows=kv_out, in_cols=dyn_embed if dynamic_embeddings else h
        )
        total_active += _linear_active_count(
            attn.o_proj, out_rows=dyn_embed if dynamic_embeddings else h, in_cols=q_out
        )

        # q_norm / k_norm (vector over head_dim). This parameter vector is reused across heads;
        # count it once (all elements are used regardless of k_heads).
        if hasattr(attn, "q_norm") and hasattr(attn.q_norm, "weight"):
            total_active += attn.q_norm.weight.numel()
        if hasattr(attn, "k_norm") and hasattr(attn.k_norm, "weight"):
            total_active += attn.k_norm.weight.numel()

        # Layer RMSNorms
        for nm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            nmod = getattr(layer, nm, None)
            if nmod is not None and hasattr(nmod, "weight"):
                if dynamic_embeddings:
                    total_active += dyn_embed  # only first dyn_embed weights are used
                else:
                    total_active += nmod.weight.numel()

        # MLP linears
        total_active += _linear_active_count(
            mlp.gate_proj,
            out_rows=dyn_ffn,
            in_cols=dyn_embed if dynamic_embeddings else h,
        )
        total_active += _linear_active_count(
            mlp.up_proj,
            out_rows=dyn_ffn,
            in_cols=dyn_embed if dynamic_embeddings else h,
        )
        total_active += _linear_active_count(
            mlp.down_proj,
            out_rows=dyn_embed if dynamic_embeddings else h,
            in_cols=dyn_ffn,
        )

    # Final norm
    if hasattr(model.model, "norm") and hasattr(model.model.norm, "weight"):
        if dynamic_embeddings:
            total_active += dyn_embed
        else:
            total_active += model.model.norm.weight.numel()

    return int(total_active)


@torch.no_grad()
def run_slicing_verification(cfg: dict):
    """
    Verify width slicing by monkeypatching forward_sliced and dynamic embedding forward,
    then running a forward pass and reporting shapes and activation magnitudes.

    Returns a summary dict with shapes and inferred widths for tests.
    """
    verify_cfg = cfg.setdefault("verification", {})
    model_path = verify_cfg.get("model_path", "google/gemma-3-270m")
    implementation = verify_cfg.get("implementation", "hydra_gemma_from_flex")
    k_heads = verify_cfg.get("k_heads", None)
    prompt = verify_cfg.get("prompt", "The quick brown fox jumps over the lazy dog.")

    if k_heads is None:
        raise ValueError("verification.k_heads must be set (or pass --k-heads).")

    # Decide device and attention backend coherently
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    resolved_attn = pick_best_attn_impl("auto", device=device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"--- Verifying Slicing for k_heads = {k_heads} ---")
    print(f"Model: {model_path}")
    print(f"Implementation: {implementation}\n")

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = load_model(
        model_path,
        implementation=implementation,
        dtype=dtype,
        attn_implementation=resolved_attn,  # ensure CPU won't use flash_attn
        device=device,
    )
    model.eval()
    model = model.to(device)

    # Safety print: which attention impl are we actually using?
    try:
        actual_attn = (
            getattr(model.config, "_attn_implementation", None) or resolved_attn
        )
        print(f"Attention backend: {actual_attn} on device: {device}\n")
    except Exception:
        pass

    if not hasattr(model, "config") or not hasattr(model.config, "num_attention_heads"):
        print(
            "[ERROR] Could not determine number of attention heads from model config."
        )
        return

    dynamic_embeddings = "without_dynamic_embeddings" not in (implementation or "")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params = _compute_active_params(
        model, int(k_heads), dynamic_embeddings=dynamic_embeddings
    )

    # Base config summary
    total_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.head_dim)
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)
    ratio = float(k_heads) / float(total_heads)
    print("Config summary:")
    print(f"  - hidden_size:        {hidden_size}")
    print(f"  - num_attention_heads:{total_heads}")
    print(f"  - head_dim:           {head_dim}")
    print(f"  - intermediate_size:  {intermediate_size}")
    print(f"  - ratio (k/full):     {ratio:.4f}")
    print("Parameters:")
    print(f"  - total:              {total_params} ({_human_count(total_params)})")
    print(
        f"  - trainable:          {trainable_params} ({_human_count(trainable_params)})"
    )
    print(
        f"  - active (k={k_heads}): {active_params} ({_human_count(active_params)})\n"
    )

    # Wrap dynamic embedding (if available) to capture sliced embedding width
    try:
        _wrap_dynamic_embedding_forward(model.model.embed_tokens)
    except Exception:
        pass

    # Monkeypatch forward_sliced on q_proj, o_proj and gate_proj to capture stats
    print(
        "Instrumenting DynamicLinear.forward_sliced on q_proj, o_proj and gate_proj; wrapping norms..."
    )
    wrapped = {"q_proj": 0, "o_proj": 0, "gate_proj": 0, "norms": 0}

    # Wrap top model.norm as well
    try:
        if _wrap_rmsnorm_forward(model.model.norm, label="model.norm"):
            wrapped["norms"] += 1
    except Exception:
        pass

    for i, layer in enumerate(model.model.layers):
        # Attention q_proj
        try:
            if _wrap_forward_sliced(
                layer.self_attn.q_proj, label=f"layer_{i}.self_attn.q_proj"
            ):
                wrapped["q_proj"] += 1
        except Exception:
            pass
        # Attention o_proj
        try:
            if _wrap_forward_sliced(
                layer.self_attn.o_proj, label=f"layer_{i}.self_attn.o_proj"
            ):
                wrapped["o_proj"] += 1
        except Exception:
            pass
        # MLP gate_proj
        try:
            if _wrap_forward_sliced(
                layer.mlp.gate_proj, label=f"layer_{i}.mlp.gate_proj"
            ):
                wrapped["gate_proj"] += 1
        except Exception:
            pass
        # Norms
        for norm_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            try:
                nm = getattr(layer, norm_name)
                if _wrap_rmsnorm_forward(nm, label=f"layer_{i}.{norm_name}"):
                    wrapped["norms"] += 1
            except Exception:
                pass

    print(f"- Wrapped q_proj:    {wrapped['q_proj']} layers")
    print(f"- Wrapped o_proj:    {wrapped['o_proj']} layers")
    print(f"- Wrapped gate_proj: {wrapped['gate_proj']} layers")
    print(f"- Wrapped norms:     {wrapped['norms']} modules")
    print("-" * 50)

    # Forward pass
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Running forward pass with prompt: '{prompt}'...")
    _ = model(**inputs, current_num_heads=int(k_heads), use_cache=False)
    print("Forward pass complete.\n")

    # Expected sizes
    # H_active * D
    expected_q_features = int(k_heads) * head_dim
    expected_mlp_features = max(1, int(intermediate_size * ratio))  # ratio * FFN width
    # ratio * hidden_size
    expected_dyn_embed = max(1, int(hidden_size * ratio))
    # o_proj out_rows = dyn_embed
    expected_o_out = expected_dyn_embed

    # Report embedding (if dynamic)
    print("--- Slicing Verification Report ---\n")
    if hasattr(model.model.embed_tokens, "slicing_info"):
        shape = model.model.embed_tokens.slicing_info.get("output_shape", None)
        if shape is not None:
            print("[Embeddings]")
            print(f"  - embed_tokens output shape: {list(shape)}")
            print(f"    - Features: {shape[-1]}")
            print(f"    - Expected (dynamic): {expected_dyn_embed}")
            print(
                f"    - Note: In non-dynamic implementation, this is full hidden_size.\n"
            )

    # Per-layer report and collect summary
    summary = {
        "config": {
            "hidden_size": hidden_size,
            "num_attention_heads": total_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "ratio": ratio,
            "k_heads": int(k_heads),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "active_parameters": int(active_params),
        },
        "embeddings": {},
        "layers": [],
    }

    if hasattr(model.model.embed_tokens, "slicing_info"):
        ei = model.model.embed_tokens.slicing_info
        if "output_shape" in ei and ei["output_shape"] is not None:
            summary["embeddings"]["features"] = int(ei["output_shape"][-1])

    for i, layer in enumerate(model.model.layers):
        layer_entry = {"index": i}
        print(f"[Layer {i}]")
        # q_proj
        q_proj = layer.self_attn.q_proj
        if hasattr(q_proj, "slicing_info"):
            info = q_proj.slicing_info
            shape = info.get("output_shape", None)
            if shape is not None:
                last = int(shape[-1])
                layer_entry["q_proj_features"] = last
            print(
                f"  - self_attn.q_proj output shape: {list(shape) if shape else shape}"
            )
            if shape is not None:
                print(f"    - Actual features:   {shape[-1]}")
                print(
                    f"    - Expected features: {expected_q_features} ({k_heads} heads * {head_dim} dim)"
                )
                print(
                    f"    - STATUS: {'OK' if shape[-1] == expected_q_features else 'MISMATCH'}"
                )
            mag = info.get("activation_magnitude", None)
            if isinstance(mag, torch.Tensor):
                active = int((mag > 1e-6).sum().item())
                first, last = _format_first_last(mag, n=5)
                print(f"    - Activated head dims: {active} / {mag.numel()}")
                print(f"    - Magnitude (first 5): {first}")
                print(f"    - Magnitude (last 5):  {last}")
        else:
            print("  - self_attn.q_proj: no capture (not wrapped or not executed).")

        # o_proj
        o_proj = layer.self_attn.o_proj
        if hasattr(o_proj, "slicing_info"):
            info = o_proj.slicing_info
            shape = info.get("output_shape", None)
            if shape is not None:
                last = int(shape[-1])
                layer_entry["o_proj_features"] = last
            print(
                f"  - self_attn.o_proj output shape: {list(shape) if shape else shape}"
            )
            if shape is not None:
                print(f"    - Actual features:   {shape[-1]}")
                print(f"    - Expected features: {expected_o_out} (dyn_embed_dim)")
                print(
                    f"    - STATUS: {'OK' if shape[-1] == expected_o_out else 'MISMATCH'}"
                )
        else:
            print("  - self_attn.o_proj: no capture (not wrapped or not executed).")

        # gate_proj
        gate_proj = layer.mlp.gate_proj
        if hasattr(gate_proj, "slicing_info"):
            info = gate_proj.slicing_info
            shape = info.get("output_shape", None)
            if shape is not None:
                last = int(shape[-1])
                layer_entry["gate_proj_features"] = last
            print(
                f"  - mlp.gate_proj output shape:    {list(shape) if shape else shape}"
            )
            if shape is not None:
                print(f"    - Actual features:   {shape[-1]}")
                print(f"    - Expected features: {expected_mlp_features}")
                print(
                    f"    - STATUS: {'OK' if shape[-1] == expected_mlp_features else 'MISMATCH'}"
                )
            mag = info.get("activation_magnitude", None)
            if isinstance(mag, torch.Tensor):
                active = int((mag > 1e-6).sum().item())
                first, last = _format_first_last(mag, n=5)
                print(f"    - Activated neurons: {active} / {mag.numel()}")
                print(f"    - Magnitude (first 5): {first}")
                print(f"    - Magnitude (last 5):  {last}")
        else:
            print("  - mlp.gate_proj: no capture (not wrapped or not executed).")

        # Norms
        norms = {}
        for norm_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            nm = getattr(layer, norm_name, None)
            if nm is not None and hasattr(nm, "slicing_info"):
                ni = nm.slicing_info
                out_shape = ni.get("output_shape", None)
                if out_shape is not None:
                    norms[norm_name] = {
                        "output_features": int(out_shape[-1]),
                        "in_cols": ni.get("in_cols", None),
                    }
        if norms:
            layer_entry["norms"] = norms

        summary["layers"].append(layer_entry)
        print("")  # spacer between layers

    # Also record top-level model.norm
    if hasattr(model.model, "norm") and hasattr(model.model.norm, "slicing_info"):
        ni = model.model.norm.slicing_info
        out_shape = ni.get("output_shape", None)
        if out_shape is not None:
            summary["final_norm"] = {
                "output_features": int(out_shape[-1]),
                "in_cols": ni.get("in_cols", None),
            }

    # Expose parameter counts at the top-level for convenience
    summary["parameters"] = {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "active_k_heads": int(active_params),
        "human_total": _human_count(total_params),
        "human_trainable": _human_count(trainable_params),
        "human_active_k_heads": _human_count(active_params),
    }
    return summary
