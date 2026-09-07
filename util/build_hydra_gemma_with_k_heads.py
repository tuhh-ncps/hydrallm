# TODO: Check equivalence

import argparse
import os
import re
import json
import math
from typing import Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
from transformers import AutoConfig


def _slice_weight(W: torch.Tensor, out_rows: Optional[int] = None, in_cols: Optional[int] = None) -> torch.Tensor:
    r = W.shape[0] if out_rows is None else min(out_rows, W.shape[0])
    c = W.shape[1] if in_cols is None else min(in_cols, W.shape[1])
    return W[:r, :c].contiguous()


def _slice_bias(b: torch.Tensor, out_rows: Optional[int] = None) -> torch.Tensor:
    r = b.shape[0] if out_rows is None else min(out_rows, b.shape[0])
    return b[:r].contiguous()


def _maybe_slice_norm(w: torch.Tensor, dyn_embed_dim: int) -> torch.Tensor:
    # RMSNorm gamma params of size hidden_size; slice first dyn_embed_dim
    return w[:min(dyn_embed_dim, w.shape[0])].contiguous()


def _is_multimodal_config(cfg) -> bool:
    return hasattr(cfg, "text_config") and hasattr(cfg, "vision_config")


def _warn(msg: str):
    print(f"WARNING: {msg}")


def _compute_targets(
    cfg,
    requested_heads: int,
    proj_out_override: Optional[int] = None,
    force: bool = False,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Returns:
      dyn_embed_dim, dyn_intermediate, q_out, kv_out, new_kv_heads, head_dim, dyn_proj_out, effective_heads

    Key behavior changes requested:
      - If requested_heads == 1: allow slicing Q projections (effective_heads=1).
      - If base has multiple KV groups (GQA, i.e. full_group_size > 1):
          - By default (force=False): only allow slicing full KV groups, i.e.
                effective_heads is a multiple of full_group_size (or 1),
                and new_kv_heads = effective_heads / full_group_size (or 1 if effective_heads==1).
            If requested_heads is not compatible, we adjust effective_heads and print a warning.
          - With --force: proceed anyway and pick KV heads as Flex does:
                new_kv_heads = ceil(effective_heads / full_group_size),
            but warn that this may not match vanilla HF auto_model semantics.
    """
    # Read base values (text vs multimodal text subconfig)
    tcfg = cfg.text_config if _is_multimodal_config(cfg) else cfg

    base_heads = int(tcfg.num_attention_heads)
    head_dim = int(tcfg.head_dim)
    base_hidden = int(tcfg.hidden_size)
    base_intermediate = int(tcfg.intermediate_size)

    base_kv_heads = int(getattr(tcfg, "num_key_value_heads", max(1, base_heads // 8)))
    base_kv_heads = max(1, min(base_kv_heads, base_heads))

    if base_heads % base_kv_heads != 0:
        # This should not happen for normal Gemma/LLaMA configs, but handle defensively.
        _warn(f"Base config has num_attention_heads ({base_heads}) not divisible by num_key_value_heads ({base_kv_heads}).")

    full_group_size = max(1, base_heads // base_kv_heads)  # Q heads per KV head in the base model

    # Determine effective_heads (possibly adjusted)
    effective_heads = int(requested_heads)
    if effective_heads <= 0:
        raise ValueError("--heads must be >= 1")

    # Special-case: single head is always allowed (explicit user request)
    if effective_heads == 1:
        new_kv_heads = 1
    else:
        if full_group_size == 1:
            # No GQA grouping; KV heads track Q heads (standard MHA case).
            new_kv_heads = min(effective_heads, base_kv_heads)
            if new_kv_heads != effective_heads:
                _warn(
                    f"Requested heads={effective_heads} but base_kv_heads={base_kv_heads}. "
                    f"Clamping num_key_value_heads to {new_kv_heads}."
                )
        else:
            # GQA present: multiple KV groups
            if not force:
                # Only allow full KV groups (uniform groups consistent with base full_group_size).
                if effective_heads % full_group_size != 0:
                    # Adjust heads to a compatible value
                    down = (effective_heads // full_group_size) * full_group_size
                    if down < full_group_size:
                        # Smallest compatible >1 is one full group
                        adjusted = full_group_size
                    else:
                        adjusted = down

                    _warn(
                        f"Base model uses GQA with full_group_size={full_group_size} (Q heads per KV head). "
                        f"Requested --heads={effective_heads} is not a full number of KV groups. "
                        f"Adjusting effective heads to {adjusted} to keep only full KV groups. "
                        f"Use --force to keep {effective_heads} anyway (may break/mismatch HF auto_model behavior)."
                    )
                    effective_heads = adjusted

                new_kv_heads = max(1, effective_heads // full_group_size)
            else:
                # Force mode: follow Flex-style KV selection
                new_kv_heads = int(math.ceil(effective_heads / full_group_size))
                if effective_heads % full_group_size != 0:
                    _warn(
                        f"--force enabled: keeping --heads={effective_heads} even though it is not a multiple of "
                        f"full_group_size={full_group_size}. This can create uneven KV groups; "
                        f"a vanilla HF auto_model implementation may not be mathematically equivalent."
                    )

            # Clamp KV heads to base_kv_heads (can't exceed rows available in K/V projection weights)
            if new_kv_heads > base_kv_heads:
                _warn(
                    f"Computed num_key_value_heads={new_kv_heads} exceeds base num_key_value_heads={base_kv_heads}. "
                    f"Clamping to {base_kv_heads}."
                )
                new_kv_heads = base_kv_heads

    # Sanity constraints
    new_kv_heads = max(1, int(new_kv_heads))
    effective_heads = max(1, int(effective_heads))

    # Derive dynamic widths by head ratio (Hydra/Flex behavior)
    head_ratio = float(effective_heads) / float(base_heads) if base_heads > 0 else 1.0
    dyn_embed_dim = max(1, int(math.floor(base_hidden * head_ratio)))
    dyn_intermediate = max(1, int(math.floor(base_intermediate * head_ratio)))

    q_out = effective_heads * head_dim
    kv_out = new_kv_heads * head_dim

    # Projector output equals dyn_embed_dim unless overridden
    dyn_proj_out = dyn_embed_dim if proj_out_override is None else int(proj_out_override)

    return dyn_embed_dim, dyn_intermediate, q_out, kv_out, new_kv_heads, head_dim, dyn_proj_out, effective_heads


def _update_config(cfg, dyn_embed_dim, dyn_intermediate, effective_heads, new_kv_heads, head_dim, multimodal: bool):
    tcfg = cfg.text_config if multimodal else cfg
    tcfg.hidden_size = int(dyn_embed_dim)
    tcfg.num_attention_heads = int(effective_heads)
    tcfg.num_key_value_heads = int(new_kv_heads)
    tcfg.intermediate_size = int(dyn_intermediate)
    tcfg.head_dim = int(head_dim)
    return cfg


def _process_tensor(
    name: str,
    tensor: torch.Tensor,
    dyn_embed_dim: int,
    dyn_intermediate: int,
    q_out: int,
    kv_out: int,
    multimodal: bool,
    dyn_proj_out: int,
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Slice rules (prefix slicing):
      - layer norms: slice gamma to dyn_embed_dim
      - q_proj: W[:q_out, :dyn_embed_dim], b[:q_out]
      - k_proj/v_proj: W[:kv_out, :dyn_embed_dim], b[:kv_out]
      - o_proj: W[:dyn_embed_dim, :q_out], b[:dyn_embed_dim]
      - MLP gate/up: W[:dyn_intermediate, :dyn_embed_dim], b[:dyn_intermediate]
      - MLP down: W[:dyn_embed_dim, :dyn_intermediate], b[:dyn_embed_dim]
      - embed_tokens: [vocab, :dyn_embed_dim]
      - lm_head: [vocab, :dyn_embed_dim]
      - final norm: slice gamma to dyn_embed_dim
      - projector (mm_input_projection_weight): [:, :dyn_proj_out]
    Other tensors: copy unchanged.

    The vision tower is ALWAYS copied unchanged: SigLIP vision layers also
    match ".layers.N." and use "self_attn.q/k/v_proj", so without this guard
    they would be sliced to text-model widths and corrupted.
    """
    # Vision tower: copy unchanged (must not be sliced to text-model widths)
    if "vision_tower." in name or "vision_model." in name:
        return name, tensor

    # Embeddings
    if name.endswith("embed_tokens.weight"):
        return name, tensor[:, :min(dyn_embed_dim, tensor.shape[1])].contiguous()

    # Per-layer blocks
    if re.search(r"\.layers\.(\d+)\.", name):
        # attention QKV/O
        if ".self_attn.q_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=q_out, in_cols=dyn_embed_dim)
        if ".self_attn.q_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=q_out)

        if ".self_attn.k_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=kv_out, in_cols=dyn_embed_dim)
        if ".self_attn.k_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=kv_out)

        if ".self_attn.v_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=kv_out, in_cols=dyn_embed_dim)
        if ".self_attn.v_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=kv_out)

        if ".self_attn.o_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=dyn_embed_dim, in_cols=q_out)
        if ".self_attn.o_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=dyn_embed_dim)

        # layer norms over model dim
        if (
            ".input_layernorm.weight" in name
            or ".post_attention_layernorm.weight" in name
            or ".pre_feedforward_layernorm.weight" in name
            or ".post_feedforward_layernorm.weight" in name
        ):
            return name, _maybe_slice_norm(tensor, dyn_embed_dim)

        # MLP
        if ".mlp.gate_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=dyn_intermediate, in_cols=dyn_embed_dim)
        if ".mlp.gate_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=dyn_intermediate)

        if ".mlp.up_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=dyn_intermediate, in_cols=dyn_embed_dim)
        if ".mlp.up_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=dyn_intermediate)

        if ".mlp.down_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=dyn_embed_dim, in_cols=dyn_intermediate)
        if ".mlp.down_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=dyn_embed_dim)

    # final model norm (text)
    if name.endswith(".norm.weight") or name == "norm.weight":
        return name, _maybe_slice_norm(tensor, dyn_embed_dim)

    # LM head
    if name.endswith("lm_head.weight"):
        return name, tensor[:, :min(dyn_embed_dim, tensor.shape[1])].contiguous()

    # Multimodal projector
    if multimodal and name.endswith("multi_modal_projector.mm_input_projection_weight"):
        return name, tensor[:, :min(dyn_proj_out, tensor.shape[1])].contiguous()

    return name, tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id or local path to original model")
    ap.add_argument("--dst", required=True, help="Output directory for sliced checkpoint")
    ap.add_argument("--heads", type=int, required=True, help="Requested target num_attention_heads (k)")
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force slicing even when the requested head count would imply uneven KV-grouping under GQA. "
            "This can make the saved model not mathematically equivalent (and possibly incompatible) "
            "with vanilla HF auto_model implementations."
        ),
    )
    ap.add_argument(
        "--proj_out",
        type=int,
        default=None,
        help="Optional projector output width (multimodal). Defaults to dyn_embed_dim.",
    )
    ap.add_argument("--max_shard_gb", type=float, default=4.0, help="Shard size in GB")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Resolve model snapshot folder
    if os.path.isdir(args.src):
        model_path = args.src
    else:
        model_path = snapshot_download(
            args.src,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=["*.bin"],
        )

    # Load config and compute slicing targets
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    multimodal = _is_multimodal_config(cfg)

    (
        dyn_embed_dim,
        dyn_intermediate,
        q_out,
        kv_out,
        new_kv_heads,
        head_dim,
        dyn_proj_out,
        effective_heads,
    ) = _compute_targets(cfg, args.heads, args.proj_out, force=args.force)

    if effective_heads != args.heads and not args.force:
        _warn(f"Effective num_attention_heads used for slicing is {effective_heads} (requested {args.heads}).")

    # Update and save config to match the physically sliced checkpoint
    cfg = _update_config(
        cfg,
        dyn_embed_dim=dyn_embed_dim,
        dyn_intermediate=dyn_intermediate,
        effective_heads=effective_heads,
        new_kv_heads=new_kv_heads,
        head_dim=head_dim,
        multimodal=multimodal,
    )
    cfg.save_pretrained(args.dst)

    # Copy tokenizer files if present
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ]:
        srcf = os.path.join(model_path, fname)
        if os.path.exists(srcf):
            try:
                import shutil
                shutil.copy(srcf, os.path.join(args.dst, fname))
            except Exception:
                pass

    # Gather shards
    shard_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    shard_files.sort()

    weight_map = {}
    new_shard_state = {}
    shard_counter = 1

    def flush_shard():
        nonlocal new_shard_state, shard_counter
        if not new_shard_state:
            return
        shard_filename = f"model-{shard_counter:05d}-of-XXXXX.safetensors"
        save_file(new_shard_state, os.path.join(args.dst, shard_filename), metadata={"format": "pt"})
        for k in new_shard_state.keys():
            weight_map[k] = shard_filename
        shard_counter += 1
        new_shard_state = {}

    bytes_limit = int(args.max_shard_gb * (1024**3))

    # Stream and slice
    for shard in shard_files:
        shard_path = os.path.join(model_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as fin:
            for key in fin.keys():
                tens = fin.get_tensor(key)

                new_name, new_tensor = _process_tensor(
                    key,
                    tens,
                    dyn_embed_dim=dyn_embed_dim,
                    dyn_intermediate=dyn_intermediate,
                    q_out=q_out,
                    kv_out=kv_out,
                    multimodal=multimodal,
                    dyn_proj_out=dyn_proj_out,
                )

                if new_tensor is None:
                    continue

                new_shard_state[new_name] = new_tensor

                current_size = sum(t.numel() * t.element_size() for t in new_shard_state.values())
                if current_size >= bytes_limit:
                    flush_shard()

    flush_shard()

    # Fix shard numbering in filenames and weight_map
    final_shards = shard_counter - 1
    for i in range(1, final_shards + 1):
        old_name = f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = f"model-{i:05d}-of-{final_shards:05d}.safetensors"
        os.rename(os.path.join(args.dst, old_name), os.path.join(args.dst, new_name))
        for k, v in list(weight_map.items()):
            if v == old_name:
                weight_map[k] = new_name

    # Write index
    total_size = sum(
        os.path.getsize(os.path.join(args.dst, f))
        for f in os.listdir(args.dst)
        if f.endswith(".safetensors")
    )
    index_json = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index_json, f, indent=2)

    print("Done.")
    print(f"- requested_heads: {args.heads}")
    print(f"- effective_heads: {effective_heads}")
    print(f"- dyn_embed_dim: {dyn_embed_dim}")
    print(f"- dyn_intermediate: {dyn_intermediate}")
    print(f"- q_out (heads*head_dim): {q_out}")
    print(f"- kv_out (kv_heads*head_dim): {kv_out}")
    print(f"- num_kv_heads: {new_kv_heads}")
    if multimodal:
        print(f"- dyn_proj_out (projector): {dyn_proj_out}")
    if args.force:
        print("- force: enabled")
    print(f"Sliced model saved to {args.dst}")


if __name__ == "__main__":
    main()