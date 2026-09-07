import argparse
import os
import re
import json
from math import floor
from typing import Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
from transformers import AutoConfig


def _slice_weight(W: torch.Tensor, out_rows: Optional[int] = None, in_cols: Optional[int] = None) -> torch.Tensor:
    r = W.shape[0] if out_rows is None else min(int(out_rows), W.shape[0])
    c = W.shape[1] if in_cols is None else min(int(in_cols), W.shape[1])
    return W[:r, :c].contiguous()


def _slice_bias(b: torch.Tensor, out_rows: Optional[int] = None) -> torch.Tensor:
    r = b.shape[0] if out_rows is None else min(int(out_rows), b.shape[0])
    return b[:r].contiguous()


def _is_multimodal_config(cfg) -> bool:
    return hasattr(cfg, "text_config") and hasattr(cfg, "vision_config")


def _round_down_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return (x // multiple) * multiple


def _compute_matformer_targets(cfg, mlp_ratio: float, round_multiple: int) -> Tuple[int, int, bool]:
    """
    MatFormer slicing target:
      - Keep hidden_size and attention heads unchanged
      - Only reduce intermediate_size by mlp_ratio
    Returns: (base_hidden, new_intermediate, multimodal)
    """
    multimodal = _is_multimodal_config(cfg)
    tcfg = cfg.text_config if multimodal else cfg

    base_hidden = int(tcfg.hidden_size)
    base_intermediate = int(tcfg.intermediate_size)

    if mlp_ratio <= 0:
        raise ValueError("--mlp_ratio must be > 0")
    if mlp_ratio > 1.0:
        # Slicing can only reduce the checkpoint: rows beyond the base
        # intermediate_size do not exist in the source weights.
        raise ValueError("--mlp_ratio must be <= 1.0 for checkpoint slicing")

    new_intermediate = max(1, int(floor(base_intermediate * float(mlp_ratio))))
    new_intermediate = _round_down_to_multiple(new_intermediate, round_multiple)
    new_intermediate = max(1, new_intermediate)

    return base_hidden, new_intermediate, multimodal


def _update_config_matformer(cfg, new_intermediate: int, multimodal: bool):
    tcfg = cfg.text_config if multimodal else cfg
    tcfg.intermediate_size = int(new_intermediate)
    # Do NOT change:
    # - hidden_size
    # - num_attention_heads / num_key_value_heads / head_dim
    return cfg


def _process_tensor_matformer(
    name: str,
    tensor: torch.Tensor,
    base_hidden: int,
    new_intermediate: int,
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    MatFormer-style checkpoint slicing:
      - Only slice MLP weights/biases to match new_intermediate
      - Everything else is copied unchanged (embeddings, attention, norms, lm_head, projector, vision tower, etc.)

    Target shapes:
      gate/up:  [new_intermediate, base_hidden]
      down:     [base_hidden, new_intermediate]
    """
    # Apply only inside decoder layers (but keep it permissive by matching substrings)
    if re.search(r"\.layers\.(\d+)\.", name):
        if ".mlp.gate_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=new_intermediate, in_cols=base_hidden)
        if ".mlp.gate_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=new_intermediate)

        if ".mlp.up_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=new_intermediate, in_cols=base_hidden)
        if ".mlp.up_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=new_intermediate)

        if ".mlp.down_proj.weight" in name:
            return name, _slice_weight(tensor, out_rows=base_hidden, in_cols=new_intermediate)
        if ".mlp.down_proj.bias" in name:
            return name, _slice_bias(tensor, out_rows=base_hidden)

    # Some model variants might not use ".layers.N." in names; allow a fallback:
    if ".mlp.gate_proj.weight" in name:
        return name, _slice_weight(tensor, out_rows=new_intermediate, in_cols=base_hidden)
    if ".mlp.gate_proj.bias" in name:
        return name, _slice_bias(tensor, out_rows=new_intermediate)

    if ".mlp.up_proj.weight" in name:
        return name, _slice_weight(tensor, out_rows=new_intermediate, in_cols=base_hidden)
    if ".mlp.up_proj.bias" in name:
        return name, _slice_bias(tensor, out_rows=new_intermediate)

    if ".mlp.down_proj.weight" in name:
        return name, _slice_weight(tensor, out_rows=base_hidden, in_cols=new_intermediate)
    if ".mlp.down_proj.bias" in name:
        return name, _slice_bias(tensor, out_rows=base_hidden)

    # Default: copy unchanged
    return name, tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id or local path to original model")
    ap.add_argument("--dst", required=True, help="Output directory for sliced checkpoint")
    ap.add_argument(
        "--mlp_ratio",
        type=float,
        required=True,
        help="MatFormer MLP width ratio in (0, 1]. Only intermediate_size is reduced; hidden_size/heads stay full.",
    )
    ap.add_argument(
        "--round_multiple",
        type=int,
        default=1,
        help="Round DOWN the computed intermediate_size to a multiple of this value (e.g., 256). Default=1.",
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

    # Load config and compute MatFormer targets
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    base_hidden, new_intermediate, multimodal = _compute_matformer_targets(
        cfg, args.mlp_ratio, args.round_multiple
    )

    # Update and save config (only intermediate_size changes)
    cfg = _update_config_matformer(cfg, new_intermediate=new_intermediate, multimodal=multimodal)
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

                new_name, new_tensor = _process_tensor_matformer(
                    key,
                    tens,
                    base_hidden=base_hidden,
                    new_intermediate=new_intermediate,
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
    print(f"- multimodal: {multimodal}")
    print(f"- base_hidden_size (unchanged): {base_hidden}")
    print(f"- new_intermediate_size: {new_intermediate}  (ratio={args.mlp_ratio}, round_multiple={args.round_multiple})")
    print("Sliced tensors: only MLP (gate/up/down) matrices (and biases if present).")
    print(f"Sliced model saved to {args.dst}")


if __name__ == "__main__":
    main()