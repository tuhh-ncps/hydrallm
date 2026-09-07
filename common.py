# common.py
from __future__ import annotations
import os
import sys
import yaml
import copy
import argparse
import json
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, List, Set

# --- Validation Constants ---
class _RequiredSentinel:
    def __repr__(self):
        return "<REQUIRED>"
    def __bool__(self):
        return False 

# Use this constant in your default dictionaries for values that MUST be provided
REQUIRED = _RequiredSentinel()

# Optional: load .env for HF token and env vars
try:
    from dotenv import load_dotenv
    _DOTENV_LOADED = False
    if os.path.exists(".env"):
        load_dotenv()
        _DOTENV_LOADED = True
except ImportError:
    pass

def str_to_bool(v):
    """
    Converts string/bool/None/REQUIRED to a boolean or None.
    """
    if isinstance(v, bool):
        return v
    if v is None: 
        return None
    if v is REQUIRED:
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    return v

def setup_environment():
    """Sets up HF token, allows TF32, and handles .env loading."""
    if '_DOTENV_LOADED' in globals() and _DOTENV_LOADED:
        tok = os.getenv("HF_TOKEN")
        if tok:
            os.environ["HUGGINGFACE_TOKEN"] = tok
            os.environ["HF_HUB_TOKEN"] = tok
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        print("[WARN] PyTorch not found. Skipping CUDA setup.")
    except Exception as e:
        print(f"[WARN] Failed to set up PyTorch CUDA: {e}")

def add_distributed_args(parser: argparse.ArgumentParser):
    """Adds standard distributed training arguments to a parser."""
    group = parser.add_argument_group("Distributed Training")
    group.add_argument("--local-rank", "--local_rank", type=int, dest="local_rank", default=None)
    group.add_argument("--rank", "--global_rank", type=int, dest="rank", default=None)
    group.add_argument("--world-size", "--world_size", type=int, dest="world_size", default=None)
    group.add_argument("--node-rank", "--node_rank", type=int, dest="node_rank", default=None)
    group.add_argument("--master-addr", "--master_addr", type=str, dest="master_addr", default=None)
    group.add_argument("--master-port", "--master_port", type=int, dest="master_port", default=None)

def set_dist_env_from_args(args: argparse.Namespace):
    """Sets environment variables based on distributed args."""
    if getattr(args, "local_rank", None) is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    if getattr(args, "rank", None) is not None:
        os.environ["RANK"] = str(args.rank)
    if getattr(args, "world_size", None) is not None:
        os.environ["WORLD_SIZE"] = str(args.world_size)
    if getattr(args, "node_rank", None) is not None:
        os.environ["NODE_RANK"] = str(args.node_rank)
    if getattr(args, "master_addr", None) is not None:
        os.environ["MASTER_ADDR"] = str(args.master_addr)
    if getattr(args, "master_port", None) is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)

# ----------------------- Config Logic ----------------------- #

def default_device_from_env(cfg: dict, section: str = "inference") -> dict:
    """Resolve cfg[section]['device'] == 'auto' to a concrete cuda/mps/cpu device."""
    try:
        from common_helpers import pick_default_device
    except ImportError:
        return cfg
    if cfg.get(section, {}).get("device", "auto") == "auto":
        cfg.setdefault(section, {})["device"] = pick_default_device()
    return cfg


def _deep_merge_with_fallbacks(defaults: dict, user: dict, path: Optional[List[str]] = None) -> Tuple[dict, Set[Tuple[str, ...]]]:
    if path is None:
        path = []
    merged = {}
    fallbacks: Set[Tuple[str, ...]] = set()
    for k, v_def in defaults.items():
        p = path + [k]
        if k not in user:
            merged[k] = copy.deepcopy(v_def)
            def _collect_leaf_paths(val, curp):
                if isinstance(val, dict):
                    if not val: 
                        fallbacks.add(tuple(curp))
                    else:
                        for kk, vv in val.items():
                            _collect_leaf_paths(vv, curp + [kk])
                else:
                    fallbacks.add(tuple(curp))
            _collect_leaf_paths(v_def, p)
        else:
            v_user = user[k]
            if isinstance(v_def, dict) and isinstance(v_user, dict):
                m, f = _deep_merge_with_fallbacks(v_def, v_user, p)
                merged[k] = m
                fallbacks |= f
            else:
                merged[k] = v_user
    for k, v_user in user.items():
        if k not in merged:
            merged[k] = v_user
    return merged, fallbacks

def _get_by_path(cfg: dict, path: Tuple[str, ...]):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def warn_fallbacks(
    final_cfg: dict,
    initial_defaults: dict,
    initial_fallback_paths: Set[Tuple[str, ...]],
    relevant_prefixes: Tuple[str, ...],
    command: str,
):
    if not initial_fallback_paths:
        return

    filtered_fallback_paths = set()
    for p in initial_fallback_paths:
        if any(p[0] == rp for rp in relevant_prefixes if p):
            filtered_fallback_paths.add(p)
    
    if not filtered_fallback_paths:
        return

    used_fallbacks = []
    for p in sorted(filtered_fallback_paths):
        v_final = _get_by_path(final_cfg, p)
        v_def = _get_by_path(initial_defaults, p)
        
        # Only warn if it wasn't overridden AND it isn't REQUIRED (validation handles REQUIRED)
        if v_final == v_def and v_final is not REQUIRED:
            used_fallbacks.append((p, v_def))
    
    if not used_fallbacks:
        return
    
    print(
        f"[WARN] [{command}] Using fallback values from internal defaults "
        f"(not provided via config or CLI):"
    )
    for p, v in used_fallbacks:
        dotted = ".".join(p)
        if isinstance(v, (dict, list)):
            print(f"  - {dotted}: <{type(v).__name__}>")
        else:
            print(f"  - {dotted}: {v}")

def validate_config(cfg: dict, command: str):
    """Recursively checks if any value in the config is still set to REQUIRED."""
    missing_keys = []

    def _check(node, path_prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _check(v, f"{path_prefix}.{k}" if path_prefix else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _check(v, f"{path_prefix}[{i}]")
        else:
            if node is REQUIRED:
                missing_keys.append(path_prefix)

    _check(cfg)
    
    if missing_keys:
        print(f"\n[ERROR] [{command}] The following required configuration keys are missing:")
        for k in missing_keys:
            print(f"  - {k}")
        print("Please provide them via a config file (-c) or CLI arguments.")
        sys.exit(1)

def maybe_print_config(print_config: bool, cfg: dict, command: str):
    if print_config:
        print(
            f"\n[{command}] Resolved configuration:\n"
            + json.dumps(cfg, indent=2, ensure_ascii=False, default=str)
        )