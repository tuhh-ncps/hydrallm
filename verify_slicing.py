# verify_slicing.py
from __future__ import annotations
import argparse
import os
import yaml
from common import (
    setup_environment, warn_fallbacks, validate_config, maybe_print_config, 
    add_distributed_args, set_dist_env_from_args, _deep_merge_with_fallbacks, REQUIRED
)

try:
    from util.verify_slicing import run_slicing_verification
except ImportError:
    def run_slicing_verification(cfg): pass

def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "verification": {
            "model_path": REQUIRED,
            "implementation": "hydra_gemma_from_flex",
            "k_heads": 4,
            "prompt": "The quick brown fox jumps over the lazy dog.",
        },
    }

def merge_overrides(cfg, args):
    v_cfg = cfg.setdefault("verification", {})
    if args.model_path is not None: v_cfg["model_path"] = args.model_path
    if args.implementation is not None: v_cfg["implementation"] = args.implementation
    if args.k_heads is not None: v_cfg["k_heads"] = args.k_heads
    if args.prompt is not None: v_cfg["prompt"] = args.prompt
    return cfg

def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="Visually inspect model slicing.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")

    parser.add_argument("--model-path", type=str)
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--k-heads", type=int)
    parser.add_argument("--prompt", type=str)

    args = parser.parse_args()
    set_dist_env_from_args(args)

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(initial_defaults, user_cfg)
    final_cfg = merge_overrides(initial_merged_cfg, args)
    
    validate_config(final_cfg, "verify-slicing")
    warn_fallbacks(final_cfg, initial_defaults, initial_fallback_paths, ("project", "verification"), "verify-slicing")
    maybe_print_config(args.print_config, final_cfg, "verify-slicing")
    
    run_slicing_verification(final_cfg)

if __name__ == "__main__":
    main()