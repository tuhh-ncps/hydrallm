# infer.py
from __future__ import annotations
import argparse
import os
import yaml
from common import (
    setup_environment,
    warn_fallbacks,
    validate_config,
    maybe_print_config,
    add_distributed_args,
    set_dist_env_from_args,
    str_to_bool,
    _deep_merge_with_fallbacks,
    REQUIRED,
    default_device_from_env,
)

try:
    from inference import run_inference_from_cfg
except ImportError:

    def run_inference_from_cfg(cfg):
        pass


def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "inference": {
            "model_path": REQUIRED,
            "tokenizer_path": None,
            "device": "cpu",
            "dtype": "auto",
            "implementation": "hydra_gemma_from_flex",
            "attn_implementation": "auto",
            "use_k_heads": True,
            "k_heads": 4,
            "ffn_granularity_ratio": None,
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": True,
            "context_size": 1024,
            "prompt": "Once upon a time",
            "image_paths": None,
        },
    }


def merge_overrides(cfg, args):
    inf = cfg.setdefault("inference", {})
    if args.model_path is not None:
        inf["model_path"] = args.model_path
    if args.tokenizer_path is not None:
        inf["tokenizer_path"] = args.tokenizer_path
    if args.device is not None:
        inf["device"] = args.device
    if args.dtype is not None:
        inf["dtype"] = args.dtype
    if args.implementation is not None:
        inf["implementation"] = args.implementation
    if args.attn_implementation is not None:
        inf["attn_implementation"] = args.attn_implementation
    if args.k_heads is not None:
        inf["k_heads"] = args.k_heads
    if args.use_k_heads is not None:
        inf["use_k_heads"] = str_to_bool(args.use_k_heads)
    if args.ffn_granularity_ratio is not None:
        values = args.ffn_granularity_ratio
        inf["ffn_granularity_ratio"] = values[0] if len(values) == 1 else values
    if args.max_new_tokens is not None:
        inf["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        inf["temperature"] = args.temperature
    if args.top_p is not None:
        inf["top_p"] = args.top_p
    if args.top_k is not None:
        inf["top_k"] = args.top_k
    if args.do_sample is not None:
        inf["do_sample"] = str_to_bool(args.do_sample)
    if args.context_size is not None:
        inf["context_size"] = args.context_size
    if args.prompt is not None:
        inf["prompt"] = args.prompt
    if args.image_paths is not None:
        inf["image_paths"] = args.image_paths
    return cfg


def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="HydraViT inference.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")

    parser.add_argument("--model-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--attn-implementation", type=str)
    parser.add_argument("--k-heads", type=int)
    parser.add_argument("--use-k-heads", type=str)
    parser.add_argument(
        "--ffn-granularity-ratio",
        type=float,
        nargs="+",
        help="One global FFN ratio or one ratio per decoder layer.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--do-sample", type=str)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--image-paths", nargs="+", type=str)

    args = parser.parse_args()
    set_dist_env_from_args(args)

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(
        initial_defaults, user_cfg
    )
    final_cfg = merge_overrides(initial_merged_cfg, args)

    validate_config(final_cfg, "infer")
    warn_fallbacks(
        final_cfg,
        initial_defaults,
        initial_fallback_paths,
        ("project", "inference"),
        "infer",
    )

    final_cfg = default_device_from_env(final_cfg, section="inference")
    maybe_print_config(args.print_config, final_cfg, "infer")

    run_inference_from_cfg(final_cfg)


if __name__ == "__main__":
    main()
