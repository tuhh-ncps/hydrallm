# prepare_dataset.py
from __future__ import annotations
import argparse
import os
import yaml
from common import (
    setup_environment, warn_fallbacks, validate_config, maybe_print_config, 
    add_distributed_args, set_dist_env_from_args, _deep_merge_with_fallbacks, REQUIRED
)

try:
    from train_llm.train_dataset import materialize_packed_dataset_to_disk
except ImportError:
    def materialize_packed_dataset_to_disk(*args, **kwargs): pass

def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "training": {
            "dataset": {
                "source": "packed_disk",
                "data_dir": REQUIRED, 
                "prepare_if_missing": True,
                "hf_dataset": REQUIRED,
                "hf_name": REQUIRED,
                "use_full": False,
                "hf_split": "train",
                "text_field": "text",
                "cache_dir": ".cache",
                "streaming_buffer_size": 100000,
                "initial_random_skip_tokens": 0,
                "max_blocks": None,
                "num_proc": 1, # Default in config
            },
            "seq_len": 3072,
            "num_samples": 120000000,
            "seed": 42,
            "tokenizer_name_or_path": None,
            "model_name_or_path": "google/gemma-3-270m",
        },
        "inference": {
             "model_path": "google/gemma-3-270m",
        },
        "data": {
            "cache_dir": ".cache",
            "seed": 42,
        }
    }

def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="Materialize dataset.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")
    # CLI arg defaults to None so we can tell if it was set
    parser.add_argument("--num-proc", type=int, default=None, help="Number of processes for dataset preparation")
    
    args = parser.parse_args()
    set_dist_env_from_args(args)

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(initial_defaults, user_cfg)
    final_cfg = initial_merged_cfg 
    
    validate_config(final_cfg, "prepare-dataset")
    warn_fallbacks(final_cfg, initial_defaults, initial_fallback_paths, ("project", "training", "data"), "prepare-dataset")
    maybe_print_config(args.print_config, final_cfg, "prepare-dataset")
    
    tr = final_cfg.get("training", {})
    ds_cfg = tr.get("dataset", {})
    
    # Priority: CLI > Config > Default(1)
    num_proc = 1
    if args.num_proc is not None:
        num_proc = args.num_proc
    elif "num_proc" in ds_cfg:
        num_proc = int(ds_cfg["num_proc"])
    
    materialize_packed_dataset_to_disk(
        out_dir=ds_cfg["data_dir"],
        hf_dataset=ds_cfg["hf_dataset"],
        hf_name=None if ds_cfg["use_full"] else ds_cfg["hf_name"],
        hf_split=ds_cfg["hf_split"],
        cache_dir=ds_cfg["cache_dir"] or final_cfg.get("data", {}).get("cache_dir"),
        text_field=ds_cfg["text_field"],
        tokenizer_name=tr.get("tokenizer_name_or_path") or tr.get("model_name_or_path") or final_cfg.get("inference", {}).get("model_path"),
        seq_len=int(tr.get("seq_len", 2048)),
        buffer_size=int(ds_cfg.get("streaming_buffer_size", 100000)),
        seed=int(tr.get("seed", 42)),
        num_samples=int(tr.get("num_samples", 0) or 0),
        initial_skip_tokens=int(ds_cfg.get("initial_random_skip_tokens", 0)),
        max_blocks=ds_cfg.get("max_blocks", None),
        num_proc=num_proc,
    )

if __name__ == "__main__":
    main()