# train_llm.py
from __future__ import annotations
import argparse
import os
import yaml
from common import (
    setup_environment, warn_fallbacks, validate_config, maybe_print_config, 
    add_distributed_args, set_dist_env_from_args, str_to_bool, 
    _deep_merge_with_fallbacks, REQUIRED, default_device_from_env
)

def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "training": {
            "implementation": "hydra_gemma_from_flex",
            "model_name_or_path": REQUIRED,
            "tokenizer_name_or_path": None,
            "output_dir": REQUIRED, 
            "seq_len": 3072,
            "num_samples": 120000000,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.01,
            "warmup_steps": None,
            "max_steps": 20000,
            "max_grad_norm_enabled": True,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "exclude_wd_norms_embeddings": False,
            "fp16": False,
            "bf16": True,
            "deepspeed": "train_llm/configs/ds_config_optimized.json",
            "gradient_checkpointing": True,
            "attn_implementation": "flash_attention_2",
            "heads": [1, 2, 3, 4],
            "head_weights": [1, 1, 1, 3],
            "ffn_granularity_ratios": [0.125, 0.25, 0.5, 1.0],
            "ffn_granularity_weights": [1, 1, 1, 3],
            "width_sampling_mode": "global",
            "seed": 42,
            "report_to": ["tensorboard"],
            "logging_dir": None,
            "logging_steps": 1,
            "save_steps": 1000,
            "save_total_limit": 20,
            "loss": {
                "type": "ce_plus_z",
                "z_loss_alpha": 1.0e-5,
            },
            "teacher": {
                "enabled": True,
                "model_name_or_path": "google/gemma-3-270m",
                "implementation": "auto_model",
                "attn_implementation": "auto",
                "dtype": "auto",
                "temperature": 2.0,
                "weight": 0.1,
                "kd_every_n_steps": 1,
                "compile": True,
            },
            "ema": {
                "enabled": True,
                "decay": 0.999,
            },
            "dataset": {
                "source": "hf_streaming",
                "data_dir": "data/dataset",
                "prepare_if_missing": True,
                "hf_dataset": "HuggingFaceFW/fineweb-edu",
                "hf_name": "sample-10BT",
                "use_full": False,
                "hf_split": "train",
                "text_field": "text",
                "cache_dir": ".cache",
                "streaming_buffer_size": 100000,
                "initial_random_skip_tokens": 0,
                "max_blocks": None,
            },
            "dataloader_num_workers": 0,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 2,
            "evaluation": {
                "enabled": False,
                "eval_on": "auto",
                "eval_steps": 5,
                "eval_max_blocks": 512,
                "eval_split_ratio": 0.01,
            },
        },
        "data": {
            "cache_dir": ".cache",
            "seed": 42,
        }
    }

def merge_overrides(cfg, args):
    tr = cfg.setdefault("training", {})
    if args.implementation is not None: tr["implementation"] = args.implementation
    if args.attn_implementation is not None: tr["attn_implementation"] = args.attn_implementation
    if args.model_name_or_path is not None: tr["model_name_or_path"] = args.model_name_or_path
    if args.tokenizer_name_or_path is not None: tr["tokenizer_name_or_path"] = args.tokenizer_name_or_path
    if args.output_dir is not None: tr["output_dir"] = args.output_dir
    if args.seq_len is not None: tr["seq_len"] = args.seq_len
    if args.num_samples is not None: tr["num_samples"] = args.num_samples
    if args.per_device_train_batch_size is not None: tr["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None: tr["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate is not None: tr["learning_rate"] = args.learning_rate
    if args.warmup_steps is not None: tr["warmup_steps"] = args.warmup_steps
    if args.max_steps is not None: tr["max_steps"] = args.max_steps
    if args.logging_steps is not None: tr["logging_steps"] = args.logging_steps
    if args.save_steps is not None: tr["save_steps"] = args.save_steps
    if args.save_total_limit is not None: tr["save_total_limit"] = args.save_total_limit
    if args.fp16 is not None: tr["fp16"] = str_to_bool(args.fp16)
    if args.bf16 is not None: tr["bf16"] = str_to_bool(args.bf16)
    if args.deepspeed is not None: tr["deepspeed"] = args.deepspeed
    if args.heads is not None: tr["heads"] = args.heads
    if args.head_weights is not None: tr["head_weights"] = args.head_weights
    if args.ffn_granularity_ratios is not None:
        tr["ffn_granularity_ratios"] = args.ffn_granularity_ratios
    if args.ffn_granularity_weights is not None:
        tr["ffn_granularity_weights"] = args.ffn_granularity_weights
    if args.width_sampling_mode is not None:
        tr["width_sampling_mode"] = args.width_sampling_mode
    if args.seed is not None: tr["seed"] = args.seed
    if args.gradient_checkpointing is not None: tr["gradient_checkpointing"] = str_to_bool(args.gradient_checkpointing)
    
    ds = tr.setdefault("dataset", {})
    if args.dataset_source is not None: ds["source"] = args.dataset_source
    if args.dataset_data_dir is not None: ds["data_dir"] = args.dataset_data_dir
    if args.dataset_cache_dir is not None: ds["cache_dir"] = args.dataset_cache_dir
    if args.dataset_hf is not None: ds["hf_dataset"] = args.dataset_hf
    if args.dataset_hf_name is not None: ds["hf_name"] = args.dataset_hf_name
    if args.dataset_hf_split is not None: ds["hf_split"] = args.dataset_hf_split
    if args.dataset_text_field is not None: ds["text_field"] = args.dataset_text_field
    
    if args.dataloader_num_workers is not None: tr["dataloader_num_workers"] = args.dataloader_num_workers
    if args.dataloader_persistent_workers is not None: tr["dataloader_persistent_workers"] = str_to_bool(args.dataloader_persistent_workers)
    if args.dataloader_prefetch_factor is not None: tr["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    if args.exclude_wd_norms_embeddings is not None: tr["exclude_wd_norms_embeddings"] = str_to_bool(args.exclude_wd_norms_embeddings)
    if args.kd_every_n_steps is not None: tr.setdefault("teacher", {})["kd_every_n_steps"] = int(args.kd_every_n_steps)
    if args.teacher_compile is not None: 
        tr.setdefault("teacher", {})["compile"] = str_to_bool(args.teacher_compile)
    return cfg

def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="HydraViT text-only training.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")
    
    # Training specific args
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--attn-implementation", type=str)
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--tokenizer-name-or-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--fp16", type=str)
    parser.add_argument("--bf16", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--heads", type=int, nargs="+", action="extend")
    parser.add_argument("--head-weights", type=float, nargs="+", action="extend")
    parser.add_argument(
        "--ffn-granularity-ratios", type=float, nargs="+", action="extend"
    )
    parser.add_argument(
        "--ffn-granularity-weights", type=float, nargs="+", action="extend"
    )
    parser.add_argument(
        "--width-sampling-mode",
        choices=("global", "per_layer"),
        help="Sample one FFN width globally or independently per decoder layer.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gradient-checkpointing", type=str)
    
    # Dataset args
    parser.add_argument("--dataset-source", type=str)
    parser.add_argument("--dataset-data-dir", type=str)
    parser.add_argument("--dataset-cache-dir", type=str)
    parser.add_argument("--dataset-hf", type=str)
    parser.add_argument("--dataset-hf-name", type=str)
    parser.add_argument("--dataset-hf-split", type=str)
    parser.add_argument("--dataset-text-field", type=str)
    
    # Dataloader args
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--dataloader-persistent-workers", type=str)
    parser.add_argument("--dataloader-prefetch-factor", type=int)
    parser.add_argument("--exclude-wd_norms_embeddings", type=str)
    parser.add_argument("--kd-every-n-steps", type=int)
    parser.add_argument("--teacher-compile", type=str, help="Whether to torch.compile the teacher model")

    args = parser.parse_args()
    set_dist_env_from_args(args)

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(initial_defaults, user_cfg)
    final_cfg = merge_overrides(initial_merged_cfg, args)
    
    validate_config(final_cfg, "train-llm")
    warn_fallbacks(final_cfg, initial_defaults, initial_fallback_paths, ("project", "training", "data"), "train-llm")
    
    final_cfg = default_device_from_env(final_cfg, section="inference")
    maybe_print_config(args.print_config, final_cfg, "train-llm")
    
    implementation = final_cfg.get("training", {}).get("implementation")
    if implementation in ("hydra_gemma", "hydra_gemma_from_flex"):
        from train_llm.train_hydravit import run_training_from_cfg
        run_training_from_cfg(final_cfg)
    elif implementation == "matformer_gemma":
        from train_llm.train_matformer import run_training_from_cfg
        run_training_from_cfg(final_cfg)
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")

if __name__ == "__main__":
    main()
