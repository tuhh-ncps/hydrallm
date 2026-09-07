# lighteval.py
from __future__ import annotations
import os
import argparse
import yaml
from common import (
    setup_environment, warn_fallbacks, validate_config, maybe_print_config, 
    add_distributed_args, set_dist_env_from_args, str_to_bool, 
    _deep_merge_with_fallbacks, REQUIRED
)

# Lazy imports handled via try/except in main or top-level if strictly required
try:
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from evaluation.custom_transformer_model_config import CustomTransformersModelConfig
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
except ImportError:
    # We don't error immediately to allow --help to work, but main execution will fail
    pass

def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "lighteval": {
            "tasks": "leaderboard|hellaswag|10,lighteval|arc:easy|0,lighteval|triviaqa|5",
            "output_dir": None, # Will auto-resolve if None
            "launcher_type": "ACCELERATE",
            "max_samples": None,
            "custom": True,
            "model": {
                # We make model_name REQUIRED because the user's default was a specific checkpoint path
                "model_name": REQUIRED, 
                "tokenizer": None,
                "subfolder": None,
                "revision": "main",
                "batch_size": 1,
                "max_length": None,
                "add_special_tokens": True,
                "skip_special_tokens": True,
                "model_parallel": False,
                "dtype": "bfloat16",
                "device": "cuda",
                "trust_remote_code": False,
                "compile": False,
                "override_chat_template": None,
                "pairwise_tokenization": False,
                "continuous_batching": False,
                "implementation": "hydra_gemma_from_flex",
                "attn_implementation": "flash_attention_2",
                "current_num_heads": None,
                "ffn_granularity_ratio": None,
                "generation_parameters": {
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "top_k": 64,
                },
            },
        },
    }

def merge_overrides(cfg, args):
    le = cfg.setdefault("lighteval", {})
    if args.tasks is not None: le["tasks"] = args.tasks
    if args.output_dir is not None: le["output_dir"] = args.output_dir
    if args.launcher_type is not None: le["launcher_type"] = args.launcher_type
    if args.max_samples is not None: le["max_samples"] = args.max_samples
    if args.custom is not None: le["custom"] = str_to_bool(args.custom)
    
    m = le.setdefault("model", {})
    if args.model_name is not None: m["model_name"] = args.model_name
    if args.tokenizer is not None: m["tokenizer"] = args.tokenizer
    if args.subfolder is not None: m["subfolder"] = args.subfolder
    if args.revision is not None: m["revision"] = args.revision
    if args.batch_size is not None: m["batch_size"] = args.batch_size
    if args.max_length is not None: m["max_length"] = args.max_length
    if args.add_special_tokens is not None: m["add_special_tokens"] = str_to_bool(args.add_special_tokens)
    if args.skip_special_tokens is not None: m["skip_special_tokens"] = str_to_bool(args.skip_special_tokens)
    if args.model_parallel is not None: m["model_parallel"] = str_to_bool(args.model_parallel)
    if args.dtype is not None: m["dtype"] = args.dtype
    if args.device is not None: m["device"] = args.device
    if args.trust_remote_code is not None: m["trust_remote_code"] = str_to_bool(args.trust_remote_code)
    if args.compile is not None: m["compile"] = str_to_bool(args.compile)
    if args.pairwise_tokenization is not None: m["pairwise_tokenization"] = str_to_bool(args.pairwise_tokenization)
    if args.continuous_batching is not None: m["continuous_batching"] = str_to_bool(args.continuous_batching)
    if args.override_chat_template is not None:
        val = str_to_bool(args.override_chat_template)
        # Logic: if string "null" is passed, set to None, else boolean
        if isinstance(val, bool) is False and str(args.override_chat_template).lower() == "null":
            m["override_chat_template"] = None
        else:
            m["override_chat_template"] = val
            
    if args.implementation is not None: m["implementation"] = args.implementation
    if args.attn_implementation is not None: m["attn_implementation"] = args.attn_implementation
    if args.current_num_heads is not None: m["current_num_heads"] = int(args.current_num_heads)
    if args.ffn_granularity_ratio is not None:
        values = args.ffn_granularity_ratio
        m["ffn_granularity_ratio"] = values[0] if len(values) == 1 else values
    
    gp = m.setdefault("generation_parameters", {})
    if args.max_new_tokens is not None: gp["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None: gp["temperature"] = args.temperature
    if args.top_p is not None: gp["top_p"] = args.top_p
    if args.top_k is not None: gp["top_k"] = args.top_k
    return cfg

def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="Run LightEval benchmarks.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")

    # CLI Overrides
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--launcher-type", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--custom", type=str)
    
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--subfolder", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--add-special-tokens", type=str)
    parser.add_argument("--skip-special-tokens", type=str)
    parser.add_argument("--model-parallel", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--trust-remote-code", type=str)
    parser.add_argument("--compile", type=str)
    parser.add_argument("--pairwise-tokenization", type=str)
    parser.add_argument("--continuous-batching", type=str)
    parser.add_argument("--override-chat-template", type=str)
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--attn-implementation", type=str)
    parser.add_argument("--current-num-heads", type=int)
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

    args = parser.parse_args()
    set_dist_env_from_args(args)

    # Load config logic
    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(initial_defaults, user_cfg)
    final_cfg = merge_overrides(initial_merged_cfg, args)
    
    validate_config(final_cfg, "lighteval")
    warn_fallbacks(final_cfg, initial_defaults, initial_fallback_paths, ("project", "lighteval"), "lighteval")
    maybe_print_config(args.print_config, final_cfg, "lighteval")
    
    # Ensure modules are available before execution
    try:
        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
        from evaluation.custom_transformer_model_config import CustomTransformersModelConfig
        from lighteval.models.transformers.transformers_model import TransformersModelConfig
    except ImportError as e:
        raise RuntimeError(f"LightEval dependencies not found: {e}")

    # Setup Execution
    le = final_cfg["lighteval"]
    model_cfg = le["model"]
    
    launcher = le.get("launcher_type", "ACCELERATE")
    if isinstance(launcher, str):
        launcher = launcher.upper()
    launcher_type = getattr(ParallelismManager, launcher, ParallelismManager.ACCELERATE)
    
    pipeline_params = PipelineParameters(
        launcher_type=launcher_type,
        custom_tasks_directory=None,
        max_samples=le.get("max_samples", None),
    )
    
    # Determine output directory
    out_dir = le.get("output_dir")
    if out_dir is None:
        model_name_or_path = model_cfg.get("model_name")
        if not model_name_or_path:
            out_dir = "outputs/eval/unknown_model"
        elif os.path.exists(model_name_or_path):
            # It's a local path. Save results inside a 'lighteval' subfolder.
            out_dir = os.path.join(model_name_or_path, "lighteval")
        else:
            # It's a HF model ID. Sanitize it for a folder name.
            safe_model_name = model_name_or_path.replace("/", "_")
            out_dir = f"outputs/eval/{safe_model_name}"
    le["output_dir"] = out_dir # Update config for tracking
    
    tracker = EvaluationTracker(
        output_dir=out_dir,
        save_details=True,
        push_to_hub=False,
        results_path_template="{output_dir}/",
    )
    
    tasks = le.get("tasks", "leaderboard|hellaswag|10")
    use_custom = bool(le.get("custom", True))
    
    if use_custom:
        # Build path to our custom model class file
        repo_root = os.path.abspath(os.path.dirname(__file__))
        model_def_path = os.path.join(repo_root, "evaluation", "custom_lighteval_transformer.py")
        
        config_obj = CustomTransformersModelConfig(
            model_name=model_cfg["model_name"],
            tokenizer=model_cfg.get("tokenizer", None),
            subfolder=model_cfg.get("subfolder", None),
            revision=model_cfg.get("revision", "main"),
            batch_size=model_cfg.get("batch_size", None),
            max_length=model_cfg.get("max_length", None),
            model_loading_kwargs=model_cfg.get("model_loading_kwargs", {}),
            add_special_tokens=bool(model_cfg.get("add_special_tokens", True)),
            skip_special_tokens=bool(model_cfg.get("skip_special_tokens", True)),
            model_parallel=bool(model_cfg.get("model_parallel", False)),
            dtype=model_cfg.get("dtype", "bfloat16"),
            device=model_cfg.get("device", "cuda"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            compile=bool(model_cfg.get("compile", False)),
            multichoice_continuations_start_space=None,
            pairwise_tokenization=bool(model_cfg.get("pairwise_tokenization", False)),
            continuous_batching=bool(model_cfg.get("continuous_batching", False)),
            override_chat_template=model_cfg.get("override_chat_template", None),
            implementation=model_cfg.get("implementation", "hydra_gemma_from_flex"),
            attn_implementation=model_cfg.get("attn_implementation", "auto"),
            current_num_heads=model_cfg.get("current_num_heads", None),
            ffn_granularity_ratio=model_cfg.get("ffn_granularity_ratio", None), 
            generation_parameters=model_cfg.get("generation_parameters", {}),
            # required by CustomModelConfig
            model_definition_file_path=model_def_path,
        )
    else:
        config_obj = TransformersModelConfig(
            model_name=model_cfg["model_name"],
            revision=model_cfg.get("revision", "main"),
            dtype=model_cfg.get("dtype", "float16"),
            compile=bool(model_cfg.get("compile", False)),
            model_parallel=bool(model_cfg.get("model_parallel", False)),
            batch_size=model_cfg.get("batch_size", 1),
            continuous_batching=bool(model_cfg.get("continuous_batching", False)),
            model_loading_kwargs=model_cfg.get("model_loading_kwargs", {"attn_implementation": "eager"}),
            generation_parameters=model_cfg.get("generation_parameters", {"temperature": 0.0}),
        )

    # Create and Run Pipeline
    pipeline = Pipeline(
        tasks=tasks,
        model_config=config_obj,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=tracker,
    )
    
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()
