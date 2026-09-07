# sort_llm.py
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
    from sorting_llm.core import run_sort_llm
    from sorting_llm.common import (
        make_dirs_for_config as make_dirs_for_sorting,
    )
except ImportError:

    def run_sort_llm(cfg):
        pass

    def make_dirs_for_sorting(cfg, command):
        pass


_METRIC_HELP_HEADS = (
    "Metric for attention heads: "
    "magnitude | variance_x_consumers | cett | cett_normalized | cett_variance | None"
)
_METRIC_HELP_NEURONS = (
    "Metric for MLP neurons: "
    "magnitude | variance_x_consumers | cett | cett_normalized | cett_variance | None"
)
_METRIC_HELP_EMB = "Metric for embeddings: " "magnitude | variance_x_consumers | None"


def get_command_defaults():
    return {
        "project": {
            "data_dir": "data",
            "outputs_dir": "outputs",
            "tmp_dir": ".cache",
        },
        "sorting": {
            "model_id": REQUIRED,
            "save_dir": REQUIRED,
            "dtype": "auto",
            "device": "auto",
            "device_map": "auto",
            "calibration_file": "data/calibration/calibration.txt",
            "chunk_len": 2048,
            "add_special_tokens": False,
            # Granular metrics
            "heads_metric": "magnitude",
            "neurons_metric": "magnitude",
            "embeddings_metric": "variance_x_consumers",
            "top_k": 3,
            "postcheck": True,
            # CETT tail curve
            "cett_tail_curve": True,
            "cett_error_bound": 0.2,
            "cett_tail_prune_fracs": [0.25, 0.50, 0.75, 0.90],
            # Plot scale controls for postcheck/analysis
            "plot_scales": {
                "heads_heatmap": {"vmin": None, "vmax": None},
                "heads_bar": {"ymin": None, "ymax": None},
                "neurons_line": {"ymin": None, "ymax": None, "log": True},
                "neurons_group_heatmap": {"vmin": None, "vmax": None},
                "neurons_grouped_bar": {"ymin": None, "ymax": None},
                "embeddings_line": {"ymin": None, "ymax": None, "log": False},
                "embeddings_grouped_bar": {"ymin": None, "ymax": None},
            },
        },
    }


def _parse_none_str(val):
    if val is None:
        return None
    if isinstance(val, str) and val.lower() == "none":
        return None
    return val


def merge_overrides(cfg, args):
    s_cfg = cfg.setdefault("sorting", {})
    if args.model_id is not None:
        s_cfg["model_id"] = args.model_id
    if args.save_dir is not None:
        s_cfg["save_dir"] = args.save_dir
    if args.device is not None:
        s_cfg["device"] = args.device
    if args.dtype is not None:
        s_cfg["dtype"] = args.dtype
    if args.device_map is not None:
        s_cfg["device_map"] = args.device_map
    if args.calibration_file is not None:
        s_cfg["calibration_file"] = args.calibration_file
    if args.chunk_len is not None:
        s_cfg["chunk_len"] = args.chunk_len
    if args.top_k is not None:
        s_cfg["top_k"] = int(args.top_k)
    if args.heads_metric is not None:
        s_cfg["heads_metric"] = _parse_none_str(args.heads_metric)
    if args.neurons_metric is not None:
        s_cfg["neurons_metric"] = _parse_none_str(args.neurons_metric)
    if args.embeddings_metric is not None:
        s_cfg["embeddings_metric"] = _parse_none_str(args.embeddings_metric)
    if args.postcheck is not None:
        s_cfg["postcheck"] = str_to_bool(args.postcheck)
    return cfg


def main():
    setup_environment()
    parser = argparse.ArgumentParser(description="Compute importance, reorder, save.")
    add_distributed_args(parser)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--device-map", type=str)
    parser.add_argument("--calibration-file", type=str)
    parser.add_argument("--chunk-len", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--heads-metric", type=str, help=_METRIC_HELP_HEADS)
    parser.add_argument("--neurons-metric", type=str, help=_METRIC_HELP_NEURONS)
    parser.add_argument("--embeddings-metric", type=str, help=_METRIC_HELP_EMB)
    parser.add_argument("--postcheck", type=str)
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
    validate_config(final_cfg, "sort-llm")
    warn_fallbacks(
        final_cfg,
        initial_defaults,
        initial_fallback_paths,
        ("project", "sorting"),
        "sort-llm",
    )
    final_cfg = default_device_from_env(final_cfg, section="sorting")
    make_dirs_for_sorting(final_cfg, command="sort-llm")
    maybe_print_config(args.print_config, final_cfg, "sort-llm")
    run_sort_llm(final_cfg)


if __name__ == "__main__":
    main()
