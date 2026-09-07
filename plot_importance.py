from __future__ import annotations

import argparse
import os

import yaml

from common import (
    REQUIRED,
    _deep_merge_with_fallbacks,
    default_device_from_env,
    maybe_print_config,
    setup_environment,
    validate_config,
    warn_fallbacks,
)

try:
    from sorting_llm.common import make_dirs_for_config as make_dirs_for_plotting
    from sorting_llm.post_analyze import run_single_model_analysis
except ImportError:

    def run_single_model_analysis(*args, **kwargs):
        pass

    def make_dirs_for_plotting(cfg, command):
        pass


_METRIC_HELP_HEADS = (
    "magnitude | variance_x_consumers | cett | cett_normalized | "
    "cett_variance | None"
)
_METRIC_HELP_NEURONS = _METRIC_HELP_HEADS
_METRIC_HELP_EMBEDDINGS = "magnitude | variance_x_consumers | None"


def get_command_defaults():
    return {
        "project": {
            "outputs_dir": "outputs",
        },
        "plot_importance": {
            "model_path": REQUIRED,
            "out_dir": REQUIRED,
            "calibration_file": "data/calibration/calibration.txt",
            "chunk_len": 2048,
            "device": "auto",
            "dtype": "auto",
            "heads_metric": "magnitude",
            "neurons_metric": "magnitude",
            "embeddings_metric": "variance_x_consumers",
            "cett_tail_prune_fracs": [0.25, 0.50, 0.75, 0.90],
            "cett_error_bound": 0.2,
            "plot_scales": None,
        },
    }


def _parse_none_str(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def merge_overrides(cfg, args):
    plot_cfg = cfg.setdefault("plot_importance", {})
    if args.model_path is not None:
        plot_cfg["model_path"] = args.model_path
    if args.out_dir is not None:
        plot_cfg["out_dir"] = args.out_dir
    if args.calibration_file is not None:
        plot_cfg["calibration_file"] = args.calibration_file
    if args.chunk_len is not None:
        plot_cfg["chunk_len"] = args.chunk_len
    if args.device is not None:
        plot_cfg["device"] = args.device
    if args.dtype is not None:
        plot_cfg["dtype"] = args.dtype
    if args.heads_metric is not None:
        plot_cfg["heads_metric"] = _parse_none_str(args.heads_metric)
    if args.neurons_metric is not None:
        plot_cfg["neurons_metric"] = _parse_none_str(args.neurons_metric)
    if args.embeddings_metric is not None:
        plot_cfg["embeddings_metric"] = _parse_none_str(args.embeddings_metric)
    return cfg


def main():
    setup_environment()
    parser = argparse.ArgumentParser(
        description="Calculate and plot importance metrics for a model."
    )
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--calibration-file", type=str)
    parser.add_argument("--chunk-len", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--heads-metric", type=str, help=_METRIC_HELP_HEADS)
    parser.add_argument("--neurons-metric", type=str, help=_METRIC_HELP_NEURONS)
    parser.add_argument(
        "--embeddings-metric", type=str, help=_METRIC_HELP_EMBEDDINGS
    )
    args = parser.parse_args()

    initial_defaults = get_command_defaults()
    user_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as config_file:
            user_cfg = yaml.safe_load(config_file) or {}

    initial_merged_cfg, initial_fallback_paths = _deep_merge_with_fallbacks(
        initial_defaults, user_cfg
    )
    final_cfg = merge_overrides(initial_merged_cfg, args)
    validate_config(final_cfg, "plot-importance")
    warn_fallbacks(
        final_cfg,
        initial_defaults,
        initial_fallback_paths,
        ("project", "plot_importance"),
        "plot-importance",
    )

    final_cfg = default_device_from_env(final_cfg, section="plot_importance")
    make_dirs_for_plotting(final_cfg, command="plot-importance")
    maybe_print_config(args.print_config, final_cfg, "plot-importance")

    plot_cfg = final_cfg["plot_importance"]
    run_single_model_analysis(
        model_path=plot_cfg["model_path"],
        out_dir=plot_cfg["out_dir"],
        calibration_path=plot_cfg["calibration_file"],
        device=plot_cfg["device"],
        dtype_str=plot_cfg["dtype"],
        chunk_tokens=int(plot_cfg["chunk_len"]),
        heads_metric=plot_cfg.get("heads_metric"),
        neurons_metric=plot_cfg.get("neurons_metric"),
        embeddings_metric=plot_cfg.get("embeddings_metric"),
        cett_tail_prune_fracs=plot_cfg.get("cett_tail_prune_fracs"),
        cett_error_bound=plot_cfg.get("cett_error_bound"),
        plot_scales=plot_cfg.get("plot_scales"),
    )


if __name__ == "__main__":
    main()
