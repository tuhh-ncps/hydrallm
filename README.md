# HydraLLM

HydraLLM provides dynamic-width Gemma 3 models and the surrounding tooling needed to reorder, train, evaluate, and export them. The repository currently supports text-only HydraGemma and MatFormer training, unified HydraGemma text-plus-vision training, dynamic-width inference, LightEval integration, and a CLIP caption benchmark.

## Model Variants

### HydraGemma

`hydra_gemma_from_flex` exposes `current_num_heads` as its primary width control. Unless an explicit FFN ratio is also supplied, one global head count determines:

- attention width;
- hidden and token-embedding width;
- FFN intermediate width;
- multimodal projector width.

The projector follows the head-derived embedding width so projected image tokens and text-token embeddings have the same final dimension. Hydra training samples one head count globally for the model. Runtime paths that accept `ffn_granularity_ratio` can explicitly override the head-derived FFN ratio while leaving the other Hydra dimensions tied to the head count.

### MatFormerGemma

`matformer_gemma` keeps attention, hidden/embedding, and multimodal projector widths full. Only the FFN intermediate width changes.

It accepts two equivalent controls:

- `granularity`: an index into `ffn_granularity_ratios`;
- `mlp_ratio`: the actual FFN ratio.

Both controls may be scalar or per-layer:

```python
# Global: every decoder layer uses half of the full FFN width.
model(input_ids=input_ids, mlp_ratio=0.5)

# Per-layer example for a model with four decoder layers.
model(input_ids=input_ids, mlp_ratio=[0.25, 0.5, 0.5, 1.0])
```

Explicit `mlp_ratio` takes precedence over `granularity`. A per-layer list must contain exactly `model.config.num_hidden_layers` values. Ratios must be greater than `0` and at most `1` in the inference and evaluation entry points.

## Repository Layout

```text
.
├── train_llm.py                 # Text-only HydraGemma/MatFormer training
├── train_unified.py             # Unified text and multimodal HydraGemma training
├── sort_llm.py                  # Structural importance scoring and reordering
├── plot_importance.py           # Standalone importance analysis and plotting
├── infer.py                     # Text and multimodal generation
├── run_lighteval.py             # LightEval integration
├── prepare_dataset.py           # Packed text dataset materialization
├── verify_slicing.py            # Dynamic head-slicing verification
├── model_implementations/       # Flex, Hydra, MatFormer, and auto-model loaders
├── train_llm/                   # Text trainers, datasets, EMA, and callbacks
├── train_unified/               # Unified dataset and Hydra trainer
├── sorting_llm/                 # Importance metrics, reordering, and analysis
├── evaluation/                  # Custom LightEval model and config
├── inference/                   # Inference implementation
├── benchmarks/                  # CLIP caption benchmark
├── configs/                     # Training, sorting, and evaluation configs
├── util/                        # Export, conversion, dataset, and diagnostic tools
└── tests/                       # Focused regression tests
```

## Installation

Python 3.11 or newer is required by the pinned NumPy and SciPy versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned core stack includes PyTorch 2.8, Transformers 4.56, Accelerate 1.10, Datasets 4.2, and LightEval 0.11. DeepSpeed and FlashAttention are intentionally commented out in `requirements.txt`; install compatible versions separately when needed. All provided training recipes select DeepSpeed, and the text-training recipes select FlashAttention. To run without them, explicitly set `deepspeed: null` in the relevant `training` or `train` block and set `attn_implementation: auto`.

An optional Hugging Face token can be placed in a root `.env` file:

```env
HF_TOKEN=hf_xxx
```

## Configuration

Root entry points generally resolve settings in this order:

1. built-in command defaults;
2. YAML passed with `-c` or `--config`;
3. explicit CLI overrides;
4. required-field validation.

Configuration paths are resolved relative to the repository root in the provided recipes. Required machine-specific model paths are left as commented examples in the unified-training, LightEval, and standalone plotting configs; set them in YAML or with the corresponding CLI option.

Use `--print-config` to inspect the resolved configuration. It prints and then continues execution; it is not a validation-only or dry-run flag. Unknown YAML keys are not rejected, so check spelling carefully.

## Text-Only Training

`train_llm.py` routes these implementations:

- `hydra_gemma_from_flex` to `HydraTrainer`;
- `matformer_gemma` to `MatFormerTrainer`.

Both trainers keep the sampled architecture fixed across every microbatch in a gradient-accumulation window. They support CE/z-loss, optional knowledge distillation, EMA, packed-disk or Hugging Face streaming datasets, and optional DeepSpeed. Runs use timestamped output directories. To resume an existing run, set `training.timestamp_override` to that run's timestamp; the trainer then resumes from the latest checkpoint under its `model/` directory.

### HydraGemma training

```yaml
training:
  implementation: hydra_gemma_from_flex
  model_name_or_path: google/gemma-3-270m
  output_dir: outputs/final_trained_llms/hydra
  seq_len: 3072
  num_samples: 2000000
  max_steps: 20000
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  bf16: true
  gradient_checkpointing: true
  heads: [1, 2, 3, 4]
  head_weights: [1, 1, 1, 3]
  dataset:
    source: packed_disk
    data_dir: data/fineweb/edu_350BT_3072
    hf_dataset: HuggingFaceFW/fineweb-edu
    hf_name: sample-350BT
    hf_split: train
    text_field: text
```

Run a provided configuration:

```bash
python train_llm.py -c configs/train_llm/pt_hydra_default.yaml
```

### MatFormer training

MatFormer supports two sampling modes:

- `global` is the default and samples one granularity for all decoder layers;
- `per_layer` independently samples one granularity for each decoder layer.

```yaml
training:
  implementation: matformer_gemma
  model_name_or_path: google/gemma-3-270m
  output_dir: outputs/final_trained_llms/matformer
  seq_len: 3072
  num_samples: 2000000
  max_steps: 20000
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  bf16: true
  gradient_checkpointing: true
  ffn_granularity_ratios: [0.125, 0.25, 0.5, 1.0]
  ffn_granularity_weights: [1, 1, 1, 3]
  width_sampling_mode: global
  dataset:
    source: packed_disk
    data_dir: data/fineweb/edu_350BT_3072
    hf_dataset: HuggingFaceFW/fineweb-edu
    hf_name: sample-350BT
    hf_split: train
    text_field: text
```

Enable per-layer sampling in YAML:

```yaml
training:
  width_sampling_mode: per_layer
```

Or through the root CLI:

```bash
python train_llm.py \
  -c configs/train_llm/pt_matformer_default.yaml \
  --width-sampling-mode per_layer \
  --ffn-granularity-ratios 0.125 0.25 0.5 1.0 \
  --ffn-granularity-weights 1 1 1 3
```

In per-layer mode, the trainer samples granularity indices and the model translates them into one FFN ratio per decoder layer. MatFormer attention, embeddings, and the multimodal projector remain full-width in both modes.

### DeepSpeed

DeepSpeed configurations are under `train_llm/configs/`. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --master_port=29501 \
  --include localhost:0,1 \
  train_llm.py -c configs/train_llm/pt_hydra_default.yaml
```

The provided command also requires a compatible FlashAttention installation. For a non-DeepSpeed fallback, set `deepspeed: null`; use `attn_implementation: auto` to let the runtime select FlashAttention, SDPA, or eager attention based on availability.

## Assemble a Multimodal HydraGemma Checkpoint

Before unified multimodal training, combine a Gemma 3 text checkpoint with the vision tower and projector from a multimodal Gemma 3 checkpoint:

```bash
python util/create_mm_hydra_gemma.py \
  --text-model google/gemma-3-270m \
  --mm-source google/gemma-3-4b-pt \
  --output outputs/assembled/gemma-3-270m-mm
```

The script builds the checkpoint on CPU. It uses the text model's configuration, language-model weights, LM head, and tokenizer; takes the multimodal source's vision configuration, vision-tower weights, projector weights, and image processor; and slices or zero-pads the projector input columns to match the text hidden size. Add `--local-files-only` when both source checkpoints are already cached.

Use the output directory as `train.model_name_or_path` in a unified-training recipe or pass it with `--model-name-or-path`.

## Unified Text and Multimodal Training

`train_unified.py` trains `hydra_gemma_from_flex` on a combined Parquet dataset containing text-only and image-text rows. It currently uses global Hydra head sampling, not MatFormer per-layer FFN sampling.

The pipeline supports independently trainable components:

- `projection`;
- `text_model`;
- `vision_encoder`.

It also supports per-component learning rates and warmup, optional EMA and KD, gradient accumulation, synchronized modality selection across distributed ranks, and text-only or multimodal batches.

```bash
python train_unified.py -c configs/train_unified/align_projection.yaml
python train_unified.py -c configs/train_unified/joint_projection_and_llm_training.yaml
python train_unified.py -c configs/train_unified/full_model_training.yaml
```

These recipes represent projection-only, projection-plus-text, and full projection/text/vision training respectively. Each leaves the required `train.model_name_or_path` as a commented example; uncomment and update it, or pass `--model-name-or-path`, before running.

The dataset block points to Parquet data and an image root:

```yaml
train:
  implementation: hydra_gemma_from_flex
  multimodal: true
  heads: [1, 2, 3, 4]
  head_weights: [1, 1, 1, 1]
  train_model_parts: [projection, text_model]
  dataset:
    # Replace these example paths with a built unified dataset.
    parquet_path: data/unified_dataset/parquet
    split: train
    images_root: data/unified_dataset/images
```

Build compatible datasets with `util/mm_dataset_builders/build_unified_dataset.py` and recipes under `util/mm_dataset_builders/`. The checked-in builder recipes contain machine-specific source and output paths and must be customized first.

```bash
python util/mm_dataset_builders/build_unified_dataset.py \
  --config util/mm_dataset_builders/dataset_experiments.yaml
```

## Inference

`infer.py` supports text-only and multimodal generation. Hydra head width is controlled with `--use-k-heads` and `--k-heads`. MatFormer FFN width is controlled with `--ffn-granularity-ratio`.

Head control currently defaults to enabled with `k_heads: 4`. Pass `--use-k-heads false` for MatFormer or for unsliced full-head inference. When selecting a Hydra width, use `1 <= k_heads <= model.config.num_attention_heads`; the inference CLI does not currently validate this range.

With no FFN ratio, inference runs at full FFN width. One value is a global ratio:

```bash
python infer.py \
  --model-path outputs/final_trained_llms/matformer/model \
  --implementation matformer_gemma \
  --device cuda \
  --use-k-heads false \
  --ffn-granularity-ratio 0.5 \
  --prompt "Explain elastic inference."
```

Multiple values specify one ratio per decoder layer. This abbreviated example assumes a checkpoint with four decoder layers:

```bash
python infer.py \
  --model-path outputs/final_trained_llms/matformer/model \
  --implementation matformer_gemma \
  --device cuda \
  --use-k-heads false \
  --ffn-granularity-ratio 0.25 0.5 0.5 1.0 \
  --prompt "Explain elastic inference."
```

The list is validated against `num_hidden_layers`. A one-element list is normalized to a global scalar. Real Gemma checkpoints generally have more than four layers, so provide the complete list for the checkpoint being loaded.

Hydra example:

```bash
python infer.py \
  --model-path outputs/final_trained_llms/hydra/model \
  --implementation hydra_gemma_from_flex \
  --device cuda \
  --use-k-heads true \
  --k-heads 3 \
  --prompt "Explain dynamic head slicing."
```

For multimodal inference, include one `<start_of_image>` placeholder per image:

```bash
python infer.py \
  --model-path outputs/trained_models/multimodal/model \
  --implementation hydra_gemma_from_flex \
  --device cuda \
  --prompt "<start_of_image>\nDescribe this image." \
  --image-paths path/to/image.jpg
```

## LightEval

`run_lighteval.py` uses the custom implementation in `evaluation/` when `lighteval.custom` is true. It forwards Hydra `current_num_heads` and MatFormer `ffn_granularity_ratio` during log-likelihood evaluation and padded generation. Unsupported model implementations are retried without these custom kwargs. Setting `custom: false` uses LightEval's upstream Transformers wrapper and does not apply the dynamic-width controls.

```yaml
lighteval:
  tasks: "leaderboard|hellaswag|10"
  custom: true
  model:
    model_name: outputs/final_trained_llms/matformer/model
    implementation: matformer_gemma
    dtype: bfloat16
    device: cuda
    current_num_heads: null
    ffn_granularity_ratio: 0.5
```

`ffn_granularity_ratio` may also be a per-layer YAML list. The default is `null`, which means full model width. The following command is illustrative for a four-layer checkpoint; production commands must provide exactly one value per actual decoder layer.

```bash
python run_lighteval.py \
  -c configs/lighteval_config.yaml \
  --model-name outputs/final_trained_llms/matformer/model \
  --implementation matformer_gemma \
  --ffn-granularity-ratio 0.25 0.5 0.5 1.0
```

Dynamic-width kwargs are not propagated by the continuous-batching LightEval path; keep `continuous_batching: false` when evaluating a sliced width.

On the custom path, device placement comes from Accelerate and the launch environment rather than directly from `lighteval.model.device`.

## Benchmarks

### CLIP caption benchmark

The CLIP benchmark generates captions and scores them with a CLIP model:

```bash
python benchmarks/clip/run_benchmark.py \
  --model_path outputs/trained_models/multimodal/model \
  --image_dir data/clip_images \
  --output_dir outputs/benchmarks/clip/matformer \
  --implementation matformer_gemma \
  --ffn_granularity_ratio 0.5 \
  --dtype bfloat16 \
  --clip_model openai/clip-vit-large-patch14
```

`--ffn_granularity_ratio` accepts either one global value or one value per decoder layer. The selected ratio is recorded in caption metadata and displayed by `benchmarks/clip/print_results.py`.

Hydra runs use `--num_heads` in `run_benchmark.py` or `--current_num_heads` in `run_hydra_gemma3.py`.

## Sorting and Importance Analysis

### Sort and reorder a checkpoint

`sort_llm.py` runs calibration text through a checkpoint, computes structural importance, and physically permutes the checkpoint so important heads, neurons, and embedding dimensions occupy leading tensor prefixes.

```bash
python sort_llm.py \
  -c configs/reorder/reorder_cett_normalized_cett_normalized_magnitude.yaml
```

Head and neuron metrics support `magnitude`, `variance_x_consumers`, `cett`, `cett_normalized`, `cett_variance`, or `null`. Embeddings support `magnitude`, `variance_x_consumers`, or `null`. A `null` metric leaves that component in its original order.

Sorting creates a timestamped directory from `sorting.save_dir`:

```text
<sorting.save_dir>_YYYYMMDD_HHMMSS/
├── sort_config.yaml
├── importance_data.pth
├── model/
├── postcheck_original/           # When postcheck is enabled
├── postcheck_reordered/          # When postcheck is enabled
└── cett_tail_curve.json          # When enabled for cett_normalized
```

CETT tail reports are generated only for head or neuron metrics set to `cett_normalized`. `cett_error_bound` and `cett_tail_prune_fracs` control the reported cumulative-error bounds.

### Plot one checkpoint

`plot_importance.py` performs standalone importance analysis without rewriting the checkpoint. It recalculates scores from the calibration text; it does not read the `importance_data.pth` produced by sorting.

The reorder configs include a `plot_importance` section, but leave `model_path` commented because the sorting directory contains a runtime timestamp. Supply the actual nested `model/` path:

```bash
python plot_importance.py \
  -c configs/reorder/reorder_cett_normalized_cett_normalized_magnitude.yaml \
  --model-path \
  outputs/reordered_llms/reorder_cett_normalized_cett_normalized_magnitude_20260907_143000/model
```

Any compatible local checkpoint or Hugging Face model ID can be analyzed directly:

```bash
python plot_importance.py \
  --model-path google/gemma-3-270m \
  --out-dir outputs/importance_analysis/gemma-3-270m \
  --heads-metric magnitude \
  --neurons-metric cett_normalized \
  --embeddings-metric variance_x_consumers
```

Standalone analysis writes its configuration, CETT reports when applicable, and available plots under `plot_importance.out_dir`. Plot files are emitted as PNG and PDF. `sorting.plot_scales` controls sorting postchecks, while `plot_importance.plot_scales` controls standalone plots. Supported scale groups are `heads_heatmap`, `heads_bar`, `neurons_line`, `neurons_group_heatmap`, `neurons_grouped_bar`, `embeddings_line`, and `embeddings_grouped_bar`; omitted or `null` bounds are derived automatically.

## Dataset Preparation

Materialize packed text data for `training.dataset.source: packed_disk`:

```bash
python prepare_dataset.py \
  -c configs/train_llm/pt_hydra_default.yaml \
  --num-proc 4
```

The multimodal toolkit includes:

- builders in `util/mm_dataset_builders/`;
- downloaders in `util/mm_dataset_downloaders/`;
- validation, cleaning, and shuffling tools in `util/mm_dataset_tools/`.

See the unified-training section for the builder command. Customize the selected recipe's source paths, `output.output_dir`, and image-store settings before running it.

## Export and Diagnostics

Build a physically sliced Hydra checkpoint:

```bash
python util/build_hydra_gemma_with_k_heads.py \
  --src google/gemma-3-270m \
  --dst outputs/converted/gemma3_k1 \
  --heads 1
```

The exporter preserves complete GQA groups, with one head as a special case. An incompatible requested head count is adjusted to a supported value unless `--force` is used; forced output may be mathematically inequivalent or incompatible.

Build a physically sliced MatFormer checkpoint with one global FFN width:

```bash
python util/build_matformer_gemma_with_w_width.py \
  --src google/gemma-3-270m \
  --dst outputs/converted/gemma3_mlp_half \
  --mlp_ratio 0.5
```

The MatFormer export utility creates a conventional checkpoint with one uniform FFN width; it does not export heterogeneous per-layer widths. Both exporters require safetensors input. They copy common tokenizer files but not all multimodal processor artifacts, so copy the source processor/image-preprocessor files separately when producing a self-contained multimodal checkpoint.

Additional tools include:

- `verify_slicing.py` and `util/verify_slicing.py` for Hydra slicing checks;
- `util/eval_ppl_and_loss.py` for text loss/perplexity;
- `util/check_model_equality.py` for checkpoint comparisons;
- `util/create_mm_hydra_gemma.py` for constructing multimodal Hydra checkpoints;
- `util/create_text_model_from_mm.py` for extracting text-only checkpoints.

## Tests

Run the current regression suite with:

```bash
python -m pytest -q
```

The tests cover global and per-layer MatFormer sampling, inference ratio normalization and validation, and mocked forwarding behavior that keeps the multimodal projector full-width while varying MatFormer FFN width.

## License

HydraLLM is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
Third-party packages, model weights, tokenizers, datasets, and evaluation tasks
remain subject to their own licenses and terms.
