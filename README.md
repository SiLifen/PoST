# Optimal Decay Spectra for Linear Recurrences

Official implementation of **PoST (Position-Adaptive Spectral Tapering)** — a principled spectral reparameterization and position-adaptive scaling method for gated linear recurrences.

## Overview

PoST modifies gated linear recurrences in two ways:

1. **Spectral Reparameterization** — replaces free decay parameters with a constrained monotonic parameterization ($\theta + \mathrm{cumsum}(\mathrm{softplus}(\delta))$), ensuring timescales are geometrically spaced from short-range to long-range.
2. **Position-Adaptive Scaling** — scales the effective decay by $t^{-\alpha}$ at position $t$, where $\alpha$ is analytically derived from the learned spectrum. This maintains geometric spacing across all positions, improving extrapolation beyond training length.

### Supported Architectures

| Architecture | Baseline | + PoST |
|---|---|---|
| **Mamba-2** | `mamba2` | `mamba2_post` |
| **RWKV-7** | `rwkv7` | `rwkv7_post` |
| **Gated DeltaNet** | `gdn` | `gdn_post` |

### Model Sizes (LM Pretraining)

| Size | `d_model` | `n_layer` (Mamba-2/GDN) | `n_layer` (RWKV-7) |
|------|-----------|---------------------|---------------------|
| 180M | 768 | 24 | 12 |
| 440M | 1024 | 48 | 24 |
| 1.5B | 2048 | 48 | 24 |

## Setup

### Requirements

- Python >= 3.11
- CUDA 12.4
- [uv](https://docs.astral.sh/uv/)

### Install

```bash
git clone https://github.com/Gunale0926/PoST.git
cd PoST

# Install with uv
uv sync
```

### Configuration

All scripts use the `DATA_DIR` environment variable to locate data and output directories (defaults to `./data`):

```bash
# Set your API tokens
export HF_TOKEN="your_huggingface_token"        # Required for gated datasets/models
export WANDB_API_KEY="your_wandb_api_key"        # Required for W&B logging

# Set data/output root (defaults to ./data if not set)
export DATA_DIR="/path/to/storage"

# HF cache paths are derived from DATA_DIR automatically.
# Override individually if needed:
# export HF_HOME="/custom/hf/cache"
```

## Training

All training is handled by `trainer.py`. It supports multi-GPU via HuggingFace Accelerate, data packing, streaming datasets, and periodic LM-eval during training.

### Quick Start

```bash
uv run accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 \
  trainer.py \
    --arch mamba2_post \
    --model_size 180M \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config sample-10BT \
    --max_seq_length 2048 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 6e-4 \
    --max_tokens 4e9 \
    --packing \
    --output_dir output/mamba2_post \
    --resume
```

### Batch Training Scripts

Pre-configured scripts for training all architectures at each scale:

```bash
# 180M models, ~4B tokens each (Mamba-2, RWKV-7, GDN × baseline + PoST)
bash scripts/run_180m_4b.sh

# 440M models, ~9B tokens each
bash scripts/run_440m_9b.sh

# 1.5B models, ~30B tokens each
bash scripts/run_1.5b_30b.sh

# Train a specific architecture only
bash scripts/run_180m_4b.sh mamba2_post
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--arch` | all | Architecture: `mamba2`, `mamba2_post`, `rwkv7`, `rwkv7_post`, `gdn`, `gdn_post` |
| `--model_size` | `180M` | Size preset: `180M`, `440M`, `880M`, `1.5B` |
| `--max_tokens` | — | Total tokens to train on (e.g. `4e9`) |
| `--packing` | off | Concatenate documents into fixed-length sequences |
| `--streaming` | off | Stream dataset from HuggingFace (no disk caching) |
| `--gradient_checkpointing` | off | Reduce VRAM usage |
| `--eval_steps` | 2000 | Run lm-eval every N steps (0 to disable) |
| `--resume` | off | Auto-resume from latest checkpoint |
| `--dry_run` | off | Print parameter counts and exit |

### Data Preparation

For offline data preparation (downloads, tokenizes, and packs the dataset to disk):

```bash
bash scripts/prepare_data.sh
```

## Evaluation

### Standard LM Benchmarks (Zero-Shot)

Evaluates on LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
bash scripts/eval_llm.sh /path/to/output
```

### NIAH (Needle-In-A-Haystack)

Evaluates single-needle and multi-needle retrieval at 1K, 2K, 4K context lengths using custom task YAMLs in `tasks/`:

```bash
bash scripts/eval_niah.sh /path/to/output

# Only 180M models
bash scripts/eval_niah.sh /path/to/output 180M
```

### Direct Evaluation

```bash
uv run accelerate launch --multi_gpu --num_processes=8 \
  eval_llm.py \
    --model_path /path/to/model \
    --batch_size 64 \
    --output_dir eval_results
```

## MQAR Experiments

MQAR (Multi-Query Associative Recall) experiments use the [Zoology](https://github.com/HazyResearch/zoology) framework in `zoology/`.

```bash
# Run all architectures × all state sizes
python -m zoology.launch zoology/zoology/experiments/post_mqar_all.py
```

See `zoology/zoology/experiments/post_mqar_all.py` for the full configuration including LR sweeps, curriculum schedule, and state-size equalization across architectures.

MQAR figure/table scripts in `figures_scripts/` fetch data from W&B. Set the `WANDB_ENTITY` env var to your W&B team/user:

```bash
export WANDB_ENTITY="your-wandb-entity"
python figures_scripts/generate_mqar_figures.py --use-cached
```

## Figure Reproduction

Analysis scripts that produce paper figures require trained checkpoint paths:

```bash
# Taper profile ($\alpha$) analysis
python figures_scripts/analyze_alphas.py \
  --model_dir_180m /path/to/post_180m_4b \
  --model_dir_440m /path/to/post_440m_9b

# Spectral distribution analysis
python figures_scripts/analyze_spectra.py \
  --model_dir_180m /path/to/post_180m_4b \
  --model_dir_440m /path/to/post_440m_9b

# Layer × Head heatmaps
python figures_scripts/plot_heatmap.py \
  --model_dir_180m /path/to/post_180m_4b \
  --model_dir_440m /path/to/post_440m_9b
```

Each checkpoint directory should contain `mamba2/mamba2/model.safetensors` and `mamba2_post/mamba2_post/model.safetensors`.
