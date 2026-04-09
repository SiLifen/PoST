#!/bin/bash
# ============================================================================
# PoST Project — Offline Data Preparation
# ============================================================================
#
# Downloads, tokenizes, and packs the dataset offline using trainer.py's
# built-in pipeline logic, then saves it to disk for all training nodes
# to load instantly.
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Override paths via environment variables:
#   DATA_DIR=/my/fast/disk bash scripts/prepare_data.sh
#
# ============================================================================
set -euo pipefail

# Configurable paths (override via environment)
DATA_DIR="${DATA_DIR:-./data}"
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"

# Dataset Target
DATASET="HuggingFaceFW/fineweb-edu"
DATASET_CONFIG="sample-100BT"
MAX_TOKENS=30e9
SEQ_LEN=2048

# Output Path
SAVE_PATH="${DATA_DIR}/fineweb-edu-100B-packed-30B"

echo "============================================================"
echo "  Preparing Dataset (Offline Packing)"
echo "============================================================"
echo "  Dataset: ${DATASET} (${DATASET_CONFIG})"
echo "  Tokens:  ${MAX_TOKENS}"
echo "  Output:  ${SAVE_PATH}"
echo "============================================================"
echo ""

# Run trainer in prepare_data_only mode (single GPU is fine since it's just data)
uv run python trainer.py \
  --data_path "${SAVE_PATH}" \
  --dataset_name "${DATASET}" \
  --dataset_config "${DATASET_CONFIG}" \
  --max_seq_length ${SEQ_LEN} \
  --max_tokens ${MAX_TOKENS} \
  --packing \
  --prepare_data_only
