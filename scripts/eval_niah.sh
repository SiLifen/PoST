#!/bin/bash
# ============================================================================
# Evaluation Script — NIAH (Needle-In-A-Haystack) Benchmark
# ============================================================================
#
# Evaluates trained models on NIAH-Single-1/2/3 at 2K, 4K, 8K context lengths.
# Uses custom task YAMLs from tasks/ directory.
#
# Usage:
#   bash scripts/eval_niah.sh [MODEL_BASE_DIR] [MODEL_SIZE_FILTER]
#
# Examples:
#   bash scripts/eval_niah.sh                                        # all models
#   bash scripts/eval_niah.sh ./data/output 180M  # only 180M
#
# ============================================================================
set -euo pipefail

# ---- Authentication ----
export HF_TOKEN=""
export WANDB_API_KEY=""

DATA_DIR="${DATA_DIR:-./data}"
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"

# ---- Configuration ----
MODEL_BASE="${1:-${DATA_DIR}/output}"
MODEL_SIZE_FILTER="${2:-}"
BATCH_SIZE=8
NUM_GPUS=8
OUTPUT_DIR="./eval_results/niah_models"

echo "============================================================"
echo "  PoST Evaluation — NIAH (Needle-In-A-Haystack)"
echo "============================================================"
echo "  Base Dir:        ${MODEL_BASE}"
echo "  Size Filter:     ${MODEL_SIZE_FILTER:-None}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Batch:           ${BATCH_SIZE}"
echo "  Context lengths: 1024, 2048, 4096"
echo "  Output:          ${OUTPUT_DIR}/"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

MODEL_PATHS=()
while IFS= read -r file; do
  dir_path="$(dirname "$file")"
  if [[ -z "${MODEL_SIZE_FILTER}" ]] || [[ "$dir_path" == *"${MODEL_SIZE_FILTER}"* ]]; then
    MODEL_PATHS+=("$dir_path")
  fi
done < <(find "$MODEL_BASE" -type d -name "checkpoint-*" -prune -o -name "config.json" -type f -print 2>/dev/null)

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "⚠️  No models found in ${MODEL_BASE}"
  exit 0
fi

TOTAL=${#MODEL_PATHS[@]}

for i in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$i]}"
  IDX=$((i + 1))
  ARCH=$(basename "$MODEL_PATH")

  echo ""
  echo "############################################################"
  echo "# [${IDX}/${TOTAL}] Evaluating NIAH: ${ARCH}"
  echo "############################################################"

  uv run accelerate launch \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    eval_llm.py \
    --model_path "${MODEL_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --output_dir "${OUTPUT_DIR}" \
    --tasks post_niah_single_1 post_niah_single_2 post_niah_single_3 post_niah_multikey_1 post_niah_multivalue post_niah_multiquery \
    --no_ruler \
    --ruler_lengths 1024 2048 4096

  echo "✅ [${IDX}/${TOTAL}] Done: ${ARCH}"
done

echo ""
echo "🏁 All NIAH evaluations complete! Results: ${OUTPUT_DIR}/"
