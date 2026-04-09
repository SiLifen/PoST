#!/bin/bash
# ============================================================================
# Evaluation Script — LLM Models (4×GPU data-parallel)
# ============================================================================
#
# Evaluates trained models in the specified directory.
#
# Usage:
#   bash scripts/eval_llm.sh [MODEL_BASE_DIR]
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
# MODEL_BASE is where the trained weights are saved
MODEL_BASE="${1:-${DATA_DIR}/output}"
NUM_GPUS=8
BATCH_SIZE_STD=64         # Standard benchmarks (loglikelihood — lightweight)
OUTPUT_DIR="./eval_results/all_models"

# ---- Print plan ----
echo "============================================================"
echo "  PoST Evaluation — All Models"
echo "============================================================"
echo "  Base Dir:        ${MODEL_BASE}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Batch (std):     ${BATCH_SIZE_STD}"
echo "  Output:          ${OUTPUT_DIR}/"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Find all directories containing config.json
MODEL_PATHS=()
while IFS= read -r file; do
  MODEL_PATHS+=("$(dirname "$file")")
done < <(find "$MODEL_BASE" -type d -name "checkpoint-*" -prune -o -name "config.json" -type f -print 2>/dev/null)

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "⚠️  No models found in ${MODEL_BASE}"
  exit 0
fi

TOTAL=${#MODEL_PATHS[@]}

# ---- Evaluate each model ----
for i in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$i]}"
  IDX=$((i + 1))
  
  # Output name is just the arch name
  ARCH=$(basename "$MODEL_PATH")

  echo ""
  echo "############################################################"
  echo "# [${IDX}/${TOTAL}] Evaluating: ${ARCH}"
  echo "#   Model path: ${MODEL_PATH}"
  echo "############################################################"

  # Phase 1: Standard benchmarks (multi-GPU, large batch)
  echo ""
  echo "📊 Phase 1: Standard benchmarks (${NUM_GPUS} GPUs, batch=${BATCH_SIZE_STD})"
  uv run accelerate launch \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    eval_llm.py \
      --model_path "${MODEL_PATH}" \
      --batch_size ${BATCH_SIZE_STD} \
      --output_dir "${OUTPUT_DIR}" \
      --no_ruler
  echo ""
  echo "✅ [${IDX}/${TOTAL}] Done: ${ARCH}"
  echo ""
done

echo ""
echo "🏁 All evaluations complete!"
echo "   Results: ${OUTPUT_DIR}/"
