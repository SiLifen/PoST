#!/bin/bash
# ============================================================================
# PoST Training Script — 180M Models × 4B Tokens per model
# ============================================================================
#
# Trains 4 models sequentially on 4×GPU:
#   1. Mamba-2      (baseline)
#   2. Mamba-2 PoST (position-adaptive state taper)
#   3. RWKV-7       (baseline)
#   4. RWKV-7 PoST  (position-adaptive state taper)
#
# Cost estimate (1×H200 @ $2/hr, ~35% MFU):
#   Mamba-2:  ~5 hr/model
#   RWKV-7:   ~5 hr/model
#   Total:    ~40 hr → ~$80
#
# Usage:
#   bash scripts/run_180m_2.5b.sh              # Run all 4
#   bash scripts/run_180m_2.5b.sh mamba2_post  # Run only one
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
MODEL_SIZE="180M"
SEQ_LEN=2048
LR=6e-4
DATASET="HuggingFaceFW/fineweb-edu"
DATASET_CONFIG="sample-10BT"

# Batch size: 8 × 8 GPUs × 4 grad_accum × 2048 = 524,288 tokens/step ≈ 0.5M
BATCH_SIZE=8
GRAD_ACCUM=4
NUM_GPUS=8

# 10B tokens / 524,288 tokens per step ≈ 19,073 steps
# MAX_STEPS=19073

SAVE_STEPS=3000
EVAL_STEPS=3000

OUTPUT_BASE="${DATA_DIR}/output/post_180m_4b"
export WANDB_PROJECT="post-180m-4b"

# ---- Architecture list ----
if [[ $# -gt 0 ]]; then
  ARCHS=("$@")
else
  ARCHS=("mamba2" "mamba2_post" "rwkv7" "rwkv7_post" "gdn" "gdn_post")
fi

# ---- Print plan ----
TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * SEQ_LEN))
# TOTAL_TOKENS=$((MAX_STEPS * TOKENS_PER_STEP))
echo "============================================================"
# echo "  PoST Training — ${MODEL_SIZE} × ${TOTAL_TOKENS} tokens"
echo "============================================================"
echo "  Models:        ${ARCHS[*]}"
echo "  Tokens/step:   ${TOKENS_PER_STEP} (≈0.5M)"
# echo "  Max steps:     ${MAX_STEPS}"
echo "  Learning rate: ${LR}"
echo "  Output:        ${OUTPUT_BASE}/"
echo "  WandB project: ${WANDB_PROJECT}"
echo "============================================================"
echo ""

# ---- Train each architecture ----
for i in "${!ARCHS[@]}"; do
  ARCH="${ARCHS[$i]}"
  IDX=$((i + 1))
  TOTAL=${#ARCHS[@]}

  echo ""
  echo "############################################################"
  echo "# [${IDX}/${TOTAL}] Training: ${ARCH}"
  echo "############################################################"
  echo ""

  # Check if custom local dataset exists
  DATA_ARGS="--dataset_name ${DATASET} --dataset_config ${DATASET_CONFIG}"
  LOCAL_DATA_PATH="${DATA_DIR}/output/data/fineweb-edu-100B-packed-30B"
  if [ -d "$LOCAL_DATA_PATH" ]; then
    echo "  [INFO] Found local packed dataset: ${LOCAL_DATA_PATH}"
    DATA_ARGS="--data_path ${LOCAL_DATA_PATH} --dataset_name ${DATASET} --dataset_config ${DATASET_CONFIG}"
  fi

  uv run accelerate launch \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    trainer.py \
      --arch "${ARCH}" \
      --model_size "${MODEL_SIZE}" \
      ${DATA_ARGS} \
      --max_seq_length ${SEQ_LEN} \
      --batch_size ${BATCH_SIZE} \
      --grad_accum ${GRAD_ACCUM} \
      --lr ${LR} \
      --output_dir "${OUTPUT_BASE}/${ARCH}" \
      --packing \
      --max_tokens 4e9 \
      --save_steps ${SAVE_STEPS} \
      --eval_steps ${EVAL_STEPS} \
      --resume
  echo ""
  echo "✅ [${IDX}/${TOTAL}] Done: ${ARCH}"
  echo ""
done

echo ""
echo "🏁 All training complete!"
echo "   Results: ${OUTPUT_BASE}/"
echo "   WandB:   https://wandb.ai (project: ${WANDB_PROJECT})"
