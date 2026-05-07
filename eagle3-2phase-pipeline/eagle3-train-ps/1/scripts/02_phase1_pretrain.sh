#!/bin/bash
# Step 2 — Phase-1 pretraining of the Eagle3 draft head for Qwen3-8B.
#
# Env vars (all have defaults; override any of them to tune):
#   PHASE1_EPOCHS            num epochs (default 10)
#   MAX_LENGTH               max seq length (default 4096)
#   BATCH_SIZE               per-device batch (default 4)
#   SGLANG_MEM_FRAC          sglang mem fraction (default 0.6)
#   DATALOADER_NUM_WORKERS   torch dataloader workers (default 8)
#
# Reads:  cache/dataset/qwen3_8b_train.jsonl, cache/dataset/qwen3_8b_eval.jsonl
# Writes: outputs/phase1/epoch_*
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
SPECFORGE_DIR=${SPECFORGE_DIR:-/workspace/SpecForge}

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export CUDA_VISIBLE_DEVICES=0

PHASE1_EPOCHS=${PHASE1_EPOCHS:-10}
MAX_LENGTH=${MAX_LENGTH:-4096}
BATCH_SIZE=${BATCH_SIZE:-4}
SGLANG_MEM_FRAC=${SGLANG_MEM_FRAC:-0.6}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-2000}

LOG="$ROOT_DIR/logs/02_phase1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 2: Phase-1 pretraining (epochs=$PHASE1_EPOCHS max_length=$MAX_LENGTH batch_size=$BATCH_SIZE mem_frac=$SGLANG_MEM_FRAC workers=$DATALOADER_NUM_WORKERS) ==="

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
sleep 2

TRAIN_DATA="$ROOT_DIR/cache/dataset/qwen3_8b_train.jsonl"
EVAL_DATA="$ROOT_DIR/cache/dataset/qwen3_8b_eval.jsonl"
OUTPUT_DIR="$ROOT_DIR/outputs/phase1"

if [ ! -s "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA missing. Run scripts/01_prepare_data.sh first."
    exit 1
fi
[ -s "$EVAL_DATA" ] || EVAL_DATA="$TRAIN_DATA"
mkdir -p "$OUTPUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node 1 \
    $SPECFORGE_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --embedding-key model.embed_tokens.weight \
    --train-data-path "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --build-dataset-num-proc 64 \
    --dataloader-num-workers $DATALOADER_NUM_WORKERS \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs $PHASE1_EPOCHS \
    --batch-size $BATCH_SIZE \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length $MAX_LENGTH \
    --ttt-length 7 \
    --max-grad-norm 0.5 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --sglang-mem-fraction-static $SGLANG_MEM_FRAC \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --dist-timeout 60 \
    --report-to tensorboard

echo ""
echo "=== Step 2 complete. Checkpoints in $OUTPUT_DIR: ==="
ls -d "$OUTPUT_DIR"/epoch_* 2>/dev/null | tail -5 || echo "(no epoch_* dirs yet)"
echo "Log: $LOG"
