#!/bin/bash
# Step 5 — Phase-2 fine-tuning on regenerated data, starting from Phase-1 checkpoint.
#
# Env vars (all have defaults):
#   PHASE2_EPOCHS          num epochs (default 6)
#   LEARNING_RATE          phase-2 LR (default 2e-5)
#   MAX_LENGTH             max seq length (default 4096)
#   BATCH_SIZE             per-device batch (default 4)
#   SGLANG_MEM_FRAC        sglang mem fraction (default 0.6)
#   DATALOADER_NUM_WORKERS torch dataloader workers (default 8)
#
# Reads:  cache/dataset/qwen3_8b_regen_combined.jsonl (from step 04)
#         outputs/phase1/epoch_* (latest)
# Writes: outputs/phase2/epoch_*
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
SPECFORGE_DIR=${SPECFORGE_DIR:-/workspace/SpecForge}
DATA_DIR="$ROOT_DIR/cache/dataset"

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export CUDA_VISIBLE_DEVICES=0

PHASE2_EPOCHS=${PHASE2_EPOCHS:-6}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_LENGTH=${MAX_LENGTH:-4096}
BATCH_SIZE=${BATCH_SIZE:-4}
SGLANG_MEM_FRAC=${SGLANG_MEM_FRAC:-0.6}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-2000}

LOG="$ROOT_DIR/logs/05_phase2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 5: Phase-2 (epochs=$PHASE2_EPOCHS lr=$LEARNING_RATE max_len=$MAX_LENGTH batch=$BATCH_SIZE mem_frac=$SGLANG_MEM_FRAC) ==="

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
sleep 2

PHASE1_CKPT=${PHASE1_CKPT:-$(ls -d $ROOT_DIR/outputs/phase1/epoch_* 2>/dev/null | sort -V | tail -n 1)}
if [ -z "$PHASE1_CKPT" ]; then
    echo "ERROR: no phase1 checkpoint in outputs/phase1/. Run 02_phase1_pretrain.sh first."
    exit 1
fi
echo "Resuming from phase1: $PHASE1_CKPT"

TRAIN_DATA="$DATA_DIR/qwen3_8b_regen_combined.jsonl"
if [ ! -s "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA missing. Run scripts/03_regenerate_data.sh then scripts/04_combine_regen.sh first."
    exit 1
fi

EVAL_DATA="$DATA_DIR/qwen3_8b_eval.jsonl"
[ -f "$EVAL_DATA" ] || EVAL_DATA="$TRAIN_DATA"

OUTPUT_DIR="$ROOT_DIR/outputs/phase2"
mkdir -p "$OUTPUT_DIR"
echo "TRAIN=$TRAIN_DATA  EVAL=$EVAL_DATA  OUT=$OUTPUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node 1 \
    $SPECFORGE_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --embedding-key model.embed_tokens.weight \
    --ckpt-dir "$PHASE1_CKPT" \
    --train-data-path "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --build-dataset-num-proc 64 \
    --dataloader-num-workers $DATALOADER_NUM_WORKERS \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs $PHASE2_EPOCHS \
    --batch-size $BATCH_SIZE \
    --tp-size 1 \
    --learning-rate $LEARNING_RATE \
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
echo "=== Step 5 complete. Checkpoints in $OUTPUT_DIR: ==="
ls -d "$OUTPUT_DIR"/epoch_* 2>/dev/null | tail -5 || echo "(no epoch_* dirs yet)"
echo "Log: $LOG"
