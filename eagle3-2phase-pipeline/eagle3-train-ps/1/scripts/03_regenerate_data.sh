#!/bin/bash
# Step 3 — Regenerate training data for ONE dataset using the target model via SGLang.
#
# Run this once per dataset you want to include in phase-2 training. Step 04 then
# concatenates every `qwen3_8b_regen_<dataset>.jsonl` into the combined file that
# step 05 reads. You can run 03 only for `sharegpt`, or for all three, or any mix.
#
# Synchronous: SpecForge's regenerate_train_data.py natively supports --num-samples N
# (stops after exactly N completed samples). No background process, no polling, no
# kill — produces deterministic, reproducible output regardless of warmup state.
#
# Env vars:
#   REGEN_TEMPERATURE  sampling temp (default 0.8; 0 → greedy, otherwise sampled)

# Usage:
#   bash scripts/03_regenerate_data.sh sharegpt     <num_samples>
#   bash scripts/03_regenerate_data.sh ultrachat    <num_samples>
#   bash scripts/03_regenerate_data.sh perfectblend <num_samples>
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
SPECFORGE_DIR=${SPECFORGE_DIR:-/workspace/SpecForge}
DATA_DIR="$ROOT_DIR/cache/dataset"

DATASET=${1:-sharegpt}
TARGET_SAMPLES=${2:-25000}
REGEN_TEMPERATURE=${REGEN_TEMPERATURE:-0.8}

LOG="$ROOT_DIR/logs/03_regen_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

MODEL=Qwen/Qwen3-8B
MEM_FRAC=${SGLANG_MEM_FRAC:-0.75}
CONCURRENCY=64
MAX_TOKENS=4096

INPUT_FILE="$DATA_DIR/${DATASET}_train.jsonl"
OUTPUT_FILE="$DATA_DIR/qwen3_8b_regen_${DATASET}.jsonl"

echo "=========================================="
echo "Step 3: Regen $DATASET (num_samples=$TARGET_SAMPLES, temp=$REGEN_TEMPERATURE)"
echo "  Input:   $INPUT_FILE"
echo "  Output:  $OUTPUT_FILE"
echo "=========================================="

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "       run scripts/01_prepare_data.sh first (with the corresponding ${DATASET^^}_N>0)."
    exit 1
fi

# regenerate_train_data.py rejects rows that start with an assistant message. Pre-process
# each row to a single {user: first_user_turn} conversation so every row is accepted.
PROMPTS_FILE="$DATA_DIR/${DATASET}_prompts_only.jsonl"
python3 - "$INPUT_FILE" "$PROMPTS_FILE" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
kept = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        if not line.strip(): continue
        row = json.loads(line)
        convs = row.get("conversations", []) or row.get("messages", [])
        user = next((m for m in convs if m.get("role") == "user"), None)
        if not user: continue
        fout.write(json.dumps({
            "id": row.get("id", kept),
            "conversations": [{"role": "user", "content": user["content"]}],
        }) + "\n")
        kept += 1
print(f"preprocessed {kept} prompts -> {dst}")
PY

# ------------------------------------------------------------------------------
# Launch SGLang server
# ------------------------------------------------------------------------------
start_server() {
    echo "=== starting SGLang server for $MODEL (mem-fraction-static=$MEM_FRAC) ==="
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
    CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
        --model $MODEL --trust-remote-code --tp 1 --dp 1 \
        --dtype bfloat16 --mem-fraction-static "$MEM_FRAC" \
        --port 30000 --host 0.0.0.0 >"$ROOT_DIR/logs/sglang_regen_server.log" 2>&1 &
    echo "Waiting for server on :30000 ..."
    for i in $(seq 1 180); do
        if curl -sf http://localhost:30000/health >/dev/null 2>&1; then
            echo "Server ready after ~${i}×5s"; return 0
        fi
        sleep 5
    done
    echo "ERROR: server did not come up"; return 1
}
cleanup() {
    echo "Stopping SGLang server ..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang::" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

start_server || exit 1

# Always start fresh: wipe prior output + the SpecForge-side _error.jsonl sibling.
# Without --resume, regenerate_train_data.py opens the output in "w" mode anyway,
# but we delete first so a midway crash from a prior run doesn't leave a partial
# file lying around between explicit invocations.
rm -f "$OUTPUT_FILE" "${OUTPUT_FILE%.jsonl}_error.jsonl"

# Synchronous regen — runs to exactly $TARGET_SAMPLES (--num-samples), then exits.
python "$SPECFORGE_DIR/scripts/regenerate_train_data.py" \
    --model $MODEL \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS \
    --server-address localhost:30000 \
    --temperature "$REGEN_TEMPERATURE" \
    --num-samples "$TARGET_SAMPLES" \
    --input-file-path "$PROMPTS_FILE" \
    --output-file-path "$OUTPUT_FILE"

FINAL_COUNT=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "=========================================="
echo "Done $DATASET: $FINAL_COUNT samples -> $OUTPUT_FILE"
echo "=========================================="
echo "Log: $LOG"
