#!/bin/bash
# Step 4 — Combine per-dataset regen outputs into a single fine-tune training file.
#
# Plain `cat` concatenation; training loader shuffles internally.
#
# Usage:
#   bash scripts/04_combine_regen.sh                                # auto-discover all regen files in cache/dataset
#   bash scripts/04_combine_regen.sh file1.jsonl file2.jsonl ...    # concat exactly these files
#
# Always writes to the same fixed path so 05_phase2_finetune.sh has a stable input:
#   $DATA_DIR/qwen3_8b_regen_combined.jsonl
# Truncates any prior combined file.
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
DATA_DIR="$ROOT_DIR/cache/dataset"
OUT="$DATA_DIR/qwen3_8b_regen_combined.jsonl"

LOG="$ROOT_DIR/logs/04_combine_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== Step 4: Combine regen outputs -> $OUT ==="

INPUTS=()
if [ "$#" -gt 0 ]; then
    # Explicit file list
    for f in "$@"; do
        if [ -s "$f" ]; then
            INPUTS+=("$f")
        else
            echo "WARN: skipping missing/empty $f"
        fi
    done
else
    # Auto-discovery — prefer full file, fall back to _quick, across common bases
    for base in qwen3_8b_regen_sharegpt qwen3_8b_regen_ultrachat qwen3_8b_regen_perfectblend; do
        if   [ -s "$DATA_DIR/${base}.jsonl"       ]; then INPUTS+=("$DATA_DIR/${base}.jsonl")
        elif [ -s "$DATA_DIR/${base}_quick.jsonl" ]; then INPUTS+=("$DATA_DIR/${base}_quick.jsonl")
        fi
    done
fi

if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "ERROR: no regen files found to combine. Run scripts/03_regenerate_data.sh first."
    exit 1
fi

echo "Combining ${#INPUTS[@]} file(s):"
for f in "${INPUTS[@]}"; do
    echo "  - $f  ($(wc -l < "$f") rows)"
done

: > "$OUT"
for f in "${INPUTS[@]}"; do
    cat "$f" >> "$OUT"
done

# Invalidate stale caches keyed only on filepath (SpecForge's processed_dataset
# + vocab_mapping, and HF datasets' generator cache). Without this, phase-2 would
# silently reuse a previous combined dataset with different content.
for p in "$ROOT_DIR/cache/processed_dataset" "$ROOT_DIR/cache/vocab_mapping" "$HOME/.cache/huggingface/datasets/generator"; do
    if [ -d "$p" ]; then
        rm -rf "$p"
        echo "invalidated cache: $p"
    fi
done

echo ""
echo "=== Step 4 complete. Combined: $(wc -l < "$OUT") rows -> $OUT ==="
echo "Log: $LOG"
