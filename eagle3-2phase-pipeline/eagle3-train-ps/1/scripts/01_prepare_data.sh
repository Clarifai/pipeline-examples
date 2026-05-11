#!/bin/bash
# Step 1 — Phase-1 data preparation for Qwen3-8B Eagle3 (full-mode blend).
#
# Downloads SG / UC / PB via SpecForge's prepare_data.py, shuffles each
# (seed=42), takes the first SG_N / UC_N / PB_N rows, mixes, shuffles again,
# writes qwen3_8b_train.jsonl. Eval = last 200 SG rows (holdout).
#
# Output: cache/dataset/qwen3_8b_train.jsonl + cache/dataset/qwen3_8b_eval.jsonl
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
SPECFORGE_DIR=${SPECFORGE_DIR:-/workspace/SpecForge}
OUT_DIR="$ROOT_DIR/cache/dataset"
mkdir -p "$OUT_DIR"

SG_N=${SG_N:-12000}
UC_N=${UC_N:-10000}
PB_N=${PB_N:-5000}

LOG="$ROOT_DIR/logs/01_prepare_data_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== Step 1: Prepare Phase-1 data (SG_N=$SG_N UC_N=$UC_N PB_N=$PB_N) ==="
cd "$SPECFORGE_DIR"

[ "$SG_N" -gt 0 ] && python scripts/prepare_data.py --dataset sharegpt     --output-path "$OUT_DIR"
[ "$UC_N" -gt 0 ] && python scripts/prepare_data.py --dataset ultrachat    --output-path "$OUT_DIR"
[ "$PB_N" -gt 0 ] && { python scripts/prepare_data.py --dataset perfectblend --output-path "$OUT_DIR" \
    || echo "WARN: perfectblend failed — continuing without it"; }

python <<PY
import json, os, random
random.seed(42)
OUT = "$OUT_DIR"
sg_n, uc_n, pb_n = int("$SG_N"), int("$UC_N"), int("$PB_N")

def load(p):
    if not os.path.isfile(p): return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

sharegpt  = load(os.path.join(OUT, "sharegpt_train.jsonl"))     if sg_n > 0 else []
ultrachat = load(os.path.join(OUT, "ultrachat_train.jsonl"))    if uc_n > 0 else []
pb        = load(os.path.join(OUT, "perfectblend_train.jsonl")) if pb_n > 0 else []
print(f"Loaded: SG={len(sharegpt)} UC={len(ultrachat)} PB={len(pb)}")

random.shuffle(sharegpt); random.shuffle(ultrachat); random.shuffle(pb)
mixed = sharegpt[:sg_n] + ultrachat[:uc_n] + pb[:pb_n]
random.shuffle(mixed)

out = os.path.join(OUT, "qwen3_8b_train.jsonl")
with open(out, "w") as f:
    for x in mixed:
        f.write(json.dumps(x) + "\n")
print(f"Wrote {len(mixed)} -> {out}")

# Filepath-keyed caches (SpecForge processed_dataset, vocab_mapping, HF generator)
# would silently reuse OLD data on the next training run. Invalidate them.
import shutil
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for p in [
    os.path.join("$ROOT_DIR", "cache", "processed_dataset"),
    os.path.join("$ROOT_DIR", "cache", "vocab_mapping"),
    os.path.join(hf_home, "datasets", "generator"),
]:
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print(f"invalidated cache: {p}")

# Eval set = last 200 SG rows (holdout). Fallback to small slice if SG is tiny.
ev = sharegpt[-200:] if len(sharegpt) > 400 else sharegpt[:max(20, len(sharegpt)//5)]
evp = os.path.join(OUT, "qwen3_8b_eval.jsonl")
with open(evp, "w") as f:
    for x in ev:
        f.write(json.dumps(x) + "\n")
print(f"Wrote {len(ev)} -> {evp}")
PY

echo ""
echo "=== Step 1 complete. Outputs in $OUT_DIR: ==="
ls -lh "$OUT_DIR" | head -15
echo "Log: $LOG"
