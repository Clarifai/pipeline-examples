#!/usr/bin/env bash
# quantize_llmcompressor.sh — generic HF model -> NVFP4 (or other format)
# quantization driver, using vLLM's llm-compressor as the backend.
#
# Usage:
#   ./quantize_llmcompressor.sh --src <hf_id_or_path> --out <output_dir> [options]
#
# Quick examples:
#   # Dense Llama
#   ./quantize_llmcompressor.sh --src meta-llama/Llama-3.3-70B-Instruct --out /data/llama
#
#   # MoE — auto-adds router + vision exclusions
#   ./quantize_llmcompressor.sh --src Qwen/Qwen3-30B-A3B-Instruct-2507 --out /data/qwen3
#
#   # Different scheme (W4A16 weight-only)
#   ./quantize_llmcompressor.sh --src meta-llama/Llama-3.3-70B-Instruct --out /data/llama-w4a16 --scheme W4A16
#
# See ./quantize_llmcompressor.sh --help for all flags.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ---------- defaults --------------------------------------------------------

SRC=""
OUT=""
SCHEME="NVFP4"
EXTRA_IGNORE=""
CALIB_DATASET="HuggingFaceH4/ultrachat_200k"
CALIB_SPLIT="train_sft"
CALIB_SIZE="512"
CALIB_SEQ="2048"
MOE_ALL_EXPERTS="true"
PIPELINE=""        # empty = let llm-compressor infer (defaults to sequential)
TRUST_REMOTE_CODE="true"
AUTO_DEFAULTS="true"
SAVE_UNCOMPRESSED="false"
DTYPE="auto"
GPUS=""
NUM_GPUS=""
DRY_RUN="false"
WORKER="${WORKER:-$SCRIPT_DIR/quantize_llmcompressor_worker.py}"
EXTRA_PY_ARGS=""

# ---------- usage -----------------------------------------------------------

usage() {
  cat <<'EOF'
quantize_llmcompressor.sh — generic HF -> NVFP4 driver via vLLM's llm-compressor

REQUIRED
  --src <hf_id_or_path>         Source HF repo id or local checkpoint dir
  --out <path>                  Output directory for the quantized checkpoint

QUANTIZATION
  --scheme <name>               Default: NVFP4. Common values:
                                  NVFP4         (W4A4 fp4 — Blackwell tensor cores)
                                  NVFP4A16      (W4A16 — fp4 weights, bf16 activations)
                                  FP8_DYNAMIC   (W8A8 fp8, dynamic per-token act)
                                  FP8           (W8A8 fp8, per-tensor static)
                                  W4A16         (int4 weight-only, AWQ-style)
                                  MXFP4         (W4A4 mxfp4, OCP standard, Hopper-friendly)
                                See compressed-tensors for the full preset list.
  --ignore <csv>                Extra patterns to add to the ignore list, comma-separated.
                                Use 're:<regex>' for regex. Merged with auto-defaults
                                (--auto-defaults true) which adds router/vision/embed
                                exclusions when MoE/VLM is detected.
  --no-auto-defaults            Disable auto-detected ignore patterns; use only --ignore.

CALIBRATION
  --calib-dataset <hf_id>       Default: HuggingFaceH4/ultrachat_200k
  --calib-split <name>          Default: train_sft
                                (For datasets without an SFT split, try 'train'.)
  --calib-size <int>            Default: 512
  --calib-seq <int>             Max sequence length. Default: 2048
  --moe-all-experts <bool>      true | false. Default: true. Forces every routed expert
                                through calibration (critical for MoE accuracy).
  --pipeline <name>             sequential | basic | independent | datafree.
                                Default: inferred (sequential for QuantizationModifier).
                                * sequential: layer-by-layer with error propagation.
                                  Highest accuracy, slow (~10 min/layer × num_layers
                                  with all-experts MoE calibration).
                                * basic: single forward pass over all calib samples.
                                  Much faster, small accuracy hit. Recommended for
                                  iteration / large MoE models.
                                * datafree: skip calibration entirely (only valid for
                                  weight-only schemes like NVFP4A16, W4A16).

EXECUTION
  --no-trust-remote-code        Disable trust_remote_code (default: enabled)
  --save-uncompressed           Save uncompressed safetensors (debug; bigger on disk)
  --dtype <auto|bfloat16|float16>
                                Model load dtype. Default: auto (uses model's torch_dtype).
  --gpus <csv>                  CUDA_VISIBLE_DEVICES override. Mutually exclusive with --num-gpus.
  --num-gpus <int>              Auto-pick this many least-loaded GPUs from nvidia-smi.
  --worker <path>               Override the Python worker script path
                                (default: <script_dir>/quantize_llmcompressor_worker.py)
  --extra '<flags>'             Extra raw flags forwarded to the Python worker.
  --dry-run                     Print every command and exit without executing.
  -h, --help                    Show this help.
EOF
}

# ---------- arg parsing -----------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)                  SRC="$2"; shift 2 ;;
    --out)                  OUT="$2"; shift 2 ;;
    --scheme)               SCHEME="$2"; shift 2 ;;
    --ignore)               EXTRA_IGNORE="$2"; shift 2 ;;
    --no-auto-defaults)     AUTO_DEFAULTS="false"; shift ;;
    --calib-dataset)        CALIB_DATASET="$2"; shift 2 ;;
    --calib-split)          CALIB_SPLIT="$2"; shift 2 ;;
    --calib-size)           CALIB_SIZE="$2"; shift 2 ;;
    --calib-seq)            CALIB_SEQ="$2"; shift 2 ;;
    --moe-all-experts)      MOE_ALL_EXPERTS="$2"; shift 2 ;;
    --pipeline)             PIPELINE="$2"; shift 2 ;;
    --no-trust-remote-code) TRUST_REMOTE_CODE="false"; shift ;;
    --save-uncompressed)    SAVE_UNCOMPRESSED="true"; shift ;;
    --dtype)                DTYPE="$2"; shift 2 ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --num-gpus)             NUM_GPUS="$2"; shift 2 ;;
    --worker)               WORKER="$2"; shift 2 ;;
    --extra)                EXTRA_PY_ARGS="$2"; shift 2 ;;
    --dry-run)              DRY_RUN="true"; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -z "$SRC" ]]   && { echo "ERROR: --src is required" >&2; exit 2; }
[[ -z "$OUT" ]]   && { echo "ERROR: --out is required" >&2; exit 2; }
[[ -f "$WORKER" ]] || { echo "ERROR: worker script not found at $WORKER" >&2; exit 2; }

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "true" ]]; then
    eval "$@"
  fi
}

# ---------- pick GPUs -------------------------------------------------------

pick_idle_gpus() {
  local n="$1"
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: --num-gpus needs nvidia-smi but it's not on PATH." >&2
    exit 2
  }
  local query
  query=$(nvidia-smi --query-gpu=index,memory.used,memory.free,name \
                    --format=csv,noheader,nounits 2>/dev/null) || {
    echo "ERROR: nvidia-smi query failed." >&2
    exit 2
  }
  if [[ -z "$query" ]]; then
    echo "ERROR: nvidia-smi returned no GPUs." >&2
    exit 2
  fi
  local total
  total=$(echo "$query" | wc -l)
  if (( n > total )); then
    echo "ERROR: --num-gpus $n requested but only $total GPU(s) visible." >&2
    exit 2
  fi
  local picked
  picked=$(echo "$query" | sort -t',' -k2,2n | head -n "$n")
  echo "==> Auto-selected $n GPU(s) by lowest memory.used:" >&2
  echo "$picked" | awk -F',' '{
    gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2);
    gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4);
    printf "    [%s] %s  used=%s MiB  free=%s MiB\n", $1, $4, $2, $3
  }' >&2
  echo "$picked" | awk -F',' '{gsub(/ /,"",$1); print $1}' | paste -sd,
}

if [[ -n "$NUM_GPUS" && -n "$GPUS" ]]; then
  echo "ERROR: --gpus and --num-gpus are mutually exclusive." >&2
  exit 2
fi
if [[ -n "$NUM_GPUS" ]]; then
  GPUS=$(pick_idle_gpus "$NUM_GPUS")
fi

# ---------- llmcompressor importable? --------------------------------------

if ! python -c "import llmcompressor" >/dev/null 2>&1; then
  echo "WARNING: 'import llmcompressor' failed in the active Python environment." >&2
  echo "         Install with: pip install -e $SCRIPT_DIR/llm-compressor" >&2
  echo "         or:           pip install llmcompressor" >&2
fi

# ---------- materialize HF repo to a local path ----------------------------
# llmcompressor's oneshot path internally does from_pretrained(model_id) which
# itself supports HF repo ids — but we still snapshot up-front so:
#   * inspect can read config.json without auth/network indirection,
#   * dtype/disk usage is predictable,
#   * the same path works whether --src is a repo or a local dir.

PYT_CKPT="$SRC"
if [[ ! -d "$SRC" ]]; then
  echo "==> Materializing HF snapshot for $SRC (cache: \${HF_HOME:-~/.cache/huggingface})"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    [dry-run] would call snapshot_download(repo_id=$SRC)"
    PYT_CKPT="<local-snapshot-of-$SRC>"
  else
    PYT_CKPT=$(SRC="$SRC" python - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
try:
    print(snapshot_download(repo_id=os.environ["SRC"]))
except Exception as e:
    print(f"SNAPSHOT_DOWNLOAD_FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    if [[ -z "$PYT_CKPT" || ! -d "$PYT_CKPT" ]]; then
      echo "ERROR: snapshot_download did not return a usable local path." >&2
      exit 1
    fi
    echo "    local path: $PYT_CKPT"
  fi
fi

# ---------- run worker -----------------------------------------------------

mkdir -p "$OUT"

PY_ARGS=()
PY_ARGS+=( --src "$PYT_CKPT" )
PY_ARGS+=( --out "$OUT" )
PY_ARGS+=( --scheme "$SCHEME" )
PY_ARGS+=( --calib-dataset "$CALIB_DATASET" )
PY_ARGS+=( --calib-split "$CALIB_SPLIT" )
PY_ARGS+=( --calib-size "$CALIB_SIZE" )
PY_ARGS+=( --calib-seq "$CALIB_SEQ" )
PY_ARGS+=( --moe-all-experts "$MOE_ALL_EXPERTS" )
PY_ARGS+=( --auto-defaults "$AUTO_DEFAULTS" )
PY_ARGS+=( --dtype "$DTYPE" )
[[ -n "$PIPELINE" ]] && PY_ARGS+=( --pipeline "$PIPELINE" )

[[ -n "$EXTRA_IGNORE" ]] && PY_ARGS+=( --ignore "$EXTRA_IGNORE" )
[[ "$TRUST_REMOTE_CODE" == "false" ]] && PY_ARGS+=( --no-trust-remote-code )
[[ "$SAVE_UNCOMPRESSED" == "true" ]]  && PY_ARGS+=( --save-uncompressed )

echo "==> Running llmcompressor worker"
echo "    src        : $PYT_CKPT"
echo "    out        : $OUT"
echo "    scheme     : $SCHEME"
echo "    calib      : $CALIB_DATASET split=$CALIB_SPLIT n=$CALIB_SIZE seq=$CALIB_SEQ"
echo "    moe-all    : $MOE_ALL_EXPERTS"
[[ -n "$PIPELINE" ]] && echo "    pipeline   : $PIPELINE"
[[ -n "$EXTRA_IGNORE" ]] && echo "    extra ignore : $EXTRA_IGNORE"

CMD=( python "$WORKER" "${PY_ARGS[@]}" )
[[ -n "$EXTRA_PY_ARGS" ]] && CMD+=( $EXTRA_PY_ARGS )

if [[ -n "$GPUS" ]]; then
  CMD=( env CUDA_VISIBLE_DEVICES="$GPUS" "${CMD[@]}" )
fi

printf '+ '
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$DRY_RUN" != "true" ]]; then
  "${CMD[@]}"
fi

# ---------- verify ---------------------------------------------------------

if [[ "$DRY_RUN" == "true" ]]; then
  echo "==> Dry run finished. No commands executed."
  exit 0
fi

echo "==> Verifying output at $OUT"
if [[ ! -f "$OUT/config.json" ]]; then
  echo "ERROR: $OUT/config.json missing — quantization did not produce a checkpoint." >&2
  exit 1
fi

du -sh "$OUT"
OUT="$OUT" python - <<'PYEOF'
import json, os
out = os.environ["OUT"]
with open(os.path.join(out, "config.json")) as f:
    cfg = json.load(f)
qc = cfg.get("quantization_config", {})
if not qc:
    print("WARNING: quantization_config missing from config.json")
else:
    print("quantization_config (first 1500 chars):")
    print(json.dumps(qc, indent=2)[:1500])
PYEOF

echo "==> Done. Checkpoint: $OUT"
