# LLM Compressor Quantize-and-Evaluate Pipeline (single-step)

One Clarifai pipeline step that does the entire **quantize → evaluate**
loop in a single container and reports the accuracy delta between the
original BF16 model and the quantized checkpoint.

## What the step does, in order

1. **Phase 1 (optional, on by default)** — serve the **SOURCE BF16** model
   with vLLM and run [`evalscope`](https://github.com/modelscope/evalscope)
   on it to record the pre-quantization baseline. Skip with
   `--set eval_source=false` when you already know the baseline.
2. **Phase 2** — quantize the source model with vLLM's
   [`llm-compressor`](https://github.com/vllm-project/llm-compressor)
   (scheme + pipeline chosen via inputs).
3. **Phase 3** — HF transformers smoke-test the produced checkpoint
   (load + chat-template generate).
4. **Phase 4** — optionally upload the compressed-tensors checkpoint as a
   versioned Clarifai artifact (`quantized-checkpoint`).
5. **Phase 5** — serve the **local quantized dir** with vLLM and run the
   same evalscope benchmark again.
6. **Phase 6** — write a top-level `summary_comparison.json` capturing
   both runs plus `delta_pp` and `retention_pct`.
7. **Phase 7** — optionally upload the full eval run dir (both phases'
   `configs/ logs/ predictions/ reviews/ reports/` plus the summary) as a
   second versioned artifact (`eval-report`).

Both phases live in **one** `pipeline_step.py` and share **one** container
image (`quantize-evaluate-step/Dockerfile`). For a two-step variant —
separate `quantize-step` and `evaluate-step` that pass the checkpoint via
a Clarifai artifact, useful when you want to evaluate an already-quantized
checkpoint without re-quantizing — see
`../llmcompressor-pipeline-quick-start/`.

---

## Defaults

The defaults run a fast int4 weight-only sanity (Qwen2.5-0.5B, W4A16,
datafree, 1 GPU) and evaluate both source and quantized on GPQA-Diamond.
For the W4A4 NVFP4 reference run on a small MoE model:

| Param | Fast sanity (default) | MoE NVFP4 reference |
|---|---|---|
| `src` | `Qwen/Qwen2.5-0.5B-Instruct` | `ibm-granite/granite-4.0-h-tiny` |
| `scheme` | `W4A16` | `NVFP4` |
| `pipeline` | `datafree` | `""` (auto-infer → sequential) |
| `num_gpus` | `1` | `1` |
| `eval_source` | `true` | `true` |
| `dataset` | `gpqa_diamond` | `gpqa_diamond` |
| `eval_batch_size` | `32` | `32` |
| `timeout` | `1500` | `1500` |
| `max_tokens` | `64000` (auto-clamped to fit context) | same |
| `user_id` / `app_id` | your Clarifai user/app | same |
| `artifact_id` | `quantized-checkpoint` | same |
| `report_artifact_id` | `eval-report` | same |

Full per-param descriptions live in
`quantize-evaluate-step/config.yaml` and in `pipeline_step.py --help`.

---

## Upload and run the pipeline

From inside this folder, upload once:

```bash
clarifai pipeline upload
```

Then run with one of the two compute options:

```bash
# (a) Simplest — auto-create or reuse compute from an instance type
clarifai pipeline run --instance=g6e.xlarge

# (b) Use your existing nodepool + compute cluster (both flags required)
clarifai pipeline run \
    --nodepool_id=<your_existing_nodepool_id> \
    --compute_cluster_id=<your_existing_compute_cluster_id>
```

Override params at runtime with `--set key=value`:

```bash
# Reproduce the MoE NVFP4 reference run with GPQA-Diamond eval afterwards
clarifai pipeline run --instance=g6e.xlarge \
    --set src=ibm-granite/granite-4.0-h-tiny \
    --set scheme=NVFP4 \
    --set pipeline= \
    --set user_id=<YOUR_USER_ID> \
    --set app_id=<YOUR_APP_ID>

# Skip the source eval (~2x faster, but no delta)
clarifai pipeline run --instance=g6e.xlarge \
    --set eval_source=false

# Pick a different evalscope benchmark
clarifai pipeline run --instance=g6e.xlarge \
    --set dataset=mmlu \
    --set user_id=<YOUR_USER_ID> \
    --set app_id=<YOUR_APP_ID>
```

When `user_id` and `app_id` are set, the step uploads two artifacts:
- `quantized-checkpoint` — the compressed-tensors model dir (zip)
- `eval-report` — the full eval bundle (zip), see *Eval-report layout* below

When they're empty the step still runs end-to-end (Phase 1 → … → Phase 6)
but skips both uploads. Useful for local debugging.

### Monitor

Visit `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` and check the
**Pipelines** tab to follow run progress. Artifacts land under the
**Artifacts** tab when the run finishes.

---

## Eval-report layout

When `eval_source=true` (default), the `eval-report` artifact's zip
contents look like:

```
summary_comparison.json     ← top-level: scores, delta_pp, retention_pct
source/
    <timestamp>/
        configs/  logs/  predictions/  reviews/  reports/
        vllm-server.log
quantized/
    <timestamp>/
        configs/  logs/  predictions/  reviews/  reports/
        vllm-server.log
```

Headline fields in `summary_comparison.json`:

```json
{
  "source_model":   "Qwen/Qwen2.5-0.5B-Instruct",
  "quantized_model": "/workspace/outputs/quantized_model",
  "scheme":         "W4A16",
  "dataset":        "gpqa_diamond",
  "source":    { "score": 0.2172, "num_samples": 198, ... },
  "quantized": { "score": 0.1364, "num_samples": 198, ... },
  "delta_pp": -8.08,
  "retention_pct": 62.80
}
```

When `eval_source=false`, only `quantized/` is present and
`summary_comparison.json` omits the `source` / `delta_pp` /
`retention_pct` keys.

---

## Notes on the eval defaults

Tuned for thinking-mode reasoning models on PhD-level benchmarks like
GPQA-Diamond. Each default has a real failure case behind it:

| Default | Why |
|---|---|
| `timeout=1500` (sec/request) | evalscope's default of 300 s truncates long reasoning traces from Qwen3-class models and forces 5× retries. 1500 s covers all observed cases. |
| `max_tokens=64000` | Bounds runaway thinking traces — without a cap, an outlier sample can run past `max_model_len` and return empty (eval scores it wrong). The step **auto-clamps** this to `model.max_position_embeddings − 4096` if the requested value exceeds the model's context window (so Qwen2.5-0.5B with 32K ctx gets clamped to 28672, while Qwen3.6 with 262K ctx keeps the requested 64000). |
| `eval_batch_size=32` | Concurrent requests to the local vLLM server. Tune up for higher GPU utilization on small models, down if you OOM. |
| `eval_source=true` | Without the BF16 baseline, the quantized score is uninterpretable. Doubles eval wall-time; set `false` only when you already know the baseline. |

Override per-run as usual:

```bash
clarifai pipeline run --instance=g6e.xlarge \
    --set timeout=3000 --set max_tokens=32000 --set eval_batch_size=16
```

---

## Container & image notes

The step's container is built `FROM vllm/vllm-openai:latest` so that
torch + CUDA libs + vLLM come pre-installed. Layered on top:

- `llm-compressor` at a pinned commit (`LLMC_REF=7a52e9e`)
- `evalscope==1.6.1`
- `huggingface-hub<1.0,>=0.34` (matches the transformers range
  llm-compressor pins; without this pin, `evalscope` pulls hub 1.x and
  breaks `transformers` import at runtime)
- A `python` → `python3` symlink (Clarifai's pipeline runner invokes
  `python` directly; the base image only ships `python3`)
- `ENTRYPOINT []` (the base image's default entrypoint is
  `["vllm","serve"]`; clear it so the container runs arbitrary commands)

The Dockerfile is small and all the reasoning lives next to the code that
needs it — read it if you want to adapt to a different base.

---

## Local development & iteration

You can build + run the entire pipeline locally (much faster iteration
than `clarifai pipeline upload`):

```bash
# Build the step image
docker build -t qe-step:local quantize-evaluate-step/

# Run end-to-end on a single GPU, defaults match the cloud run
docker run --rm --gpus '"device=0"' --ipc=host --shm-size=32g \
    -v "$HOME/.cache/huggingface:/home/nonroot/cache/huggingface" \
    qe-step:local \
    python /home/nonroot/main/1/pipeline_step.py

# Override any param the same way `--set` would
docker run --rm --gpus '"device=0"' --ipc=host --shm-size=32g \
    -v "$HOME/.cache/huggingface:/home/nonroot/cache/huggingface" \
    qe-step:local \
    python /home/nonroot/main/1/pipeline_step.py \
        --src ibm-granite/granite-4.0-h-tiny \
        --scheme NVFP4 \
        --pipeline "" \
        --eval_source true
```

Uploads are skipped automatically when `--user_id` / `--app_id` aren't
passed (and there's no `CLARIFAI_USER_ID` / `CLARIFAI_APP_ID` env vars).

---

## Download artifacts and verify them locally

Two helpers ship with this folder:

| Script | What it does |
|---|---|
| `test_download_and_sanity.py` | Downloads the **quantized-checkpoint** artifact, unzips, then HF-loads + chat-template-generates to confirm the checkpoint is downstream-deployable. |
| `test_download_eval_report.py` | Downloads the **eval-report** artifact, unzips, prints `summary_comparison.json`, then for each phase (source / quantized) prints score / macro / n plus pipeline-health stats from `reviews/` (recomputed accuracy, empty extracts, letter distribution, prediction-length percentiles). |

Run them inside the same container image used by the pipeline so
torch / transformers / compressed-tensors / vllm / evalscope versions
match what the artifacts were saved with:

```bash
# Verify the latest version of the quantized checkpoint
docker run --rm --gpus all --ipc=host --network=host --shm-size=32g --ulimit memlock=-1 \
    -v "$PWD/test_download_and_sanity.py:/tmp/test_download_and_sanity.py:ro" \
    -e CLARIFAI_PAT="<your_pat>" \
    -e CLARIFAI_API_BASE="https://api.clarifai.com" \
    qe-step:local \
    python /tmp/test_download_and_sanity.py \
        --user_id <YOUR_USER_ID> \
        --app_id <YOUR_APP_ID> \
        --artifact_id quantized-checkpoint \
        --workdir /workspace/artifact-roundtrip

# Verify the latest eval-report (dual-eval bundle)
docker run --rm --ipc=host --network=host \
    -v "$PWD/test_download_eval_report.py:/tmp/test_download_eval_report.py:ro" \
    -e CLARIFAI_PAT="<your_pat>" \
    -e CLARIFAI_API_BASE="https://api.clarifai.com" \
    qe-step:local \
    python /tmp/test_download_eval_report.py \
        --user_id <YOUR_USER_ID> \
        --app_id <YOUR_APP_ID> \
        --artifact_id eval-report \
        --workdir /workspace/eval-roundtrip
```

Pass `--version_id <id>` to inspect a specific version; omit it and the
script picks the most recent.

---

## Files in this folder

```
.
├── README.md                         ← this file
├── config.yaml                       ← top-level Argo workflow + inputs
├── config-lock.yaml                  ← generated by `clarifai pipeline upload` (commit)
├── quantize-evaluate-step/
│   ├── config.yaml                   ← step interface (pipeline_step_input_params)
│   ├── Dockerfile                    ← step image
│   ├── requirements.txt
│   └── 1/
│       ├── pipeline_step.py          ← phase orchestrator
│       └── scripts/
│           ├── quantize_llmcompressor.sh
│           └── quantize_llmcompressor_worker.py
├── test_download_and_sanity.py       ← quantized-checkpoint round-trip
└── test_download_eval_report.py      ← eval-report round-trip + summary
```
