# LLM Compressor Quantization Pipeline

Clarifai pipeline that quantizes a HuggingFace causal LM with vLLM's
[llm-compressor](https://github.com/vllm-project/llm-compressor) and
uploads the resulting compressed-tensors checkpoint as a Clarifai
artifact you can later download.

One `pipeline_step.py`, given a target model + a few params, will:

1. Download the source model from HuggingFace.
2. Quantize with the chosen scheme + pipeline (datafree / basic / sequential).
3. Smoke-test the produced checkpoint with a load + chat-template generate.
4. Upload the checkpoint zip to Clarifai as a versioned `Artifact`.

The container bakes a pinned `llm-compressor` commit and a tested set of
`compressed-tensors` / `transformers` / `huggingface-hub` / `torchvision`
versions, so the run is reproducible.

---

## Defaults and the MoE NVFP4 variant

The defaults run a fast int4 weight-only sanity (Qwen2.5-0.5B, W4A16,
datafree, 1 GPU). For the W4A4 NVFP4 reference run on a small MoE model
(Granite-4-h-tiny) — pairing with NVFP4 means picking a scheme that needs
activation calibration, so `pipeline` must be empty (auto-infer →
sequential):

| Param | Fast sanity (default) | MoE NVFP4 reference |
|---|---|---|
| `src` | `Qwen/Qwen2.5-0.5B-Instruct` | `ibm-granite/granite-4.0-h-tiny` |
| `scheme` | `W4A16` | `NVFP4` |
| `pipeline` | `datafree` | `""` (auto-infer → sequential) |
| `num_gpus` | `1` | `1` |
| `user_id` / `app_id` / `artifact_id` | your Clarifai user/app / `quantized-checkpoint` | same |

Full param descriptions live in `pipeline_step.py --help`.

---

## Upload and run the pipeline

From inside this folder (`llmcompressor_pipeline/`), upload once:

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

Override params at runtime by repeating `--set key=value`:

```bash
# Reproduce the MoE NVFP4 reference run
clarifai pipeline run --instance=g6e.xlarge \
    --set src=ibm-granite/granite-4.0-h-tiny \
    --set scheme=NVFP4 \
    --set pipeline=
```

### Monitor

Visit `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` and check the
**Pipelines** tab to follow run progress. The uploaded checkpoint lands
under the **Artifacts** tab when the run finishes.

---

## Download the artifact and verify it locally

`test_download_and_sanity.py` (in this folder) downloads the produced
artifact via the Clarifai SDK, unzips it, then runs an HF transformers
load + chat-template generate to verify the checkpoint is
downstream-deployable. Run it inside the same container image used by
the pipeline so the torch / transformers / compressed-tensors versions
match what the artifact was saved with:

```bash
# One-time build of the container image
docker build -t llmcompressor-pipeline:local quantize-step/

# Download + verify the latest version of the artifact
docker run --rm --gpus all --ipc=host --network=host --shm-size=32g --ulimit memlock=-1 \
    -v "$PWD/test_download_and_sanity.py:/tmp/test_download_and_sanity.py:ro" \
    -e CLARIFAI_PAT="<your_pat>" \
    -e CLARIFAI_API_BASE="https://api.clarifai.com" \
    llmcompressor-pipeline:local \
    python /tmp/test_download_and_sanity.py \
        --user_id <YOUR_USER_ID> \
        --app_id <YOUR_APP_ID> \
        --artifact_id quantized-checkpoint \
        --workdir /workspace/artifact-roundtrip
```

Pass `--version_id <id>` to download a specific version; omit it and the
script picks the most recent version automatically. A coherent
`def reverse_list(lst): return lst[::-1]` answer at the end of the
script output means the artifact contains everything needed for
inference (`model.safetensors`, `config.json`, tokenizer files,
`recipe.yaml`).
