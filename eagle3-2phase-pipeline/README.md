# Eagle3 Two-Phase Speculative Decoding Pipeline (`eagle3-2phase-pipeline`)

A Clarifai pipeline that trains an Eagle3 speculative decoding draft head for Qwen3-8B. Supports full two-phase training (pretrain + finetune) or finetune-only from a pretrained checkpoint. Uploads the trained model to Clarifai for deployment with SGLang.

## What's in this example

A Clarifai pipeline consists of two things; this example illustrates both of them:

- `config.yaml` — pipeline-level config (Argo orchestration spec + pipeline parameters).
- `eagle3-train-ps/` — the pipeline step:
  - `config.yaml` — compute requirements + per-step input parameters.
  - `Dockerfile` + `requirements.txt` — container/runtime environment.
  - `1/pipeline_step.py` — orchestrator that runs all training steps sequentially.
  - `1/scripts/` — training bash scripts (data prep, pretrain, regen, combine, finetune).
  - `1/model_upload_helper.py` — Clarifai model + artifact upload.
  - `1/qwen-with-eagle3-model/` — model runner template (uploaded to Clarifai for serving).
  - `1/configs/` — draft model configs.

## What this pipeline does

- **Prepare data**: downloads ShareGPT, UltraChat, and PerfectBlend datasets, blends and shuffles them into a training set.
- **Phase-1 pretrain**: trains the Eagle3 draft head from scratch on the blended data using SpecForge's `train_eagle3.py`.
- **Phase-1 artifact upload**: uploads the pretrained checkpoint as a Clarifai artifact for future finetune-only runs.
- **Regenerate data**: uses the target model (Qwen3-8B) to regenerate training data per dataset via SGLang.
- **Combine regen**: merges regenerated outputs into a single training file.
- **Phase-2 finetune**: finetunes the draft head on regenerated data, starting from the phase-1 checkpoint.
- **Upload**: uploads the trained model to Clarifai (served via SGLang with Eagle3 speculative decoding) and stores the checkpoint as an artifact.

**Finetune-only mode** (`pretrained_ckpt` set): skips phase-1 entirely — downloads a pretrained checkpoint artifact and starts from regeneration. When set, phase-1 params (`sg_n`, `uc_n`, `pb_n`, `phase1_epochs`) are ignored.

## Parameters

`config.yaml` exposes the parameters users typically override (`pretrained_ckpt`, `sg_n`, `uc_n`, `pb_n`, `sg_target`, `uc_target`, `pb_target`, `batch_size`, `max_length`, `sglang_mem_frac`, `dataloader_num_workers`, `phase1_epochs`, `phase2_epochs`, `learning_rate`, `regen_temperature`, `app_id`, `model_id`). For full descriptions see `eagle3-train-ps/config.yaml`.

## Quick Start Guide

### Step 1: Set Up Your Environment

```bash
pip install clarifai
clarifai login   # interactive: paste your Personal Access Token when prompted
```

Alternatively, set the PAT non-interactively:

```bash
export CLARIFAI_PAT=<your_personal_access_token>
```

### Step 2: Initialize the Pipeline from Template

```bash
clarifai pipeline init --template=eagle3-2phase-pipeline
cd eagle3-2phase-pipeline
```

### Step 3: Upload and Run the Pipeline

```bash
clarifai pipeline upload .
```

Quick sanity check (~1 hr):

```bash
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set sg_n="100" --set uc_n="100" --set pb_n="100" \
    --set sg_target="300" --set uc_target="300" --set pb_target="300" \
    --set batch_size="2" --set max_length="1024" --set sglang_mem_frac="0.5" \
    --set phase1_epochs="1" --set phase2_epochs="3" --set learning_rate="5e-5" \
    --set regen_temperature="0" \
    --set model_id="qwen3-8b-eagle3-test"
```

Full training with defaults (~1.5 days on a single GPU):

```bash
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance>
```

Finetune-only sanity check (~1 hr):

```bash
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set pretrained_ckpt="<artifact_id>" \
    --set sg_target="300" --set uc_target="300" --set pb_target="300" \
    --set batch_size="2" --set max_length="1024" --set sglang_mem_frac="0.5" \
    --set phase2_epochs="3" --set learning_rate="5e-5" \
    --set regen_temperature="0" \
    --set model_id="qwen3-8b-eagle3-ft-test"
```

Full finetune with defaults:

```bash
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set pretrained_ckpt="<artifact_id>"
```

To use your own cluster and nodepool, replace `--instance <gpu-instance>` with `--compute_cluster_id <your-cluster-id> --nodepool_id <your-nodepool-id>`. Override parameters at run time by repeating `--set key=value`.

### Step 4: Monitor

Go to `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` — the **Pipelines** tab tracks the run, and the **Models** tab is where the trained model lands when training finishes.

## Run locally

Requires a GPU with at least 80GB VRAM. GPU must be free.

```bash
export CLARIFAI_PAT="..."           # Clarifai PAT for model upload
export CLARIFAI_USER_ID="..."       # Your Clarifai user ID

# Build
cd eagle3-train-ps
docker build -t eagle3-train-ps:local .

# Start container
docker run -d --name eagle3-test \
    --gpus all --ipc=host --network=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CLARIFAI_PAT -e CLARIFAI_USER_ID -e CLARIFAI_APP_ID \
    -v $HOME/.cache/huggingface:/workspace/eagle3/cache/huggingface \
    --entrypoint bash eagle3-train-ps:local -c "tail -f /dev/null"

# Quick sanity check — full 2-phase (~1 hr)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --sg_n 100 --uc_n 100 --pb_n 100 \
    --sg_target 300 --uc_target 300 --pb_target 300 \
    --batch_size 2 --max_length 1024 --sglang_mem_frac 0.5 \
    --phase1_epochs 1 --phase2_epochs 3 --learning_rate 5e-5 \
    --regen_temperature 0 \
    --app_id <YOUR_APP_ID> \
    --model_id qwen3-8b-eagle3-test"

# Quick finetune-only sanity check (~1 hr)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --pretrained_ckpt <artifact_id> \
    --sg_target 300 --uc_target 300 --pb_target 300 \
    --batch_size 2 --max_length 1024 --sglang_mem_frac 0.5 \
    --phase2_epochs 3 --learning_rate 5e-5 \
    --regen_temperature 0 \
    --app_id <YOUR_APP_ID> \
    --model_id qwen3-8b-eagle3-ft-test"

# Full finetune-only from artifact
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --pretrained_ckpt <artifact_id> \
    --app_id <YOUR_APP_ID>"

# Full 2-phase training with defaults (~2 days on a single GPU)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --app_id <YOUR_APP_ID>"

# Cleanup
docker rm -f eagle3-test
```

In cloud, `CLARIFAI_PAT` and `CLARIFAI_USER_ID` are auto-injected by the platform.

## Datasets, artifacts, and deployment

- Pipelines overview: https://docs.clarifai.com/compute/pipelines/
- This pipeline downloads public datasets (ShareGPT, UltraChat, PerfectBlend) automatically via SpecForge. To adjust the data blend, override the `sg_n`, `uc_n`, `pb_n` parameters.
- Manage Clarifai artifacts (the pipeline uploads both the model and a checkpoint artifact for future finetune-only runs): https://docs.clarifai.com/create/artifacts/manage

## Custom

This is an example — grab it, modify the scripts and pipeline step with the Clarifai SDK, and adapt it for your own target model, datasets, and training schedule.
