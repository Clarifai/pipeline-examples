# Eagle3 Two-Phase Training Pipeline

Pipeline template for Eagle3 speculative decoding draft head training using SpecForge.
Supports two modes:

- **Full 2-phase** (default): pretrain + regen + finetune end-to-end
- **Finetune-only**: skip pretraining, start from a pretrained checkpoint artifact

Both modes upload the trained model to Clarifai for deployment.

## Structure

```
eagle3-2phase-pipeline/
├── config.yaml                          # Pipeline orchestration (Argo spec)
├── .gitignore
└── eagle3-train-ps/                     # Pipeline step
    ├── config.yaml                      # Step compute config + input params
    ├── Dockerfile                       # sglang:v0.5.9 + SpecForge
    ├── requirements.txt
    └── 1/
        ├── pipeline_step.py             # Orchestrator: runs all steps sequentially
        ├── model_upload_helper.py       # Clarifai model + artifact upload
        ├── qwen-with-eagle3-model/      # Model runner template (uploaded to Clarifai)
        ├── scripts/                     # Training bash scripts
        └── configs/                     # Draft model configs
```

## Setup

```bash
pip install clarifai
clarifai login

# Initialize — auto-fills your user_id and app_id
clarifai pipeline init --template=eagle3-2phase-pipeline

cd eagle3-2phase-pipeline
```

## Upload pipeline and run

```bash
clarifai pipeline upload .

# Quick sanity check (~1 hr)
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set sg_n="100" --set uc_n="100" --set pb_n="100" \
    --set sg_target="300" --set uc_target="300" --set pb_target="300" \
    --set batch_size="2" --set max_length="1024" --set sglang_mem_frac="0.5" \
    --set phase1_epochs="1" --set phase2_epochs="3" --set learning_rate="5e-5" \
    --set regen_temperature="0" \
    --set model_id="qwen3-8b-eagle3-test"

# Full training with defaults (~1.5 days on a single GPU)
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance>

```

To use your own cluster and nodepool, replace `--instance <gpu-instance>` with `--compute_cluster_id <your-cluster-id> --nodepool_id <your-nodepool-id>`.

## Finetune-only mode

If you already have a pretrained Eagle3 checkpoint (e.g. from a previous phase-1 run), you can skip pretraining and go straight to finetuning. 

```bash
# Quick finetune sanity check (~1 hr)
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set pretrained_ckpt="<artifact_id>" \
    --set sg_target="300" --set uc_target="300" --set pb_target="300" \
    --set batch_size="2" --set max_length="1024" --set sglang_mem_frac="0.5" \
    --set phase2_epochs="3" --set learning_rate="5e-5" \
    --set regen_temperature="0" \
    --set model_id="qwen3-8b-eagle3-ft-test"

# Full finetune with defaults
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set pretrained_ckpt="<artifact_id>"
```

When `pretrained_ckpt` is set, phase-1 params (`sg_n`, `uc_n`, `pb_n`, `phase1_epochs`) are ignored. The pipeline downloads the checkpoint, prepares regen datasets, and runs finetune + upload.

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

## Tunable parameters

All parameters have defaults. Override only what you need.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_ckpt` | *(empty)* | Pretrained checkpoint artifact. If set, skips phase-1. |
| `sg_n` | 12000 | ShareGPT rows in phase-1 blend (0 disables) |
| `uc_n` | 10000 | UltraChat rows (0 disables) |
| `pb_n` | 5000 | PerfectBlend rows (0 disables) |
| `sg_target` | 15000 | Target ShareGPT rows for phase-2 regen |
| `uc_target` | 15000 | Target UltraChat rows for phase-2 regen |
| `pb_target` | 5000 | Target PerfectBlend rows for phase-2 regen |
| `batch_size` | 2 | Per-device batch size |
| `max_length` | 2048 | Max sequence length |
| `sglang_mem_frac` | 0.6 | SGLang memory fraction |
| `dataloader_num_workers` | 8 | Torch dataloader workers |
| `phase1_epochs` | 3 | Phase-1 pretrain epochs (ignored in finetune-only) |
| `phase2_epochs` | 3 | Phase-2 finetune epochs |
| `learning_rate` | 2e-5 | Phase-2 LR (phase-1 is hardcoded 1e-4) |
| `regen_temperature` | 0.8 | Regen sampling temp (0 = greedy/deterministic) |
| `app_id` | pipeline-app | Clarifai app ID for model upload |
| `model_id` | qwen3-8b-eagle3 | Clarifai model ID for upload |

## Training flow

**Full 2-phase mode** (default):

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_prepare_data.sh` | Prepare training data blend (SG + UC + PB) |
| 2 | `02_phase1_pretrain.sh` | Phase-1: pretrain Eagle3 draft head |
| — | Phase-1 artifact upload | Upload pretrained checkpoint for future finetune runs |
| 3 | `03_regenerate_data.sh` | Regenerate data per dataset |
| 4 | `04_combine_regen.sh` | Merge regen outputs into single file |
| 5 | `05_phase2_finetune.sh` | Phase-2: finetune on regenerated data |
| 6 | Model upload | Upload model version + checkpoint artifact to Clarifai |

**Finetune-only mode** (`--pretrained_ckpt` set):

| Step | Description |
|------|-------------|
| Download checkpoint | Fetch pretrained Eagle3 draft head from Clarifai artifact |
| Download datasets | Prepare regen input datasets |
| 3 | Regenerate data per dataset |
| 4 | Merge regen outputs |
| 5 | Finetune on regenerated data |
| 6 | Upload model + checkpoint artifact |
