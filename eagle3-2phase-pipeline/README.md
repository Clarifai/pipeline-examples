# Eagle3 Two-Phase Training Pipeline

Pipeline template for Eagle3 speculative decoding draft head training using SpecForge.
Runs the complete two-phase training end-to-end in a single pipeline run, followed by
model upload to Clarifai.

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

# Quick sanity check (~1 hr)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --sg_n 100 --uc_n 100 --pb_n 100 \
    --sg_target 300 --uc_target 300 --pb_target 300 \
    --batch_size 2 --max_length 1024 --sglang_mem_frac 0.5 \
    --phase1_epochs 1 --phase2_epochs 3 --learning_rate 5e-5 \
    --regen_temperature 0 \
    --app_id <YOUR_APP_ID> \
    --model_id qwen3-8b-eagle3-test"

# Full training with defaults (~2 days on a single GPU)
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
| `phase1_epochs` | 3 | Phase-1 pretrain epochs |
| `phase2_epochs` | 3 | Phase-2 finetune epochs |
| `learning_rate` | 2e-5 | Phase-2 LR (phase-1 is hardcoded 1e-4) |
| `regen_temperature` | 0.8 | Regen sampling temp (0 = greedy/deterministic) |
| `app_id` | pipeline-app | Clarifai app ID for model upload |
| `model_id` | qwen3-8b-eagle3 | Clarifai model ID for upload |

## Training flow

`pipeline_step.py` runs these steps sequentially:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_prepare_data.sh` | Prepare training data blend (SG + UC + PB) |
| 2 | `02_phase1_pretrain.sh` | Phase-1: pretrain Eagle3 draft head |
| 3 | `03_regenerate_data.sh` | Regenerate data per dataset (sharegpt, ultrachat, perfectblend) |
| 4 | `04_combine_regen.sh` | Merge regen outputs into single file |
| 5 | `05_phase2_finetune.sh` | Phase-2: finetune on regenerated data |
| 6 | Model upload | Upload model version + checkpoint artifact to Clarifai |
