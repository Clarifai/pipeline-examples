# Eagle3 Two-Phase Training Pipeline

Pipeline template for Eagle3 speculative decoding draft head training using SpecForge.
Runs the complete two-phase training end-to-end in a single pipeline run, followed by
model upload to Clarifai.

All parameters have production defaults. Override any subset for a quick sanity check
or custom training run.

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

### 1. Initialize from template

```bash
pip install clarifai
clarifai login

# Initialize — auto-fills your user_id and app_id
clarifai pipeline init --template=eagle3-2phase-pipeline

cd eagle3-2phase-pipeline
```

Or if cloning directly, replace `<YOUR_USER_ID>` and `<YOUR_APP_ID>` in `config.yaml` and `eagle3-train-ps/config.yaml`.

### 2. Upload pipeline

```bash
clarifai pipeline upload .
```

### 3. Run pipeline

```bash
# Full production training (all defaults, several hours)
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance>

# Quick sanity check (~40 min)
clarifai pipeline run --config config-lock.yaml --instance <gpu-instance> \
    --set sg_n="100" --set uc_n="100" --set pb_n="100" \
    --set sg_target="300" --set uc_target="300" --set pb_target="300" \
    --set batch_size="2" --set max_length="1024" --set sglang_mem_frac="0.5" \
    --set phase1_epochs="1" --set phase2_epochs="3" --set learning_rate="5e-5" \
    --set regen_temperature="0" \
    --set model_id="qwen3-8b-eagle3-test"
```

## Tuneable parameters

All parameters have defaults. Override only what you need.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sg_n` | 24000 | ShareGPT rows in phase-1 blend (0 disables) |
| `uc_n` | 19000 | UltraChat rows (0 disables) |
| `pb_n` | 11000 | PerfectBlend rows (0 disables) |
| `sg_target` | 25000 | Target ShareGPT rows for phase-2 regen |
| `uc_target` | 25000 | Target UltraChat rows for phase-2 regen |
| `pb_target` | 11000 | Target PerfectBlend rows for phase-2 regen |
| `batch_size` | 4 | Per-device batch size |
| `max_length` | 4096 | Max sequence length |
| `sglang_mem_frac` | 0.6 | SGLang memory fraction |
| `dataloader_num_workers` | 8 | Torch dataloader workers |
| `phase1_epochs` | 10 | Phase-1 pretrain epochs |
| `phase2_epochs` | 6 | Phase-2 finetune epochs |
| `learning_rate` | 2e-5 | Phase-2 LR (phase-1 is hardcoded 1e-4) |
| `regen_temperature` | 0.8 | Regen sampling temp (0 = greedy/deterministic) |
| `model_id` | qwen3-8b-eagle3 | Clarifai model ID for upload |

## Local testing

Build and test in Docker before deploying to cloud.
Requires a GPU with at least 80GB VRAM (e.g. GH200, A100 80GB, H100). GPU must be free.

### Required environment variables

```bash
export CLARIFAI_PAT="..."           # Clarifai PAT for model upload
export CLARIFAI_USER_ID="..."       # Your Clarifai user ID
export CLARIFAI_APP_ID="..."        # Your Clarifai app ID
```

```bash
# Build
cd eagle3-2phase-pipeline/eagle3-train-ps
docker build -t eagle3-train-ps:local .

# Start container
docker run -d --name eagle3-test \
    --gpus all --ipc=host --network=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CLARIFAI_PAT -e CLARIFAI_USER_ID -e CLARIFAI_APP_ID \
    -v $HOME/.cache/huggingface:/workspace/eagle3/cache/huggingface \
    --entrypoint bash eagle3-train-ps:local -c "tail -f /dev/null"

# Quick sanity check (~40 min)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py \
    --sg_n 100 --uc_n 100 --pb_n 100 \
    --sg_target 300 --uc_target 300 --pb_target 300 \
    --batch_size 2 --max_length 1024 --sglang_mem_frac 0.5 \
    --phase1_epochs 1 --phase2_epochs 3 --learning_rate 5e-5 \
    --regen_temperature 0 \
    --model_id qwen3-8b-eagle3-test"

# Full production training (several hours)
docker exec eagle3-test bash -c \
    "cd /home/nonroot/main && python3 1/pipeline_step.py"

# Cleanup
docker rm -f eagle3-test
```

In cloud, `CLARIFAI_PAT` and `CLARIFAI_USER_ID` are auto-injected by the platform.

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
