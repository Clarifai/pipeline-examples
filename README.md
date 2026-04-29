# Clarifai Pipeline Templates

This repository contains pipeline templates for training machine learning models on the Clarifai platform.

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

### Step 2: Browse Available Templates

```bash
clarifai pipelinetemplate ls
```

### Step 3: Initialize a Pipeline from Template

```bash
clarifai pipeline init --template=classifier-pipeline-resnet-quick-start
```

This creates a new folder named after the template. **`cd` into that folder before
running any of the subsequent `clarifai pipeline ...` commands** — they read the
local `config.yaml` / `config-lock.yaml`:

```bash
cd classifier-pipeline-resnet-quick-start
```

Optional — override defaults at init time (different user/app from your `clarifai login`,
a custom pipeline ID, or a model parameter default):

```bash
clarifai pipeline init --template=classifier-pipeline-resnet-quick-start \
  --user_id MY_CUSTOM_USER_ID --app_id MY_CUSTOM_APP_ID \
  --set id=MY_CUSTOM_PIPELINE_ID --set num_epochs=20
```

### Step 4: Upload and Run the Pipeline

Make sure you are inside the generated pipeline folder (e.g. `classifier-pipeline-resnet-quick-start/`)
from Step 3, then upload:

```bash
clarifai pipeline upload
```

Then run the pipeline using **one** of the two compute options:

```bash
# (a) Simplest — auto-create or reuse compute from an instance type
clarifai pipeline run --instance=g6e.xlarge

# (b) Use your existing nodepool + compute cluster (both flags required)
clarifai pipeline run \
  --nodepool_id=<your_existing_nodepool_id> \
  --compute_cluster_id=<your_existing_compute_cluster_id>
```

To override pipeline parameters at run time, repeat `--set key=value`:

```bash
clarifai pipeline run --instance=g6e.xlarge --set num_epochs=20 --set batch_size=32
```

### Step 5: Monitor Your Pipeline

Go to https://clarifai.com/YOUR_USER_ID/YOUR_APP_ID, check the Pipelines tab to monitor your pipeline and check the Models tab to find your model once training is done.

## Available Templates

### Quick-Start Pipelines — Try These First!

Quick-start pipelines come with **default public datasets** pre-configured, so you can launch them right away to see an end-to-end training run — no data preparation needed.

| Template | Description |
|----------|-------------|
| `classifier-pipeline-resnet-quick-start` | Image classification with ResNet and sample dataset |
| `detector-pipeline-yolof-quick-start` | Object detection with YOLOF and sample dataset |
| `detector-pipeline-eval-yolof-quick-start` | Evaluation pipeline for a pretrained YOLOF detector — runs inference on a dataset and reports COCO detection metrics (no training) |
| `lora-pipeline-unsloth-quick-start` | LLM LoRA fine-tuning with Unsloth and sample dataset |

### Other Pipeline Examples

These are diverse pipelines (some of them may require additional setting up, e.g. a Clarifai dataset as a prerequisite).

| Template | Description |
|----------|-------------|
| `classifier-pipeline-resnet` | ResNet-based image classifier |
| `detector-pipeline-yolof` | YOLOF-based object detector |
| `detector-pipeline-dfine` | D-FINE-based object detector |