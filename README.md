# Clarifai Pipeline Templates

This repository contains pipeline templates for training machine learning models on the Clarifai platform.

## Quick Start Guide

### Step 1: Set Up Your Environment

```bash
pip install clarifai
export CLARIFAI_PAT=<your_personal_access_token>
```

### Step 2: Browse Available Templates

```bash
clarifai pipelinetemplate ls
```

### Step 3: Initialize a Pipeline from Template

```bash
clarifai pipeline init --template=classifier-pipeline-resnet-quick-start
cd classifier-pipeline-resnet-quick-start
```

### Step 4: Upload and Run the Pipeline

```bash
clarifai pipeline upload
clarifai pipeline run --nodepool_id=<your_nodepool_id> --compute_cluster_id=<your_compute_cluster_id>
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
| `lora-pipeline-unsloth-quick-start` | LLM LoRA fine-tuning with Unsloth and sample dataset |

### Standard Pipelines

These are diverse pipelines (some of them may require additional setting up, e.g. a Clarifai dataset as a prerequisite).

| Template | Description |
|----------|-------------|
| `classifier-pipeline-resnet` | ResNet-based image classifier |
| `detector-pipeline-yolof` | YOLOF-based object detector |
| `detector-pipeline-dfine` | D-FINE-based object detector |