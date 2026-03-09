# Clarifai Pipeline Templates

This repository contains pipeline templates for training machine learning models on the Clarifai platform.

## Quick Start Guide

### Step 1: Set Up Your Environment

```bash
pip install clarifai
export CLARIFAI_PAT=<your_personal_access_token>
```

### Step 2: Initialize a Pipeline from Template

```bash
clarifai pipeline init --template=classifier-pipeline-resnet-quick-start
cd classifier-pipeline-resnet-quick-start
```

### Step 3: Upload and Run the Pipeline

```bash
clarifai pipeline upload
clarifai pipeline run --nodepool_id=<your_nodepool_id> --compute_cluster_id=<your_compute_cluster_id>
```

### Step 4: Monitor Your Pipeline

Go to https://clarifai.com/YOUR_USER_ID/YOUR_APP_ID, check the Pipelines tab to monitor your pipeline and check the Models tab to find your model once training is done.

## Available Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `classifier-pipeline-resnet-quick-start` | Quick start ResNet-based image classifier | Image classification with sample dataset |
| `detector-pipeline-yolof-quick-start` | Quick start YOLOF-based object detector | Object detection with sample dataset |
| `classifier-pipeline-resnet` | ResNet-based image classifier | Image classification tasks |
| `detector-pipeline-yolof` | YOLOF-based object detector | Object detection tasks |
| `lora-pipeline-unsloth` | LoRA fine-tuning with Unsloth | LLM fine-tuning tasks |
| `benchmark-gpu-memory-pipeline` | GPU memory benchmark pipeline | GPU memory testing and benchmarking |

## Prerequisites

Before getting started, ensure you have:

1. **Clarifai CLI installed**
   ```bash
   pip install clarifai
   ```

2. **A Clarifai account** with access to:
   - An **App** 
   - A **Compute Cluster** with GPU support
   - A **Nodepool** configured with GPU instances (e.g., `g6exlarge`)

3. **Your Personal Access Token (PAT)** from [Clarifai Settings](https://clarifai.com/settings/security)
