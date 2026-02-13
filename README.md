# Clarifai Pipeline Templates

This repository contains pipeline templates for training machine learning models on the Clarifai platform.

## Available Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `classifier-pipeline-resnet` | ResNet-based image classifier | Image classification tasks |
| `detector-pipeline-yolof` | YOLOF-based object detector | Object detection tasks |

## Prerequisites

Before getting started, ensure you have:

1. **Clarifai CLI installed**
   ```bash
   pip install clarifai
   ```

2. **A Clarifai account** with access to:
   - An **App** (note your `app_id`)
   - A **Dataset** uploaded to your app (note your `dataset_id`)
   - A **Compute Cluster** with GPU support
   - A **Nodepool** configured with GPU instances (e.g., `g6exlarge`)

3. **Your Personal Access Token (PAT)** from [Clarifai Settings](https://clarifai.com/settings/security)

## Quick Start Guide

### Step 1: Set Up Your Environment

Export your Clarifai PAT:

```bash
export CLARIFAI_PAT=<your_personal_access_token>
```

### Step 2: List Available Pipeline Templates

View all available pipeline templates:

```bash
clarifai pipelinetemplate list
```

### Step 3: Initialize a Pipeline from Template

Choose a template and initialize your pipeline:

**For image classification:**
```bash
clarifai pipeline init --template=classifier-pipeline-resnet
```

**For object detection:**
```bash
clarifai pipeline init --template=detector-pipeline-yolof
```

This creates a local pipeline directory with configuration files.

### Step 4: Configure Your Pipeline

Open and update the existing `config.yaml` file with your customized ID and hyperparameter values, e.g. set:

- `user_id`: Your Clarifai user ID
- `app_id`: Your application ID
- `dataset_id`: Your dataset ID
- `model_id`: The ID you want to assign to the model created by this pipeline
- Other training hyperparameters as needed

> **Note:** Need to upload a dataset first? Follow some examples at [https://github.com/Clarifai/examples/tree/main/datasets/upload](https://github.com/Clarifai/examples/tree/main/datasets/upload):
> - For **object detection**: Use the VOC dataset example
> - For **image classification**: Use the Food101 dataset example

### Step 5: Upload the Pipeline

Upload your configured pipeline to Clarifai:

```bash
clarifai pipeline upload
```

**Important:** After entering the above command, record the **Pipeline Version ID** displayed in the terminal output. You'll need this for the next step.

Example output:
```
Pipeline version created with ID: b42eaf86fa32434a901a33b779cb828c
```

### Step 6: Launch the Pipeline Run

Run the pipeline with your compute resources:

```bash
clarifai pipeline run \
  --pipeline_id=<your_pipeline_id> \
  --user_id=<your_user_id> \
  --app_id=<your_app_id> \
  --pipeline_version_id=<pipeline_version_id_from_step_5> \
  --nodepool_id=<your_nodepool_id> \
  --compute_cluster_id=<your_compute_cluster_id>
```

**Example:**
```bash
clarifai pipeline run \
  --pipeline_id=classifier-pipeline \
  --user_id=john_doe \
  --app_id=my_training_app \
  --pipeline_version_id=b42eaf86fa32434a901a33b779cb828c \
  --nodepool_id=pool-g6exlarge-abc123 \
  --compute_cluster_id=cluster-aws-us-east-1-xyz789
```

**Important:** After running the command, record the **Pipeline Version Run ID** for monitoring.

Example output:
```
Pipeline version run created with ID: d98503678c4d42e3b1e77cd335302763
```

## Monitoring Your Pipeline

### View Logs

1. Go to the [Clarifai Platform](https://clarifai.com)
2. Navigate to your App
3. Find your pipeline under the **Pipelines** section
4. Click on the pipeline name and find the run ID to view logs and status

### Check Training Results

After the pipeline completes successfully:

1. **Trained Model**: A new model version will be created for your chosen App -> Models -> Your chosen model ID.
2. **Fine-tuned Checkpoint**: The checkpoint is usually uploaded to pipeline artifacts
   - View and download via the Platform UI (TO BE IMPLEMENTED BY UI)
   - Or download via CLI/SDK