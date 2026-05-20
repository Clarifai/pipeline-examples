# D-FINE Object Detection Pipeline (`detector-pipeline-dfine`)

A Clarifai pipeline example that fine-tunes a D-FINE object detector (HuggingFace `transformers`) on either a Clarifai dataset or a pre-packaged artifact dataset, exports ONNX, and uploads the model to Clarifai.

## What's in this example

A Clarifai pipeline consists of three things; this example illustrates all of them:

- `config.yaml` — pipeline-level config (Argo orchestration spec + pipeline parameters).
- `detector-pipeline-dfine-ps/` — the pipeline step:
  - `config.yaml` — compute requirements + per-step input parameters.
  - `Dockerfile` + `requirements.txt` — container/runtime environment.
  - `1/pipeline_step.py` — entry point that wires CLI args into the model's `train()` method.
  - `1/models/model/1/model.py` — the actual training/export logic (D-FINE `VisualDetectorClass`).

## What this pipeline does

- **Get data**: either downloads a Clarifai dataset (set `clarifai_dataset_id`) and converts it to COCO format, or downloads a pre-packaged VOC-style dataset bundled as a Clarifai artifact (default).
- **Pretrained checkpoint**: pulls a D-FINE pretrained model from HuggingFace (`pretrained_model`, default `ustc-community/dfine-small-obj2coco`).
- **Train**: fine-tunes D-FINE using the HuggingFace `Trainer` on the COCO-format dataset.
- **Export**: exports the fine-tuned model to ONNX and places the checkpoint into a Clarifai model layout.
- **Upload**: uploads the checkpoint artifact and the model to Clarifai, ready for deployment (with optional TensorRT inference path).

## Parameters

`config.yaml` exposes parameters users typically override (`clarifai_dataset_id`, `dataset_version_id`, `concepts`, `pretrained_model`, `num_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_steps`, `seed`, `streaming_video`, etc.). For full descriptions see `detector-pipeline-dfine-ps/config.yaml` and the `train()` method signature in `detector-pipeline-dfine-ps/1/models/model/1/model.py`.

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
clarifai pipeline init --template=detector-pipeline-dfine
cd detector-pipeline-dfine
```

Optional — override defaults at init time (different user/app, a custom pipeline ID, or a parameter default):

```bash
clarifai pipeline init --template=detector-pipeline-dfine \
  --user_id MY_USER_ID --app_id MY_APP_ID \
  --set id=MY_PIPELINE_ID --set num_epochs=100
```

### Step 3: Upload and Run the Pipeline

```bash
clarifai pipeline upload
```

Then run with one of the two compute options:

```bash
# (a) Simplest — auto-create or reuse compute from an instance type
clarifai pipeline run --instance=g6e.xlarge

# (b) Use your existing nodepool + compute cluster
clarifai pipeline run \
  --nodepool_id=<your_existing_nodepool_id> \
  --compute_cluster_id=<your_existing_compute_cluster_id>
```

To train on your own Clarifai dataset instead of the default artifact, pass it at run time:

```bash
clarifai pipeline run --instance=g6e.xlarge \
  --set clarifai_dataset_id=<YOUR_DATASET_ID> \
  --set concepts='["bird","cat"]'
```

### Step 4: Monitor

Go to `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` — the **Pipelines** tab tracks the run, and the **Models** tab is where the fine-tuned detector lands when training finishes.

## Datasets, artifacts, and deployment

- Pipelines overview: https://docs.clarifai.com/compute/pipelines/
- Prepare a Clarifai dataset for training (object detection with bounding boxes): https://docs.clarifai.com/create/datasets/create
- Manage Clarifai artifacts (this pipeline uploads the checkpoint as an artifact and the model for deployment): https://docs.clarifai.com/create/artifacts/manage

The final step in `model.py` uploads the trained checkpoint as a Clarifai artifact (so it can be retrieved later) and pushes the model to Clarifai for deployment.

## Custom

This is an example — grab it, refactor `model.py` and the pipeline step with the Clarifai SDK, and shape the workflow to your own dataset, pretrained backbone, augmentations, and export targets (ONNX/TensorRT).
