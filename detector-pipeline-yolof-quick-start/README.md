# YOLOF Detector Quick-Start Pipeline (`detector-pipeline-yolof-quick-start`)

A quick-start Clarifai pipeline that fine-tunes a YOLOF object detector on a **pre-packaged public VOC-style dataset**. No dataset setup needed — launch it and watch an end-to-end training run.

## What's in this example

A Clarifai pipeline consists of two things; this example illustrates both of them:

- `config.yaml` — pipeline-level config (Argo orchestration spec + pipeline parameters).
- `model-version-train-ps/` — the pipeline step:
  - `config.yaml` — compute requirements + per-step input parameters.
  - `Dockerfile` + `requirements.txt` — container/runtime environment.
  - `1/pipeline_step.py` — entry point that wires CLI args into the model's `train()` method.
  - `1/models/model/1/model.py` — the actual training logic (YOLOF `VisualDetectorClass`).

## What this pipeline does

- **Get data**: downloads a default public VOC-style detection dataset packaged as a Clarifai artifact and unzips it into COCO format.
- **Pretrained checkpoint**: downloads a COCO-pretrained YOLOF ResNet-50-C5 checkpoint from Clarifai's artifact store.
- **Train**: generates an MMDetection config for the dataset and trains YOLOF with the MMEngine `Runner`.
- **Benchmark**: profiles the trained checkpoint to size GPU requirements and updates the model's `config.yaml`.
- **Upload**: exports the checkpoint + config into a Clarifai model, ready for deployment.

## Parameters

`config.yaml` exposes the parameters users typically override (`image_size`, `keep_aspect_ratio`, `max_aspect_ratio`, `batch_size`, `num_epochs`, `min_samples_per_epoch`, `per_item_lrate`, `pretrained_weights`, `frozen_stages`, `inference_max_batch_size`, `seed`). For full descriptions see `model-version-train-ps/config.yaml` and the `train()` method signature in `model-version-train-ps/1/models/model/1/model.py`.

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
clarifai pipeline init --template=detector-pipeline-yolof-quick-start
cd detector-pipeline-yolof-quick-start
```

Optional — override defaults at init time (different user/app, a custom pipeline ID, or a parameter default):

```bash
clarifai pipeline init --template=detector-pipeline-yolof-quick-start \
  --user_id MY_USER_ID --app_id MY_APP_ID \
  --set id=MY_PIPELINE_ID --set num_epochs=50
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

Override parameters at run time by repeating `--set key=value`:

```bash
clarifai pipeline run --instance=g6e.xlarge --set num_epochs=20 --set batch_size=8
```

### Step 4: Monitor

Go to `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` — the **Pipelines** tab tracks the run, and the **Models** tab is where the trained detector lands when training finishes.

## Datasets, artifacts, and deployment

- Pipelines overview: https://docs.clarifai.com/compute/pipelines/
- This quick-start uses a **public artifact dataset** out of the box. To train on your own data instead, prepare a Clarifai dataset: https://docs.clarifai.com/create/datasets/create
- Manage Clarifai artifacts (the example uploads the trained checkpoint as a model so it can be deployed; you can also persist intermediate objects to artifacts): https://docs.clarifai.com/create/artifacts/manage

The final step in `model.py` uploads the trained checkpoint as a Clarifai model so it is immediately ready for deployment.

## Custom

This is an example — grab it, refactor `model.py` and the pipeline step with the Clarifai SDK, and shape the workflow to your own dataset, augmentations, and training loop.
