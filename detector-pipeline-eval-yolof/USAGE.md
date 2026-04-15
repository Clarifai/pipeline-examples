# How to Use the YOLOF Evaluation Pipeline

This guide covers how to run the `detector-pipeline-eval-yolof` pipeline both locally and on the Clarifai platform.

## What This Pipeline Does

Evaluates a pretrained YOLOF (You Only Look One-level Feature) object detector on a dataset and computes standard COCO detection metrics. **It does not train a model.**

## Prerequisites

- Python 3.11
- A [Clarifai Personal Access Token (PAT)](https://clarifai.com/settings/security)
- For local runs: ~200MB disk space for the checkpoint + dataset

## Local Setup

### 1. Install Dependencies

```bash
# PyTorch (CPU-only for local testing)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# MMDetection stack
pip install openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"

# Other dependencies
pip install clarifai==12.1.4 "clarifai-protocol>=0.0.23,<0.0.36" clarifai-grpc==12.0.2 \
  boto3==1.42.4 scipy pycocotools Pillow yapf
```

> **Note:** On macOS, `mmcv` builds from source and requires `setuptools<82`. Run `pip install "setuptools<82,>=69.0"` if you hit a `pkg_resources` error.

### 2. Set Your Clarifai PAT

```bash
export CLARIFAI_PAT="your_personal_access_token"
```

### 3. Run the Pipeline

Navigate to the pipeline step directory and run:

```bash
cd detector-pipeline-eval-yolof/detector-pipeline-eval-yolof-ps/1

python3 pipeline_step.py \
  --user_id=clarifai \
  --app_id=train_pipelines \
  --dataset_source=artifact \
  --concepts='["bird","cat"]' \
  --image_size='[512]' \
  --max_aspect_ratio=1.5 \
  --keep_aspect_ratio=true \
  --batch_size=4 \
  --score_threshold=0.05 \
  --iou_threshold=0.6 \
  --pretrained_weights=coco
```

Results are saved to `/tmp/yolof_eval_work_dir/eval_results.json`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | required | Clarifai user ID |
| `app_id` | string | required | Clarifai application ID |
| `dataset_source` | string | `"artifact"` | `"artifact"` for pre-packaged dataset, `"clarifai"` for a Clarifai platform dataset |
| `dataset_id` | string | `""` | Dataset ID (required when `dataset_source=clarifai`) |
| `dataset_version_id` | string | `""` | Dataset version ID (required when `dataset_source=clarifai`) |
| `concepts` | string | `'["bird","cat"]'` | JSON array of class names to evaluate |
| `image_size` | string | `'[512]'` | Image size as JSON array. Single value = min side length |
| `max_aspect_ratio` | float | `1.5` | Max aspect ratio multiplier when `keep_aspect_ratio=true` |
| `keep_aspect_ratio` | bool | `true` | Preserve image aspect ratio during resize |
| `batch_size` | int | `8` | Images per inference batch (reduce for CPU or low-memory GPU) |
| `score_threshold` | float | `0.05` | Minimum confidence to keep a detection |
| `iou_threshold` | float | `0.6` | IoU threshold for Non-Maximum Suppression |
| `pretrained_weights` | string | `"coco"` | Checkpoint to use. Currently only `"coco"` is available |

## Dataset Sources

### Option 1: Pre-packaged Artifact (default)

Uses a small VOC-format dataset bundled as a Clarifai artifact. Good for quick smoke tests.

```bash
--dataset_source=artifact
```

No `dataset_id` or `dataset_version_id` needed.

### Option 2: Clarifai Platform Dataset

Exports a dataset from your Clarifai app, downloads images, and converts annotations to COCO format.

```bash
--dataset_source=clarifai \
--dataset_id=your_dataset_id \
--dataset_version_id=your_version_id \
--concepts='["class1","class2","class3"]'
```

The `concepts` list must match concept IDs in your dataset's bounding box annotations. Only images with at least one matching annotation are included.

## Understanding the Output

Results are saved to `/tmp/yolof_eval_work_dir/eval_results.json`:

```json
{
  "model": "YOLOF (ResNet-50-C5)",
  "pretrained_weights": "coco",
  "num_classes": 2,
  "classes": ["bird", "cat"],
  "num_images": 3,
  "num_detections": 300,
  "metrics": {
    "AP": 0.001,
    "AP50": 0.003,
    ...
  }
}
```

### Key Metrics

| Metric | What It Means | Good Range |
|--------|---------------|------------|
| **AP** | Primary metric. Mean AP across IoU thresholds 0.50–0.95 | 0.30–0.50+ |
| **AP50** | AP at IoU=0.50 (loose matching) | 0.50–0.70+ |
| **AP75** | AP at IoU=0.75 (strict matching) | 0.20–0.40+ |
| **AR100** | Recall with up to 100 detections per image | 0.40–0.60+ |
| **APsmall/medium/large** | AP broken down by object area | varies |

A value of `-1.0` means there were no ground truth objects in that size category, so the metric is undefined.

## Common Issues

### Low metrics with the pre-packaged artifact dataset

The default COCO-pretrained checkpoint has **80 classes**, but the eval config sets `num_classes` based on your `concepts` list. If you pass 2 concepts, the classification head (80 → 2 classes) won't load from the checkpoint, resulting in near-zero metrics. This is expected behavior for a smoke test. For meaningful metrics, use a model fine-tuned on your target classes.

### `CLARIFAI_PAT environment variable not set`

Export the token before running:

```bash
export CLARIFAI_PAT="your_pat_here"
```

### `mmcv` install fails with `ModuleNotFoundError: No module named 'pkg_resources'`

```bash
pip install "setuptools<82,>=69.0"
```

Then retry the `mim install` command.

### Out of memory on GPU

Reduce `batch_size`:

```bash
--batch_size=1
```

### Slow on CPU

This is expected. The model downloads a 177MB checkpoint and runs inference through a ResNet-50 backbone. For faster evaluation, use a CUDA-capable GPU.

## Running on the Clarifai Platform

The pipeline is designed to run as an Argo Workflow on Clarifai's compute infrastructure:

1. Fill in `config.yaml` with your `pipeline_id`, `user_id`, `app_id`, and pipeline step `version_id`
2. Build and push the Docker image using the provided `Dockerfile`
3. Register the pipeline step and deploy the workflow through the Clarifai platform
4. The workflow runs on a GPU node (4 CPU, 16 GiB RAM, 1× 16 GiB GPU) as defined in `detector-pipeline-eval-yolof-ps/config.yaml`

## Pipeline Steps (Internal)

| Step | Description |
|------|-------------|
| 1. Download checkpoint | Fetches pretrained YOLOF weights from Clarifai artifact store |
| 2. Obtain dataset | Downloads artifact ZIP or exports from Clarifai platform via gRPC |
| 3. Generate config | Creates MMDetection config with model architecture and eval settings |
| 4. Load model | Initializes YOLOF via `DetInferencer` on GPU/CPU |
| 5. Run inference | Batch processes images, collects detections in COCO format |
| 6. Compute metrics | Runs `pycocotools.COCOeval` against ground truth |
| 7. Save results | Writes `eval_results.json` with metrics+metadata |
