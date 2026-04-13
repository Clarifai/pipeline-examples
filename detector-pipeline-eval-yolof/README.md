# YOLOF Evaluation Pipeline (`detector-pipeline-eval-yolof`)

An evaluation pipeline for the YOLOF (You Only Look One-level Feature) object detector. It runs inference using a pretrained YOLOF model on a dataset and computes standard COCO detection metrics.

**This pipeline does not train a model.** It evaluates a pretrained checkpoint and reports metrics.

## Directory Structure

```
detector-pipeline-eval-yolof/
├── config.yaml                          # Argo Workflow orchestration config
├── template.yaml                        # Clarifai UI template with parameter definitions
├── README.md
└── detector-pipeline-eval-yolof-ps/     # Pipeline step
    ├── config.yaml                      # Compute requirements and input parameters
    ├── Dockerfile                       # Docker build (PyTorch + MMDetection)
    ├── requirements.txt                 # Python dependencies
    └── 1/
        ├── pipeline_step.py             # Entry point
        └── models/model/
            ├── config.yaml              # Model metadata (placeholder)
            └── 1/
                ├── eval_model.py        # Core evaluation logic (YOLOFEvaluator)
                └── dataset_helpers.py   # Dataset download and COCO format conversion
```

## How It Works

The pipeline is a single-step Argo Workflow. The step runs inside a Docker container with PyTorch 2.1.2, CUDA 11.8, and MMDetection 3.3.0.

### Entry Point

`pipeline_step.py` dynamically imports `eval_model.py`, finds the `YOLOFEvaluator` class (by looking for a class with an `evaluate` method), builds an argparse parser from the method signature, and calls `evaluate()` with the parsed arguments.

### Evaluation Steps (inside `YOLOFEvaluator.evaluate()`)

| Step | Description |
|------|-------------|
| **1. Download checkpoint** | Downloads a pretrained YOLOF ResNet-50-C5 checkpoint (trained on COCO) from Clarifai's artifact store. |
| **2. Obtain dataset** | Either downloads a pre-packaged VOC dataset artifact (`dataset_source=artifact`) or exports a dataset from the Clarifai platform via gRPC (`dataset_source=clarifai`). Clarifai datasets are downloaded as protobuf exports, filtered by target concepts, and converted to COCO format. |
| **3. Generate MMDet config** | Programmatically creates a self-contained MMDetection config file for YOLOF with the correct number of classes, image scale, score/IoU thresholds, and checkpoint path. All backbone stages are frozen. |
| **4. Load model** | Loads the YOLOF model using MMDetection's `DetInferencer` on GPU (falls back to CPU). |
| **5. Run inference** | Processes images in batches through the model, collecting bounding box predictions in COCO format (`[x, y, width, height]`). |
| **6. Compute metrics** | Uses `pycocotools.COCOeval` to compute the standard 12 COCO metrics against ground truth annotations. |
| **7. Save results** | Writes a JSON file with all metrics, model info, and dataset details to the working directory. |

### Reported Metrics

The pipeline outputs these COCO evaluation metrics:

| Metric | Description |
|--------|-------------|
| `AP` | Average Precision @ IoU=0.50:0.95 |
| `AP50` | Average Precision @ IoU=0.50 |
| `AP75` | Average Precision @ IoU=0.75 |
| `APsmall` | AP for small objects (area < 32²) |
| `APmedium` | AP for medium objects (32² < area < 96²) |
| `APlarge` | AP for large objects (area > 96²) |
| `AR1` | Average Recall with 1 detection per image |
| `AR10` | Average Recall with 10 detections per image |
| `AR100` | Average Recall with 100 detections per image |
| `ARsmall` | AR for small objects |
| `ARmedium` | AR for medium objects |
| `ARlarge` | AR for large objects |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | — | Clarifai user ID |
| `app_id` | string | — | Clarifai application ID |
| `dataset_source` | string | `"artifact"` | `"artifact"` for pre-packaged dataset or `"clarifai"` for Clarifai dataset export |
| `dataset_id` | string | `""` | Clarifai dataset ID (required when `dataset_source=clarifai`) |
| `dataset_version_id` | string | `""` | Clarifai dataset version ID (required when `dataset_source=clarifai`) |
| `concepts` | string | `'["bird","cat"]'` | JSON array of concept names to evaluate |
| `image_size` | string | `"[512]"` | Input image size as JSON array (single value = min side) |
| `max_aspect_ratio` | float | `1.5` | Max aspect ratio when `keep_aspect_ratio=true` |
| `keep_aspect_ratio` | bool | `true` | Preserve original image aspect ratio during resize |
| `batch_size` | int | `8` | Number of images per inference batch |
| `score_threshold` | float | `0.05` | Minimum confidence score to keep a detection |
| `iou_threshold` | float | `0.6` | IoU threshold for Non-Maximum Suppression |
| `pretrained_weights` | string | `"coco"` | Pretrained weights source (only `"coco"` is supported) |

## Usage

### 1. Configure the pipeline

Edit `config.yaml` and replace the placeholder values:

```yaml
pipeline:
  id: "<YOUR_PIPELINE_ID>"
  user_id: "<YOUR_USER_ID>"
  app_id: "<YOUR_APP_ID>"
```

To evaluate on your own Clarifai dataset, set:

```yaml
- name: dataset_source
  value: "clarifai"
- name: dataset_id
  value: "<YOUR_DATASET_ID>"
- name: dataset_version_id
  value: "<YOUR_DATASET_VERSION_ID>"
```

To use the pre-packaged VOC artifact dataset (default), keep `dataset_source` as `"artifact"`.

### 2. Configure the pipeline step

Edit `detector-pipeline-eval-yolof-ps/config.yaml` and set your `user_id` and `app_id`.

### 3. Environment

Set the `CLARIFAI_PAT` environment variable with your Clarifai Personal Access Token. The pipeline reads this at runtime to authenticate API calls.

### 4. Compute Requirements

The pipeline step requires:
- **CPU**: 4 cores, 16 GiB memory
- **GPU**: 1x NVIDIA GPU with 16 GiB memory

### 5. Docker Build

The Dockerfile is based on `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` and installs:
- MMDetection 3.3.0 (via OpenMIM) with mmcv 2.1.0 and mmengine 0.10.7
- pycocotools for COCO metric computation
- Clarifai SDK 12.1.4 for artifact downloads and dataset exports

## Key Implementation Details

- **No training**: The `YOLOFEvaluator` class is standalone and does not extend Clarifai's `VisualDetectorClass`. It only runs evaluation and has no `predict()`, `load_model()`, or model upload logic.
- **Config generation**: The MMDetection config is generated entirely in Python (via `_get_eval_config()`), not loaded from a file. This avoids dependency on MMDetection's config inheritance system.
- **Dataset helpers**: `dataset_helpers.py` handles Clarifai dataset export downloads via gRPC, multi-threaded image downloading (20 workers), concept filtering by bounding box annotations, and conversion to COCO annotation format.
- **Checkpoint source**: The YOLOF checkpoint is downloaded from Clarifai's artifact store (`mmdetectionyolof-coco`), not from a URL or local path.
