# YOLOF Evaluation Pipeline (`detector-pipeline-eval-yolof-quick-start`)

An evaluation pipeline for the YOLOF (You Only Look One-level Feature) object detector. It runs inference using a pretrained YOLOF model on a dataset and computes standard COCO detection metrics.

**This pipeline does not train a model.** It evaluates a pretrained checkpoint and reports metrics.

## Directory Structure

```
detector-pipeline-eval-yolof-quick-start/
├── config.yaml                          # Argo Workflow orchestration config
├── README.md
└── detector-pipeline-eval-yolof-quick-start-ps/     # Pipeline step
    ├── config.yaml                      # Compute requirements and input parameters
    ├── Dockerfile                       # Docker build (PyTorch + MMDetection)
    ├── requirements.txt                 # Python dependencies
    └── 1/
        ├── pipeline_step.py             # Entry point
        └── models/model/1/
            ├── model.py                 # Core evaluation logic (YOLOFEvaluator)
            └── dataset_helpers.py       # Dataset download and COCO format conversion
```

## How It Works

The pipeline is a single-step Argo Workflow. The step runs inside a Docker container with PyTorch 2.1.2, CUDA 11.8, and MMDetection 3.3.0.

### Entry Point

`pipeline_step.py` dynamically imports `model.py`, finds the `YOLOFEvaluator` class (by looking for a class with an `evaluate` method), builds an argparse parser from the method signature, and calls `evaluate()` with the parsed arguments.

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

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | `"clarifai"` | Clarifai user ID |
| `app_id` | string | `"train_pipelines"` | Clarifai application ID |
| `dataset_source` | string | `"artifact"` | `"artifact"` for pre-packaged dataset or `"clarifai"` for Clarifai dataset export |
| `dataset_id` | string | `""` | Clarifai dataset ID (required when `dataset_source=clarifai`) |
| `dataset_version_id` | string | `""` | Clarifai dataset version ID (required when `dataset_source=clarifai`) |
| `concepts` | string | `'["bird","cat"]'` | JSON array of concept names to evaluate |
| `image_size` | string | `"[512]"` | Input image size as JSON array (single value = min side) |
| `max_aspect_ratio` | float | `1.5` | Max aspect ratio when `keep_aspect_ratio=true` |
| `keep_aspect_ratio` | bool | `true` | Preserve original image aspect ratio during resize |
| `batch_size` | int | `4` | Number of images per inference batch |
| `score_threshold` | float | `0.05` | Minimum confidence score to keep a detection |
| `iou_threshold` | float | `0.6` | IoU threshold for Non-Maximum Suppression |
| `pretrained_weights` | string | `"coco"` | Pretrained weights source (only `"coco"` is supported) |

## Dataset Sources

### Pre-packaged Artifact (default)

Uses a small VOC-format dataset bundled as a Clarifai artifact. Good for quick smoke tests. No `dataset_id` or `dataset_version_id` needed.

### Clarifai Platform Dataset

Exports a dataset from your Clarifai app, downloads images, and converts annotations to COCO format. Requires `dataset_source=clarifai`, `dataset_id`, and `dataset_version_id`. The `concepts` list must match concept IDs in your dataset's bounding box annotations.

## Usage

### Prerequisites

- A [Clarifai Personal Access Token (PAT)](https://clarifai.com/settings/security)
- `CLARIFAI_PAT` environment variable set:
  ```bash
  export CLARIFAI_PAT="your_personal_access_token"
  ```

### Running on the Clarifai Platform

1. Edit `config.yaml` and replace the pipeline placeholder values:
   ```yaml
   pipeline:
     id: "<YOUR_PIPELINE_ID>"
     user_id: "<YOUR_USER_ID>"
     app_id: "<YOUR_APP_ID>"
   ```

2. To evaluate on your own Clarifai dataset instead of the default artifact, set:
   ```yaml
   - name: dataset_source
     value: "clarifai"
   - name: dataset_id
     value: "<YOUR_DATASET_ID>"
   - name: dataset_version_id
     value: "<YOUR_DATASET_VERSION_ID>"
   ```
   To use the pre-packaged VOC artifact dataset (default), keep `dataset_source` as `"artifact"`.

3. Build and push the Docker image using the provided `Dockerfile`, then register the pipeline step and deploy through the Clarifai platform.

### Running Locally

All defaults are pre-configured to use public artifacts, so you only need a valid PAT:

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.3.0"
pip install clarifai==12.4.0 "clarifai-protocol>=0.0.35,<0.1.0" clarifai-grpc==12.3.1 \
  boto3==1.42.4 scipy pycocotools Pillow yapf

# Set your PAT and run (no arguments needed)
export CLARIFAI_PAT="your_personal_access_token"
cd detector-pipeline-eval-yolof-quick-start/detector-pipeline-eval-yolof-quick-start-ps/1
python3 pipeline_step.py
```

To evaluate on your own Clarifai dataset:

```bash
python3 pipeline_step.py \
  --user_id=your_user_id \
  --app_id=your_app_id \
  --dataset_source=clarifai \
  --dataset_id=your_dataset_id \
  --dataset_version_id=your_version_id \
  --concepts='["class1","class2"]'
```

Results are saved to `/tmp/yolof_eval_work_dir/eval_results.json`.

### Compute Requirements

- **CPU**: 4 cores, 16 GiB memory
- **GPU**: 1x NVIDIA GPU with 16 GiB memory

## Output

The pipeline writes `eval_results.json` with COCO evaluation metrics:

| Metric | Description |
|--------|-------------|
| `AP` | Average Precision @ IoU=0.50:0.95 (primary metric) |
| `AP50` | Average Precision @ IoU=0.50 |
| `AP75` | Average Precision @ IoU=0.75 |
| `APsmall` / `APmedium` / `APlarge` | AP by object size |
| `AR1` / `AR10` / `AR100` | Average Recall at 1, 10, 100 detections per image |
| `ARsmall` / `ARmedium` / `ARlarge` | AR by object size |

A value of `-1.0` means there were no ground truth objects in that size category.

## Troubleshooting

- **Low metrics with artifact dataset**: The COCO-pretrained checkpoint has 80 classes, but the eval config uses `num_classes` from your `concepts` list. Mismatched head dimensions cause near-zero metrics. This is expected for smoke tests.
- **`mmcv` install fails on macOS**: Run `pip install "setuptools<82,>=69.0"` first.
- **Out of memory**: Reduce `batch_size` (e.g., `--batch_size=1`).

## Key Implementation Details

- **No training**: The `YOLOFEvaluator` class is standalone and does not extend Clarifai's `VisualDetectorClass`. It only runs evaluation.
- **Config generation**: The MMDetection config is generated entirely in Python (via `_get_eval_config()`), not loaded from a file.
- **Dataset helpers**: `dataset_helpers.py` handles Clarifai dataset export downloads via gRPC, multi-threaded image downloading, concept filtering, and COCO format conversion.
- **Checkpoint source**: The YOLOF checkpoint is downloaded from Clarifai's artifact store (`mmdetectionyolof-coco`).
