import logging
import os
import json
import inspect
import zipfile
import torch
from pathlib import Path
from mmdet.apis import DetInferencer
from clarifai.client.artifact_version import ArtifactVersion

try:
    from .dataset_helpers import (
        download_dataset,
        convert_dataset_to_coco_format,
        create_classes_file,
        _safe_extract,
    )
except ImportError:
    import sys
    model_dir = Path(__file__).parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from dataset_helpers import (
        download_dataset,
        convert_dataset_to_coco_format,
        create_classes_file,
        _safe_extract,
    )

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class YOLOFEvaluator:

    @staticmethod
    def _get_argparse_type(param_annotation):
        if param_annotation == int:
            return int
        elif param_annotation == float:
            return float
        elif param_annotation == str:
            return str
        elif param_annotation == bool:
            return lambda x: str(x).lower() == 'true'
        else:
            return str

    @classmethod
    def to_pipeline_parser(cls):
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate a YOLOF detector model")

        sig = inspect.signature(cls.evaluate)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            arg_type = cls._get_argparse_type(param.annotation)

            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)

        return parser

    def evaluate(
        self,
        user_id: str = "YOUR_USER_ID",
        app_id: str = "YOUR_APP_ID",
        dataset_source: str = "artifact",
        dataset_id: str = "",
        dataset_version_id: str = "",
        concepts: str = '["bird","cat"]',
        image_size: str = "[512]",
        max_aspect_ratio: float = 1.5,
        keep_aspect_ratio: bool = True,
        batch_size: int = 8,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.6,
        pretrained_weights: str = "coco",
    ) -> str:
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        concepts_list = json.loads(concepts)
        image_size_list = json.loads(image_size) if isinstance(image_size, str) else image_size

        work_dir = "/tmp/yolof_eval_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        logging.info("Starting YOLOF Evaluation Pipeline")

        # ================================================================
        # STEP 1: Download pretrained checkpoint
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 1: Downloading Pretrained YOLOF Checkpoint")
        logging.info("=" * 80)

        pretrained_weights_artifacts = {
            'coco': {
                'artifact_id': 'mmdetectionyolof-coco',
                'user_id': 'clarifai',
                'app_id': 'train_pipelines',
                'version_id': 'efbbbe7f8c7743de9db5e85bba43af2a',
                'filename': 'yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
            }
        }

        artifact_info = pretrained_weights_artifacts.get(pretrained_weights)
        if artifact_info is None:
            raise ValueError(f"Unknown pretrained_weights: {pretrained_weights}. Available: {list(pretrained_weights_artifacts.keys())}")

        checkpoint_dir = "/tmp/pretrain_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, artifact_info['filename'])
        version = ArtifactVersion()
        checkpoint_root = version.download(
            artifact_id=artifact_info['artifact_id'],
            user_id=artifact_info['user_id'],
            app_id=artifact_info['app_id'],
            version_id=artifact_info['version_id'],
            output_path=checkpoint_path,
            force=True,
        )
        logging.info(f"Downloaded checkpoint to {checkpoint_root}")

        # ================================================================
        # STEP 2: Download evaluation dataset
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 2: Obtaining Evaluation Dataset")
        logging.info("=" * 80)

        if dataset_source == "clarifai":
            if not dataset_id:
                raise ValueError("dataset_id is required when dataset_source='clarifai'")
            if not dataset_version_id:
                raise ValueError("dataset_version_id is required when dataset_source='clarifai'")

            dataset_name = download_dataset(
                user_id=user_id,
                app_id=app_id,
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                pat=pat,
                output_dir=work_dir,
                concepts=concepts_list,
            )
            logging.info(f"Dataset name: {dataset_name}")

            convert_output = convert_dataset_to_coco_format(
                dataset_name=dataset_name,
                dataset_split="eval",
                output_root=work_dir,
            )
            images_dir = convert_output.images_output_root
            annotations_path = convert_output.annotations_path

            classes_path = create_classes_file(
                dataset_name=dataset_name,
                output_dir=images_dir,
                concepts=None,
            )

            if classes_path and os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    dataset_classes = [line.strip() for line in f if line.strip()]
                concepts_list = dataset_classes
                logging.info(f"Using {len(dataset_classes)} classes from dataset: {dataset_classes}")

        elif dataset_source == "artifact":
            dataset_artifact = {
                'artifact_id': 'mmdetectionyolof-voc-dataset',
                'user_id': 'clarifai',
                'app_id': 'train_pipelines',
                'version_id': '08c64f4529e3485baf0016aaca046b86',
            }
            dataset_zip_path = os.path.join(work_dir, "dataset.zip")
            ArtifactVersion().download(
                artifact_id=dataset_artifact['artifact_id'],
                user_id=dataset_artifact['user_id'],
                app_id=dataset_artifact['app_id'],
                version_id=dataset_artifact['version_id'],
                output_path=dataset_zip_path,
                force=True,
            )
            logging.info(f"Downloaded dataset to: {dataset_zip_path}")

            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                _safe_extract(zip_ref, work_dir)

            images_dir = os.path.join(work_dir, "train")
            annotations_path = os.path.join(images_dir, "annotations.json")
            classes_path = os.path.join(images_dir, "classes.txt")

            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    dataset_classes = [line.strip() for line in f if line.strip()]
                concepts_list = dataset_classes
                logging.info(f"Using {len(dataset_classes)} classes from dataset: {dataset_classes}")
        else:
            raise ValueError(f"Unknown dataset_source: {dataset_source}. Must be 'artifact' or 'clarifai'")

        num_classes = len(concepts_list)
        logging.info(f"Images directory: {images_dir}")
        logging.info(f"Annotations file: {annotations_path}")
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Classes: {concepts_list}")

        # ================================================================
        # STEP 3: Generate MMDetection config for inference
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 3: Generating MMDetection Config")
        logging.info("=" * 80)

        if keep_aspect_ratio:
            if not (len(image_size_list) == 1 or (len(image_size_list) == 2 and image_size_list[0] == image_size_list[1])):
                raise ValueError('image_size must be single element with min side length when keep_aspect_ratio=True')
            min_side = min(image_size_list)
            img_scale = (int(max_aspect_ratio * min_side), min_side)
        else:
            if len(image_size_list) != 2 or not all(isinstance(v, (int, float)) for v in image_size_list):
                raise ValueError(
                    "image_size must be a list of exactly 2 numeric values [height, width] "
                    "when keep_aspect_ratio=False"
                )
            img_scale = tuple(image_size_list)

        config_path = os.path.join(work_dir, 'eval_config.py')
        config_content = self._get_eval_config(
            num_classes=num_classes,
            img_scale=img_scale,
            keep_aspect_ratio=keep_aspect_ratio,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            checkpoint_path=checkpoint_root,
        )
        with open(config_path, 'w') as f:
            f.write(config_content)
        logging.info(f"Config generated at {config_path}")

        # ================================================================
        # STEP 4: Load model
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 4: Loading YOLOF Model")
        logging.info("=" * 80)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model on {device}")

        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            inferencer = DetInferencer(
                model=config_path, weights=checkpoint_root, device=device
            )
        finally:
            torch.load = original_load

        logging.info("Model loaded successfully")

        # ================================================================
        # STEP 5: Run inference on evaluation dataset
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 5: Running Inference on Evaluation Data")
        logging.info("=" * 80)

        with open(annotations_path, 'r') as f:
            coco_gt_data = json.load(f)

        images_subdir = os.path.join(images_dir, "images")
        if not os.path.isdir(images_subdir):
            images_subdir = images_dir

        image_id_map = {}
        image_paths = []
        for img_info in coco_gt_data["images"]:
            img_path = os.path.join(images_subdir, img_info["file_name"])
            if os.path.exists(img_path):
                image_paths.append((img_info["id"], img_path, img_info["width"], img_info["height"]))
                image_id_map[img_path] = img_info["id"]

        logging.info(f"Found {len(image_paths)} images for evaluation")

        coco_predictions = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_start:batch_start + batch_size]
            batch_paths = [item[1] for item in batch]

            results = inferencer(batch_paths, return_vis=False)

            for idx, (image_id, img_path, img_w, img_h) in enumerate(batch):
                if results and 'predictions' in results and idx < len(results['predictions']):
                    pred = results['predictions'][idx]

                    bboxes = pred.get('bboxes', [])
                    labels = pred.get('labels', [])
                    scores = pred.get('scores', [])

                    for bbox, label, score in zip(bboxes, labels, scores):
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        coco_predictions.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score),
                        })

            processed = min(batch_start + batch_size, len(image_paths))
            logging.info(f"Processed {processed}/{len(image_paths)} images, {len(coco_predictions)} detections so far")

        logging.info(f"Total detections: {len(coco_predictions)}")

        # ================================================================
        # STEP 6: Compute COCO metrics
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 6: Computing COCO Evaluation Metrics")
        logging.info("=" * 80)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(annotations_path)

        if not coco_predictions:
            logging.warning("No predictions generated! All metrics will be 0.")
            metrics = {
                "AP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "APsmall": 0.0,
                "APmedium": 0.0,
                "APlarge": 0.0,
                "AR1": 0.0,
                "AR10": 0.0,
                "AR100": 0.0,
                "ARsmall": 0.0,
                "ARmedium": 0.0,
                "ARlarge": 0.0,
            }
        else:
            predictions_path = os.path.join(work_dir, "predictions.json")
            with open(predictions_path, 'w') as f:
                json.dump(coco_predictions, f)

            coco_dt = coco_gt.loadRes(predictions_path)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            metrics = {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "APsmall": float(stats[3]),
                "APmedium": float(stats[4]),
                "APlarge": float(stats[5]),
                "AR1": float(stats[6]),
                "AR10": float(stats[7]),
                "AR100": float(stats[8]),
                "ARsmall": float(stats[9]),
                "ARmedium": float(stats[10]),
                "ARlarge": float(stats[11]),
            }

        # ================================================================
        # STEP 7: Log and save results
        # ================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("EVALUATION RESULTS")
        logging.info("=" * 80)
        logging.info(f"  AP   (IoU=0.50:0.95) = {metrics['AP']:.4f}")
        logging.info(f"  AP50 (IoU=0.50)       = {metrics['AP50']:.4f}")
        logging.info(f"  AP75 (IoU=0.75)       = {metrics['AP75']:.4f}")
        logging.info(f"  AP   (small)          = {metrics['APsmall']:.4f}")
        logging.info(f"  AP   (medium)         = {metrics['APmedium']:.4f}")
        logging.info(f"  AP   (large)          = {metrics['APlarge']:.4f}")
        logging.info(f"  AR1                   = {metrics['AR1']:.4f}")
        logging.info(f"  AR10                  = {metrics['AR10']:.4f}")
        logging.info(f"  AR100                 = {metrics['AR100']:.4f}")
        logging.info(f"  AR   (small)          = {metrics['ARsmall']:.4f}")
        logging.info(f"  AR   (medium)         = {metrics['ARmedium']:.4f}")
        logging.info(f"  AR   (large)          = {metrics['ARlarge']:.4f}")

        results = {
            "model": "YOLOF (ResNet-50-C5)",
            "pretrained_weights": pretrained_weights,
            "dataset_source": dataset_source,
            "num_classes": num_classes,
            "classes": concepts_list,
            "num_images": len(image_paths),
            "num_detections": len(coco_predictions),
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "image_size": image_size_list,
            "metrics": metrics,
        }

        results_path = os.path.join(work_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to: {results_path}")

        logging.info("")
        logging.info("=" * 80)
        logging.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)

        return results_path

    def _get_eval_config(self, num_classes, img_scale, keep_aspect_ratio,
                         score_threshold, iou_threshold, checkpoint_path):
        """Generate self-contained YOLOF config for evaluation inference."""
        config = f"""# Self-contained YOLOF config for MMDetection evaluation
# Based on: configs/yolof/yolof_r50-c5_8xb8-1x_coco.py

default_scope = 'mmdet'

# Model architecture
model = dict(
    type='YOLOF',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=None),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes={num_classes},
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr={score_threshold},
        nms=dict(type='nms', iou_threshold={iou_threshold}),
        max_per_img=100))

# Required by mmdet but not used for eval-only
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

param_scheduler = []

train_cfg = None
val_cfg = None
test_cfg = dict()

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = '{checkpoint_path}'
resume = False

# Launcher
launcher = 'none'

# Data pipeline
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale={tuple(img_scale)}, keep_ratio={keep_aspect_ratio}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataset
dataset_type = 'CocoDataset'

train_dataloader = None

val_dataloader = None
val_evaluator = None

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='',
        data_prefix=dict(img=''),
        metainfo=dict(classes=()),
        pipeline=test_pipeline,
        backend_args=backend_args))

test_evaluator = dict(type='CocoMetric', metric='bbox')
"""
        return config
