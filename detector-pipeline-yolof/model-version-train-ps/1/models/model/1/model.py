import logging
import os
import math
import subprocess
import tempfile
import yaml
import torch
import inspect
from io import BytesIO
from PIL import Image as PILImage
from time import perf_counter_ns
from typing import List
from pathlib import Path
import json
import shutil
from mmdet.apis import DetInferencer
from clarifai.runners.models.visual_detector_class import VisualDetectorClass
from clarifai.runners.utils.data_types import Image, Region, Concept
from clarifai.client.artifact_version import ArtifactVersion

try:
    from .dataset_helpers import (
        download_dataset,
        convert_dataset_to_coco_format,
        create_classes_file,
    )
    from .benchmark_model_helper import benchmark_and_update_config
    from .model_export_helper import export_and_upload_detector
except ImportError:
    import sys
    from pathlib import Path
    model_dir = Path(__file__).parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from dataset_helpers import (
        download_dataset,
        convert_dataset_to_coco_format,
        create_classes_file,
    )
    from benchmark_model_helper import benchmark_and_update_config
    from model_export_helper import export_and_upload_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MMDetectionYoloF(VisualDetectorClass):

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
        parser = argparse.ArgumentParser(description="Train a Clarifai model")

        sig = inspect.signature(cls.train)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            arg_type = cls._get_argparse_type(param.annotation)

            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)

        return parser

    def to_pipeline(self, pipeline_folder_path):
        step_name = "model-version-train-ps"
        model_dir = Path(__file__).parent.parent
        pipeline_path = Path(pipeline_folder_path)
        step_path = pipeline_path / step_name

        pipeline_path.mkdir(parents=True, exist_ok=True)
        (step_path / "1").mkdir(parents=True, exist_ok=True)

        sig = inspect.signature(self.train)
        params_info = {}
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                if param.default != inspect.Parameter.empty:
                    params_info[param_name] = param.default
                else:
                    params_info[param_name] = ""

        with open(model_dir / "1" / "pipe_config.yaml") as f:
            parent_config = f.read()

        param_lines = []
        step_param_lines = []

        for param_name, default_val in params_info.items():
            if default_val == "":
                val_str = '""'
            elif isinstance(default_val, str):
                val_str = f'"{default_val}"' if not default_val.startswith('[') else f"'{default_val}'"
            elif isinstance(default_val, bool):
                val_str = str(default_val).lower()
            else:
                val_str = str(default_val)

            param_lines.append(f"            - name: {param_name}\n              value: {val_str}")
            step_param_lines.append(f'                  - name: {param_name}\n                    value: "{{{{workflow.parameters.{param_name}}}}}"')

        parent_config += "\n".join(param_lines)
        parent_config += "\n        templates:\n        - name: sequence\n          steps:\n"
        parent_config += f"          - - name: {step_name}-name\n"
        parent_config += f"              templateRef:\n"
        parent_config += f"                name: users/YOUR_USER_ID/apps/YOUR_APP_ID/pipeline_steps/{step_name}\n"
        parent_config += f"                template: users/YOUR_USER_ID/apps/YOUR_APP_ID/pipeline_steps/{step_name}\n"
        parent_config += "              arguments:\n                parameters:\n"
        parent_config += "\n".join(step_param_lines) + "\n"

        with open(pipeline_path / "config.yaml", "w") as f:
            f.write(parent_config)

        with open(model_dir / "1" / "pipe_step_config.yaml") as f:
            step_config = yaml.safe_load(f)

        step_config.pop('pipeline_step_input_params', None)

        with open(step_path / "config.yaml", "w") as f:
            yaml.dump(step_config, f, default_flow_style=False, sort_keys=False)

            f.write("\npipeline_step_input_params:\n")
            for param_name in params_info:
                f.write(f"  - name: {param_name}\n")
                f.write(f"    description: \"\"\n")

        shutil.copy(model_dir / "train_Dockerfile", step_path / "Dockerfile")
        shutil.copy(model_dir / "train_requirements.txt", step_path / "requirements.txt")
        shutil.copy(model_dir / "1" / "pipeline_step.py", step_path / "1" / "pipeline_step.py")

        model_copy_path = step_path / "1" / "models" / "model"
        shutil.copytree(model_dir, model_copy_path)

        if (model_copy_path / "Dockerfile").exists():
            (model_copy_path / "Dockerfile").rename(model_copy_path / "Dockerfil")
        if (model_copy_path / "requirements.txt").exists():
            (model_copy_path / "requirements.txt").rename(model_copy_path / "requiremen.txt")

        logger.info(f"Pipeline structure created at {pipeline_path}")

    def train(self,
              user_id: str = "YOUR_USER_ID",
              app_id: str = "YOUR_APP_ID",
              model_id: str = "test_detector",
              dataset_id: str = "YOUR_DATASET_ID",
              seed: int = -1,
              num_gpus: int = 1,
              image_size: str = "[512]",
              max_aspect_ratio: float = 1.5,
              keep_aspect_ratio: bool = True,
              batch_size: int = 16,
              num_epochs: int = 100,
              min_samples_per_epoch: int = 300,
              per_item_lrate: float = 0.001875,
              pretrained_weights: str = "coco",
              frozen_stages: int = 1,
              inference_max_batch_size: int = 2,
              is_cpu: int = 0,
              concepts: str = '["bird","cat"]'
              ) -> str:
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        concepts = json.loads(concepts)
        image_size_list = json.loads(image_size) if isinstance(image_size, str) else image_size

        work_dir = "/tmp/mmdetection_work_dir"

        logging.info("Starting MMDetection YOLOF training pipeline")

        pretrained_weights_artifacts = {
            'coco': {
                'artifact_id': 'mmdetectionyolof-coco',
                'user_id': 'clarifai',
                'app_id': 'train_pipelines',
                'version_id': '7abb95c6ca62423ba8fc0b211b0e08e9',
                'filename': 'yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
            }
        }

        artifact_info = pretrained_weights_artifacts[pretrained_weights]
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

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 1: Downloading Dataset from Clarifai API")
        logging.info("=" * 80)

        os.makedirs(work_dir, exist_ok=True)

        dataset_name = download_dataset(
            user_id=user_id,
            app_id=app_id,
            dataset_id=dataset_id,
            pat=pat,
            output_dir=work_dir,
            concepts=concepts,
        )

        logging.info(f"Dataset name: {dataset_name}")

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 2: Converting Dataset to COCO Format")
        logging.info("=" * 80)

        convert_output = convert_dataset_to_coco_format(
            dataset_name=dataset_name,
            dataset_split="train",
            output_root=work_dir,
        )

        images_output_root = convert_output.images_output_root
        annotations_path = convert_output.annotations_path

        logging.info(f"Images directory: {images_output_root}")
        logging.info(f"Annotations file: {annotations_path}")

        classes_path = create_classes_file(
            dataset_name=dataset_name,
            output_dir=images_output_root,
            concepts=None,
        )
        logging.info(f"Classes file: {classes_path}")

        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                dataset_classes = [line.strip() for line in f if line.strip()]
            num_classes = len(dataset_classes)
            concepts = dataset_classes
            logging.info(f"Using {len(dataset_classes)} classes from dataset: {dataset_classes}")

        self.seed = seed
        self.is_cpu = is_cpu
        self.num_gpus = 0 if self.is_cpu else num_gpus
        self.image_size = image_size_list
        self.max_aspect_ratio = max_aspect_ratio
        self.keep_aspect_ratio = keep_aspect_ratio
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.min_samples_per_epoch = min_samples_per_epoch
        self.per_item_lrate = per_item_lrate
        self.frozen_stages = frozen_stages
        self.pretrained_weights = pretrained_weights
        self.inference_max_batch_size = inference_max_batch_size
        self.load_from = checkpoint_root
        self.work_dir = work_dir
        self.data_dir = images_output_root
        self.num_classes = num_classes

        if self.keep_aspect_ratio:
            if not (len(self.image_size) == 1 or (len(self.image_size) == 2 and self.image_size[0] == self.image_size[1])):
                raise ValueError('image_size must be single element with min side length when keep_aspect_ratio=True')
            if self.max_aspect_ratio < 1.0:
                raise ValueError('max_aspect_ratio should be >= 1 (multiple of min side length)')
            min_side = min(self.image_size)
            self.img_scale = (int(self.max_aspect_ratio * min_side), min_side)
        else:
            self.img_scale = tuple(self.image_size)

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 3-6: Training Model")
        logging.info("=" * 80)

        os.makedirs(self.work_dir, exist_ok=True)

        logging.info("STEP 3: Generating self-contained config...")
        config_path = os.path.join(self.work_dir, 'config.py')
        with open(config_path, 'w') as f:
            f.write(self._get_config_contents())
        logging.info(f"Config generated at {config_path}")

        logging.info("STEP 4: Configuring config for dataset...")
        train_annotations_path = annotations_path
        train_images_path = images_output_root
        configured_config_path = os.path.join(self.work_dir, 'configured_config.py')

        self._configure(
            config_path,
            train_annotations_path,
            train_images_path,
            classes_path,
            configured_config_path
        )
        logging.info(f"Config configured at {configured_config_path}")

        logging.info("STEP 5: Training model using mmdetection/mmengine API...")
        from mmengine.config import Config
        from mmengine.runner import Runner

        cfg = Config.fromfile(configured_config_path)
        cfg.work_dir = self.work_dir

        if self.seed > 0:
            cfg.randomness = dict(seed=self.seed)
            logging.info(f"Set random seed to {self.seed}")

        logging.info("Building runner and starting training...")
        if self.is_cpu == 0:
            if torch.cuda.is_available():
                logging.info(f"✓ CUDA enabled: {torch.cuda.get_device_name(0)}")
            else:
                logging.warning("is_cpu=0 but CUDA not available, training on CPU")
        else:
            logging.info("Training on CPU (is_cpu=1)")

        runner = Runner.from_cfg(cfg)
        logging.info(next(runner.model.parameters()).device)
        runner.train()

        logging.info("Training completed")

        if self.is_cpu == 0:
            mem_before_mb = torch.cuda.memory_reserved(0) / 1024**2
            logging.info(f"GPU memory reserved before cleanup: {mem_before_mb:.2f} MB (matches nvidia-smi)")
            del runner
            torch.cuda.empty_cache()
            mem_after_mb = torch.cuda.memory_reserved(0) / 1024**2
            logging.info(f"GPU memory reserved after cleanup: {mem_after_mb:.2f} MB (matches nvidia-smi)")
            logging.info(f"GPU memory freed: {mem_before_mb - mem_after_mb:.2f} MB")

        latest_checkpoint = os.path.join(self.work_dir, f'epoch_{self.num_epochs}.pth')
        if os.path.exists(latest_checkpoint):
            self.weights_path = latest_checkpoint
            logging.info(f"Checkpoint found: {self.weights_path}")
        else:
            latest_checkpoint = os.path.join(self.work_dir, 'latest.pth')
            if os.path.exists(latest_checkpoint):
                self.weights_path = latest_checkpoint
                logging.info(f"Checkpoint found: {self.weights_path}")
            else:
                logging.warning("No checkpoint found at expected locations")
                self.weights_path = latest_checkpoint

        self.config_py_path = configured_config_path

        logging.info(f"Training completed. Checkpoint: {self.weights_path}")
        logging.info(f"Config: {self.config_py_path}")

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 6.5: Benchmarking Model for GPU Requirements")
        logging.info("=" * 80)

        model_template_dir = Path(__file__).parent.parent
        config_yaml_path = model_template_dir / "config.yaml"

        if config_yaml_path.exists():
            min_side = min(self.image_size)
            input_shape = (3, min_side, min_side)
            logging.info(f"Benchmarking with input shape: {input_shape}")

            benchmark_and_update_config(
                checkpoint_path=self.weights_path,
                config_py_path=self.config_py_path,
                config_yaml_path=str(config_yaml_path),
                input_shape=input_shape,
                batch_size=self.inference_max_batch_size,
                device_id=0,
                is_cpu=self.is_cpu,
                save_benchmark_json="benchmark.json",
            )
            logging.info("✅ Benchmark complete and config.yaml updated")
        else:
            logging.warning(f"config.yaml not found at {config_yaml_path}, skipping benchmark")

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 7: Exporting and Uploading Model to Clarifai")
        logging.info("=" * 80)

        clarifai_api_base = os.getenv("CLARIFAI_API_BASE", "https://api.clarifai.com")

        export_and_upload_detector(
            weights_path=self.weights_path,
            config_py_path=self.config_py_path,
            classes=concepts,
            source_model_dir=model_template_dir,
            clarifai_pat=pat,
            clarifai_api_base=clarifai_api_base,
            user_id=user_id,
            app_id=app_id,
            model_id=model_id,
        )

        logging.info("")
        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"Final checkpoint: {self.weights_path}")

        return self.weights_path

    @property
    def learning_rate(self):
        return self.batch_size * max(1, self.num_gpus) * self.per_item_lrate

    def _get_config_contents(self):
        """Self-contained YOLOF config based on mmdetection"""
        config = f"""# Self-contained YOLOF config for MMDetection
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
        frozen_stages={self.frozen_stages},
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
        num_classes={self.num_classes},
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
    train_cfg=dict(
        assigner=dict(type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr={self.learning_rate},
        momentum=0.9,
        weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        custom_keys=dict(backbone=dict(lr_mult=0.3333))),
    clip_grad=dict(max_norm=8, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00066667,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end={self.num_epochs},
        milestones=[{int(math.ceil(self.num_epochs / 2.0))}, {int(math.ceil(3 * self.num_epochs / 4.0))}],
        gamma=0.3333)
]

# Training config
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs={int(self.num_epochs)},
    val_interval=1)
val_cfg = None
test_cfg = dict()

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

custom_hooks = [dict(type='CheckInvalidLossHook')]

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
load_from = '{self.load_from}'
resume = False

# Launcher
launcher = 'none'

# Data pipeline
backend_args = None
min_samples_per_epoch = {int(self.min_samples_per_epoch)}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale={tuple(self.img_scale)}, keep_ratio={self.keep_aspect_ratio}),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomShift', prob=0.5, max_shift_px=32),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale={tuple(self.img_scale)}, keep_ratio={self.keep_aspect_ratio}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataset
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size={int(self.batch_size)},
    num_workers={0 if self.is_cpu else 2},
    persistent_workers={False if self.is_cpu else True},
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='',
        data_prefix=dict(img=''),
        metainfo=dict(classes=()),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


val_dataloader = None
val_evaluator = None

# Test dataloader (required for DetInferencer during inference/benchmark)
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

    def _configure(self, config_path, train_annotations_path, train_images_path, classes_path, output_path):
        """Configure the MMDetection config file with dataset paths"""
        from mmengine import Config

        cfg = Config.fromfile(config_path)
        logging.info(f"Loaded config from file: {config_path}")

        if train_annotations_path:
            cfg.train_dataloader.dataset.ann_file = train_annotations_path
            logging.info(f"Set ann_file to {train_annotations_path}")

        if train_images_path:
            cfg.train_dataloader.dataset.data_root = train_images_path
            cfg.train_dataloader.dataset.data_prefix.img = 'images'
            logging.info(f"Set data_root to {train_images_path}")

        if classes_path:
            with open(classes_path, 'r') as f:
                classes = tuple([line.strip() for line in f if line.strip()])
            cfg.train_dataloader.dataset.metainfo.classes = classes
            logging.info(f"Set classes to {classes}")

        cfg.dump(output_path)
        logging.info(f"Dumped updated config to file: {output_path}")

    def load_model(self):
        model_folder = os.path.dirname(os.path.dirname(__file__))
        checkpoint_path = os.path.join(
            model_folder, "1", "model_files", "checkpoint.pth"
        )
        config_path = os.path.join(model_folder, "1", "model_files", "config.py")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found at {config_path}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model with MMDetection inferencer on {device}")

        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            self.inferencer = DetInferencer(
                model=config_path, weights=checkpoint_path, device=device
            )
        finally:
            torch.load = original_load

        logger.info("Loaded MMDetection DetInferencer")

        with open(os.path.join(model_folder, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
            concepts = config.get("concepts", [])

        self.id2label = {i: c["name"] for i, c in enumerate(concepts)}
        self.num_classes = len(self.id2label)
        logger.info(
            f"Loaded {self.num_classes} classes: {list(self.id2label.values())}"
        )

    @VisualDetectorClass.method
    def predict(self, image: Image) -> List[Region]:
        pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_image.save(tmp, format="JPEG")
            tmp_path = tmp.name

        try:
            start = perf_counter_ns()
            results = self.inferencer(tmp_path, return_vis=False)
            inference_time_ms = (perf_counter_ns() - start) / 1_000_000
            logger.info(f"Inference took {inference_time_ms:.2f} ms")
        finally:
            os.unlink(tmp_path)

        regions = []
        if results and 'predictions' in results and len(results['predictions']) > 0:
            pred = results['predictions'][0]

            bboxes = pred.get('bboxes', [])
            labels = pred.get('labels', [])
            scores = pred.get('scores', [])

            for bbox, label, score in zip(bboxes, labels, scores):
                x1, y1, x2, y2 = bbox
                class_name = self.id2label.get(label, f"class_{label}")

                regions.append(
                    Region(
                        box=[
                            float(x1) / pil_image.width,
                            float(y1) / pil_image.height,
                            float(x2) / pil_image.width,
                            float(y2) / pil_image.height
                        ],
                        concepts=[
                            Concept(
                                id=str(label),
                                name=class_name,
                                value=float(score)
                            )
                        ]
                    )
                )

        return regions
