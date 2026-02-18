import logging
import os
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
from mmpretrain import ImageClassificationInferencer
from clarifai.runners.models.visual_classifier_class import VisualClassifierClass
from clarifai.runners.utils.data_types import Image, Concept
from clarifai.client.artifact_version import ArtifactVersion

try:
    from .dataset_helpers import (
        download_dataset,
        convert_dataset_to_imagenet_format,
        create_classes_file,
    )
    from .benchmark_model_helper import benchmark_and_update_config
    from .model_export_helper import export_and_upload_classifier
except ImportError:
    import sys
    from pathlib import Path
    model_dir = Path(__file__).parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from dataset_helpers import (
        download_dataset,
        convert_dataset_to_imagenet_format,
        create_classes_file,
    )
    from benchmark_model_helper import benchmark_and_update_config
    from model_export_helper import export_and_upload_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MMClassificationResNet50(VisualClassifierClass):

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
            return str  # Default to str for other types

    @classmethod
    def to_pipeline_parser(cls):
        import argparse
        parser = argparse.ArgumentParser(description="Train a Clarifai model")

        # Get train() method signature
        sig = inspect.signature(cls.train)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Get type from annotation
            arg_type = cls._get_argparse_type(param.annotation)

            # Add argument with or without default
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)

        return parser

    def train(self,
              # Resource IDs
              user_id: str = "YOUR_USER_ID",
              app_id: str = "YOUR_APP_ID",
              model_id: str = "test_model",
              dataset_id: str = "YOUR_DATASET_ID",
              concepts: str = '["beignets","hamburger","prime_rib","ramen"]',
              # Training hyperparameters with defaults
              num_epochs: int = 200,
              batch_size: int = 64,
              image_size: int = 224,
              per_item_lrate: float = 0.00001953125,
              weight_decay: float = 0.01,
              per_item_min_lrate: float = 1.5625e-8,
              warmup_iters: int = 5,
              warmup_ratio: float = 0.0001,
              flip_probability: float = 0.5,
              flip_direction: str = "horizontal",
              concepts_mutually_exclusive: bool = False,
              pretrained_weights: str = "ImageNet-1k",
              seed: int = -1,
              ) -> str:
        # Get PAT from environment
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        # Convert concepts string to list
        concepts = json.loads(concepts)

        work_dir = "/tmp/mmpretrain_work_dir"

        logging.info("Starting MMClassification ResNet-50 training pipeline")

        # Hardcode is_cpu and num_gpus
        is_cpu = 0
        num_gpus = 1

        # Map pretrained_weights to checkpoint paths (similar to EfficientNet pattern)
        pretrained_weights_artifacts = {
            '': None,  # No pretrained weights
            'ImageNet-1k': {
                'artifact_id': 'mmclassificationresnet50-imagenet-1k',
                'user_id': 'clarifai',
                'app_id': 'train_pipelines',
                'version_id': '91b08c5cd505452a80c5a8c54c59e4c2',
                'filename': 'resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
            }
        }

        artifact_info = pretrained_weights_artifacts.get(pretrained_weights)
        if artifact_info is not None:
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
        else:
            checkpoint_root = ''
            logging.info("Training from scratch (no pretrained weights)")

        # STEP 1: Download Dataset from Clarifai API
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

        # STEP 2: Convert Dataset to ImageNet Format
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 2: Converting Dataset to ImageNet Format")
        logging.info("=" * 80)

        convert_output = convert_dataset_to_imagenet_format(
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

        with open(classes_path, 'r') as f:
            dataset_classes = [line.strip() for line in f if line.strip()]
        num_classes = len(dataset_classes)
        concepts = dataset_classes
        logging.info(f"Using {len(dataset_classes)} classes from dataset: {dataset_classes}")

        self.seed = seed
        self.is_cpu = is_cpu  # Hardcoded to 0
        self.num_gpus = num_gpus  # Hardcoded to 1
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.per_item_lrate = per_item_lrate
        self.weight_decay = weight_decay
        self.per_item_min_lrate = per_item_min_lrate
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.pretrained_weights = pretrained_weights
        self.flip_probability = flip_probability
        self.flip_direction = flip_direction
        self.concepts_mutually_exclusive = concepts_mutually_exclusive
        self.checkpoint_root = checkpoint_root
        self.load_from = self.checkpoint_root
        self.work_dir = work_dir
        self.data_dir = images_output_root
        self.num_classes = num_classes

        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 3-6: Training Model")
        logging.info("=" * 80)

        os.makedirs(self.work_dir, exist_ok=True)

        # STEP 3: Generate Config
        logging.info("STEP 3: Generating self-contained config...")
        config_path = os.path.join(self.work_dir, 'config.py')
        with open(config_path, 'w') as f:
            f.write(self._get_config_contents())
        logging.info(f"Config generated at {config_path}")

        # STEP 4: Configure for Dataset
        logging.info("STEP 4: Configuring config for dataset...")
        train_annotations_path = os.path.join(self.data_dir, 'train_annotations.txt')
        train_images_path = os.path.join(self.data_dir, 'train')
        classes_path = os.path.join(self.data_dir, 'classes.txt')
        configured_config_path = os.path.join(self.work_dir, 'configured_config.py')

        self._configure(
            config_path,
            train_annotations_path,
            train_images_path,
            classes_path,
            configured_config_path
        )
        logging.info(f"Config configured at {configured_config_path}")

        # STEP 5: Train Model (using mmpretrain with mmengine Runner)
        logging.info("STEP 5: Training model using mmpretrain/mmengine API...")
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

        # Clear GPU memory
        if self.is_cpu == 0:
            mem_before_mb = torch.cuda.memory_reserved(0) / 1024**2
            logging.info(f"GPU memory reserved before cleanup: {mem_before_mb:.2f} MB (matches nvidia-smi)")
            del runner
            torch.cuda.empty_cache()
            mem_after_mb = torch.cuda.memory_reserved(0) / 1024**2
            logging.info(f"GPU memory reserved after cleanup: {mem_after_mb:.2f} MB (matches nvidia-smi)")
            logging.info(f"GPU memory freed: {mem_before_mb - mem_after_mb:.2f} MB")

        # STEP 6: Locate trained checkpoint
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

        # STEP 6.5: Benchmark Model and Update Config
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 6.5: Benchmarking Model for GPU Requirements")
        logging.info("=" * 80)

        model_template_dir = Path(__file__).parent.parent
        config_yaml_path = model_template_dir / "config.yaml"

        if config_yaml_path.exists():
            input_shape = (3, self.image_size, self.image_size)
            logging.info(f"Benchmarking with input shape: {input_shape}")

            benchmark_and_update_config(
                checkpoint_path=self.weights_path,
                config_py_path=self.config_py_path,
                config_yaml_path=str(config_yaml_path),
                input_shape=input_shape,
                batch_size=16,
                device_id=0,
                is_cpu=self.is_cpu,
                save_benchmark_json="benchmark.json",
            )
            logging.info("✅ Benchmark complete and config.yaml updated")
        else:
            logging.warning(f"config.yaml not found at {config_yaml_path}, skipping benchmark")

        # STEP 7: Export and Upload Model to Clarifai
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 7: Exporting and Uploading Model to Clarifai")
        logging.info("=" * 80)

        clarifai_api_base = os.getenv("CLARIFAI_API_BASE", "https://api.clarifai.com")

        export_and_upload_classifier(
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

    @property
    def min_learning_rate(self):
        return self.batch_size * max(1, self.num_gpus) * self.per_item_min_lrate

    def _get_config_contents(self):
        config = f"""# Minimal ResNet-50 config for MMPretrain
# Based on: configs/resnet/resnet50_8xb256-rsb-a1-600e_in1k.py


default_scope = 'mmpretrain'

# Data preprocessor
data_preprocessor = dict(
    num_classes={self.num_classes},
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Model (only override num_classes and loss)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(3,), style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes={self.num_classes},
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, min(5, {self.num_classes}))),
    data_preprocessor=data_preprocessor)

# Pretrained weights
load_from = '{self.load_from}'

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='Lamb', lr={self.learning_rate}, weight_decay={self.weight_decay}))

# Learning rate schedule
param_scheduler = [
    dict(type='LinearLR', start_factor={self.warmup_ratio}, by_epoch=True,
         begin=0, end={self.warmup_iters}, convert_to_iter_based=True),
    # Main schedule: cosine annealing after warmup completes
    dict(type='CosineAnnealingLR', T_max={int(self.num_epochs)-int(self.warmup_iters)}, eta_min={self.min_learning_rate},
         by_epoch=True, begin={self.warmup_iters}, end={int(self.num_epochs)})
]

# Training config
train_cfg = dict(by_epoch=True, max_epochs={int(self.num_epochs)}, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Launcher
launcher = 'none'

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale={self.image_size}),
    dict(type='RandomFlip', prob={self.flip_probability}, direction='{self.flip_direction}'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
            dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
            dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
            dict(type='SolarizeAdd', magnitude_key='magnitude', magnitude_range=(0, 110)),
            dict(type='ColorTransform', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
            dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
            dict(type='Brightness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
            dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
            dict(type='Shear', magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='horizontal'),
            dict(type='Shear', magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='vertical'),
            dict(type='Translate', magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='horizontal'),
            dict(type='Translate', magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(type='PackInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale={int(self.image_size * 256/224)}, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size={self.image_size}),
    dict(type='PackInputs')
]

# Dataset config (new dataloader format for mmengine)
dataset_type = 'ImageNet' if {self.concepts_mutually_exclusive} else 'CustomDataset'

train_dataloader = dict(
    batch_size={self.batch_size},
    num_workers={0 if self.is_cpu else 4},
    persistent_workers={False if self.is_cpu else True},
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size={self.batch_size},
    num_workers={0 if self.is_cpu else 4},
    persistent_workers={False if self.is_cpu else True},
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='',
        pipeline=val_pipeline))

val_evaluator = dict(type='Accuracy', topk=(1,))

# Test dataloader (required for inference)
test_dataloader = dict(
    batch_size={self.batch_size},
    num_workers={0 if self.is_cpu else 4},
    persistent_workers={False if self.is_cpu else True},
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='',
        pipeline=val_pipeline))

test_evaluator = dict(type='Accuracy', topk=(1,))

# Launcher
launcher = 'none'  # 'none', 'pytorch', 'slurm', 'mpi'
"""
        return config

    def _configure(self, config_path, train_annotations_path, train_images_path, classes_path, output_path):
        from mmengine import Config

        dataset_type = "ImageNet" if self.concepts_mutually_exclusive else "CustomDataset"

        cfg = Config.fromfile(config_path)
        logging.info(f"Loaded config from file: {config_path}")

        if train_annotations_path:
            cfg.train_dataloader.dataset.ann_file = train_annotations_path
            cfg.val_dataloader.dataset.ann_file = train_annotations_path
            logging.info(f"Set ann_file to {train_annotations_path}")

        if train_images_path:
            cfg.train_dataloader.dataset.data_prefix = train_images_path
            cfg.val_dataloader.dataset.data_prefix = train_images_path
            logging.info(f"Set data_prefix to {train_images_path}")

        if classes_path:
            cfg.train_dataloader.dataset.classes = classes_path
            cfg.val_dataloader.dataset.classes = classes_path
            logging.info(f"Set classes to {classes_path}")

        cfg.train_dataloader.dataset.type = dataset_type
        cfg.val_dataloader.dataset.type = dataset_type
        logging.info(f"Set dataset_type to {dataset_type}")

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
        logger.info(f"Loading model with MMPreTrain inferencer on {device}")

        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            self.inferencer = ImageClassificationInferencer(
                model=config_path, pretrained=checkpoint_path, device=device
            )
        finally:
            torch.load = original_load

        logger.info("Loaded MMPreTrain ImageClassificationInferencer")

        with open(os.path.join(model_folder, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
            concepts = config.get("concepts", [])

        self.id2label = {i: c["name"] for i, c in enumerate(concepts)}
        self.num_classes = len(self.id2label)
        logger.info(
            f"Loaded {self.num_classes} classes: {list(self.id2label.values())}"
        )

    @VisualClassifierClass.method
    def predict(self, image: Image) -> List[Concept]:
        pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_image.save(tmp, format="JPEG")
            tmp_path = tmp.name

        try:
            start = perf_counter_ns()
            results = self.inferencer(tmp_path)
            inference_time_ms = (perf_counter_ns() - start) / 1_000_000
            logger.info(f"Inference took {inference_time_ms:.2f} ms")
        finally:
            os.unlink(tmp_path)

        result = []
        if results and len(results) > 0:
            pred_scores = results[0].get("pred_scores", None)
            if pred_scores is not None:
                for idx, score in enumerate(pred_scores):
                    if idx < len(self.id2label):
                        result.append(
                            Concept(
                                id=str(idx), name=self.id2label[idx], value=float(score)
                            )
                        )
            else:
                pred_class = results[0].get("pred_class", "")
                pred_score = results[0].get("pred_score", 0.0)
                for idx, name in self.id2label.items():
                    if name == pred_class:
                        result.append(
                            Concept(id=str(idx), name=name, value=float(pred_score))
                        )
                    else:
                        result.append(Concept(id=str(idx), name=name, value=0.0))

        result.sort(key=lambda x: x.value, reverse=True)
        return result
