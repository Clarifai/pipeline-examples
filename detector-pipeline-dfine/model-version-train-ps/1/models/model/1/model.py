# Standard library imports
import inspect
import json
import os
import shutil
import signal
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple

# Third-party imports
import torch
import yaml
from PIL import Image as PILImage
from transformers import DFineForObjectDetection, AutoImageProcessor
from torchvision.ops import nms

# Clarifai imports
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.visual_detector_class import VisualDetectorClass
from clarifai.runners.utils.data_types import Image, Video, Region, Frame
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger

# TensorRT imports (optional)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None


def signal_handler(sig, frame):
    """Handle SIGINT and SIGTERM signals for graceful shutdown."""
    logger.info("\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# Inference helpers — identical to dfine_model/1/model.py
# ============================================================================

class TensorRTInference:
    """TensorRT inference engine wrapper for D-FINE model."""

    def __init__(self, engine_path: str):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Install tensorrt package.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None

        logger.info(f"Loading TensorRT engine from {engine_path}...")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(1, self.engine.num_io_tensors)
        ]

        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.channels = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]

        logger.info(f"TensorRT engine loaded: input shape = {input_shape}")
        logger.info(f"Output tensors: {self.output_names}")

    def __call__(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = pixel_values.shape[0]

        if not pixel_values.is_cuda:
            pixel_values = pixel_values.cuda()
        pixel_values = pixel_values.contiguous()

        self.context.set_input_shape(self.input_name, pixel_values.shape)

        outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            shape[0] = batch_size
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda").contiguous()

        self.context.set_tensor_address(self.input_name, pixel_values.data_ptr())
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        logits = outputs.get("logits")
        pred_boxes = outputs.get("pred_boxes")

        return logits, pred_boxes

    def __del__(self):
        self.context = None
        self.engine = None


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: float = 4.0,
) -> str:
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available")

    logger.info(f"Building TensorRT engine from {onnx_path}...")
    logger.info("This may take several minutes on first run...")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    logger.info("ONNX model parsed successfully")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    profile.set_shape(input_name, (1, channels, height, width), (1, channels, height, width), (8, channels, height, width))
    config.add_optimization_profile(profile)

    logger.info(f"Building engine with input shape: (1-8, {channels}, {height}, {width})")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to {engine_path}")
    return engine_path


def center_to_corners_format(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def detect_objects_tensorrt(
    images: List[PILImage.Image],
    trt_engine: TensorRTInference,
    processor: AutoImageProcessor,
    id2label: Dict[int, str],
    threshold: float = 0.25
) -> List[Dict[str, Any]]:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()
    logger.info(f"TRT input shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

    logits, pred_boxes = trt_engine(pixel_values)
    logger.info(f"TRT output - logits shape: {logits.shape}, pred_boxes shape: {pred_boxes.shape}")
    logger.info(f"TRT output - logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
    logger.info(f"TRT output - pred_boxes min/max: {pred_boxes.min().item():.4f}/{pred_boxes.max().item():.4f}")

    results = []
    batch_size = logits.shape[0]

    for i in range(batch_size):
        img_logits = logits[i]
        img_boxes = pred_boxes[i]

        probs = torch.softmax(img_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        logger.info(f"Post-softmax max score: {scores.max().item():.4f}, threshold: {threshold}")

        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = img_boxes[keep]

        boxes = center_to_corners_format(boxes)
        boxes = boxes.clamp(0, 1)

        results.append({
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        })

    return results


def detect_objects(
    images: List[PILImage.Image],
    model: DFineForObjectDetection,
    processor: AutoImageProcessor,
    device: str,
    threshold: float = 0.25
) -> List[Dict[str, Any]]:
    model_inputs = processor(images=images, return_tensors="pt").to(device)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    model_output = model(**model_inputs)
    results = processor.post_process_object_detection(model_output, threshold=threshold)

    for i, (result, img) in enumerate(zip(results, images)):
        img_width, img_height = img.size
        boxes = result["boxes"]
        if len(boxes) > 0:
            boxes[:, [0, 2]] /= img_width
            boxes[:, [1, 3]] /= img_height
            boxes = boxes.clamp(0, 1)
            results[i]["boxes"] = boxes

    return results


def apply_nms(
    results: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.2
) -> List[Dict[str, torch.Tensor]]:
    filtered_results = []
    for result in results:
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold)
            filtered_results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep]
            })
        else:
            filtered_results.append(result)

    return filtered_results


# ============================================================================
# Training helpers
# ============================================================================

DFINE_PRETRAINED = "ustc-community/dfine-small-obj2coco"


class CocoDetectionDataset(torch.utils.data.Dataset):
    """COCO-format dataset for D-FINE fine-tuning."""

    def __init__(self, image_dir, annotation_file, processor):
        with open(annotation_file) as f:
            self.coco = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

        self.img_id_to_anns = {}
        for ann in self.coco['annotations']:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.cat_id_to_label = {}
        for i, cat in enumerate(self.coco['categories']):
            self.cat_id_to_label[cat['id']] = i

        self.images = self.coco['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        image = PILImage.open(os.path.join(self.image_dir, img_info['file_name'])).convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])

        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "image_id": img_id,
                    "category_id": self.cat_id_to_label[ann['category_id']],
                    "bbox": ann['bbox'],
                    "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    "iscrowd": ann.get('iscrowd', 0),
                }
                for ann in anns
            ],
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels,
    }


# ============================================================================
# Main class: identical inference to dfine_model + train()
# ============================================================================

class MyRunner(VisualDetectorClass):
    """D-FINE object detection model with inference + fine-tuning."""

    def __init__(self):
        super().__init__()
        self._model: Optional[DFineForObjectDetection] = None
        self._processor: Optional[AutoImageProcessor] = None
        self._model_labels: Optional[Dict[int, str]] = None
        self._concepts_map: Optional[Dict[int, Dict[str, str]]] = None
        self._device: Optional[str] = None
        self._checkpoint_path: Optional[str] = None
        self._trt_engine: Optional[TensorRTInference] = None
        self._use_tensorrt: bool = False

    def load_model(self):
        """Download checkpoints and initialize model/TensorRT engine."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        self._checkpoint_path = builder.download_checkpoints(stage="runtime")
        self._concepts_map = VisualDetectorClass.load_concepts_from_config(model_path)
        logger.info(f"Checkpoints ready at: {self._checkpoint_path}")

        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Lazy load model on first use - called from worker process after fork."""
        if self._model is not None or self._trt_engine is not None:
            return

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing model on device: {self._device}")

        # Save TensorRT engine to /tmp so it works even if checkpoint dir is read-only
        os.makedirs("/tmp/.cache", exist_ok=True)
        engine_path = os.path.join("/tmp/.cache", "dfine_fp32.engine")
        onnx_path = os.path.join(self._checkpoint_path, "dfine.onnx")

        model_dir = os.path.dirname(__file__)
        model_dir_onnx = os.path.join(model_dir, "dfine.onnx")
        if not os.path.exists(onnx_path) and os.path.exists(model_dir_onnx):
            onnx_path = model_dir_onnx

        if TENSORRT_AVAILABLE and self._device == 'cuda':
            if not os.path.exists(engine_path) and os.path.exists(onnx_path):
                try:
                    logger.info("TensorRT engine not found, building from ONNX...")
                    build_tensorrt_engine(onnx_path, engine_path, fp16=False)
                except Exception as e:
                    logger.warning(f"Failed to build TensorRT engine: {e}")

            if os.path.exists(engine_path):
                try:
                    self._trt_engine = TensorRTInference(engine_path)
                    self._use_tensorrt = True
                    logger.info("TensorRT engine loaded successfully!")

                    self._processor = AutoImageProcessor.from_pretrained(self._checkpoint_path, use_fast=True)

                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(self._checkpoint_path)
                    self._model_labels = config.id2label

                    logger.info("D-Fine model ready with TensorRT backend!")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load TensorRT engine: {e}")
                    logger.info("Falling back to PyTorch backend...")
                    self._trt_engine = None
                    self._use_tensorrt = False

        logger.info("Loading PyTorch model...")
        self._model = DFineForObjectDetection.from_pretrained(self._checkpoint_path).to(self._device)
        self._processor = AutoImageProcessor.from_pretrained(self._checkpoint_path, use_fast=True)
        self._model.eval()
        self._model_labels = self._model.config.id2label

        logger.info("D-Fine model loaded successfully with PyTorch backend!")

    def _detect_objects(
        self,
        images: List[PILImage.Image],
        threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        if self._use_tensorrt and self._trt_engine is not None:
            return detect_objects_tensorrt(
                images, self._trt_engine, self._processor, self._model_labels, threshold=threshold
            )
        else:
            return detect_objects(
                images, self._model, self._processor, self._device, threshold=threshold
            )

    # ------------------------------------------------------------------
    # Inference methods — identical to dfine_model/1/model.py
    # ------------------------------------------------------------------

    @VisualDetectorClass.method
    def predict(
        self,
        image: Image,
        threshold: float = Param(default=0.25, min_value=0., max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid."),
        use_nms: bool = Param(default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based)."),
        iou_threshold: float = Param(default=0.2, min_value=0., max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True)."),
    ) -> List[Region]:
        self._ensure_model_loaded()
        logger.info(f"predict() called, image bytes length: {len(image.bytes) if image.bytes else 0}")
        pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")
        logger.info(f"PIL image size: {pil_image.size}")
        with torch.no_grad():
            results = self._detect_objects([pil_image], threshold=threshold)
            logger.info(f"Detection results: {len(results)} images, first has {len(results[0]['boxes'])} boxes")
            if use_nms:
                results = apply_nms(results, iou_threshold=iou_threshold)
            outputs = VisualDetectorClass.process_detections(results, threshold, self._model_labels, self._concepts_map)
            logger.info(f"Returning {len(outputs[0])} regions")
            return outputs[0]

    @VisualDetectorClass.method
    def generate(
        self,
        video: Video,
        threshold: float = Param(default=0.25, min_value=0., max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid."),
        use_nms: bool = Param(default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based)."),
        iou_threshold: float = Param(default=0.2, min_value=0., max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True)."),
    ) -> Iterator[Frame]:
        self._ensure_model_loaded()
        frame_generator = VisualDetectorClass.video_to_frames(video.bytes)
        for frame in frame_generator:
            with torch.no_grad():
                pil_image = PILImage.open(BytesIO(frame.image.bytes)).convert("RGB")
                results = self._detect_objects([pil_image], threshold=threshold)
                if use_nms:
                    results = apply_nms(results, iou_threshold=iou_threshold)
                outputs = VisualDetectorClass.process_detections(results, threshold, self._model_labels, self._concepts_map)
                frame.regions = outputs[0]
                yield frame

    @VisualDetectorClass.method
    def stream_image(
        self,
        image_stream: Iterator[Image],
        threshold: float = Param(default=0.25, min_value=0., max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid."),
        use_nms: bool = Param(default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based)."),
        iou_threshold: float = Param(default=0.2, min_value=0., max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True)."),
        batch_size: int = Param(default=1, min_value=1, max_value=32,
            description="Number of images to batch together. Use 1 for lowest latency (real-time streaming), or higher (4-8) for maximum throughput (offline processing)."),
    ) -> Iterator[List[Region]]:
        self._ensure_model_loaded()
        batch = []
        for image in image_stream:
            pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")
            batch.append(pil_image)
            if len(batch) >= batch_size:
                with torch.no_grad():
                    results = self._detect_objects(batch, threshold=threshold)
                    if use_nms:
                        results = apply_nms(results, iou_threshold=iou_threshold)
                    outputs = VisualDetectorClass.process_detections(results, threshold, self._model_labels, self._concepts_map)
                    for output in outputs:
                        yield output
                batch = []
        if batch:
            with torch.no_grad():
                results = self._detect_objects(batch, threshold=threshold)
                if use_nms:
                    results = apply_nms(results, iou_threshold=iou_threshold)
                outputs = VisualDetectorClass.process_detections(results, threshold, self._model_labels, self._concepts_map)
                for output in outputs:
                    yield output

    @VisualDetectorClass.method
    def stream_video(
        self,
        video_stream: Iterator[Video],
        threshold: float = Param(default=0.25, min_value=0., max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid."),
        use_nms: bool = Param(default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based)."),
        iou_threshold: float = Param(default=0.2, min_value=0., max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True)."),
    ) -> Iterator[Frame]:
        for video in video_stream:
            for frame_result in self.generate(
                video, threshold=threshold, use_nms=use_nms, iou_threshold=iou_threshold
            ):
                yield frame_result

    # ------------------------------------------------------------------
    # Training — pipeline_step.py calls this via to_pipeline_parser()
    # ------------------------------------------------------------------

    @classmethod
    def to_pipeline_parser(cls):
        parser = __import__('argparse').ArgumentParser()
        sig = inspect.signature(cls.train)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = str if param.annotation == inspect.Parameter.empty else param.annotation
            if arg_type == bool:
                arg_type = lambda x: x.lower() in ('true', '1', 'yes')
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def train(self,
              user_id: str = "YOUR_USER_ID",
              app_id: str = "YOUR_APP_ID",
              model_id: str = "dfine_detector",
              clarifai_dataset_id: str = "None",
              dataset_version_id: str = "",
              concepts: str = '["bird","cat"]',
              pretrained_model: str = DFINE_PRETRAINED,
              num_epochs: int = 50,
              batch_size: int = 8,
              learning_rate: float = 4e-4,
              weight_decay: float = 1e-4,
              warmup_steps: int = 500,
              seed: int = -1,
              max_steps: int = -1,
              ) -> str:
        """Fine-tune D-FINE on a COCO-format dataset, export ONNX, place into model/1/ for local-runner."""
        from clarifai.client.artifact_version import ArtifactVersion
        from transformers import TrainingArguments, Trainer

        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        work_dir = "/tmp/dfine_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        # STEP 1: Download dataset
        logger.info("=" * 80)
        logger.info("STEP 1: Downloading and Extracting Dataset")
        logger.info("=" * 80)

        if clarifai_dataset_id != "None":
            # Download from Clarifai dataset API (same as detector-pipeline-yolof)
            sys.path.insert(0, os.path.dirname(__file__))
            from dataset_helpers import download_dataset, convert_dataset_to_coco_format, create_classes_file
            concepts_list = json.loads(concepts) if isinstance(concepts, str) else concepts

            dataset_name = download_dataset(
                user_id=user_id,
                app_id=app_id,
                dataset_id=clarifai_dataset_id,
                dataset_version_id=dataset_version_id,
                pat=pat,
                output_dir=work_dir,
                concepts=concepts_list,
            )

            convert_output = convert_dataset_to_coco_format(
                dataset_name=dataset_name,
                dataset_split="train",
                output_root=work_dir,
            )

            images_dir = os.path.join(convert_output.images_output_root, "images")
            annotations_path = convert_output.annotations_path

            classes_path = create_classes_file(
                dataset_name=dataset_name,
                output_dir=convert_output.images_output_root,
                concepts=None,
            )
        else:
            # Download from artifact (default behavior)
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

            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(work_dir)

            images_dir = os.path.join(work_dir, "train", "images")
            annotations_path = os.path.join(work_dir, "train", "annotations.json")
            classes_path = os.path.join(work_dir, "train", "classes.txt")

        with open(classes_path) as f:
            classes = [line.strip() for line in f if line.strip()]
        num_classes = len(classes)
        id2label = {i: name for i, name in enumerate(classes)}
        label2id = {name: i for i, name in enumerate(classes)}
        logger.info(f"Dataset: {num_classes} classes: {classes}")

        # STEP 2: Load pretrained D-FINE
        logger.info("=" * 80)
        logger.info(f"STEP 2: Loading pretrained D-FINE from {pretrained_model}")
        logger.info("=" * 80)

        processor = AutoImageProcessor.from_pretrained(pretrained_model)
        model = DFineForObjectDetection.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        # STEP 3: Create dataset
        train_dataset = CocoDetectionDataset(images_dir, annotations_path, processor)
        logger.info(f"Training samples: {len(train_dataset)}")

        # STEP 4: Fine-tune
        logger.info("=" * 80)
        logger.info("STEP 4: Fine-tuning D-FINE")
        logger.info("=" * 80)

        output_dir = os.path.join(work_dir, "output")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.999,
            lr_scheduler_type="linear",
            warmup_steps=warmup_steps,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=10,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            seed=seed if seed > 0 else 42,
            max_steps=max_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )

        logger.info(f"Config: lr={learning_rate}, wd={weight_decay}, bs={batch_size}, "
                     f"epochs={num_epochs}, warmup={warmup_steps}")
        trainer.train()
        logger.info("Training completed!")

        checkpoint_dir = os.path.join(work_dir, "final_checkpoint")
        trainer.save_model(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        logger.info(f"Saved HF checkpoint to {checkpoint_dir}")

        # STEP 5: Export ONNX and place checkpoint + ONNX into model/1/
        logger.info("=" * 80)
        logger.info("STEP 5: Exporting ONNX and setting up model for local-runner")
        logger.info("=" * 80)

        model_template_dir = Path(__file__).parent.parent  # model/
        model_1_dir = model_template_dir / "1"

        # Copy HF checkpoint into model/1/checkpoints/
        ckpt_dest = model_1_dir / "checkpoints"
        if ckpt_dest.exists():
            shutil.rmtree(ckpt_dest)
        shutil.copytree(checkpoint_dir, ckpt_dest)
        logger.info(f"Copied HF checkpoint to {ckpt_dest}")

        # Export ONNX into model/1/dfine.onnx (same location as dfine_model)
        sys.path.insert(0, os.path.dirname(__file__))
        from model_export_helper import export_onnx
        onnx_path = str(model_1_dir / "dfine.onnx")
        export_onnx(checkpoint_path=checkpoint_dir, output_path=onnx_path)
        logger.info(f"Exported ONNX to {onnx_path}")

        # Update config.yaml with trained classes
        config_yaml_path = model_template_dir / "config.yaml"
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        config['concepts'] = [
            {'id': str(i), 'name': name} for i, name in enumerate(classes)
        ]
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated config.yaml with {num_classes} classes")

        # STEP 6: Upload checkpoint artifact and model to Clarifai
        logger.info("=" * 80)
        logger.info("STEP 6: Uploading checkpoint artifact and model to Clarifai")
        logger.info("=" * 80)

        from clarifai.client.artifact import Artifact
        from clarifai.runners.models.model_builder import ModelBuilder
        import tempfile

        # 6a: Upload checkpoint + ONNX as artifact
        artifact_id = f"{model_id}_checkpoint"
        try:
            artifacts = Artifact().list(user_id=user_id, app_id=app_id)
            if not any(a.id == artifact_id for a in artifacts):
                Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
            # Copy ONNX into checkpoint dir so artifact has everything needed
            shutil.copy2(onnx_path, os.path.join(checkpoint_dir, "dfine.onnx"))
            checkpoint_zip = os.path.join(work_dir, "checkpoint.zip")
            shutil.make_archive(checkpoint_zip.replace('.zip', ''), 'zip', checkpoint_dir)
            version = ArtifactVersion().upload(
                file_path=checkpoint_zip,
                artifact_id=artifact_id,
                user_id=user_id,
                app_id=app_id,
                visibility="private",
            )
            logger.info(f"Artifact uploaded: {version.id}")
        except Exception as e:
            logger.warning(f"Artifact upload failed (non-fatal): {e}")

        # 6b: Prepare a temp copy of the model dir and upload to Clarifai
        clarifai_api_base = os.getenv("CLARIFAI_API_BASE", "https://api.clarifai.com")
        temp_model_dir = Path(tempfile.mkdtemp()) / "dfine_model"

        try:
            shutil.copytree(model_template_dir, temp_model_dir)
            logger.info(f"Prepared model package at {temp_model_dir}")

            # Copy HF checkpoint into the temp model's 1/checkpoints/
            temp_ckpt = temp_model_dir / "1" / "checkpoints"
            if temp_ckpt.exists():
                shutil.rmtree(temp_ckpt)
            shutil.copytree(checkpoint_dir, temp_ckpt)

            # Copy ONNX into the temp model
            temp_onnx = temp_model_dir / "1" / "dfine.onnx"
            shutil.copy2(onnx_path, temp_onnx)

            # Update config.yaml with classes, user_id, app_id, model_id
            temp_config_path = temp_model_dir / "config.yaml"
            with open(temp_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            model_config['concepts'] = [
                {'id': str(i), 'name': name} for i, name in enumerate(classes)
            ]
            if 'model' not in model_config:
                model_config['model'] = {}
            model_config['model']['user_id'] = user_id
            model_config['model']['app_id'] = app_id
            model_config['model']['id'] = model_id
            with open(temp_config_path, 'w') as f:
                yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

            # Upload model to Clarifai
            builder = ModelBuilder(str(temp_model_dir), app_not_found_action="prompt")
            exists = builder.check_model_exists()
            if exists:
                logger.info(f"Model exists at {builder.model_ui_url}, creating new version.")
            else:
                logger.info(f"New model will be created at {builder.model_ui_url}")

            try:
                builder.upload_model_version()
            except KeyError as e:
                # This can happen when clarifai config context is not set up locally
                # The model upload itself already succeeded at this point
                logger.warning(f"Post-upload snippet generation failed (non-fatal): {e}")
            logger.info(f"Model uploaded: {builder.model_ui_url}")
            logger.info(f"Model Version ID: {builder.model_version_id}")
        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED!")
        logger.info(f"Checkpoint: {ckpt_dest}")
        logger.info(f"ONNX: {onnx_path}")
        logger.info(f"Model uploaded to Clarifai successfully!")
        logger.info("=" * 80)
        return checkpoint_dir
