#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# Try to use clarifai.utils.logging, fallback to standard logging
try:
    from clarifai.utils.logging import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class GPUMemoryMonitor:

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.use_pynvml = HAS_PYNVML
        if self.use_pynvml:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_id)

    def get_memory_mb(self) -> float:
        if self.use_pynvml:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / 1024 / 1024
        return torch.cuda.memory_allocated(self.device_id) / 1024 / 1024

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats(self.device_id)

    def cleanup(self):
        if self.use_pynvml:
            nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def benchmark_model(
    checkpoint_path: str,
    config_py_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    device_id: int = 0,
    is_cpu: int = 0,
) -> dict:
    if is_cpu == 1 or not torch.cuda.is_available():
        if is_cpu == 1:
            logger.info("Skipping GPU benchmark (is_cpu=1)")
        else:
            logger.warning("CUDA not available, returning zero GPU requirements")
        return {
            "gpu_mb_required_minimum": 0,
            "gpu_mb_model_load": 0,
            "requested_gpu_mb_temporary": 0,
        }

    from mmpretrain import ImageClassificationInferencer

    device = f"cuda:{device_id}"
    C, H, W = input_shape

    logger.info(f"✓ CUDA enabled for benchmark: {torch.cuda.get_device_name(device_id)}")
    mem_before_load_mb = torch.cuda.memory_reserved(device_id) / 1024**2
    logger.info(f"GPU memory before loading benchmark model: {mem_before_load_mb:.2f} MB (should match cleanup value)")

    with GPUMemoryMonitor(device_id) as monitor:
        # Measure baseline
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_mb = monitor.get_memory_mb()
        logger.info(f"Baseline GPU memory: {baseline_mb:.2f} MB")

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        # Patch torch.load to use weights_only=False for mmengine checkpoints (PyTorch 2.6+)
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            inferencer = ImageClassificationInferencer(
                model=config_py_path, pretrained=checkpoint_path, device=device
            )
        finally:
            torch.load = original_load
        torch.cuda.synchronize()
        # Check if the inferencer is a GPU inferencer
        logging.info(f"Benchmark model device: {next(inferencer.model.parameters()).device}")

        model_load_mb = monitor.get_memory_mb()
        mem_after_load_mb = torch.cuda.memory_reserved(device_id) / 1024**2
        logger.info(f"After model load: {model_load_mb:.2f} MB")
        logger.info(f"GPU memory reserved after loading: {mem_after_load_mb:.2f} MB")
        logger.info(f"GPU memory increase: {mem_after_load_mb - mem_before_load_mb:.2f} MB")

        # Create dummy images
        logger.info(f"Creating {batch_size} dummy images ({C}x{H}x{W})")
        temp_files = []
        for _ in range(batch_size):
            img_array = np.random.randint(0, 255, (H, W, C), dtype=np.uint8)
            img = PILImage.fromarray(img_array)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp, format="JPEG")
            tmp.close()
            temp_files.append(tmp.name)

        try:
            # Warmup
            logger.info("Running warmup (3 iterations)...")
            for _ in range(3):
                # TRUE BATCH INFERENCE: pass list of images
                _ = inferencer(temp_files)
                torch.cuda.synchronize()

            # Benchmark
            monitor.reset_peak_memory()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            logger.info("Running benchmark (10 iterations)...")
            memories = []
            for i in range(10):
                # TRUE BATCH INFERENCE: process all images in single forward
                _ = inferencer(temp_files)
                torch.cuda.synchronize()
                memories.append(monitor.get_memory_mb())
        finally:
            for img_path in temp_files:
                try:
                    os.unlink(img_path)
                except Exception:
                    pass

        peak_mb = max(memories)
        working_mb = max(peak_mb - model_load_mb, 0)

        logger.info(f"Peak GPU memory: {peak_mb:.2f} MB")
        logger.info(f"Working memory: {working_mb:.2f} MB")

        return {
            "gpu_mb_required_minimum": int(peak_mb),
            "gpu_mb_model_load": int(model_load_mb),
            "requested_gpu_mb_temporary": int(working_mb),
            "baseline_mb": float(baseline_mb),
            "batch_size_tested": batch_size,
        }


def update_config_with_benchmark(benchmark_results: dict, config_yaml_path: str):
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)

    gpu_mb_required = benchmark_results["gpu_mb_required_minimum"]
    logger.info(f"Benchmark result: {gpu_mb_required} MB required")

    # Add 20% buffer (like Triton orchestrator)
    gpu_mb_with_buffer = int(gpu_mb_required * 1.2)
    gpu_gb = math.ceil(gpu_mb_with_buffer / 1024)

    logger.info(f"With 20% buffer: {gpu_mb_with_buffer} MB → {gpu_gb} GB")

    # Update config
    if "inference_compute_info" not in config:
        config["inference_compute_info"] = {}

    config["inference_compute_info"]["num_accelerators"] = 1
    config["inference_compute_info"]["accelerator_memory"] = f"{gpu_gb}Gi"

    # Save IN-PLACE
    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ Updated {config_yaml_path} with GPU requirement: {gpu_gb}Gi")


def benchmark_and_update_config(
    checkpoint_path: str,
    config_py_path: str,
    config_yaml_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    device_id: int = 0,
    is_cpu: int = 0,
    save_benchmark_json: str = None,
):
    benchmark_results = benchmark_model(
        checkpoint_path, config_py_path, input_shape, batch_size, device_id, is_cpu
    )

    if save_benchmark_json:
        with open(save_benchmark_json, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Saved benchmark results to {save_benchmark_json}")

    update_config_with_benchmark(benchmark_results, config_yaml_path)

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model and update config.yaml with GPU requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark_model_helper.py \\
      --checkpoint model_files/checkpoint.pth \\
      --config-py model_files/config.py \\
      --config-yaml classifier_model/config.yaml

  # With custom batch size and save results
  python benchmark_model_helper.py \\
      --checkpoint checkpoint.pth \\
      --config-py config.py \\
      --config-yaml config.yaml \\
      --batch-size 4 \\
      --save-json benchmark_results.json
        """,
    )

    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--config-py", required=True, help="Path to config.py from training"
    )
    parser.add_argument(
        "--config-yaml", required=True, help="Path to config.yaml to update"
    )
    parser.add_argument(
        "--input-shape",
        default="3,224,224",
        help="Input shape as C,H,W (default: 3,224,224)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to test (default: 1)"
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--save-json", help="Optional: save benchmark results to JSON file"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(",")))
        if len(input_shape) != 3:
            raise ValueError("Input shape must be C,H,W (3 dimensions)")
    except ValueError as e:
        logger.error(f"Invalid input shape: {e}")
        return 1

    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        return 1

    logger.info("=" * 70)
    logger.info("Model Benchmark - GPU Memory Analysis")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Config (py): {args.config_py}")
    logger.info(f"Config (yaml): {args.config_yaml}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: cuda:{args.device_id}")
    logger.info("=" * 70)

    try:
        benchmark_and_update_config(
            checkpoint_path=args.checkpoint,
            config_py_path=args.config_py,
            config_yaml_path=args.config_yaml,
            input_shape=input_shape,
            batch_size=args.batch_size,
            device_id=args.device_id,
            save_benchmark_json=args.save_json,
        )

        logger.info("\n" + "=" * 70)
        logger.info("✅ Benchmark completed successfully!")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
