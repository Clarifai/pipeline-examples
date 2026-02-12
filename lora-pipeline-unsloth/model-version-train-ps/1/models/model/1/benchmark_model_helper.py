#!/usr/bin/env python3
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

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

try:
    from clarifai.utils.logging import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

PYTHON_EXEC = sys.executable


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
        import torch
        return torch.cuda.memory_allocated(self.device_id) / 1024 / 1024

    def reset_peak_memory(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.device_id)

    def cleanup(self):
        if self.use_pynvml:
            nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def benchmark_model(
    base_model_name: str,
    adapter_path: str = None,
    max_model_len: int = 4096,
    device_id: int = 0,
) -> dict:
    """Benchmark vLLM serving by starting a subprocess and measuring GPU memory with pynvml."""
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning zero GPU requirements")
        return {
            "gpu_mb_required_minimum": 0,
            "gpu_mb_model_load": 0,
            "requested_gpu_mb_temporary": 0,
        }

    logger.info(f"CUDA enabled for benchmark: {torch.cuda.get_device_name(device_id)}")

    with GPUMemoryMonitor(device_id) as monitor:
        # Measure baseline GPU memory
        baseline_mb = monitor.get_memory_mb()
        logger.info(f"Baseline GPU memory: {baseline_mb:.2f} MB")

        # Start vLLM server subprocess
        cmds = [
            PYTHON_EXEC, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', base_model_name,
            '--max-model-len', str(max_model_len),
            '--gpu-memory-utilization', '0.9',
            '--dtype', 'auto',
            '--port', '23334',
            '--host', 'localhost',
            '--trust-remote-code',
        ]

        if adapter_path:
            cmds.extend(['--enable-lora', '--lora-modules', f'benchmark-adapter={adapter_path}'])

        logger.info(f"Starting vLLM benchmark server...")
        server_proc = subprocess.Popen(cmds, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        try:
            # Wait for server to be ready
            from clarifai.runners.utils.model_utils import wait_for_server
            wait_for_server("http://localhost:23334", timeout=300)
            logger.info("Benchmark server started")

            # Measure GPU memory after model load
            time.sleep(5)  # Allow memory to stabilize
            model_load_mb = monitor.get_memory_mb()
            logger.info(f"After model load: {model_load_mb:.2f} MB")

            # Send test requests to measure peak memory
            import openai
            client = openai.OpenAI(api_key="notset", base_url="http://localhost:23334/v1")
            models = client.models.list()
            model_id = models.data[0].id

            logger.info("Running benchmark requests...")
            memories = []
            for i in range(5):
                try:
                    client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": f"Write a short paragraph about the number {i}."}],
                        max_tokens=128,
                    )
                except Exception as e:
                    logger.warning(f"Benchmark request {i} failed: {e}")
                time.sleep(1)
                memories.append(monitor.get_memory_mb())

            peak_mb = max(memories) if memories else model_load_mb

        finally:
            # Shutdown server
            logger.info("Shutting down benchmark server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()

        working_mb = max(peak_mb - baseline_mb, 0)

        logger.info(f"Peak GPU memory: {peak_mb:.2f} MB")
        logger.info(f"Working memory: {working_mb:.2f} MB")

        return {
            "gpu_mb_required_minimum": int(peak_mb),
            "gpu_mb_model_load": int(model_load_mb),
            "requested_gpu_mb_temporary": int(working_mb),
            "baseline_mb": float(baseline_mb),
        }


def update_config_with_benchmark(benchmark_results: dict, config_yaml_path: str):
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)

    gpu_mb_required = benchmark_results["gpu_mb_required_minimum"]
    logger.info(f"Benchmark result: {gpu_mb_required} MB required")

    # Add 20% buffer
    gpu_mb_with_buffer = int(gpu_mb_required * 1.2)
    gpu_gb = math.ceil(gpu_mb_with_buffer / 1024)

    logger.info(f"With 20% buffer: {gpu_mb_with_buffer} MB -> {gpu_gb} GB")

    if "inference_compute_info" not in config:
        config["inference_compute_info"] = {}

    config["inference_compute_info"]["num_accelerators"] = 1
    config["inference_compute_info"]["accelerator_memory"] = f"{gpu_gb}Gi"

    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated {config_yaml_path} with GPU requirement: {gpu_gb}Gi")


def benchmark_and_update_config(
    base_model_name: str,
    adapter_path: str = None,
    config_yaml_path: str = None,
    max_model_len: int = 4096,
    device_id: int = 0,
    save_benchmark_json: str = None,
):
    benchmark_results = benchmark_model(
        base_model_name, adapter_path, max_model_len, device_id
    )

    if save_benchmark_json:
        with open(save_benchmark_json, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Saved benchmark results to {save_benchmark_json}")

    if config_yaml_path:
        update_config_with_benchmark(benchmark_results, config_yaml_path)

    return benchmark_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark LLM model and update config.yaml with GPU requirements",
    )
    parser.add_argument("--base-model", required=True, help="HuggingFace model ID")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--config-yaml", required=True, help="Path to config.yaml to update")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model context length")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--save-json", default=None, help="Save benchmark results to JSON file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    benchmark_and_update_config(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        config_yaml_path=args.config_yaml,
        max_model_len=args.max_model_len,
        device_id=args.device_id,
        save_benchmark_json=args.save_json,
    )
