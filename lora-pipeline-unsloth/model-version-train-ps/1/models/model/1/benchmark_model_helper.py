#!/usr/bin/env python3
"""
GPU memory benchmark: spawns vLLM or sglang server subprocess,
measures peak GPU memory via torch.cuda.max_memory_reserved + pynvml,
returns max of both.
"""
import json
import math
import subprocess
import sys
import time
import logging
import urllib.request

import yaml

try:
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PYTHON_EXEC = sys.executable
SERVER_PORT = 23334
SERVER_URL = f"http://localhost:{SERVER_PORT}"


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

    def cleanup(self):
        if self.use_pynvml:
            nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()


def _vllm_cmd(model, max_model_len, adapter_path):
    cmd = [
        PYTHON_EXEC, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.9",
        "--dtype", "auto",
        "--port", str(SERVER_PORT),
        "--host", "localhost",
        "--trust-remote-code",
    ]
    if adapter_path:
        cmd += ["--enable-lora", "--lora-modules", f"adapter={adapter_path}"]
    return cmd


def _sglang_cmd(model, max_model_len, adapter_path):
    cmd = [
        PYTHON_EXEC, "-m", "sglang.launch_server",
        "--model-path", model,
        "--context-length", str(max_model_len),
        "--mem-fraction-static", "0.9",
        "--dtype", "auto",
        "--port", str(SERVER_PORT),
        "--host", "localhost",
        "--trust-remote-code",
    ]
    if adapter_path:
        cmd += ["--enable-lora", "--lora-paths", f"adapter={adapter_path}"]
    return cmd


def _wait_for_server(url, timeout=300, poll=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return
        except Exception:
            time.sleep(poll)
    raise TimeoutError(f"Server at {url} not ready within {timeout}s")


def _benchmark_via_server(
    base_model_name, adapter_path, max_model_len, device_id, backend,
    n_warmup_requests, server_timeout,
):
    """Benchmark by launching vLLM/sglang server subprocess."""
    import torch

    torch.cuda.reset_peak_memory_stats(device_id)

    build_cmd = _vllm_cmd if backend == "vllm" else _sglang_cmd
    cmd = build_cmd(base_model_name, max_model_len, adapter_path)

    logger.info(f"[{backend}] launching server for {base_model_name}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    pynvml_peak_mb = 0.0

    try:
        _wait_for_server(SERVER_URL, timeout=server_timeout)
        logger.info("Server ready - running warmup requests")
        time.sleep(3)

        import openai
        client = openai.OpenAI(api_key="notset", base_url=f"{SERVER_URL}/v1")
        model_id = client.models.list().data[0].id

        with GPUMemoryMonitor(device_id) as mon:
            for i in range(n_warmup_requests):
                try:
                    client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": f"Count to {i + 1}."}],
                        max_tokens=128,
                    )
                except Exception as e:
                    logger.warning(f"Warmup request {i} failed: {e}")
                pynvml_peak_mb = max(pynvml_peak_mb, mon.get_memory_mb())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    torch_peak_mb = torch.cuda.max_memory_reserved(device_id) / 1024 / 1024
    peak_mb = max(torch_peak_mb, pynvml_peak_mb)
    return torch_peak_mb, pynvml_peak_mb, peak_mb


def _benchmark_via_transformers(base_model_name, adapter_path, device_id):
    """Fallback benchmark: load model with transformers + peft, measure GPU memory."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    torch.cuda.reset_peak_memory_stats(device_id)

    logger.info(f"[transformers] loading {base_model_name} for memory measurement")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        logger.info(f"[transformers] loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    # Measure peak GPU memory after loading (weights are the main memory consumer)
    pynvml_peak_mb = 0.0
    with GPUMemoryMonitor(device_id) as mon:
        pynvml_peak_mb = mon.get_memory_mb()

    torch_peak_mb = torch.cuda.max_memory_reserved(device_id) / 1024 / 1024
    peak_mb = max(torch_peak_mb, pynvml_peak_mb)

    del model
    torch.cuda.empty_cache()

    return torch_peak_mb, pynvml_peak_mb, peak_mb


def benchmark_peak_gpu_mb(
    base_model_name,
    adapter_path=None,
    max_model_len=4096,
    device_id=0,
    backend="vllm",
    n_warmup_requests=5,
    server_timeout=300,
):
    """
    Measure peak GPU memory for model inference.

    Tries to launch a vLLM/sglang server first. If that fails (e.g. backend not
    installed), falls back to loading the model with transformers + peft.
    """
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - returning zeros")
        return {"torch_peak_mb": 0, "pynvml_peak_mb": 0, "peak_mb": 0}

    # Check if the backend is importable before wasting time on a timeout
    _backend_module = "vllm" if backend == "vllm" else "sglang"
    try:
        __import__(_backend_module)
        _backend_available = True
    except ImportError:
        _backend_available = False
        logger.info(f"{_backend_module} not installed, skipping server-based benchmark")

    try:
        if not _backend_available:
            raise ImportError(f"{_backend_module} not installed")
        torch_peak_mb, pynvml_peak_mb, peak_mb = _benchmark_via_server(
            base_model_name, adapter_path, max_model_len, device_id, backend,
            n_warmup_requests, server_timeout,
        )
    except Exception as e:
        logger.warning(f"Server-based benchmark failed ({e}), falling back to transformers")
        torch_peak_mb, pynvml_peak_mb, peak_mb = _benchmark_via_transformers(
            base_model_name, adapter_path, device_id,
        )
        backend = "transformers"

    logger.info(f"torch peak: {torch_peak_mb:.1f} MB | pynvml peak: {pynvml_peak_mb:.1f} MB | final: {peak_mb:.1f} MB")
    return {
        "torch_peak_mb": round(torch_peak_mb, 2),
        "pynvml_peak_mb": round(pynvml_peak_mb, 2),
        "peak_mb": round(peak_mb, 2),
        "backend": backend,
        "model": base_model_name,
    }


def update_config_with_benchmark(benchmark_results, config_yaml_path):
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)

    gpu_mb = benchmark_results["peak_mb"]
    gpu_gb = math.ceil(gpu_mb * 1.2 / 1024)  # 20% buffer
    logger.info(f"Benchmark: {gpu_mb:.0f} MB -> {gpu_gb} Gi (with 20% buffer)")

    config.setdefault("inference_compute_info", {})
    config["inference_compute_info"]["num_accelerators"] = 1
    config["inference_compute_info"]["accelerator_memory"] = f"{gpu_gb}Gi"

    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def benchmark_and_update_config(
    base_model_name,
    adapter_path=None,
    config_yaml_path=None,
    max_model_len=4096,
    device_id=0,
    backend="vllm",
    save_benchmark_json=None,
):
    results = benchmark_peak_gpu_mb(
        base_model_name, adapter_path, max_model_len, device_id, backend
    )
    if save_benchmark_json:
        with open(save_benchmark_json, "w") as f:
            json.dump(results, f, indent=2)
    if config_yaml_path:
        update_config_with_benchmark(results, config_yaml_path)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Benchmark LLM peak GPU memory")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--config-yaml", default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    p.add_argument("--save-json", default=None)
    args = p.parse_args()

    results = benchmark_and_update_config(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        config_yaml_path=args.config_yaml,
        max_model_len=args.max_model_len,
        device_id=args.device_id,
        backend=args.backend,
        save_benchmark_json=args.save_json,
    )
    print(json.dumps(results, indent=2))
