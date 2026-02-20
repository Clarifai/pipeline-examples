import logging
import os
import sys
import subprocess
from typing import List, Iterator
from pathlib import Path

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger

logging.basicConfig(level=logging.INFO)


PYTHON_EXEC = sys.executable


def vllm_openai_server(checkpoints, **kwargs):
    """Start vLLM OpenAI compatible server."""
    from clarifai.runners.utils.model_utils import wait_for_server, terminate_process

    cmds = [PYTHON_EXEC, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoints]
    for key, value in kwargs.items():
        if value is None:
            continue
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmds.append(f'--{param_name}')
        elif key == 'lora_modules':
            cmds.append(f'--{param_name}')
            modules = value if isinstance(value, (list, tuple)) else [value]
            cmds.extend(modules)
        else:
            cmds.extend([f'--{param_name}', str(value)])

    server = type('Server', (), {
        'host': kwargs.get('host', '0.0.0.0'),
        'port': kwargs.get('port', 23333),
        'backend': "vllm",
        'process': None
    })()

    try:
        server.process = subprocess.Popen(cmds, text=True, stdout=None, stderr=subprocess.STDOUT)
        logger.info(f"Waiting for http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info(f"Server started at http://{server.host}:{server.port}")
    except Exception as e:
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {e}")

    return server


class UnslothLoRAVLLM(OpenAIModelClass):
    """vLLM serving with optional LoRA adapter, integrated with Clarifai platform."""

    client = True
    model = True

    LORA_MODEL_ID = "lora-adapter"

    def load_model(self):
        """Load base model + optional LoRA adapter and start vLLM server."""
        from openai import OpenAI
        from clarifai.runners.models.model_builder import ModelBuilder

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        # LoRA adapter folder sits alongside model.py: 1/{model_id}_lora/
        model_id = model_config.get("model", {}).get("id")
        lora_path = os.path.join(os.path.dirname(__file__), f"{model_id}_lora")

        # Download base model checkpoints
        stage = model_config["checkpoints"]["when"]
        checkpoints = model_config["checkpoints"]["repo_id"]
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        server_args = {
            'gpu_memory_utilization': 0.7,
            'kv_cache_dtype': 'auto',
            'tensor_parallel_size': 1,
            'port': 23333,
            'host': 'localhost',
            'trust_remote_code': True,
            'enable_lora': True,
            'max_lora_rank': 64,
            'enforce_eager': False,
            'max_model_len': 4096,
        }

        if os.path.isdir(lora_path):
            server_args['lora_modules'] = [f"{self.LORA_MODEL_ID}={lora_path}"]

        self.server = vllm_openai_server(checkpoints, **server_args)
        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1'
        )
        self.model = self.LORA_MODEL_ID

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=2048, description="Maximum tokens to generate."),
                temperature: float = Param(default=0.7, description="Sampling temperature."),
                top_p: float = Param(default=0.95, description="Top-p sampling threshold."),
                ) -> str:
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        self._set_usage(response)
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(self,
                 prompt: str,
                 chat_history: List[dict] = None,
                 max_tokens: int = Param(default=2048, description="Maximum tokens to generate."),
                 temperature: float = Param(default=0.7, description="Sampling temperature."),
                 top_p: float = Param(default=0.95, description="Top-p sampling threshold."),
                 ) -> Iterator[str]:
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature, top_p=top_p,
            stream=True, stream_options={"include_usage": True},
        ):
            self._set_usage(chunk)
            if chunk.choices:
                text = chunk.choices[0].delta.content
                yield text if text is not None else ''
            else:
                yield ""

    def benchmark(self, n_warmup_requests=5, device_id=0):
        """Benchmark GPU memory using the already-running vLLM server.

        Runs warmup inference requests while tracking peak GPU memory via
        pynvml (driver-level, works cross-process) and optionally torch.cuda
        (in-process only). Returns max of both.
        Must be called after load_model().
        """
        from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        # torch.cuda stats only work if CUDA is initialized in this process.
        # vLLM runs as a subprocess so torch.cuda may not be available here.
        torch_peak_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device_id)
                has_torch = True
            else:
                has_torch = False
        except Exception:
            has_torch = False

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_id)
        pynvml_peak_mb = 0.0

        try:
            for i in range(n_warmup_requests):
                try:
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": f"Count to {i + 1}."}],
                        max_tokens=64,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                except Exception as e:
                    logger.warning(f"Warmup request {i} failed: {e}")

                info = nvmlDeviceGetMemoryInfo(handle)
                pynvml_peak_mb = max(pynvml_peak_mb, info.used / 1024 / 1024)
        finally:
            nvmlShutdown()

        if has_torch:
            torch_peak_mb = torch.cuda.max_memory_reserved(device_id) / 1024 / 1024

        peak_mb = max(torch_peak_mb, pynvml_peak_mb)

        print(f"torch peak:  {torch_peak_mb:,.1f} MB")
        print(f"pynvml peak: {pynvml_peak_mb:,.1f} MB")
        print(f"final (max): {peak_mb:,.1f} MB")

        return {
            "torch_peak_mb": round(torch_peak_mb, 2),
            "pynvml_peak_mb": round(pynvml_peak_mb, 2),
            "peak_mb": round(peak_mb, 2),
        }

    def test(self):
        print("Testing predict...")
        print(self.predict(prompt="Hello, how are you?"))
        print("\nTesting generate...")
        for tok in self.generate(prompt="Hello, how are you?"):
            print(tok, end="")
        print()


if __name__ == "__main__":
    m = UnslothLoRAVLLM()
    m.load_model()
    m.test()
