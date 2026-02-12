#!/usr/bin/env python
"""Test script: load LoRA adapter with vLLM and run inference + benchmark."""
import sys
import os

# Point to model module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import importlib
_mod = importlib.import_module("model.1.model")
UnslothLoRAVLLM = _mod.UnslothLoRAVLLM
vllm_openai_server = _mod.vllm_openai_server
from openai import OpenAI

BASE_MODEL = "unsloth/Qwen3-32B"
ADAPTER_PATH = "/tmp/lora_work_dir/lora_adapter"
LORA_MODEL_ID = "lora-adapter"

# Qwen3 thinking mode generates very long chains; disable for fast tests
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def main():
    print("=" * 60)
    print("Starting vLLM server with LoRA adapter...")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Adapter:    {ADAPTER_PATH}")
    print("=" * 60)

    server_args = {
        'gpu_memory_utilization': 0.9,
        'kv_cache_dtype': 'auto',
        'tensor_parallel_size': 1,
        'port': 23333,
        'host': 'localhost',
        'trust_remote_code': True,
        'enable_lora': True,
        'max_lora_rank': 64,
        'lora_modules': [f"{LORA_MODEL_ID}={ADAPTER_PATH}"],
    }

    server = vllm_openai_server(BASE_MODEL, **server_args)

    client = OpenAI(api_key="notset", base_url=f"http://{server.host}:{server.port}/v1")

    # Test predict-style call
    print("\n--- Test 1: predict (non-streaming) ---")
    response = client.chat.completions.create(
        model=LORA_MODEL_ID,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=64,
        temperature=0.7,
        extra_body=NO_THINK,
    )
    print(response.choices[0].message.content)

    # Test generate-style call (streaming)
    print("\n--- Test 2: generate (streaming) ---")
    for chunk in client.chat.completions.create(
        model=LORA_MODEL_ID,
        messages=[{"role": "user", "content": "Write a haiku about coding."}],
        max_tokens=64,
        temperature=0.7,
        stream=True,
        extra_body=NO_THINK,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    # Test using the class predict method directly
    print("\n--- Test 3: UnslothLoRAVLLM.predict() ---")
    m = UnslothLoRAVLLM()
    m.client = client
    m.model = LORA_MODEL_ID
    result = m.predict(prompt="What is 2+2?", max_tokens=32)
    print(result)

    # Benchmark GPU memory using the running server
    print("\n--- Test 4: benchmark() ---")
    m.server = server
    results = m.benchmark(n_warmup_requests=5)
    print(f"Benchmark results: {results}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Cleanup
    from clarifai.runners.utils.model_utils import terminate_process
    terminate_process(server.process)


if __name__ == "__main__":
    main()
