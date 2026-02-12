import os
import sys

sys.path.append(os.path.dirname(__file__))
from typing import Iterator, List

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger
from openai import OpenAI


class UnslothLoRASGLang(OpenAIModelClass):
    """SGLang-based serving for LoRA fine-tuned models.

    To use: rename model.py -> model_vllm.py, rename model_sglang.py -> model.py

    SGLang LoRA support:
    - All 7 target modules supported: q/k/v/o_proj, gate/up/down_proj
    - Uses --lora-paths name=path (NOT --lora-modules like vLLM)
    - API model field uses "base-model:adapter-name" syntax
    - Default csgmv backend works for all linear modules
    - Does NOT support embed_tokens/lm_head with csgmv (use --lora-backend triton)
    Docs: https://docs.sglang.io/advanced_features/lora.html
    """

    client = True
    model = True

    LORA_MODEL_ID = "lora-adapter"

    def load_model(self):
        """Load the model and start the SGLang server with LoRA adapter.

        SGLang LoRA CLI syntax (differs from vLLM):
          --enable-lora --lora-paths name=path --lora-backend csgmv
        API syntax for routing to adapter:
          model="base-model-name:adapter-name"
        """
        from openai_server_starter import OpenAI_APIServer

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        checkpoints_config = model_config.get("checkpoints", {})
        base_model = checkpoints_config.get("repo_id")

        # Download checkpoints (adapter files)
        stage = checkpoints_config.get("when", "runtime")
        checkpoints = base_model
        if stage in ["build", "runtime"]:
            downloaded = builder.download_checkpoints(stage=stage)
            if downloaded and downloaded != checkpoints:
                checkpoints = downloaded

        # Resolve LoRA adapter path
        adapter_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lora_adapter")
        has_adapter = os.path.isdir(adapter_path)

        additional_args = [
            "--trust-remote-code",
            "--allow-auto-truncate",
            "--max-running-requests", "128",
            "--sampling-defaults", "openai",
            "--served-model-name", "lora-fine-tuned-model",
            "--load-format", "auto",
            "--model-impl", "auto",
            "--stream-interval", "5",
        ]

        # Add LoRA flags if adapter exists
        # SGLang syntax: --enable-lora --lora-paths name=path
        if has_adapter:
            additional_args += [
                "--enable-lora",
                "--lora-paths", f"{self.LORA_MODEL_ID}={adapter_path}",
                "--lora-backend", "csgmv",
            ]

        server_args = {
            "checkpoints": checkpoints,
            "host": "0.0.0.0",
            "port": 23333,
            "context_length": 4096,
            "mem_fraction_static": 0.85,
            "tp_size": 1,
            "additional_list_args": additional_args,
        }

        self.server = OpenAI_APIServer.from_sglang_backend(**server_args)

        self.client = OpenAI(
            api_key="notset",
            base_url=f"http://{self.server.host}:{self.server.port}/v1"
        )

        if has_adapter:
            # SGLang routes to adapter via "base-model:adapter-name" in model field
            base_id = self.client.models.list().data[0].id
            self.model = f"{base_id}:{self.LORA_MODEL_ID}"
        else:
            self.model = self.client.models.list().data[0].id

        logger.info(f"SGLang model loaded: {self.model}")

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="Maximum tokens to generate."),
                temperature: float = Param(default=0.7, description="Sampling temperature."),
                top_p: float = Param(default=0.8, description="Top-p sampling threshold."),
                ) -> str:
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        if response.usage and response.usage.prompt_tokens and response.usage.completion_tokens:
            self.set_output_context(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(self,
                 prompt: str,
                 chat_history: List[dict] = None,
                 max_tokens: int = Param(default=512, description="Maximum tokens to generate."),
                 temperature: float = Param(default=0.7, description="Sampling temperature."),
                 top_p: float = Param(default=0.8, description="Top-p sampling threshold."),
                 ) -> Iterator[str]:
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature, top_p=top_p,
            stream=True,
        ):
            if chunk.choices:
                text = chunk.choices[0].delta.content
                yield text if text is not None else ''

    def test(self):
        print("Testing predict...")
        print(self.predict(prompt="Hello, how are you?"))
        print("\nTesting generate...")
        for tok in self.generate(prompt="Hello, how are you?"):
            print(tok, end="")
        print()


if __name__ == "__main__":
    m = UnslothLoRASGLang()
    m.load_model()
    m.test()
