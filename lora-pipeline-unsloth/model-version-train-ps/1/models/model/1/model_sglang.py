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
    """
    SGLang-based serving for LoRA fine-tuned models.
    To use: rename model.py -> model_vllm.py, rename model_sglang.py -> model.py
    """

    client = True
    model = True

    def load_model(self):
        """Load the model and start the SGLang server."""
        from openai_server_starter import OpenAI_APIServer

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        # Get base model repo_id from config
        checkpoints_config = model_config.get("checkpoints", {})
        base_model = checkpoints_config.get("repo_id")

        # Download checkpoints if configured
        stage = checkpoints_config.get("when", "runtime")
        checkpoints = base_model
        if stage in ["build", "runtime"]:
            try:
                downloaded = builder.download_checkpoints(stage=stage)
                if downloaded and downloaded != checkpoints:
                    checkpoints = downloaded
            except Exception:
                logger.info(f"Using HuggingFace model ID directly: {checkpoints}")

        # Log SGLang version for debugging
        try:
            import sglang
            logger.info(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
        except Exception as e:
            logger.info(f"Could not determine SGLang info: {e}")

        server_args = {
            "checkpoints": checkpoints,
            "host": "0.0.0.0",
            "port": 23333,
            "context_length": 4096,
            "mem_fraction_static": 0.85,
            'tp_size': 1,
            "additional_list_args": [
                "--trust-remote-code",
                "--allow-auto-truncate",
                "--max-running-requests",
                "128",
                "--sampling-defaults",
                "openai",
                "--served-model-name",
                "lora-fine-tuned-model",
                "--load-format",
                "auto",
                "--model-impl",
                "auto",
                '--stream-interval',
                '5',
            ]
        }

        if server_args.get("additional_list_args") == ['']:
            server_args.pop("additional_list_args")

        self.server = OpenAI_APIServer.from_sglang_backend(**server_args)

        self.client = OpenAI(
            api_key="notset",
            base_url=UnslothLoRASGLang.make_api_url(self.server.host, self.server.port)
        )
        self.model = self._get_model()

        logger.info(f"SGLang model loaded successfully: {self.model}")

    def _get_model(self):
        try:
            return self.client.models.list().data[0].id
        except Exception as e:
            raise ConnectionError("Failed to retrieve model ID from API") from e

    @staticmethod
    def make_api_url(host: str, port: int, version: str = "v1") -> str:
        return f"http://{host}:{port}/{version}"

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str,
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="A decimal number that determines the degree of randomness in the response.",
        ),
        top_p: float = Param(
            default=0.8,
            description="An alternative to sampling with temperature.",
        ),
    ) -> str:
        """Predict response for given prompt and chat history."""
        openai_messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if response.usage and response.usage.prompt_tokens and response.usage.completion_tokens:
            self.set_output_context(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str,
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="A decimal number that determines the degree of randomness in the response.",
        ),
        top_p: float = Param(
            default=0.8,
            description="An alternative to sampling with temperature.",
        ),
    ) -> Iterator[str]:
        """Stream generated text tokens."""
        openai_messages = build_openai_messages(prompt=prompt, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            if chunk.choices:
                text = (
                    chunk.choices[0].delta.content
                    if (chunk and chunk.choices[0].delta.content) is not None
                    else ''
                )
                yield text

    def test(self):
        """Test the model locally."""
        try:
            print("Testing predict...")
            print(
                self.predict(
                    prompt="Hello, how are you?",
                )
            )
        except Exception as e:
            print("Error in predict", e)

        try:
            print("Testing generate...")
            for each in self.generate(
                prompt="Hello, how are you?",
            ):
                print(each, end="")
            print()
        except Exception as e:
            print("Error in generate", e)


if __name__ == "__main__":
    m = UnslothLoRASGLang()
    m.load_model()
    m.test()
