import os
import sys
from typing import Iterator, List

sys.path.append(os.path.dirname(__file__))

import requests
from openai import OpenAI
from openai_server_starter import OpenAI_APIServer

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger


class SGLangEagle3Model(OpenAIModelClass):
    """Qwen3-8B with Eagle3 speculative decoding via SGLang."""

    client = True
    model = True

    def load_model(self):
        """Load the model and start the SGLang server with Eagle3 speculation."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        draft_path = os.path.join(model_path, "1", "eagle3_draft")

        # Download checkpoints if configured
        builder = ModelBuilder(model_path, download_validation_only=True)
        config = builder.config
        stage = config["checkpoints"]["when"]
        checkpoints = config["checkpoints"]["repo_id"]
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        server_args = {
            'checkpoints': checkpoints,
            'tp_size': 1,
            'mem_fraction_static': 0.75,
            'port': 23333,
            'host': '0.0.0.0',
            'additional_list_args': [
                "--dtype", "bfloat16",
                "--speculative-algorithm", "EAGLE3",
                "--speculative-draft-model-path", draft_path,
                "--speculative-num-steps", "3",
                "--speculative-eagle-topk", "1",
                "--speculative-num-draft-tokens", "4",
            ],
        }

        self.server = OpenAI_APIServer.from_sglang_backend(**server_args)

        self.base_url = f"http://{self.server.host}:{self.server.port}"
        self.client = OpenAI(
            api_key="notset",
            base_url=f"{self.base_url}/v1",
            timeout=1800.0,
        )
        self.model = self.client.models.list().data[0].id
        logger.info(f"Eagle3 model loaded: {self.model}")

    def handle_readiness_probe(self) -> bool:
        """Handle readiness probe by checking SGLang health endpoint."""
        try:
            res = requests.get(f"{self.base_url}/health", timeout=10)
            return res.status_code == 200
        except Exception as e:
            logger.error(f"Readiness probe failed: {e}")
            return False

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> str:
        """Return a single completion."""
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self._set_usage(response)
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> Iterator[str]:
        """Stream a completion response."""
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stream_options={"include_usage": True},
        ):
            self._set_usage(chunk)
            if chunk.choices:
                text = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ''
                yield text
            else:
                yield ''
