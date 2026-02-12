import logging
import os
import sys
import subprocess
import inspect
import shutil
from typing import List, Iterator
from pathlib import Path

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger

try:
    from .benchmark_model_helper import benchmark_and_update_config
    from .model_export_helper import export_and_upload_lora_model
except ImportError:
    _dir = Path(__file__).parent
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
    from benchmark_model_helper import benchmark_and_update_config
    from model_export_helper import export_and_upload_lora_model

logging.basicConfig(level=logging.INFO)

PYTHON_EXEC = sys.executable

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

CONVERSATION_COLUMN = "conversations"
TEXT_COLUMN = "text"
TRAIN_SPLIT_RATIO = 0.9
PREPROCESSING_WORKERS = 2


def vllm_openai_server(checkpoints, **kwargs):
    """Start vLLM OpenAI compatible server.

    Matches production pattern from vllm-gemma-3-4b-it/1/model.py.
    lora_modules kwarg is handled specially: --lora-modules name=path [name=path ...]
    """
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
            # --lora-modules accepts name=path [name=path ...]
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
    """LoRA fine-tuning with Unsloth + vLLM serving, integrated with Clarifai platform."""

    client = True
    model = True

    LORA_MODEL_ID = "lora-adapter"

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
        return str

    @classmethod
    def to_pipeline_parser(cls):
        import argparse
        parser = argparse.ArgumentParser(description="Train a Clarifai LoRA model")
        sig = inspect.signature(cls.train)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser
    
    def train(self,
              user_id: str = "christine_yu",
              app_id: str = "test_lora_pipeline_app",
              model_id: str = "test_model",
              base_model_name: str = "unsloth/Qwen3-32B",
              dataset_name: str = "mlabonne/FineTome-100k",
              max_seq_length: int = 2048,
              load_in_4bit: bool = True,
              lora_r: int = 16,
              lora_alpha: int = 16,
              lora_dropout: float = 0.0,
              num_epochs: int = 1,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              lr_scheduler_type: str = "cosine",
              warmup_ratio: float = 0.06,
              weight_decay: float = 0.01,
              max_steps: int = -1,
              logging_steps: int = 10,
              save_steps: int = 100,
              seed: int = 105,
              num_gpus: int = 1,
              ) -> str:
        # Disable torch.compile/dynamo - cut_cross_entropy's backward pass uses
        # torch.compile which can fail on certain platforms (e.g. aarch64 GH200).
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        work_dir = "/tmp/lora_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        # STEP 1: Load base model with Unsloth
        # Source: unsloth_finetune.py load_or_cache_model()
        logging.info("=" * 80)
        logging.info("STEP 1: Loading base model with Unsloth")
        logging.info("=" * 80)

        # Unsloth must be imported first to apply patches to transformers/peft
        import unsloth  # noqa: F401,I001
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import standardize_sharegpt

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # STEP 2: Load and format dataset from HuggingFace
        # Source: unsloth_finetune.py load_or_cache_dataset() + format_chat_template()
        logging.info("=" * 80)
        logging.info("STEP 2: Loading and formatting dataset")
        logging.info("=" * 80)

        import datasets as hf_datasets

        dataset = hf_datasets.load_dataset(dataset_name, split="train")
        dataset = standardize_sharegpt(dataset)
        dataset = dataset.train_test_split(test_size=1.0 - TRAIN_SPLIT_RATIO, seed=seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        def format_chat_template(examples):
            texts = [
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for conv in examples[CONVERSATION_COLUMN]
            ]
            return {TEXT_COLUMN: texts}

        train_dataset = train_dataset.map(
            format_chat_template, batched=True,
            num_proc=PREPROCESSING_WORKERS, remove_columns=train_dataset.column_names,
        )
        eval_dataset = eval_dataset.map(
            format_chat_template, batched=True,
            num_proc=PREPROCESSING_WORKERS, remove_columns=eval_dataset.column_names,
        )
        logging.info(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

        # STEP 3: Configure LoRA adapters
        # Source: unsloth_finetune.py setup_model_for_training()
        logging.info("=" * 80)
        logging.info("STEP 3: Configuring LoRA adapters")
        logging.info("=" * 80)

        model = FastLanguageModel.get_peft_model(
            model, r=lora_r, target_modules=LORA_TARGET_MODULES,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", use_gradient_checkpointing="unsloth",
            random_state=seed, use_rslora=False, loftq_config=None,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # STEP 4: Training with SFTTrainer
        # Source: unsloth_finetune.py create_training_arguments() + trainer init
        logging.info("=" * 80)
        logging.info("STEP 4: Training")
        logging.info("=" * 80)

        from transformers import TrainingArguments
        from trl import SFTTrainer

        checkpoint_path = os.path.join(work_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            save_strategy="steps",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            output_dir=checkpoint_path,
            report_to=None,
            seed=seed,
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            dataset_text_field=TEXT_COLUMN, max_seq_length=max_seq_length,
            dataset_num_proc=PREPROCESSING_WORKERS, packing=False,
            args=training_args,
        )
        trainer.train()
        logging.info("Training completed")

        # STEP 5: Save LoRA adapters
        # Source: unsloth_finetune.py L349-352
        logging.info("=" * 80)
        logging.info("STEP 5: Saving LoRA adapters")
        logging.info("=" * 80)

        adapter_path = os.path.join(work_dir, "lora_adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logging.info(f"LoRA adapter saved to {adapter_path}")

        del trainer, model
        torch.cuda.empty_cache()

        # STEP 6: Benchmark Model and Update Config
        logging.info("=" * 80)
        logging.info("STEP 6: Benchmarking Model for GPU Requirements")
        logging.info("=" * 80)

        model_template_dir = Path(__file__).parent.parent
        config_yaml_path = model_template_dir / "config.yaml"

        if config_yaml_path.exists():
            benchmark_and_update_config(
                base_model_name=base_model_name,
                adapter_path=adapter_path,
                config_yaml_path=str(config_yaml_path),
                max_model_len=min(max_seq_length, 4096),
                save_benchmark_json=os.path.join(work_dir, "benchmark.json"),
            )

        # STEP 7: Export and Upload Model to Clarifai
        logging.info("=" * 80)
        logging.info("STEP 7: Exporting and Uploading Model to Clarifai")
        logging.info("=" * 80)

        export_and_upload_lora_model(
            adapter_path=adapter_path,
            base_model_name=base_model_name,
            source_model_dir=model_template_dir,
            clarifai_pat=pat,
            clarifai_api_base=os.getenv("CLARIFAI_API_BASE", "https://api.clarifai.com"),
            user_id=user_id, app_id=app_id, model_id=model_id,
        )

        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        return adapter_path

    def load_model(self):
        """Load the fine-tuned model and start the vLLM server with LoRA adapter.

        Follows production pattern from vllm-gemma-3-4b-it/1/model.py:
        - lora_modules as list of name=path
        - max_lora_rank matching training rank
        - self.model = adapter name (routes requests to LoRA adapter)
        """
        from openai import OpenAI
        from clarifai.runners.models.model_builder import ModelBuilder

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        checkpoints_config = model_config.get("checkpoints", {})
        base_model = checkpoints_config.get("repo_id")

        # Download checkpoints (adapter files)
        stage = checkpoints_config.get("when", "runtime")
        adapter_path = None
        if stage in ["build", "runtime"]:
            adapter_path = builder.download_checkpoints(stage=stage)

        server_args = {
            'gpu_memory_utilization': 0.9,
            'kv_cache_dtype': 'auto',
            'tensor_parallel_size': 1,
            'port': 23333,
            'host': 'localhost',
            'trust_remote_code': True,
            'enable_lora': True,
            'max_lora_rank': 64,  # accommodate up to rank 64 adapters
        }

        if adapter_path:
            server_args['lora_modules'] = [f"{self.LORA_MODEL_ID}={adapter_path}"]

        self.server = vllm_openai_server(base_model, **server_args)

        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1'
        )
        # Use LoRA adapter name as model id so requests route to the adapter
        self.model = self.LORA_MODEL_ID
        logger.info(f"vLLM model loaded with LoRA adapter: {self.model}")

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="Maximum tokens to generate."),
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
                 max_tokens: int = Param(default=512, description="Maximum tokens to generate."),
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
