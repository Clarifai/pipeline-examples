import logging
import os
import sys
import subprocess
import time
import inspect
import json
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
    model_dir = Path(__file__).parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from benchmark_model_helper import benchmark_and_update_config
    from model_export_helper import export_and_upload_lora_model

logging.basicConfig(level=logging.INFO)

PYTHON_EXEC = sys.executable

# LoRA target modules - all linear layers for comprehensive fine-tuning
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Dataset constants
CONVERSATION_COLUMN = "conversations"
TEXT_COLUMN = "text"
TRAIN_SPLIT_RATIO = 0.9
PREPROCESSING_WORKERS = 2


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
        logger.info("Waiting for " + f"http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info("Server started successfully at " + f"http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start vllm server: {str(e)}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {str(e)}")

    return server


class UnslothLoRAVLLM(OpenAIModelClass):
    """LoRA fine-tuning with Unsloth + vLLM serving, integrated with Clarifai platform."""

    client = True
    model = True

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
        else:
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

    def to_pipeline(self, pipeline_folder_path):
        step_name = "model-version-train-ps"
        model_dir = Path(__file__).parent.parent
        pipeline_path = Path(pipeline_folder_path)
        step_path = pipeline_path / step_name

        pipeline_path.mkdir(parents=True, exist_ok=True)
        (step_path / "1").mkdir(parents=True, exist_ok=True)

        sig = inspect.signature(self.train)
        params_info = {}
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                if param.default != inspect.Parameter.empty:
                    params_info[param_name] = param.default
                else:
                    params_info[param_name] = ""

        with open(model_dir / "1" / "pipe_config.yaml") as f:
            parent_config = f.read()

        param_lines = []
        step_param_lines = []

        for param_name, default_val in params_info.items():
            if default_val == "":
                val_str = '""'
            elif isinstance(default_val, str):
                val_str = f'"{default_val}"' if not default_val.startswith('[') else f"'{default_val}'"
            elif isinstance(default_val, bool):
                val_str = str(default_val).lower()
            else:
                val_str = str(default_val)

            param_lines.append(f"            - name: {param_name}\n              value: {val_str}")
            step_param_lines.append(f'                  - name: {param_name}\n                    value: "{{{{workflow.parameters.{param_name}}}}}"')

        parent_config += "\n".join(param_lines)
        parent_config += "\n        templates:\n        - name: sequence\n          steps:\n"
        parent_config += f"          - - name: {step_name}-name\n"
        parent_config += f"              templateRef:\n"
        parent_config += f"                name: users/YOUR_USER_ID/apps/YOUR_APP_ID/pipeline_steps/{step_name}\n"
        parent_config += f"                template: users/YOUR_USER_ID/apps/YOUR_APP_ID/pipeline_steps/{step_name}\n"
        parent_config += "              arguments:\n                parameters:\n"
        parent_config += "\n".join(step_param_lines) + "\n"

        with open(pipeline_path / "config.yaml", "w") as f:
            f.write(parent_config)

        import yaml
        with open(model_dir / "1" / "pipe_step_config.yaml") as f:
            step_config = yaml.safe_load(f)

        step_config.pop('pipeline_step_input_params', None)

        with open(step_path / "config.yaml", "w") as f:
            yaml.dump(step_config, f, default_flow_style=False, sort_keys=False)
            f.write("\npipeline_step_input_params:\n")
            for param_name in params_info:
                f.write(f"  - name: {param_name}\n")
                f.write(f"    description: \"\"\n")

        shutil.copy(model_dir / "train_Dockerfile", step_path / "Dockerfile")
        shutil.copy(model_dir / "train_requirements.txt", step_path / "requirements.txt")
        shutil.copy(model_dir / "1" / "pipeline_step.py", step_path / "1" / "pipeline_step.py")

        model_copy_path = step_path / "1" / "models" / "model"
        shutil.copytree(model_dir, model_copy_path)

        if (model_copy_path / "Dockerfile").exists():
            (model_copy_path / "Dockerfile").rename(model_copy_path / "Dockerfil")
        if (model_copy_path / "requirements.txt").exists():
            (model_copy_path / "requirements.txt").rename(model_copy_path / "requiremen.txt")

        logger.info(f"Pipeline structure created at {pipeline_path}")

    def train(self,
              # Resource IDs
              user_id: str = "YOUR_USER_ID",
              app_id: str = "YOUR_APP_ID",
              model_id: str = "test_model",
              # Model configuration
              base_model_name: str = "unsloth/Qwen3-32B",
              dataset_name: str = "mlabonne/FineTome-100k",
              max_seq_length: int = 2048,
              load_in_4bit: bool = True,
              # LoRA configuration
              lora_r: int = 16,
              lora_alpha: int = 16,
              lora_dropout: float = 0.0,
              # Training hyperparameters
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
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        work_dir = "/tmp/lora_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        logging.info("Starting Unsloth LoRA fine-tuning pipeline")

        # =====================================================================
        # STEP 1: Load base model with Unsloth
        # Source: modal_unsloth/unsloth_finetune.py load_or_cache_model()
        # =====================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 1: Loading base model with Unsloth")
        logging.info("=" * 80)

        # Unsloth must be imported first to apply patches to transformers/peft
        import unsloth  # noqa: F401,I001
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import standardize_sharegpt

        logging.info(f"Loading model: {base_model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        logging.info("Base model loaded successfully")

        # =====================================================================
        # STEP 2: Load and format dataset from HuggingFace
        # Source: unsloth_finetune.py load_or_cache_dataset() + format_chat_template()
        # =====================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 2: Loading and formatting dataset")
        logging.info("=" * 80)

        import datasets as hf_datasets

        logging.info(f"Downloading dataset: {dataset_name}")
        dataset = hf_datasets.load_dataset(dataset_name, split="train")
        dataset = standardize_sharegpt(dataset)

        # Split into train/eval
        dataset = dataset.train_test_split(
            test_size=1.0 - TRAIN_SPLIT_RATIO, seed=seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Apply chat template formatting
        logging.info("Formatting datasets with chat template...")

        def format_chat_template(examples):
            texts = []
            for conversation in examples[CONVERSATION_COLUMN]:
                formatted_text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
                texts.append(formatted_text)
            return {TEXT_COLUMN: texts}

        train_dataset = train_dataset.map(
            format_chat_template,
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )
        eval_dataset = eval_dataset.map(
            format_chat_template,
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

        logging.info(f"Training dataset size: {len(train_dataset):,}")
        logging.info(f"Evaluation dataset size: {len(eval_dataset):,}")

        # =====================================================================
        # STEP 3: Configure LoRA adapters
        # Source: unsloth_finetune.py setup_model_for_training()
        # =====================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 3: Configuring LoRA adapters")
        logging.info("=" * 80)

        logging.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

        # =====================================================================
        # STEP 4: Training with SFTTrainer
        # Source: unsloth_finetune.py create_training_arguments() + trainer init
        # =====================================================================
        logging.info("")
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

        logging.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=TEXT_COLUMN,
            max_seq_length=max_seq_length,
            dataset_num_proc=PREPROCESSING_WORKERS,
            packing=False,
            args=training_args,
        )

        logging.info("Starting training...")
        trainer.train()
        logging.info("Training completed")

        # =====================================================================
        # STEP 5: Save LoRA adapters
        # Source: unsloth_finetune.py L349-352
        # =====================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 5: Saving LoRA adapters")
        logging.info("=" * 80)

        adapter_path = os.path.join(work_dir, "lora_adapter")
        logging.info(f"Saving LoRA adapter to: {adapter_path}")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logging.info("LoRA adapter saved successfully")

        # List saved files
        for f in os.listdir(adapter_path):
            fpath = os.path.join(adapter_path, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            logging.info(f"  {f}: {size_mb:.1f} MB")

        # Clear GPU memory after training
        del trainer
        del model
        torch.cuda.empty_cache()
        logging.info("GPU memory cleared after training")

        # =====================================================================
        # STEP 6: Benchmark Model and Update Config
        # Source: classifier model.py benchmark call pattern
        # =====================================================================
        logging.info("")
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
            logging.info("Benchmark complete and config.yaml updated")
        else:
            logging.warning(f"config.yaml not found at {config_yaml_path}, skipping benchmark")

        # =====================================================================
        # STEP 7: Export and Upload Model to Clarifai
        # Source: classifier model.py export call pattern
        # =====================================================================
        logging.info("")
        logging.info("=" * 80)
        logging.info("STEP 7: Exporting and Uploading Model to Clarifai")
        logging.info("=" * 80)

        clarifai_api_base = os.getenv("CLARIFAI_API_BASE", "https://api.clarifai.com")

        export_and_upload_lora_model(
            adapter_path=adapter_path,
            base_model_name=base_model_name,
            source_model_dir=model_template_dir,
            clarifai_pat=pat,
            clarifai_api_base=clarifai_api_base,
            user_id=user_id,
            app_id=app_id,
            model_id=model_id,
        )

        logging.info("")
        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"LoRA adapter: {adapter_path}")

        return adapter_path

    def load_model(self):
        """Load the fine-tuned model and start the vLLM server with LoRA adapter."""
        from openai import OpenAI
        from clarifai.runners.models.model_builder import ModelBuilder
        import yaml

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        # Get base model repo_id and adapter path from config
        checkpoints_config = model_config.get("checkpoints", {})
        base_model = checkpoints_config.get("repo_id")
        adapter_name = model_config.get("model", {}).get("id", "lora-adapter")

        # Download checkpoints (adapter files) if configured
        stage = checkpoints_config.get("when", "runtime")
        adapter_path = None
        if stage in ["build", "runtime"]:
            adapter_path = builder.download_checkpoints(stage=stage)

        server_args = {
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.9,
            'dtype': 'auto',
            'kv_cache_dtype': 'auto',
            'tensor_parallel_size': 1,
            'port': '23333',
            'host': 'localhost',
            'trust_remote_code': True,
            'enable_lora': True,
        }

        # If we have a local adapter path, configure LoRA modules
        if adapter_path:
            server_args['lora_modules'] = f"{adapter_name}={adapter_path}"

        self.server = vllm_openai_server(base_model, **server_args)

        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1'
        )
        self.model = self.client.models.list().data[0].id
        logger.info(f"vLLM model loaded with LoRA adapter, model id: {self.model}")

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="The maximum number of tokens to generate."),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response."),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature."),
                ) -> str:
        """Predict response for given prompt and chat history."""
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
    def generate(self,
                 prompt: str,
                 chat_history: List[dict] = None,
                 max_tokens: int = Param(default=512, description="The maximum number of tokens to generate."),
                 temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response."),
                 top_p: float = Param(default=0.95, description="An alternative to sampling with temperature."),
                 ) -> Iterator[str]:
        """Stream generated text tokens from a prompt + optional chat history."""
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in response:
            self._set_usage(chunk)
            if chunk.choices:
                text = (chunk.choices[0].delta.content
                        if (chunk and chunk.choices[0].delta.content) is not None else '')
                yield text
            else:
                yield ""

    def test(self):
        """Test the model locally."""
        try:
            print("Testing predict...")
            print(self.predict(prompt="Hello, how are you?"))
        except Exception as e:
            print("Error in predict", e)

        try:
            print("Testing generate...")
            for each in self.generate(prompt="Hello, how are you?"):
                print(each, end="")
            print()
        except Exception as e:
            print("Error in generate", e)


if __name__ == "__main__":
    m = UnslothLoRAVLLM()
    m.load_model()
    m.test()
