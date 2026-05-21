#!/usr/bin/env python3
"""Clarifai pipeline DSL definition for the LoRA fine-tuning pipeline.

This file is a DSL equivalent of model-version-train-ps/1/pipeline_step.py.
It defines a single pipeline step that:
  - loads the UnslothLoRAVLLM model class from the bundled assets
  - calls its train() method with all hyperparameters passed as pipeline inputs
  - uploads the resulting LoRA adapter to the Clarifai platform

Usage:

  # Generate a uploadable pipeline bundle:
  python dsl_def.py --generate ./generated-pipeline

  # Or upload directly:
  clarifai pipeline upload dsl_def.py
"""

import argparse
import sys
from pathlib import Path

from clarifai.runners.pipelines import ComputeInfo, Pipeline, step


def _load_train_class():
    """Load UnslothLoRAVLLM from the asset-bundled models directory.

    Mirrors the dynamic import in model-version-train-ps/1/pipeline_step.py so
    the same logic works both locally and inside the generated step container.
    """
    models_dir = str(Path(__file__).parent / "models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    model_module = __import__("model.1.model", fromlist=[""])
    return next(
        obj
        for name in dir(model_module)
        if isinstance(obj := getattr(model_module, name), type)
        and hasattr(obj, "train")
    )


@step(
    id="model-version-train-ps",
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    platform="linux/amd64",
    requirements=[
        "clarifai>=12.3",
        "clarifai-grpc>=12.0.2",
        "torch<2.6.0",
        "torchvision<2.6.0",
        "torchaudio<2.6.0",
        "boto3==1.42.40",
        "pynvml==13.0.1",
        "PyYAML==6.0.3",
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub",
        "peft==0.16.0",
        "transformers<5.0.0",
        "tokenizers==0.21.4",
        "trl==0.19.1",
        "unsloth==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "torchao==0.9.0",
        "bitsandbytes==0.49.1",
        "cut-cross-entropy==25.1.1",
        "openai==2.15.0",
    ],
    # Bundle the full model package so the step container has all source code
    # (model.py, model_export_helper.py, config.yaml, Dockerfile, downloader, …)
    assets=["./model-version-train-ps/1/models"],
    compute=ComputeInfo(
        cpu_limit="8",
        cpu_memory="10Gi",
        num_accelerators=1,
        accelerator_memory="10Gi",
        accelerator_type=["NVIDIA-*"],
    ),
)
def train_lora(
    user_id: str,
    app_id: str,
    model_id: str,
    base_model_name: str = "unsloth/Qwen3-0.6B",
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
) -> str:
    """Fine-tune a base model with Unsloth LoRA and upload it to Clarifai."""
    model_class = _load_train_class()
    return model_class().train(
        user_id=user_id,
        app_id=app_id,
        model_id=model_id,
        base_model_name=base_model_name,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
    )


with Pipeline(
    id="lora-train-pipeline",
) as pipeline:
    result = train_lora(
        user_id=pipeline.input("user_id", pipeline.user_id),
        app_id=pipeline.input("app_id", pipeline.app_id),
        model_id=pipeline.input("model_id"),
        base_model_name=pipeline.input("base_model_name", default="unsloth/Qwen3-0.6B"),
        dataset_name=pipeline.input("dataset_name", default="mlabonne/FineTome-100k"),
        max_seq_length=pipeline.input("max_seq_length", default=2048),
        load_in_4bit=pipeline.input("load_in_4bit", default=True),
        lora_r=pipeline.input("lora_r", default=16),
        lora_alpha=pipeline.input("lora_alpha", default=16),
        lora_dropout=pipeline.input("lora_dropout", default=0.0),
        num_epochs=pipeline.input("num_epochs", default=1),
        batch_size=pipeline.input("batch_size", default=4),
        gradient_accumulation_steps=pipeline.input("gradient_accumulation_steps", default=4),
        learning_rate=pipeline.input("learning_rate", default=2e-4),
        lr_scheduler_type=pipeline.input("lr_scheduler_type", default="cosine"),
        warmup_ratio=pipeline.input("warmup_ratio", default=0.06),
        weight_decay=pipeline.input("weight_decay", default=0.01),
        max_steps=pipeline.input("max_steps", default=-1),
        logging_steps=pipeline.input("logging_steps", default=10),
        save_steps=pipeline.input("save_steps", default=100),
        seed=pipeline.input("seed", default=105),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Clarifai pipeline bundle for LoRA fine-tuning."
    )
    parser.add_argument(
        "--generate",
        default="./generated-pipeline",
        help="Directory where the generated pipeline bundle should be written.",
    )
    args = parser.parse_args()

    config_path = pipeline.generate(args.generate)
    print(f"Generated pipeline bundle at: {config_path}")


if __name__ == "__main__":
    main()
