"""LoRA evaluation pipeline step.

Downloads the trained LoRA adapter from artifact store, loads base model + adapter,
runs eval on a subset of the dataset, and outputs eval metrics.
"""
import json
import inspect
import logging
import os
import tarfile
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)

CONVERSATION_COLUMN = "conversations"
TEXT_COLUMN = "text"


class LoRAEvaluator:

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
        parser = argparse.ArgumentParser(description="Evaluate a LoRA fine-tuned model")
        sig = inspect.signature(cls.evaluate)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def evaluate(
        self,
        user_id: str = "clarifai",
        app_id: str = "train_pipelines",
        model_id: str = "test_model",
        base_model_name: str = "unsloth/Qwen3-0.6B",
        dataset_name: str = "mlabonne/FineTome-100k",
        max_seq_length: int = 2048,
        max_eval_samples: int = 200,
    ) -> str:
        """Evaluate LoRA adapter from artifact store on eval subset.

        Returns path to eval_results JSON.
        """
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        work_dir = "/tmp/lora_eval_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        # ================================================================
        # STEP 1: Download adapter from artifact store
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 1: Downloading LoRA Adapter from Artifact Store")
        logging.info("=" * 80)

        from clarifai.client.artifact_version import ArtifactVersion

        artifact_id = f"{model_id}_checkpoint"
        adapter_tar_path = os.path.join(work_dir, "lora_adapter.tar.gz")

        version = ArtifactVersion()
        downloaded_path = version.download(
            artifact_id=artifact_id,
            user_id=user_id,
            app_id=app_id,
            output_path=adapter_tar_path,
            force=True,
        )
        logging.info(f"Downloaded adapter archive to {downloaded_path}")

        # Extract adapter
        adapter_path = os.path.join(work_dir, "lora_adapter")
        if tarfile.is_tarfile(downloaded_path):
            with tarfile.open(downloaded_path, "r:gz") as tar:
                tar.extractall(work_dir, filter='data')
            logging.info(f"Extracted adapter to {adapter_path}")
        else:
            adapter_path = downloaded_path
            logging.info(f"Using adapter directly from {adapter_path}")

        # ================================================================
        # STEP 2: Load base model + adapter
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 2: Loading Base Model + LoRA Adapter")
        logging.info("=" * 80)

        # Pre-initialize CUDA context
        torch.zeros(1, device="cuda")
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        logging.info("Model + adapter loaded successfully")

        # ================================================================
        # STEP 3: Load eval dataset
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 3: Loading Evaluation Dataset")
        logging.info("=" * 80)

        import datasets as hf_datasets
        from unsloth.chat_templates import standardize_sharegpt

        dataset = hf_datasets.load_dataset(dataset_name, split="train")
        dataset = standardize_sharegpt(dataset)
        # Use last portion as eval
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        eval_dataset = dataset["test"]

        if len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logging.info(f"Eval samples: {len(eval_dataset)}")

        # Format dataset
        def format_chat_template(examples):
            texts = [
                tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                for conv in examples[CONVERSATION_COLUMN]
            ]
            return {TEXT_COLUMN: texts}

        eval_dataset = eval_dataset.map(
            format_chat_template, batched=True,
            num_proc=1, remove_columns=eval_dataset.column_names,
        )

        # ================================================================
        # STEP 4: Compute eval loss
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 4: Computing Eval Loss")
        logging.info("=" * 80)

        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        with torch.no_grad():
            for i, example in enumerate(eval_dataset):
                text = example[TEXT_COLUMN]
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=max_seq_length,
                ).to(model.device)

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].shape[1]

                total_loss += loss * num_tokens
                total_tokens += num_tokens
                batch_count += 1

                if (i + 1) % 50 == 0:
                    avg = total_loss / total_tokens if total_tokens > 0 else 0
                    logging.info(f"  Processed {i+1}/{len(eval_dataset)} samples, running avg loss: {avg:.4f}")

        avg_eval_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        import math
        perplexity = math.exp(min(avg_eval_loss, 100))  # Cap to avoid overflow

        metrics = {
            "eval_loss": avg_eval_loss,
            "perplexity": perplexity,
            "eval_samples": batch_count,
        }

        # ================================================================
        # STEP 5: Write eval results
        # ================================================================
        logging.info("=" * 80)
        logging.info("EVALUATION RESULTS")
        logging.info("=" * 80)
        logging.info(f"  Eval Loss:   {metrics['eval_loss']:.4f}")
        logging.info(f"  Perplexity:  {metrics['perplexity']:.4f}")
        logging.info(f"  Samples:     {metrics['eval_samples']}")

        eval_results = {"metrics": metrics}

        # Write Argo output parameters
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "eval_results"), 'w') as f:
            json.dump(eval_results, f)

        results_path = os.path.join(work_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logging.info(f"Results saved to: {results_path}")
        logging.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")

        return results_path
