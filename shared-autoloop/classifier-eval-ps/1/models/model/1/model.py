"""Classifier evaluation pipeline step.

Downloads the trained checkpoint from artifact store, runs MMPretrain validation
on the dataset, and outputs eval metrics as Argo output parameters.
"""
import json
import inspect
import logging
import os
from pathlib import Path

import torch
from clarifai.client.artifact_version import ArtifactVersion
from mmpretrain import ImageClassificationInferencer

logging.basicConfig(level=logging.INFO)


class ClassifierEvaluator:

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
        parser = argparse.ArgumentParser(description="Evaluate a ResNet classifier model")
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
        dataset_id: str = "",
        dataset_version_id: str = "",
        concepts: str = '["beignets","hamburger","prime_rib","ramen"]',
        image_size: int = 224,
        batch_size: int = 64,
    ) -> str:
        """Evaluate classifier checkpoint from artifact store on validation data.

        Returns path to eval_results JSON.
        """
        pat = os.getenv("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")

        concepts_list = json.loads(concepts)
        work_dir = "/tmp/classifier_eval_work_dir"
        os.makedirs(work_dir, exist_ok=True)

        # ================================================================
        # STEP 1: Download checkpoint from artifact store
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 1: Downloading Checkpoint from Artifact Store")
        logging.info("=" * 80)

        artifact_id = f"{model_id}_checkpoint"
        checkpoint_path = os.path.join(work_dir, "checkpoint.pth")

        version = ArtifactVersion()
        checkpoint_root = version.download(
            artifact_id=artifact_id,
            user_id=user_id,
            app_id=app_id,
            output_path=checkpoint_path,
            force=True,
        )
        logging.info(f"Downloaded checkpoint to {checkpoint_root}")

        # Also try to download the config.py (uploaded as second artifact version)
        config_path = os.path.join(work_dir, "config.py")
        try:
            version.download(
                artifact_id=artifact_id,
                user_id=user_id,
                app_id=app_id,
                output_path=config_path,
                force=True,
            )
            logging.info(f"Downloaded config to {config_path}")
        except Exception:
            config_path = None
            logging.warning("Could not download config.py from artifact, will use default")

        # ================================================================
        # STEP 2: Download evaluation dataset
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 2: Downloading Evaluation Dataset")
        logging.info("=" * 80)

        # Import dataset helpers (same as train step uses)
        from dataset_helpers import (
            download_dataset,
            convert_dataset_to_imagenet_format,
            create_classes_file,
        )

        dataset_name = download_dataset(
            user_id=user_id,
            app_id=app_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            pat=pat,
            output_dir=work_dir,
            concepts=concepts_list,
        )

        convert_output = convert_dataset_to_imagenet_format(
            dataset_name=dataset_name,
            dataset_split="train",
            output_root=work_dir,
        )
        images_output_root = convert_output.images_output_root

        classes_path = create_classes_file(
            dataset_name=dataset_name,
            output_dir=images_output_root,
            concepts=None,
        )

        with open(classes_path, 'r') as f:
            dataset_classes = [line.strip() for line in f if line.strip()]
        num_classes = len(dataset_classes)
        logging.info(f"Using {num_classes} classes: {dataset_classes}")

        # ================================================================
        # STEP 3: Run inference on validation split
        # ================================================================
        logging.info("=" * 80)
        logging.info("STEP 3: Running Evaluation Inference")
        logging.info("=" * 80)

        # Build a minimal config for the inferencer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if config_path and os.path.exists(config_path):
            inferencer = ImageClassificationInferencer(
                model=config_path,
                pretrained=checkpoint_root,
                device=device,
            )
        else:
            inferencer = ImageClassificationInferencer(
                model='resnet50_8xb256-rsb-a1-600e_in1k',
                pretrained=checkpoint_root,
                device=device,
            )

        # Collect val images
        val_dir = os.path.join(images_output_root, "val")
        if not os.path.isdir(val_dir):
            val_dir = os.path.join(images_output_root, "train")
            logging.warning("No val split found, using train split for eval")

        image_paths = []
        labels = []
        for class_idx, class_name in enumerate(dataset_classes):
            class_dir = os.path.join(val_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)
                        labels.append(class_idx)

        logging.info(f"Found {len(image_paths)} images for evaluation")

        if not image_paths:
            logging.warning("No evaluation images found!")
            metrics = {"accuracy/top1": 0.0, "accuracy/top5": 0.0}
        else:
            # Run inference in batches
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            for batch_start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[batch_start:batch_start + batch_size]
                batch_labels = labels[batch_start:batch_start + batch_size]

                results = inferencer(batch_paths, batch_size=batch_size)

                for result, true_label in zip(results, batch_labels):
                    pred_scores = result.get('pred_scores', [])
                    if len(pred_scores) > 0:
                        top5_indices = sorted(
                            range(len(pred_scores)),
                            key=lambda i: pred_scores[i],
                            reverse=True
                        )[:5]
                        pred_label = top5_indices[0]
                        if pred_label == true_label:
                            correct_top1 += 1
                        if true_label in top5_indices:
                            correct_top5 += 1
                    total += 1

                processed = min(batch_start + batch_size, len(image_paths))
                logging.info(f"Processed {processed}/{len(image_paths)} images")

            metrics = {
                "accuracy/top1": correct_top1 / total if total > 0 else 0.0,
                "accuracy/top5": correct_top5 / total if total > 0 else 0.0,
            }

        # ================================================================
        # STEP 4: Write eval results
        # ================================================================
        logging.info("=" * 80)
        logging.info("EVALUATION RESULTS")
        logging.info("=" * 80)
        logging.info(f"  Top-1 Accuracy: {metrics.get('accuracy/top1', 0):.4f}")
        logging.info(f"  Top-5 Accuracy: {metrics.get('accuracy/top5', 0):.4f}")

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
