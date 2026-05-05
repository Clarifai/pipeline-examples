"""Upload helper for Eagle3 model — copies to temp dir before upload
to avoid mutating the original config.yaml.

Adapted from pipeline-examples lora model_export_helper.py pattern.

Two operations:
  - upload_model_version: copy model dir to temp, override config.yaml,
    then call ModelBuilder.upload_model_version().
  - upload_checkpoint_to_artifact: create artifact (if missing) and upload
    a single file as an artifact version.
"""
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import yaml

from clarifai.utils.logging import logger


def upload_model_version(model_dir: str, user_id: str, app_id: str, model_id: str):
    """Copy model dir to temp, set user/app/model IDs, then upload.

    Original files are not modified.
    """
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="eagle3_upload_"))
        temp_model_dir = temp_dir / "model"

        logger.info(f"Creating model package in: {temp_model_dir}")
        shutil.copytree(
            model_dir, temp_model_dir,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )

        # Update config.yaml in the temp copy
        config_path = temp_model_dir / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config.setdefault("model", {})
        config["model"]["user_id"] = user_id
        config["model"]["app_id"] = app_id
        config["model"]["id"] = model_id
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"config.yaml -> user_id={user_id} app_id={app_id} id={model_id}")

        # Upload
        from clarifai.runners.models.model_builder import ModelBuilder

        builder = ModelBuilder(str(temp_model_dir), app_not_found_action="prompt")

        exists = builder.check_model_exists()
        if exists:
            logger.info(f"Model exists at {builder.model_ui_url}, creating new version.")
        else:
            logger.info(f"New model at {builder.model_ui_url}")

        builder.upload_model_version()
        logger.info(f"Uploaded model version: {builder.model_ui_url}")
        logger.info(f"Model Version ID: {builder.model_version_id}")

    except Exception as e:
        logger.error(f"Error during model upload: {e}")
        traceback.print_exc()
        raise


def upload_checkpoint_to_artifact(checkpoint_path: str, user_id: str, app_id: str, model_id: str):
    from clarifai.client.artifact import Artifact
    from clarifai.client.artifact_version import ArtifactVersion

    artifact_id = f"{model_id}_checkpoint"
    if not any(a.id == artifact_id for a in Artifact().list(user_id=user_id, app_id=app_id)):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
    version = ArtifactVersion().upload(
        file_path=str(checkpoint_path),
        artifact_id=artifact_id,
        user_id=user_id,
        app_id=app_id,
        visibility="private",
    )
    logger.info(f"Uploaded artifact version: {version.id}")
