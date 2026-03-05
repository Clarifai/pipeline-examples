import os
import shutil
import tempfile
import traceback
import yaml
from pathlib import Path
from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion

try:
    from clarifai.utils.logging import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def upload_checkpoint_to_artifact(checkpoint_path, user_id, app_id, model_id):
    artifact_id = f"{model_id}_checkpoint"
    artifacts = Artifact().list(user_id=user_id, app_id=app_id)
    if not any(a.id == artifact_id for a in artifacts):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
    version = ArtifactVersion().upload(file_path=str(checkpoint_path), artifact_id=artifact_id, user_id=user_id, app_id=app_id, visibility="private")
    logger.info(f"📦 Artifact version: {version.id}")


def export_and_upload_detector(
    weights_path: str,
    config_py_path: str,
    classes: list,
    source_model_dir: Path,
    clarifai_pat: str,
    clarifai_api_base: str,
    user_id: str = None,
    app_id: str = None,
    model_id: str = None,
):
    logger.info("=" * 80)
    logger.info("MODEL EXPORT: Starting detector model export and upload")
    logger.info("=" * 80)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.error(f"❌ Weights file not found: {weights_path}")
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"✅ Found trained weights: {weights_path}")
    logger.info(f"📋 Number of classes: {len(classes)}")
    logger.info(f"📋 Classes: {classes[:10]}{'...' if len(classes) > 10 else ''}")

    logger.info("📦 Preparing model package for upload...")
    copy_model_files_and_upload(
        source_model_dir=source_model_dir,
        weights_path=weights_path,
        config_py_path=config_py_path,
        classes=classes,
        clarifai_pat=clarifai_pat,
        clarifai_api_base=clarifai_api_base,
        user_id=user_id,
        app_id=app_id,
        model_id=model_id,
    )


def copy_model_files_and_upload(
    source_model_dir: Path,
    weights_path: Path,
    config_py_path: str,
    classes: list,
    clarifai_pat: str,
    clarifai_api_base: str,
    user_id: str = None,
    app_id: str = None,
    model_id: str = None,
):
    try:
        temp_dir = tempfile.mkdtemp()
        temp_dir = Path("trained_model_temp")
        model_dir = Path(temp_dir) / "detector_model"

        logger.info(f"📁 Creating model package in: {model_dir}")

        if model_dir.exists():
            logger.warning(f"⚠️  Directory already exists from previous run: {model_dir}")
            logger.warning("⚠️  Removing existing directory before creating new one...")
            shutil.rmtree(model_dir)
            logger.info("✅ Existing directory removed")

        shutil.copytree(source_model_dir, model_dir)

        model_files_path = model_dir / "1" / "model_files"
        model_files_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = model_files_path / "checkpoint.pth"
        logger.info(f"📦 Copying weights to: {checkpoint_path}")
        shutil.copy2(weights_path, checkpoint_path)
        logger.info("✅ Weights copied successfully")

        upload_checkpoint_to_artifact(checkpoint_path, user_id, app_id, model_id)

        config_dest = model_files_path / "config.py"
        logger.info(f"📦 Copying config.py to: {config_dest}")
        shutil.copy2(config_py_path, config_dest)
        logger.info("✅ Config.py copied successfully")

        config_path = model_dir / "config.yaml"
        logger.info(f"📝 Updating config.yaml with {len(classes)} classes")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['concepts'] = [
            {'id': str(i), 'name': class_name}
            for i, class_name in enumerate(classes)
        ]

        if user_id:
            config['model']['user_id'] = user_id
            logger.info(f"📝 Set user_id: {user_id}")
        if app_id:
            config['model']['app_id'] = app_id
            logger.info(f"📝 Set app_id: {app_id}")
        if model_id:
            config['model']['id'] = model_id
            logger.info(f"📝 Set model_id: {model_id}")

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("✅ Config updated successfully")

        logger.info("=" * 80)
        logger.info("📤 Uploading model to Clarifai platform...")
        logger.info("=" * 80)

        from clarifai.runners.models.model_builder import ModelBuilder

        logger.info("📦 Initializing ModelBuilder...")
        builder = ModelBuilder(str(model_dir), app_not_found_action="prompt")

        exists = builder.check_model_exists()
        if exists:
            logger.info(
                f"Model already exists at {builder.model_ui_url}, "
                "this upload will create a new version for it."
            )
        else:
            logger.info(
                f"New model will be created at {builder.model_ui_url} "
                "with its first version."
            )

        logger.info("📤 Uploading model version...")
        builder.upload_model_version()

        logger.info("-" * 80)
        logger.info("✅ Model upload completed successfully!")
        logger.info(f"📍 Model URL: {builder.model_ui_url}")
        logger.info(f"🆔 Model Version ID: {builder.model_version_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Error during model export: {e}")
        traceback.print_exc()
        raise
