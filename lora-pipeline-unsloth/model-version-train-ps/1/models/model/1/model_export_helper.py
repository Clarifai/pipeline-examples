import os
import shutil
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
    version = ArtifactVersion().upload(
        file_path=str(checkpoint_path),
        artifact_id=artifact_id,
        user_id=user_id,
        app_id=app_id,
        visibility="private",
    )
    logger.info(f"Artifact version: {version.id}")


def export_and_upload_lora_model(
    adapter_path: str,
    base_model_name: str,
    source_model_dir: Path,
    clarifai_pat: str,
    clarifai_api_base: str,
    user_id: str = None,
    app_id: str = None,
    model_id: str = None,
):
    logger.info("=" * 80)
    logger.info("MODEL EXPORT: Starting LoRA model export and upload")
    logger.info("=" * 80)

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter directory not found: {adapter_path}")
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    # List adapter files
    adapter_files = list(adapter_path.iterdir())
    total_size_mb = sum(f.stat().st_size for f in adapter_files if f.is_file()) / 1024 / 1024
    logger.info(f"Found {len(adapter_files)} adapter files, total size: {total_size_mb:.1f} MB")

    logger.info("Preparing model package for upload...")
    copy_model_files_and_upload(
        source_model_dir=source_model_dir,
        adapter_path=adapter_path,
        base_model_name=base_model_name,
        clarifai_pat=clarifai_pat,
        clarifai_api_base=clarifai_api_base,
        user_id=user_id,
        app_id=app_id,
        model_id=model_id,
    )


def copy_model_files_and_upload(
    source_model_dir: Path,
    adapter_path: Path,
    base_model_name: str,
    clarifai_pat: str,
    clarifai_api_base: str,
    user_id: str = None,
    app_id: str = None,
    model_id: str = None,
):
    try:
        temp_dir = Path("trained_model_temp")
        model_dir = temp_dir / "lora_model"

        logger.info(f"Creating model package in: {model_dir}")

        if model_dir.exists():
            logger.warning(f"Directory already exists from previous run: {model_dir}")
            logger.warning("Removing existing directory before creating new one...")
            shutil.rmtree(model_dir)
            logger.info("Existing directory removed")

        shutil.copytree(source_model_dir, model_dir)

        # Handle naming conventions for Clarifai compatibility
        requirement_txt = model_dir / "requiremen.txt"
        requirements_txt = model_dir / "requirements.txt"
        if requirement_txt.exists() and not requirements_txt.exists():
            logger.info("Renaming requiremen.txt -> requirements.txt")
            requirement_txt.rename(requirements_txt)

        dockerfil = model_dir / "Dockerfil"
        dockerfile = model_dir / "Dockerfile"
        if dockerfil.exists() and not dockerfile.exists():
            logger.info("Renaming Dockerfil -> Dockerfile")
            dockerfil.rename(dockerfile)

        # Copy LoRA adapter files into checkpoints directory
        checkpoints_dir = model_dir / "1" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        adapter_dest = checkpoints_dir / "lora_adapter"
        logger.info(f"Copying LoRA adapter to: {adapter_dest}")
        shutil.copytree(adapter_path, adapter_dest)
        logger.info("LoRA adapter copied successfully")

        # Upload adapter as artifact
        # Create a tar of the adapter for artifact upload
        import tarfile
        adapter_tar_path = temp_dir / "lora_adapter.tar.gz"
        with tarfile.open(adapter_tar_path, "w:gz") as tar:
            tar.add(adapter_path, arcname="lora_adapter")
        upload_checkpoint_to_artifact(adapter_tar_path, user_id, app_id, model_id)

        # Update config.yaml with base model info and checkpoints
        config_path = model_dir / "config.yaml"
        logger.info("Updating config.yaml with base model and adapter info")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set checkpoints to point to the base model repo_id
        # The adapter will be loaded from the local checkpoints directory
        config["checkpoints"] = {
            "repo_id": base_model_name,
            "when": "runtime",
        }

        if user_id:
            config["model"]["user_id"] = user_id
            logger.info(f"Set user_id: {user_id}")
        if app_id:
            config["model"]["app_id"] = app_id
            logger.info(f"Set app_id: {app_id}")
        if model_id:
            config["model"]["id"] = model_id
            logger.info(f"Set model_id: {model_id}")

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("Config updated successfully")

        # Upload to Clarifai
        logger.info("=" * 80)
        logger.info("Uploading model to Clarifai platform...")
        logger.info("=" * 80)

        from clarifai.runners.models.model_builder import ModelBuilder

        logger.info("Initializing ModelBuilder...")
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

        logger.info("Uploading model version...")
        builder.upload_model_version()

        logger.info("-" * 80)
        logger.info("Model upload completed successfully!")
        logger.info(f"Model URL: {builder.model_ui_url}")
        logger.info(f"Model Version ID: {builder.model_version_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during model export: {e}")
        traceback.print_exc()
        raise
