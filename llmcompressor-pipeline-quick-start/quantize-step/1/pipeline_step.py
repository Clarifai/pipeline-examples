#!/usr/bin/env python3
"""LLM Compressor quantization pipeline orchestrator.

Drives scripts/quantize_llmcompressor.sh with the wrapper's defaults, then
runs a HuggingFace transformers smoke test on the produced checkpoint, then
optionally uploads the checkpoint as a Clarifai artifact.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Step directory holding scripts/quantize_llmcompressor.sh + worker.
STEP_DIR = Path(__file__).resolve().parent

# Default output dir: /workspace/outputs/quantized_model in container, otherwise next to step.
DEFAULT_OUT = "/workspace/outputs/quantized_model" if Path("/workspace").is_dir() \
    else str(STEP_DIR.parent / "outputs" / "quantized_model")


def run_quantization(src: str, scheme: str, out: str, num_gpus: str, pipeline: str) -> None:
    cmd = [
        "bash", str(STEP_DIR / "scripts" / "quantize_llmcompressor.sh"),
        "--src", src,
        "--out", out,
        "--scheme", scheme,
        "--num-gpus", num_gpus,
        "--pipeline", pipeline,
    ]
    print(f"\n>>> running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(STEP_DIR), check=True)


def sanity_check(out: str) -> None:
    print(f"\n>>> sanity check: load {out} and generate", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(out, dtype="auto", device_map="auto")
    tok = AutoTokenizer.from_pretrained(out)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that reverses a list."}],
        return_tensors="pt", add_generation_prompt=True,
    )
    import torch
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) or hasattr(inputs, "data") else inputs
    if not isinstance(input_ids, torch.Tensor):
        input_ids = inputs.input_ids
    input_ids = input_ids.to(m.device)
    out_ids = m.generate(input_ids, max_new_tokens=200, do_sample=False)
    print(tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True), flush=True)


def upload_artifact(out: str, artifact_id: str, user_id: str, app_id: str) -> None:
    """Zip the quantized checkpoint dir and upload as a Clarifai artifact version."""
    from clarifai.client.artifact import Artifact
    from clarifai.client.artifact_version import ArtifactVersion
    from clarifai.client.user import User

    archive_base = str(Path(out).parent / Path(out).name)
    archive = shutil.make_archive(archive_base, "zip", root_dir=out)
    print(f"\n>>> upload: {archive} -> user={user_id} app={app_id} artifact_id={artifact_id}",
          flush=True)

    user = User(user_id=user_id)
    try:
        user.app(app_id=app_id)
    except Exception as e:
        if "does not exist" in str(e).lower():
            print(f">>> app '{app_id}' not found, creating it", flush=True)
            user.create_app(app_id=app_id)
        else:
            raise

    existing = list(Artifact().list(user_id=user_id, app_id=app_id))
    if not any(a.id == artifact_id for a in existing):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)

    version = ArtifactVersion().upload(
        file_path=archive,
        artifact_id=artifact_id,
        user_id=user_id,
        app_id=app_id,
    )
    print(f">>> uploaded: artifact_id={artifact_id} version_id={version.id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",       default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF repo id or local path. "
                             "To try quantizing an MoE model, set to ibm-granite/granite-4.0-h-tiny "
                             "(or any other MoE variant with a shim under "
                             "llm-compressor/src/llmcompressor/modeling/).")
    parser.add_argument("--scheme",    default="W4A16",
                        help="Quantization scheme: W4A16, NVFP4, NVFP4A16, FP8, FP8_DYNAMIC, MXFP4, W8A8_INT8. "
                             "To try the NVFP4 W4A4 variant (e.g. paired with the MoE example above), "
                             "set to NVFP4 — or pick any other NVFP4 variant.")
    parser.add_argument("--out",       default=DEFAULT_OUT)
    parser.add_argument("--num_gpus",  default="1",
                        help="GPUs auto-picked by lowest memory.used.")
    parser.add_argument("--pipeline",  default="datafree",
                        help="datafree (weight-only schemes), basic, sequential. "
                             "When pairing with NVFP4 (or any scheme that needs activation calibration), "
                             "set to '' (empty) and llm-compressor auto-infers sequential.")
    parser.add_argument("--user_id",     default=os.environ.get("CLARIFAI_USER_ID", ""),
                        help="Clarifai user id for artifact upload. Empty = skip upload.")
    parser.add_argument("--app_id",      default=os.environ.get("CLARIFAI_APP_ID", ""),
                        help="Clarifai app id for artifact upload. Empty = skip upload.")
    parser.add_argument("--artifact_id", default="quantized-checkpoint",
                        help="Clarifai artifact id (created if missing).")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    run_quantization(args.src, args.scheme, args.out, args.num_gpus, args.pipeline)
    sanity_check(args.out)
    upload_artifact(args.out, args.artifact_id, args.user_id, args.app_id)
    print("\n=== pipeline_step.py done ===", flush=True)


if __name__ == "__main__":
    main()
