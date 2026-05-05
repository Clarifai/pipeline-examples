#!/usr/bin/env python3
"""Download a quantized-checkpoint Clarifai artifact, unzip it, and run a
load + chat-template generate sanity check on the produced model.

Required env: CLARIFAI_PAT.
Required argv: --user_id --app_id --artifact_id [--version_id]
If --version_id is omitted, the most recent version of the artifact is used.
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path


def resolve_latest_version(user_id: str, app_id: str, artifact_id: str) -> str:
    from clarifai.client.artifact_version import ArtifactVersion
    versions = list(ArtifactVersion().list(
        artifact_id=artifact_id, user_id=user_id, app_id=app_id, per_page=50,
    ))
    if not versions:
        raise SystemExit(f"No versions found for artifact {user_id}/{app_id}/{artifact_id}")
    versions.sort(
        key=lambda v: getattr(v, "created_at", None).seconds if getattr(v, "created_at", None) else 0,
        reverse=True,
    )
    print(f">>> resolved latest version_id={versions[0].id} (out of {len(versions)} versions)",
          flush=True)
    return versions[0].id


def download_artifact(user_id: str, app_id: str, artifact_id: str, version_id: str,
                      out_dir: str) -> str:
    from clarifai.client.artifact_version import ArtifactVersion

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    zip_path = str(Path(out_dir) / f"{artifact_id}.zip")
    print(f">>> downloading artifact_id={artifact_id} version_id={version_id} -> {zip_path}",
          flush=True)
    ArtifactVersion().download(
        output_path=zip_path,
        artifact_id=artifact_id,
        version_id=version_id,
        user_id=user_id,
        app_id=app_id,
        force=True,
    )
    return zip_path


def unzip(zip_path: str, dest: str) -> str:
    Path(dest).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f">>> unzipped to {dest}", flush=True)
    print(">>> contents:", flush=True)
    for p in sorted(Path(dest).iterdir()):
        size = p.stat().st_size
        print(f"    {p.name:40s} {size:>12} bytes", flush=True)
    return dest


def sanity_check(model_dir: str) -> None:
    print(f"\n>>> sanity check: load {model_dir} and generate", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto", device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_dir)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that reverses a list."}],
        return_tensors="pt", add_generation_prompt=True,
    )
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) or hasattr(inputs, "data") else inputs
    if not isinstance(input_ids, torch.Tensor):
        input_ids = inputs.input_ids
    input_ids = input_ids.to(m.device)
    out_ids = m.generate(input_ids, max_new_tokens=200, do_sample=False)
    print(tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user_id",     required=True)
    p.add_argument("--app_id",      required=True)
    p.add_argument("--artifact_id", required=True)
    p.add_argument("--version_id",  default="",
                   help="Optional. If empty, the latest version of the artifact is used.")
    p.add_argument("--workdir",     default="/tmp/artifact-roundtrip")
    args = p.parse_args()

    if not os.environ.get("CLARIFAI_PAT"):
        raise SystemExit("CLARIFAI_PAT env var not set")

    workdir = args.workdir
    if Path(workdir).exists():
        shutil.rmtree(workdir)
    Path(workdir).mkdir(parents=True)

    version_id = args.version_id or resolve_latest_version(args.user_id, args.app_id, args.artifact_id)
    zip_path = download_artifact(args.user_id, args.app_id, args.artifact_id, version_id,
                                 workdir)
    model_dir = unzip(zip_path, str(Path(workdir) / "model"))
    sanity_check(model_dir)
    print("\n=== test_download_and_sanity.py done ===", flush=True)


if __name__ == "__main__":
    main()
