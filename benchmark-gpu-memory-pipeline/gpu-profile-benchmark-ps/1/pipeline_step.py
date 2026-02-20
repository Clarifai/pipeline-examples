#!/usr/bin/env python
"""Benchmark pipeline step: loads the vLLM serving model, shoots warmup requests
while tracking peak GPU memory via pynvml, then uploads the result as a
Clarifai artifact JSON file and prints it.
"""
import argparse
import importlib
import json
import signal
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "models"))

_mod = importlib.import_module("model.1.model")
UnslothLoRAVLLM = _mod.UnslothLoRAVLLM


def upload_benchmark_artifact(result_path, user_id, app_id, artifact_id):
    from clarifai.client.artifact import Artifact
    from clarifai.client.artifact_version import ArtifactVersion

    artifacts = Artifact().list(user_id=user_id, app_id=app_id)
    if not any(a.id == artifact_id for a in artifacts):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
    version = ArtifactVersion().upload(
        file_path=str(result_path),
        artifact_id=artifact_id,
        user_id=user_id,
        app_id=app_id,
        visibility="private",
    )
    print(f"Artifact version: {version.id}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU memory for vLLM serving model")
    parser.add_argument("--user_id", type=str, default="<YOUR_USER_ID>")
    parser.add_argument("--app_id", type=str, default="<YOUR_APP_ID>")
    parser.add_argument("--n_warmup_requests", type=int, default=5)
    args = parser.parse_args()

    model = UnslothLoRAVLLM()

    def _cleanup():
        if hasattr(model, 'server') and model.server and model.server.process:
            from clarifai.runners.utils.model_utils import terminate_process
            print("Terminating vLLM server...")
            terminate_process(model.server.process)

    # SIGTERM (e.g. container stop) won't trigger finally by default — raise so it does.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    try:
        print("Loading model...")
        model.load_model()
        print("Model loaded. Running benchmark...")

        result = model.benchmark(n_warmup_requests=args.n_warmup_requests, device_id=0)
        print(f"Benchmark result:\n{json.dumps(result, indent=2)}")

        result_path = "/tmp/benchmark_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {result_path}")

        model_config_path = Path(__file__).parent / "models" / "model" / "config.yaml"
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        model_id = model_config.get("model", {}).get("id", "benchmark-model")
        artifact_id = f"{model_id}_benchmark"

        print(f"Uploading benchmark result as artifact '{artifact_id}'...")
        upload_benchmark_artifact(result_path, args.user_id, args.app_id, artifact_id)
        print("Done.")
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
