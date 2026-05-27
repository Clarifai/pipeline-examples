#!/usr/bin/env python3
"""Quantize-and-evaluate pipeline step (single step variant).

End-to-end flow in one container:
  1. (Optional, on by default) Serve the SOURCE BF16 model with vLLM and run
     evalscope to establish a pre-quantization baseline.
  2. Quantize the source model with vLLM's llm-compressor.
  3. Smoke-test the produced checkpoint with HF transformers (load + generate).
  4. Optionally upload the quantized checkpoint as a Clarifai artifact.
  5. Serve the local quantized dir with vLLM and run evalscope again.
  6. Write `summary_comparison.json` with both scores + delta.
  7. Optionally upload the combined eval run dir as a second Clarifai artifact.

Set `--eval_source false` to skip step 1 (faster, but you lose the delta).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = STEP_DIR / "scripts"

# Defaults aware of whether we're inside the container or running locally.
DEFAULT_QUANT_OUT = "/workspace/outputs/quantized_model" if Path("/workspace").is_dir() \
    else str(STEP_DIR.parent / "outputs" / "quantized_model")
DEFAULT_EVAL_OUT = "/workspace/outputs/eval_report" if Path("/workspace").is_dir() \
    else str(STEP_DIR.parent / "outputs" / "eval_report")


# ---------------------------------------------------------------------------
# Quantization (delegates to scripts/quantize_llmcompressor.sh).
# ---------------------------------------------------------------------------

def run_quantization(src: str, scheme: str, out: str, num_gpus: str, pipeline: str) -> None:
    cmd = [
        "bash", str(SCRIPTS_DIR / "quantize_llmcompressor.sh"),
        "--src", src,
        "--out", out,
        "--scheme", scheme,
        "--num-gpus", num_gpus,
        "--pipeline", pipeline,
    ]
    print(f"\n>>> quantize: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(STEP_DIR), check=True)


# ---------------------------------------------------------------------------
# Smoke check the produced checkpoint via transformers.
# ---------------------------------------------------------------------------

def sanity_check(out: str) -> None:
    print(f"\n>>> sanity check: load {out} and generate", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(out, dtype="auto", device_map="auto")
    tok = AutoTokenizer.from_pretrained(out)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that reverses a list."}],
        return_tensors="pt", add_generation_prompt=True,
    )
    # `apply_chat_template(return_tensors="pt", ...)` returns a 2-D LongTensor of
    # token ids directly when only the prompt is asked for, but a BatchEncoding /
    # dict when extra options (e.g., return_dict=True) are set. Handle both.
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs
    elif isinstance(inputs, dict):
        input_ids = inputs["input_ids"]
    else:
        input_ids = getattr(inputs, "input_ids", inputs)
    input_ids = input_ids.to(m.device)
    out_ids = m.generate(input_ids, max_new_tokens=200, do_sample=False)
    print(tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True), flush=True)
    # Free the GPU before vLLM grabs it.
    del m
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Clarifai artifact upload (used twice: checkpoint + eval report).
# ---------------------------------------------------------------------------

def upload_artifact(src_dir: str, artifact_id: str, user_id: str, app_id: str,
                    label: str = "artifact") -> None:
    archive_base = src_dir.rstrip("/")
    archive = shutil.make_archive(archive_base, "zip", root_dir=src_dir)
    print(f"\n>>> upload {label}: {archive} -> user={user_id} app={app_id} "
          f"artifact_id={artifact_id}", flush=True)

    from clarifai.client.artifact import Artifact
    from clarifai.client.artifact_version import ArtifactVersion
    from clarifai.client.user import User

    user = User(user_id=user_id)
    try:
        user.app(app_id=app_id)
    except Exception as e:  # noqa: BLE001
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
    print(f">>> uploaded {label}: artifact_id={artifact_id} version_id={version.id}",
          flush=True)


# ---------------------------------------------------------------------------
# vLLM lifecycle helpers.
# ---------------------------------------------------------------------------

def start_vllm(model: str, served_name: str, num_gpus: int, port: int,
               log_path: Path) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", served_name,
        "--tensor-parallel-size", str(num_gpus),
    ]
    print(f"\n>>> starting vllm: {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_health(port: int, proc: subprocess.Popen, timeout_s: int = 1800) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"vLLM exited with code {proc.returncode} before becoming healthy. "
                f"See server log."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f">>> vLLM healthy at {url}", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise SystemExit(f"vLLM did not become healthy at {url} within {timeout_s}s")


def stop_vllm(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("\n>>> stopping vllm server", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


# ---------------------------------------------------------------------------
# evalscope.
# ---------------------------------------------------------------------------

def cap_max_tokens(model_path: str, requested: int, reserve_for_prompt: int = 4096) -> int:
    """Cap requested max_tokens to fit the model's context window.

    vLLM rejects requests where `max_tokens > max_model_len - prompt_len`.
    Our default 64000 fits Qwen3.6-class (262K ctx) but blows past smaller
    models like Qwen2.5-0.5B (32K ctx). Read max_position_embeddings from
    the model's config.json and clamp. Works for both local paths and HF
    repo ids — for repo ids we snapshot-download just config.json.
    """
    p = Path(model_path)
    if not p.is_dir():
        # Try resolving as an HF repo id (downloads only config files; tiny + idempotent).
        try:
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(
                repo_id=model_path,
                allow_patterns=["config.json", "*.json"],
            ))
        except Exception as e:  # noqa: BLE001
            print(f">>> cap_max_tokens: could not resolve {model_path!r} ({e}); "
                  f"leaving max_tokens={requested}", flush=True)
            return requested
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        return requested
    cfg = json.loads(cfg_path.read_text())
    max_pos = (
        cfg.get("max_position_embeddings")
        or (cfg.get("text_config") or {}).get("max_position_embeddings")
        or 0
    )
    if max_pos <= 0:
        return requested
    capped = max(1024, max_pos - reserve_for_prompt)
    if requested > capped:
        print(f">>> capping max_tokens {requested} -> {capped} "
              f"(model max_position_embeddings={max_pos}, "
              f"reserved {reserve_for_prompt} for prompt)", flush=True)
        return capped
    return requested


def run_evalscope(served_name: str, port: int, dataset: str,
                  eval_batch_size: int, timeout: int, max_tokens: int,
                  work_dir: str) -> None:
    cmd = [
        "evalscope", "eval",
        "--model", served_name,
        "--api-url", f"http://127.0.0.1:{port}/v1/chat/completions",
        "--api-key", "EMPTY",
        "--eval-type", "openai_api",
        "--datasets", dataset,
        "--eval-batch-size", str(eval_batch_size),
        "--timeout", str(timeout),
        "--generation-config", f"max_tokens={max_tokens}",
        "--ignore-errors",
        "--work-dir", work_dir,
    ]
    print(f"\n>>> running evalscope: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def find_latest_report(work_dir: str, dataset: str) -> Path | None:
    matches = sorted(Path(work_dir).glob(f"*/reports/*/{dataset}.json"))
    if not matches:
        matches = sorted(Path(work_dir).glob("*/reports/*/*.json"))
    return matches[-1] if matches else None


def summarize_report(label: str, report_path: Path) -> dict:
    r = json.loads(report_path.read_text())
    metric = (r.get("metrics") or [{}])[0]
    summary = {
        "label": label,
        "model": r.get("model_name"),
        "dataset": r.get("dataset_name"),
        "score": r.get("score"),
        "macro_score": metric.get("macro_score"),
        "num_samples": metric.get("num"),
        "report_path": str(report_path),
    }
    print(f"\n=== {label} eval result ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<14} {v:.4f}", flush=True)
        else:
            print(f"  {k:<14} {v}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Composite eval phase: vllm-serve + evalscope + summarize.
# ---------------------------------------------------------------------------

def run_eval_phase(
    label: str,
    model: str,
    served_name: str,
    dataset: str,
    eval_batch_size: int,
    timeout: int,
    max_tokens_request: int,
    num_gpus: int,
    port: int,
    work_dir: Path,
) -> dict:
    """Serve `model` with vLLM, run evalscope, stop vLLM, return the summary dict."""
    work_dir.mkdir(parents=True, exist_ok=True)
    server_log = work_dir / "vllm-server.log"
    proc = start_vllm(
        model=model,
        served_name=served_name,
        num_gpus=num_gpus,
        port=port,
        log_path=server_log,
    )
    try:
        wait_for_health(port, proc, timeout_s=1800)
        # If `model` is a local dir we can clamp max_tokens up front; for an HF
        # repo id we let vLLM enforce its own cap.
        effective_max_tokens = cap_max_tokens(model, max_tokens_request)
        run_evalscope(
            served_name=served_name,
            port=port,
            dataset=dataset,
            eval_batch_size=eval_batch_size,
            timeout=timeout,
            max_tokens=effective_max_tokens,
            work_dir=str(work_dir),
        )
    finally:
        stop_vllm(proc)

    report_path = find_latest_report(str(work_dir), dataset)
    if report_path is None:
        raise SystemExit(f"No report .json found under {work_dir}. See {server_log}.")
    return summarize_report(label, report_path)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()

    # Quantize phase.
    p.add_argument("--src",       default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF repo id or local path. To try quantizing an MoE model, set to "
                        "ibm-granite/granite-4.0-h-tiny (or any MoE variant with a shim under "
                        "llm-compressor/src/llmcompressor/modeling/).")
    p.add_argument("--scheme",    default="W4A16",
                   help="Quantization scheme: W4A16, NVFP4, NVFP4A16, FP8, FP8_DYNAMIC, "
                        "MXFP4, W8A8_INT8.")
    p.add_argument("--out",       default=DEFAULT_QUANT_OUT,
                   help="Output dir for the quantized checkpoint.")
    p.add_argument("--num_gpus",  default="1",
                   help="GPUs auto-picked by lowest memory.used (also used as TP size in eval).")
    p.add_argument("--pipeline",  default="datafree",
                   help="datafree (weight-only schemes), basic, sequential. "
                        "Empty = let llm-compressor auto-infer (sequential for activation-aware schemes).")

    # Evaluate phase.
    p.add_argument("--eval_source",     default="true",
                   help="true/false. When true (default), also evaluate the SOURCE BF16 model "
                        "before quantization so summary_comparison.json includes the delta. "
                        "Disable for ~2x faster runs when the baseline is already known.")
    p.add_argument("--dataset",         default="gpqa_diamond",
                   help="evalscope benchmark id.")
    p.add_argument("--eval_batch_size", default="32")
    p.add_argument("--timeout",         default="1500",
                   help="Per-request timeout seconds; 1500 covers long thinking traces on GPQA.")
    p.add_argument("--max_tokens",      default="64000",
                   help="Bound on generated tokens per response (auto-clamped to fit model's context).")
    p.add_argument("--port",            default="8000",
                   help="Local port for the vLLM server.")
    p.add_argument("--eval_out",        default=DEFAULT_EVAL_OUT,
                   help="Eval work dir (will contain source/, quantized/, summary_comparison.json).")

    # Uploads.
    p.add_argument("--user_id",            default=os.environ.get("CLARIFAI_USER_ID", ""),
                   help="Clarifai user id for artifact uploads. Empty = skip uploads.")
    p.add_argument("--app_id",             default=os.environ.get("CLARIFAI_APP_ID", ""),
                   help="Clarifai app id for artifact uploads. Empty = skip uploads.")
    p.add_argument("--artifact_id",        default="quantized-checkpoint",
                   help="Clarifai artifact id for the quantized checkpoint zip.")
    p.add_argument("--report_artifact_id", default="eval-report",
                   help="Clarifai artifact id for the evalscope report zip.")

    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_out)
    eval_root.mkdir(parents=True, exist_ok=True)
    source_eval_dir = eval_root / "source"
    quant_eval_dir = eval_root / "quantized"

    eval_source = args.eval_source.strip().lower() in ("1", "true", "yes", "on")
    source_summary: dict | None = None

    # ---- Phase 1 (optional): evaluate the SOURCE BF16 model.
    if eval_source:
        print(f"\n=== Phase 1: evaluate SOURCE model ({args.src}) ===", flush=True)
        source_summary = run_eval_phase(
            label="source",
            model=args.src,
            served_name=args.src,
            dataset=args.dataset,
            eval_batch_size=int(args.eval_batch_size),
            timeout=int(args.timeout),
            max_tokens_request=int(args.max_tokens),
            num_gpus=int(args.num_gpus),
            port=int(args.port),
            work_dir=source_eval_dir,
        )
    else:
        print("\n=== Phase 1: skipping source-model eval (eval_source=false) ===", flush=True)

    # ---- Phase 2: quantize.
    print(f"\n=== Phase 2: quantize ({args.src} -> {args.out}, scheme={args.scheme}) ===",
          flush=True)
    run_quantization(args.src, args.scheme, args.out, args.num_gpus, args.pipeline)

    # ---- Phase 3: smoke-check the local checkpoint.
    sanity_check(args.out)

    # ---- Phase 4: upload the checkpoint (optional).
    if args.user_id and args.app_id:
        upload_artifact(args.out, args.artifact_id, args.user_id, args.app_id,
                        label="quantized checkpoint")
    else:
        print("\n>>> checkpoint upload skipped: --user_id / --app_id not set", flush=True)

    # ---- Phase 5: evaluate the QUANTIZED model.
    print(f"\n=== Phase 5: evaluate QUANTIZED model ({args.out}) ===", flush=True)
    quant_summary = run_eval_phase(
        label="quantized",
        model=args.out,
        served_name=args.out,
        dataset=args.dataset,
        eval_batch_size=int(args.eval_batch_size),
        timeout=int(args.timeout),
        max_tokens_request=int(args.max_tokens),
        num_gpus=int(args.num_gpus),
        port=int(args.port),
        work_dir=quant_eval_dir,
    )

    # ---- Phase 6: write summary_comparison.json.
    comparison = {
        "source_model":   args.src,
        "quantized_model": args.out,
        "scheme":         args.scheme,
        "dataset":        args.dataset,
        "quantized":      quant_summary,
    }
    if source_summary is not None:
        comparison["source"] = source_summary
        src_score = source_summary.get("score")
        q_score = quant_summary.get("score")
        if isinstance(src_score, (int, float)) and isinstance(q_score, (int, float)):
            comparison["delta_pp"] = round((q_score - src_score) * 100, 4)
            comparison["retention_pct"] = (
                round(100 * q_score / src_score, 4) if src_score else None
            )
    (eval_root / "summary_comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== Final comparison ===", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)

    # ---- Phase 7: upload the combined eval-report (optional).
    if args.user_id and args.app_id:
        upload_artifact(str(eval_root), args.report_artifact_id, args.user_id, args.app_id,
                        label="eval report")
    else:
        print("\n>>> eval-report upload skipped: --user_id / --app_id not set", flush=True)

    print("\n=== pipeline_step.py done ===", flush=True)


if __name__ == "__main__":
    main()
