#!/usr/bin/env python3
"""Eagle3 2-phase orchestrator (drives scripts/01..05, then uploads).

Supports two modes:
  - Full 2-phase: pretrain + regen + finetune (default)
  - Finetune-only: provide --pretrained_ckpt to skip phase-1 and go
    straight to regen + finetune from a Clarifai artifact checkpoint.

After phase-2 finetune, uploads the model version + checkpoint artifact
to Clarifai.
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORKSPACE = "/workspace/eagle3" if os.path.isdir("/workspace/eagle3") else \
    str(Path(__file__).parent)
RUNNER_DIR = os.path.join(WORKSPACE, "qwen-with-eagle3-model")
DRAFT_DEST = os.path.join(RUNNER_DIR, "1", "eagle3_draft")
DATA_DIR = os.path.join(WORKSPACE, "cache", "dataset")
CKPT_CACHE = os.path.join(WORKSPACE, "cache", "pretrained_ckpt")
SPECFORGE_DIR = os.environ.get("SPECFORGE_DIR", "/workspace/SpecForge")

ARTIFACT_SHORT_RE = re.compile(r"^[A-Za-z0-9_.\-]+(@[A-Za-z0-9_.\-]+)?$")


# ── Checkpoint resolution (artifact only) ──────────────────────────────

def _resolve_artifact(artifact_id, version_id, user_id, app_id):
    """Download checkpoint from Clarifai artifact."""
    from clarifai.client.artifact_version import ArtifactVersion
    dest_dir = os.path.join(CKPT_CACHE, f"{user_id}_{app_id}_{artifact_id}")
    os.makedirs(dest_dir, exist_ok=True)
    print(
        f">>> clarifai artifact download user={user_id} app={app_id} "
        f"id={artifact_id} version={version_id or '<latest>'}",
        flush=True,
    )
    if not version_id:
        versions = list(ArtifactVersion().list(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id,
        ))
        if not versions:
            raise SystemExit(f"no versions found for artifact {artifact_id}")
        version_id = versions[0].id
        print(f">>> resolved latest artifact version: {version_id}", flush=True)
    download_path = ArtifactVersion().download(
        artifact_id=artifact_id,
        version_id=version_id,
        user_id=user_id,
        app_id=app_id,
        output_path=dest_dir,
        force=True,
    )
    is_archive = download_path.endswith((".tar.gz", ".tgz", ".tar")) or tarfile.is_tarfile(download_path)
    if os.path.isfile(download_path) and is_archive:
        extract_dir = dest_dir + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        print(f">>> extracting {download_path} -> {extract_dir}", flush=True)
        with tarfile.open(download_path, "r:*") as tar:
            tar.extractall(extract_dir)
        entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            return os.path.join(extract_dir, entries[0])
        return extract_dir
    return download_path


def resolve_pretrained_ckpt(uri, run_user_id, run_app_id):
    """Resolve --pretrained_ckpt to a local directory.

    Accepts: <artifact_id>[@<version_id>]
    Uses the run's CLARIFAI_USER_ID and --app_id to locate the artifact.
    """
    if not uri:
        return None

    os.makedirs(CKPT_CACHE, exist_ok=True)

    if not ARTIFACT_SHORT_RE.match(uri):
        raise SystemExit(
            f"unrecognized pretrained_ckpt: {uri!r}\n"
            "  expected: <artifact_id>[@<version_id>]"
        )

    if not run_user_id or not run_app_id:
        raise SystemExit(
            "pretrained_ckpt requires CLARIFAI_USER_ID env and --app_id to locate the artifact."
        )

    art, _, ver = uri.partition("@")
    return _resolve_artifact(art, ver or None, run_user_id, run_app_id)




# ── Dataset preparation for finetune-only mode ─────────────────────────

def prepare_finetune_datasets(active_targets):
    """Download datasets needed for regen (skipping step 01).

    Calls SpecForge's prepare_data.py for each active dataset and creates
    the eval holdout from the first available dataset (sharegpt > ultrachat > perfectblend).

    Returns list of (dataset, target) that are ready.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ready = []
    for dataset, target in active_targets:
        train_file = os.path.join(DATA_DIR, f"{dataset}_train.jsonl")
        if os.path.isfile(train_file) and os.path.getsize(train_file) > 0:
            print(f">>> {train_file} already exists, skipping download", flush=True)
            ready.append((dataset, target))
            continue
        print(f"\n>>> downloading {dataset} dataset via SpecForge prepare_data.py", flush=True)
        result = subprocess.run(
            [sys.executable, os.path.join(SPECFORGE_DIR, "scripts", "prepare_data.py"),
             "--dataset", dataset, "--output-path", DATA_DIR],
            cwd=SPECFORGE_DIR,
        )
        if result.returncode != 0:
            if dataset == "perfectblend":
                print(f"WARN: {dataset} download failed — skipping", flush=True)
                continue
            raise SystemExit(f"failed to download {dataset} dataset (exit {result.returncode})")
        ready.append((dataset, target))

    if not ready:
        raise SystemExit("no regen input datasets available after download")

    # Create eval holdout — prefer sharegpt, fall back to ultrachat, then perfectblend
    eval_file = os.path.join(DATA_DIR, "qwen3_8b_eval.jsonl")
    if not os.path.isfile(eval_file):
        for source_name in ["sharegpt_train.jsonl", "ultrachat_train.jsonl", "perfectblend_train.jsonl"]:
            source_file = os.path.join(DATA_DIR, source_name)
            if os.path.isfile(source_file):
                print(f">>> creating eval holdout from {source_name}", flush=True)
                with open(source_file) as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                random.seed(42)
                random.shuffle(rows)
                ev = rows[-200:] if len(rows) > 400 else rows[:max(20, len(rows) // 5)]
                with open(eval_file, "w") as f:
                    for x in ev:
                        f.write(json.dumps(x) + "\n")
                print(f">>> wrote {len(ev)} eval rows -> {eval_file}", flush=True)
                break

    return ready


# ── Shared helpers ──────────────────────────────────────────────────────

def latest_phase2_ckpt():
    ckpts = glob.glob(os.path.join(WORKSPACE, "outputs/phase2/epoch_*"))
    if not ckpts:
        raise SystemExit("no phase2 checkpoint in outputs/phase2/")
    def step_num(p):
        parts = Path(p).name.split("_step_")
        return int(parts[-1]) if len(parts) == 2 else 0
    return max(ckpts, key=step_num)


def cleanup_checkpoints(phase_dir):
    """Keep only the latest checkpoint, delete the rest to save disk."""
    ckpts = glob.glob(os.path.join(phase_dir, "epoch_*"))
    if len(ckpts) <= 1:
        return
    def step_num(p):
        parts = Path(p).name.split("_step_")
        return int(parts[-1]) if len(parts) == 2 else 0
    latest = max(ckpts, key=step_num)
    for ckpt in ckpts:
        if ckpt != latest:
            shutil.rmtree(ckpt)
            print(f"cleaned up checkpoint: {ckpt}", flush=True)
    print(f"kept latest: {latest}", flush=True)


def package_and_upload(env, model_id):
    user_id = env.get("CLARIFAI_USER_ID")
    app_id = env.get("CLARIFAI_APP_ID")
    if not user_id or not app_id:
        raise SystemExit("CLARIFAI_USER_ID and CLARIFAI_APP_ID must be set")

    ckpt = latest_phase2_ckpt()
    print(f"\n>>> injecting draft from {ckpt} -> {DRAFT_DEST}", flush=True)
    if os.path.isdir(DRAFT_DEST):
        shutil.rmtree(DRAFT_DEST)
    shutil.copytree(ckpt, DRAFT_DEST)

    sys.path.insert(0, WORKSPACE)
    from model_upload_helper import upload_model_version, upload_checkpoint_to_artifact

    print(f"\n>>> upload_model_version user={user_id} app={app_id} id={model_id}", flush=True)
    upload_model_version(RUNNER_DIR, user_id, app_id, model_id)

    archive = shutil.make_archive(
        os.path.join(WORKSPACE, "outputs", f"{Path(ckpt).name}_eagle3_draft"),
        "gztar", root_dir=ckpt,
    )
    print(f"\n>>> upload_checkpoint_to_artifact {archive}", flush=True)
    upload_checkpoint_to_artifact(archive, user_id, app_id, model_id)


def upload_phase1_artifact(env):
    """Upload phase-1 checkpoint as artifact so it can be used for finetune-only runs."""
    user_id = env.get("CLARIFAI_USER_ID")
    app_id = env.get("CLARIFAI_APP_ID")
    if not user_id or not app_id:
        return

    phase1_ckpts = glob.glob(os.path.join(WORKSPACE, "outputs/phase1/epoch_*"))
    if not phase1_ckpts:
        return

    def step_num(p):
        parts = Path(p).name.split("_step_")
        return int(parts[-1]) if len(parts) == 2 else 0
    latest = max(phase1_ckpts, key=step_num)

    sys.path.insert(0, WORKSPACE)
    from model_upload_helper import upload_checkpoint_to_artifact

    archive = shutil.make_archive(
        os.path.join(WORKSPACE, "outputs", f"{Path(latest).name}_phase1_draft"),
        "gztar", root_dir=latest,
    )
    phase1_model_id = env.get("CLARIFAI_APP_ID", "eagle3") + "_phase1"
    print(f"\n>>> uploading phase-1 checkpoint artifact: {phase1_model_id}_checkpoint", flush=True)
    upload_checkpoint_to_artifact(archive, user_id, app_id, phase1_model_id)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Finetune-only mode
    parser.add_argument("--pretrained_ckpt",        default="",      help="Pretrained Eagle3 checkpoint (Clarifai artifact). If set, skips phase-1 and goes straight to regen + finetune.")
    # Phase-1 params (ignored in finetune-only mode)
    parser.add_argument("--sg_n",                   default="12000", help="ShareGPT rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--uc_n",                   default="10000", help="UltraChat rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--pb_n",                   default="5000",  help="PerfectBlend rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--phase1_epochs",          default="3",     help="Number of epochs for phase-1 pretrain")
    # Shared params
    parser.add_argument("--sg_target",              default="15000", help="Target ShareGPT rows regenerated for phase-2 training")
    parser.add_argument("--uc_target",              default="15000", help="Target UltraChat rows regenerated for phase-2 training")
    parser.add_argument("--pb_target",              default="5000",  help="Target PerfectBlend rows regenerated for phase-2 training")
    parser.add_argument("--batch_size",             default="2",     help="Per-device batch size, used by phase-1 and phase-2")
    parser.add_argument("--max_length",             default="2048",  help="Max sequence length, used by phase-1 and phase-2")
    parser.add_argument("--sglang_mem_frac",        default="0.6",   help="SGLang mem fraction, used by phase-1 and phase-2 (target-model backend)")
    parser.add_argument("--dataloader_num_workers", default="8",     help="Torch dataloader workers, used by phase-1 and phase-2")
    parser.add_argument("--phase2_epochs",          default="3",     help="Number of epochs for phase-2 finetune")
    parser.add_argument("--learning_rate",          default="2e-5",  help="Learning rate for phase-2 (phase-1 LR is hardcoded at 1e-4)")
    parser.add_argument("--regen_temperature",      default="0.8",   help="Sampling temperature for phase-2 data regeneration")
    parser.add_argument("--app_id",                 default="pipeline-app", help="Clarifai app id for the upload")
    parser.add_argument("--model_id",               default="qwen3-8b-eagle3", help="Clarifai model id for the upload")
    args = parser.parse_args()

    finetune_only = bool(args.pretrained_ckpt)

    env = {
        **os.environ,
        "CLARIFAI_APP_ID":        args.app_id,
        "SG_N":                   args.sg_n,
        "UC_N":                   args.uc_n,
        "PB_N":                   args.pb_n,
        "BATCH_SIZE":             args.batch_size,
        "MAX_LENGTH":             args.max_length,
        "SGLANG_MEM_FRAC":        args.sglang_mem_frac,
        "DATALOADER_NUM_WORKERS": args.dataloader_num_workers,
        "PHASE1_EPOCHS":          args.phase1_epochs,
        "PHASE2_EPOCHS":          args.phase2_epochs,
        "LEARNING_RATE":          args.learning_rate,
        "REGEN_TEMPERATURE":      args.regen_temperature,
    }

    if finetune_only:
        # ── Finetune-only mode ──────────────────────────────────────────
        print("=== FINETUNE-ONLY MODE ===", flush=True)
        print(f">>> resolving pretrained_ckpt={args.pretrained_ckpt!r}", flush=True)

        run_user_id = os.environ.get("CLARIFAI_USER_ID")
        ckpt_dir = resolve_pretrained_ckpt(args.pretrained_ckpt, run_user_id, args.app_id)
        print(f">>> pretrained checkpoint: {ckpt_dir}", flush=True)

        # Set PHASE1_CKPT so 05_phase2_finetune.sh uses the artifact checkpoint
        env["PHASE1_CKPT"] = ckpt_dir

        # Download datasets for regen (replaces step 01)
        targets = [
            ("sharegpt", args.sg_target),
            ("ultrachat", args.uc_target),
            ("perfectblend", args.pb_target),
        ]
        active = [(d, t) for d, t in targets if int(t) > 0]
        if not active:
            raise SystemExit("at least one of --sg_target / --uc_target / --pb_target must be > 0")

        print("\n>>> preparing datasets for regen", flush=True)
        active = prepare_finetune_datasets(active)

        # Steps 3-5: regen → combine → finetune
        for dataset, target in active:
            subprocess.run(
                ["bash", "scripts/03_regenerate_data.sh", dataset, str(target)],
                cwd=WORKSPACE, env=env, check=True,
            )
        subprocess.run(["bash", "scripts/04_combine_regen.sh"], cwd=WORKSPACE, env=env, check=True)
        subprocess.run(["bash", "scripts/05_phase2_finetune.sh"], cwd=WORKSPACE, env=env, check=True)
        cleanup_checkpoints(os.path.join(WORKSPACE, "outputs/phase2"))

    else:
        # ── Full 2-phase mode ───────────────────────────────────────────
        print("=== FULL 2-PHASE MODE ===", flush=True)

        subprocess.run(["bash", "scripts/01_prepare_data.sh"], cwd=WORKSPACE, env=env, check=True)
        subprocess.run(["bash", "scripts/02_phase1_pretrain.sh"], cwd=WORKSPACE, env=env, check=True)
        cleanup_checkpoints(os.path.join(WORKSPACE, "outputs/phase1"))

        # Upload phase-1 checkpoint as artifact for future finetune-only runs
        upload_phase1_artifact(env)

        if int(args.sg_target) > 0:
            subprocess.run(["bash", "scripts/03_regenerate_data.sh", "sharegpt",     args.sg_target], cwd=WORKSPACE, env=env, check=True)
        if int(args.uc_target) > 0:
            subprocess.run(["bash", "scripts/03_regenerate_data.sh", "ultrachat",    args.uc_target], cwd=WORKSPACE, env=env, check=True)
        if int(args.pb_target) > 0:
            subprocess.run(["bash", "scripts/03_regenerate_data.sh", "perfectblend", args.pb_target], cwd=WORKSPACE, env=env, check=True)
        subprocess.run(["bash", "scripts/04_combine_regen.sh"], cwd=WORKSPACE, env=env, check=True)
        subprocess.run(["bash", "scripts/05_phase2_finetune.sh"], cwd=WORKSPACE, env=env, check=True)
        cleanup_checkpoints(os.path.join(WORKSPACE, "outputs/phase2"))

    # Step 6: upload model + artifact
    package_and_upload(env, args.model_id)
    print("\n=== pipeline_step.py done ===", flush=True)


if __name__ == "__main__":
    main()
