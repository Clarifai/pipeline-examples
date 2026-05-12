#!/usr/bin/env python3
"""Eagle3 2-phase orchestrator (drives scripts/01..05, then uploads).

Argparse with full-preset defaults, one explicit `env` dict that maps
each env var name to the corresponding argparse attribute, then the
pipeline as a flat list of subprocess.run calls.

After phase-2 finetune, uploads the model version + checkpoint artifact
to Clarifai.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = "/workspace/eagle3" if os.path.isdir("/workspace/eagle3") else \
    str(Path(__file__).parent)
RUNNER_DIR = os.path.join(WORKSPACE, "qwen-with-eagle3-model")
DRAFT_DEST = os.path.join(RUNNER_DIR, "1", "eagle3_draft")


def latest_phase2_ckpt():
    ckpts = glob.glob(os.path.join(WORKSPACE, "outputs/phase2/epoch_*"))
    if not ckpts:
        raise SystemExit("no phase2 checkpoint in outputs/phase2/")
    # Names are epoch_{N}_step_{M}; sort by step number to handle double digits
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sg_n",                   default="12000", help="ShareGPT rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--uc_n",                   default="10000", help="UltraChat rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--pb_n",                   default="5000",  help="PerfectBlend rows in the phase-1 training blend (0 disables)")
    parser.add_argument("--sg_target",              default="15000", help="Target ShareGPT rows regenerated for phase-2 training")
    parser.add_argument("--uc_target",              default="15000", help="Target UltraChat rows regenerated for phase-2 training")
    parser.add_argument("--pb_target",              default="5000",  help="Target PerfectBlend rows regenerated for phase-2 training")
    parser.add_argument("--batch_size",             default="2",     help="Per-device batch size, used by phase-1 and phase-2")
    parser.add_argument("--max_length",             default="2048",  help="Max sequence length, used by phase-1 and phase-2")
    parser.add_argument("--sglang_mem_frac",        default="0.6",   help="SGLang mem fraction, used by phase-1 and phase-2 (target-model backend)")
    parser.add_argument("--dataloader_num_workers", default="8",     help="Torch dataloader workers, used by phase-1 and phase-2")
    parser.add_argument("--phase1_epochs",          default="3",     help="Number of epochs for phase-1 pretrain")
    parser.add_argument("--phase2_epochs",          default="3",     help="Number of epochs for phase-2 finetune")
    parser.add_argument("--learning_rate",          default="2e-5",  help="Learning rate for phase-2 (phase-1 LR is hardcoded at 1e-4)")
    parser.add_argument("--regen_temperature",      default="0.8",   help="Sampling temperature for phase-2 data regeneration; 0 = greedy (byte-identical regen across reruns), >0 = sampled (different bytes each run, regen has no seed)")
    parser.add_argument("--app_id",                  default="pipeline-app", help="Clarifai app id for the upload")
    parser.add_argument("--model_id",               default="qwen3-8b-eagle3", help="Clarifai model id for the upload")
    args = parser.parse_args()

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

    subprocess.run(["bash", "scripts/01_prepare_data.sh"], cwd=WORKSPACE, env=env, check=True)
    subprocess.run(["bash", "scripts/02_phase1_pretrain.sh"], cwd=WORKSPACE, env=env, check=True)
    cleanup_checkpoints(os.path.join(WORKSPACE, "outputs/phase1"))
    if int(args.sg_target) > 0:
        subprocess.run(["bash", "scripts/03_regenerate_data.sh", "sharegpt",     args.sg_target], cwd=WORKSPACE, env=env, check=True)
    if int(args.uc_target) > 0:
        subprocess.run(["bash", "scripts/03_regenerate_data.sh", "ultrachat",    args.uc_target], cwd=WORKSPACE, env=env, check=True)
    if int(args.pb_target) > 0:
        subprocess.run(["bash", "scripts/03_regenerate_data.sh", "perfectblend", args.pb_target], cwd=WORKSPACE, env=env, check=True)
    subprocess.run(["bash", "scripts/04_combine_regen.sh"], cwd=WORKSPACE, env=env, check=True)
    subprocess.run(["bash", "scripts/05_phase2_finetune.sh"], cwd=WORKSPACE, env=env, check=True)
    cleanup_checkpoints(os.path.join(WORKSPACE, "outputs/phase2"))

    package_and_upload(env, args.model_id)
    print("\n=== pipeline_step.py done ===", flush=True)


if __name__ == "__main__":
    main()
