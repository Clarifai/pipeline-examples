# Autoloop Self-Contained Architecture

## Overview

Each autoloop pipeline folder is **fully self-contained** — it has no runtime dependency on the base (non-autoloop) pipeline folders. The training step (`train-ps/`) in every autoloop folder is a **modified copy** of the corresponding base pipeline's training step, with autoloop-specific additions.

Shared decision steps (`hp-adjust-ps`, `metric-decision-ps`) live in `shared-autoloop/` and are symlinked into each autoloop folder.

## Directory Structure

```
shared-autoloop/
├── hp-adjust-ps/              ← Hyperparameter adjustment step (shared)
├── metric-decision-ps/        ← Metric-based decision step (shared)
├── llm-decision-ps/           ← LLM-based decision step (shared)
└── retune-pipeline-design/    ← Design documentation

classifier-pipeline-resnet-autoloop/
├── config.yaml
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
├── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)
└── train-ps/                  ← Modified copy of classifier-pipeline-resnet-ps

detector-pipeline-yolof-autoloop/
├── config.yaml
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
├── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)
└── train-ps/                  ← Modified copy of detector-pipeline-yolof-ps

lora-pipeline-unsloth-autoloop/
├── config.yaml
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
├── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)
└── train-ps/                  ← Modified copy of model-version-train-ps
```

### Why symlinks for decision steps but copies for train-ps?

The decision steps (`hp-adjust-ps`, `metric-decision-ps`, `llm-decision-ps`) are **pipeline-agnostic** — they operate on metrics and hyperparameter definitions without any knowledge of the underlying model type. A single shared implementation works across all pipelines.

The training step (`train-ps`), however, is **pipeline-specific** — each model type (ResNet classifier, YOLOF detector, LoRA LLM) has fundamentally different training code, hyperparameters, and export logic. These cannot be shared and need autoloop-specific modifications layered on top.

---

## Why Not Use Base Pipeline Code Directly?

The base pipeline training steps are **single-shot trainers** — they train once, export a model, upload it to Clarifai, and exit. The autoloop training step must participate in an **iterative retraining loop** and therefore needs three categories of modifications.

### Modification 1: New Parameters

Two new parameters are added to the `train()` method signature:

```python
def train(self,
          ...existing params...,
          skip_export: bool = False,       # NEW — controls export behavior
          hyperparams_json: str = "{}",    # NEW — receives HP overrides
          ) -> str:
```

| Parameter | Type | Purpose |
|-----------|------|---------|
| `skip_export` | `bool` | When `True`, skips full Clarifai model export and instead uploads training artifacts (checkpoints, configs) to the artifact store. Used in intermediate loop iterations. |
| `hyperparams_json` | `str` | JSON string containing hyperparameter overrides suggested by the decision step (hp-adjust, metric-decision, or llm-decision). |

### Modification 2: Hyperparameter Override Block

A block is inserted early in the `train()` method (after PAT validation, before any training logic) that parses the incoming HP overrides and applies them to local variables:

```python
# Apply HP overrides from autoloop decision step
hp_overrides = json.loads(hyperparams_json) if isinstance(hyperparams_json, str) else hyperparams_json
if hp_overrides:
    logging.info(f"Applying hyperparameter overrides: {hp_overrides}")
    if "per_item_lrate" in hp_overrides:
        per_item_lrate = float(hp_overrides["per_item_lrate"])
    if "num_epochs" in hp_overrides:
        num_epochs = int(hp_overrides["num_epochs"])
    # ... additional HP keys specific to each pipeline
```

The overrideable hyperparameters vary per pipeline because each model type exposes different tuning knobs:

| Pipeline | Overrideable Hyperparameters |
|----------|------------------------------|
| **ResNet Classifier** | `per_item_lrate`, `weight_decay`, `num_epochs`, `batch_size` |
| **YOLOF Detector** | `per_item_lrate`, `frozen_stages`, `num_epochs`, `batch_size` |
| **LoRA Unsloth** | `learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `weight_decay` |

### Modification 3: Conditional Export (`skip_export` Branch)

The base pipeline's export step (typically "STEP 7") always calls the full export-and-upload function. The autoloop version wraps this in a conditional:

```python
if skip_export:
    # 1. Upload checkpoint/adapter to artifact store
    from model_export_helper import upload_checkpoint_to_artifact
    upload_checkpoint_to_artifact(weights_path, user_id, app_id, model_id)

    # 2. Upload config to artifact store (if applicable)
    artifact_id = f"{model_id}_checkpoint"
    ArtifactVersion().upload(file_path=config_path, ...)

    # 3. Extract eval metrics from training logs
    #    (ResNet: MMPretrain JSON logs, LoRA: trainer_state.json, YOLOF: separate eval step)
    metrics = { ... }
    eval_results = {"metrics": metrics}

    # 4. Write Argo output parameters to /tmp/
    outputs = {
        "checkpoint_path": ...,
        "config_path": ...,
        "artifact_id": artifact_id,
        "eval_results": json.dumps(eval_results),
    }
    for name, value in outputs.items():
        Path(f"/tmp/{name}").write_text(str(value))
else:
    # Standard full export — identical to base pipeline
    export_and_upload_model(...)
```

**Why `skip_export` exists:** In the autoloop's iterative loop, intermediate training runs should NOT publish a new model version to Clarifai. Instead, they upload raw artifacts (checkpoints, configs) and report eval metrics back to the decision step. Only the final iteration (after the decision step declares success) runs the full export path.

---

## LoRA-Specific Additional Changes

The LoRA autoloop `train-ps` has two extra modifications beyond the standard 3-part pattern:

### Evaluation Strategy

The base LoRA training does not run mid-training evaluations. The autoloop version adds evaluation during training so eval metrics are available for the decision step:

```python
# Base
training_args = TrainingArguments(
    save_strategy="steps",
    fp16=...,

# Autoloop — adds eval during training
training_args = TrainingArguments(
    save_strategy="steps",
    eval_strategy="steps",                                        # NEW
    eval_steps=max(1, max_steps if max_steps > 0 else logging_steps),  # NEW
    fp16=...,
```

### Trainer State Persistence

After training completes, the autoloop version saves the trainer state to a JSON file for metric extraction:

```python
trainer.train()
logging.info("Training completed")
trainer.state.save_to_json(os.path.join(work_dir, "trainer_state.json"))  # NEW
```

---

## How the Autoloop Pipeline Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Autoloop Pipeline (DAG)                   │
│                                                             │
│   ┌──────────┐     ┌──────────────────┐     ┌───────────┐  │
│   │ train-ps │────▶│ decision step    │────▶│ train-ps  │  │
│   │ (iter 1) │     │ (metric-decision │     │ (iter 2)  │  │
│   │          │     │  or llm-decision │     │           │  │
│   └──────────┘     │  or hp-adjust)   │     └───────────┘  │
│                    └──────────────────┘                     │
│                           │                                 │
│                    Outputs:                                  │
│                    - continue/stop decision                  │
│                    - hyperparams_json (tuned HPs)            │
│                    - skip_export flag                        │
└─────────────────────────────────────────────────────────────┘
```

**Loop flow:**

1. **train-ps (iteration N)** — Trains with current hyperparameters. If `skip_export=True`, uploads artifacts and reports eval metrics instead of publishing a model.
2. **Decision step** — Receives eval metrics and HP history. Decides whether to continue or stop. If continuing, produces new `hyperparams_json` with tuned values.
3. **train-ps (iteration N+1)** — Receives the new `hyperparams_json`, applies HP overrides, and trains again. Repeats until the decision step signals convergence or `max_retrain_iterations` is reached.
4. **Final iteration** — Decision step signals stop; train-ps runs with `skip_export=False` and performs the standard full export to Clarifai.

---

## Summary of Changes Per File

| File (relative to autoloop folder) | Changes vs Base |
|-------------------------------------|-----------------|
| `train-ps/1/models/model/1/model.py` | +2 params (`skip_export`, `hyperparams_json`), HP override block, conditional export branch |
| `train-ps/1/models/model/1/model_export_helper.py` | New helper: `upload_checkpoint_to_artifact()` for artifact store uploads |
| `train-ps/config.yaml` | Updated `templateRef` to point to own `train-ps` |
| `train-ps/1/models/model/Dockerfile` | Copied from base, no changes needed |
| `train-ps/1/models/model/requirements.txt` | Copied from base, no changes needed |
| `config.yaml` (pipeline root) | Autoloop-specific params: `search_space`, `metric_threshold`, `max_retrain_iterations`, `tuning_strategy`, LLM config, loop state params |

---

## Testing

All autoloop tests live in `shared-autoloop/` and cover the shared decision steps:

```bash
# Run all 160 tests (requires .venv-llm-test with Python 3.11, clarifai 12.4.1)
source .venv-llm-test/bin/activate
RUN_LIVE_LLM_TESTS=1 python -m pytest shared-autoloop/ --tb=short -q
```

- **151 unit tests** — Run without any env vars
- **9 LLM integration tests** — Require `RUN_LIVE_LLM_TESTS=1` and `CLARIFAI_PAT`
