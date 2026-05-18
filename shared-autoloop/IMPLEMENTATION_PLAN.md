# Implementation Plan: Artifact-Based Autoloop (Zero Code Duplication)

## Goal

Eliminate the 3 copied `train-ps/` folders from autoloop pipelines by making the **base pipelines** (classifier-resnet, detector-yolof, lora-unsloth) reusable as-is within the autoloop workflow; using **Clarifai artifacts** for inter-step data transfer and a **conditional no-op** pattern for export control.

---

## Current Architecture (What Exists)

```
classifier-pipeline-resnet-autoloop/
├── train-ps/         ← FULL COPY of base pipeline with 3 modifications
├── hp-adjust-ps →    symlink to shared-autoloop/
├── metric-decision-ps → symlink to shared-autoloop/

detector-pipeline-yolof-autoloop/
├── train-ps/         ← FULL COPY of base pipeline with 3 modifications
├── hp-adjust-ps →    symlink to shared-autoloop/
├── metric-decision-ps → symlink to shared-autoloop/

lora-pipeline-unsloth-autoloop/
├── train-ps/         ← FULL COPY of base pipeline with 3 modifications
├── hp-adjust-ps →    symlink to shared-autoloop/
├── metric-decision-ps → symlink to shared-autoloop/
```

### Modifications in copied train-ps (what makes them different from base):

1. **+2 params** in `train()` signature: `skip_export: bool = False`, `hyperparams_json: str = "{}"`
2. **HP override block** (~10 lines) that parses `hyperparams_json` and overrides local variables
3. **Conditional export** (`if skip_export: ... else: normal_export`)

---

## Target Architecture (What We Want)

```
classifier-pipeline-resnet-autoloop/
├── config.yaml       ← References base pipeline via templateRef (no train-ps copy)
├── hp-adjust-ps →    symlink to shared-autoloop/
├── metric-decision-ps → symlink to shared-autoloop/

Base pipelines (UNCHANGED signatures):
├── classifier-pipeline-resnet/classifier-pipeline-resnet-ps/   ← Used directly
├── detector-pipeline-yolof/detector-pipeline-yolof-ps/         ← Used directly
├── lora-pipeline-unsloth-quick-start/model-version-train-ps/   ← Used directly
```

---

## Manager's Approach (Two Principles)

### Principle 1: Artifacts as Inter-Step Communication

Instead of passing `hyperparams_json` as an Argo workflow parameter → the **decision step uploads** HP overrides to a known artifact ID, and the **train step downloads** them at startup.

### Principle 2: Conditional No-Op for Export Control

Instead of a `skip_export` param → the train step **always runs** but reads a "loop state" artifact at export time. If it finds `should_export: false`, it uploads checkpoint artifacts and skips full Clarifai model export.

---

## Detailed Implementation Steps

### STEP 1: Define Artifact ID Convention

All artifact IDs use a **deterministic naming convention** so steps can find each other's data without parameter passing:

| Artifact ID | Written By | Read By | Contents |
|---|---|---|---|
| `{model_id}_hp_overrides` | Decision step (llm-decision / hp-adjust) | Train step | JSON dict of hyperparameter overrides |
| `{model_id}_loop_state` | Decision step | Train step | JSON: `{"should_export": bool, "iteration": int}` |
| `{model_id}_checkpoint` | Train step (when not exporting) | Export step / next iteration decision step | Model weights/adapter tarball |
| `{model_id}_eval_results` | Train step | Decision step | JSON: `{"metrics": {...}}` |

**Convention:** All artifacts scoped to `user_id/app_id` from environment variables or train params.

---

### STEP 2: Modify Decision Steps (shared-autoloop/)

#### 2a. `shared-autoloop/llm-decision-ps/1/models/model/1/model.py`

After the decision is made and `hyperparams_json` is computed, **upload artifacts**:

```python
# At end of decide(), after writing /tmp/ outputs:

from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion

# Get model_id from current_hyperparams or a new param
# NOTE: model_id must be passed as a param to the decision step
#       OR derived from a convention (e.g., workflow name)

# Upload HP overrides artifact
hp_artifact_id = f"{model_id}_hp_overrides"
_ensure_artifact_exists(hp_artifact_id, user_id, app_id)
ArtifactVersion().upload(
    data=hyperparams_json,
    artifact_id=hp_artifact_id,
    user_id=user_id, app_id=app_id,
    visibility="private",
)

# Upload loop state artifact
loop_state = {
    "should_export": (decision == "deploy"),
    "iteration": int(current_iteration),
    "decision": decision,  # "deploy" / "retrain" / "stop"
}
state_artifact_id = f"{model_id}_loop_state"
_ensure_artifact_exists(state_artifact_id, user_id, app_id)
ArtifactVersion().upload(
    data=json.dumps(loop_state),
    artifact_id=state_artifact_id,
    user_id=user_id, app_id=app_id,
    visibility="private",
)
```

Helper function:
```python
def _ensure_artifact_exists(artifact_id, user_id, app_id):
    """Create artifact container if it doesn't exist."""
    from clarifai.client.artifact import Artifact
    artifacts = Artifact().list(user_id=user_id, app_id=app_id)
    if not any(a.id == artifact_id for a in artifacts):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
```

#### 2b. `shared-autoloop/metric-decision-ps/1/models/model/1/model.py`

Same artifact upload pattern. This step currently only outputs a decision (no hyperparams), so just write the `loop_state` artifact.

#### 2c. `shared-autoloop/hp-adjust-ps/1/models/model/1/model.py`

Upload `{model_id}_hp_overrides` artifact with the adjusted hyperparameters.

#### 2d. New parameter needed

All decision steps need `model_id` and `user_id`/`app_id` as input params to construct artifact IDs. Add to signatures:

```python
def decide(self,
           ...existing params...,
           model_id: str = "",       # NEW
           user_id: str = "",        # NEW
           app_id: str = "",         # NEW
           ) -> str:
```

---

### STEP 3: Modify Base Pipeline Train Steps

Each base pipeline's `model.py` gets a **small addition** (~20 lines) that reads artifacts. The `train()` signature stays **UNCHANGED**.

#### 3a. HP Override via Artifact (inserted after PAT check, before training logic)

Add to each base pipeline `model.py`:

```python
        # ── Autoloop HP override (no-op if artifact doesn't exist) ──
        try:
            from clarifai.client.artifact_version import ArtifactVersion
            hp_artifact_id = f"{model_id}_hp_overrides"
            hp_data = ArtifactVersion().download(
                artifact_id=hp_artifact_id,
                user_id=user_id, app_id=app_id,
            )
            if hp_data:
                hp_overrides = json.loads(hp_data)
                logging.info(f"[Autoloop] Applying HP overrides from artifact: {hp_overrides}")
                # Pipeline-specific HP application (see below)
        except Exception:
            pass  # No artifact = single-shot mode, use defaults
```

**Pipeline-specific HP application blocks:**

For **classifier-resnet**:
```python
                if "per_item_lrate" in hp_overrides:
                    per_item_lrate = float(hp_overrides["per_item_lrate"])
                if "weight_decay" in hp_overrides:
                    weight_decay = float(hp_overrides["weight_decay"])
                if "num_epochs" in hp_overrides:
                    num_epochs = int(hp_overrides["num_epochs"])
                if "batch_size" in hp_overrides:
                    batch_size = int(hp_overrides["batch_size"])
```

For **detector-yolof**:
```python
                if "per_item_lrate" in hp_overrides:
                    per_item_lrate = float(hp_overrides["per_item_lrate"])
                if "frozen_stages" in hp_overrides:
                    frozen_stages = int(hp_overrides["frozen_stages"])
                if "num_epochs" in hp_overrides:
                    num_epochs = int(hp_overrides["num_epochs"])
                if "batch_size" in hp_overrides:
                    batch_size = int(hp_overrides["batch_size"])
```

For **lora-unsloth**:
```python
                if "learning_rate" in hp_overrides:
                    learning_rate = float(hp_overrides["learning_rate"])
                if "lora_r" in hp_overrides:
                    lora_r = int(hp_overrides["lora_r"])
                if "lora_alpha" in hp_overrides:
                    lora_alpha = int(hp_overrides["lora_alpha"])
                if "num_epochs" in hp_overrides:
                    num_epochs = int(hp_overrides["num_epochs"])
                if "weight_decay" in hp_overrides:
                    weight_decay = float(hp_overrides["weight_decay"])
```

#### 3b. Conditional No-Op Export (wraps existing export section)

Replace the existing export section in each base pipeline:

```python
        # ── Check loop state artifact for export control ──
        _should_export = True
        try:
            from clarifai.client.artifact_version import ArtifactVersion
            state_data = ArtifactVersion().download(
                artifact_id=f"{model_id}_loop_state",
                user_id=user_id, app_id=app_id,
            )
            if state_data:
                loop_state = json.loads(state_data)
                _should_export = loop_state.get("should_export", True)
        except Exception:
            pass  # No state artifact = single-shot mode, always export

        if not _should_export:
            logging.info("[Autoloop] skip_export: uploading checkpoint to artifact store")
            from model_export_helper import upload_checkpoint_to_artifact
            upload_checkpoint_to_artifact(checkpoint_path, user_id, app_id, model_id)

            # Upload eval results as artifact for decision step
            eval_results = {"metrics": metrics_dict}  # pipeline-specific extraction
            eval_artifact_id = f"{model_id}_eval_results"
            _ensure_artifact_exists(eval_artifact_id, user_id, app_id)
            ArtifactVersion().upload(
                data=json.dumps(eval_results),
                artifact_id=eval_artifact_id,
                user_id=user_id, app_id=app_id,
                visibility="private",
            )
            logging.info(f"[Autoloop] Eval results uploaded: {eval_results}")
            return "autoloop_checkpoint_uploaded"
        
        # ── Normal export (unchanged base pipeline logic) ──
        export_and_upload_model(...)
```

#### 3c. Metric Extraction (pipeline-specific)

When `_should_export = False`, the train step must extract eval metrics before uploading:

**Classifier ResNet** — Parse MMPretrain JSON log:
```python
            import glob
            log_files = sorted(glob.glob(os.path.join(self.work_dir, "*.log.json")))
            metrics = {}
            if log_files:
                with open(log_files[-1], 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("mode") == "val":
                                metrics = {k: v for k, v in entry.items()
                                           if k not in ("mode", "epoch", "iter", "lr", "step")}
                        except json.JSONDecodeError:
                            continue
```

**Detector YOLOF** — No metric extraction needed in train step (separate eval step handles this). Just upload checkpoint.

**LoRA Unsloth** — Parse HF Trainer state:
```python
            state_path = os.path.join(work_dir, "trainer_state.json")
            metrics = {}
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                log_history = state.get("log_history", [])
                for entry in reversed(log_history):
                    if "eval_loss" in entry:
                        metrics["eval_loss"] = entry["eval_loss"]
                        if "eval_runtime" in entry:
                            metrics["eval_runtime"] = entry["eval_runtime"]
                        break
                for entry in reversed(log_history):
                    if "loss" in entry and "eval_loss" not in entry:
                        metrics["train_loss"] = entry["loss"]
                        break
```

#### 3d. LoRA-Specific: Add eval_strategy

The LoRA base pipeline needs `eval_strategy` in TrainingArguments so eval metrics are available:

```python
# Add to lora-pipeline-unsloth-quick-start TrainingArguments:
            eval_strategy="steps",
            eval_steps=max(1, max_steps if max_steps > 0 else logging_steps),
```

And save trainer state after training:
```python
        trainer.train()
        logging.info("Training completed")
        trainer.state.save_to_json(os.path.join(work_dir, "trainer_state.json"))
```

**NOTE:** These additions are safe for single-shot mode — they just produce extra files that are ignored.

---

### STEP 4: Add `model_export_helper.py` to Base Pipelines

Each base pipeline needs the `upload_checkpoint_to_artifact()` function. Copy from the current autoloop versions:

**Files to create/update:**
- `classifier-pipeline-resnet/classifier-pipeline-resnet-ps/1/models/model/1/model_export_helper.py`
- `detector-pipeline-yolof/detector-pipeline-yolof-ps/1/models/model/1/model_export_helper.py`
- `lora-pipeline-unsloth-quick-start/model-version-train-ps/1/models/model/1/model_export_helper.py`

Each already has an `export_and_upload_*` function — just add `upload_checkpoint_to_artifact()`:

```python
def upload_checkpoint_to_artifact(checkpoint_path, user_id, app_id, model_id):
    """Upload a training checkpoint to Clarifai artifact store."""
    from clarifai.client.artifact import Artifact
    from clarifai.client.artifact_version import ArtifactVersion

    artifact_id = f"{model_id}_checkpoint"
    artifacts = Artifact().list(user_id=user_id, app_id=app_id)
    if not any(a.id == artifact_id for a in artifacts):
        Artifact().create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)

    upload_path = str(checkpoint_path)
    tarball = None
    if Path(checkpoint_path).is_dir():
        import shutil, tempfile
        tarball = Path(tempfile.mkdtemp()) / f"{Path(checkpoint_path).name}.tar.gz"
        shutil.make_archive(str(tarball).removesuffix(".tar.gz"), "gztar", root_dir=checkpoint_path)
        upload_path = str(tarball)
    try:
        ArtifactVersion().upload(
            file_path=upload_path,
            artifact_id=artifact_id,
            user_id=user_id, app_id=app_id,
            visibility="private",
        )
    finally:
        if tarball and tarball.exists():
            tarball.unlink()
```

---

### STEP 5: Modify Decision Steps to Read Eval Results from Artifact

Currently, decision steps receive `eval_results_json` as an Argo parameter (passed from train step's `/tmp/eval_results` output). With the artifact approach, they should **also** support reading from artifact:

```python
# At start of decide():
if not eval_results_json or eval_results_json == "{}":
    # Try reading from artifact
    try:
        from clarifai.client.artifact_version import ArtifactVersion
        eval_data = ArtifactVersion().download(
            artifact_id=f"{model_id}_eval_results",
            user_id=user_id, app_id=app_id,
        )
        if eval_data:
            eval_results_json = eval_data
    except Exception:
        pass
```

---

### STEP 6: Update Autoloop `config.yaml` (Argo Workflow)

#### 6a. Remove `skip_export` and `hyperparams_json` from train step arguments

Since these are now artifact-based, they're no longer passed as Argo params:

```yaml
# BEFORE:
- name: hyperparams_json
  value: "{{inputs.parameters.current_hyperparams}}"
- name: skip_export
  value: "true"

# AFTER: (remove both lines entirely)
```

#### 6b. Change `templateRef` to point to base pipeline

```yaml
# BEFORE:
templateRef:
  name: users/.../pipeline_steps/train-ps/versions/<TRAIN_PS_VERSION>

# AFTER:
templateRef:
  name: users/.../pipeline_steps/classifier-pipeline-resnet-ps/versions/<VERSION>
```

#### 6c. Add `model_id`, `user_id`, `app_id` to decision step arguments

```yaml
# Add to llm-decide arguments:
- name: model_id
  value: "{{workflow.parameters.model_id}}"
- name: user_id
  value: "{{workflow.parameters.user_id}}"
- name: app_id
  value: "{{workflow.parameters.app_id}}"
```

#### 6d. Decision step must write loop_state BEFORE train runs

The loop requires that the **decision step from the previous iteration** writes the `loop_state` artifact. For iteration 1, a lightweight **init step** is needed:

```yaml
# New init step at start of DAG:
- name: init-loop-state
  template: init-loop-state

# Template:
- name: init-loop-state
  container:
    image: python:3.11-slim
    command: [python, -c]
    args:
      - |
        from clarifai.client.artifact_version import ArtifactVersion
        import json
        loop_state = {"should_export": False, "iteration": 1, "decision": "retrain"}
        ArtifactVersion().upload(
            data=json.dumps(loop_state),
            artifact_id="{{workflow.parameters.model_id}}_loop_state",
            user_id="{{workflow.parameters.user_id}}",
            app_id="{{workflow.parameters.app_id}}",
            visibility="private",
        )
```

Or simpler: have the **train step** check if loop_state artifact exists — if not, assume single-shot mode (`_should_export = True`). This is already handled by the `except` clause.

For the autoloop case, the **decision step** writes `should_export: false` on "retrain" and `should_export: true` on "deploy". The train step in the **next iteration** reads it.

**Workflow sequence:**
```
[init-loop-state] → [train] → [decision] → [train] → [decision] → ... → [train (final export)]
```

On the final iteration, the decision step writes `should_export: true`, and the subsequent train run does the full export.

**Alternative (simpler):** Keep `skip_export` as an Argo param (it's just one boolean) and only use artifacts for `hyperparams_json`. This avoids the chicken-and-egg problem of loop_state needing to exist before the first train run.

---

### STEP 7: Remove Copied `train-ps` Folders

After all changes are verified working:

```bash
rm -rf classifier-pipeline-resnet-autoloop/train-ps/
rm -rf detector-pipeline-yolof-autoloop/train-ps/
rm -rf lora-pipeline-unsloth-autoloop/train-ps/
```

---

### STEP 8: Remove Design Docs from pipeline-examples

Per manager feedback, move design markdowns out:

```bash
rm -rf shared-autoloop/retune-pipeline-design/
# Move to wiki or internal docs repo instead
```

---

## Files Modified (Summary)

### Base Pipelines (minimal additions, signature unchanged):
| File | Change |
|------|--------|
| `classifier-pipeline-resnet/classifier-pipeline-resnet-ps/1/models/model/1/model.py` | +HP artifact read block (~10 lines), +conditional export block (~25 lines) |
| `classifier-pipeline-resnet/classifier-pipeline-resnet-ps/1/models/model/1/model_export_helper.py` | +`upload_checkpoint_to_artifact()` function |
| `detector-pipeline-yolof/detector-pipeline-yolof-ps/1/models/model/1/model.py` | +HP artifact read block (~10 lines), +conditional export block (~20 lines) |
| `detector-pipeline-yolof/detector-pipeline-yolof-ps/1/models/model/1/model_export_helper.py` | +`upload_checkpoint_to_artifact()` function |
| `lora-pipeline-unsloth-quick-start/model-version-train-ps/1/models/model/1/model.py` | +HP artifact read block (~10 lines), +conditional export block (~30 lines), +eval_strategy, +trainer_state save |
| `lora-pipeline-unsloth-quick-start/model-version-train-ps/1/models/model/1/model_export_helper.py` | +`upload_checkpoint_to_artifact()` function |

### Shared Decision Steps:
| File | Change |
|------|--------|
| `shared-autoloop/llm-decision-ps/1/models/model/1/model.py` | +artifact upload for HP overrides and loop state, +model_id/user_id/app_id params |
| `shared-autoloop/metric-decision-ps/1/models/model/1/model.py` | +artifact upload for loop state, +model_id/user_id/app_id params |
| `shared-autoloop/hp-adjust-ps/1/models/model/1/model.py` | +artifact upload for HP overrides, +model_id/user_id/app_id params |

### Autoloop Configs:
| File | Change |
|------|--------|
| `classifier-pipeline-resnet-autoloop/config.yaml` | Update templateRef to base pipeline, remove skip_export/hyperparams_json args, add model_id to decision steps |
| `detector-pipeline-yolof-autoloop/config.yaml` | Same |
| `lora-pipeline-unsloth-autoloop/config.yaml` | Same |

### Deleted:
| Path | Reason |
|------|--------|
| `classifier-pipeline-resnet-autoloop/train-ps/` | No longer needed |
| `detector-pipeline-yolof-autoloop/train-ps/` | No longer needed |
| `lora-pipeline-unsloth-autoloop/train-ps/` | No longer needed |
| `shared-autoloop/retune-pipeline-design/` | Move to wiki |

---

## Artifact API Reference

```python
from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion

# Create artifact container (one-time)
Artifact().create(artifact_id="my_artifact", user_id="...", app_id="...")

# List existing artifacts
artifacts = Artifact().list(user_id="...", app_id="...")

# Upload data (file or string)
ArtifactVersion().upload(
    file_path="/path/to/file",   # OR data="json string"
    artifact_id="my_artifact",
    user_id="...", app_id="...",
    visibility="private",
)

# Download latest version
data = ArtifactVersion().download(
    artifact_id="my_artifact",
    user_id="...", app_id="...",
)
```

**Note:** Verify the exact `ArtifactVersion().download()` API signature against the installed `clarifai` SDK version. The API above is based on patterns observed in existing code.

---

## Testing Strategy

1. **Unit tests:** Mock artifact API calls in shared-autoloop tests (existing 160 tests should still pass with minor updates to mock the new artifact calls)
2. **Integration test:** Deploy one pipeline (e.g., ResNet classifier) in single-shot mode → verify artifact read fails gracefully and normal export works
3. **Autoloop integration test:** Run full autoloop with artifact-based communication → verify HP overrides arrive at train step, eval results arrive at decision step
4. **Regression:** Run base pipelines standalone → verify no behavioral change (artifact reads silently fail, normal flow)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Artifact API latency adds overhead to train startup | ~1 API call (< 1s) vs minutes of training. Negligible. |
| Artifact download API doesn't exist or differs from expected | Verify against `clarifai` SDK docs/source before implementing. May need `Artifact().get_version()` instead. |
| Race condition: decision step and train step run concurrently | Not possible — Argo DAG ensures sequential execution via `depends:` |
| Stale artifacts from previous runs | Use versioning or clear artifacts at workflow start (init step) |
| Base pipeline single-shot users see import errors | All artifact imports inside try/except — graceful fallback |

---

## Decision Points (Ask Manager)

1. **Keep `skip_export` as Argo param?** Simplifies first-iteration bootstrapping. Only `hyperparams_json` moves to artifacts. Recommended: YES — it's a simple boolean flag, not data transfer.

2. **Eval results: artifact or Argo output param?** Current system writes to `/tmp/eval_results` (Argo output). Could keep this for eval results and only use artifacts for HP overrides + checkpoints. Recommended: Keep Argo outputs for small JSON, use artifacts for large files (checkpoints).

3. **Artifact cleanup:** Who deletes stale artifacts after workflow completes? Options: cleanup step at end of workflow, TTL on artifacts, manual.

---

## Recommended Hybrid Approach

Based on the trade-offs, the cleanest implementation is:

| Data | Transfer Method | Reason |
|------|----------------|--------|
| `hyperparams_json` | **Artifact** | Decouples decision step from train step signature |
| `skip_export` | **Argo param** (keep as-is) | Simple boolean, avoids init-step complexity |
| `eval_results` | **Argo output param** (keep as-is) | Small JSON, native workflow feature |
| Checkpoints | **Artifact** (keep as-is) | Large files, already using artifacts |

This means:
- Base pipeline signature gets ONE new param: `skip_export: bool = False` (minimal change)
- HP overrides read from artifact (no param needed)
- eval_results still written to `/tmp/` as Argo outputs (no change to decision step inputs)
- Only the `hyperparams_json` param is eliminated from the train step

**OR go fully artifact-based** (zero param changes to base train signature) and accept the init-step complexity.
