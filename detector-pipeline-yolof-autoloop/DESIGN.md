# Design Document: Autonomous YOLOF ML Training Loop Pipeline

**Pipeline ID**: `detector-pipeline-yolof-autoloop`
**Status**: Draft

---

## 1. Overview

This document describes the design for an autonomous ML training loop pipeline for YOLOF object detection on the Clarifai platform. The pipeline orchestrates a closed-loop cycle:

```
Train Model → Evaluate (COCO Benchmarks) → Compare AP Against Threshold
    → Pass: Export & Upload to Clarifai
    → Fail (iterations remaining): Adjust Hyperparameters → Retrain
    → Fail (max iterations exhausted): Report Failure
```

The pipeline is implemented as an Argo Workflow with **conditional branching** using two supported approaches:
- **Approach A** — DAG with recursive WorkflowTemplate (recommended for production)
- **Approach B** — Unrolled multi-step sequential (fallback for simpler environments)

Both approaches reuse the existing `detector-pipeline-yolof` training step and `detector-pipeline-eval-yolof-quick-start` evaluation step, adding a new lightweight **metric-decision** step for threshold comparison and hyperparameter adjustment.

---

## 2. Motivation

Current YOLOF pipelines on Clarifai are **single-shot**: they train once, optionally evaluate, and export. If the model underperforms, a human must manually inspect metrics, adjust hyperparameters, and re-trigger the pipeline. This is:

- **Slow** — round-trip time between training runs includes human review latency
- **Error-prone** — manual HP tuning is ad-hoc and inconsistent
- **Wasteful** — failed runs still consume full GPU compute with no automated recovery

An autonomous loop eliminates this gap by closing the feedback loop within a single Argo Workflow execution.

---

## 3. Pipeline DAG Architecture

### 3.1 High-Level Flow

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│  Train   │────▶│ Evaluate │────▶│  Decide  │
│ (GPU)    │     │  (GPU)   │     │ (CPU)    │
└─────────┘     └──────────┘     └────┬─────┘
                                      │
                    ┌─────────────────┬┴────────────────┐
                    ▼                 ▼                  ▼
            decision="pass"   decision="retrain"  decision="fail"
                    │                 │                  │
              ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
              │  Export &  │   │ Adjust HPs  │   │   Report    │
              │  Upload    │   │ & Re-enter  │   │  Failure    │
              │ (CPU/GPU)  │   │   Loop      │   │  (CPU)      │
              └───────────┘   └──────┬──────┘   └─────────────┘
                                     │
                                     ▼
                              ┌─────────┐
                              │  Train   │  (iteration N+1)
                              │  ...     │
                              └─────────┘
```

### 3.2 Pipeline Steps

| Step | Reuse/New | Compute | Purpose |
|------|-----------|---------|---------|
| **yolof-train** | Reuse `detector-pipeline-yolof-ps` (modified) | 4 CPU, 16Gi RAM, 1× GPU 16Gi | Train YOLOF model, output checkpoint |
| **yolof-eval** | Reuse `detector-pipeline-eval-yolof-quick-start-ps` (modified) | 4 CPU, 16Gi RAM, 1× GPU 16Gi | Run COCO evaluation, output metrics JSON |
| **metric-decision** | **New** | 1 CPU, 2Gi RAM, **no GPU** | Compare AP against threshold, output decision + adjusted HPs |
| **model-export** | Reuse export logic from train step | 2 CPU, 4Gi RAM, no GPU | Upload passing model to Clarifai platform |

### 3.3 Data Flow Between Steps

```
Train Step                    Eval Step                     Decision Step
───────────                   ─────────                     ─────────────
Outputs:                      Inputs:                       Inputs:
 • checkpoint_path ──────────▶ checkpoint_path               • eval_results_json ◀── eval results
 • config_path ──────────────▶ config_path                   • ap_threshold
 • train_output.json          Outputs:                       • current_iteration
   (Argo output param)         • eval_results.json ────────▶ • max_retrain_iterations
                                (Argo output param)          • current_per_item_lrate
                                                             • current_frozen_stages
                                                            Outputs:
                                                             • decision (pass|retrain|fail)
                                                             • new_per_item_lrate
                                                             • new_frozen_stages
                                                             • next_iteration
```

---

## 4. Workflow Parameters

### 4.1 Existing Parameters (from `detector-pipeline-yolof`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | — | Clarifai user ID |
| `app_id` | string | — | Clarifai application ID |
| `model_id` | string | `test_detector` | Target model identifier |
| `dataset_id` | string | — | Training dataset ID |
| `dataset_version_id` | string | `""` | Dataset version (optional) |
| `concepts` | string | `'["bird","cat"]'` | JSON list of detection classes |
| `seed` | int | `-1` | Random seed (-1 = no seed) |
| `image_size` | string | `[512]` | Input image dimensions |
| `max_aspect_ratio` | float | `1.5` | Max aspect ratio multiplier |
| `keep_aspect_ratio` | bool | `true` | Preserve original aspect ratio |
| `batch_size` | int | `16` | Training batch size |
| `num_epochs` | int | `100` | Training epochs per iteration |
| `min_samples_per_epoch` | int | `300` | Minimum data samples per epoch |
| `per_item_lrate` | float | `0.001875` | Per-item learning rate |
| `pretrained_weights` | string | `coco` | Pretrained weight source (`coco` or `None`) |
| `frozen_stages` | int | `1` | Number of frozen ResNet backbone stages (1–4) |
| `inference_max_batch_size` | int | `2` | Batch size for GPU benchmarking |

### 4.2 New Parameters (autoloop-specific)

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `ap_threshold` | float | `0.50` | 0.0–1.0 | Minimum AP (IoU=0.50:0.95) to accept model |
| `max_retrain_iterations` | int | `3` | 1–10 | Maximum retrain attempts before failing |
| `lr_decay_factor` | float | `0.5` | 0.1–1.0 | Multiply learning rate by this factor on each retry |
| `unfreeze_on_retry` | bool | `true` | — | Reduce `frozen_stages` by 1 on each retry |
| `score_threshold` | float | `0.05` | 0.0–1.0 | Detection confidence cutoff for evaluation |
| `iou_threshold` | float | `0.6` | 0.0–1.0 | NMS IoU threshold for evaluation |
| `warm_start` | bool | `true` | — | Resume from previous checkpoint on retrain (vs fresh start) |

---

## 5. Argo Workflow Specifications

### 5.1 Approach A — DAG with Recursive WorkflowTemplate (Recommended)

This approach uses Argo's DAG template with `when` conditions for branching and recursive template invocation for the retrain loop.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: yolof-autoloop-
spec:
  entrypoint: autoloop
  arguments:
    parameters:
      # ... all parameters from Section 4 ...
      - name: current_iteration
        value: "1"
      - name: current_per_item_lrate
        value: "{{workflow.parameters.per_item_lrate}}"
      - name: current_frozen_stages
        value: "{{workflow.parameters.frozen_stages}}"

  templates:
    # ── Main DAG ──────────────────────────────────────────────
    - name: autoloop
      inputs:
        parameters:
          - name: current_iteration
          - name: current_per_item_lrate
          - name: current_frozen_stages
      dag:
        tasks:
          # ── STEP 1: Train ──
          - name: train
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: user_id
                  value: "{{workflow.parameters.user_id}}"
                - name: app_id
                  value: "{{workflow.parameters.app_id}}"
                - name: model_id
                  value: "{{workflow.parameters.model_id}}"
                - name: dataset_id
                  value: "{{workflow.parameters.dataset_id}}"
                - name: dataset_version_id
                  value: "{{workflow.parameters.dataset_version_id}}"
                - name: concepts
                  value: "{{workflow.parameters.concepts}}"
                - name: seed
                  value: "{{workflow.parameters.seed}}"
                - name: image_size
                  value: "{{workflow.parameters.image_size}}"
                - name: batch_size
                  value: "{{workflow.parameters.batch_size}}"
                - name: num_epochs
                  value: "{{workflow.parameters.num_epochs}}"
                - name: per_item_lrate
                  value: "{{inputs.parameters.current_per_item_lrate}}"
                - name: frozen_stages
                  value: "{{inputs.parameters.current_frozen_stages}}"
                - name: pretrained_weights
                  value: "{{workflow.parameters.pretrained_weights}}"
                # ... remaining training params ...

          # ── STEP 2: Evaluate ──
          - name: eval
            depends: "train.Succeeded"
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            arguments:
              parameters:
                - name: user_id
                  value: "{{workflow.parameters.user_id}}"
                - name: app_id
                  value: "{{workflow.parameters.app_id}}"
                - name: checkpoint_source
                  value: "path"
                - name: checkpoint_path
                  value: "{{tasks.train.outputs.parameters.checkpoint_path}}"
                - name: config_path
                  value: "{{tasks.train.outputs.parameters.config_path}}"
                - name: dataset_id
                  value: "{{workflow.parameters.dataset_id}}"
                - name: concepts
                  value: "{{workflow.parameters.concepts}}"
                - name: image_size
                  value: "{{workflow.parameters.image_size}}"
                - name: batch_size
                  value: "4"
                - name: score_threshold
                  value: "{{workflow.parameters.score_threshold}}"
                - name: iou_threshold
                  value: "{{workflow.parameters.iou_threshold}}"

          # ── STEP 3: Decide ──
          - name: decide
            depends: "eval.Succeeded"
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: eval_results_json
                  value: "{{tasks.eval.outputs.parameters.eval_results}}"
                - name: ap_threshold
                  value: "{{workflow.parameters.ap_threshold}}"
                - name: current_iteration
                  value: "{{inputs.parameters.current_iteration}}"
                - name: max_retrain_iterations
                  value: "{{workflow.parameters.max_retrain_iterations}}"
                - name: current_per_item_lrate
                  value: "{{inputs.parameters.current_per_item_lrate}}"
                - name: current_frozen_stages
                  value: "{{inputs.parameters.current_frozen_stages}}"
                - name: lr_decay_factor
                  value: "{{workflow.parameters.lr_decay_factor}}"
                - name: unfreeze_on_retry
                  value: "{{workflow.parameters.unfreeze_on_retry}}"

          # ── BRANCH: Pass → Export ──
          - name: export
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == pass"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: user_id
                  value: "{{workflow.parameters.user_id}}"
                - name: app_id
                  value: "{{workflow.parameters.app_id}}"
                - name: model_id
                  value: "{{workflow.parameters.model_id}}"
                - name: checkpoint_path
                  value: "{{tasks.train.outputs.parameters.checkpoint_path}}"
                - name: config_path
                  value: "{{tasks.train.outputs.parameters.config_path}}"

          # ── BRANCH: Retrain → Recurse ──
          - name: retrain
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == retrain"
            template: autoloop
            arguments:
              parameters:
                - name: current_iteration
                  value: "{{tasks.decide.outputs.parameters.next_iteration}}"
                - name: current_per_item_lrate
                  value: "{{tasks.decide.outputs.parameters.new_per_item_lrate}}"
                - name: current_frozen_stages
                  value: "{{tasks.decide.outputs.parameters.new_frozen_stages}}"

          # ── BRANCH: Fail → Report ──
          - name: report-failure
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == fail"
            template: failure-report
            arguments:
              parameters:
                - name: final_ap
                  value: "{{tasks.decide.outputs.parameters.ap}}"
                - name: iterations_run
                  value: "{{inputs.parameters.current_iteration}}"

    # ── Failure Report Template ───────────────────────────────
    - name: failure-report
      inputs:
        parameters:
          - name: final_ap
          - name: iterations_run
      container:
        image: python:3.11-slim
        command: [python, -c]
        args:
          - |
            import json, sys
            report = {
              "status": "FAILED",
              "reason": "AP threshold not met after max iterations",
              "final_ap": float("{{inputs.parameters.final_ap}}"),
              "iterations_run": int("{{inputs.parameters.iterations_run}}"),
            }
            print(json.dumps(report, indent=2))
            sys.exit(1)
```

**Advantages**: True looping, configurable iteration count at runtime, clean DAG structure.
**Limitations**: Requires Argo Workflows support for recursive template calls (available in Argo ≥ 3.0).

### 5.2 Approach B — Unrolled Multi-Step Sequential (Fallback)

For environments where recursive templates are not supported, the loop is unrolled to a fixed maximum depth (default 3 iterations).

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: yolof-autoloop-seq-
spec:
  entrypoint: sequence
  arguments:
    parameters:
      # ... all parameters from Section 4 ...

  templates:
    - name: sequence
      steps:
        # ══════════ Iteration 1 ══════════
        - - name: train-1
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: per_item_lrate
                  value: "{{workflow.parameters.per_item_lrate}}"
                - name: frozen_stages
                  value: "{{workflow.parameters.frozen_stages}}"
                # ... remaining training params ...

        - - name: eval-1
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            arguments:
              parameters:
                - name: checkpoint_source
                  value: "path"
                - name: checkpoint_path
                  value: "{{steps.train-1.outputs.parameters.checkpoint_path}}"
                # ... remaining eval params ...

        - - name: decide-1
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: eval_results_json
                  value: "{{steps.eval-1.outputs.parameters.eval_results}}"
                - name: current_iteration
                  value: "1"
                # ... remaining decision params ...

        - - name: export-1
            when: "{{steps.decide-1.outputs.parameters.decision}} == pass"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-1.outputs.parameters.checkpoint_path}}"
                # ...

        # ══════════ Iteration 2 (conditional) ══════════
        - - name: train-2
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: per_item_lrate
                  value: "{{steps.decide-1.outputs.parameters.new_per_item_lrate}}"
                - name: frozen_stages
                  value: "{{steps.decide-1.outputs.parameters.new_frozen_stages}}"
                # ...

        - - name: eval-2
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-2.outputs.parameters.checkpoint_path}}"
                # ...

        - - name: decide-2
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: eval_results_json
                  value: "{{steps.eval-2.outputs.parameters.eval_results}}"
                - name: current_iteration
                  value: "2"
                # ...

        - - name: export-2
            when: "{{steps.decide-2.outputs.parameters.decision}} == pass"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-2.outputs.parameters.checkpoint_path}}"
                # ...

        # ══════════ Iteration 3 (conditional) ══════════
        - - name: train-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: per_item_lrate
                  value: "{{steps.decide-2.outputs.parameters.new_per_item_lrate}}"
                - name: frozen_stages
                  value: "{{steps.decide-2.outputs.parameters.new_frozen_stages}}"
                # ...

        - - name: eval-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            # ...

        - - name: decide-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: current_iteration
                  value: "3"
                # ...

        - - name: export-3
            when: "{{steps.decide-3.outputs.parameters.decision}} == pass"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            # ...

        # ══════════ Final failure report ══════════
        - - name: report-failure
            when: "{{steps.decide-3.outputs.parameters.decision}} == fail"
            template: failure-report
```

**Advantages**: Simpler to debug, no recursive template dependency, deterministic step count.
**Limitations**: Fixed maximum depth (must be baked in at workflow definition time), verbose YAML.

---

## 6. Component Designs

### 6.1 Metric Decision Step (New)

The `metric-decision-ps` is a lightweight, GPU-free pipeline step that reads evaluation results and outputs a routing decision with adjusted hyperparameters.

#### 6.1.1 Interface

```python
class MetricDecision:
    def decide(
        self,
        eval_results_json: str,          # Path to eval_results.json or inline JSON
        ap_threshold: float = 0.50,      # Minimum AP to accept
        current_iteration: int = 1,      # Current loop iteration (1-indexed)
        max_retrain_iterations: int = 3, # Maximum retrain attempts
        current_per_item_lrate: float = 0.001875,
        current_frozen_stages: int = 1,
        lr_decay_factor: float = 0.5,    # LR multiplier on retry
        unfreeze_on_retry: bool = True,  # Reduce frozen_stages by 1
    ) -> str:
        """Returns path to decision.json"""
```

#### 6.1.2 Decision Logic

```python
# 1. Load evaluation metrics
eval_results = json.load(open(eval_results_json))
current_ap = eval_results["metrics"]["AP"]

# 2. Compare against threshold
if current_ap >= ap_threshold:
    decision = {
        "decision": "pass",
        "ap": current_ap,
        "iteration": current_iteration,
    }

elif current_iteration >= max_retrain_iterations:
    decision = {
        "decision": "fail",
        "ap": current_ap,
        "iteration": current_iteration,
        "reason": f"AP {current_ap:.4f} < {ap_threshold} after {current_iteration} iterations",
    }

else:
    # Adjust hyperparameters
    new_lr = current_per_item_lrate * lr_decay_factor
    new_frozen = max(0, current_frozen_stages - 1) if unfreeze_on_retry else current_frozen_stages

    decision = {
        "decision": "retrain",
        "ap": current_ap,
        "iteration": current_iteration,
        "next_iteration": current_iteration + 1,
        "new_per_item_lrate": new_lr,
        "new_frozen_stages": new_frozen,
        "reason": f"AP {current_ap:.4f} < {ap_threshold}, retrying with lr={new_lr}, frozen={new_frozen}",
    }

# 3. Write outputs for Argo parameter extraction
with open("/tmp/decision.json", "w") as f:
    json.dump(decision, f, indent=2)

# 4. Write individual Argo output parameters
for key, value in decision.items():
    path = f"/tmp/{key}"
    with open(path, "w") as f:
        f.write(str(value))
```

#### 6.1.3 Argo Output Parameters

The step must declare output parameters so Argo can extract them:

```yaml
outputs:
  parameters:
    - name: decision
      valueFrom:
        path: /tmp/decision
    - name: ap
      valueFrom:
        path: /tmp/ap
    - name: new_per_item_lrate
      valueFrom:
        path: /tmp/new_per_item_lrate
        default: "0"
    - name: new_frozen_stages
      valueFrom:
        path: /tmp/new_frozen_stages
        default: "0"
    - name: next_iteration
      valueFrom:
        path: /tmp/next_iteration
        default: "0"
```

#### 6.1.4 Compute Requirements

```yaml
pipeline_step_compute_info:
  cpu_limit: "1000m"           # 1 CPU core
  cpu_memory: "2Gi"            # 2 GB RAM
  num_accelerators: 0          # No GPU
  accelerator_memory: "0"
  accelerator_type: []
```

### 6.2 Modifications to Existing Training Step

**File**: `detector-pipeline-yolof-ps/1/models/model/1/model.py`
**Class**: `MMDetectionYoloF`
**Method**: `train()`

**Change**: After training completes and before export, write a JSON file with checkpoint metadata:

```python
# After STEP 5 (Training), before STEP 6 (Benchmark):
train_output = {
    "checkpoint_path": self.weights_path,     # e.g., /tmp/mmdetection_work_dir/epoch_100.pth
    "config_path": self.config_py_path,       # e.g., /tmp/mmdetection_work_dir/configured_config.py
    "num_epochs": self.num_epochs,
    "final_lr": self.learning_rate,
}
with open("/tmp/train_output.json", "w") as f:
    json.dump(train_output, f)

# Write individual files for Argo parameter extraction
with open("/tmp/checkpoint_path", "w") as f:
    f.write(self.weights_path)
with open("/tmp/config_path", "w") as f:
    f.write(self.config_py_path)
```

**Impact**: Additive-only change. Existing single-shot training pipelines are unaffected (the extra files are simply ignored if not consumed).

### 6.3 Modifications to Existing Evaluation Step

**File**: `detector-pipeline-eval-yolof-quick-start-ps/1/models/model/1/model.py`
**Class**: `YOLOFEvaluator`
**Method**: `evaluate()`

**Change**: Add two new parameters and a conditional checkpoint loading path:

```python
def evaluate(
    self,
    # ... existing params ...
    checkpoint_source: str = "artifact",  # NEW: "artifact" or "path"
    checkpoint_path: str = "",            # NEW: path to .pth when source="path"
) -> str:
    # STEP 1: Download checkpoint (modified)
    if checkpoint_source == "path" and checkpoint_path:
        checkpoint_root = checkpoint_path
        logging.info(f"Using checkpoint from train step: {checkpoint_root}")
    else:
        # Existing artifact download logic (unchanged)
        artifact_info = pretrained_weights_artifacts.get(pretrained_weights)
        checkpoint_root = version.download(...)
```

Also add Argo output parameter writing after computing metrics:

```python
# After STEP 7 (save results), add:
with open("/tmp/eval_results", "w") as f:
    f.write(results_path)
```

**Impact**: Backward-compatible. Default `checkpoint_source="artifact"` preserves existing behavior exactly.

---

## 7. Hyperparameter Adjustment Strategy

### 7.1 Schedule

The decision step applies a **predefined decay schedule** combined with **rule-based backbone unfreezing**:

| Iteration | `per_item_lrate` | `frozen_stages` | Effective LR (batch=16) | Rationale |
|-----------|-----------------|-----------------|------------------------|-----------|
| 1 (initial) | 0.001875 | 1 | 0.03 | Default training config |
| 2 (retry 1) | 0.0009375 | 1 | 0.015 | Halve LR to stabilize convergence |
| 3 (retry 2) | 0.00046875 | 0 | 0.0075 | Halve LR + fully unfreeze backbone |
| 4 (retry 3) | 0.000234375 | 0 | 0.00375 | Continue halving, full adaptation |

> **Effective LR formula**: `batch_size × num_gpus × per_item_lrate`

### 7.2 Adjustment Rules

```
For each retry iteration:
  1. new_per_item_lrate = current_per_item_lrate × lr_decay_factor
  2. if unfreeze_on_retry:
       new_frozen_stages = max(0, current_frozen_stages - 1)
     else:
       new_frozen_stages = current_frozen_stages
  3. num_epochs remains unchanged
  4. batch_size remains unchanged
```

### 7.3 Why This Strategy

- **LR halving** is the most common and well-understood recovery strategy when training underperforms — it reduces oscillation around minima and helps convergence on difficult distributions
- **Backbone unfreezing** allows the model to adapt lower-level features to the target domain, which is critical when the dataset diverges significantly from COCO (the pretrain domain)
- **Conservative schedule** (only halving, not grid search) avoids excessive compute cost while still covering the most impactful hyperparameter axis

---

## 8. Artifact & Parameter Passing

### 8.1 Argo Artifact Strategy

All inter-step data passes through **Argo output parameters** (small JSON values) rather than Argo artifacts (S3/GCS). This is because:

1. Checkpoint files remain on the shared filesystem within the same workflow execution
2. Evaluation results are small JSON documents (< 1KB)
3. Decision outputs are scalar values

For environments with **ephemeral pods** (no shared filesystem), the train step should upload the checkpoint to the Clarifai artifact store and pass the `artifact_id` + `version_id` as parameters instead of a filesystem path.

### 8.2 Output Parameter Declarations

Each step must declare its outputs in the pipeline step config:

**Train step** outputs:
```yaml
outputs:
  parameters:
    - name: checkpoint_path
      valueFrom:
        path: /tmp/checkpoint_path
    - name: config_path
      valueFrom:
        path: /tmp/config_path
```

**Eval step** outputs:
```yaml
outputs:
  parameters:
    - name: eval_results
      valueFrom:
        path: /tmp/eval_results
```

**Decision step** outputs: (see Section 6.1.3)

---

## 9. Files to Create

```
detector-pipeline-yolof-autoloop/
├── config.yaml                    # Approach A: DAG with recursive template
├── config-sequential.yaml         # Approach B: Unrolled sequential
├── template.yaml                  # Clarifai UI template definition
├── metric-decision-ps/
│   ├── config.yaml                # Pipeline step compute config (CPU-only)
│   ├── Dockerfile                 # Lightweight Python 3.11 image
│   ├── requirements.txt           # Minimal: clarifai SDK only
│   └── 1/
│       ├── pipeline_step.py       # Entry point (reflection pattern)
│       └── models/
│           └── model/
│               └── 1/
│                   └── model.py   # MetricDecision.decide() logic
```

## 10. Files to Modify

| File | Change | Impact |
|------|--------|--------|
| `detector-pipeline-yolof-ps/.../model.py` | Add `/tmp/train_output.json` and individual output param files after training | Additive, no breaking change |
| `detector-pipeline-eval-yolof-quick-start-ps/.../model.py` | Add `checkpoint_source` and `checkpoint_path` params to `evaluate()` | Backward-compatible (defaults to existing behavior) |

---

## 11. Evaluation Metrics Reference

The eval step computes all 12 standard COCO metrics. Only **AP** (primary) is used for the threshold decision, but all metrics are logged for observability:

| Metric | IoU Range | Object Size | Used for Decision |
|--------|-----------|-------------|-------------------|
| **AP** | 0.50:0.95 | All | **Yes (primary)** |
| AP50 | 0.50 | All | No (logged) |
| AP75 | 0.75 | All | No (logged) |
| APsmall | 0.50:0.95 | Small (<32²px) | No (logged) |
| APmedium | 0.50:0.95 | Medium (32²–96²px) | No (logged) |
| APlarge | 0.50:0.95 | Large (>96²px) | No (logged) |
| AR@1 | 0.50:0.95 | All | No (logged) |
| AR@10 | 0.50:0.95 | All | No (logged) |
| AR@100 | 0.50:0.95 | All | No (logged) |
| ARsmall | 0.50:0.95 | Small | No (logged) |
| ARmedium | 0.50:0.95 | Medium | No (logged) |
| ARlarge | 0.50:0.95 | Large | No (logged) |

---

## 12. Warm-Start vs Cold-Start Retraining

### Option A: Warm-Start (Recommended)

On retrain, resume from the checkpoint produced by the previous iteration:

```
Iteration 1: pretrained_weights="coco" → train → checkpoint_v1.pth
Iteration 2: load checkpoint_v1.pth → train with lower LR → checkpoint_v2.pth
Iteration 3: load checkpoint_v2.pth → train with lower LR + unfrozen backbone → checkpoint_v3.pth
```

**Pros**: Faster convergence, previous training compute is not wasted.
**Cons**: Risk of compounding errors if initial training diverged badly.

### Option B: Cold-Start

On retrain, always start fresh from the pretrained COCO weights:

```
Iteration 1: pretrained_weights="coco" → train → checkpoint_v1.pth (fails)
Iteration 2: pretrained_weights="coco" → train with lower LR → checkpoint_v2.pth
Iteration 3: pretrained_weights="coco" → train with lower LR + unfrozen backbone → checkpoint_v3.pth
```

**Pros**: Each iteration is independent, no compounding errors.
**Cons**: Slower, wastes prior compute.

**Design Choice**: Use warm-start by default (`warm_start=true`), with cold-start available as a fallback parameter.

To implement warm-start, the `retrain` DAG task must pass the previous checkpoint path:

```yaml
- name: retrain
  depends: "decide.Succeeded"
  when: "{{tasks.decide.outputs.parameters.decision}} == retrain"
  template: autoloop
  arguments:
    parameters:
      # ... adjusted HPs ...
      - name: pretrained_weights
        value: "{{tasks.train.outputs.parameters.checkpoint_path}}"
```

The training step already supports arbitrary checkpoint paths via the `pretrained_weights` parameter — when set to a file path instead of `"coco"`, it loads that checkpoint directly.

---

## 13. Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Training OOM | Argo pod exit code 137 | Halve `batch_size`, retry (future enhancement) |
| Training divergence (NaN loss) | MMEngine raises RuntimeError | Retry with halved LR (handled by loop) |
| Eval produces 0 detections | AP = 0.0, triggers retrain | Retrain with adjusted HPs |
| Eval step OOM | Argo pod exit code 137 | Reduce eval `batch_size` (manual intervention) |
| Dataset download timeout | Step fails with non-zero exit | Argo `retryStrategy` on train step (max 2 retries) |
| Max iterations exhausted | Decision step outputs `"fail"` | `report-failure` step logs summary, workflow exits with error |
| Checkpoint file missing | Eval step fails on file not found | Argo default retry or manual re-trigger |

---

## 14. Cost Analysis

Assuming a single **g5.xlarge** (1× A10G GPU, 24 GiB VRAM) instance on AWS:

| Scenario | GPU Hours | Est. Cost (on-demand) |
|----------|-----------|----------------------|
| Pass on iteration 1 | ~2h (train) + ~0.5h (eval) = **2.5h** | ~$2.50 |
| Pass on iteration 2 | ~5h | ~$5.00 |
| Fail after 3 iterations | ~7.5h | ~$7.50 |
| Decision step (all iterations) | 0 GPU hours (CPU only) | ~$0.02 |

> The decision step adds negligible cost since it runs on CPU-only compute.

---

## 15. Verification Plan

| # | Test | Method | Expected Result |
|---|------|--------|-----------------|
| 1 | Decision logic — pass path | Unit test with `AP=0.65`, `threshold=0.50` | `decision="pass"` |
| 2 | Decision logic — retrain path | Unit test with `AP=0.30`, `threshold=0.50`, `iter=1`, `max=3` | `decision="retrain"`, `new_lr=0.0009375` |
| 3 | Decision logic — fail path | Unit test with `AP=0.30`, `threshold=0.50`, `iter=3`, `max=3` | `decision="fail"` |
| 4 | LR math | Unit test: `0.001875 × 0.5 × 0.5 × 0.5` | `0.000234375` |
| 5 | Frozen stages floor | Unit test: `frozen=1`, 3 decrements | `1 → 0 → 0 → 0` (floors at 0) |
| 6 | Argo YAML validity | `argo lint config.yaml` | No errors |
| 7 | Forced retrain cycle | Deploy with `ap_threshold=0.99`, `max_retrain_iterations=1` | 1 train + 1 eval + 1 retrain + 1 eval |
| 8 | End-to-end (quick-start) | Use artifact dataset, `max_retrain_iterations=2` | Full loop completes without manual intervention |
| 9 | Parameter chain | Verify Argo output params from each step propagate correctly | No missing/empty params |
| 10 | Backward compatibility | Run existing single-shot training pipeline | No regressions from train/eval step modifications |

---

## 16. Future Enhancements

1. **Epoch scaling on retry**: Add `epoch_scale_factor` parameter (default 1.0) — multiply `num_epochs` by this factor on each retry to give lower LR more time to converge.

2. **Batch size auto-tuning**: If train step OOMs, automatically halve `batch_size` and retry (requires capturing exit codes in the DAG).

3. **Notification on failure**: Add a Slack/email webhook step at the `report-failure` node to alert the team when max iterations are exhausted.

4. **Multi-metric thresholds**: Support compound conditions like `AP >= 0.50 AND AP50 >= 0.70` for stricter quality gates.

5. **Hyperparameter search**: Replace the fixed schedule with Bayesian optimization (Optuna) or grid search over LR/frozen_stages/batch_size — run parallel training experiments in a single iteration.

6. **Model versioning & comparison**: Store all iteration checkpoints with metadata in the Clarifai artifact store, enabling post-hoc analysis of the training trajectory.

7. **Early stopping within training**: Add MMEngine hooks to monitor validation loss during training and stop early if the model plateaus, reducing per-iteration compute cost.

---

## Appendix A: Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.1.2 | Deep learning framework |
| CUDA | 11.8 | GPU acceleration |
| MMDetection | 3.3.0 | Object detection toolkit |
| MMEngine | 0.10.7 | Training runner |
| MMCV | 2.1.0 | Computer vision ops |
| pycocotools | 2.0.7 | COCO metric computation |
| Clarifai SDK | 12.4.0 | Platform integration |
| Argo Workflows | ≥ 3.0 | Pipeline orchestration |
| Python | 3.11 | Runtime |

## Appendix B: YOLOF Architecture Reference

```
Input Image
    │
    ▼
ResNet-50 Backbone (4 stages, configurable freezing)
    │ (C5 features, 2048 channels)
    ▼
DilatedEncoder Neck (2048 → 512, 4 residual blocks, dilations [2,4,6,8])
    │ (512 channels)
    ▼
YOLOFHead
    ├── Classification: FocalLoss (γ=2.0, α=0.25)
    ├── Regression: GIoU Loss
    └── Anchors: ratios=[1.0], scales=[1,2,4,8,16], stride=32
    │
    ▼
NMS Post-processing (score_thr, iou_threshold, max_per_img=100)
    │
    ▼
Detections [x, y, w, h, class, score]
```

## Appendix C: Argo `when` Condition Syntax Reference

```yaml
# String equality
when: "{{tasks.decide.outputs.parameters.decision}} == pass"

# Numeric comparison
when: "{{tasks.decide.outputs.parameters.ap}} > 0.5"

# Boolean (Argo treats as string)
when: "{{tasks.decide.outputs.parameters.should_retrain}} == true"

# Negation
when: "{{tasks.decide.outputs.parameters.decision}} != fail"
```

Conditions are evaluated as string comparisons by default. For numeric comparisons, ensure the output parameter is formatted as a plain number string (no quotes or whitespace).
