# Design Document: Autonomous YOLOF ML Training Loop Pipeline

**Pipeline ID**: `detector-pipeline-yolof-autoloop`
**Status**: Draft

---

## 1. Overview

This document describes the design for an autonomous ML training loop pipeline for YOLOF object detection on the Clarifai platform. The pipeline orchestrates a closed-loop cycle:

```
Train Model → Evaluate → Compare Metric Against Threshold
    → Deploy: Export & Upload to Clarifai
    → Retrain (iterations remaining): Adjust Hyperparameters → Retrain
    → Stop (max iterations exhausted or plateau): Report Failure
```

The pipeline is implemented as an Argo Workflow with **conditional branching** using two supported approaches:
- **Approach A** — DAG with recursive WorkflowTemplate (recommended for production)
- **Approach B** — Unrolled multi-step sequential (fallback for simpler environments)

Both approaches reuse the existing `detector-pipeline-yolof` training step and `detector-pipeline-eval-yolof-quick-start` evaluation step, adding two new lightweight CPU-only steps:
- **metric-decision** step for threshold comparison and routing decisions ([design doc](metric-decision-design.md))
- **hp-adjust** step for hyperparameter adjustment on retrain ([design doc](hp-adjustment-design.md))

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
┌─────────┐     ┌──────────┐     ┌───────────────┐
│  Train   │────▶│ Evaluate │────▶│ Metric        │
│ (GPU)    │     │  (GPU)   │     │ Decision      │
└─────────┘     └──────────┘     │ (CPU)         │
                                  └───────┬───────┘
                                          │
                    ┌─────────────────────┬┴──────────────────┐
                    ▼                     ▼                    ▼
            decision="deploy"     decision="retrain"   decision="stop"
                    │                     │                    │
              ┌─────▼─────┐       ┌──────▼───────┐    ┌──────▼──────┐
              │  Export &  │       │  HP Adjust   │    │   Report    │
              │  Upload    │       │  (CPU)       │    │  Failure    │
              │ (CPU/GPU)  │       └──────┬───────┘    └─────────────┘
              └───────────┘              │
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
| **metric-decision** | **New** ([design](metric-decision-design.md)) | 1 CPU, 2Gi RAM, **no GPU** | Compare metrics against threshold, output routing decision (deploy/retrain/stop) |
| **hp-adjust** | **New** ([design](hp-adjustment-design.md)) | 1 CPU, 2Gi RAM, **no GPU** | Generate adjusted hyperparameters for next training iteration (only runs on retrain) |
| **model-export** | Reuse export logic from train step | 2 CPU, 4Gi RAM, no GPU | Upload passing model to Clarifai platform |

### 3.3 Data Flow Between Steps

```
Train Step              Eval Step               Decision Step            HP Adjust Step
───────────             ─────────               ─────────────            ──────────────
Outputs:                Inputs:                  Inputs:                  Inputs:
 • checkpoint_path ────▶ checkpoint_path          • eval_results_json     • hp_history
 • config_path ────────▶ config_path              • task_type             • current_hyperparams
 • train_output.json    Outputs:                  • primary_metric        • task_type
                         • eval_results.json ────▶ • metric_threshold     • tuning_strategy
                                                  • current_iteration     • search_space
                                                  • max_retrain_iters     • is_overfitting ◀── decision step
                                                  • hp_history            • lr_decay_factor
                                                  • current_hyperparams   • seed
                                                 Outputs:                Outputs:
                                                  • decision ────────────▶ (gate: retrain only)
                                                    (deploy|retrain|stop)
                                                  • hp_history ──────────▶ hp_history (to next iter)
                                                  • is_overfitting ──────▶ is_overfitting
                                                  • metric_value          • hyperparams_json ──▶ Train (next iter)
                                                  • next_iteration        • strategy_metadata
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
| `task_type` | string | `detection` | `detection`, `classification`, `llm_finetune` | Task discriminator — routes metric selection, HP adjustment rules, and eval method |
| `metric_threshold` | float | `0.50` | 0.0–inf | Quality gate — metric must meet this to deploy (detection AP: 0.50, classification accuracy: 0.85, LLM eval_loss: 1.5) |
| `primary_metric` | string | `auto` | — | Metric for threshold comparison — `auto` resolves per `task_type` (AP, accuracy/top1, eval_loss) |
| `metric_direction` | string | `auto` | `maximize`, `minimize`, `auto` | Whether higher or lower metric is better |
| `max_retrain_iterations` | int | `3` | 1–10 | Maximum training iterations before stopping |
| `tuning_strategy` | string | `schedule` | `schedule`, `grid`, `random` | HP adjustment strategy for retrain iterations |
| `search_space` | string (JSON) | `auto` | Valid JSON or `auto` | Search space for grid/random strategies — `auto` generates per task_type |
| `lr_decay_factor` | float | `0.5` | 0.1–1.0 | Multiply learning rate by this factor on each retry (schedule strategy) |
| `unfreeze_on_retry` | bool | `true` | — | Reduce `frozen_stages` by 1 on each retry (detection only) |
| `early_stop_min_delta` | float | `0.0` | 0.0–1.0 | Minimum metric improvement between iterations — 0 disables early stopping |
| `overfitting_detection` | bool | `false` | — | Check train_loss vs eval_loss divergence — sets `is_overfitting` flag for HP adjust step |
| `score_threshold` | float | `0.05` | 0.0–1.0 | Detection confidence cutoff for evaluation |
| `iou_threshold` | float | `0.6` | 0.0–1.0 | NMS IoU threshold for evaluation |

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
      - name: current_hyperparams
        value: "{}"  # JSON dict — empty on first iteration, populated by hp-adjust step on retrain
      - name: hp_history
        value: "[]"  # JSON array — accumulated by decide step across iterations

  templates:
    # ── Main DAG ──────────────────────────────────────────────
    - name: autoloop
      inputs:
        parameters:
          - name: current_iteration
          - name: current_hyperparams
          - name: hp_history
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
                  value: "{{workflow.parameters.per_item_lrate}}"
                - name: frozen_stages
                  value: "{{workflow.parameters.frozen_stages}}"
                - name: pretrained_weights
                  value: "{{workflow.parameters.pretrained_weights}}"
                - name: hyperparams_json
                  value: "{{inputs.parameters.current_hyperparams}}"
                  # JSON dict of HP overrides — "{}" on iteration 1 (use workflow defaults),
                  # populated by hp-adjust step on retrain. Train step parses this and
                  # applies overrides on top of named params.
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

          # ── STEP 3: Metric Decision ──
          - name: decide
            depends: "eval.Succeeded"
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: eval_results_json
                  value: "{{tasks.eval.outputs.parameters.eval_results}}"
                - name: task_type
                  value: "{{workflow.parameters.task_type}}"
                - name: primary_metric
                  value: "{{workflow.parameters.primary_metric}}"
                - name: metric_direction
                  value: "{{workflow.parameters.metric_direction}}"
                - name: metric_threshold
                  value: "{{workflow.parameters.metric_threshold}}"
                - name: current_iteration
                  value: "{{inputs.parameters.current_iteration}}"
                - name: max_retrain_iterations
                  value: "{{workflow.parameters.max_retrain_iterations}}"
                - name: hp_history
                  value: "{{inputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{{inputs.parameters.current_hyperparams}}"
                - name: early_stop_min_delta
                  value: "{{workflow.parameters.early_stop_min_delta}}"
                - name: overfitting_detection
                  value: "{{workflow.parameters.overfitting_detection}}"

          # ── BRANCH: Deploy → Export ──
          - name: export
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == deploy"
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

          # ── BRANCH: Retrain → HP Adjust → Recurse ──
          # ── STEP 4: HP Adjustment (only when decision=retrain) ──
          - name: hp-adjust
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <hp-adjust-ps-ref>
              template: <hp-adjust-ps-template>
            arguments:
              parameters:
                - name: hp_history
                  value: "{{tasks.decide.outputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{{inputs.parameters.current_hyperparams}}"
                - name: task_type
                  value: "{{workflow.parameters.task_type}}"
                - name: tuning_strategy
                  value: "{{workflow.parameters.tuning_strategy}}"
                - name: search_space
                  value: "{{workflow.parameters.search_space}}"
                - name: lr_decay_factor
                  value: "{{workflow.parameters.lr_decay_factor}}"
                - name: unfreeze_on_retry
                  value: "{{workflow.parameters.unfreeze_on_retry}}"
                - name: is_overfitting
                  value: "{{tasks.decide.outputs.parameters.is_overfitting}}"
                - name: seed
                  value: "{{workflow.parameters.seed}}"

          # ── STEP 5: Recursive Retrain ──
          - name: retrain
            depends: "hp-adjust.Succeeded"
            template: autoloop
            arguments:
              parameters:
                - name: current_iteration
                  value: "{{tasks.decide.outputs.parameters.next_iteration}}"
                - name: current_hyperparams
                  value: "{{tasks.hp-adjust.outputs.parameters.hyperparams_json}}"
                - name: hp_history
                  value: "{{tasks.decide.outputs.parameters.hp_history}}"

          # ── BRANCH: Stop → Report ──
          - name: report-failure
            depends: "decide.Succeeded"
            when: "{{tasks.decide.outputs.parameters.decision}} == stop"
            template: failure-report
            arguments:
              parameters:
                - name: final_metric
                  value: "{{tasks.decide.outputs.parameters.metric_value}}"
                - name: metric_name
                  value: "{{tasks.decide.outputs.parameters.metric_name}}"
                - name: iterations_run
                  value: "{{tasks.decide.outputs.parameters.current_iteration}}"

    # ── Failure Report Template ───────────────────────────────
    - name: failure-report
      inputs:
        parameters:
          - name: final_metric
          - name: metric_name
          - name: iterations_run
      container:
        image: python:3.11-slim
        command: [python, -c]
        args:
          - |
            import json, sys
            report = {
              "status": "FAILED",
              "reason": "Metric threshold not met after max iterations",
              "metric_name": "{{inputs.parameters.metric_name}}",
              "final_metric": float("{{inputs.parameters.final_metric}}"),
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
                - name: hyperparams_json
                  value: "{}"
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
                - name: task_type
                  value: "{{workflow.parameters.task_type}}"
                - name: primary_metric
                  value: "{{workflow.parameters.primary_metric}}"
                - name: metric_direction
                  value: "{{workflow.parameters.metric_direction}}"
                - name: metric_threshold
                  value: "{{workflow.parameters.metric_threshold}}"
                - name: current_iteration
                  value: "1"
                - name: max_retrain_iterations
                  value: "{{workflow.parameters.max_retrain_iterations}}"
                - name: hp_history
                  value: "[]"
                - name: current_hyperparams
                  value: "{}"
                - name: early_stop_min_delta
                  value: "{{workflow.parameters.early_stop_min_delta}}"
                - name: overfitting_detection
                  value: "{{workflow.parameters.overfitting_detection}}"

        - - name: export-1
            when: "{{steps.decide-1.outputs.parameters.decision}} == deploy"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-1.outputs.parameters.checkpoint_path}}"
                # ...

        # ══════════ Iteration 2 (conditional) ══════════
        - - name: hp-adjust-1
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <hp-adjust-ps-ref>
              template: <hp-adjust-ps-template>
            arguments:
              parameters:
                - name: hp_history
                  value: "{{steps.decide-1.outputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{}"
                - name: task_type
                  value: "{{workflow.parameters.task_type}}"
                - name: tuning_strategy
                  value: "{{workflow.parameters.tuning_strategy}}"
                - name: search_space
                  value: "{{workflow.parameters.search_space}}"
                - name: lr_decay_factor
                  value: "{{workflow.parameters.lr_decay_factor}}"
                - name: unfreeze_on_retry
                  value: "{{workflow.parameters.unfreeze_on_retry}}"
                - name: is_overfitting
                  value: "{{steps.decide-1.outputs.parameters.is_overfitting}}"
                - name: seed
                  value: "{{workflow.parameters.seed}}"

        - - name: train-2
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: hyperparams_json
                  value: "{{steps.hp-adjust-1.outputs.parameters.hyperparams_json}}"
                # ... remaining training params ...

        - - name: eval-2
            when: "{{steps.decide-1.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-2.outputs.parameters.checkpoint_path}}"
                # ... remaining eval params ...

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
                - name: hp_history
                  value: "{{steps.decide-1.outputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{{steps.hp-adjust-1.outputs.parameters.hyperparams_json}}"
                # ... remaining decision params (same workflow params as decide-1) ...

        - - name: export-2
            when: "{{steps.decide-2.outputs.parameters.decision}} == deploy"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-2.outputs.parameters.checkpoint_path}}"
                # ...

        # ══════════ Iteration 3 (conditional) ══════════
        - - name: hp-adjust-2
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <hp-adjust-ps-ref>
              template: <hp-adjust-ps-template>
            arguments:
              parameters:
                - name: hp_history
                  value: "{{steps.decide-2.outputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{{steps.hp-adjust-1.outputs.parameters.hyperparams_json}}"
                - name: is_overfitting
                  value: "{{steps.decide-2.outputs.parameters.is_overfitting}}"
                # ... remaining hp-adjust params (same workflow params as hp-adjust-1) ...

        - - name: train-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-train-ps-ref>
              template: <yolof-train-ps-template>
            arguments:
              parameters:
                - name: hyperparams_json
                  value: "{{steps.hp-adjust-2.outputs.parameters.hyperparams_json}}"
                # ... remaining training params ...

        - - name: eval-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <yolof-eval-ps-ref>
              template: <yolof-eval-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-3.outputs.parameters.checkpoint_path}}"
                # ...

        - - name: decide-3
            when: "{{steps.decide-2.outputs.parameters.decision}} == retrain"
            templateRef:
              name: <metric-decision-ps-ref>
              template: <metric-decision-ps-template>
            arguments:
              parameters:
                - name: eval_results_json
                  value: "{{steps.eval-3.outputs.parameters.eval_results}}"
                - name: current_iteration
                  value: "3"
                - name: hp_history
                  value: "{{steps.decide-2.outputs.parameters.hp_history}}"
                - name: current_hyperparams
                  value: "{{steps.hp-adjust-2.outputs.parameters.hyperparams_json}}"
                # ... remaining decision params ...

        - - name: export-3
            when: "{{steps.decide-3.outputs.parameters.decision}} == deploy"
            templateRef:
              name: <model-export-ps-ref>
              template: <model-export-ps-template>
            arguments:
              parameters:
                - name: checkpoint_path
                  value: "{{steps.train-3.outputs.parameters.checkpoint_path}}"
                # ...

        # ══════════ Final failure report ══════════
        - - name: report-failure
            when: "{{steps.decide-3.outputs.parameters.decision}} == stop"
            template: failure-report
            arguments:
              parameters:
                - name: final_metric
                  value: "{{steps.decide-3.outputs.parameters.metric_value}}"
                - name: metric_name
                  value: "{{steps.decide-3.outputs.parameters.metric_name}}"
                - name: iterations_run
                  value: "3"
```

**Advantages**: Simpler to debug, no recursive template dependency, deterministic step count.
**Limitations**: Fixed maximum depth (must be baked in at workflow definition time), verbose YAML.

---

## 6. Component Designs

### 6.1 Metric Decision Step (New)

The `metric-decision-ps` is a lightweight, GPU-free pipeline step that reads evaluation results, compares metrics against thresholds, and outputs a routing decision (`deploy` / `retrain` / `stop`). It does **not** adjust hyperparameters — that is the responsibility of the HP Adjustment Step.

**Full design**: [metric-decision-design.md](metric-decision-design.md)

Key responsibilities:
- Resolve primary metric name and direction from `task_type` (or explicit overrides)
- Compare metric value against configurable threshold
- Detect early stopping (plateau below `min_delta`)
- Detect overfitting (train_loss << eval_loss) and pass flag downstream
- Maintain iteration history (`hp_history` accumulation)
- Output routing decision + metadata for Argo conditional branching

### 6.2 HP Adjustment Step (New)

The `hp-adjust-ps` is a lightweight, GPU-free pipeline step that generates the next set of hyperparameters when the metric decision step outputs `decision="retrain"`. It is **only invoked on the retrain branch**.

**Full design**: [hp-adjustment-design.md](hp-adjustment-design.md)

Key responsibilities:
- Apply one of three tuning strategies: `schedule` (deterministic decay), `grid` (exhaustive enumeration), or `random` (seeded sampling)
- Task-aware defaults: each task type (detection, classification, llm_finetune) has its own schedule rules, grid defaults, and random distributions
- Apply overfitting corrections when `is_overfitting=true` (reduce capacity, increase regularization)
- Output new hyperparameter JSON for the next training iteration

### 6.3 Modifications to Existing Training Step

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

### 6.4 Modifications to Existing Evaluation Step

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

The HP adjustment logic is fully decoupled into its own pipeline step (`hp-adjust-ps`). See [hp-adjustment-design.md](hp-adjustment-design.md) for the complete design including:

- **Three strategies**: `schedule` (deterministic decay), `grid` (exhaustive enumeration), `random` (seeded sampling)
- **Task-specific rules**: Different schedule rules and default search spaces for detection, classification, and LLM fine-tuning
- **Overfitting corrections**: Automatic regularization adjustments when overfitting is detected by the metric decision step

### 7.1 Quick Reference — Detection Schedule (Default)

| Iteration | `per_item_lrate` | `frozen_stages` | Effective LR (batch=16) | Rationale |
|-----------|-----------------|-----------------|------------------------|-----------|
| 1 (initial) | 0.001875 | 1 | 0.03 | Default training config |
| 2 (retry 1) | 0.0009375 | 0 | 0.015 | Halve LR + unfreeze backbone |
| 3 (retry 2) | 0.00046875 | 0 | 0.0075 | Halve LR again |

> **Effective LR formula**: `batch_size × num_gpus × per_item_lrate`

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

**Decision step** outputs: (see [metric-decision-design.md](metric-decision-design.md), Section 3.3)

**HP Adjust step** outputs:
```yaml
outputs:
  parameters:
    - name: hyperparams_json
      valueFrom:
        path: /tmp/hyperparams_json
    - name: strategy_metadata
      valueFrom:
        path: /tmp/strategy_metadata
        default: "{}"
```

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
├── hp-adjust-ps/
│   ├── config.yaml                # Pipeline step compute config (CPU-only)
│   ├── Dockerfile                 # Lightweight Python 3.11 image
│   ├── requirements.txt           # Minimal: clarifai SDK only
│   └── 1/
│       ├── pipeline_step.py       # Entry point (reflection pattern)
│       └── models/
│           └── model/
│               └── 1/
│                   ├── model.py     # HPAdjustment.adjust() logic
│                   └── strategies.py # schedule/grid/random implementations
```

## 10. Files to Modify

| File | Change | Impact |
|------|--------|--------|
| `detector-pipeline-yolof-ps/.../model.py` | Add `/tmp/train_output.json` and individual output param files after training | Additive, no breaking change |
| `detector-pipeline-eval-yolof-quick-start-ps/.../model.py` | Add `checkpoint_source` and `checkpoint_path` params to `evaluate()` | Backward-compatible (defaults to existing behavior) |

---

## 11. Evaluation Metrics Reference

The eval step computes all 12 standard COCO metrics. The **primary metric** used for the threshold decision is configurable via `primary_metric` and `task_type` (default: AP for detection). All metrics are logged for observability:

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

## 12. Cold-Start Retraining (Current) & Warm-Start (Future)

### Current Behavior: Cold-Start

On retrain, each iteration starts fresh from the original pretrained weights (e.g., COCO checkpoint):

```
Iteration 1: pretrained_weights="coco" → train → checkpoint_v1.pth (fails threshold)
Iteration 2: pretrained_weights="coco" → train with lower LR → checkpoint_v2.pth
Iteration 3: pretrained_weights="coco" → train with lower LR + unfrozen backbone → checkpoint_v3.pth
```

**Pros**: Each iteration is independent, no compounding errors from a diverged initial training.
**Cons**: Slower, wastes prior compute.

This is the current implementation across all autoloop pipelines. The training step's `pretrained_weights` parameter only supports keyed values (`"coco"`, `"None"`) that map to artifact downloads via `pretrained_weights_artifacts`. It does not accept arbitrary filesystem checkpoint paths.

### Future Enhancement: Warm-Start

Warm-start would resume from the checkpoint produced by the previous iteration:

```
Iteration 1: pretrained_weights="coco" → train → checkpoint_v1.pth
Iteration 2: load checkpoint_v1.pth → train with lower LR → checkpoint_v2.pth
Iteration 3: load checkpoint_v2.pth → train with lower LR + unfrozen backbone → checkpoint_v3.pth
```

**Pros**: Faster convergence, previous training compute is not wasted.
**Cons**: Risk of compounding errors if initial training diverged badly.

To implement warm-start in the future, two changes would be needed:

1. **Training step**: Add support for an optional `checkpoint_path` parameter that, when provided, is used as `load_from` directly — bypassing the `pretrained_weights_artifacts` lookup.
2. **Autoloop config**: Forward the train step's `checkpoint_path` output to the next iteration's train step input, and add a `warm_start` workflow parameter to gate this behavior.

```yaml
# Future: warm-start retrain task
- name: retrain
  depends: "hp-adjust.Succeeded"
  template: autoloop
  arguments:
    parameters:
      - name: current_iteration
        value: "{{tasks.decide.outputs.parameters.next_iteration}}"
      - name: current_hyperparams
        value: "{{tasks.hp-adjust.outputs.parameters.hyperparams_json}}"
      - name: hp_history
        value: "{{tasks.decide.outputs.parameters.hp_history}}"
      # Future: pass previous checkpoint for warm-start
      # - name: checkpoint_path
      #   value: "{{tasks.train.outputs.parameters.checkpoint_path}}"
```

---

## 13. Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Training OOM | Argo pod exit code 137 | Halve `batch_size`, retry (future enhancement) |
| Training divergence (NaN loss) | MMEngine raises RuntimeError | Retry with halved LR (handled by loop) |
| Eval produces 0 detections | AP = 0.0, triggers retrain | Retrain with adjusted HPs |
| Eval step OOM | Argo pod exit code 137 | Reduce eval `batch_size` (manual intervention) |
| Dataset download timeout | Step fails with non-zero exit | Argo `retryStrategy` on train step (max 2 retries) |
| Max iterations exhausted | Decision step outputs `"stop"` | `report-failure` step logs summary, workflow exits with error |
| Checkpoint file missing | Eval step fails on file not found | Argo default retry or manual re-trigger |

---

## 14. Cost Analysis

Assuming a single **g5.xlarge** (1× A10G GPU, 24 GiB VRAM) instance on AWS:

| Scenario | GPU Hours | Est. Cost (on-demand) |
|----------|-----------|----------------------|
| Pass on iteration 1 | ~2h (train) + ~0.5h (eval) = **2.5h** | ~$2.50 |
| Pass on iteration 2 | ~5h | ~$5.00 |
| Fail after 3 iterations | ~7.5h | ~$7.50 |
| Decision + HP adjust steps (all iterations) | 0 GPU hours (CPU only) | ~$0.04 |

> The decision and HP adjust steps add negligible cost since they both run on CPU-only compute (< 2 seconds each).

---

## 15. Verification Plan

| # | Test | Method | Expected Result |
|---|------|--------|-----------------|
| 1 | Decision logic — deploy path | Unit test: `AP=0.65`, `threshold=0.50` | `decision="deploy"` |
| 2 | Decision logic — retrain path | Unit test: `AP=0.30`, `threshold=0.50`, `iter=1`, `max=3` | `decision="retrain"`, `is_overfitting="false"` |
| 3 | Decision logic — stop path (max iters) | Unit test: `AP=0.30`, `threshold=0.50`, `iter=3`, `max=3` | `decision="stop"` |
| 4 | Decision logic — stop path (plateau) | Unit test: AP improves by 0.005, `early_stop_min_delta=0.01` | `decision="stop"`, reason contains "Plateau" |
| 5 | Decision logic — overfitting detection | Unit test: `train_loss=0.5`, `eval_loss=2.0`, `overfitting_detection=true` | `is_overfitting="true"` |
| 6 | Metric resolution — auto detection | Unit test: `task_type="detection"`, `primary_metric="auto"` | `metric_name="AP"`, `direction="maximize"` |
| 7 | Metric resolution — auto LLM | Unit test: `task_type="llm_finetune"`, `primary_metric="auto"` | `metric_name="eval_loss"`, `direction="minimize"` |
| 8 | HP adjust — schedule detection | Unit test: `LR=0.001875`, `factor=0.5`, `frozen=1` | `LR=0.0009375`, `frozen=0` |
| 9 | HP adjust — schedule LLM | Unit test: `LR=2e-4`, `lora_r=16` | `LR=1e-4`, `lora_r=32` |
| 10 | HP adjust — grid skip tried | Unit test: 1 combo tried, 6-combo grid | Returns 2nd untried combo |
| 11 | HP adjust — random deterministic | Unit test: `seed=42`, same history | Same output on repeated calls |
| 12 | HP adjust — overfit corrections | Unit test: `is_overfitting=true`, detection | `num_epochs` halved |
| 13 | History accumulation | Unit test: 2 prior entries, decide outputs retrain | History has 3 entries |
| 14 | Argo YAML validity | `argo lint config.yaml` | No errors |
| 15 | Forced retrain cycle | Deploy with `metric_threshold=0.99`, `max_retrain_iterations=2` | train → eval → decide → hp-adjust → train → eval → decide |
| 16 | End-to-end (quick-start) | Use artifact dataset, `max_retrain_iterations=2` | Full loop completes without manual intervention |
| 17 | Parameter chain | Verify Argo output params from each step propagate correctly | No missing/empty params |
| 18 | Backward compatibility | Run existing single-shot training pipeline | No regressions from train/eval step modifications |

---

## 16. Future Enhancements

1. **Epoch scaling on retry**: Add `epoch_scale_factor` parameter (default 1.0) — multiply `num_epochs` by this factor on each retry to give lower LR more time to converge.

2. **Batch size auto-tuning**: If train step OOMs, automatically halve `batch_size` and retry (requires capturing exit codes in the DAG).

3. **Notification on failure**: Add a Slack/email webhook step at the `report-failure` node to alert the team when max iterations are exhausted.

4. **Multi-metric thresholds**: Support compound conditions like `AP >= 0.50 AND AP50 >= 0.70` for stricter quality gates.

5. **Bayesian optimization**: Add an Optuna-based `"bayesian"` tuning strategy that models the HP→metric relationship and proposes smarter candidates than grid/random.

6. **Parallel HP trials**: Use Argo's `withParam` fan-out to run N training jobs in parallel per iteration (grid/random strategies output N HP sets instead of 1).

7. **LLM-guided HP selection**: A future `"llm_guided"` strategy that sends history + search space to an LLM API for context-aware HP suggestions.

8. **Model versioning & comparison**: Store all iteration checkpoints with metadata in the Clarifai artifact store, enabling post-hoc analysis of the training trajectory.

9. **Early stopping within training**: Add MMEngine hooks to monitor validation loss during training and stop early if the model plateaus, reducing per-iteration compute cost.

10. **Warm-start retraining**: Resume from the previous iteration's checkpoint instead of cold-starting from pretrained weights each time. Requires adding a `checkpoint_path` input parameter to the training step and forwarding checkpoint outputs between autoloop iterations (see Section 12 for details).

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
