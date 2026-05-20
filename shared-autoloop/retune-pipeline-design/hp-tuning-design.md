# Design Document: Hyperparameter Tuning Step

**Step ID**: `metric-decision-ps` (renamed from metric-decision to reflect broader scope)
**Status**: Draft
**Date**: 2026-04-22

---

## 1. Overview

The Hyperparameter Tuning Step is a lightweight, GPU-free pipeline step that sits between the Evaluate and Train steps in the autonomous ML loop. It performs three functions:

1. **Metric Comparison** — Reads evaluation results, compares the primary metric against a threshold
2. **Decision Routing** — Outputs one of three decisions: `pass`, `retrain`, or `fail`
3. **HP Adjustment** — When decision is `retrain`, produces an adjusted hyperparameter set for the next training iteration

```
┌──────────┐     ┌───────────────────────┐     ┌──────────┐
│ Eval Step│────▶│  HP Tuning Step (CPU)  │────▶│ Train /  │
│ (GPU)    │     │                        │     │ Export / │
│          │     │  1. Compare metrics    │     │ Fail     │
│ outputs: │     │  2. Route decision     │     └──────────┘
│ eval_    │     │  3. Adjust HPs         │
│ results  │     │                        │
│ .json    │     │  outputs:              │
└──────────┘     │  • decision            │
                 │  • hyperparams_json    │
                 │  • hp_history          │
                 └───────────────────────┘
```

The step is generic across task types (detection, classification, LLM fine-tuning) and supports three tuning strategies: predefined schedule, grid search, and random search.

---

## 2. Compute Requirements

```yaml
pipeline_step_compute_info:
  cpu_limit: "1000m"       # 1 CPU core
  cpu_memory: "2Gi"        # 2 GB RAM
  num_accelerators: 0      # No GPU
  accelerator_memory: "0"
  accelerator_type: []
```

This step is deliberately GPU-free. It only reads JSON and performs arithmetic. Estimated execution time: < 5 seconds.

---

## 3. Interface

### 3.1 Method Signature

```python
class HyperparameterTuner:
    def decide(
        self,
        # ── Eval Results ──
        eval_results_json: str,              # Path to eval_results.json from eval step

        # ── Task Context ──
        task_type: str = "detection",        # "detection" | "classification" | "llm_finetune"
        primary_metric: str = "auto",        # Metric name or "auto" (resolved from task_type)
        metric_direction: str = "auto",      # "maximize" | "minimize" | "auto"
        metric_threshold: float = 0.50,      # Quality gate value

        # ── Loop State ──
        current_iteration: int = 1,          # Current iteration (1-indexed)
        max_retrain_iterations: int = 3,     # Max retries before failure
        hp_history: str = "[]",              # JSON array of prior iterations

        # ── Tuning Strategy ──
        tuning_strategy: str = "schedule",   # "schedule" | "grid" | "random"
        search_space: str = "auto",          # JSON search space definition
        lr_decay_factor: float = 0.5,        # LR multiplier per retry (schedule strategy)
        unfreeze_on_retry: bool = True,      # Decrement frozen_stages (detection only)
        seed: int = -1,                      # Random seed for reproducibility

        # ── Current HPs (carried from train step) ──
        current_hyperparams: str = "{}",     # JSON of current iteration's HPs

        # ── Stopping Controls ──
        early_stop_min_delta: float = 0.0,   # 0 = disabled; positive = min improvement
        overfitting_detection: bool = False,  # Check train_loss vs eval_loss divergence
    ) -> str:
        """Returns path to /tmp/decision_output.json"""
```

### 3.2 Inputs

| Input | Source | Format |
|-------|--------|--------|
| `eval_results_json` | Eval step Argo output param | File path to JSON with `{"metrics": {"AP": 0.32, ...}}` |
| `current_hyperparams` | Workflow params / prior decision output | JSON string: `{"per_item_lrate": 0.001875, "frozen_stages": 1}` |
| `hp_history` | Prior decision step output (accumulated) | JSON array: `[{"iteration": 1, "hyperparams": {...}, "metrics": {...}}, ...]` |

### 3.3 Outputs

The step writes individual files for Argo parameter extraction:

| Output File | Argo Param Name | Values | Description |
|-------------|-----------------|--------|-------------|
| `/tmp/decision` | `decision` | `"pass"`, `"retrain"`, `"fail"` | Routing decision |
| `/tmp/hyperparams_json` | `hyperparams_json` | JSON string | Adjusted HPs for next iteration |
| `/tmp/hp_history` | `hp_history` | JSON array string | Updated history with current iteration appended |
| `/tmp/ap` | `ap` | Float string | Current metric value (for logging) |
| `/tmp/next_iteration` | `next_iteration` | Int string | Next iteration number |
| `/tmp/decision_output.json` | — | Full JSON | Complete decision record (for debugging) |

Argo output parameter declarations:

```yaml
outputs:
  parameters:
    - name: decision
      valueFrom:
        path: /tmp/decision
    - name: hyperparams_json
      valueFrom:
        path: /tmp/hyperparams_json
        default: "{}"
    - name: hp_history
      valueFrom:
        path: /tmp/hp_history
        default: "[]"
    - name: ap
      valueFrom:
        path: /tmp/ap
        default: "0"
    - name: next_iteration
      valueFrom:
        path: /tmp/next_iteration
        default: "0"
```

---

## 4. Decision Logic

### 4.1 Flowchart

```
                    ┌──────────────────────┐
                    │  Load eval_results    │
                    │  Extract primary_metric│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Resolve metric       │
                    │  direction & name     │
                    │  from task_type       │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │  metric passes threshold?        │
              │  (maximize: metric >= threshold) │
              │  (minimize: metric <= threshold) │
              └───────┬──────────────┬──────────┘
                      │ YES          │ NO
               ┌──────▼──────┐       │
               │ decision =  │       │
               │ "pass"      │       │
               └─────────────┘       │
                              ┌──────▼──────────────┐
                              │ early_stop check:    │
                              │ delta < min_delta?   │
                              └───┬──────────┬──────┘
                                  │ YES      │ NO
                           ┌──────▼──────┐   │
                           │ decision =  │   │
                           │ "fail"      │   │
                           │ (plateau)   │   │
                           └─────────────┘   │
                                      ┌──────▼──────────────┐
                                      │ overfitting check:  │
                                      │ train_loss <<       │
                                      │   eval_loss?        │
                                      └───┬──────────┬──────┘
                                          │ YES      │ NO
                                   ┌──────▼──────┐   │
                                   │ adjust HPs  │   │
                                   │ for overfit  │   │
                                   │ (regularize) │   │
                                   └──────┬──────┘   │
                                          │          │
                                   ┌──────▼──────────▼──────┐
                                   │ iteration < max?       │
                                   └───┬──────────┬─────────┘
                                       │ YES      │ NO
                                ┌──────▼──────┐ ┌─▼───────────┐
                                │ decision =  │ │ decision =   │
                                │ "retrain"   │ │ "fail"       │
                                │ + adjust HPs│ │ (max iters)  │
                                └─────────────┘ └──────────────┘
```

### 4.2 Metric Resolution

```python
METRIC_DEFAULTS = {
    "detection":      {"metric": "AP",           "direction": "maximize"},
    "classification": {"metric": "accuracy/top1", "direction": "maximize"},
    "llm_finetune":   {"metric": "eval_loss",    "direction": "minimize"},
}

MINIMIZE_PATTERNS = ["loss", "perplexity", "error"]  # metric names matching these → minimize

def resolve_metric(primary_metric, metric_direction, task_type):
    if primary_metric == "auto":
        primary_metric = METRIC_DEFAULTS[task_type]["metric"]
    if metric_direction == "auto":
        if any(p in primary_metric.lower() for p in MINIMIZE_PATTERNS):
            metric_direction = "minimize"
        else:
            metric_direction = METRIC_DEFAULTS.get(task_type, {}).get("direction", "maximize")
    return primary_metric, metric_direction
```

### 4.3 Threshold Comparison

```python
def metric_passes(value, threshold, direction):
    if direction == "maximize":
        return value >= threshold
    else:  # minimize
        return value <= threshold
```

### 4.4 Early Stop Check

```python
def should_early_stop(hp_history, current_metric, metric_direction, min_delta):
    if min_delta <= 0 or len(hp_history) < 1:
        return False
    prev_metric = hp_history[-1]["metrics"][primary_metric]
    if metric_direction == "maximize":
        improvement = current_metric - prev_metric
    else:
        improvement = prev_metric - current_metric  # for loss, decrease = improvement
    return improvement < min_delta
```

### 4.5 Overfitting Check (LLM only)

```python
def check_overfitting(eval_results, task_type, overfitting_detection):
    if not overfitting_detection or task_type != "llm_finetune":
        return False, {}
    train_loss = eval_results.get("metrics", {}).get("train_loss", None)
    eval_loss = eval_results.get("metrics", {}).get("eval_loss", None)
    if train_loss is None or eval_loss is None:
        return False, {}
    if train_loss < eval_loss * 0.5:  # train_loss is less than half of eval_loss
        return True, {
            "lora_dropout": min(0.1, current_hyperparams.get("lora_dropout", 0.0) + 0.05),
            "num_epochs": max(1, current_hyperparams.get("num_epochs", 1) - 1),
        }
    return False, {}
```

---

## 5. Tuning Strategies

### 5.1 Strategy: `schedule` (Default)

A deterministic, predefined decay schedule. No search space required — adjustments are hardcoded per task type.

#### Detection Adjustments

| Retry | `per_item_lrate` | `frozen_stages` | Rule |
|-------|-----------------|-----------------|------|
| 1 | × 0.5 | unchanged | Halve LR |
| 2 | × 0.5 | − 1 | Halve LR + unfreeze 1 stage |
| 3 | × 0.5 | − 1 (floor 0) | Continue |

```python
def schedule_detection(current_hps, lr_decay_factor, unfreeze_on_retry, iteration):
    new_hps = dict(current_hps)
    new_hps["per_item_lrate"] = current_hps["per_item_lrate"] * lr_decay_factor
    if unfreeze_on_retry and iteration >= 2:
        new_hps["frozen_stages"] = max(0, current_hps.get("frozen_stages", 1) - 1)
    return new_hps
```

#### Classification Adjustments

| Retry | `per_item_lrate` | `weight_decay` | `num_epochs` | Rule |
|-------|-----------------|----------------|-------------|------|
| 1 | × 0.5 | unchanged | × 1.5 | Halve LR, extend epochs |
| 2 | × 0.5 | × 0.1 | × 1.5 | + reduce regularization |
| 3 | × 0.5 | unchanged | × 1.5 | Continue pattern |

```python
def schedule_classification(current_hps, lr_decay_factor, iteration):
    new_hps = dict(current_hps)
    new_hps["per_item_lrate"] = current_hps["per_item_lrate"] * lr_decay_factor
    new_hps["num_epochs"] = min(500, int(current_hps.get("num_epochs", 200) * 1.5))
    if iteration == 2:
        new_hps["weight_decay"] = current_hps.get("weight_decay", 0.01) * 0.1
    return new_hps
```

#### LLM Fine-tuning Adjustments

| Retry | `learning_rate` | `lora_r` | `num_epochs` | `lora_dropout` | Rule |
|-------|----------------|---------|-------------|---------------|------|
| 1 | × 0.5 | × 2 | unchanged | unchanged | Halve LR, double rank |
| 2 | × 0.5 | unchanged | + 1 | unchanged | Halve LR, extend epochs |
| 3 | × 0.5 | unchanged | unchanged | + 0.05 | Halve LR, add dropout |

```python
def schedule_llm(current_hps, lr_decay_factor, iteration):
    new_hps = dict(current_hps)
    new_hps["learning_rate"] = current_hps["learning_rate"] * lr_decay_factor
    if iteration == 1:
        new_hps["lora_r"] = min(128, current_hps.get("lora_r", 16) * 2)
        new_hps["lora_alpha"] = new_hps["lora_r"]  # keep alpha = r
    elif iteration == 2:
        new_hps["num_epochs"] = current_hps.get("num_epochs", 1) + 1
    elif iteration >= 3:
        new_hps["lora_dropout"] = min(0.1, current_hps.get("lora_dropout", 0.0) + 0.05)
    return new_hps
```

**Why this order?** For LLM LoRA training:
1. **Rank first** — `lora_r` has the highest impact on model capacity. If the model underfits, more rank helps more than lower LR.
2. **Epochs second** — More data passes are the next cheapest lever. LLM defaults to only 1 epoch.
3. **Dropout last** — Only if overfitting is suspected (high capacity but poor generalization).

### 5.2 Strategy: `grid`

Systematically enumerate all combinations from the search space, trying one per iteration.

#### Search Space Format

```json
{
  "per_item_lrate": {
    "type": "grid",
    "values": [0.001875, 0.0009375, 0.00046875]
  },
  "frozen_stages": {
    "type": "grid",
    "values": [1, 0]
  }
}
```

#### Grid Enumeration

```python
import itertools

def generate_grid(search_space):
    """Generate all grid combinations."""
    param_names = []
    param_values = []
    for name, spec in search_space.items():
        if spec["type"] == "grid":
            param_names.append(name)
            param_values.append(spec["values"])

    combos = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in combos]

def grid_select(search_space, hp_history, current_iteration):
    """Select next untried grid config."""
    all_combos = generate_grid(search_space)
    tried_configs = [h["hyperparams"] for h in hp_history]

    for combo in all_combos:
        # Check if this combo was already tried
        already_tried = False
        for tried in tried_configs:
            if all(tried.get(k) == v for k, v in combo.items()):
                already_tried = True
                break
        if not already_tried:
            return combo

    # All grid points exhausted
    return None  # Triggers "fail" decision
```

#### Grid Behavior
- Grid points are enumerated in lexicographic order of parameter values
- History is checked to skip already-tried configs
- If all grid points are exhausted before `max_retrain_iterations`, the pipeline fails early
- Total possible iterations = `min(max_retrain_iterations, product of grid sizes)`

Example: `per_item_lrate` (3 values) × `frozen_stages` (2 values) = 6 grid points. With `max_retrain_iterations=3`, only the first 3 are tried.

### 5.3 Strategy: `random`

Sample hyperparameters from defined distributions, using history to avoid near-duplicates.

#### Search Space Format

```json
{
  "per_item_lrate": {
    "type": "log_uniform",
    "min": 1e-5,
    "max": 0.01
  },
  "frozen_stages": {
    "type": "discrete_uniform",
    "min": 0,
    "max": 4
  },
  "num_epochs": {
    "type": "choice",
    "values": [50, 100, 150, 200]
  }
}
```

#### Sampling

```python
import math
import random

def sample_hp(spec, rng):
    """Sample a single hyperparameter from its distribution."""
    if spec["type"] == "log_uniform":
        log_min = math.log(spec["min"])
        log_max = math.log(spec["max"])
        return math.exp(rng.uniform(log_min, log_max))

    elif spec["type"] == "uniform":
        return rng.uniform(spec["min"], spec["max"])

    elif spec["type"] == "discrete_uniform":
        return rng.randint(spec["min"], spec["max"])

    elif spec["type"] == "choice":
        return rng.choice(spec["values"])

    elif spec["type"] == "grid":
        return rng.choice(spec["values"])

def random_select(search_space, hp_history, seed, iteration):
    """Sample a random HP config."""
    effective_seed = seed + iteration if seed >= 0 else None
    rng = random.Random(effective_seed)

    new_hps = {}
    for name, spec in search_space.items():
        new_hps[name] = sample_hp(spec, rng)

    return new_hps
```

#### Distribution Recommendations

| Parameter | Distribution | Rationale |
|-----------|-------------|-----------|
| `per_item_lrate` / `learning_rate` | `log_uniform` | LR spans multiple orders of magnitude; log-uniform gives equal probability to each order |
| `frozen_stages` | `discrete_uniform(0, 4)` | All backbone configurations equally likely |
| `lora_r` | `choice([8, 16, 32, 64])` | Powers of 2 are standard; discrete choices avoid odd values |
| `num_epochs` | `choice` or `discrete_uniform` | Depends on task; choice for LLM (1–3), uniform for vision (50–300) |
| `weight_decay` | `log_uniform(1e-5, 0.1)` | Similar to LR — spans orders of magnitude |
| `lora_dropout` | `uniform(0.0, 0.15)` | Small range, uniform is fine |

---

## 6. History Management

### 6.1 History Format

The `hp_history` parameter accumulates a JSON array across iterations:

```json
[
  {
    "iteration": 1,
    "hyperparams": {
      "per_item_lrate": 0.001875,
      "frozen_stages": 1,
      "num_epochs": 100
    },
    "metrics": {
      "AP": 0.32,
      "AP50": 0.55,
      "AP75": 0.28
    },
    "decision": "retrain",
    "reason": "AP 0.3200 < 0.50, retrying with lr=0.0009375, frozen=1"
  },
  {
    "iteration": 2,
    "hyperparams": {
      "per_item_lrate": 0.0009375,
      "frozen_stages": 1,
      "num_epochs": 100
    },
    "metrics": {
      "AP": 0.41,
      "AP50": 0.65,
      "AP75": 0.38
    },
    "decision": "retrain",
    "reason": "AP 0.4100 < 0.50, retrying with lr=0.00046875, frozen=0"
  }
]
```

### 6.2 History Update

Each invocation of the HP tuning step:
1. Parses the incoming `hp_history` JSON array
2. Appends the current iteration's record (HPs + metrics + decision)
3. Writes the updated array to `/tmp/hp_history`
4. The Argo DAG passes this forward to the next iteration

```python
def update_history(hp_history_json, current_iteration, current_hps, metrics, decision, reason):
    history = json.loads(hp_history_json)
    history.append({
        "iteration": current_iteration,
        "hyperparams": current_hps,
        "metrics": metrics,
        "decision": decision,
        "reason": reason,
    })
    return json.dumps(history)
```

### 6.3 History Size

Each entry is ~200–500 bytes. At 10 iterations max, the full history is < 5KB — well within Argo's parameter size limits (default 256KB).

### 6.4 Best-Seen Tracking

The step tracks the best metric value and its corresponding config across all iterations:

```python
def get_best_seen(hp_history, primary_metric, metric_direction):
    if not hp_history:
        return None, None
    if metric_direction == "maximize":
        best = max(hp_history, key=lambda h: h["metrics"].get(primary_metric, float("-inf")))
    else:
        best = min(hp_history, key=lambda h: h["metrics"].get(primary_metric, float("inf")))
    return best["metrics"][primary_metric], best["hyperparams"]
```

Used for:
- **Convergence detection**: If current metric is worse than best-seen by > `early_stop_min_delta`, consider reverting
- **Final export**: If the last iteration isn't the best, the export step uses the best-seen checkpoint (requires checkpoint versioning)

---

## 7. Task-Specific Behavior Summary

### 7.1 Detection

| Aspect | Behavior |
|--------|----------|
| **Primary metric** | AP (mAP@IoU=0.50:0.95) |
| **Direction** | Maximize |
| **Default threshold** | 0.50 |
| **Tunable HPs** | `per_item_lrate`, `frozen_stages` |
| **Schedule** | Halve LR each retry; unfreeze 1 backbone stage starting from retry 2 |
| **Overfitting check** | Disabled (MMDet validates every epoch internally) |
| **Typical retrain cost** | ~2.5h GPU per iteration |

### 7.2 Classification

| Aspect | Behavior |
|--------|----------|
| **Primary metric** | accuracy/top1 |
| **Direction** | Maximize |
| **Default threshold** | 0.85 |
| **Tunable HPs** | `per_item_lrate`, `weight_decay`, `num_epochs` |
| **Schedule** | Halve LR + scale epochs ×1.5 each retry; reduce weight_decay on retry 2 |
| **Overfitting check** | Disabled |
| **Typical retrain cost** | ~1.5h GPU per iteration |

### 7.3 LLM Fine-tuning

| Aspect | Behavior |
|--------|----------|
| **Primary metric** | eval_loss |
| **Direction** | Minimize |
| **Default threshold** | 1.5 |
| **Tunable HPs** | `learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `lora_dropout` |
| **Schedule** | Halve LR + double lora_r on retry 1; extend epochs on retry 2; add dropout on retry 3 |
| **Overfitting check** | Enabled by default (train_loss vs eval_loss divergence) |
| **Typical retrain cost** | ~0.5–1h GPU per iteration |

---

## 8. Implementation Plan

### 8.1 File Structure

```
detector-pipeline-yolof-autoloop/
└── metric-decision-ps/
    ├── config.yaml                          # Pipeline step compute config
    ├── Dockerfile                           # Lightweight Python 3.11 image
    ├── requirements.txt                     # Minimal deps
    └── 1/
        ├── pipeline_step.py                 # Entry point (reflection pattern)
        └── models/
            └── model/
                ├── config.yaml              # Model config
                └── 1/
                    ├── model.py             # HyperparameterTuner class
                    ├── strategies.py        # Schedule, grid, random implementations
                    └── metric_utils.py      # Metric resolution, threshold comparison
```

### 8.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /home/nonroot/main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 1 /home/nonroot/main/1

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main

ENTRYPOINT ["python3", "1/pipeline_step.py"]
```

No PyTorch, no CUDA, no ML frameworks. This image is < 200MB.

### 8.3 requirements.txt

```
clarifai>=12.1.4,<13.0.0
```

Only the Clarifai SDK is needed (for `to_pipeline_parser()` base class). The step uses only Python stdlib for JSON parsing and math.

### 8.4 Pipeline Step Config

```yaml
pipeline_step:
  id: "metric-decision-ps"
  user_id: "<YOUR_USER_ID>"
  app_id: "<YOUR_APP_ID>"

pipeline_step_compute_info:
  cpu_limit: "1000m"
  cpu_memory: "2Gi"
  num_accelerators: 0
  accelerator_memory: "0"
  accelerator_type: []

build_info:
  python_version: "3.11"
  platform: "linux/amd64"

pipeline_step_input_params:
  - eval_results_json
  - task_type
  - primary_metric
  - metric_direction
  - metric_threshold
  - current_iteration
  - max_retrain_iterations
  - hp_history
  - tuning_strategy
  - search_space
  - lr_decay_factor
  - unfreeze_on_retry
  - seed
  - current_hyperparams
  - early_stop_min_delta
  - overfitting_detection
```

### 8.5 Core Implementation Skeleton

```python
# model.py

import json
import logging
import inspect
import os

from .strategies import ScheduleStrategy, GridStrategy, RandomStrategy
from .metric_utils import resolve_metric, metric_passes, should_early_stop, check_overfitting

logging.basicConfig(level=logging.INFO)


class HyperparameterTuner:

    STRATEGIES = {
        "schedule": ScheduleStrategy,
        "grid": GridStrategy,
        "random": RandomStrategy,
    }

    @staticmethod
    def _get_argparse_type(param_annotation):
        if param_annotation == int:
            return int
        elif param_annotation == float:
            return float
        elif param_annotation == str:
            return str
        elif param_annotation == bool:
            return lambda x: str(x).lower() == 'true'
        else:
            return str

    @classmethod
    def to_pipeline_parser(cls):
        import argparse
        parser = argparse.ArgumentParser(description="HP Tuning Decision Step")
        sig = inspect.signature(cls.decide)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def decide(
        self,
        eval_results_json: str,
        task_type: str = "detection",
        primary_metric: str = "auto",
        metric_direction: str = "auto",
        metric_threshold: float = 0.50,
        current_iteration: int = 1,
        max_retrain_iterations: int = 3,
        hp_history: str = "[]",
        tuning_strategy: str = "schedule",
        search_space: str = "auto",
        lr_decay_factor: float = 0.5,
        unfreeze_on_retry: bool = True,
        seed: int = -1,
        current_hyperparams: str = "{}",
        early_stop_min_delta: float = 0.0,
        overfitting_detection: bool = False,
    ) -> str:

        # ── Parse inputs ──
        with open(eval_results_json, 'r') as f:
            eval_results = json.load(f)
        history = json.loads(hp_history)
        current_hps = json.loads(current_hyperparams)
        metrics = eval_results.get("metrics", {})

        # ── Resolve metric ──
        metric_name, direction = resolve_metric(primary_metric, metric_direction, task_type)
        current_value = metrics.get(metric_name, 0.0)
        logging.info(f"Iteration {current_iteration}: {metric_name} = {current_value:.4f} "
                     f"(threshold: {metric_threshold}, direction: {direction})")

        # ── Decision logic ──
        if metric_passes(current_value, metric_threshold, direction):
            decision = "pass"
            reason = f"{metric_name} {current_value:.4f} meets threshold {metric_threshold}"
            new_hps = current_hps

        elif early_stop_min_delta > 0 and should_early_stop(
            history, current_value, metric_name, direction, early_stop_min_delta
        ):
            decision = "fail"
            reason = f"Plateau detected: improvement below {early_stop_min_delta}"
            new_hps = current_hps

        elif current_iteration >= max_retrain_iterations:
            decision = "fail"
            reason = (f"{metric_name} {current_value:.4f} < {metric_threshold} "
                      f"after {current_iteration} iterations")
            new_hps = current_hps

        else:
            decision = "retrain"

            # Check overfitting (LLM)
            is_overfit, overfit_adjustments = check_overfitting(
                eval_results, task_type, overfitting_detection, current_hps
            )

            # Select strategy
            strategy_cls = self.STRATEGIES.get(tuning_strategy, ScheduleStrategy)
            strategy = strategy_cls(
                task_type=task_type,
                lr_decay_factor=lr_decay_factor,
                unfreeze_on_retry=unfreeze_on_retry,
                seed=seed,
            )

            # Parse search space
            if search_space != "auto":
                space = json.loads(search_space)
            else:
                space = strategy.default_search_space(task_type)

            # Generate new HPs
            new_hps = strategy.suggest(
                current_hps=current_hps,
                history=history,
                search_space=space,
                iteration=current_iteration,
            )

            # Apply overfitting overrides
            if is_overfit:
                new_hps.update(overfit_adjustments)
                reason = f"Overfitting detected. {metric_name} {current_value:.4f} < {metric_threshold}"
            else:
                reason = (f"{metric_name} {current_value:.4f} < {metric_threshold}, "
                          f"retrying with {json.dumps(new_hps)}")

        # ── Update history ──
        history.append({
            "iteration": current_iteration,
            "hyperparams": current_hps,
            "metrics": metrics,
            "decision": decision,
            "reason": reason,
        })

        # ── Write outputs ──
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)

        outputs = {
            "decision": decision,
            "ap": str(current_value),
            "hyperparams_json": json.dumps(new_hps),
            "hp_history": json.dumps(history),
            "next_iteration": str(current_iteration + 1),
        }

        for key, value in outputs.items():
            with open(os.path.join(output_dir, key), 'w') as f:
                f.write(value)

        # Full decision record for debugging
        full_output = {
            "decision": decision,
            "reason": reason,
            "metric_name": metric_name,
            "metric_value": current_value,
            "threshold": metric_threshold,
            "direction": direction,
            "iteration": current_iteration,
            "new_hyperparams": new_hps,
            "history_length": len(history),
        }
        output_path = os.path.join(output_dir, "decision_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"Decision: {decision} | Reason: {reason}")
        return output_path
```

### 8.6 Strategy Base Class

```python
# strategies.py

from abc import ABC, abstractmethod

class TuningStrategy(ABC):
    def __init__(self, task_type, lr_decay_factor=0.5,
                 unfreeze_on_retry=True, seed=-1):
        self.task_type = task_type
        self.lr_decay_factor = lr_decay_factor
        self.unfreeze_on_retry = unfreeze_on_retry
        self.seed = seed

    @abstractmethod
    def suggest(self, current_hps, history, search_space, iteration):
        """Return dict of new hyperparameters."""
        pass

    def default_search_space(self, task_type):
        """Return default search space for task type."""
        # ... (defined per strategy subclass)
        pass


class ScheduleStrategy(TuningStrategy):
    def suggest(self, current_hps, history, search_space, iteration):
        if self.task_type == "detection":
            return self._detection(current_hps, iteration)
        elif self.task_type == "classification":
            return self._classification(current_hps, iteration)
        elif self.task_type == "llm_finetune":
            return self._llm(current_hps, iteration)

    # ... _detection(), _classification(), _llm() as defined in Section 5.1


class GridStrategy(TuningStrategy):
    def suggest(self, current_hps, history, search_space, iteration):
        # ... grid_select() as defined in Section 5.2


class RandomStrategy(TuningStrategy):
    def suggest(self, current_hps, history, search_space, iteration):
        # ... random_select() as defined in Section 5.3
```

---

## 9. Argo Integration

### 9.1 DAG Task Definition (Approach A)

```yaml
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
      - name: tuning_strategy
        value: "{{workflow.parameters.tuning_strategy}}"
      - name: search_space
        value: "{{workflow.parameters.search_space}}"
      - name: lr_decay_factor
        value: "{{workflow.parameters.lr_decay_factor}}"
      - name: unfreeze_on_retry
        value: "{{workflow.parameters.unfreeze_on_retry}}"
      - name: seed
        value: "{{workflow.parameters.seed}}"
      - name: current_hyperparams
        value: "{{inputs.parameters.current_hyperparams}}"
```

### 9.2 Conditional Branches Consuming Outputs

```yaml
# Pass → Export
- name: export
  depends: "decide.Succeeded"
  when: "{{tasks.decide.outputs.parameters.decision}} == pass"

# Retrain → Recurse with new HPs
- name: retrain
  depends: "decide.Succeeded"
  when: "{{tasks.decide.outputs.parameters.decision}} == retrain"
  template: autoloop
  arguments:
    parameters:
      - name: current_iteration
        value: "{{tasks.decide.outputs.parameters.next_iteration}}"
      - name: current_hyperparams
        value: "{{tasks.decide.outputs.parameters.hyperparams_json}}"
      - name: hp_history
        value: "{{tasks.decide.outputs.parameters.hp_history}}"

# Fail → Report
- name: report-failure
  depends: "decide.Succeeded"
  when: "{{tasks.decide.outputs.parameters.decision}} == fail"
```

---

## 10. Testing Plan

### 10.1 Unit Tests

| # | Test | Input | Expected Output |
|---|------|-------|-----------------|
| 1 | Pass — detection | AP=0.65, threshold=0.50 | `decision="pass"` |
| 2 | Pass — LLM (minimize) | eval_loss=1.2, threshold=1.5 | `decision="pass"` |
| 3 | Retrain — schedule detection | AP=0.30, iter=1, max=3 | `decision="retrain"`, LR halved |
| 4 | Retrain — schedule LLM | eval_loss=2.0, iter=1, max=3 | `decision="retrain"`, LR halved, lora_r doubled |
| 5 | Retrain — grid | AP=0.30, grid with 3 combos, iter=1 | `decision="retrain"`, next grid point |
| 6 | Retrain — random | AP=0.30, log_uniform LR | `decision="retrain"`, valid random sample |
| 7 | Fail — max iterations | AP=0.30, iter=3, max=3 | `decision="fail"` |
| 8 | Fail — plateau | AP=0.31 (prev 0.30), min_delta=0.05 | `decision="fail"`, reason="Plateau" |
| 9 | Fail — grid exhausted | All 4 grid points tried, iter=3 | `decision="fail"` |
| 10 | Overfitting — LLM | train_loss=0.5, eval_loss=2.0 | lora_dropout increased |
| 11 | History accumulation | 2 prior entries | History has 3 entries after |
| 12 | Metric resolution — auto | task_type="classification" | metric="accuracy/top1", direction="maximize" |
| 13 | Metric resolution — explicit | primary_metric="AP50" | metric="AP50", direction="maximize" |
| 14 | Direction — loss metric | primary_metric="eval_loss" | direction="minimize" |
| 15 | Seed reproducibility | seed=42, random strategy | Same output on re-run |
| 16 | LR decay math | 0.001875 × 0.5³ | 0.000234375 |
| 17 | Frozen stages floor | frozen=1, 3 decrements | 1 → 0 → 0 → 0 |
| 18 | lora_r cap | lora_r=64, double | 128 (capped at 128) |
| 19 | num_epochs cap (classification) | epochs=400, × 1.5 | 500 (capped) |
| 20 | Backward compat — empty history | hp_history="[]" | Works without error |

### 10.2 Integration Tests

| # | Test | Method |
|---|------|--------|
| 1 | Argo YAML lint | `argo lint config.yaml` — validates param references |
| 2 | Forced retrain | Set threshold=0.99, run pipeline, verify retrain branch triggers |
| 3 | Forced pass | Set threshold=0.01, run pipeline, verify export branch triggers |
| 4 | Parameter chain | Verify `hyperparams_json` from decide step propagates to train step as `hyperparams_override` |
| 5 | History passthrough | Run 2 iterations, inspect `hp_history` param — should contain both entries |

---

## 11. Future Enhancements

| Enhancement | Version | Description |
|-------------|---------|-------------|
| LLM-guided tuning | v3 | Add `"llm"` strategy that calls a Clarifai-hosted LLM with the HP history as context, requesting structured JSON output |
| Parallel search | v2 | Add `withParam` fan-out for grid/random — run N configs simultaneously, pick best |
| Best-checkpoint revert | v2 | Track best-seen checkpoint path, export that instead of latest on "pass after degradation" |
| Bayesian optimization | v3 | Add `"bayesian"` strategy using a Gaussian Process surrogate (requires `scikit-optimize` dep) |
| Cross-validation | v3 | Run eval on multiple dataset splits, use mean metric for more robust decisions |
