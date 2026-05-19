# Design Document: Metric-Based Conditional Branching Step

**Step ID**: `metric-decision-ps`
**Status**: Draft
**Date**: 2026-04-23

---

## 1. Overview

The Metric Decision Step is a lightweight, GPU-free pipeline step that reads evaluation metrics from a previous eval step, compares them against configurable thresholds, and outputs a routing decision. It enables **conditional DAG execution in Argo Workflows** — branching the pipeline into deploy, retrain, or stop paths.

This step is **purely a decision gate**. It does not adjust hyperparameters — that is the responsibility of the separate [HP Adjustment Step](hp-adjustment-design.md).

```
┌──────────┐     ┌─────────────────────────┐
│ Eval Step│────▶│ Metric Decision Step     │
│ (GPU)    │     │ (CPU — no GPU)           │
│          │     │                          │
│ outputs: │     │  1. Load eval metrics    │
│ eval_    │     │  2. Resolve metric name  │
│ results  │     │  3. Compare vs threshold │
│ .json    │     │  4. Check early stopping │──── decision="deploy" ──▶ [Export Step]
└──────────┘     │  5. Check overfitting    │
                 │  6. Output decision      │──── decision="retrain" ──▶ [HP Adjust Step]
                 │                          │
                 │  outputs:                │──── decision="stop" ──▶ [Failure Report]
                 │  • decision              │
                 │  • metric_value          │
                 │  • hp_history (updated)  │
                 │  • is_overfitting        │
                 └─────────────────────────┘
```

### 1.1 Separation of Concerns

| Responsibility | This Step (metric-decision-ps) | HP Adjustment Step (hp-adjust-ps) |
|---|---|---|
| Read eval metrics | **Yes** | No |
| Compare against threshold | **Yes** | No |
| Detect early stopping / plateau | **Yes** | No |
| Detect overfitting | **Yes** | No |
| Output routing decision | **Yes** | No |
| Maintain loop history | **Yes** (append current metrics) | No (reads history, doesn't modify) |
| Select tuning strategy | No | **Yes** |
| Generate new hyperparameters | No | **Yes** |
| Define search space | No | **Yes** |

### 1.2 Why Decouple?

1. **Independent replacement** — You can swap the HP strategy (grid → Bayesian → LLM-guided) without touching the decision logic, and vice versa.
2. **Reuse outside the loop** — The metric decision step can be used in any pipeline that needs conditional branching based on eval metrics, even without retraining.
3. **Simpler testing** — Each step has fewer code paths. The decision step is pure comparison logic; the HP step is pure generation logic.
4. **Different evolution cadences** — Decision logic is stable (thresholds don't change often); HP strategies evolve rapidly (new search algorithms, LLM-guided tuning).

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

Execution time: < 2 seconds. This step only parses JSON and does float comparisons.

---

## 3. Interface

### 3.1 Method Signature

```python
class MetricDecision:
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

        # ── Current HPs (for history tracking only) ──
        current_hyperparams: str = "{}",     # JSON of current iteration's HPs

        # ── Stopping Controls ──
        early_stop_min_delta: float = 0.0,   # 0 = disabled; positive = min improvement
        overfitting_detection: bool = False,  # Check train_loss vs eval_loss divergence
    ) -> str:
        """Returns path to /tmp/decision_output.json"""
```

**Notable absence**: No `tuning_strategy`, `search_space`, `lr_decay_factor`, `unfreeze_on_retry`, or `seed` — these belong to the HP Adjustment Step.

### 3.2 Inputs

| Input | Source | Format |
|-------|--------|--------|
| `eval_results_json` | Eval step Argo output param | File path to JSON: `{"metrics": {"AP": 0.32, ...}, "train_loss": 0.5}` |
| `current_hyperparams` | Workflow params or prior HP Adjust step output | JSON string: `{"per_item_lrate": 0.001875, "frozen_stages": 1}` |
| `hp_history` | Prior metric-decision step output (accumulated) | JSON array: `[{"iteration": 1, "hyperparams": {...}, "metrics": {...}, "decision": "retrain"}, ...]` |

### 3.3 Outputs

| Output File | Argo Param Name | Type | Values | Description |
|-------------|-----------------|------|--------|-------------|
| `/tmp/decision` | `decision` | string | `"deploy"`, `"retrain"`, `"stop"` | Routing decision for Argo `when` conditions |
| `/tmp/metric_value` | `metric_value` | string (float) | e.g., `"0.3200"` | Current primary metric value |
| `/tmp/metric_name` | `metric_name` | string | e.g., `"AP"` | Resolved metric name |
| `/tmp/hp_history` | `hp_history` | string (JSON array) | `[{...}, ...]` | Updated history with current iteration appended |
| `/tmp/is_overfitting` | `is_overfitting` | string (bool) | `"true"` / `"false"` | Whether overfitting was detected (passed to HP step) |
| `/tmp/current_iteration` | `current_iteration` | string (int) | e.g., `"2"` | Passthrough for downstream steps |
| `/tmp/next_iteration` | `next_iteration` | string (int) | e.g., `"3"` | Incremented iteration counter |
| `/tmp/decision_output.json` | — | JSON file | Full record | Complete decision record for debugging |

#### Decision Value Naming

The decision values are renamed from the original design for clarity:
- `"pass"` → **`"deploy"`** — unambiguous intent: export and upload the model
- `"retrain"` → **`"retrain"`** — stays the same
- `"fail"` → **`"stop"`** — clearer: the pipeline stops, no more retries

#### Argo Output Parameter Declarations

```yaml
outputs:
  parameters:
    - name: decision
      valueFrom:
        path: /tmp/decision
    - name: metric_value
      valueFrom:
        path: /tmp/metric_value
        default: "0"
    - name: metric_name
      valueFrom:
        path: /tmp/metric_name
        default: "unknown"
    - name: hp_history
      valueFrom:
        path: /tmp/hp_history
        default: "[]"
    - name: is_overfitting
      valueFrom:
        path: /tmp/is_overfitting
        default: "false"
    - name: current_iteration
      valueFrom:
        path: /tmp/current_iteration
        default: "1"
    - name: next_iteration
      valueFrom:
        path: /tmp/next_iteration
        default: "2"
```

---

## 4. Decision Logic

### 4.1 Flowchart

```
                    ┌──────────────────────────┐
                    │  Load eval_results.json   │
                    │  Parse metrics dict       │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  Resolve metric name      │
                    │  & direction from         │
                    │  task_type + overrides     │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  Extract metric value     │
                    │  from eval results        │
                    └──────────┬───────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │  Metric meets threshold?             │
              │  maximize: value >= threshold        │
              │  minimize: value <= threshold         │
              └───────┬──────────────────┬──────────┘
                      │ YES              │ NO
               ┌──────▼──────┐           │
               │ decision =  │           │
               │ "deploy"    │           │
               └─────────────┘           │
                                         │
                              ┌──────────▼──────────────┐
                              │ Early stop check:        │
                              │ improvement from last    │
                              │ iteration < min_delta?   │
                              └───┬──────────────┬──────┘
                                  │ YES          │ NO
                           ┌──────▼──────┐       │
                           │ decision =  │       │
                           │ "stop"      │       │
                           │ reason:     │       │
                           │ "plateau"   │       │
                           └─────────────┘       │
                                                 │
                                      ┌──────────▼──────────┐
                                      │ iteration >= max?    │
                                      └───┬──────────┬──────┘
                                          │ YES      │ NO
                                   ┌──────▼──────┐   │
                                   │ decision =  │   │
                                   │ "stop"      │   │
                                   │ reason:     │   │
                                   │ "max_iters" │   │
                                   └─────────────┘   │
                                                     │
                                          ┌──────────▼──────────┐
                                          │ Overfitting check   │
                                          │ (if enabled)        │
                                          │ train_loss <<       │
                                          │   eval_loss?        │
                                          └───┬─────────┬──────┘
                                              │ YES     │ NO
                                              ▼         ▼
                                      is_overfitting  is_overfitting
                                      = true          = false
                                              │         │
                                              └────┬────┘
                                                   │
                                            ┌──────▼──────┐
                                            │ decision =  │
                                            │ "retrain"   │
                                            └─────────────┘
```

### 4.2 Metric Resolution

```python
METRIC_DEFAULTS = {
    "detection":      {"metric": "AP",            "direction": "maximize"},
    "classification": {"metric": "accuracy/top1",  "direction": "maximize"},
    "llm_finetune":   {"metric": "eval_loss",     "direction": "minimize"},
}

MINIMIZE_PATTERNS = ["loss", "perplexity", "error", "cer", "wer"]

def resolve_metric(primary_metric: str, metric_direction: str, task_type: str):
    """Resolve 'auto' values to concrete metric name and direction."""
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
def metric_passes(value: float, threshold: float, direction: str) -> bool:
    """Check if metric meets the quality gate."""
    if direction == "maximize":
        return value >= threshold
    else:
        return value <= threshold
```

### 4.4 Early Stop Check

```python
def should_early_stop(
    hp_history: list,
    current_value: float,
    primary_metric: str,
    metric_direction: str,
    min_delta: float,
) -> bool:
    """Detect diminishing returns between latest two iterations."""
    if min_delta <= 0 or len(hp_history) < 1:
        return False

    prev_value = hp_history[-1]["metrics"].get(primary_metric)
    if prev_value is None:
        return False

    if metric_direction == "maximize":
        improvement = current_value - prev_value
    else:
        improvement = prev_value - current_value  # for loss, decrease = improvement

    return improvement < min_delta
```

### 4.5 Overfitting Detection

Enabled primarily for LLM fine-tuning where train_loss often diverges sharply from eval_loss.

```python
def detect_overfitting(
    eval_results: dict,
    task_type: str,
    overfitting_detection: bool,
) -> bool:
    """Check if train_loss is significantly below eval_loss (heavy overfitting)."""
    if not overfitting_detection:
        return False

    metrics = eval_results.get("metrics", {})
    train_loss = metrics.get("train_loss")
    eval_loss = metrics.get("eval_loss")

    if train_loss is None or eval_loss is None:
        return False

    # train_loss less than half of eval_loss → heavy overfitting
    return train_loss < eval_loss * 0.5
```

The step does NOT adjust HPs for overfitting — it simply sets `is_overfitting=true` as an output parameter. The downstream HP Adjustment Step reads this flag and applies appropriate regularization adjustments (increase dropout, reduce epochs, etc.).

### 4.6 History Management

The decision step is responsible for maintaining the iteration history. On each invocation:

1. Parse incoming `hp_history` JSON array
2. Append the current iteration's record
3. Write the updated array to `/tmp/hp_history`

```python
def update_history(
    hp_history: list,
    current_iteration: int,
    current_hyperparams: dict,
    metrics: dict,
    decision: str,
    reason: str,
) -> list:
    """Append current iteration to history."""
    hp_history.append({
        "iteration": current_iteration,
        "hyperparams": current_hyperparams,
        "metrics": metrics,
        "decision": decision,
        "reason": reason,
    })
    return hp_history
```

**History format:**
```json
[
  {
    "iteration": 1,
    "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
    "metrics": {"AP": 0.32, "AP50": 0.55, "AP75": 0.28},
    "decision": "retrain",
    "reason": "AP 0.3200 < threshold 0.50"
  }
]
```

Each entry is ~200–500 bytes. At 10 iterations max, the full history is < 5KB — well within Argo's output parameter size limits (256KB default).

---

## 5. Implementation

### 5.1 File Structure

```
metric-decision-ps/
├── config.yaml                # Pipeline step compute config
├── Dockerfile                 # Lightweight Python 3.11 image
├── requirements.txt           # Minimal: clarifai SDK only
└── 1/
    ├── pipeline_step.py       # Entry point (reflection pattern)
    └── models/
        └── model/
            ├── config.yaml    # Model config
            └── 1/
                └── model.py   # MetricDecision.decide()
```

### 5.2 Core Implementation

```python
# model.py

import json
import logging
import inspect
import os

logging.basicConfig(level=logging.INFO)


METRIC_DEFAULTS = {
    "detection":      {"metric": "AP",            "direction": "maximize"},
    "classification": {"metric": "accuracy/top1",  "direction": "maximize"},
    "llm_finetune":   {"metric": "eval_loss",     "direction": "minimize"},
}

MINIMIZE_PATTERNS = ["loss", "perplexity", "error", "cer", "wer"]


class MetricDecision:

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
        parser = argparse.ArgumentParser(description="Metric-based conditional branching")
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
        metric_name, direction = self._resolve_metric(primary_metric, metric_direction, task_type)
        current_value = metrics.get(metric_name, 0.0)
        logging.info(f"[Iteration {current_iteration}] {metric_name} = {current_value:.4f} "
                     f"(threshold: {metric_threshold}, direction: {direction})")

        # ── Decision ──
        reason = ""
        is_overfit = False

        if self._metric_passes(current_value, metric_threshold, direction):
            decision = "deploy"
            reason = f"{metric_name} {current_value:.4f} meets threshold {metric_threshold}"

        elif early_stop_min_delta > 0 and self._should_early_stop(
            history, current_value, metric_name, direction, early_stop_min_delta
        ):
            decision = "stop"
            reason = f"Plateau: improvement below {early_stop_min_delta}"

        elif current_iteration >= max_retrain_iterations:
            decision = "stop"
            reason = (f"{metric_name} {current_value:.4f} below threshold {metric_threshold} "
                      f"after {current_iteration} iterations")

        else:
            decision = "retrain"
            reason = f"{metric_name} {current_value:.4f} below threshold {metric_threshold}"
            is_overfit = self._detect_overfitting(eval_results, task_type, overfitting_detection)
            if is_overfit:
                reason += " (overfitting detected)"

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
            "metric_value": f"{current_value:.6f}",
            "metric_name": metric_name,
            "hp_history": json.dumps(history),
            "is_overfitting": str(is_overfit).lower(),
            "current_iteration": str(current_iteration),
            "next_iteration": str(current_iteration + 1),
        }
        for key, value in outputs.items():
            with open(os.path.join(output_dir, key), 'w') as f:
                f.write(value)

        # Full record for debugging
        full_output = {
            **outputs,
            "threshold": metric_threshold,
            "direction": direction,
            "task_type": task_type,
            "history_length": len(history),
        }
        output_path = os.path.join(output_dir, "decision_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"Decision: {decision} | Reason: {reason}")
        return output_path

    @staticmethod
    def _resolve_metric(primary_metric, metric_direction, task_type):
        if primary_metric == "auto":
            primary_metric = METRIC_DEFAULTS[task_type]["metric"]
        if metric_direction == "auto":
            if any(p in primary_metric.lower() for p in MINIMIZE_PATTERNS):
                metric_direction = "minimize"
            else:
                metric_direction = METRIC_DEFAULTS.get(task_type, {}).get("direction", "maximize")
        return primary_metric, metric_direction

    @staticmethod
    def _metric_passes(value, threshold, direction):
        if direction == "maximize":
            return value >= threshold
        return value <= threshold

    @staticmethod
    def _should_early_stop(hp_history, current_value, primary_metric, direction, min_delta):
        if min_delta <= 0 or len(hp_history) < 1:
            return False
        prev_value = hp_history[-1]["metrics"].get(primary_metric)
        if prev_value is None:
            return False
        if direction == "maximize":
            improvement = current_value - prev_value
        else:
            improvement = prev_value - current_value
        return improvement < min_delta

    @staticmethod
    def _detect_overfitting(eval_results, task_type, overfitting_detection):
        if not overfitting_detection:
            return False
        metrics = eval_results.get("metrics", {})
        train_loss = metrics.get("train_loss")
        eval_loss = metrics.get("eval_loss")
        if train_loss is None or eval_loss is None:
            return False
        return train_loss < eval_loss * 0.5
```

### 5.3 Pipeline Step Entry Point

```python
# pipeline_step.py

import sys
from pathlib import Path

model_module = __import__("model.1.model", fromlist=[''])
model_class = [
    obj for name in dir(model_module)
    if isinstance(obj := getattr(model_module, name), type)
    and hasattr(obj, 'decide')
][0]

def main():
    args = model_class.to_pipeline_parser().parse_args()
    model_class().decide(**vars(args))

if __name__ == "__main__":
    main()
```

### 5.4 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /home/nonroot/main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 1 /home/nonroot/main/1

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main

ENTRYPOINT ["python3", "1/pipeline_step.py"]
```

### 5.5 requirements.txt

```
clarifai>=12.1.4,<13.0.0
```

### 5.6 Pipeline Step Config

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
  - current_hyperparams
  - early_stop_min_delta
  - overfitting_detection
```

---

## 6. Argo Integration

### 6.1 DAG Task Definition

```yaml
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
```

### 6.2 Downstream Conditional Branches

```yaml
# ── BRANCH: Deploy ──
- name: export
  depends: "decide.Succeeded"
  when: "{{tasks.decide.outputs.parameters.decision}} == deploy"
  templateRef:
    name: <model-export-ps-ref>
    template: <model-export-ps-template>
  arguments:
    parameters:
      - name: checkpoint_path
        value: "{{tasks.train.outputs.parameters.checkpoint_path}}"

# ── BRANCH: Retrain → HP Adjust → Recurse ──
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
      - name: is_overfitting
        value: "{{tasks.decide.outputs.parameters.is_overfitting}}"
      # ... tuning strategy params ...

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

# ── BRANCH: Stop ──
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
```

### 6.3 Updated DAG Flow (with decoupled steps)

```
┌─────────┐     ┌──────────┐     ┌──────────────┐
│  Train   │────▶│ Evaluate │────▶│ Metric       │
│ (GPU)    │     │ (GPU)    │     │ Decision     │
└─────────┘     └──────────┘     │ (CPU)        │
                                  └──────┬───────┘
                                         │
                   ┌─────────────────────┬┴──────────────────┐
                   ▼                     ▼                    ▼
           decision="deploy"     decision="retrain"   decision="stop"
                   │                     │                    │
             ┌─────▼─────┐       ┌──────▼───────┐    ┌──────▼──────┐
             │  Export &  │       │  HP Adjust   │    │  Report     │
             │  Upload    │       │  (CPU)       │    │  Failure    │
             └───────────┘       └──────┬───────┘    └─────────────┘
                                        │
                                        ▼
                                 ┌─────────┐
                                 │  Train   │ (iteration N+1)
                                 │  ...     │
                                 └─────────┘
```

---

## 7. Testing Plan

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | Deploy — detection | AP=0.65, threshold=0.50 | `decision="deploy"` |
| 2 | Deploy — LLM (minimize) | eval_loss=1.2, threshold=1.5 | `decision="deploy"` |
| 3 | Deploy — classification | accuracy=0.90, threshold=0.85 | `decision="deploy"` |
| 4 | Retrain — below threshold | AP=0.30, threshold=0.50, iter=1, max=3 | `decision="retrain"` |
| 5 | Stop — max iterations | AP=0.30, threshold=0.50, iter=3, max=3 | `decision="stop"` |
| 6 | Stop — plateau | AP=0.31 (prev 0.30), min_delta=0.05 | `decision="stop"`, reason="Plateau" |
| 7 | Retrain + overfitting | train_loss=0.5, eval_loss=2.0, detection=true | `decision="retrain"`, `is_overfitting="true"` |
| 8 | Retrain — no overfitting flag | train_loss=0.5, eval_loss=2.0, detection=false | `is_overfitting="false"` |
| 9 | Metric resolution — auto detection | task_type="detection", primary_metric="auto" | metric_name="AP", direction="maximize" |
| 10 | Metric resolution — auto LLM | task_type="llm_finetune", primary_metric="auto" | metric_name="eval_loss", direction="minimize" |
| 11 | Metric resolution — explicit override | primary_metric="AP50", metric_direction="maximize" | metric_name="AP50" |
| 12 | Metric resolution — loss pattern | primary_metric="eval_loss" | direction="minimize" (auto) |
| 13 | History accumulation | 2 prior entries, current decision="retrain" | History has 3 entries |
| 14 | Empty history | hp_history="[]" | Works, no early stop |
| 15 | Threshold boundary — exact match | AP=0.50, threshold=0.50, direction="maximize" | `decision="deploy"` (>=) |
| 16 | Threshold boundary — minimize exact | eval_loss=1.5, threshold=1.5, direction="minimize" | `decision="deploy"` (<=) |

---

## 8. Standalone Usage (Outside Autoloop)

This step can be used independently in any Argo pipeline that needs metric-based branching:

```yaml
# Example: Simple train → eval → deploy-or-stop (no retrain loop)
templates:
  - name: conditional-deploy
    dag:
      tasks:
        - name: train
          templateRef: {name: <any-train-step>}
        - name: eval
          depends: "train.Succeeded"
          templateRef: {name: <any-eval-step>}
        - name: decide
          depends: "eval.Succeeded"
          templateRef: {name: <metric-decision-ps>}
          arguments:
            parameters:
              - name: max_retrain_iterations
                value: "0"  # No retrain — just deploy or stop
        - name: export
          depends: "decide.Succeeded"
          when: "{{tasks.decide.outputs.parameters.decision}} == deploy"
          templateRef: {name: <export-step>}
```

Setting `max_retrain_iterations=0` turns this into a simple quality gate with no loop.
