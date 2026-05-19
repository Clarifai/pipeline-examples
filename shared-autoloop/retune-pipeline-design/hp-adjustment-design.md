# Design Document: Hyperparameter Adjustment Step

**Step ID**: `hp-adjust-ps`
**Status**: Draft
**Date**: 2026-04-23

---

## 1. Overview

The HP Adjustment Step generates the next set of hyperparameters for a retrain iteration. It is invoked **only when the upstream [Metric Decision Step](metric-decision-design.md) outputs `decision="retrain"`**.

This step is purely a **parameter generation engine**. It does not evaluate models, compare metrics, or make routing decisions. It takes the loop history and current hyperparameters, applies a tuning strategy, and outputs a new hyperparameter JSON for the next training step.

```
┌────────────────────┐        ┌───────────────────────────┐
│ Metric Decision    │ ──────▶│  HP Adjustment Step       │
│ Step               │        │  (CPU — no GPU)           │
│                    │        │                           │
│ decision="retrain" │        │  1. Load history          │
│ is_overfitting     │        │  2. Select strategy       │
│ hp_history         │        │  3. Apply task-specific   │
│                    │        │     HP adjustments        │
└────────────────────┘        │  4. Apply overfit fixes   │
                              │  5. Output new HPs        │
  ┌──────────────────┐        │                           │
  │ Workflow Params   │ ──────▶│  outputs:                │
  │                   │        │  • hyperparams_json      │
  │ tuning_strategy   │        │  • strategy_metadata     │
  │ search_space      │        └───────────┬──────────────┘
  │ lr_decay_factor   │                    │
  │ seed              │                    ▼
  └──────────────────┘              ┌─────────────┐
                                    │  Next Train  │
                                    │  Step        │
                                    └─────────────┘
```

### 1.1 Separation of Concerns

| Responsibility | Metric Decision Step | This Step (hp-adjust-ps) |
|---|---|---|
| Read eval metrics | Yes | No |
| Compare against threshold | Yes | No |
| Output routing decision | Yes | No |
| Maintain loop history | Yes | No |
| Read history for HP selection | No | **Yes** |
| Select tuning strategy | No | **Yes** |
| Generate new hyperparameters | No | **Yes** |
| Apply overfitting corrections | No | **Yes** |
| Define search space | No | **Yes** |

### 1.2 Design Goals

1. **Pluggable strategies** — Schedule, grid, and random strategies are interchangeable via the `tuning_strategy` parameter. Future strategies (Bayesian, LLM-guided) add new modules without touching the interface.
2. **Task-aware defaults** — Each task type (detection, classification, llm_finetune) has its own default search space and schedule rules.
3. **Overfitting-reactive** — When `is_overfitting=true` is passed from the decision step, HP adjustments prioritize regularization over exploration.
4. **Deterministic when needed** — Schedule strategy is fully deterministic. Grid and random support seeded reproducibility.

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

Execution time: < 1 second. This step only performs arithmetic and JSON manipulation.

---

## 3. Interface

### 3.1 Method Signature

```python
class HPAdjustment:
    def adjust(
        self,
        # ── Loop Context ──
        hp_history: str = "[]",                # JSON array from metric-decision step
        current_hyperparams: str = "{}",       # JSON of current iteration's HPs

        # ── Task Context ──
        task_type: str = "detection",          # "detection" | "classification" | "llm_finetune"

        # ── Strategy Selection ──
        tuning_strategy: str = "schedule",     # "schedule" | "grid" | "random"
        search_space: str = "auto",            # JSON dict or "auto" for task-type defaults

        # ── Schedule Strategy Knobs ──
        lr_decay_factor: float = 0.5,          # Multiply LR by this factor each iteration
        unfreeze_on_retry: bool = True,        # Reduce frozen_stages for vision tasks

        # ── Overfitting Signal ──
        is_overfitting: bool = False,          # From metric-decision step output

        # ── Reproducibility ──
        seed: int = 42,                        # Random seed for grid/random strategies
    ) -> str:
        """Returns path to /tmp/hp_output.json"""
```

**Notable absence**: No `eval_results_json`, `metric_threshold`, `max_retrain_iterations`, or `early_stop_*` params — those belong to the Metric Decision Step.

### 3.2 Inputs

| Input | Source | Purpose |
|-------|--------|---------|
| `hp_history` | Metric Decision Step output `hp_history` | Iteration history for history-aware selection |
| `current_hyperparams` | Workflow param or prior HP Adjust output | Base HPs to modify |
| `task_type` | Workflow param | Determines default search space and schedule rules |
| `tuning_strategy` | Workflow param | Which strategy to use |
| `search_space` | Workflow param (optional) | Custom search space override; `"auto"` uses task defaults |
| `lr_decay_factor` | Workflow param | Controls LR decay rate for schedule strategy |
| `unfreeze_on_retry` | Workflow param | Layer unfreezing toggle for vision schedule strategy |
| `is_overfitting` | Metric Decision Step output `is_overfitting` | Triggers regularization adjustments |
| `seed` | Workflow param | Deterministic random selection |

### 3.3 Outputs

| Output File | Argo Param Name | Type | Description |
|-------------|-----------------|------|-------------|
| `/tmp/hyperparams_json` | `hyperparams_json` | string (JSON dict) | New HP dict for the next training step |
| `/tmp/strategy_metadata` | `strategy_metadata` | string (JSON dict) | Debug info: strategy used, changes made, grid index, etc. |
| `/tmp/hp_output.json` | — | JSON file | Full output record for debugging |

**Output parameter declarations:**
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

**`hyperparams_json` example (detection):**
```json
{
  "per_item_lrate": 0.0009375,
  "frozen_stages": 0,
  "num_epochs": 12,
  "batch_size": 8
}
```

**`strategy_metadata` example:**
```json
{
  "strategy": "schedule",
  "iteration": 2,
  "changes": {
    "per_item_lrate": {"from": 0.001875, "to": 0.0009375, "rule": "decay_by_0.5"},
    "frozen_stages": {"from": 1, "to": 0, "rule": "unfreeze_one_stage"}
  },
  "overfit_adjustments": false
}
```

---

## 4. Tuning Strategies

### 4.1 Strategy: `schedule` (Default)

A deterministic, rule-based strategy that applies predictable HP adjustments per iteration. No randomness. Designed for production reliability.

#### 4.1.1 Detection (YOLOF / MMDetection)

```python
def schedule_detection(current_hps: dict, iteration: int, lr_decay: float, unfreeze: bool) -> dict:
    new_hps = current_hps.copy()

    # Learning rate decay
    current_lr = new_hps.get("per_item_lrate", 0.001875)
    new_hps["per_item_lrate"] = current_lr * lr_decay

    # Layer unfreezing
    if unfreeze:
        current_frozen = new_hps.get("frozen_stages", 1)
        if current_frozen > 0:
            new_hps["frozen_stages"] = current_frozen - 1

    return new_hps
```

**Schedule rules (per iteration):**

| Iteration | LR (factor=0.5) | frozen_stages | Notes |
|-----------|-----------------|---------------|-------|
| 1 (initial) | 0.001875 | 1 | User-supplied defaults |
| 2 | 0.0009375 | 0 | Halved LR, unfrozen 1 stage |
| 3 | 0.00046875 | 0 | Halved LR again, already at 0 |

#### 4.1.2 Classification (ResNet / MMPretrain)

```python
def schedule_classification(current_hps: dict, iteration: int, lr_decay: float, unfreeze: bool) -> dict:
    new_hps = current_hps.copy()

    # Learning rate decay
    current_lr = new_hps.get("per_item_lrate", 1.95e-5)
    new_hps["per_item_lrate"] = current_lr * lr_decay

    # Weight decay adjustment (slight increase for regularization)
    current_wd = new_hps.get("weight_decay", 0.01)
    new_hps["weight_decay"] = min(current_wd * 1.5, 0.1)  # cap at 0.1

    return new_hps
```

#### 4.1.3 LLM Fine-tuning (LoRA / Unsloth)

```python
def schedule_llm(current_hps: dict, iteration: int, lr_decay: float, **kwargs) -> dict:
    new_hps = current_hps.copy()

    # Learning rate decay
    current_lr = new_hps.get("learning_rate", 2e-4)
    new_hps["learning_rate"] = current_lr * lr_decay

    # LoRA rank increase (more capacity per retry)
    current_r = new_hps.get("lora_r", 16)
    if current_r < 128:
        new_hps["lora_r"] = min(current_r * 2, 128)
        new_hps["lora_alpha"] = new_hps["lora_r"]  # keep alpha = r

    return new_hps
```

**Schedule rules (per iteration):**

| Iteration | LR (factor=0.5) | lora_r | lora_alpha | Notes |
|-----------|-----------------|--------|------------|-------|
| 1 (initial) | 2e-4 | 16 | 16 | User-supplied defaults |
| 2 | 1e-4 | 32 | 32 | Double LoRA rank for more capacity |
| 3 | 5e-5 | 64 | 64 | Double again |

### 4.2 Strategy: `grid`

Exhaustive enumeration of predefined HP combinations. The grid index is determined by the current iteration number (modulo grid size).

#### 4.2.1 Search Space Format

```json
{
  "per_item_lrate": [0.001875, 0.0009375, 0.001],
  "frozen_stages": [0, 1],
  "num_epochs": [12]
}
```

Each key maps to a **list of values** to try. The full grid is the Cartesian product of all lists.

#### 4.2.2 Default Search Spaces

```python
GRID_DEFAULTS = {
    "detection": {
        "per_item_lrate": [0.001875, 0.0009375, 0.003],
        "frozen_stages": [0, 1],
    },
    "classification": {
        "per_item_lrate": [1.95e-5, 1e-5, 5e-5],
        "weight_decay": [0.01, 0.05],
    },
    "llm_finetune": {
        "learning_rate": [2e-4, 1e-4, 5e-5],
        "lora_r": [16, 32, 64],
    },
}
```

#### 4.2.3 Grid Selection Logic

```python
import itertools

def grid_select(
    current_hps: dict,
    search_space: dict,
    hp_history: list,
    seed: int,
) -> dict:
    """Select the next untried grid combination."""

    # Build full grid
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Exclude already-tried combinations
    tried = set()
    for entry in hp_history:
        combo_key = tuple(entry["hyperparams"].get(k) for k in keys)
        tried.add(combo_key)

    remaining = [
        combo for combo in all_combos
        if tuple(combo.get(k) for k in keys) not in tried
    ]

    if not remaining:
        # All combinations tried — restart from beginning
        remaining = all_combos

    # Take next candidate
    new_hps = current_hps.copy()
    new_hps.update(remaining[0])
    return new_hps
```

### 4.3 Strategy: `random`

Samples hyperparameters from defined distributions. Each iteration produces an independent random draw (seeded by `seed + iteration` for reproducibility).

#### 4.3.1 Search Space Format (Random)

```json
{
  "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
  "frozen_stages": {"type": "choice", "values": [0, 1]},
  "num_epochs": {"type": "discrete_uniform", "low": 6, "high": 24, "step": 6}
}
```

#### 4.3.2 Sampling Functions

```python
import random
import math

def sample_hp(spec: dict, rng: random.Random) -> float:
    """Sample a single HP value from its distribution specification."""
    dist_type = spec["type"]

    if dist_type == "log_uniform":
        log_low = math.log(spec["low"])
        log_high = math.log(spec["high"])
        return math.exp(rng.uniform(log_low, log_high))

    elif dist_type == "uniform":
        return rng.uniform(spec["low"], spec["high"])

    elif dist_type == "discrete_uniform":
        low = spec["low"]
        high = spec["high"]
        step = spec.get("step", 1)
        steps = list(range(low, high + 1, step))
        return rng.choice(steps)

    elif dist_type == "choice":
        return rng.choice(spec["values"])

    elif dist_type == "int_log_uniform":
        log_low = math.log(spec["low"])
        log_high = math.log(spec["high"])
        return round(math.exp(rng.uniform(log_low, log_high)))

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def random_select(
    current_hps: dict,
    search_space: dict,
    hp_history: list,
    seed: int,
) -> dict:
    """Sample new HPs from distributions defined in search_space."""
    iteration = len(hp_history) + 1
    rng = random.Random(seed + iteration)

    new_hps = current_hps.copy()
    for key, spec in search_space.items():
        new_hps[key] = sample_hp(spec, rng)

    return new_hps
```

#### 4.3.3 Default Random Distributions

| Task Type | Parameter | Distribution |
|-----------|-----------|-------------|
| detection | per_item_lrate | log_uniform(1e-5, 1e-2) |
| detection | frozen_stages | choice([0, 1]) |
| classification | per_item_lrate | log_uniform(1e-6, 1e-3) |
| classification | weight_decay | log_uniform(0.001, 0.1) |
| llm_finetune | learning_rate | log_uniform(1e-5, 5e-4) |
| llm_finetune | lora_r | choice([8, 16, 32, 64, 128]) |
| llm_finetune | lora_alpha | choice([8, 16, 32, 64, 128]) |

```python
RANDOM_DEFAULTS = {
    "detection": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "frozen_stages": {"type": "choice", "values": [0, 1]},
    },
    "classification": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
        "weight_decay": {"type": "log_uniform", "low": 0.001, "high": 0.1},
    },
    "llm_finetune": {
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
        "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
        "lora_alpha": {"type": "choice", "values": [8, 16, 32, 64, 128]},
    },
}
```

---

## 5. Overfitting Adjustments

When `is_overfitting=true` is passed from the metric decision step, the HP adjustment step applies regularization overrides **on top of** the selected strategy's output. These are applied regardless of which strategy was used.

```python
def apply_overfit_corrections(new_hps: dict, task_type: str) -> dict:
    """Override specific HPs to counter overfitting."""
    corrected = new_hps.copy()

    if task_type == "detection":
        # Reduce epochs to prevent further memorization
        current_epochs = corrected.get("num_epochs", 12)
        corrected["num_epochs"] = max(current_epochs // 2, 4)

    elif task_type == "classification":
        # Increase weight decay + reduce epochs
        current_wd = corrected.get("weight_decay", 0.01)
        corrected["weight_decay"] = min(current_wd * 3, 0.1)
        current_epochs = corrected.get("num_epochs", 200)
        corrected["num_epochs"] = max(current_epochs // 2, 20)

    elif task_type == "llm_finetune":
        # For LoRA: reduce rank (less capacity = less memorization)
        current_r = corrected.get("lora_r", 16)
        corrected["lora_r"] = max(current_r // 2, 4)
        corrected["lora_alpha"] = corrected["lora_r"]
        # Increase weight_decay if present
        current_wd = corrected.get("weight_decay", 0.0)
        corrected["weight_decay"] = max(current_wd, 0.01)

    return corrected
```

**Overfit adjustments override the strategy output.** Order of operations:

```
1. current_hyperparams (input)
       │
       ▼
2. Strategy applied (schedule / grid / random)
       │
       ▼
3. Overfitting corrections (if is_overfitting=true)
       │
       ▼
4. hyperparams_json (output)
```

---

## 6. Search Space Resolution

When `search_space="auto"`, the step resolves default search spaces based on `task_type` and `tuning_strategy`:

```python
def resolve_search_space(search_space: str, task_type: str, tuning_strategy: str) -> dict:
    """Resolve 'auto' to concrete search space based on task and strategy."""
    if search_space != "auto":
        return json.loads(search_space)

    if tuning_strategy == "grid":
        return GRID_DEFAULTS.get(task_type, {})
    elif tuning_strategy == "random":
        return RANDOM_DEFAULTS.get(task_type, {})
    else:
        return {}  # schedule doesn't use search_space
```

The schedule strategy ignores the search space entirely — it only uses `lr_decay_factor` and `unfreeze_on_retry`.

---

## 7. Implementation

### 7.1 File Structure

```
hp-adjust-ps/
├── config.yaml                # Pipeline step compute config
├── Dockerfile                 # Lightweight Python 3.11 image
├── requirements.txt           # Minimal: clarifai SDK only
└── 1/
    ├── pipeline_step.py       # Entry point (reflection pattern)
    └── models/
        └── model/
            ├── config.yaml    # Model config
            └── 1/
                ├── model.py     # HPAdjustment.adjust()
                └── strategies.py  # schedule/grid/random implementations
```

### 7.2 Core Implementation

```python
# model.py

import json
import logging
import inspect
import os

from strategies import (
    schedule_select,
    grid_select,
    random_select,
    apply_overfit_corrections,
    resolve_search_space,
    GRID_DEFAULTS,
    RANDOM_DEFAULTS,
)

logging.basicConfig(level=logging.INFO)


STRATEGY_DISPATCH = {
    "schedule": schedule_select,
    "grid": grid_select,
    "random": random_select,
}


class HPAdjustment:

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
        parser = argparse.ArgumentParser(description="Hyperparameter adjustment for retrain loop")
        sig = inspect.signature(cls.adjust)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def adjust(
        self,
        hp_history: str = "[]",
        current_hyperparams: str = "{}",
        task_type: str = "detection",
        tuning_strategy: str = "schedule",
        search_space: str = "auto",
        lr_decay_factor: float = 0.5,
        unfreeze_on_retry: bool = True,
        is_overfitting: bool = False,
        seed: int = 42,
    ) -> str:

        # ── Parse inputs ──
        history = json.loads(hp_history)
        current_hps = json.loads(current_hyperparams)
        resolved_space = resolve_search_space(search_space, task_type, tuning_strategy)
        iteration = len(history) + 1

        logging.info(f"[HP Adjust] Strategy: {tuning_strategy}, task: {task_type}, "
                     f"iteration: {iteration}, overfit: {is_overfitting}")

        # ── Select strategy and generate new HPs ──
        strategy_fn = STRATEGY_DISPATCH.get(tuning_strategy)
        if strategy_fn is None:
            raise ValueError(f"Unknown tuning_strategy: {tuning_strategy}. "
                             f"Valid options: {list(STRATEGY_DISPATCH.keys())}")

        if tuning_strategy == "schedule":
            new_hps = strategy_fn(
                current_hps=current_hps,
                task_type=task_type,
                iteration=iteration,
                lr_decay=lr_decay_factor,
                unfreeze=unfreeze_on_retry,
            )
        else:  # grid or random
            new_hps = strategy_fn(
                current_hps=current_hps,
                search_space=resolved_space,
                hp_history=history,
                seed=seed,
            )

        # ── Track changes before overfit corrections ──
        changes = {
            k: {"from": current_hps.get(k), "to": v}
            for k, v in new_hps.items()
            if current_hps.get(k) != v
        }

        # ── Apply overfitting corrections ──
        overfit_applied = False
        if is_overfitting:
            pre_overfit = new_hps.copy()
            new_hps = apply_overfit_corrections(new_hps, task_type)
            overfit_applied = any(new_hps.get(k) != pre_overfit.get(k) for k in new_hps)

        # ── Build metadata ──
        metadata = {
            "strategy": tuning_strategy,
            "task_type": task_type,
            "iteration": iteration,
            "changes": changes,
            "overfit_adjustments": overfit_applied,
            "search_space_used": resolved_space if tuning_strategy != "schedule" else None,
            "seed": seed if tuning_strategy in ("grid", "random") else None,
        }

        # ── Write outputs ──
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)

        outputs = {
            "hyperparams_json": json.dumps(new_hps),
            "strategy_metadata": json.dumps(metadata),
        }
        for key, value in outputs.items():
            with open(os.path.join(output_dir, key), 'w') as f:
                f.write(value)

        # Full record for debugging
        full_output = {
            "input_hyperparams": current_hps,
            "output_hyperparams": new_hps,
            "metadata": metadata,
        }
        output_path = os.path.join(output_dir, "hp_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"[HP Adjust] Changes: {changes}")
        if overfit_applied:
            logging.info(f"[HP Adjust] Overfit corrections applied for {task_type}")

        return output_path
```

### 7.3 Strategies Module

```python
# strategies.py

import itertools
import json
import math
import random


# ═══════════════════════════════════════════════════════════
# SCHEDULE STRATEGY
# ═══════════════════════════════════════════════════════════

def schedule_select(current_hps, task_type, iteration, lr_decay, unfreeze):
    dispatch = {
        "detection": _schedule_detection,
        "classification": _schedule_classification,
        "llm_finetune": _schedule_llm,
    }
    fn = dispatch.get(task_type, _schedule_detection)
    return fn(current_hps, iteration, lr_decay, unfreeze)


def _schedule_detection(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["per_item_lrate"] = current_hps.get("per_item_lrate", 0.001875) * lr_decay
    if unfreeze:
        frozen = new_hps.get("frozen_stages", 1)
        if frozen > 0:
            new_hps["frozen_stages"] = frozen - 1
    return new_hps


def _schedule_classification(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["per_item_lrate"] = current_hps.get("per_item_lrate", 1.95e-5) * lr_decay
    current_wd = new_hps.get("weight_decay", 0.01)
    new_hps["weight_decay"] = min(current_wd * 1.5, 0.1)
    return new_hps


def _schedule_llm(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["learning_rate"] = current_hps.get("learning_rate", 2e-4) * lr_decay
    current_r = new_hps.get("lora_r", 16)
    if current_r < 128:
        new_hps["lora_r"] = min(current_r * 2, 128)
        new_hps["lora_alpha"] = new_hps["lora_r"]
    return new_hps


# ═══════════════════════════════════════════════════════════
# GRID STRATEGY
# ═══════════════════════════════════════════════════════════

GRID_DEFAULTS = {
    "detection": {
        "per_item_lrate": [0.001875, 0.0009375, 0.003],
        "frozen_stages": [0, 1],
    },
    "classification": {
        "per_item_lrate": [1.95e-5, 1e-5, 5e-5],
        "weight_decay": [0.01, 0.05],
    },
    "llm_finetune": {
        "learning_rate": [2e-4, 1e-4, 5e-5],
        "lora_r": [16, 32, 64],
    },
}


def grid_select(current_hps, search_space, hp_history, seed):
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    tried = set()
    for entry in hp_history:
        combo_key = tuple(entry["hyperparams"].get(k) for k in keys)
        tried.add(combo_key)

    remaining = [
        combo for combo in all_combos
        if tuple(combo.get(k) for k in keys) not in tried
    ]

    if not remaining:
        remaining = all_combos

    new_hps = current_hps.copy()
    new_hps.update(remaining[0])
    return new_hps


# ═══════════════════════════════════════════════════════════
# RANDOM STRATEGY
# ═══════════════════════════════════════════════════════════

RANDOM_DEFAULTS = {
    "detection": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "frozen_stages": {"type": "choice", "values": [0, 1]},
    },
    "classification": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
        "weight_decay": {"type": "log_uniform", "low": 0.001, "high": 0.1},
    },
    "llm_finetune": {
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
        "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
        "lora_alpha": {"type": "choice", "values": [8, 16, 32, 64, 128]},
    },
}


def _sample_hp(spec, rng):
    dist_type = spec["type"]
    if dist_type == "log_uniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    elif dist_type == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    elif dist_type == "discrete_uniform":
        steps = list(range(spec["low"], spec["high"] + 1, spec.get("step", 1)))
        return rng.choice(steps)
    elif dist_type == "choice":
        return rng.choice(spec["values"])
    elif dist_type == "int_log_uniform":
        return round(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def random_select(current_hps, search_space, hp_history, seed):
    iteration = len(hp_history) + 1
    rng = random.Random(seed + iteration)
    new_hps = current_hps.copy()
    for key, spec in search_space.items():
        new_hps[key] = _sample_hp(spec, rng)
    return new_hps


# ═══════════════════════════════════════════════════════════
# OVERFITTING CORRECTIONS
# ═══════════════════════════════════════════════════════════

def apply_overfit_corrections(new_hps, task_type):
    corrected = new_hps.copy()
    if task_type == "detection":
        current_epochs = corrected.get("num_epochs", 12)
        corrected["num_epochs"] = max(current_epochs // 2, 4)
    elif task_type == "classification":
        current_wd = corrected.get("weight_decay", 0.01)
        corrected["weight_decay"] = min(current_wd * 3, 0.1)
        current_epochs = corrected.get("num_epochs", 200)
        corrected["num_epochs"] = max(current_epochs // 2, 20)
    elif task_type == "llm_finetune":
        current_r = corrected.get("lora_r", 16)
        corrected["lora_r"] = max(current_r // 2, 4)
        corrected["lora_alpha"] = corrected["lora_r"]
        current_wd = corrected.get("weight_decay", 0.0)
        corrected["weight_decay"] = max(current_wd, 0.01)
    return corrected


# ═══════════════════════════════════════════════════════════
# SEARCH SPACE RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_search_space(search_space, task_type, tuning_strategy):
    if search_space != "auto":
        return json.loads(search_space)
    if tuning_strategy == "grid":
        return GRID_DEFAULTS.get(task_type, {})
    elif tuning_strategy == "random":
        return RANDOM_DEFAULTS.get(task_type, {})
    return {}
```

### 7.4 Pipeline Step Entry Point

```python
# pipeline_step.py

import sys
from pathlib import Path

model_module = __import__("model.1.model", fromlist=[''])
model_class = [
    obj for name in dir(model_module)
    if isinstance(obj := getattr(model_module, name), type)
    and hasattr(obj, 'adjust')
][0]

def main():
    args = model_class.to_pipeline_parser().parse_args()
    model_class().adjust(**vars(args))

if __name__ == "__main__":
    main()
```

### 7.5 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /home/nonroot/main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 1 /home/nonroot/main/1

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main

ENTRYPOINT ["python3", "1/pipeline_step.py"]
```

### 7.6 requirements.txt

```
clarifai>=12.1.4,<13.0.0
```

### 7.7 Pipeline Step Config

```yaml
pipeline_step:
  id: "hp-adjust-ps"
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
  - hp_history
  - current_hyperparams
  - task_type
  - tuning_strategy
  - search_space
  - lr_decay_factor
  - unfreeze_on_retry
  - is_overfitting
  - seed
```

---

## 8. Argo Integration

### 8.1 DAG Task Definition

```yaml
# ── STEP 4: HP Adjustment (only on retrain) ──
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
```

### 8.2 Data Flow to Next Training Iteration

```yaml
# ── STEP 5: Recursive retrain ──
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
```

Note: `hp_history` comes from the **decide** step (which maintains history), while `current_hyperparams` comes from **hp-adjust** (which generates the next set). This clean separation means the HP adjust step never needs to know about history management.

---

## 9. Testing Plan

| # | Test | Strategy | Input | Expected |
|---|------|----------|-------|----------|
| 1 | Schedule — detection LR decay | schedule | LR=0.001875, factor=0.5 | LR=0.0009375 |
| 2 | Schedule — detection unfreeze | schedule | frozen=1, unfreeze=true | frozen=0 |
| 3 | Schedule — detection no unfreeze | schedule | frozen=1, unfreeze=false | frozen=1 |
| 4 | Schedule — classification WD increase | schedule | WD=0.01 | WD=0.015 |
| 5 | Schedule — LLM rank double | schedule | lora_r=16 | lora_r=32, lora_alpha=32 |
| 6 | Schedule — LLM rank cap | schedule | lora_r=128 | lora_r=128 (no change) |
| 7 | Grid — first combo | grid | empty history, 3x2 grid | First combo (index 0) |
| 8 | Grid — skip tried | grid | 1 tried combo | 2nd combo |
| 9 | Grid — exhausted restart | grid | all combos tried | Back to first |
| 10 | Random — deterministic | random | seed=42, iter=1 | Same output every time |
| 11 | Random — different seeds differ | random | seed=42 vs seed=99 | Different outputs |
| 12 | Overfit — detection | schedule | is_overfitting=true, epochs=12 | epochs=6 |
| 13 | Overfit — LLM | schedule | is_overfitting=true, lora_r=64 | lora_r=32, WD≥0.01 |
| 14 | Overfit — classification | schedule | is_overfitting=true, WD=0.01, epochs=200 | WD=0.03, epochs=100 |
| 15 | Overfit + grid interaction | grid | is_overfitting=true | Grid applied, then overfit corrections override |
| 16 | Auto search space — grid detection | grid | search_space="auto" | Uses GRID_DEFAULTS["detection"] |
| 17 | Auto search space — random LLM | random | search_space="auto" | Uses RANDOM_DEFAULTS["llm_finetune"] |
| 18 | Custom search space | grid | explicit JSON | Uses provided space, not defaults |
| 19 | Unknown strategy | — | tuning_strategy="bayesian" | ValueError raised |
| 20 | Empty current_hyperparams | schedule | current_hyperparams="{}" | Uses default values from schedule fn |

---

## 10. Future Extensions (v2/v3)

| Version | Feature | Impact on This Step |
|---------|---------|---------------------|
| v2 | Bayesian optimization (Optuna) | New strategy module in strategies.py, new `"bayesian"` dispatch entry. Requires `optuna` in requirements.txt. |
| v2 | Parallel trials (`withParam` fan-out) | This step outputs N HP sets as a JSON array instead of a single dict. Argo's `withParam` iterates over the array. |
| v3 | LLM-guided HP selection | New strategy that sends history + search space to an LLM API, parses structured response. Requires API key management. |
| v3 | Multi-objective optimization | Accept multiple metrics + weights. Strategy must balance exploration across objectives. |
