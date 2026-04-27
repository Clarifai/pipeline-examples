# Parameter Reference: HP Adjustment Step (`hp-adjust-ps`)

This document describes every parameter accepted and produced by the HP adjustment step. The step is a lightweight, GPU-free pipeline component that generates the next set of hyperparameters for a retrain iteration. It is only invoked when the upstream metric decision step outputs `decision="retrain"`.

See also: [hp-adjustment-design.md](hp-adjustment-design.md) | [hp-adjustment-parameters.csv](hp-adjustment-parameters.csv)

---

## 1. Loop Context

#### `hp_history`
| | |
|---|---|
| **Type** | string (JSON array) |
| **Default** | `"[]"` |
| **Source** | Metric decision step output `hp_history` |

Iteration history maintained by the metric decision step. This step reads it but **does not modify it** — history management is solely the decision step's responsibility.

**How each strategy uses history:**

| Strategy | Usage |
|----------|-------|
| `schedule` | Reads `len(hp_history)` to know the iteration number for applying rules |
| `grid` | Reads `hp_history[*].hyperparams` to skip already-tried combinations |
| `random` | Uses `len(hp_history) + 1` for iteration-based seed derivation |

**Schema per entry:**
```json
{
  "iteration": 1,
  "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
  "metrics": {"AP": 0.3200},
  "decision": "retrain",
  "reason": "AP 0.3200 below threshold 0.50"
}
```

#### `current_hyperparams`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Default** | `"{}"` |
| **Source** | Workflow parameter (initial) or prior hp-adjust step output `hyperparams_json` |

The base hyperparameters that the strategy modifies. The output `hyperparams_json` is built by copying this dict, then applying strategy-specific changes on top.

**Detection example:**
```json
{"per_item_lrate": 0.001875, "frozen_stages": 1, "num_epochs": 100, "batch_size": 16}
```

**LLM example:**
```json
{"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 16, "num_epochs": 1, "batch_size": 4}
```

Keys not present in the strategy's scope are passed through unchanged.

---

## 2. Task Context

#### `task_type`
| | |
|---|---|
| **Type** | string |
| **Default** | `"detection"` |
| **Values** | `"detection"`, `"classification"`, `"llm_finetune"` |
| **Source** | Workflow parameter (immutable across iterations) |

Determines:
1. **Schedule strategy rules**: Which HPs to decay, which to unfreeze, task-specific defaults
2. **Default search spaces**: Grid values and random distributions when `search_space="auto"`
3. **Overfitting corrections**: Task-specific regularization adjustments

| task_type | Schedule adjusts | Default grid HPs | Default random HPs |
|-----------|-----------------|-------------------|---------------------|
| `detection` | `per_item_lrate` (decay), `frozen_stages` (unfreeze) | `per_item_lrate`, `frozen_stages` | `per_item_lrate`, `frozen_stages` |
| `classification` | `per_item_lrate` (decay), `weight_decay` (increase) | `per_item_lrate`, `weight_decay` | `per_item_lrate`, `weight_decay` |
| `llm_finetune` | `learning_rate` (decay), `lora_r` (double) | `learning_rate`, `lora_r` | `learning_rate`, `lora_r`, `lora_alpha` |

---

## 3. Strategy Selection

#### `tuning_strategy`
| | |
|---|---|
| **Type** | string |
| **Default** | `"schedule"` |
| **Values** | `"schedule"`, `"grid"`, `"random"` |
| **Source** | Workflow parameter |

Which HP adjustment algorithm to use:

**`schedule` (recommended default)**
Deterministic, rule-based decay. Applies the same predictable adjustments per iteration regardless of history content. Zero randomness, zero overhead. Best for production pipelines where reliability matters more than exploration.

- LR is multiplied by `lr_decay_factor` each iteration
- Vision detection: additionally unfreezes one backbone stage (if `unfreeze_on_retry=true`)
- Classification: additionally increases `weight_decay` by 1.5×
- LLM: additionally doubles `lora_r` (capped at 128)

**`grid`**
Exhaustive enumeration of HP combinations from `search_space`. Each iteration tries the next untried combination from the Cartesian product. When all combinations are exhausted, restarts from the beginning.

Best when you have a small, well-defined set of promising configs to try (e.g., 3 LR values × 2 frozen_stages = 6 combos).

**`random`**
Samples HPs from probability distributions defined in `search_space`. Each iteration draws independently, seeded by `seed + iteration` for reproducibility. Better coverage of high-dimensional spaces than grid.

Best when the search space is large or continuous and you want diverse exploration.

#### `search_space`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Default** | `"auto"` |
| **Source** | Workflow parameter |

Defines the tunable HPs and their values/distributions. Format depends on `tuning_strategy`:

**Grid format** — each key maps to a list of values:
```json
{
  "per_item_lrate": [0.001875, 0.0009375, 0.003],
  "frozen_stages": [0, 1]
}
```
Full grid = Cartesian product (6 combos in this example).

**Random format** — each key maps to a sampling distribution:
```json
{
  "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
  "frozen_stages": {"type": "choice", "values": [0, 1]},
  "num_epochs": {"type": "discrete_uniform", "low": 6, "high": 24, "step": 6}
}
```

Supported distribution types:

| Type | Parameters | Use for |
|------|-----------|---------|
| `log_uniform` | `low`, `high` | Learning rates (spans orders of magnitude) |
| `uniform` | `low`, `high` | Continuous params with narrow range |
| `discrete_uniform` | `low`, `high`, `step` | Integer params like epochs |
| `choice` | `values` (list) | Categorical or small discrete sets |
| `int_log_uniform` | `low`, `high` | Integer params spanning orders of magnitude (e.g., lora_r) |

**`"auto"` resolution** — when left as `"auto"`, defaults are generated per task_type:

Detection defaults:
| Strategy | Search space |
|----------|-------------|
| grid | `per_item_lrate: [0.001875, 0.0009375, 0.003]`, `frozen_stages: [0, 1]` |
| random | `per_item_lrate: log_uniform(1e-5, 1e-2)`, `frozen_stages: choice([0, 1])` |

Classification defaults:
| Strategy | Search space |
|----------|-------------|
| grid | `per_item_lrate: [1.95e-5, 1e-5, 5e-5]`, `weight_decay: [0.01, 0.05]` |
| random | `per_item_lrate: log_uniform(1e-6, 1e-3)`, `weight_decay: log_uniform(0.001, 0.1)` |

LLM defaults:
| Strategy | Search space |
|----------|-------------|
| grid | `learning_rate: [2e-4, 1e-4, 5e-5]`, `lora_r: [16, 32, 64]` |
| random | `learning_rate: log_uniform(1e-5, 5e-4)`, `lora_r: choice([8, 16, 32, 64, 128])`, `lora_alpha: choice([8, 16, 32, 64, 128])` |

**Note**: The `schedule` strategy ignores `search_space` entirely — it uses `lr_decay_factor` and `unfreeze_on_retry` instead.

---

## 4. Schedule Strategy Knobs

These parameters only affect the `schedule` strategy. They are ignored by `grid` and `random`.

#### `lr_decay_factor`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.5` |
| **Constraints** | 0.1–1.0 |
| **Source** | Workflow parameter |

Multiplier applied to the learning rate on each iteration. Applied to:
- `per_item_lrate` for detection and classification
- `learning_rate` for LLM fine-tuning

**Example progression** (factor=0.5, detection):
| Iteration | per_item_lrate | Effective LR (batch=16) |
|-----------|---------------|------------------------|
| 1 | 0.001875 | 0.03 |
| 2 | 0.0009375 | 0.015 |
| 3 | 0.00046875 | 0.0075 |

**Tuning guidance**:
- **0.5** (default): Standard halving — moderate exploration, 3 iterations covers 8× LR range
- **0.3**: Aggressive — faster convergence to low LR, useful when initial LR is likely too high
- **0.7**: Conservative — gentle decay, useful when you want more iterations at moderate LR
- **1.0**: No LR decay (effectively disables LR adjustment in schedule mode)

#### `unfreeze_on_retry`
| | |
|---|---|
| **Type** | bool |
| **Default** | `true` |
| **Source** | Workflow parameter |

Only applies to **detection** tasks with the `schedule` strategy. When `true`, `frozen_stages` is decremented by 1 each iteration (floored at 0), progressively unfreezing ResNet backbone stages.

**Example progression** (unfreeze=true):
| Iteration | frozen_stages | Backbone behavior |
|-----------|--------------|-------------------|
| 1 | 1 | Stage 1 frozen, stages 2-4 trainable |
| 2 | 0 | Fully trainable |
| 3 | 0 | Already fully trainable (no change) |

**When to disable**: Set to `false` when your dataset is similar to COCO and backbone adaptation is unnecessary. Also disable when training from scratch (`pretrained_weights="None"`) since there are no pretrained features to preserve.

---

## 5. Overfitting Signal

#### `is_overfitting`
| | |
|---|---|
| **Type** | bool |
| **Default** | `false` |
| **Source** | Metric decision step output `is_overfitting` |

When `true`, the step applies regularization corrections **on top of** the strategy's output. These overrides are applied regardless of which strategy was used.

**Corrections by task type:**

| task_type | Correction | Rationale |
|-----------|-----------|-----------|
| `detection` | `num_epochs = max(num_epochs // 2, 4)` | Reduce training time to prevent further memorization |
| `classification` | `weight_decay × 3` (capped at 0.1), `num_epochs = max(num_epochs // 2, 20)` | Stronger regularization + reduced exposure |
| `llm_finetune` | `lora_r = max(lora_r // 2, 4)`, `lora_alpha = lora_r`, `weight_decay = max(weight_decay, 0.01)` | Reduce adapter capacity + add weight penalty |

**Order of operations**: Strategy output is computed first, then overfitting corrections override specific values. For example, if the schedule strategy doubles `lora_r` from 16 to 32, and overfitting is detected, the correction halves it back to 16.

---

## 6. Reproducibility

#### `seed`
| | |
|---|---|
| **Type** | int |
| **Default** | `42` |
| **Constraints** | 0+ |
| **Source** | Workflow parameter |

Random seed for deterministic HP selection. Used differently by each strategy:

| Strategy | Seed usage |
|----------|-----------|
| `schedule` | Not used (fully deterministic by design) |
| `grid` | Not used (deterministic enumeration order) |
| `random` | `random.Random(seed + iteration)` — each iteration gets a unique but reproducible RNG |

Setting the same seed guarantees identical HP selections across pipeline reruns with the same history.

---

## 7. Outputs

### 7.1 Primary Output

#### `hyperparams_json`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Argo path** | `/tmp/hyperparams_json` |

The new hyperparameter set for the next training iteration. Built by copying `current_hyperparams`, applying strategy-specific changes, then applying overfitting corrections if applicable.

**Detection example:**
```json
{"per_item_lrate": 0.0009375, "frozen_stages": 0, "num_epochs": 100, "batch_size": 16}
```

**LLM example:**
```json
{"learning_rate": 1e-4, "lora_r": 32, "lora_alpha": 32, "num_epochs": 1, "batch_size": 4}
```

Keys from `current_hyperparams` not modified by the strategy are passed through unchanged. The train step parses this and applies as overrides to its default parameters.

### 7.2 Debug Output

#### `strategy_metadata`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Argo path** | `/tmp/strategy_metadata` |
| **Default** | `"{}"` |

Diagnostic metadata about what the step did. Useful for debugging HP selection and tracking which changes were made.

**Example:**
```json
{
  "strategy": "schedule",
  "task_type": "detection",
  "iteration": 2,
  "changes": {
    "per_item_lrate": {"from": 0.001875, "to": 0.0009375, "rule": "decay_by_0.5"},
    "frozen_stages": {"from": 1, "to": 0, "rule": "unfreeze_one_stage"}
  },
  "overfit_adjustments": false,
  "search_space_used": null,
  "seed": null
}
```

Fields:
- `strategy`: Which strategy was applied
- `task_type`: Task type used for defaults
- `iteration`: Current iteration number (derived from history length)
- `changes`: Dict of parameters that were modified, with before/after values
- `overfit_adjustments`: Whether overfitting corrections were applied on top
- `search_space_used`: Resolved search space (for grid/random; null for schedule)
- `seed`: Seed used (for grid/random; null for schedule)

#### `hp_output.json`
| | |
|---|---|
| **Type** | JSON file |
| **Path** | `/tmp/hp_output.json` |

Full output record for debugging. Contains `input_hyperparams`, `output_hyperparams`, and `metadata`. Not consumed by any downstream step — purely for human inspection of Argo pod logs.
