# Parameter Reference: Metric Decision Step (`metric-decision-ps`)

This document describes every parameter accepted and produced by the metric decision step. The step is a lightweight, GPU-free pipeline component that compares evaluation metrics against configurable thresholds and outputs a routing decision (`deploy` / `retrain` / `stop`).

See also: [metric-decision-design.md](metric-decision-design.md) | [metric-decision-parameters.csv](metric-decision-parameters.csv)

---

## 1. Evaluation Input

#### `eval_results_json`
| | |
|---|---|
| **Type** | string (file path) |
| **Required** | Yes |
| **Source** | Eval step Argo output parameter |

Path to the JSON file produced by the evaluation step. Expected schema:

```json
{
  "metrics": {
    "AP": 0.3200,
    "AP50": 0.5500,
    "AP75": 0.2800,
    "accuracy/top1": 0.87,
    "eval_loss": 1.42,
    "train_loss": 0.55
  }
}
```

The step reads `metrics[primary_metric]` for the threshold comparison. The `train_loss` and `eval_loss` fields are only required when `overfitting_detection=true`.

---

## 2. Task Context

#### `task_type`
| | |
|---|---|
| **Type** | string |
| **Default** | `"detection"` |
| **Values** | `"detection"`, `"classification"`, `"llm_finetune"` |
| **Source** | Workflow parameter (immutable across iterations) |

Determines the default primary metric and direction when they are set to `"auto"`:

| task_type | Default metric | Default direction |
|-----------|---------------|-------------------|
| `detection` | `AP` | `maximize` |
| `classification` | `accuracy/top1` | `maximize` |
| `llm_finetune` | `eval_loss` | `minimize` |

#### `primary_metric`
| | |
|---|---|
| **Type** | string |
| **Default** | `"auto"` |
| **Source** | Workflow parameter |

Which metric from the eval results to compare against the threshold. When `"auto"`, resolved from `task_type` using the table above.

Can be overridden to any key present in the eval results `metrics` dict. Examples:
- `"AP50"` — use AP at IoU=0.50 instead of AP@0.50:0.95
- `"accuracy/top5"` — use top-5 accuracy instead of top-1
- `"perplexity"` — any custom metric the eval step produces

**Resolution logic**: If the metric name contains any of `["loss", "perplexity", "error", "cer", "wer"]`, direction auto-resolves to `"minimize"`. Otherwise it inherits from the `task_type` default.

#### `metric_direction`
| | |
|---|---|
| **Type** | string |
| **Default** | `"auto"` |
| **Values** | `"maximize"`, `"minimize"`, `"auto"` |
| **Source** | Workflow parameter |

Whether a higher or lower metric value is better:
- **`maximize`**: Model passes when `metric_value >= metric_threshold` (AP, accuracy, F1)
- **`minimize`**: Model passes when `metric_value <= metric_threshold` (loss, perplexity, error rate)
- **`auto`**: Resolved from `primary_metric` name or `task_type` default

#### `metric_threshold`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.50` |
| **Constraints** | 0.0–inf |
| **Source** | Workflow parameter |

The quality gate value. The model is accepted (`decision="deploy"`) when the primary metric meets this threshold.

Task-specific guidance:
- **Detection**: 0.50 AP is a reasonable production baseline for custom detectors
- **Classification**: 0.85 top-1 accuracy for most practical use cases
- **LLM Fine-tuning**: 1.5 eval_loss (lower is better — set based on baseline model's loss on your eval set)

**Boundary behavior**: Exact equality passes. If `metric_direction="maximize"` and `metric_value == metric_threshold`, the decision is `"deploy"`.

---

## 3. Loop State

#### `current_iteration`
| | |
|---|---|
| **Type** | int |
| **Default** | `1` |
| **Constraints** | 1+ |
| **Source** | Workflow parameter (iteration 1) or prior decide step output `next_iteration` |

The current loop iteration counter, 1-indexed. Used to determine if max iterations have been reached.

#### `max_retrain_iterations`
| | |
|---|---|
| **Type** | int |
| **Default** | `3` |
| **Constraints** | 1–10 |
| **Source** | Workflow parameter |

Maximum number of training iterations (including the initial run). When `current_iteration >= max_retrain_iterations` and the metric still doesn't meet the threshold, the step outputs `decision="stop"`.

Total training runs = up to `max_retrain_iterations`. Cost ceiling: `max_retrain_iterations × (train_cost + eval_cost)`.

#### `hp_history`
| | |
|---|---|
| **Type** | string (JSON array) |
| **Default** | `"[]"` |
| **Source** | Prior decide step output (accumulated) |

JSON array of all previous iteration records. This step **appends** the current iteration's record before writing the updated array to its output.

**Schema per entry:**
```json
{
  "iteration": 1,
  "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
  "metrics": {"AP": 0.3200, "AP50": 0.5500},
  "decision": "retrain",
  "reason": "AP 0.3200 below threshold 0.50"
}
```

Size: ~200–500 bytes per entry. At 10 iterations max, total is < 5KB — well within Argo's 256KB output parameter limit.

**Used for**: Early stopping (comparing current metric against previous iteration's metric to detect plateaus).

#### `current_hyperparams`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Default** | `"{}"` |
| **Source** | Workflow parameter (initial) or prior hp-adjust step output `hyperparams_json` |

JSON object of the current iteration's hyperparameters. This step does **not** modify these — it stores them in the history record for tracking and debugging. The actual HP adjustment happens in the downstream HP Adjustment Step.

Example:
```json
{"per_item_lrate": 0.0009375, "frozen_stages": 0}
```

---

## 4. Stopping Controls

#### `early_stop_min_delta`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.0` |
| **Constraints** | 0.0–1.0 |
| **Source** | Workflow parameter |

Minimum improvement in the primary metric between the current and previous iteration. If improvement falls below this delta, the step outputs `decision="stop"` with reason `"Plateau"` instead of continuing to retrain.

- **0.0 (default)**: Early stopping disabled — retrain continues until threshold is met or max iterations reached
- **0.01**: Stop if AP/accuracy improves by less than 1% per iteration
- **0.05**: More aggressive — stop on < 5% improvement

**Improvement calculation** (direction-aware):
- `maximize`: improvement = current_value − previous_value
- `minimize`: improvement = previous_value − current_value (decrease counts as positive improvement)

**Prerequisite**: Requires at least one previous iteration in `hp_history`. On iteration 1, early stopping is skipped.

#### `overfitting_detection`
| | |
|---|---|
| **Type** | bool |
| **Default** | `false` |
| **Source** | Workflow parameter |

When enabled, the step checks whether `train_loss < eval_loss × 0.5` (i.e., training loss is less than half of eval loss), indicating severe overfitting. If detected:
- The step sets `is_overfitting="true"` in its output
- The downstream HP Adjustment Step reads this flag and applies regularization corrections (reduce capacity, increase dropout, etc.)
- The **decision is still `"retrain"`** — overfitting does not trigger a stop

**When to enable**: Primarily useful for LLM fine-tuning where train/eval loss divergence is a common failure mode. For vision tasks, MMDetection/MMPretrain run validation within the training step, making external overfitting detection less necessary.

**Requirements**: The eval results JSON must contain both `train_loss` and `eval_loss` in its `metrics` dict. If either is missing, overfitting detection is silently skipped.

---

## 5. Outputs

### 5.1 Routing Decision

#### `decision`
| | |
|---|---|
| **Type** | string |
| **Values** | `"deploy"`, `"retrain"`, `"stop"` |
| **Argo path** | `/tmp/decision` |

The primary output. Drives Argo's `when` conditional branching:

| Value | Meaning | Downstream action |
|-------|---------|-------------------|
| `deploy` | Metric meets threshold | Export & upload model |
| `retrain` | Metric below threshold, iterations remaining | HP Adjustment → Train again |
| `stop` | Metric below threshold, max iterations reached OR plateau detected | Report failure |

**Decision priority** (evaluated in order):
1. Metric ≥ threshold → `deploy`
2. Early stop triggered (plateau) → `stop`
3. `current_iteration >= max_retrain_iterations` → `stop`
4. Otherwise → `retrain`

### 5.2 Metric Information

#### `metric_value`
| | |
|---|---|
| **Type** | string (formatted float, 6 decimal places) |
| **Argo path** | `/tmp/metric_value` |

The primary metric's value from the current evaluation. Examples: `"0.320000"`, `"0.870000"`, `"1.420000"`.

#### `metric_name`
| | |
|---|---|
| **Type** | string |
| **Argo path** | `/tmp/metric_name` |

The resolved primary metric name. Examples: `"AP"`, `"accuracy/top1"`, `"eval_loss"`. Useful for the failure report step to display which metric was being tracked.

### 5.3 Loop State Outputs

#### `hp_history` (output)
| | |
|---|---|
| **Type** | string (JSON array) |
| **Argo path** | `/tmp/hp_history` |

Updated history array with the current iteration appended. Consumed by:
- **HP Adjustment Step**: For history-aware strategy selection (grid: skip tried combos; random: seeded iteration)
- **Next decide step** (via recursive retrain): For early stopping comparison

#### `is_overfitting`
| | |
|---|---|
| **Type** | string (`"true"` / `"false"`) |
| **Argo path** | `/tmp/is_overfitting` |
| **Default** | `"false"` |

Overfitting detection result. When `"true"`, the downstream HP Adjustment Step applies regularization overrides:
- **Detection**: Halve `num_epochs`
- **Classification**: Increase `weight_decay` by 3×, halve `num_epochs`
- **LLM**: Halve `lora_r`, set `weight_decay ≥ 0.01`

#### `current_iteration` (output)
| | |
|---|---|
| **Type** | string (int) |
| **Argo path** | `/tmp/current_iteration` |

Passthrough of the input `current_iteration` value. Used by downstream steps (e.g., failure report).

#### `next_iteration`
| | |
|---|---|
| **Type** | string (int) |
| **Argo path** | `/tmp/next_iteration` |

`current_iteration + 1`. Passed to the recursive retrain DAG task as the next iteration's `current_iteration` input.

### 5.4 Debug Output

#### `decision_output.json`
| | |
|---|---|
| **Type** | JSON file |
| **Path** | `/tmp/decision_output.json` |

Full decision record for debugging. Contains all output fields plus threshold, direction, task_type, and history_length. Not consumed by any downstream step — purely for human inspection of Argo pod logs.
