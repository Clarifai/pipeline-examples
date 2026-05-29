# Parameter Reference: LLM Decision Step (`llm-decision-ps`)

This document describes every parameter accepted and produced by the LLM decision step. The step is a lightweight, GPU-free pipeline component that uses a Clarifai-hosted LLM to analyze the full training history and produce a routing decision (`deploy` / `retrain` / `stop`) along with adjusted hyperparameters when retraining.

It replaces both `metric-decision-ps` and `hp-adjust-ps` in a single step, with automatic fallback to the existing hardcoded logic when the LLM is unavailable or produces invalid output.

See also: [llm-decision-design.md](llm-decision-design.md) | [llm-decision-parameters.csv](llm-decision-parameters.csv)

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

The step reads ALL metrics from this file and includes them in the LLM prompt for comprehensive analysis. The `primary_metric` value is used for the quality gate comparison. The `train_loss` and `eval_loss` fields enable overfitting assessment by the LLM.

**Used by**: LLM prompt (all metrics shown), fallback (only `primary_metric` compared to threshold).

---

## 2. Task Context

#### `task_type`
| | |
|---|---|
| **Type** | string |
| **Default** | `"detection"` |
| **Values** | `"detection"`, `"classification"`, `"llm_finetune"`, or any custom string |
| **Source** | Workflow parameter (immutable across iterations) |

Determines the default primary metric and direction when set to `"auto"`:

| task_type | Default metric | Default direction |
|-----------|---------------|-------------------|
| `detection` | `AP` | `maximize` |
| `classification` | `accuracy/top1` | `maximize` |
| `llm_finetune` | `eval_loss` | `minimize` |

**LLM usage**: Included in the prompt as context for the LLM to understand the optimization target and suggest task-appropriate hyperparameter adjustments.

**Fallback usage**: Dispatches to task-specific strategy functions in the hardcoded fallback path.

**Custom task types**: When using a custom string (e.g., `"spec_decoder_tuning"`), you must explicitly set `primary_metric` and `metric_direction` (not `"auto"`) since no defaults exist for custom types. The LLM handles custom types gracefully via `task_description`.

#### `task_description`
| | |
|---|---|
| **Type** | string |
| **Default** | `""` (empty) |
| **Source** | Workflow parameter |

Free-text description of the training task and goals. Included verbatim in the LLM prompt to provide domain context that `task_type` alone cannot convey.

**Examples:**
- `"Fine-tuning a food image classifier for a mobile app. Latency is critical — prefer smaller models."`
- `"Training a document detector for OCR preprocessing. Must achieve high recall even at the cost of precision."`
- `"LoRA fine-tuning an instruction-following model on customer support conversations. The model should not hallucinate product information."`

**When empty**: The LLM prompt uses a generic description auto-generated from `task_type` (e.g., "Object detection model training with COCO-style evaluation").

**Impact**: Helps the LLM make better trade-off decisions. For example, if the description mentions latency constraints, the LLM might prefer stopping with a good-enough model rather than pursuing marginal accuracy gains with more parameters.

#### `primary_metric`
| | |
|---|---|
| **Type** | string |
| **Default** | `"auto"` |
| **Source** | Workflow parameter |

Which metric from the eval results to use as the quality gate. When `"auto"`, resolved from `task_type` using the defaults table above.

Can be overridden to any key present in the eval results `metrics` dict:
- `"AP50"` — use AP at IoU=0.50 instead of AP@0.50:0.95
- `"accuracy/top5"` — use top-5 accuracy instead of top-1
- `"perplexity"` — any custom metric the eval step produces

**Resolution logic**: If the metric name contains any of `["loss", "perplexity", "error", "cer", "wer"]`, direction auto-resolves to `"minimize"`. Otherwise inherits from `task_type` default.

**LLM usage**: The quality gate value is highlighted in the prompt. The LLM sees ALL metrics but knows which one gates deployment.

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
- **`auto`**: Resolved from `primary_metric` name patterns or `task_type` defaults

**LLM usage**: Stated explicitly in the prompt so the LLM knows which direction means improvement.

#### `metric_threshold`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.50` |
| **Constraints** | 0.0–inf |
| **Source** | Workflow parameter |

The quality gate value. The model is deployed when the primary metric meets this threshold.

**Task-specific guidance:**
- **Detection**: 0.50 AP is a reasonable production baseline for custom detectors
- **Classification**: 0.85 top-1 accuracy for most practical use cases
- **LLM Fine-tuning**: 1.5 eval_loss (lower is better — set based on baseline model's loss)

**Boundary behavior**: Exact equality passes. If `metric_direction="maximize"` and `metric_value == metric_threshold`, the decision is `"deploy"`.

**LLM usage**: Shown in the prompt. The LLM should output `"deploy"` when the metric meets this threshold, though it may override with `"stop"` if the threshold is met but the model shows instability (this is a feature of LLM reasoning vs. rigid rules).

---

## 3. Loop State

#### `current_iteration`
| | |
|---|---|
| **Type** | int |
| **Default** | `1` |
| **Constraints** | 1+ |
| **Source** | Workflow parameter (iteration 1) or prior llm-decide output `next_iteration` |

The current loop iteration counter, 1-indexed. Included in the LLM prompt to convey budget pressure.

**LLM usage**: The prompt shows "Iteration X of Y, Z remaining" — the LLM can balance exploration (try bold changes early) vs. exploitation (refine what works when budget is low).

**Fallback usage**: Compared against `max_retrain_iterations` for stop condition.

#### `max_retrain_iterations`
| | |
|---|---|
| **Type** | int |
| **Default** | `3` |
| **Constraints** | 0–10 |
| **Source** | Workflow parameter |

Maximum number of training iterations allowed. When `current_iteration >= max_retrain_iterations` and the metric doesn't meet the threshold, the decision is `"stop"`.

**Set to `0`**: Deploy-or-stop quality gate only — no retraining allowed.

**Cost ceiling**: `max_retrain_iterations × (train_cost + eval_cost + ~$0.001 LLM)`.

**LLM usage**: Shown in the prompt as the total budget. The LLM understands that on the last iteration, suggesting "retrain" is pointless and should choose "deploy" (if close enough) or "stop".

#### `hp_history`
| | |
|---|---|
| **Type** | string (JSON array) |
| **Default** | `"[]"` |
| **Source** | Prior llm-decide output (accumulated) |

JSON array of all previous iteration records. This is the **primary context** the LLM uses to reason about training progress. The step appends the current iteration before outputting the updated array.

**Schema per entry:**
```json
{
  "iteration": 1,
  "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
  "metrics": {"AP": 0.3200, "AP50": 0.5500, "AP75": 0.2800},
  "decision": "retrain",
  "reason": "AP 0.3200 below threshold 0.50",
  "reasoning": "AP improved 31% from baseline. Trend suggests further gains with LR reduction.",
  "method": "llm"
}
```

**LLM usage**: The full history is formatted as a table/list in the prompt. The LLM can identify:
- Improvement trends (monotonic improvement → keep going)
- Oscillations (AP going up/down → reduce LR, increase regularization)
- Plateau (diminishing returns → stop or try a radically different config)
- Overfitting pattern (train metrics improve but eval metrics degrade)

**Size**: ~400–800 bytes per entry (with reasoning). At 10 max iterations = 4–8 KB. Well within Argo's 256KB limit.

#### `current_hyperparams`
| | |
|---|---|
| **Type** | string (JSON object) |
| **Default** | `"{}"` |
| **Source** | Workflow parameter (initial) or prior llm-decide output `hyperparams_json` |

The hyperparameters used for the current iteration's training. Shown to the LLM as the baseline it should modify when suggesting retraining.

**Detection example:**
```json
{"per_item_lrate": 0.001875, "frozen_stages": 1, "num_epochs": 100, "batch_size": 16}
```

**LLM example:**
```json
{"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 16, "num_epochs": 1, "batch_size": 4}
```

**LLM usage**: Shown in the prompt alongside the search space. The LLM adjusts these values based on the observed metrics.

---

## 4. HP Adjustment Context

#### `search_space`
| | |
|---|---|
| **Type** | string (JSON or `"auto"`) |
| **Default** | `"auto"` |
| **Source** | Workflow parameter |

Defines the valid ranges/choices for hyperparameters. The LLM must produce `next_hyperparams` values within these bounds. Values outside bounds are clamped (not rejected).

**When `"auto"`**: Resolved to task-specific defaults:

| task_type | Default search space |
|-----------|---------------------|
| `detection` | `{"per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}, "frozen_stages": {"type": "choice", "values": [0, 1]}}` |
| `classification` | `{"per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3}, "weight_decay": {"type": "log_uniform", "low": 0.001, "high": 0.1}}` |
| `llm_finetune` | `{"learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4}, "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]}, "lora_alpha": {"type": "choice", "values": [8, 16, 32, 64, 128]}}` |

**Custom search space format:**
```json
{
  "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
  "frozen_stages": {"type": "choice", "values": [0, 1]},
  "num_epochs": {"type": "discrete_uniform", "low": 6, "high": 24, "step": 6},
  "batch_size": {"type": "choice", "values": [4, 8, 16, 32]}
}
```

**Supported distribution types** (for display and clamping):
- `log_uniform` — continuous range, logarithmic scale (for learning rates)
- `uniform` — continuous range, linear scale
- `discrete_uniform` — integer steps within a range
- `choice` — categorical selection from a list
- `int_log_uniform` — integer log-uniform

**LLM usage**: Formatted as a human-readable table in the prompt showing parameter names, types, and valid ranges. The LLM references these when generating HP suggestions.

**Fallback usage**: Used by `grid_select()` to enumerate combinations or by `random_select()` to sample.

---

## 5. Fallback-Only Parameters

These parameters configure the hardcoded fallback logic. They are ignored when the LLM path succeeds. They exist for backward compatibility and to ensure the fallback behaves identically to the existing `metric-decision-ps` + `hp-adjust-ps` steps.

#### `tuning_strategy`
| | |
|---|---|
| **Type** | string |
| **Default** | `"schedule"` |
| **Values** | `"schedule"`, `"grid"`, `"random"` |
| **Source** | Workflow parameter |

Which HP adjustment strategy the fallback uses. The LLM ignores this — it chooses its own approach based on context.

| Strategy | Fallback behavior |
|----------|------------------|
| `schedule` | Deterministic decay: LR × `lr_decay_factor`, unfreeze stages, etc. |
| `grid` | Enumerate search space combinations, skip already-tried configs |
| `random` | Sample from search space distributions using seeded RNG |

#### `lr_decay_factor`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.5` |
| **Constraints** | 0.1–1.0 |
| **Source** | Workflow parameter |

Fallback schedule strategy: multiply the learning rate by this factor on each retrain iteration.

Example with default 0.5: `0.001875 → 0.0009375 → 0.00046875 → ...`

**Not used by LLM** — the LLM decides its own LR adjustments based on training history.

#### `unfreeze_on_retry`
| | |
|---|---|
| **Type** | bool |
| **Default** | `true` |
| **Source** | Workflow parameter |

Fallback schedule strategy (detection only): decrement `frozen_stages` by 1 each iteration, allowing more backbone layers to be fine-tuned.

**Not used by LLM** — the LLM can independently decide to adjust `frozen_stages` if it appears in the search space.

#### `early_stop_min_delta`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.0` |
| **Constraints** | 0.0–1.0 |
| **Source** | Workflow parameter |

Fallback early stopping: minimum metric improvement between consecutive iterations to avoid plateau detection.

- `0.0` — early stopping disabled (fallback relies on max iterations only)
- `0.01` — stop if improvement is < 1% between iterations

**Not used by LLM** — the LLM performs its own trend analysis on the full history, identifying plateaus, oscillations, and diminishing returns without a fixed delta threshold.

#### `overfitting_detection`
| | |
|---|---|
| **Type** | bool |
| **Default** | `false` |
| **Source** | Workflow parameter |

Fallback overfitting detection: check if `train_loss < eval_loss × 0.5` (heavy overfitting signal).

When the fallback detects overfitting, it passes `is_overfitting=true` to the HP adjustment logic, which then applies task-specific regularization corrections.

**Not used by LLM** — the LLM assesses overfitting holistically from the full metric history (train vs. eval loss trajectory across iterations, not just a single-point ratio).

---

## 6. LLM Configuration

#### `llm_model_url`
| | |
|---|---|
| **Type** | string |
| **Default** | `""` (empty) |
| **Source** | Workflow parameter |

Clarifai model URL for LLM inference. Must point to a text-generation model hosted on the Clarifai platform.

**Examples:**
- `"https://clarifai.com/openai/chat-completion/models/gpt-4o-mini"` — Recommended default (fast, cheap, good JSON compliance)
- `"https://clarifai.com/openai/chat-completion/models/gpt-4o"` — Higher quality reasoning
- `"https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet"` — Alternative provider

**Empty string behavior**: When empty, the step skips the LLM entirely and uses the fallback path immediately. This enables running in air-gapped environments or when no LLM budget is available. Functionally equivalent to running `metric-decision-ps` + `hp-adjust-ps` sequentially.

**Authentication**: Uses the `CLARIFAI_PAT` environment variable (Personal Access Token) for API authentication. This is typically injected as a Kubernetes secret in the pipeline runtime environment.

#### `llm_temperature`
| | |
|---|---|
| **Type** | float |
| **Default** | `0.1` |
| **Constraints** | 0.0–2.0 |
| **Source** | Workflow parameter |

Sampling temperature for the LLM call. Controls the randomness of the response.

| Range | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Fully deterministic (greedy) | Maximum consistency across runs |
| 0.1 (default) | Nearly deterministic with minimal variation | Production (recommended) |
| 0.3–0.5 | Moderate variation | When you want the LLM to explore novel HP combinations |
| 1.0+ | High creativity | Not recommended for production decisions |

**Impact on decisions**: At temperature=0.1, the same history/metrics should produce the same decision ~95% of the time. At 0.5, there may be 10-20% variation in HP suggestions (but decision deploy/retrain/stop remains stable).

#### `llm_max_retries`
| | |
|---|---|
| **Type** | int |
| **Default** | `3` |
| **Constraints** | 1–10 |
| **Source** | Workflow parameter |

Maximum number of LLM call attempts before falling back to hardcoded logic.

**Retry triggers:**
- LLM returns non-JSON response (e.g., markdown-wrapped)
- Response missing required fields (`decision`, `reasoning`, `confidence`)
- Invalid decision value (not in deploy/retrain/stop)
- `decision="retrain"` but `next_hyperparams` missing or empty
- Network timeout (30s default)
- HTTP 429 (rate limited) / 500 / 502 / 503

**Non-retryable errors** (immediate fallback):
- HTTP 401/403 (authentication failure)
- HTTP 404 (model not found)

**Latency impact**: Worst case (all retries fail) adds ~30–90s before fallback. In practice, retries succeed on attempt 1-2 for well-behaved models.

---

## 7. Reproducibility

#### `seed`
| | |
|---|---|
| **Type** | int |
| **Default** | `42` |
| **Constraints** | 0+ |
| **Source** | Workflow parameter |

Random seed for the fallback path's grid and random strategies. Uses `(seed + iteration)` to produce different but deterministic results per iteration.

**Not used by LLM path** — LLM decisions are influenced by temperature, not a seed. For maximum reproducibility of decisions, set `llm_temperature=0.0`.

---

## 8. Outputs

#### `decision`
| | |
|---|---|
| **Type** | string |
| **Values** | `"deploy"`, `"retrain"`, `"stop"` |
| **File** | `/tmp/decision` |

The routing decision used by Argo `when` conditions to branch the DAG:
- **`deploy`** — model meets quality gate → proceed to export/upload
- **`retrain`** — model can improve → use `hyperparams_json` for next training iteration
- **`stop`** — further training is unlikely to help → report final metrics and end

**Produced by**: LLM decision (validated), or fallback logic on LLM failure.

#### `hyperparams_json`
| | |
|---|---|
| **Type** | string (JSON object) |
| **File** | `/tmp/hyperparams_json` |

The hyperparameter configuration for the next training iteration. Only meaningful when `decision="retrain"`.

**LLM path**: Generated by the LLM based on history analysis. Values are validated against `search_space` and clamped to bounds. Unknown keys from the LLM are passed through (allowing the LLM to suggest parameters not in the explicit search space, if they exist in the training step's interface).

**Fallback path**: Generated by the configured `tuning_strategy` (schedule/grid/random) with task-specific rules.

**Example (detection, retrain):**
```json
{"per_item_lrate": 0.0009375, "frozen_stages": 0, "num_epochs": 12, "batch_size": 8}
```

**Example (deploy or stop):**
```json
{}
```

#### `reasoning`
| | |
|---|---|
| **Type** | string |
| **File** | `/tmp/reasoning` |

Human-readable explanation of the decision. Useful for debugging, audit trails, and understanding why the LLM made a particular choice.

**LLM path example:**
```
AP improved from 0.32 to 0.42 (+31%) after unfreezing the backbone. The learning rate 
was halved which helped stabilize training. Given 1 iteration remaining, I recommend 
one more attempt with a further LR reduction to 4.7e-4 — the improvement trajectory 
suggests we can reach the 0.50 threshold.
```

**Fallback example:**
```
[FALLBACK] AP 0.4200 below threshold 0.50 (LLM unavailable: 3 retries exhausted — 
invalid JSON responses)
```

#### `metric_value`
| | |
|---|---|
| **Type** | string (float, 6 decimal places) |
| **File** | `/tmp/metric_value` |

The current iteration's primary metric value, formatted to 6 decimal places. Read directly from `eval_results_json`.

Example: `"0.420000"`

If the metric key is missing from eval results: `"N/A"`.

#### `metric_name`
| | |
|---|---|
| **Type** | string |
| **File** | `/tmp/metric_name` |

The resolved primary metric name (after `"auto"` resolution).

Example: `"AP"`, `"accuracy/top1"`, `"eval_loss"`

#### `hp_history` (output)
| | |
|---|---|
| **Type** | string (JSON array) |
| **File** | `/tmp/hp_history` |

The updated history array with the current iteration appended. Includes the new `reasoning` and `method` fields:

```json
[
  {
    "iteration": 1,
    "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
    "metrics": {"AP": 0.3200, "AP50": 0.5500},
    "decision": "retrain",
    "reason": "AP 0.3200 below threshold 0.50",
    "reasoning": "Initial run with default HPs. AP is below threshold but shows the model is learning. Recommending LR reduction and backbone unfreezing.",
    "method": "llm"
  }
]
```

This output feeds into the next iteration's `hp_history` input parameter via Argo parameter passing.

#### `is_overfitting`
| | |
|---|---|
| **Type** | string (`"true"` / `"false"`) |
| **File** | `/tmp/is_overfitting` |

Whether overfitting was detected for the current iteration.

**LLM path**: The LLM explicitly outputs `is_overfitting` as part of its JSON response, based on its analysis of train vs. eval metric trajectories.

**Fallback path**: Determined by the `train_loss < eval_loss × 0.5` heuristic (only when `overfitting_detection=true`).

#### `current_iteration` (output)
| | |
|---|---|
| **Type** | string (int) |
| **File** | `/tmp/current_iteration` |

Passthrough of the input `current_iteration` value. Used by downstream steps for reference.

#### `next_iteration`
| | |
|---|---|
| **Type** | string (int) |
| **File** | `/tmp/next_iteration` |

`current_iteration + 1`. Fed to the recursive autoloop template as the next iteration's counter.

#### `strategy_metadata`
| | |
|---|---|
| **Type** | string (JSON object) |
| **File** | `/tmp/strategy_metadata` |

Debug metadata documenting how the decision was made:

**LLM success:**
```json
{
  "method": "llm",
  "model_url": "https://clarifai.com/openai/chat-completion/models/gpt-4o-mini",
  "temperature": 0.1,
  "attempts": 1,
  "prompt_tokens": 2480,
  "completion_tokens": 185,
  "latency_ms": 4200,
  "confidence": 0.85
}
```

**Fallback:**
```json
{
  "method": "fallback",
  "fallback_reason": "LLM retries exhausted: [invalid decision 'continue']",
  "attempts": 3,
  "fallback_strategy": "schedule",
  "fallback_task_type": "detection",
  "changes": {"per_item_lrate": {"from": 0.001875, "to": 0.0009375}},
  "overfit_adjustments": false
}
```
