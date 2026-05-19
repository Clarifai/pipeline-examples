# Design Document: LLM-Based Decision Step

**Step ID**: `llm-decision-ps`
**Status**: Draft
**Date**: 2026-04-30

---

## 1. Overview

The LLM Decision Step replaces the separate `metric-decision-ps` and `hp-adjust-ps` pipeline steps with a single LLM-powered reasoning step. It receives the full training history, current evaluation metrics, and available hyperparameter ranges, then uses a Clarifai-hosted LLM to produce a structured decision (deploy/retrain/stop) along with the next hyperparameter configuration when retraining is needed.

The hardcoded logic from the two existing steps is retained as an automatic fallback in case the LLM call fails.

```
┌──────────┐     ┌─────────────────────────────────────────────────────────────────┐
│ Eval Step│────▶│  LLM Decision Step                                               │
│ (GPU)    │     │  (CPU — no GPU, network access for LLM API)                     │
│          │     │                                                                  │
│ outputs: │     │  1. Load eval metrics + full history                             │
│ eval_    │     │  2. Build prompt with task context, history, HP ranges            │
│ results  │     │  3. Call Clarifai-hosted LLM (GPT-4o-mini / Claude / etc.)       │
│ .json    │     │  4. Parse structured JSON response                               │
│          │     │  5. Validate decision + hyperparameters                           │──▶ decision="deploy" ──▶ [Export]
│          │     │  6. On failure: retry up to N times                               │
└──────────┘     │  7. On all retries exhausted: fall back to hardcoded logic        │──▶ decision="retrain" ──▶ [Train]
                 │                                                                  │
  ┌────────┐     │  outputs:                                                        │──▶ decision="stop" ──▶ [Report]
  │Workflow │────▶│  • decision (deploy | retrain | stop)                            │
  │ Params  │     │  • hyperparams_json (next HPs, if retrain)                      │
  │         │     │  • reasoning (LLM explanation)                                  │
  │llm_model│     │  • hp_history (updated with reasoning)                          │
  │_url     │     │  • strategy_metadata (method=llm|fallback)                      │
  │task_desc│     │  • metric_value, metric_name, is_overfitting, next_iteration    │
  └────────┘     └─────────────────────────────────────────────────────────────────┘
```

### 1.1 Why Replace Two Steps With One?

| Aspect | Before (metric-decision + hp-adjust) | After (llm-decision) |
|--------|--------------------------------------|---------------------|
| History analysis | Only compares last 2 data points (early stopping) | Considers full history — identifies trends, oscillations, diminishing returns |
| Strategy | Hardcoded per-task rules (schedule/grid/random) | LLM adapts strategy per context; no per-task code required |
| New task types | Requires new code in `strategies.py` dispatch table | Works for any task type described in `task_description` |
| Overfitting | Binary flag (train_loss < 0.5 × eval_loss) | Nuanced assessment of metric trajectories |
| Explainability | Opaque reason strings ("AP 0.32 below threshold 0.50") | Structured reasoning explaining tradeoffs and strategy choice |
| DAG complexity | 2 pipeline steps, intermediate `is_overfitting` hand-off | 1 pipeline step, simpler Argo DAG |
| Latency | < 2 sec (both steps combined) | 3-10 sec (LLM API call), < 2 sec (fallback) |
| Reliability | Always deterministic | LLM output validated + retry + deterministic fallback |

### 1.2 Separation of Concerns

| Responsibility | This Step (llm-decision-ps) | Train/Eval Steps |
|---|---|---|
| Read eval metrics | **Yes** | No (eval step produces them) |
| Analyze full training history | **Yes** | No |
| Output routing decision | **Yes** | No |
| Generate next hyperparameters (if retrain) | **Yes** | No |
| Explain reasoning | **Yes** | No |
| Maintain loop history | **Yes** | No |
| Actually train the model | No | **Yes** |
| Actually evaluate the model | No | **Yes** |

### 1.3 Design Principles

1. **LLM-first, fallback-safe** — Use LLM reasoning for better decisions, but never let an LLM failure stop the pipeline. The hardcoded logic from `metric-decision-ps` + `hp-adjust-ps` is always available as a deterministic fallback.
2. **Structured output** — The LLM is prompted to return a strict JSON schema. The step validates the response against an expected schema before accepting it.
3. **Full context** — The prompt includes ALL iterations (not just the last), allowing the LLM to identify trends, oscillations, and plateaus that rule-based logic misses.
4. **Task-agnostic** — The LLM doesn't need task-specific code. A free-text `task_description` parameter provides domain context. This supports new task types (spec decoder tuning, serving config optimization) without code changes.
5. **Observability** — Every decision includes a `reasoning` field explaining why the LLM chose that action. The `strategy_metadata` tracks whether the LLM or fallback produced the decision.
6. **Budget-aware** — The prompt includes the iteration budget remaining, so the LLM can weigh exploration vs. deployment pressure.

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

**Execution time**:
- LLM path: 3–10 seconds (network round-trip to Clarifai model API)
- Fallback path: < 2 seconds (same as existing hardcoded logic)

**Network**: Requires outbound HTTPS access to Clarifai API (`api.clarifai.com`).

---

## 3. Interface

### 3.1 Method Signature

```python
class LLMDecision:
    def decide(
        self,
        # ── Eval Results ──
        eval_results_json: str,              # Path to eval_results.json from eval step

        # ── Task Context ──
        task_type: str = "detection",        # "detection" | "classification" | "llm_finetune" | any custom
        task_description: str = "",          # Free-text description of the task & goals
        primary_metric: str = "auto",        # Metric name or "auto" (resolved from task_type)
        metric_direction: str = "auto",      # "maximize" | "minimize" | "auto"
        metric_threshold: float = 0.50,      # Quality gate value

        # ── Loop State ──
        current_iteration: int = 1,          # Current iteration (1-indexed)
        max_retrain_iterations: int = 3,     # Max retries before stop
        hp_history: str = "[]",              # JSON array of prior iterations
        current_hyperparams: str = "{}",     # JSON of current iteration's HPs

        # ── HP Adjustment Context ──
        search_space: str = "auto",          # JSON dict or "auto" for task-type defaults
        tuning_strategy: str = "schedule",   # Hint for fallback; LLM ignores this
        lr_decay_factor: float = 0.5,        # Fallback only: schedule LR decay factor
        unfreeze_on_retry: bool = True,      # Fallback only: layer unfreezing toggle

        # ── Stopping Controls ──
        early_stop_min_delta: float = 0.0,   # Fallback only: min improvement for early stop
        overfitting_detection: bool = False,  # Fallback only: train/eval loss divergence check

        # ── LLM Configuration ──
        llm_model_url: str = "",             # Clarifai model URL for LLM inference
        llm_temperature: float = 0.1,        # Sampling temperature (low = deterministic)
        llm_max_retries: int = 3,            # Max retry attempts on LLM failure/invalid output

        # ── Reproducibility ──
        seed: int = 42,                      # Fallback random seed
    ) -> str:
        """Returns path to /tmp/decision_output.json"""
```

### 3.2 Inputs

| Input | Source | Purpose |
|-------|--------|---------|
| `eval_results_json` | Eval step Argo output | Current iteration's metric values |
| `task_type` | Workflow param | Context for LLM prompt + fallback strategy dispatch |
| `task_description` | Workflow param | Free-text context for the LLM (e.g., "Fine-tuning a food classifier for a mobile app with latency constraints") |
| `primary_metric` | Workflow param | Which metric to optimize (or "auto") |
| `metric_direction` | Workflow param | Higher is better or lower is better |
| `metric_threshold` | Workflow param | Quality gate for deployment |
| `current_iteration` | Prior decide output or workflow | Loop counter |
| `max_retrain_iterations` | Workflow param | Budget ceiling |
| `hp_history` | Prior decide output | Complete iteration history for LLM context |
| `current_hyperparams` | Workflow param or prior output | Current HP set to show the LLM |
| `search_space` | Workflow param | Available HP ranges/choices for LLM to sample from |
| `tuning_strategy` | Workflow param | Used by fallback only (LLM decides its own strategy) |
| `lr_decay_factor` | Workflow param | Fallback schedule strategy knob |
| `unfreeze_on_retry` | Workflow param | Fallback schedule strategy knob |
| `early_stop_min_delta` | Workflow param | Fallback early stop threshold |
| `overfitting_detection` | Workflow param | Fallback overfitting detection toggle |
| `llm_model_url` | Workflow param | Clarifai-hosted model to use for reasoning |
| `llm_temperature` | Workflow param | Controls LLM randomness |
| `llm_max_retries` | Workflow param | Retry budget before falling back |
| `seed` | Workflow param | For fallback random/grid strategy determinism |

### 3.3 Outputs

| Output File | Argo Param Name | Type | Values | Description |
|-------------|-----------------|------|--------|-------------|
| `/tmp/decision` | `decision` | string | `"deploy"`, `"retrain"`, `"stop"` | Routing decision for Argo conditional branching |
| `/tmp/hyperparams_json` | `hyperparams_json` | string (JSON) | e.g., `{"per_item_lrate": 0.0009375}` | Next HP config (only meaningful when decision=retrain) |
| `/tmp/reasoning` | `reasoning` | string | Free text | LLM explanation of the decision |
| `/tmp/metric_value` | `metric_value` | string (float) | e.g., `"0.320000"` | Current primary metric value |
| `/tmp/metric_name` | `metric_name` | string | e.g., `"AP"` | Resolved primary metric name |
| `/tmp/hp_history` | `hp_history` | string (JSON array) | `[{...}, ...]` | Updated history with current iteration + reasoning appended |
| `/tmp/is_overfitting` | `is_overfitting` | string (bool) | `"true"` / `"false"` | Whether the LLM (or fallback) detected overfitting |
| `/tmp/current_iteration` | `current_iteration` | string (int) | e.g., `"2"` | Passthrough |
| `/tmp/next_iteration` | `next_iteration` | string (int) | e.g., `"3"` | Incremented iteration counter |
| `/tmp/strategy_metadata` | `strategy_metadata` | string (JSON) | `{"method": "llm", ...}` | Debug metadata: method used, prompt tokens, retries, etc. |
| `/tmp/decision_output.json` | — | JSON file | Full record | Complete decision record for debugging |

**Argo output parameter declarations:**
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
    - name: reasoning
      valueFrom:
        path: /tmp/reasoning
        default: ""
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
    - name: strategy_metadata
      valueFrom:
        path: /tmp/strategy_metadata
        default: "{}"
```

---

## 4. LLM Prompt Design

### 4.1 System Prompt

```
You are an ML training loop optimizer. You analyze training history and metrics
to decide whether a model should be deployed, retrained with adjusted
hyperparameters, or stopped.

You always respond with valid JSON matching the required schema. You never
include markdown formatting, code fences, or explanatory text outside the JSON.
```

### 4.2 User Prompt Structure

The user prompt is constructed dynamically from the step's inputs:

```
## Task
{task_description or auto-generated from task_type}

## Quality Gate
- Primary metric: {metric_name} ({metric_direction})
- Threshold: {metric_threshold}
- Current value: {current_metric_value}

## Iteration Budget
- Current iteration: {current_iteration} of {max_retrain_iterations}
- Iterations remaining: {max_retrain_iterations - current_iteration}

## Current Hyperparameters
{json.dumps(current_hyperparams, indent=2)}

## Available Hyperparameter Ranges
{formatted search_space — showing parameter names, types, and valid ranges/choices}

## Training History
{formatted table of all iterations: iteration, hyperparams, metrics, decision, reason}

## Current Evaluation Results
{json.dumps(eval_results.metrics, indent=2)}

## Instructions
Analyze the training history and decide the next action:
1. "deploy" — if the model meets the quality threshold
2. "retrain" — if improvement is likely with different hyperparameters
3. "stop" — if further training is unlikely to help (plateau, oscillating, budget exhausted)

If you choose "retrain", also provide the next hyperparameter values. Values MUST
be within the ranges defined in "Available Hyperparameter Ranges".

Respond with JSON only:
{
  "decision": "deploy" | "retrain" | "stop",
  "reasoning": "2-4 sentence explanation of your analysis and decision",
  "confidence": 0.0 to 1.0,
  "is_overfitting": true | false,
  "next_hyperparams": { ... }  // Required only when decision="retrain"
}
```

### 4.3 Prompt Design Rationale

| Design Choice | Rationale |
|---|---|
| Full history in prompt | Enables trend analysis (oscillations, diminishing returns, divergence) |
| Search space in prompt | Constrains LLM outputs to valid ranges; prevents hallucinated HPs |
| Explicit JSON schema | Reduces parsing failures; LLMs perform well with clear output schemas |
| Low temperature (0.1) | Prioritizes consistency over creativity; training decisions should be stable |
| Few-shot examples omitted | Keeps prompt short; schema specification is sufficient for capable models |
| Task description as free text | Allows domain context without code changes for new task types |
| Iteration budget shown | Enables LLM to trade exploration vs exploitation pressure |

### 4.4 Token Budget Analysis

| Component | Estimated Tokens |
|---|---|
| System prompt | ~80 |
| Task context + quality gate | ~150 |
| Current hyperparams (typical) | ~100 |
| Search space (typical) | ~200 |
| Full history (10 iterations max) | ~1500 |
| Current eval results | ~200 |
| Instructions + schema | ~250 |
| **Total prompt** | **~2,500** |
| Expected response | ~200 |

Well within context windows of GPT-4o-mini (128K), Claude (200K), etc. Cost per call at GPT-4o-mini rates: ~$0.001.

---

## 5. LLM Response Validation

### 5.1 Schema Validation

The LLM response must be valid JSON conforming to:

```python
RESPONSE_SCHEMA = {
    "decision": str,         # Must be one of: "deploy", "retrain", "stop"
    "reasoning": str,        # Non-empty explanation
    "confidence": float,     # 0.0–1.0
    "is_overfitting": bool,  # Overfitting assessment
    "next_hyperparams": dict # Required when decision="retrain"; ignored otherwise
}
```

### 5.2 Validation Rules

```python
def validate_llm_response(response: dict, search_space: dict, decision: str) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []

    # 1. Decision is valid enum
    if response.get("decision") not in ("deploy", "retrain", "stop"):
        errors.append(f"Invalid decision: {response.get('decision')}")

    # 2. Reasoning is present and non-empty
    if not response.get("reasoning", "").strip():
        errors.append("Missing or empty reasoning")

    # 3. Confidence is in range
    conf = response.get("confidence", -1)
    if not (0.0 <= conf <= 1.0):
        errors.append(f"Confidence out of range: {conf}")

    # 4. next_hyperparams required for retrain
    if response["decision"] == "retrain":
        next_hps = response.get("next_hyperparams")
        if not isinstance(next_hps, dict) or not next_hps:
            errors.append("decision=retrain requires non-empty next_hyperparams")
        else:
            # 5. HP values within search space bounds
            for key, value in next_hps.items():
                if key in search_space:
                    hp_errors = _validate_hp_value(key, value, search_space[key])
                    errors.extend(hp_errors)

    return errors
```

### 5.3 HP Value Clamping

Rather than rejecting HPs that are slightly out of bounds (e.g., LLM returns 0.011 for a range of [0.001, 0.01]), values are **clamped** to the nearest bound:

```python
def clamp_hyperparams(next_hps: dict, search_space: dict) -> dict:
    """Clamp values to search space bounds. Unknown keys pass through."""
    clamped = next_hps.copy()
    for key, value in clamped.items():
        if key not in search_space:
            continue
        spec = search_space[key]
        if isinstance(spec, list):
            # Grid: snap to nearest value
            clamped[key] = min(spec, key=lambda x: abs(x - value))
        elif isinstance(spec, dict):
            if "low" in spec and "high" in spec:
                clamped[key] = max(spec["low"], min(spec["high"], value))
            elif "values" in spec:
                clamped[key] = min(spec["values"], key=lambda x: abs(x - value))
    return clamped
```

### 5.4 Retry Strategy

```
Attempt 1: Call LLM → Validate response
  ✓ Valid → Use LLM decision
  ✗ Invalid → Log validation errors

Attempt 2: Call LLM again (same prompt) → Validate
  ✓ Valid → Use LLM decision
  ✗ Invalid → Log errors

Attempt 3: Call LLM again → Validate
  ✓ Valid → Use LLM decision
  ✗ Invalid → Log "All LLM retries exhausted"

Fallback: Run hardcoded metric-decision + hp-adjust logic
  → Always produces a valid decision
  → strategy_metadata.method = "fallback"
  → strategy_metadata.fallback_reason = "LLM retries exhausted: {errors}"
```

**Retry conditions** (trigger a retry):
- JSON parse error (non-JSON response, markdown-wrapped JSON)
- Missing required fields
- Invalid decision value
- `decision="retrain"` but `next_hyperparams` missing or empty
- Network timeout (configurable, default 30s)
- HTTP 429/500/502/503 from API

**Non-retryable errors** (immediate fallback):
- Authentication failure (401/403) — likely a misconfigured `llm_model_url`
- Model not found (404)

---

## 6. Fallback Logic

### 6.1 Purpose

The fallback guarantees pipeline progress even when the LLM is unavailable, rate-limited, or producing invalid outputs. It runs the exact same logic as the existing `metric-decision-ps` + `hp-adjust-ps` steps, ensuring backward compatibility.

### 6.2 Implementation

```python
def fallback_decide(
    eval_results: dict,
    task_type: str,
    primary_metric: str,
    metric_direction: str,
    metric_threshold: float,
    current_iteration: int,
    max_retrain_iterations: int,
    hp_history: list,
    current_hyperparams: dict,
    early_stop_min_delta: float,
    overfitting_detection: bool,
    tuning_strategy: str,
    search_space: dict,
    lr_decay_factor: float,
    unfreeze_on_retry: bool,
    seed: int,
) -> dict:
    """
    Run the existing hardcoded decision + HP adjustment logic.
    Returns the same output format as the LLM path.
    """
    # Step 1: Metric decision (threshold comparison, early stopping, overfitting)
    #   → decision, metric_value, metric_name, is_overfitting, reason
    # (imports MetricDecision from shared-autoloop/metric-decision-ps)

    # Step 2: If decision == "retrain", run HP adjustment
    #   → hyperparams_json
    # (imports HPAdjustment from shared-autoloop/hp-adjust-ps)

    # Step 3: Combine into unified output format
    return {
        "decision": decision,
        "reasoning": f"[FALLBACK] {reason}",
        "confidence": 1.0,  # Deterministic logic has full "confidence"
        "is_overfitting": is_overfitting,
        "next_hyperparams": new_hps if decision == "retrain" else {},
    }
```

### 6.3 Fallback Triggering

| Scenario | Behavior |
|---|---|
| `llm_model_url` is empty string | Fallback immediately (no LLM configured) |
| LLM API returns invalid JSON 3x | Fallback after exhausting retries |
| LLM API timeout 3x | Fallback after exhausting retries |
| LLM API 401/403/404 | Fallback immediately (non-retryable) |
| LLM returns valid JSON | Use LLM decision (normal path) |

### 6.4 Observability

The `strategy_metadata` output always indicates which path was taken:

**LLM success:**
```json
{
  "method": "llm",
  "model_url": "https://clarifai.com/openai/chat-completion/models/gpt-4o-mini",
  "temperature": 0.1,
  "attempts": 1,
  "prompt_tokens": 2480,
  "completion_tokens": 185,
  "latency_ms": 4200
}
```

**LLM failure → Fallback:**
```json
{
  "method": "fallback",
  "fallback_reason": "LLM retries exhausted: [invalid decision 'continue', missing next_hyperparams]",
  "attempts": 3,
  "fallback_strategy": "schedule",
  "fallback_task_type": "detection"
}
```

---

## 7. History Management

### 7.1 Updated History Entry Format

The LLM decision step extends the history entry with a `reasoning` field:

```json
{
  "iteration": 2,
  "hyperparams": {"per_item_lrate": 0.0009375, "frozen_stages": 0},
  "metrics": {"AP": 0.42, "AP50": 0.66, "AP75": 0.38},
  "decision": "retrain",
  "reason": "AP 0.4200 below threshold 0.50",
  "reasoning": "AP improved from 0.32 to 0.42 (+31%) after unfreezing backbone. The trend suggests further improvement is likely with a lower LR and more epochs. Recommending continued exploration.",
  "method": "llm"
}
```

**Backward compatibility**: If the fallback is used, `reasoning` contains the fallback reason string prefixed with `[FALLBACK]`, and `method` is `"fallback"`.

### 7.2 Size Budget

| Field | Typical Size |
|---|---|
| Base entry (without reasoning) | ~200–300 bytes |
| Reasoning (LLM) | ~200–500 bytes |
| **Total per entry** | ~400–800 bytes |
| **10 iterations** | ~4–8 KB |

Still well within Argo's 256KB output limit.

---

## 8. Simplified DAG Architecture

### 8.1 Before (Two Decision Steps)

```
Train → Eval → metric-decision → (deploy: export)
                                → (retrain: hp-adjust → recursive Train)
                                → (stop: report)
```
5 tasks in DAG, 2 CPU-only decision steps, intermediate `is_overfitting` hand-off.

### 8.2 After (Single LLM Step)

```
Train → Eval → llm-decide → (deploy: export)
                           → (retrain: recursive Train)
                           → (stop: report)
```
4 tasks in DAG, 1 CPU-only decision step, `hyperparams_json` produced directly.

### 8.3 Argo DAG Task Definition

```yaml
# ── STEP 3: LLM Decision (replaces metric-decision + hp-adjust) ──
- name: llm-decide
  depends: "eval.Succeeded"
  templateRef:
    name: users/<USER>/apps/<APP>/pipeline_steps/llm-decision-ps/versions/<VERSION>
    template: users/<USER>/apps/<APP>/pipeline_steps/llm-decision-ps/versions/<VERSION>
  arguments:
    parameters:
      - name: eval_results_json
        value: "{{tasks.eval.outputs.parameters.eval_results}}"
      - name: task_type
        value: "{{workflow.parameters.task_type}}"
      - name: task_description
        value: "{{workflow.parameters.task_description}}"
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
      - name: search_space
        value: "{{workflow.parameters.search_space}}"
      - name: tuning_strategy
        value: "{{workflow.parameters.tuning_strategy}}"
      - name: lr_decay_factor
        value: "{{workflow.parameters.lr_decay_factor}}"
      - name: unfreeze_on_retry
        value: "{{workflow.parameters.unfreeze_on_retry}}"
      - name: early_stop_min_delta
        value: "{{workflow.parameters.early_stop_min_delta}}"
      - name: overfitting_detection
        value: "{{workflow.parameters.overfitting_detection}}"
      - name: llm_model_url
        value: "{{workflow.parameters.llm_model_url}}"
      - name: llm_temperature
        value: "{{workflow.parameters.llm_temperature}}"
      - name: llm_max_retries
        value: "{{workflow.parameters.llm_max_retries}}"
      - name: seed
        value: "{{workflow.parameters.seed}}"

# ── BRANCH: Retrain (recursive) ──
- name: retrain
  depends: "llm-decide.Succeeded"
  when: "{{tasks.llm-decide.outputs.parameters.decision}} == retrain"
  template: autoloop
  arguments:
    parameters:
      - name: current_iteration
        value: "{{tasks.llm-decide.outputs.parameters.next_iteration}}"
      - name: current_hyperparams
        value: "{{tasks.llm-decide.outputs.parameters.hyperparams_json}}"
      - name: hp_history
        value: "{{tasks.llm-decide.outputs.parameters.hp_history}}"
```

---

## 9. LLM Model Selection

### 9.1 Recommended Models (Clarifai-hosted)

| Model | URL Pattern | Strengths |
|---|---|---|
| GPT-4o-mini | `https://clarifai.com/openai/chat-completion/models/gpt-4o-mini` | Low cost (~$0.001/call), fast (2–4s), excellent JSON compliance |
| GPT-4o | `https://clarifai.com/openai/chat-completion/models/gpt-4o` | Higher reasoning quality, higher cost (~$0.01/call) |
| Claude 3.5 Sonnet | `https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet` | Strong at structured analysis, 200K context |

### 9.2 Guidance

- **Default recommendation**: GPT-4o-mini — best cost/quality tradeoff for this use case (structured decision-making, not creative generation)
- **High-stakes production**: GPT-4o or Claude 3.5 Sonnet for better reasoning on complex training trajectories
- **Air-gapped environments**: If no LLM is available, leave `llm_model_url=""` to use the fallback path exclusively

---

## 10. Implementation Plan

### 10.1 File Structure

```
shared-autoloop/llm-decision-ps/
├── config.yaml                # Pipeline step compute config
├── Dockerfile                 # Python 3.11-slim + clarifai SDK
├── requirements.txt           # clarifai>=12.3,<13.0.0
└── 1/
    ├── pipeline_step.py       # Entry point (reflection pattern)
    └── models/
        └── model/
            └── 1/
                ├── model.py       # LLMDecision class with decide() method
                ├── prompts.py     # Prompt construction (system + user)
                └── fallback.py    # Hardcoded fallback (wraps existing steps)
```

### 10.2 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `clarifai` | >=12.3,<13.0.0 | Model predict API for LLM inference |
| Python stdlib `json` | — | JSON parsing/serialization |
| Python stdlib `logging` | — | Structured logging |
| Python stdlib `os` | — | File output |

No additional dependencies. The step imports `MetricDecision` and `HPAdjustment` from the co-located `shared-autoloop/` steps for fallback logic.

### 10.3 Clarifai Model Predict API Usage

```python
from clarifai.client.user import User

def call_llm(model_url: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call a Clarifai-hosted LLM and return the text response."""
    user = User(user_id="...", pat="...")  # Resolved from environment
    model = user.model(url=model_url)

    response = model.predict_by_bytes(
        input_bytes=user_prompt.encode("utf-8"),
        input_type="text",
        inference_params={
            "temperature": temperature,
            "max_tokens": 1024,
            "system_prompt": system_prompt,
        },
    )
    return response.outputs[0].data.text.raw
```

---

## 11. Migration Path

### 11.1 Backward Compatibility

- **Existing `metric-decision-ps` and `hp-adjust-ps` remain untouched** in `shared-autoloop/`. Pipelines that use the old two-step pattern continue to work.
- **The new step is additive** — teams can migrate one autoloop at a time by updating their `config.yaml` to reference `llm-decision-ps` instead of the two separate steps.
- **Fallback guarantees deterministic behavior** — if `llm_model_url=""`, the step behaves identically to the old two-step flow (same thresholds, same strategies, same outputs).

### 11.2 Migration Checklist

For each autoloop pipeline (`classifier-pipeline-resnet-autoloop`, `detector-pipeline-yolof-autoloop`, `lora-pipeline-unsloth-autoloop`):

1. Add new workflow parameters: `llm_model_url`, `task_description`, `llm_temperature`, `llm_max_retries`
2. Remove DAG tasks: `decide` (metric-decision-ps) and `hp-adjust` (hp-adjust-ps)
3. Add DAG task: `llm-decide` (llm-decision-ps) with combined parameter set
4. Update `retrain` task dependency: `hp-adjust.Succeeded` → `llm-decide.Succeeded`
5. Update `retrain` task arguments: `hyperparams_json` from `llm-decide` output
6. Update `export` and `report-failure` dependencies: `decide` → `llm-decide`
7. Verify conditional expressions use `llm-decide.outputs.parameters.decision`

---

## 12. Testing Strategy

### 12.1 Unit Tests

| Category | Tests | Mock Strategy |
|---|---|---|
| Prompt construction | Verify all context included, proper formatting, edge cases (empty history, custom metrics) | No mocks needed |
| LLM response parsing | Valid JSON, malformed JSON, markdown-wrapped, partial JSON | No mocks needed (string input) |
| Schema validation | All valid/invalid combinations of decision + next_hyperparams + confidence | No mocks needed |
| HP clamping | Values within bounds, out of bounds, edge values, unknown keys | No mocks needed |
| Fallback logic | Each task type × each decision | Mock eval_results file |
| Retry logic | Succeed on attempt 1, 2, 3, all fail | Mock LLM call |
| End-to-end (LLM path) | Deploy, retrain, stop for each task type | Mock LLM returning valid JSON |
| End-to-end (fallback) | All LLM retries fail → fallback runs correctly | Mock LLM raising exception |
| History accumulation | Verify entries appended correctly with reasoning field | Mock LLM call |

### 12.2 Integration Tests

| Scenario | Validation |
|---|---|
| Real LLM call (GPT-4o-mini) → deploy decision | Schema valid, decision correct, reasoning present |
| Real LLM call → retrain decision | HP values within search space, reasoning justifies choice |
| Pipeline step CLI invocation | All output files written to /tmp/ |

### 12.3 Regression Tests

Ensure the fallback produces **identical outputs** to running `metric-decision-ps` + `hp-adjust-ps` sequentially for the same inputs. This is tested by:
1. Running both paths with identical inputs across all 3 task types
2. Comparing `decision`, `hyperparams_json`, `is_overfitting`, `hp_history`
3. Only `reasoning` and `strategy_metadata` should differ

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM produces invalid JSON | Decision step fails | JSON parsing with retry + fallback; low temperature reduces this |
| LLM recommends HPs outside valid range | Training may fail or waste resources | Strict validation + clamping to bounds |
| LLM API rate limiting | Decision step delayed | Retry with backoff; fallback if exhausted |
| LLM API outage | No decision possible from LLM | Deterministic fallback — pipeline continues regardless |
| LLM "hallucinates" strategy changes | Unexpected training behavior | HP values validated against search space; clamped to bounds |
| Cost at scale (many pipelines) | Increased API costs | GPT-4o-mini at ~$0.001/call; 10 iterations = ~$0.01/pipeline run |
| LLM decision inconsistency across identical inputs | Non-deterministic behavior | Temperature=0.1 minimizes variance; fallback is fully deterministic |
| Prompt injection via task_description | LLM produces unintended output | Output is schema-validated; even adversarial prompts can only produce deploy/retrain/stop |

---

## 14. Future Extensions

1. **Bayesian optimization integration** — LLM can suggest promising regions in HP space, combined with a GP surrogate model for better exploration
2. **Cost-aware decisions** — Include compute cost estimates in the prompt; LLM can balance quality vs. budget
3. **Multi-objective optimization** — LLM reasons about tradeoffs (accuracy vs. latency, quality vs. model size)
4. **Cross-pipeline learning** — Share training histories across similar pipelines; LLM can transfer knowledge from past runs
5. **Confidence-gated deployment** — Use the `confidence` score to gate deployment: only deploy if confidence > 0.8, otherwise request human review
