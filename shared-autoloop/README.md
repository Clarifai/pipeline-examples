# Autoloop Architecture

## Overview

Autoloop pipelines enable **iterative retraining** of models on Clarifai. Each autoloop pipeline references the base training pipeline directly via Argo `templateRef` — there are **no copied training steps**. The base pipelines already include autoloop support parameters (`skip_export`, `hyperparams_json`).

Shared decision steps (`hp-adjust-ps`, `metric-decision-ps`, `llm-decision-ps`) and evaluation steps (`classifier-eval-ps`, `lora-eval-ps`) live in `shared-autoloop/` and are symlinked into each autoloop folder.

## Directory Structure

```
shared-autoloop/
├── hp-adjust-ps/              ← Hyperparameter adjustment step (shared)
├── metric-decision-ps/        ← Metric-based decision step (shared)
├── llm-decision-ps/           ← LLM-based decision step (shared)
├── classifier-eval-ps/        ← Classifier evaluation step (shared)
├── lora-eval-ps/              ← LoRA evaluation step (shared)
├── retune-pipeline-design/    ← Design documentation
└── tests/                     ← Unit & integration tests (160+)

classifier-pipeline-resnet-autoloop/
├── config.yaml                ← Argo DAG referencing base pipeline via templateRef
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
└── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)

detector-pipeline-yolof-autoloop/
├── config.yaml                ← Argo DAG referencing base pipeline via templateRef
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
└── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)

lora-pipeline-unsloth-autoloop/
├── config.yaml                ← Argo DAG referencing base pipeline via templateRef
├── hp-adjust-ps  → ../shared-autoloop/hp-adjust-ps     (symlink)
└── metric-decision-ps → ../shared-autoloop/metric-decision-ps  (symlink)
```

### Key Design Decisions

- **No train-ps copies** — Autoloop `config.yaml` uses `templateRef` to invoke the base pipeline's training step directly (e.g., `classifier-pipeline-resnet-ps`). This eliminates code duplication.
- **Symlinked decision steps** — Decision logic is pipeline-agnostic and shared across all autoloop pipelines.
- **Base pipelines have autoloop built in** — Each base pipeline's `train()` method already accepts `skip_export` and `hyperparams_json` parameters, making it reusable in iterative loops.

---

## How Base Pipelines Support Autoloop

The base pipeline training steps accept two parameters that enable autoloop integration:

```python
def train(self,
          ...existing params...,
          skip_export: bool = False,       # Controls export behavior
          hyperparams_json: str = "{}",    # Receives HP overrides from decision step
          ) -> str:
```

| Parameter | Type | Purpose |
|-----------|------|---------|
| `skip_export` | `bool` | When `True`, skips full Clarifai model export and instead uploads training artifacts (checkpoints, configs) to the artifact store. Used in intermediate loop iterations. |
| `hyperparams_json` | `str` | JSON string containing hyperparameter overrides suggested by the decision step (hp-adjust, metric-decision, or llm-decision). |

### Hyperparameter Override

Early in the `train()` method, the incoming HP overrides are parsed and applied:

```python
hp_overrides = json.loads(hyperparams_json) if isinstance(hyperparams_json, str) else hyperparams_json
if hp_overrides:
    logging.info(f"Applying hyperparameter overrides: {hp_overrides}")
    if "per_item_lrate" in hp_overrides:
        per_item_lrate = float(hp_overrides["per_item_lrate"])
    if "num_epochs" in hp_overrides:
        num_epochs = int(hp_overrides["num_epochs"])
    # ... additional HP keys specific to each pipeline
```

The overrideable hyperparameters vary per pipeline:

| Pipeline | Overrideable Hyperparameters |
|----------|------------------------------|
| **ResNet Classifier** | `per_item_lrate`, `weight_decay`, `num_epochs`, `batch_size` |
| **YOLOF Detector** | `per_item_lrate`, `frozen_stages`, `num_epochs`, `batch_size` |
| **LoRA Unsloth** | `learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `weight_decay` |

### Conditional Export (`skip_export` Branch)

When `skip_export=True` (intermediate loop iterations), the train step uploads artifacts and eval metrics instead of publishing a model. When `skip_export=False` (final iteration), the standard full export runs.

---

## How the Autoloop Pipeline Works

The autoloop `config.yaml` defines an Argo DAG with recursive self-invocation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Autoloop Pipeline (Argo DAG)                      │
│                                                                     │
│   ┌──────────┐     ┌──────────┐     ┌──────────────┐               │
│   │  train   │────▶│   eval   │────▶│  llm-decide  │               │
│   │(base PS) │     │(eval PS) │     │ (decision PS)│               │
│   └──────────┘     └──────────┘     └──────┬───────┘               │
│                                            │                        │
│                        ┌───────────────────┼───────────────────┐    │
│                        ▼                   ▼                   ▼    │
│                    [deploy]           [retrain]             [stop]   │
│                  (export done)    (recurse w/ new HPs)    (fail)    │
└─────────────────────────────────────────────────────────────────────┘
```

**Loop flow:**

1. **Train step** (via `templateRef` to base pipeline) — Trains with current hyperparameters and `skip_export=true`. Uploads checkpoint artifacts and reports eval metrics.
2. **Eval step** — Evaluates the trained model and produces standardized metrics.
3. **Decision step** (llm-decision or metric-decision) — Receives eval metrics and HP history. Outputs one of:
   - `deploy` → Model already exported, report success
   - `retrain` → Recurse with new `hyperparams_json`
   - `stop` → Max iterations reached, report failure
4. **Retrain branch** — Recursively invokes the same DAG with updated `current_hyperparams`, `hp_history`, and incremented `current_iteration`.

### HP Expand Step

Before invoking the train step, an `hp-expand` template merges HP overrides with workflow defaults:

```python
hp = json.loads(current_hyperparams)
defaults = {"per_item_lrate": "...", "num_epochs": "...", ...}
merged = {**defaults, **{k: str(v) for k, v in hp.items() if k in defaults}}
# Writes each HP to /tmp/<key> as Argo output parameters
```

This ensures the train step receives individual HP parameters (not a JSON blob) for compatibility with the base pipeline's interface.

---

## Config.yaml Structure

Each autoloop `config.yaml` is an Argo Workflow template with:

1. **Training parameters** — Same as the base pipeline (user_id, dataset_id, num_epochs, etc.)
2. **Autoloop-specific parameters** — Control the loop behavior:
   - `metric_threshold` — Target metric to achieve
   - `max_retrain_iterations` — Maximum loop iterations (default: 3)
   - `tuning_strategy` — HP adjustment strategy (`schedule`, `bayesian`, etc.)
   - `search_space` — HP search bounds (`auto` or JSON)
3. **LLM decision parameters** — For llm-decision step:
   - `llm_model_url` — Clarifai model URL for the decision LLM
   - `task_description` — Context for the LLM
4. **Loop state parameters** — Track iteration state:
   - `current_iteration`, `current_hyperparams`, `hp_history`

Placeholder values (`<YOUR_USER_ID>`, `<YOUR_APP_ID>`, etc.) must be replaced with actual Clarifai resource IDs when deploying.

---

## Testing

All autoloop tests live in `shared-autoloop/` subdirectories and cover the shared decision/eval steps:

```bash
# Run all 160+ tests (requires .venv-llm-test with Python 3.11, clarifai 12.4.1)
source .venv-llm-test/bin/activate
RUN_LIVE_LLM_TESTS=1 python -m pytest shared-autoloop/ --tb=short -q
```

- **151 unit tests** — Run without any env vars
- **9 LLM integration tests** — Require `RUN_LIVE_LLM_TESTS=1` and `CLARIFAI_PAT`
