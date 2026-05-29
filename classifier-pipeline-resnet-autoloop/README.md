# Classifier ResNet Auto-Loop Pipeline

Trains a ResNet image classifier with automated hyperparameter retuning. After
each training iteration the pipeline:

1. Evaluates the trained model and writes metrics.
2. Runs **`metric-decision-ps`** to decide whether the loop should stop or
   retry — comparing against a primary-metric threshold, max-iterations cap,
   early-stop plateau, and (optionally) overfitting detection.
3. If retry, runs **`hp-adjust-ps`** to propose a new hyperparameter set using
   one of the supported strategies (`schedule` / `grid` / `random`) and feeds
   that back into the next training iteration.

The two decision steps are shared with the detector and LoRA auto-loops via
symlink to [`../shared-autoloop/`](../shared-autoloop/).

## Design

See the design docs under
[`../shared-autoloop/retune-pipeline-design/`](../shared-autoloop/retune-pipeline-design/):

- [`DESIGN.md`](../shared-autoloop/retune-pipeline-design/DESIGN.md) — overall flow.
- [`hp-adjustment-design.md`](../shared-autoloop/retune-pipeline-design/hp-adjustment-design.md) — adjustment strategies and parameters.
- [`metric-decision-design.md`](../shared-autoloop/retune-pipeline-design/metric-decision-design.md) — decision rules.

## Usage

1. Fill in the placeholder values in [`config.yaml`](config.yaml)
   (`<YOUR_USER_ID>`, `<YOUR_APP_ID>`, `<YOUR_PIPELINE_ID>`, `<YOUR_DATASET_ID>`,
   etc.) and tune the auto-loop parameters (`max_retrain_iterations`,
   `metric_threshold`, `hp_strategy`, …) to taste.
2. Upload the pipeline:
   ```
   clarifai pipeline upload ./config.yaml
   ```
3. Trigger a run via the Clarifai CLI / UI / API.

## Tests

Unit tests for the shared steps live under
[`../shared-autoloop/hp-adjust-ps/tests/`](../shared-autoloop/hp-adjust-ps/tests/) and
[`../shared-autoloop/metric-decision-ps/tests/`](../shared-autoloop/metric-decision-ps/tests/).
