# Detector YOLOF Auto-Loop Pipeline

Trains a YOLOF object detector with automated hyperparameter retuning. After
each training iteration the pipeline:

1. Evaluates the trained model and writes metrics.
2. Runs **`metric-decision-ps`** to decide whether the loop should stop or
   retry — comparing against a primary-metric threshold (default `AP`),
   max-iterations cap, early-stop plateau, and (optionally) overfitting
   detection.
3. If retry, runs **`hp-adjust-ps`** to propose a new hyperparameter set using
   one of the supported strategies (`schedule` / `grid` / `random`) — for the
   detector this typically tunes `per_item_lrate`, `frozen_stages`, etc.

The two decision steps are shared with the classifier and LoRA auto-loops via
symlink to [`../shared-autoloop/`](../shared-autoloop/).

## Design

See the design docs under
[`../shared-autoloop/retune-pipeline-design/`](../shared-autoloop/retune-pipeline-design/):

- [`DESIGN.md`](../shared-autoloop/retune-pipeline-design/DESIGN.md) — overall flow.
- [`hp-adjustment-design.md`](../shared-autoloop/retune-pipeline-design/hp-adjustment-design.md) — adjustment strategies and parameters.
- [`metric-decision-design.md`](../shared-autoloop/retune-pipeline-design/metric-decision-design.md) — decision rules.

## Usage

1. Fill in the placeholder values in [`config.yaml`](config.yaml) and tune the
   auto-loop parameters to taste.
2. Upload the pipeline:
   ```
   clarifai pipeline upload ./config.yaml
   ```
3. Trigger a run via the Clarifai CLI / UI / API.

## Tests

Unit tests for the shared steps live under
[`../shared-autoloop/hp-adjust-ps/tests/`](../shared-autoloop/hp-adjust-ps/tests/) and
[`../shared-autoloop/metric-decision-ps/tests/`](../shared-autoloop/metric-decision-ps/tests/).
