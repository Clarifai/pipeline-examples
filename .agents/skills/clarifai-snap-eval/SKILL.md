---
name: clarifai-snap-eval
description: >
  Analyse evaluation results of a vision model trained for snap moment
  identification in American football videos. Accepts a CSV of predictions
  vs ground truth, normalises timestamps, ranks errors, classifies
  pass/fail, and produces per-play_type, per-view, and per-league
  breakdowns with cross-tabulated pattern analysis.
metadata:
  author: clarifai
  version: "1.0.0"
---

# Clarifai Snap Moment Eval Analysis Skill

A Clarifai pipeline step that ingests model-evaluation CSVs for the
snap-detection task in American football and produces a structured,
deterministic performance report.

## When to Use

- User asks to **analyse or evaluate snap detection results** from a football model
- User provides a **CSV file** with prediction vs ground-truth snap times
- User wants to **identify failure patterns** across play types, camera views, or leagues
- User needs a **per-category breakdown** (play_type, view, league) of model accuracy
- User asks to **compare model predictions** against ground-truth snap moments
- User wants to find which **play types, views, or leagues** the model struggles with

## Execution Steps

Follow these steps in order when analysing snap-detection eval results:

1. **Load and validate the CSV** — Read the evaluation CSV file (from path, URL, or inline text). Verify that the required columns exist: `video`, `game_id`, `play_id`, `team_id`, `play_type`, `league`, `view`, `has_gt`, `gt_snap_time`, `pred_time_sec`, `error_sec`, `detected`. Log the total row count and any missing columns.
2. **Normalise time columns** — Convert `gt_snap_time`, `pred_time_sec`, and `error_sec` to a consistent numeric format (float seconds). If any column contains timestamps in MM:SS or HH:MM:SS format, parse them into seconds. Drop rows where `has_gt` is false (no ground truth available).
3. **Recompute absolute error** — Calculate `abs_error = abs(pred_time_sec - gt_snap_time)` for every row. If the existing `error_sec` column differs from the recomputed value, log a warning and prefer the recomputed value.
4. **Sort by descending error** — Sort the full dataset by `abs_error` descending so the worst predictions appear first. This makes the worst failures immediately visible.
5. **Classify pass / fail** — Mark each row as `PASS` (abs_error ≤ 0.5 s) or `FAIL` (abs_error > 0.5 s). Add a boolean `is_failure` column.
6. **Compute overall summary** — Report: total rows, detected count, undetected count, pass count, fail count, overall accuracy (pass / total), mean absolute error (MAE), median absolute error, 95th-percentile error, and the worst single error.
7. **Analyse per play_type** — Group by `play_type` and compute for each: count, pass/fail counts, accuracy %, MAE, median error, worst error. Rank play types by failure rate descending.
8. **Analyse per view** — Same breakdown grouped by camera `view`. Rank views by failure rate descending.
9. **Analyse per league** — Same breakdown grouped by `league`. Rank leagues by failure rate descending.
10. **Cross-tabulate for pattern extraction** — Build pivot tables for `play_type × view` and `play_type × league` showing failure rates. Highlight cells where failure rate exceeds the overall average by more than 10 percentage points.
11. **Compile the worst-N list** — Output the top-20 worst predictions (highest abs_error) with all columns, so the user can inspect individual failures.
12. **Generate the LLM summary** — If a model URL is provided, send the aggregated statistics to the LLM and ask for a concise narrative: which categories are weakest, likely root causes, and recommended next steps.

## Project Structure

```
snap-eval-pipeline/
├── config.yaml                       # Root pipeline + Argo workflow
├── template.json                     # UI parameter definitions
├── run_demo.py                       # Demo with synthetic CSV data
└── snap-eval-ps/                     # Pipeline step
    ├── config.yaml                   # Step identity + input params
    ├── Dockerfile                    # Container image
    ├── requirements.txt              # Python dependencies
    └── 1/
        ├── pipeline_step.py          # Entry point (SnapEvalStep)
        └── eval_analyzer.py          # Core analysis engine
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `csv_path` | no* | `""` | Local path to the eval-results CSV |
| `csv_url` | no* | `""` | URL to download the CSV |
| `csv_text` | no* | `""` | Raw CSV content as a string |
| `failure_threshold` | no | `0.5` | Seconds; abs error above this = FAIL |
| `top_n_worst` | no | `20` | Number of worst predictions to list |
| `llm_model_url` | no | `""` | Clarifai LLM for narrative summary |
| `inference_params` | no | `{"temperature":0.3,"max_tokens":4096}` | LLM inference settings |
| `user_id` | yes | — | Clarifai user ID |
| `app_id` | yes | — | Clarifai app ID |
| `save_results` | no | `false` | Persist results to Clarifai |

*At least one of `csv_path`, `csv_url`, or `csv_text` must be provided.

## Quick Start (Local)

```python
import sys
sys.path.insert(0, "snap-eval-pipeline/snap-eval-ps/1")
from pipeline_step import SnapEvalStep

step = SnapEvalStep()
result = step.analyze(
    csv_path="/path/to/eval_results.csv",
    failure_threshold=0.5,
    llm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o",
)
```

## CSV Format

The minimum required columns:

| Column | Type | Description |
|--------|------|-------------|
| `video` | str | Video filename or ID |
| `game_id` | str | Game identifier |
| `play_id` | str | Play identifier |
| `team_id` | str | Team identifier |
| `play_type` | str | e.g. `pass`, `run`, `punt`, `kickoff` |
| `league` | str | e.g. `NFL`, `NCAA`, `XFL` |
| `view` | str | Camera angle, e.g. `Sideline`, `Endzone`, `All22` |
| `has_gt` | bool | Whether ground-truth snap time is available |
| `gt_snap_time` | float | Ground-truth snap time in seconds |
| `pred_time_sec` | float | Model-predicted snap time in seconds |
| `error_sec` | float | Pre-computed error (recomputed by skill) |
| `detected` | bool | Whether the model detected a snap at all |

Additional columns are preserved but not used in the analysis.

## Plugging Into an Existing Pipeline

```yaml
- name: snap-eval
  templateRef:
    name: users/<USER>/apps/<APP>/pipeline_steps/snap-eval-ps
    template: users/<USER>/apps/<APP>/pipeline_steps/snap-eval-ps
  arguments:
    parameters:
      - name: csv_path
        value: "{{workflow.parameters.csv_path}}"
      - name: llm_model_url
        value: "https://clarifai.com/openai/chat-completion/models/gpt-4o"
```

## Supported LLM Models

Any Clarifai-hosted text-generation model for the optional narrative summary:

| Model | URL |
|-------|-----|
| GPT-4o | `https://clarifai.com/openai/chat-completion/models/gpt-4o` |
| Claude 3.5 Sonnet | `https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet` |
| Llama 3 70B | `https://clarifai.com/meta/llama-3/models/llama-3-70b-instruct` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing required columns" | Ensure CSV has at least `gt_snap_time`, `pred_time_sec`, `play_type`, `view`, `league` |
| "No rows with ground truth" | Check `has_gt` column — analysis only covers rows where ground truth exists |
| Empty LLM summary | Verify `llm_model_url` points to a valid text-generation model |
| Wrong time format | The skill auto-detects MM:SS and HH:MM:SS — ensure times use `:` separators |
