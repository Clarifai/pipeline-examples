# LoRA LLM Fine-Tuning Quick-Start Pipeline (`lora-pipeline-unsloth-quick-start`)

A quick-start Clarifai pipeline that LoRA-fine-tunes an LLM using [Unsloth](https://github.com/unslothai/unsloth) on a **default public HuggingFace dataset** (`mlabonne/FineTome-100k`). No dataset setup needed — launch it and watch an end-to-end fine-tuning run.

## What's in this example

A Clarifai pipeline consists of two things; this example illustrates both of them:

- `config.yaml` — pipeline-level config (Argo orchestration spec + pipeline parameters).
- `model-version-train-ps/` — the pipeline step:
  - `config.yaml` — compute requirements + per-step input parameters.
  - `Dockerfile` + `requirements.txt` — container/runtime environment.
  - `1/pipeline_step.py` — entry point that wires CLI args into the model's `train()` method.
  - `1/models/model/1/model.py` — the actual fine-tuning logic (Unsloth + TRL `SFTTrainer`).

## What this pipeline does

- **Get data**: loads the default HuggingFace dataset (`dataset_name`, default `mlabonne/FineTome-100k`), normalizes to ShareGPT format, and applies the model's chat template.
- **Base model**: loads the base LLM via Unsloth's `FastLanguageModel.from_pretrained` (`base_model_name`, default `unsloth/Qwen3-0.6B`), optionally in 4-bit.
- **LoRA setup**: attaches LoRA adapters (`lora_r`, `lora_alpha`, `lora_dropout`) on top of the base model.
- **Train**: fine-tunes with TRL `SFTTrainer` using the supplied schedule (`num_epochs`, `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type`, `warmup_ratio`, `weight_decay`, `max_steps`).
- **Save & upload**: saves the LoRA adapter and uploads a Clarifai model that loads the base model + adapter (served via vLLM), ready for deployment.

## Parameters

`config.yaml` exposes the parameters users typically override (`base_model_name`, `dataset_name`, `max_seq_length`, `load_in_4bit`, `lora_r`, `lora_alpha`, `lora_dropout`, `num_epochs`, `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type`, `warmup_ratio`, `weight_decay`, `max_steps`, `logging_steps`, `save_steps`, `seed`). For full descriptions see `model-version-train-ps/config.yaml` and the `train()` method signature in `model-version-train-ps/1/models/model/1/model.py`.

## Quick Start Guide

### Step 1: Set Up Your Environment

```bash
pip install clarifai
clarifai login   # interactive: paste your Personal Access Token when prompted
```

Alternatively, set the PAT non-interactively:

```bash
export CLARIFAI_PAT=<your_personal_access_token>
```

### Step 2: Initialize the Pipeline from Template

```bash
clarifai pipeline init --template=lora-pipeline-unsloth-quick-start
cd lora-pipeline-unsloth-quick-start
```

Optional — override defaults at init time (different user/app, a custom pipeline ID, or a parameter default):

```bash
clarifai pipeline init --template=lora-pipeline-unsloth-quick-start \
  --user_id MY_USER_ID --app_id MY_APP_ID \
  --set id=MY_PIPELINE_ID --set max_steps=100
```

### Step 3: Upload and Run the Pipeline

```bash
clarifai pipeline upload
```

Then run with one of the two compute options:

```bash
# (a) Simplest — auto-create or reuse compute from an instance type
clarifai pipeline run --instance=g6e.xlarge

# (b) Use your existing nodepool + compute cluster
clarifai pipeline run \
  --nodepool_id=<your_existing_nodepool_id> \
  --compute_cluster_id=<your_existing_compute_cluster_id>
```

Override parameters at run time by repeating `--set key=value`:

```bash
clarifai pipeline run --instance=g6e.xlarge \
  --set base_model_name=unsloth/Qwen3-0.6B \
  --set max_steps=100 \
  --set learning_rate=0.0002
```

### Step 4: Monitor

Go to `https://clarifai.com/<YOUR_USER_ID>/<YOUR_APP_ID>` — the **Pipelines** tab tracks the run, and the **Models** tab is where the fine-tuned model lands when training finishes.

## Datasets, artifacts, and deployment

- Pipelines overview: https://docs.clarifai.com/compute/pipelines/
- This quick-start uses a **public HuggingFace dataset** out of the box. To fine-tune on your own data, prepare a Clarifai dataset or point `dataset_name` at another HuggingFace dataset: https://docs.clarifai.com/create/datasets/create
- Manage Clarifai artifacts (the example uploads the model so it can be deployed; you can also persist the raw LoRA adapter as an artifact): https://docs.clarifai.com/create/artifacts/manage

The final step in `model.py` uploads the fine-tuned LoRA adapter packaged as a Clarifai model (base model + adapter served via vLLM) so it is immediately ready for deployment.

## Custom

This is an example — grab it, refactor `model.py` and the pipeline step with the Clarifai SDK, and shape the workflow to your own base model, dataset, chat template, and training schedule.
