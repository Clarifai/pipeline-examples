---
name: clarifai-code-review
description: >
  Create, configure, and troubleshoot the Python Code Review Clarifai Pipeline.
  Use when asked to "review Python code", "set up a code review pipeline",
  "run static analysis", or "plug a code review step into my pipeline".
  Works with any Clarifai-hosted LLM/VLM.
metadata:
  author: clarifai
  version: "1.0.0"
---

# Clarifai Python Code Review Pipeline Skill

A reusable Clarifai pipeline step that reviews Python code using static
analysis (pylint, flake8, bandit) and generates LLM-powered reviews via
any Clarifai-hosted model.

## When to Use

- User asks to review Python code or set up a code review pipeline.
- User wants to add static analysis to an existing Clarifai pipeline.
- User wants to integrate an LLM-based code reviewer into their workflow.
- User asks how to plug a review step into their LLM/VLM pipeline.

## Execution Steps

When the user requests a code review, the LLM should follow these steps
in order to produce a deterministic, high-quality review:

1. **Identify the code scope** — Determine which files or inline code the user wants reviewed. If a repository URL is provided, note the branch and file patterns. If inline code is given, treat it as a single file.
2. **Run static analysis** — Execute pylint, flake8, and bandit on every Python file in scope. Collect all findings normalized by severity (error, warning, convention, refactor, info).
3. **Filter findings by strictness** — Apply the user's strictness setting: `high` keeps only errors, `medium` keeps errors + warnings, `low` keeps everything.
4. **Organize findings by category** — Group findings into: Security (bandit), Bugs/Errors (pylint E/F), Warnings (pylint W, flake8), Style/Convention (pylint C/R), and Info.
5. **Generate the structured review** — Produce sections in this exact order: Summary (2-3 sentences + quality score 1-10), Critical Issues, Improvements, Style & Conventions, Positive Observations.
6. **Reference specific locations** — Every issue must cite the file name and line number. Explain WHY it is a problem and HOW to fix it.
7. **Prioritize by impact** — Lead with the most impactful issues. If there are security findings from bandit, always highlight them first.
8. **Compile the final output** — Return a JSON object with `status`, `files_reviewed`, `total_findings`, `findings` array, `review` text, and `strictness`.

## Project Structure

```
code-review-pipeline/
├── config.yaml                          # Root pipeline definition (Argo workflow)
├── template.json                        # UI parameter template for Clarifai platform
└── python-code-review-ps/               # The reusable pipeline step
    ├── config.yaml                      # Step identity, inputs, compute resources
    ├── requirements.txt                 # pylint, flake8, bandit, clarifai SDK
    ├── Dockerfile                       # Slim Python 3.11 + git
    └── 1/
        ├── pipeline_step.py             # Entry point (argparse → review())
        ├── analyzers.py                 # Static analysis runner
        └── llm_reviewer.py             # LLM integration via Clarifai SDK
```

## How It Works

The step runs three phases:
1. **Acquire code** — clone a git repo (`--repo_url`) or accept inline code (`--code_text`).
2. **Static analysis** — run pylint, flake8, and bandit; normalize findings by severity.
3. **LLM review** — send code + findings to any Clarifai-hosted LLM for a structured review.

## Parameters

| Parameter             | Required | Default                           | Description |
|-----------------------|----------|-----------------------------------|-------------|
| `repo_url`            | no*      | `""`                              | Git URL to clone and review |
| `branch`              | no       | `main`                            | Branch to checkout |
| `file_patterns`       | no       | `**/*.py`                         | Comma-separated glob patterns |
| `code_text`           | no*      | `""`                              | Raw Python code (alternative to repo_url) |
| `review_strictness`   | no       | `medium`                          | `low` / `medium` / `high` |
| `llm_model_url`       | no       | `""`                              | Any Clarifai model URL |
| `llm_inference_params`| no       | `{"temperature":0.3,"max_tokens":4096}` | JSON inference params |
| `user_id`             | yes      | —                                 | Clarifai user ID |
| `app_id`              | yes      | —                                 | Clarifai app ID |
| `save_results`        | no       | `false`                           | Persist results to Clarifai |

*Either `repo_url` or `code_text` must be provided.

## Quick Start

### 1. Set credentials

```bash
export CLARIFAI_PAT=YOUR_PAT_HERE
```

### 2. Edit placeholders

Replace `<YOUR_USER_ID>`, `<YOUR_APP_ID>`, `<YOUR_PIPELINE_ID>` in:
- `code-review-pipeline/config.yaml`
- `code-review-pipeline/python-code-review-ps/config.yaml`

### 3. Upload

```bash
cd code-review-pipeline
clarifai pipeline upload
```

### 4. Run

```bash
clarifai pipeline run \
  --compute_cluster_id YOUR_CLUSTER \
  --nodepool_id YOUR_NODEPOOL \
  --set repo_url="https://github.com/org/repo.git" \
  --set branch="main" \
  --set review_strictness="medium" \
  --set llm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o"
```

## Plugging Into an Existing Pipeline

Upload the step independently, then reference it from any pipeline:

```bash
cd code-review-pipeline/python-code-review-ps
clarifai pipelinestep upload
```

Then in your pipeline's `config.yaml`, add it as a step:

```yaml
step_directories:
  - your-existing-step
  - python-code-review-ps   # add this

# In the Argo workflow spec, add:
- - name: code-review
    templateRef:
      name: users/<OWNER_USER>/apps/<OWNER_APP>/pipeline_steps/python-code-review-ps
      template: users/<OWNER_USER>/apps/<OWNER_APP>/pipeline_steps/python-code-review-ps
    arguments:
      parameters:
        - name: repo_url
          value: "{{workflow.parameters.repo_url}}"
        - name: llm_model_url
          value: "{{workflow.parameters.llm_model_url}}"
        # ... other params as needed
```

## Supported LLM Models

Any Clarifai-hosted text-generation model works. Examples:

| Model | URL |
|-------|-----|
| GPT-4o | `https://clarifai.com/openai/chat-completion/models/gpt-4o` |
| Claude 3.5 Sonnet | `https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet` |
| Llama 3 70B | `https://clarifai.com/meta/llama-3/models/llama-3-70b-instruct` |
| Mistral Large | `https://clarifai.com/mistralai/completion/models/mistral-large` |

## Strictness Levels

| Level    | What gets reported |
|----------|--------------------|
| `high`   | Errors and fatal issues only |
| `medium` | Errors + warnings |
| `low`    | Everything including conventions, refactoring suggestions, and info |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "git is not installed" | Ensure the Dockerfile includes `apt-get install git` |
| Empty LLM response | Verify `llm_model_url` points to a valid text-generation model |
| "CLARIFAI_PAT not set" | Set it as an env var or configure it in pipeline secrets |
| Clone timeout | Increase `timeout` in pipeline_step.py or use `--depth 1` (already default) |
| Too many findings | Increase `review_strictness` to `high` to focus on critical issues |
