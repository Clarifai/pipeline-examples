---
name: clarifai-smart-router
description: >
  Intelligent intent-based router that combines Python Code Review and
  Chart Analysis skills into a single pipeline. Classifies user prompts
  using keyword matching + optional LLM fallback and dispatches to the
  correct skill automatically.
metadata:
  author: clarifai
  version: "1.0.0"
---

# Clarifai Smart Router Pipeline Skill

A unified Clarifai pipeline step that acts as an intelligent router:
it classifies the user's intent from a natural-language prompt and
delegates to the appropriate downstream skill.

## When to Use

- User sends a **free-form prompt** and the system must decide whether
  to run code review or chart analysis.
- You want a **single pipeline endpoint** that handles both skills.
- You're building a **chat-like interface** where users describe tasks
  in natural language and the pipeline auto-selects the right tool.
- You want to **combine multiple skills** behind one Clarifai pipeline
  without requiring the caller to pick the skill manually.

## How Intent Classification Works

The router uses a **three-tier** classification strategy:

### Tier 0 — Input Modality Signal (instant, deterministic)
If the user provides an **image** (URL / base64 / file) without code → `chart_analysis`.
If the user provides **code** (code_text / repo_url) without an image → `code_review`.

### Tier 1 — Keyword + Regex Pattern Matching (fast, no API calls)
Each skill has a set of keywords and regex patterns derived from its
SKILL.md "When to Use" section:

**Code Review triggers:**
- Keywords: `review`, `code review`, `static analysis`, `pylint`, `flake8`,
  `bandit`, `lint`, `bug`, `vulnerability`, `refactor`, `code quality`...
- Patterns: `review.*code`, `check.*python`, `static analysis`, etc.

**Chart Analysis triggers:**
- Keywords: `chart`, `graph`, `plot`, `visualization`, `trend analysis`,
  `anomaly detection`, `extract data`, `bar chart`, `pie chart`...
- Patterns: `analyze.*chart`, `read.*graph`, `trend.*analysis`, etc.

### Tier 2 — LLM Classification (fallback for ambiguous prompts)
When Tier 1 confidence is below 0.35 or the top two skills are within
0.15 of each other, the router sends the prompt to any Clarifai-hosted
LLM for classification. The LLM receives the full "When to Use"
descriptions and returns a structured JSON verdict.

## Project Structure

```
smart-router-pipeline/
├── config.yaml                      # Root pipeline + Argo workflow
├── template.json                    # UI parameter definitions
├── run_demo.py                      # 7-scenario demo script
└── smart-router-ps/                 # Pipeline step
    ├── config.yaml                  # Step identity + all input params
    ├── Dockerfile                   # Container (includes both skills)
    ├── requirements.txt             # Combined dependencies
    └── 1/
        ├── pipeline_step.py         # SmartRouterStep (entry point)
        └── intent_classifier.py     # Two-tier intent classifier
```

## Parameters

### Router Parameters
| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `user_prompt` | **yes** | — | Natural-language task description (used for classification) |
| `model_url` | no | `""` | Clarifai model URL — serves as LLM, VLM, and Tier 2 classifier |

### Code Review Parameters (forwarded when intent = code_review)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `repo_url` | `""` | Git repository URL |
| `branch` | `main` | Git branch |
| `file_patterns` | `**/*.py` | Glob patterns |
| `code_text` | `""` | Inline Python code |
| `review_strictness` | `medium` | low / medium / high |

### Chart Analysis Parameters (forwarded when intent = chart_analysis)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_url` | `""` | Chart image URL |
| `image_base64` | `""` | Base64-encoded chart |
| `image_path` | `""` | Local file path |
| `analysis_type` | `general` | general / data_extraction / trend_analysis / comparison / anomaly_detection |
| `output_format` | `detailed` | summary / detailed / json_table / markdown |
| `additional_context` | `""` | Domain context |

### Shared Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_params` | `{"temperature": 0.3}` | Model inference settings |
| `user_id` | — | Clarifai user ID |
| `app_id` | — | Clarifai app ID |
| `save_results` | `false` | Persist to Clarifai |

## Quick Start

```python
import sys, os
sys.path.insert(0, "smart-router-pipeline/smart-router-ps/1")
from pipeline_step import SmartRouterStep

step = SmartRouterStep()

# The router auto-classifies and dispatches:
result = step.route(
    user_prompt="Review this Python code for security issues",
    code_text="import pickle; data = pickle.loads(user_input)",
    model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o",
    user_id="me", app_id="myapp",
)

result = step.route(
    user_prompt="Analyze this bar chart and extract all data points",
    image_url="https://example.com/chart.png",
    model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o",
    user_id="me", app_id="myapp",
)
```

## Output Format

The router returns a JSON object with two top-level keys:

```json
{
  "router": {
    "intent": "code_review",
    "confidence": 0.85,
    "method": "pattern",
    "reasoning": "Matched on: keyword:review, pattern:review.*code",
    "all_scores": {"code_review": 0.85, "chart_analysis": 0.0}
  },
  "skill_output": {
    "status": "completed",
    "...": "full output from the matched skill"
  }
}
```

## Adding New Skills

1. Create a new `SkillDescriptor` in `intent_classifier.py` with
   `name`, `when_to_use`, `keywords`, and `patterns`.
2. Add it to `SKILL_REGISTRY`.
3. Add a `_run_<skill_name>()` method in `pipeline_step.py`.
4. Add a routing branch in `SmartRouterStep.route()`.

## Supported Models

Any Clarifai-hosted LLM/VLM:
- `https://clarifai.com/openai/chat-completion/models/gpt-4o`
- `https://clarifai.com/anthropic/completion/models/claude-sonnet-4`
- `https://clarifai.com/google/gemini/models/gemini-2.0-flash`
- `https://clarifai.com/meta/llama/models/llama-3.2-90b-vision`
