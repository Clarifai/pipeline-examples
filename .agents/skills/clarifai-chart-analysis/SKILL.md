---
name: clarifai-chart-analysis
description: Analyze charts, graphs, and data visualizations using Clarifai pipelines with any VLM (GPT-4o, Claude, Gemini, LLaVA). Extracts data points, identifies trends, detects anomalies, and generates actionable insights.
---

# Clarifai Chart Analysis Skill

## When to Use
- User asks to **analyze a chart, graph, plot, or data visualization**
- User wants to **extract data** from a chart image
- User needs **trend analysis** or **anomaly detection** on visual data
- User wants to **compare categories** shown in a chart
- User provides a chart **image URL, file path, or base64** and asks for insights

## Execution Steps

When the user requests chart analysis, the VLM should follow these steps
in order to produce a deterministic, thorough analysis:

1. **Load and validate the image** — Accept the chart image from one of: URL, base64 string, or local file path. Validate the image format (PNG, JPEG, GIF, WebP, SVG supported) and reject images over 20 MB.
2. **Preprocess the image** — Auto-resize images larger than 2048px on any dimension while maintaining aspect ratio. Detect MIME type for proper encoding.
3. **Select the analysis prompt** — Based on the `analysis_type` parameter, select the specialized system prompt: `general` (full overview), `data_extraction` (structured data tables), `trend_analysis` (temporal patterns), `comparison` (category ranking), or `anomaly_detection` (outliers & quality).
4. **Extract chart metadata** — Identify the chart type (bar, line, pie, scatter, etc.), read axis labels and units, identify all data series from the legend, and note the chart title.
5. **Extract all data points** — Read every visible data point from the chart. For unclear values, estimate from gridlines and mark with ≈. Present in a structured table when possible.
6. **Analyze patterns and trends** — Identify primary trends, notable comparisons, outliers, and anomalies. Quantify changes where possible (e.g., "+25% growth").
7. **Apply domain context** — If `additional_context` is provided, use it to give domain-specific insights (e.g., "quarterly revenue in USD" helps interpret axis values).
8. **Format the output** — Structure the response according to `output_format`: `summary` (max 200 words), `detailed` (full sections), `json_table` (JSON code block), or `markdown` (clean report format).
9. **Compile the final output** — Return a JSON object with `status`, `analysis` text, `analysis_type`, `output_format`, `image_source`, and `model_used`.

## Project Structure
```
chart-analysis-pipeline/
├── config.yaml                     # Root pipeline + Argo workflow
├── template.json                   # UI parameter definitions
└── chart-analysis-ps/              # Pipeline step
    ├── config.yaml                 # Step identity + input params
    ├── Dockerfile                  # Container image
    ├── requirements.txt            # Python dependencies
    └── 1/
        ├── pipeline_step.py        # Entry point (ChartAnalysisStep)
        ├── image_preprocessor.py   # Image loading & validation
        └── vlm_analyzer.py         # VLM-powered analysis engine
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_url` | `""` | URL of the chart image |
| `image_base64` | `""` | Base64-encoded chart image |
| `image_path` | `""` | Local file path to chart image |
| `analysis_type` | `"general"` | `general`, `data_extraction`, `trend_analysis`, `comparison`, `anomaly_detection` |
| `output_format` | `"detailed"` | `summary`, `detailed`, `json_table`, `markdown` |
| `vlm_model_url` | `""` | Any Clarifai VLM URL (required for analysis) |
| `vlm_inference_params` | `{"temperature":0.2}` | VLM inference settings |
| `additional_context` | `""` | Domain context (e.g. "quarterly revenue in USD") |
| `save_results` | `false` | Persist results to Clarifai |

## Quick Start (Local)

```python
import sys
sys.path.insert(0, "chart-analysis-pipeline/chart-analysis-ps/1")
from pipeline_step import ChartAnalysisStep

step = ChartAnalysisStep()

# Analyze a chart from URL
result = step.analyze(
    image_url="https://example.com/chart.png",
    analysis_type="general",
    vlm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o",
)

# Analyze from local file
result = step.analyze(
    image_path="/path/to/my-chart.png",
    analysis_type="data_extraction",
    output_format="json_table",
    vlm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o",
)
```

## Analysis Types

### `general` — Full Overview
Identifies chart type, extracts all data, provides trends and insights.

### `data_extraction` — Structured Data
Extracts every data point into a markdown table. Best for converting charts back into data.

### `trend_analysis` — Temporal Patterns
Focuses on direction, rate of change, inflection points, and seasonality.

### `comparison` — Category Analysis
Ranks categories, computes gaps, identifies proportional relationships.

### `anomaly_detection` — Outlier & Quality Check
Spots outliers, pattern breaks, missing data, and misleading visual elements.

## Supported VLM Models
Any Clarifai-hosted vision-language model:
- `https://clarifai.com/openai/chat-completion/models/gpt-4o`
- `https://clarifai.com/anthropic/completion/models/claude-sonnet-4`
- `https://clarifai.com/google/gemini/models/gemini-2.0-flash`
- `https://clarifai.com/meta/llama/models/llama-3.2-90b-vision`

## Plugging into Existing Pipelines
Add this step to any Argo workflow via `templateRef`:
```yaml
- name: analyze-chart
  templateRef:
    name: users/<USER>/apps/<APP>/pipeline_steps/chart-analysis-ps
    template: users/<USER>/apps/<APP>/pipeline_steps/chart-analysis-ps
  arguments:
    parameters:
      - name: image_url
        value: "{{workflow.parameters.chart_image_url}}"
      - name: analysis_type
        value: "data_extraction"
      - name: vlm_model_url
        value: "https://clarifai.com/openai/chat-completion/models/gpt-4o"
```

## Image Input Modes
1. **URL** — Downloads from any public/authenticated URL
2. **Base64** — Accepts raw base64 or `data:image/png;base64,...` format
3. **File path** — Reads from local filesystem (useful in multi-step pipelines)

Images larger than 2048px on any dimension are auto-resized to preserve VLM input limits.

## Troubleshooting
- **"Provide at least one of: image_url, image_base64, or image_path"** — No image input given
- **"Empty response from VLM model"** — Check that vlm_model_url supports image+text inputs
- **"Image too large"** — Images over 20MB are rejected; use smaller files
- **Poor extraction results** — Try `additional_context` to give the model domain hints
