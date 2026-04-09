"""
VLM Chart Analyzer Module

Sends chart/graph images to any Clarifai-hosted VLM (Vision Language Model)
for analysis. Works with GPT-4o, Claude 3.5 Sonnet, Gemini Pro Vision,
LLaVA, and any other multimodal model on the platform.

The model is specified at runtime via `vlm_model_url`, making this module
completely model-agnostic.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("chart-analysis-step.vlm_analyzer")


# ── Analysis-type-specific system prompts ────────────────────────────

ANALYSIS_PROMPTS: Dict[str, str] = {
    "general": """\
You are an expert data visualization analyst. Analyze the provided chart/graph image and produce a comprehensive report.

Your analysis must include:
## Chart Type & Overview
Identify the chart type (bar, line, pie, scatter, heatmap, etc.) and give a 2-sentence summary.

## Data Points
Extract all visible data points, labels, axis values, and legends. Present them in a structured table format when possible.

## Key Findings
- Primary trends or patterns visible in the data.
- Notable comparisons between categories/series.
- Any outliers or anomalies.

## Insights & Interpretation
What does this data tell us? Provide actionable insights based on the visualized data.

## Data Quality Notes
Missing labels, truncated axes, misleading scales, or any visual issues that affect interpretation.

Rules:
- Be precise with numbers — read values from axes carefully.
- If exact values are unclear, provide best estimates with ranges.
- Use the additional context if provided to give domain-specific insights.
- Always mention the units shown on axes.
""",

    "data_extraction": """\
You are a data extraction specialist. Your job is to extract ALL numerical and categorical data from the chart image into a structured format.

Produce your output as:
## Extracted Data Table
Present ALL data points in a markdown table with proper column headers.

## Axis Information
- X-axis: label, unit, range
- Y-axis: label, unit, range
- Any secondary axes

## Legend / Series
List all data series with their labels and visual identifiers (colors, patterns).

## Raw Values
List every readable value as: Category/Label → Value (Unit)

Rules:
- Extract EVERY visible data point, not just highlights.
- Estimate values from axis gridlines when exact labels aren't shown.
- Clearly mark estimated values with ≈.
- Note any data points that are obscured or unclear.
""",

    "trend_analysis": """\
You are a trend analysis expert. Analyze the temporal or sequential patterns in this chart.

Your analysis must include:
## Trend Direction
Overall direction: upward, downward, flat, cyclical, or mixed. Quantify the change (e.g., +25% over the period).

## Rate of Change
Is the trend accelerating, decelerating, or constant?

## Inflection Points
Identify any turning points where the trend changes direction. Note approximate dates/values.

## Seasonality & Patterns
Any repeating patterns, cycles, or periodic behavior visible.

## Forecast Implications
Based on the visible trend, what might we expect going forward? (State clearly this is visual extrapolation, not a statistical forecast.)

## Confidence Assessment
How reliable is this trend? Consider: data density, noise level, time range, and any visible outliers.
""",

    "comparison": """\
You are a comparative data analyst. Analyze how different categories, series, or groups compare in this chart.

Your analysis must include:
## Categories Compared
List all groups/series being compared.

## Rankings
Rank all categories from highest to lowest on the primary metric.

## Key Differences
- Largest gap between categories.
- Closest/most similar categories.
- Any surprising rankings.

## Proportional Analysis
What percentage of the total does each category represent? (If applicable.)

## Statistical Observations
Mean, median (if estimable), range, and any visible distribution characteristics.
""",

    "anomaly_detection": """\
You are a data anomaly detection specialist. Examine this chart for unusual patterns, outliers, and data quality issues.

Your analysis must include:
## Outliers
Identify any data points significantly different from the general pattern. Specify their location and magnitude.

## Pattern Breaks
Any sudden changes, gaps, or discontinuities in the data.

## Missing Data
Any visible gaps, interpolated sections, or missing categories.

## Scale Issues
Check for: truncated axes, logarithmic vs linear scale misuse, broken axes, dual axes with mismatched scales.

## Visual Anomalies
Any rendering issues, overlapping data, or visual elements that could misrepresent the data.

## Severity Assessment
Rate each anomaly: CRITICAL (affects conclusions), WARNING (notable but minor), INFO (cosmetic).
""",
}


class VLMAnalyzer:
    """Analyze charts/graphs using any Clarifai-hosted VLM."""

    def __init__(
        self,
        model_url: str,
        pat: str = "",
        inference_params: Optional[dict] = None,
    ):
        """
        Args:
            model_url: Clarifai VLM model URL, e.g.
                https://clarifai.com/openai/chat-completion/models/gpt-4o
            pat: Personal Access Token (falls back to CLARIFAI_PAT env).
            inference_params: Dict with temperature, max_tokens, etc.
        """
        self.model_url = model_url
        self.pat = pat or os.environ.get("CLARIFAI_PAT", "")
        self.inference_params = inference_params or {
            "temperature": 0.2,
            "max_tokens": 4096,
        }

    def analyze(
        self,
        image_bytes: bytes,
        analysis_type: str = "general",
        additional_context: str = "",
        output_format: str = "detailed",
    ) -> Dict[str, Any]:
        """
        Send the chart image to the VLM for analysis.

        Args:
            image_bytes: Raw image bytes.
            analysis_type: One of general, data_extraction, trend_analysis,
                comparison, anomaly_detection.
            additional_context: Extra domain context from the user.
            output_format: summary, detailed, json_table, markdown.

        Returns:
            Dict with analysis_text, analysis_type, model_used, and metadata.
        """
        system_prompt = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS["general"])

        # Build user prompt
        user_parts: List[str] = ["Analyze the chart/graph image provided."]

        if additional_context:
            user_parts.append(f"\n**Additional Context:** {additional_context}")

        if output_format == "summary":
            user_parts.append(
                "\nKeep your response concise — max 200 words, "
                "focusing on the single most important takeaway."
            )
        elif output_format == "json_table":
            user_parts.append(
                "\nReturn ALL extracted data as a JSON object with keys: "
                "'chart_type', 'title', 'axes', 'data_points' (array of objects), "
                "'summary'. Wrap the JSON in a ```json code block."
            )
        elif output_format == "markdown":
            user_parts.append(
                "\nFormat your entire response as clean Markdown "
                "suitable for embedding in a report."
            )

        user_prompt = "\n".join(user_parts)

        try:
            analysis_text = self._call_vlm(system_prompt, user_prompt, image_bytes)
        except Exception as exc:
            logger.error("VLM call failed: %s", exc)
            analysis_text = self._fallback_response(analysis_type)

        return {
            "analysis_text": analysis_text,
            "analysis_type": analysis_type,
            "output_format": output_format,
            "model_used": self.model_url,
        }

    # ------------------------------------------------------------------
    # Clarifai VLM call
    # ------------------------------------------------------------------

    def _call_vlm(
        self, system_prompt: str, user_prompt: str, image_bytes: bytes
    ) -> str:
        """Call the Clarifai-hosted VLM with image + text (multimodal)."""
        from clarifai.client.model import Model
        from clarifai.client.input import Inputs

        logger.info("Calling VLM: %s", self.model_url)

        model = Model(url=self.model_url, pat=self.pat)

        # Build the combined text prompt
        full_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # Build inference params
        inf_params = {}
        if "temperature" in self.inference_params:
            inf_params["temperature"] = float(self.inference_params["temperature"])
        if "max_tokens" in self.inference_params:
            inf_params["max_tokens"] = int(self.inference_params["max_tokens"])

        # Create a single multimodal input proto with BOTH image and text
        input_proto = Inputs.get_input_from_bytes(
            input_id="chart_analysis",
            image_bytes=image_bytes,
            text_bytes=full_text.encode("utf-8"),
        )

        response = model.predict(
            inputs=[input_proto],
            inference_params=inf_params if inf_params else {},
        )

        # Extract text from response
        if response.outputs:
            output = response.outputs[0]
            if hasattr(output, "data") and hasattr(output.data, "text"):
                text = output.data.text.raw
                if text:
                    return text

        raise RuntimeError(
            "Empty response from VLM model. "
            "Verify the vlm_model_url points to a valid vision-language model."
        )

    @staticmethod
    def _fallback_response(analysis_type: str) -> str:
        """Return a fallback message when the VLM is unavailable."""
        return (
            f"## Chart Analysis (Fallback)\n\n"
            f"The VLM model could not be reached. Analysis type requested: "
            f"**{analysis_type}**.\n\n"
            f"To get a full analysis, ensure:\n"
            f"1. `vlm_model_url` points to a valid Clarifai VLM model.\n"
            f"2. `CLARIFAI_PAT` environment variable is set.\n"
            f"3. The model supports image+text (multimodal) inputs.\n\n"
            f"**Supported models:** GPT-4o, Claude 3.5 Sonnet, Gemini Pro Vision, "
            f"LLaVA, Qwen-VL, InternVL, etc."
        )

    @staticmethod
    def get_supported_analysis_types() -> List[str]:
        """Return the list of supported analysis types."""
        return list(ANALYSIS_PROMPTS.keys())
