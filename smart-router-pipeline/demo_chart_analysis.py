#!/usr/bin/env python3
"""
Demo: Smart Router → Chart Analysis

Routes a single chart-analysis request through the smart router.
GPT-4o (VLM) analyses a programmatically generated bar chart.
"""
import json
import os
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "smart-router-ps", "1"))

from pipeline_step import SmartRouterStep

MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"


def _generate_chart() -> str:
    """Create a simple bar chart PNG and return its path."""
    from PIL import Image, ImageDraw

    chart_path = os.path.join(SCRIPT_DIR, "demo_chart.png")
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 10), "Sales by Region", fill="black")

    data = {"North": 85, "South": 62, "East": 44, "West": 73}
    bar_w, gap, left, bottom = 60, 20, 40, 260
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853"]
    for i, (region, val) in enumerate(data.items()):
        x = left + i * (bar_w + gap)
        h = int(val * 2)
        draw.rectangle([(x, bottom - h), (x + bar_w, bottom)], fill=colors[i])
        draw.text((x + 15, bottom + 5), region, fill="black")
        draw.text((x + 20, bottom - h - 15), str(val), fill="black")
    draw.line([(left, bottom), (left + 4 * (bar_w + gap), bottom)], fill="black", width=2)
    draw.line([(left, 50), (left, bottom)], fill="black", width=2)
    img.save(chart_path)
    return chart_path


def main():
    print("=" * 72)
    print("  SMART ROUTER — CHART ANALYSIS DEMO")
    print("=" * 72)

    pat = os.environ.get("CLARIFAI_PAT", "")
    if not pat:
        print("\n  ⚠  CLARIFAI_PAT not set — VLM analysis will be skipped.\n")

    chart_path = _generate_chart()
    print(f"\n  Generated chart: {chart_path}")

    step = SmartRouterStep()
    result_json = step.route(
        user_prompt="Analyze this sales chart and extract the revenue by region",
        model_url=MODEL_URL,
        image_path=chart_path,
        analysis_type="data_extraction",
        output_format="json_table",
        additional_context="Regional sales figures in millions USD.",
        user_id="demo",
        app_id="demo",
    )

    data = json.loads(result_json)
    router = data.get("router", {})
    skill = data.get("skill_output", {})

    print(f"\n  Intent     : {router.get('intent')}")
    print(f"  Confidence : {router.get('confidence')}")
    print(f"  Method     : {router.get('method')}")
    print(f"  Reasoning  : {router.get('reasoning')}")
    print(f"  Scores     : {router.get('all_scores')}")

    print(f"\n  Status     : {skill.get('status')}")

    analysis = skill.get("analysis", "")
    if analysis:
        print(f"\n  ── VLM ANALYSIS ──\n{textwrap.indent(str(analysis)[:2000], '    ')}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
