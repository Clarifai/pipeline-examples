#!/usr/bin/env python3
"""
Demo: Run the Chart Analysis pipeline step locally.

This demo downloads a sample chart image from the web and runs the
ChartAnalysisStep without a VLM (to verify image loading, preprocessing,
and the full pipeline flow). To get actual VLM analysis, set
vlm_model_url to a valid Clarifai VLM model.
"""
import sys
import os

# Add the step's source directory to the path
STEP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chart-analysis-ps", "1",
)
sys.path.insert(0, STEP_DIR)

from pipeline_step import ChartAnalysisStep


def demo_url_image(vlm_model_url=""):
    """Demo: Analyze a chart from a public URL."""
    print("\n" + "=" * 70)
    print("DEMO 1: Analyze chart from URL")
    print("=" * 70 + "\n")

    step = ChartAnalysisStep()
    result = step.analyze(
        image_url="https://quickchart.io/chart?c={type:'bar',data:{labels:['Q1','Q2','Q3','Q4'],datasets:[{label:'Revenue',data:[12,18,15,22]}]}}&w=600&h=400&format=png",
        analysis_type="general",
        output_format="detailed",
        vlm_model_url=vlm_model_url,
        additional_context="This is a quarterly revenue bar chart.",
        user_id="demo_user",
        app_id="demo_app",
    )
    return result


def demo_local_image(vlm_model_url=""):
    """Demo: Analyze a chart from a locally generated PNG."""
    print("\n" + "=" * 70)
    print("DEMO 2: Analyze chart from local file (generated test image)")
    print("=" * 70 + "\n")

    # Generate a simple chart image using PIL
    chart_path = _generate_sample_chart()

    step = ChartAnalysisStep()
    result = step.analyze(
        image_path=chart_path,
        analysis_type="data_extraction",
        output_format="json_table",
        vlm_model_url=vlm_model_url,
        additional_context="Quarterly sales data in millions USD for 2025.",
        user_id="demo_user",
        app_id="demo_app",
    )
    return result


def _generate_sample_chart() -> str:
    """Generate a simple bar chart PNG using Pillow (no matplotlib needed)."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 600, 400
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((180, 15), "Quarterly Sales 2025", fill="black")

    # Data
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [12.5, 18.3, 15.7, 22.1]
    max_val = max(values)
    bar_width = 80
    gap = 30
    chart_left = 80
    chart_bottom = 340
    chart_top = 60
    chart_height = chart_bottom - chart_top

    # Y-axis
    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill="black", width=2)
    # X-axis
    draw.line([(chart_left, chart_bottom), (width - 30, chart_bottom)], fill="black", width=2)

    # Y-axis labels
    for i in range(6):
        y_val = max_val * i / 5
        y_pos = chart_bottom - int((y_val / max_val) * chart_height)
        draw.text((10, y_pos - 8), f"${y_val:.0f}M", fill="gray")
        draw.line([(chart_left, y_pos), (width - 30, y_pos)], fill="#EEEEEE", width=1)

    # Bars
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853"]
    for i, (q, v) in enumerate(zip(quarters, values)):
        x = chart_left + gap + i * (bar_width + gap)
        bar_height = int((v / max_val) * chart_height)
        y_top = chart_bottom - bar_height

        draw.rectangle([(x, y_top), (x + bar_width, chart_bottom)], fill=colors[i])
        draw.text((x + 25, chart_bottom + 8), q, fill="black")
        draw.text((x + 15, y_top - 18), f"${v}M", fill="black")

    # Y-axis label
    draw.text((5, chart_top - 25), "Revenue (USD M)", fill="black")

    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_chart.png")
    img.save(chart_path)
    print(f"Generated sample chart: {chart_path}")
    return chart_path


def demo_base64_image(vlm_model_url=""):
    """Demo: Analyze a chart from base64-encoded image."""
    import base64

    print("\n" + "=" * 70)
    print("DEMO 3: Analyze chart from base64 input")
    print("=" * 70 + "\n")

    # Generate a chart and encode it
    chart_path = _generate_sample_chart()
    with open(chart_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    step = ChartAnalysisStep()
    result = step.analyze(
        image_base64=b64,
        analysis_type="trend_analysis",
        output_format="summary",
        vlm_model_url=vlm_model_url,
        additional_context="Quarterly sales showing growth trajectory.",
        user_id="demo_user",
        app_id="demo_app",
    )
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("CHART ANALYSIS PIPELINE — LOCAL DEMO")
    print("=" * 70)
    VLM_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"
    print(
        f"\nUsing VLM: {VLM_URL}\n"
        "CLARIFAI_PAT must be set in the environment.\n"
    )

    # Demo 2: Local file (uses Pillow — always available)
    demo_local_image(vlm_model_url=VLM_URL)

    # Demo 3: Base64 input
    demo_base64_image(vlm_model_url=VLM_URL)

    # Demo 1: URL download (requires internet)
    try:
        demo_url_image(vlm_model_url=VLM_URL)
    except Exception as e:
        print(f"\nURL demo skipped (network issue): {e}")

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
