#!/usr/bin/env python3
"""
Demo: Smart Router Pipeline — Intent Classification + Skill Dispatch

This demo sends a variety of natural-language prompts to the Smart Router
and shows how it classifies each one and routes to the correct skill.

Test scenarios:
  1. Code review prompt (keyword match)        → code_review
  2. Chart analysis prompt (keyword match)      → chart_analysis
  3. Ambiguous prompt (LLM classification)      → depends on model
  4. Code input signal (code_text provided)     → code_review  (Tier 0)
  5. Image input signal (image_url provided)    → chart_analysis (Tier 0)
  6. Chart analysis with REAL VLM               → GPT-4o response
  7. Code review with REAL LLM                  → GPT-4o response
  8. Snap-eval prompt (keyword match)           → snap_eval
  9. Snap-eval with CSV + LLM                   → full analysis report
"""
import json
import os
import sys
import textwrap

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTER_DIR = os.path.join(SCRIPT_DIR, "smart-router-ps", "1")
sys.path.insert(0, ROUTER_DIR)

from pipeline_step import SmartRouterStep

VLM_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"


def banner(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def print_result(result_json: str):
    """Pretty-print the router result."""
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        print(result_json)
        return

    router = data.get("router", {})
    skill = data.get("skill_output", {})

    print(f"  Intent     : {router.get('intent', '?')}")
    print(f"  Confidence : {router.get('confidence', '?')}")
    print(f"  Method     : {router.get('method', '?')}")
    print(f"  Reasoning  : {router.get('reasoning', '?')}")
    print(f"  Scores     : {router.get('all_scores', {})}")
    print()

    # Show a trimmed version of the skill output
    status = skill.get("status", "")
    routed = skill.get("routed", True)
    if status:
        print(f"  Skill Status: {status}")
    if not routed:
        note = skill.get("note") or skill.get("message", "")
        print(f"  ⚠ Routed   : NO (passthrough)")
        if note:
            print(f"  Note       : {note}")
        response = skill.get("response", "")
        if response:
            print(f"  Response (first 500 chars):\n{textwrap.indent(str(response)[:500], '    ')}")

    # For code review
    if "total_findings" in skill:
        print(f"  Findings   : {skill['total_findings']}")
        files = skill.get("files_reviewed", [])
        print(f"  Files      : {files}")
        review = skill.get("review", "")
        if review:
            print(f"  Review (first 500 chars):\n{textwrap.indent(review[:500], '    ')}")

    # For chart analysis
    if "analysis" in skill:
        analysis = skill["analysis"]
        print(f"  Analysis (first 500 chars):\n{textwrap.indent(str(analysis)[:500], '    ')}")

    # For snap-eval
    if "report" in skill:
        report = skill.get("report", "")
        print(f"  Report (first 800 chars):\n{textwrap.indent(str(report)[:800], '    ')}")
    if "summary" in skill and isinstance(skill["summary"], dict):
        summ = skill["summary"]
        print(f"  Total rows : {summ.get('total_rows', '?')}")
        print(f"  Overall accuracy: {summ.get('overall_accuracy', '?')}")
        print(f"  Mean error : {summ.get('mean_error_sec', '?')}")
    if "llm_narrative" in skill:
        narr = skill.get("llm_narrative", "")
        print(f"  LLM Narrative (first 500 chars):\n{textwrap.indent(str(narr)[:500], '    ')}")

    if "model_used" in skill:
        print(f"  Model Used : {skill['model_used']}")


# ── Test Scenarios ────────────────────────────────────────────────────

def test_1_code_review_keyword():
    """Clear code review prompt — should match on keywords."""
    banner("TEST 1: Code Review (keyword match)")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="Review my Python code for bugs and security vulnerabilities",
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_2_chart_analysis_keyword():
    """Clear chart analysis prompt — should match on keywords.
    No image provided, so skill returns a helpful error message."""
    banner("TEST 2: Chart Analysis (keyword match, no image)")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="Analyze this bar chart and extract all the data points from the graph",
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_3_ambiguous_prompt():
    """Ambiguous prompt — Tier 1 might be low-confidence."""
    banner("TEST 3: Ambiguous Prompt")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="Help me understand this data",
        model_url=VLM_URL,
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_4_code_with_vague_prompt():
    """Code provided with vague prompt — model handles it directly,
    code_text is just a soft hint (not enough to force code_review)."""
    banner("TEST 4: Code + vague prompt (passthrough — model handles it)")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="What do you think?",
        code_text="def add(a, b): return a+b",
        model_url=VLM_URL,
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_5_image_with_vague_prompt():
    """Image provided with vague prompt — model handles it directly,
    image is just a soft hint (not enough to force chart_analysis)."""
    banner("TEST 5: Image + vague prompt (passthrough — model handles it)")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="What do you think?",
        image_url="https://quickchart.io/chart?c={type:'bar',data:{labels:['A','B'],datasets:[{data:[10,20]}]}}&w=400&h=300",
        model_url=VLM_URL,
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_6_chart_with_vlm():
    """Full chart analysis with real VLM (GPT-4o)."""
    banner("TEST 6: Chart Analysis with REAL VLM (GPT-4o)")

    # Generate a sample chart
    from PIL import Image, ImageDraw
    chart_path = os.path.join(SCRIPT_DIR, "sample_test_chart.png")
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 10), "Sales by Region", fill="black")
    # Simple bars
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

    step = SmartRouterStep()
    result = step.route(
        user_prompt="Analyze this sales chart and extract the revenue by region",
        model_url=VLM_URL,
        image_path=chart_path,
        analysis_type="data_extraction",
        output_format="json_table",
        additional_context="Regional sales figures in millions USD.",
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_7_code_review_with_llm():
    """Full code review with real LLM (GPT-4o)."""
    banner("TEST 7: Code Review with REAL LLM (GPT-4o)")
    buggy_code = textwrap.dedent("""\
        import pickle
        import os

        def load_data(user_input):
            data = pickle.loads(user_input)  # Security: deserializing untrusted data
            return data

        def process(items):
            result = []
            for i in range(len(items)):  # Style: should use enumerate
                result.append(items[i] * 2)
            return result

        x = 10
        y = 0
        print(x / y)  # Bug: division by zero
    """)

    step = SmartRouterStep()
    result = step.route(
        user_prompt="Review this Python code for bugs, security issues, and style problems",
        model_url=VLM_URL,
        code_text=buggy_code,
        review_strictness="low",
        user_id="demo", app_id="demo",
    )
    print_result(result)


def test_8_snap_eval_keyword():
    """Clear snap-eval prompt — should match on keywords (no CSV)."""
    banner("TEST 8: Snap-Eval (keyword match, no CSV)")
    step = SmartRouterStep()
    result = step.route(
        user_prompt="Analyze the snap detection evaluation results and show failure rates per play type and camera view",
        user_id="demo", app_id="demo",
    )
    print_result(result)


def _generate_sample_csv() -> str:
    """Generate a small synthetic eval CSV for testing."""
    import random
    random.seed(42)
    lines = ["video,game_id,play_id,team_id,play_type,league,view,has_gt,gt_snap_time,pred_time_sec,error_sec,detected"]
    play_types = ["pass", "run", "punt", "kickoff"]
    views = ["Sideline", "Endzone"]
    leagues = ["NFL", "NCAA"]
    for i in range(50):
        pt = random.choice(play_types)
        view = random.choice(views)
        league = random.choice(leagues)
        gt = round(random.uniform(5.0, 25.0), 2)
        # Difficulty scaling
        scale = 0.3 if pt == "pass" else 0.6 if pt == "run" else 1.0
        err = round(abs(random.gauss(0, scale)), 3)
        pred = round(gt + random.choice([-1, 1]) * err, 2)
        detected = 1 if err < 1.5 else 0
        lines.append(f"vid_{i:03d}.mp4,G{i//10},P{i},{i%6},{pt},{league},{view},1,{gt},{pred},{err},{detected}")
    return "\n".join(lines)


def test_9_snap_eval_with_csv():
    """Full snap-eval analysis with CSV data + optional LLM narrative."""
    banner("TEST 9: Snap-Eval with CSV data + LLM narrative")
    csv_text = _generate_sample_csv()
    step = SmartRouterStep()
    result = step.route(
        user_prompt="Analyze these snap detection eval results, show per-play-type and per-view breakdown, and highlight worst predictions",
        model_url=VLM_URL,
        csv_text=csv_text,
        failure_threshold=0.5,
        top_n_worst=10,
        user_id="demo", app_id="demo",
    )
    print_result(result)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  SMART ROUTER PIPELINE — INTENT CLASSIFICATION DEMO")
    print("=" * 72)

    pat = os.environ.get("CLARIFAI_PAT", "")
    if pat:
        print(f"\n  CLARIFAI_PAT is set ({pat[:8]}...)")
        print(f"  VLM/LLM: {VLM_URL}\n")
    else:
        print("\n  ⚠  CLARIFAI_PAT not set — tests 3, 6, 7 will use fallback.\n")

    # ── Classification-only tests (fast, no API calls) ──
    test_1_code_review_keyword()
    test_2_chart_analysis_keyword()
    test_8_snap_eval_keyword()

    # ── Tests that use Clarifai models ──
    if pat:
        test_3_ambiguous_prompt()
        test_4_code_with_vague_prompt()
        test_5_image_with_vague_prompt()
        test_6_chart_with_vlm()
        test_7_code_review_with_llm()
        test_9_snap_eval_with_csv()
    else:
        print("\n  Skipping VLM/LLM tests (no CLARIFAI_PAT).\n")

    banner("ALL TESTS COMPLETE")
