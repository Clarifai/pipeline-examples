#!/usr/bin/env python3
"""
Demo: Smart Router → Snap-Moment Eval Analysis

Routes a single snap-eval request through the smart router.
Generates a synthetic 50-row CSV, analyses it, and optionally
calls GPT-4o for a narrative summary.
"""
import json
import os
import random
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "smart-router-ps", "1"))

from pipeline_step import SmartRouterStep

MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"


def _generate_csv() -> str:
    """Build a 50-row synthetic eval CSV."""
    random.seed(42)
    lines = [
        "video,game_id,play_id,team_id,play_type,league,view,"
        "has_gt,gt_snap_time,pred_time_sec,error_sec,detected"
    ]
    play_types = ["pass", "run", "punt", "kickoff"]
    views = ["Sideline", "Endzone"]
    leagues = ["NFL", "NCAA"]
    for i in range(50):
        pt = random.choice(play_types)
        view = random.choice(views)
        league = random.choice(leagues)
        gt = round(random.uniform(5.0, 25.0), 2)
        scale = 0.3 if pt == "pass" else 0.6 if pt == "run" else 1.0
        err = round(abs(random.gauss(0, scale)), 3)
        pred = round(gt + random.choice([-1, 1]) * err, 2)
        detected = 1 if err < 1.5 else 0
        lines.append(
            f"vid_{i:03d}.mp4,G{i // 10},P{i},{i % 6},"
            f"{pt},{league},{view},1,{gt},{pred},{err},{detected}"
        )
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  SMART ROUTER — SNAP-EVAL DEMO")
    print("=" * 72)

    pat = os.environ.get("CLARIFAI_PAT", "")
    if not pat:
        print("\n  ⚠  CLARIFAI_PAT not set — LLM narrative will be skipped.\n")

    csv_text = _generate_csv()
    print(f"\n  Generated synthetic CSV: {csv_text.count(chr(10))} rows")

    step = SmartRouterStep()
    result_json = step.route(
        user_prompt=(
            "Analyze these snap detection eval results, show per-play-type "
            "and per-view breakdown, and highlight worst predictions"
        ),
        model_url=MODEL_URL,
        csv_text=csv_text,
        failure_threshold=0.5,
        top_n_worst=10,
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

    # Summary stats
    summary = skill.get("summary", {})
    if isinstance(summary, dict) and summary:
        print(f"  Total rows : {summary.get('total_rows', '?')}")
        print(f"  Accuracy   : {summary.get('overall_accuracy', '?')}")
        print(f"  Mean error : {summary.get('mean_error_sec', '?')}")

    # Markdown report
    report = skill.get("report", "")
    if report:
        print(f"\n  ── EVAL REPORT ──\n{textwrap.indent(report[:2000], '    ')}")
        if len(report) > 2000:
            print(f"    ... ({len(report) - 2000} more chars)")

    # LLM narrative
    narrative = skill.get("llm_narrative", "")
    if narrative:
        print(f"\n  ── LLM NARRATIVE ──\n{textwrap.indent(narrative, '    ')}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
