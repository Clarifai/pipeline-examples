#!/usr/bin/env python3
"""
Demo: Snap-Moment Eval Analysis Pipeline

Generates a realistic synthetic evaluation CSV and runs the full
analysis pipeline to demonstrate all output sections.
"""
import json
import os
import random
import sys
import textwrap

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP_DIR = os.path.join(SCRIPT_DIR, "snap-eval-ps", "1")
sys.path.insert(0, STEP_DIR)

from pipeline_step import SnapEvalStep

VLM_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"


def banner(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


# ── Generate synthetic eval CSV ───────────────────────────────────────

def generate_sample_csv(n: int = 200, seed: int = 42) -> str:
    """Create a realistic synthetic snap-detection eval CSV."""
    random.seed(seed)

    play_types = ["pass", "run", "punt", "kickoff", "field_goal"]
    views = ["Sideline", "Endzone", "All22"]
    leagues = ["NFL", "NCAA"]
    teams = ["KC", "SF", "DAL", "GB", "BUF", "PHI", "CIN", "MIA"]

    # Define per-category difficulty (some combos are harder)
    difficulty = {
        ("punt", "Endzone"): 1.8,
        ("kickoff", "Endzone"): 1.5,
        ("punt", "All22"): 1.4,
        ("kickoff", "All22"): 1.3,
        ("field_goal", "Sideline"): 1.2,
        ("pass", "Sideline"): 0.6,
        ("run", "Sideline"): 0.7,
        ("pass", "All22"): 0.8,
        ("run", "All22"): 0.8,
    }

    rows = ["video,game_id,play_id,team_id,play_type,league,view,"
            "has_gt,gt_snap_time,pred_time_sec,error_sec,detected"]

    for i in range(n):
        play_type = random.choice(play_types)
        view = random.choice(views)
        league = random.choice(leagues)
        team = random.choice(teams)
        game_id = f"G{random.randint(1000, 9999)}"
        play_id = f"P{i+1:04d}"
        video = f"{game_id}_{play_id}_{view}.mp4"

        gt_snap = round(random.uniform(3.0, 25.0), 3)
        has_gt = True

        # Scale error by difficulty
        diff_key = (play_type, view)
        scale = difficulty.get(diff_key, 1.0)
        # NCAA is slightly harder
        if league == "NCAA":
            scale *= 1.15

        # Most predictions are good, some are bad
        if random.random() < 0.85:
            # Good prediction
            error = random.gauss(0, 0.2 * scale)
        else:
            # Bad prediction — larger error
            error = random.gauss(0, 0.8 * scale)

        pred = round(gt_snap + error, 3)
        abs_error = round(abs(error), 3)

        # Some predictions miss entirely (undetected)
        detected = True
        if random.random() < 0.03 * scale:
            detected = False
            pred = 0.0
            abs_error = round(gt_snap, 3)

        rows.append(
            f"{video},{game_id},{play_id},{team},{play_type},{league},{view},"
            f"{has_gt},{gt_snap},{pred},{abs_error},{detected}"
        )

    return "\n".join(rows)


# ── Test functions ────────────────────────────────────────────────────

def test_1_basic_analysis():
    """Run basic analysis on synthetic CSV (no LLM)."""
    banner("TEST 1: Basic Snap-Eval Analysis (no LLM)")

    csv_data = generate_sample_csv(200)
    step = SnapEvalStep()
    result = step.analyze(
        csv_text=csv_data,
        failure_threshold=0.5,
        top_n_worst=10,
        user_id="demo",
        app_id="demo",
    )

    data = json.loads(result)
    ov = data.get("overall", {})
    print(f"  Total rows    : {ov.get('total_rows')}")
    print(f"  Accuracy      : {ov.get('accuracy_pct')}%")
    print(f"  Failure rate  : {ov.get('failure_rate_pct')}%")
    print(f"  MAE           : {ov.get('mean_abs_error')}s")
    print(f"  Median error  : {ov.get('median_abs_error')}s")
    print(f"  P95 error     : {ov.get('p95_error')}s")
    print(f"  Worst error   : {ov.get('worst_error')}s")
    print()

    print("  Per Play Type:")
    for row in data.get("per_play_type", []):
        print(
            f"    {row['play_type']:15s}  "
            f"acc={row['accuracy_pct']:5.1f}%  "
            f"fail={row['fail']:3d}/{row['total']:3d}  "
            f"MAE={row['mae']:.4f}s"
        )
    print()

    print("  Per View:")
    for row in data.get("per_view", []):
        print(
            f"    {row['view']:15s}  "
            f"acc={row['accuracy_pct']:5.1f}%  "
            f"fail={row['fail']:3d}/{row['total']:3d}  "
            f"MAE={row['mae']:.4f}s"
        )
    print()

    print("  Per League:")
    for row in data.get("per_league", []):
        print(
            f"    {row['league']:15s}  "
            f"acc={row['accuracy_pct']:5.1f}%  "
            f"fail={row['fail']:3d}/{row['total']:3d}  "
            f"MAE={row['mae']:.4f}s"
        )
    print()

    hotspots = data.get("hotspots", [])
    if hotspots:
        print(f"  Failure Hotspots ({len(hotspots)}):")
        for h in hotspots[:5]:
            ctx = h.get("view") or h.get("league", "?")
            print(
                f"    {h['play_type']:12s} × {ctx:10s}  "
                f"fail={h['failure_rate_pct']:.1f}%  "
                f"(+{h['excess_pp']:.1f}pp, n={h['sample_size']})"
            )
    print()

    worst = data.get("worst_predictions", [])
    if worst:
        print(f"  Top-{len(worst)} Worst Predictions:")
        for w in worst[:5]:
            print(
                f"    {str(w.get('video', '?')):40s}  "
                f"type={str(w.get('play_type', '?')):10s}  "
                f"gt={str(w.get('gt_snap_time', '?')):>8s}  "
                f"pred={str(w.get('pred_time_sec', '?')):>8s}  "
                f"err={str(w.get('abs_error', '?')):>8s}  "
                f"{w.get('result', '?')}"
            )

    # Also print the full markdown report
    report = data.get("report", "")
    if report:
        print("\n  ── FULL MARKDOWN REPORT ─────────────────────")
        for line in report.split("\n")[:50]:
            print(f"  {line}")
        if report.count("\n") > 50:
            print(f"  ... ({report.count(chr(10)) - 50} more lines)")


def test_2_with_llm():
    """Run analysis with LLM narrative summary (GPT-4o)."""
    banner("TEST 2: Snap-Eval Analysis with LLM Summary (GPT-4o)")

    csv_data = generate_sample_csv(150, seed=99)
    step = SnapEvalStep()
    result = step.analyze(
        csv_text=csv_data,
        failure_threshold=0.5,
        top_n_worst=5,
        llm_model_url=VLM_URL,
        user_id="demo",
        app_id="demo",
    )

    data = json.loads(result)
    ov = data.get("overall", {})
    print(f"  Accuracy : {ov.get('accuracy_pct')}%")
    print(f"  MAE      : {ov.get('mean_abs_error')}s")
    print()

    summary = data.get("llm_summary", "")
    if summary:
        print("  ── LLM NARRATIVE SUMMARY ──")
        for line in textwrap.wrap(summary, width=70):
            print(f"    {line}")
    else:
        print("  (No LLM summary generated)")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  SNAP-MOMENT EVAL ANALYSIS — DEMO")
    print("=" * 72)

    pat = os.environ.get("CLARIFAI_PAT", "")
    if pat:
        print(f"\n  CLARIFAI_PAT is set ({pat[:8]}...)")
        print(f"  LLM: {VLM_URL}\n")
    else:
        print("\n  ⚠  CLARIFAI_PAT not set — test 2 will skip LLM summary.\n")

    test_1_basic_analysis()

    if pat:
        test_2_with_llm()
    else:
        print("\n  Skipping LLM test (no CLARIFAI_PAT).\n")

    banner("ALL TESTS COMPLETE")
