"""
Snap-Moment Eval Analyzer — Core Analysis Engine

Takes a pandas DataFrame of snap-detection evaluation results and
produces a structured, deterministic performance report following
the execution steps defined in SKILL.md.
"""

import io
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("snap-eval-step.eval_analyzer")

# Columns the analysis absolutely requires
REQUIRED_COLUMNS = [
    "gt_snap_time",
    "pred_time_sec",
    "play_type",
    "view",
    "league",
]

# Columns expected but that can be synthesised / defaulted
OPTIONAL_COLUMNS = {
    "video": "",
    "game_id": "",
    "play_id": "",
    "team_id": "",
    "has_gt": True,
    "error_sec": None,  # recomputed
    "detected": True,
}


# ── Time Parsing ──────────────────────────────────────────────────────

def _parse_time_value(val) -> Optional[float]:
    """Convert a time value to float seconds.

    Handles:
      - Already numeric (int / float)
      - "MM:SS" or "MM:SS.fff"
      - "HH:MM:SS" or "HH:MM:SS.fff"
      - String-encoded floats ("12.345")
    """
    if val is None:
        return None

    # Already numeric
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return None
        return float(val)

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return None

    # Try plain float first
    try:
        return float(s)
    except ValueError:
        pass

    # HH:MM:SS.fff or MM:SS.fff
    m = re.match(
        r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d+))?$", s,
    )
    if m:
        hours = int(m.group(1)) if m.group(1) else 0
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        frac = float(f"0.{m.group(4)}") if m.group(4) else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac

    logger.warning("Could not parse time value: %r", val)
    return None


# ── Group Breakdown Helper ────────────────────────────────────────────

def _group_breakdown(
    df, group_col: str, failure_threshold: float,
) -> List[dict]:
    """Compute pass/fail stats per unique value of *group_col*."""
    import pandas as pd

    results = []
    for name, grp in df.groupby(group_col, dropna=False):
        total = len(grp)
        fails = int((grp["abs_error"] > failure_threshold).sum())
        passes = total - fails
        accuracy = round(passes / total * 100, 2) if total else 0.0
        mae = round(grp["abs_error"].mean(), 4) if total else 0.0
        median_err = round(grp["abs_error"].median(), 4) if total else 0.0
        worst = round(grp["abs_error"].max(), 4) if total else 0.0

        results.append({
            group_col: str(name) if name is not None else "UNKNOWN",
            "total": total,
            "pass": passes,
            "fail": fails,
            "accuracy_pct": accuracy,
            "failure_rate_pct": round(100 - accuracy, 2),
            "mae": mae,
            "median_error": median_err,
            "worst_error": worst,
        })

    # Sort by failure rate descending
    results.sort(key=lambda x: x["failure_rate_pct"], reverse=True)
    return results


def _cross_tabulate(
    df, row_col: str, col_col: str, failure_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """Build a pivot of failure rates for row_col × col_col."""
    import pandas as pd

    pivot: Dict[str, Dict[str, Any]] = {}
    for (r, c), grp in df.groupby([row_col, col_col], dropna=False):
        r_str = str(r) if r is not None else "UNKNOWN"
        c_str = str(c) if c is not None else "UNKNOWN"
        total = len(grp)
        fails = int((grp["abs_error"] > failure_threshold).sum())
        rate = round(fails / total * 100, 2) if total else 0.0
        pivot.setdefault(r_str, {})[c_str] = {
            "total": total,
            "fail": fails,
            "failure_rate_pct": rate,
        }
    return pivot


# ── Main Analysis Function ────────────────────────────────────────────

class EvalAnalyzer:
    """Deterministic snap-detection evaluation analyser."""

    def __init__(
        self,
        failure_threshold: float = 0.5,
        top_n_worst: int = 20,
    ):
        self.failure_threshold = failure_threshold
        self.top_n_worst = top_n_worst

    def analyze(self, df) -> dict:
        """Run the full analysis pipeline on a DataFrame.

        Args:
            df: pandas DataFrame with at least the REQUIRED_COLUMNS.

        Returns:
            Dict with sections: overall, per_play_type, per_view,
            per_league, cross_tabs, worst_predictions.
        """
        import pandas as pd

        logger.info("Starting analysis on %d rows.", len(df))

        # ── Step 1: Validate columns ─────────────────────────────────
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Fill optional columns
        for col, default in OPTIONAL_COLUMNS.items():
            if col not in df.columns:
                df[col] = default

        # ── Step 2: Normalise time columns ────────────────────────────
        for col in ("gt_snap_time", "pred_time_sec", "error_sec"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_time_value)

        # Filter to rows with ground truth
        if "has_gt" in df.columns:
            before = len(df)
            df["has_gt"] = df["has_gt"].apply(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
                if isinstance(v, str) else bool(v)
            )
            df = df[df["has_gt"]].copy()
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows without ground truth.", dropped)

        # Drop rows where gt or pred is None
        df = df.dropna(subset=["gt_snap_time", "pred_time_sec"]).copy()
        logger.info("Rows with valid gt + pred: %d", len(df))

        if len(df) == 0:
            return {
                "status": "no_data",
                "message": "No valid rows with ground truth and predictions.",
            }

        # ── Step 3: Recompute absolute error ──────────────────────────
        df["abs_error"] = (df["pred_time_sec"] - df["gt_snap_time"]).abs()

        # Warn if existing error_sec differs
        if "error_sec" in df.columns:
            orig = df["error_sec"].dropna()
            if len(orig) > 0:
                diff = (orig - df.loc[orig.index, "abs_error"]).abs()
                mismatch = (diff > 0.001).sum()
                if mismatch > 0:
                    logger.warning(
                        "%d rows have error_sec != abs(pred - gt). "
                        "Using recomputed abs_error.",
                        mismatch,
                    )

        # ── Step 4: Sort by descending error ──────────────────────────
        df = df.sort_values("abs_error", ascending=False).reset_index(drop=True)

        # ── Step 5: Classify pass / fail ──────────────────────────────
        df["is_failure"] = df["abs_error"] > self.failure_threshold
        df["result"] = df["is_failure"].map({True: "FAIL", False: "PASS"})

        # ── Step 6: Overall summary ───────────────────────────────────
        total = len(df)
        detected_col = df.get("detected")
        if detected_col is not None:
            detected_count = int(
                df["detected"].apply(
                    lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
                    if isinstance(v, str) else bool(v)
                ).sum()
            )
        else:
            detected_count = total

        pass_count = int((~df["is_failure"]).sum())
        fail_count = int(df["is_failure"].sum())
        accuracy = round(pass_count / total * 100, 2) if total else 0.0
        mae = round(df["abs_error"].mean(), 4)
        median_error = round(df["abs_error"].median(), 4)
        p95_error = round(df["abs_error"].quantile(0.95), 4)
        worst_error = round(df["abs_error"].max(), 4)

        overall = {
            "total_rows": total,
            "detected": detected_count,
            "undetected": total - detected_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "accuracy_pct": accuracy,
            "failure_rate_pct": round(100 - accuracy, 2),
            "mean_abs_error": mae,
            "median_abs_error": median_error,
            "p95_error": p95_error,
            "worst_error": worst_error,
            "failure_threshold_sec": self.failure_threshold,
        }
        logger.info(
            "Overall: %d rows, accuracy=%.1f%%, MAE=%.4fs",
            total, accuracy, mae,
        )

        # ── Step 7–9: Per-category breakdowns ─────────────────────────
        per_play_type = _group_breakdown(df, "play_type", self.failure_threshold)
        per_view = _group_breakdown(df, "view", self.failure_threshold)
        per_league = _group_breakdown(df, "league", self.failure_threshold)

        # ── Step 10: Cross-tabulation for pattern extraction ──────────
        cross_play_view = _cross_tabulate(
            df, "play_type", "view", self.failure_threshold,
        )
        cross_play_league = _cross_tabulate(
            df, "play_type", "league", self.failure_threshold,
        )

        # Flag outlier cells (failure rate > overall + 10pp)
        overall_fail_rate = overall["failure_rate_pct"]
        hotspots = []
        for row_key, cols in cross_play_view.items():
            for col_key, cell in cols.items():
                if cell["failure_rate_pct"] > overall_fail_rate + 10:
                    hotspots.append({
                        "play_type": row_key,
                        "view": col_key,
                        "failure_rate_pct": cell["failure_rate_pct"],
                        "excess_pp": round(
                            cell["failure_rate_pct"] - overall_fail_rate, 2
                        ),
                        "sample_size": cell["total"],
                    })
        for row_key, cols in cross_play_league.items():
            for col_key, cell in cols.items():
                if cell["failure_rate_pct"] > overall_fail_rate + 10:
                    hotspots.append({
                        "play_type": row_key,
                        "league": col_key,
                        "failure_rate_pct": cell["failure_rate_pct"],
                        "excess_pp": round(
                            cell["failure_rate_pct"] - overall_fail_rate, 2
                        ),
                        "sample_size": cell["total"],
                    })
        hotspots.sort(key=lambda x: x["failure_rate_pct"], reverse=True)

        # ── Step 11: Worst-N predictions ──────────────────────────────
        worst_cols = [
            "video", "game_id", "play_id", "play_type", "league", "view",
            "gt_snap_time", "pred_time_sec", "abs_error", "result",
        ]
        available_cols = [c for c in worst_cols if c in df.columns]
        worst_df = df.head(self.top_n_worst)[available_cols]
        worst_predictions = []
        for _, row in worst_df.iterrows():
            entry = {}
            for c in available_cols:
                val = row[c]
                if isinstance(val, float):
                    entry[c] = round(val, 4)
                else:
                    entry[c] = str(val) if val is not None else ""
            worst_predictions.append(entry)

        return {
            "status": "completed",
            "overall": overall,
            "per_play_type": per_play_type,
            "per_view": per_view,
            "per_league": per_league,
            "cross_tabs": {
                "play_type_x_view": cross_play_view,
                "play_type_x_league": cross_play_league,
            },
            "hotspots": hotspots,
            "worst_predictions": worst_predictions,
        }

    def format_report(self, results: dict) -> str:
        """Format the analysis results as a human-readable markdown report."""
        if results.get("status") == "no_data":
            return "## No Data\nNo valid rows with ground truth and predictions."

        lines = []
        ov = results["overall"]

        lines.append("# Snap-Moment Detection — Evaluation Report\n")

        # ── Overall ──
        lines.append("## Overall Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Evaluated | {ov['total_rows']} |")
        lines.append(f"| Detected | {ov['detected']} |")
        lines.append(f"| Undetected | {ov['undetected']} |")
        lines.append(f"| PASS (≤{ov['failure_threshold_sec']}s) | {ov['pass_count']} |")
        lines.append(f"| FAIL (>{ov['failure_threshold_sec']}s) | {ov['fail_count']} |")
        lines.append(f"| Accuracy | {ov['accuracy_pct']}% |")
        lines.append(f"| MAE | {ov['mean_abs_error']}s |")
        lines.append(f"| Median Error | {ov['median_abs_error']}s |")
        lines.append(f"| 95th Percentile | {ov['p95_error']}s |")
        lines.append(f"| Worst Error | {ov['worst_error']}s |")
        lines.append("")

        # ── Per-category tables ──
        for section, key in [
            ("Per Play Type", "per_play_type"),
            ("Per Camera View", "per_view"),
            ("Per League", "per_league"),
        ]:
            data = results[key]
            group_col = key.replace("per_", "")
            lines.append(f"## {section}\n")
            lines.append(
                f"| {group_col} | Total | Pass | Fail | Accuracy% | "
                f"Failure% | MAE | Median | Worst |"
            )
            lines.append("|" + "---|" * 9)
            for row in data:
                group_val = row.get(
                    group_col,
                    row.get("play_type", row.get("view", row.get("league", "?")))
                )
                lines.append(
                    f"| {group_val} | {row['total']} | {row['pass']} | "
                    f"{row['fail']} | {row['accuracy_pct']} | "
                    f"{row['failure_rate_pct']} | {row['mae']}s | "
                    f"{row['median_error']}s | {row['worst_error']}s |"
                )
            lines.append("")

        # ── Hotspots ──
        hotspots = results.get("hotspots", [])
        if hotspots:
            lines.append("## Failure Hotspots\n")
            lines.append(
                "Combinations with failure rate >10 percentage points above "
                f"the overall rate ({ov['failure_rate_pct']}%):\n"
            )
            for h in hotspots[:15]:
                ctx = h.get("view") or h.get("league", "?")
                lines.append(
                    f"- **{h['play_type']} × {ctx}**: "
                    f"{h['failure_rate_pct']}% failure "
                    f"(+{h['excess_pp']}pp, n={h['sample_size']})"
                )
            lines.append("")

        # ── Worst predictions ──
        worst = results.get("worst_predictions", [])
        if worst:
            lines.append(f"## Top-{len(worst)} Worst Predictions\n")
            cols = list(worst[0].keys())
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "---|" * len(cols))
            for row in worst:
                vals = [str(row.get(c, "")) for c in cols]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

        return "\n".join(lines)

    def generate_llm_summary(
        self,
        results: dict,
        model_url: str,
        pat: str = "",
        inference_params: Optional[dict] = None,
    ) -> str:
        """Send aggregated stats to an LLM for a narrative summary."""
        import json
        from clarifai.client.model import Model

        if not model_url:
            return ""

        inf = inference_params or {"temperature": 0.3, "max_tokens": 4096}

        # Build a concise context from the results
        context = {
            "overall": results["overall"],
            "per_play_type": results["per_play_type"],
            "per_view": results["per_view"],
            "per_league": results["per_league"],
            "hotspots": results.get("hotspots", [])[:10],
            "worst_5": results.get("worst_predictions", [])[:5],
        }

        system_prompt = (
            "You are an expert sports-AI evaluation analyst. You are given "
            "aggregated evaluation metrics for a snap-moment detection model "
            "trained on American football videos.\n\n"
            "Produce a concise narrative report (300-500 words) covering:\n"
            "1. Overall model performance assessment.\n"
            "2. Which play types, camera views, or leagues are weakest.\n"
            "3. Failure hotspots (specific category combinations).\n"
            "4. Likely root causes for the worst failures.\n"
            "5. Concrete recommendations to improve model accuracy.\n\n"
            "Be data-driven — cite specific numbers from the stats."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"## Evaluation Statistics\n\n"
            f"```json\n{json.dumps(context, indent=2)}\n```"
        )

        logger.info("Calling LLM for narrative summary: %s", model_url)
        model = Model(url=model_url, pat=pat or os.environ.get("CLARIFAI_PAT", ""))
        response = model.predict_by_bytes(
            input_bytes=prompt.encode("utf-8"),
            input_type="text",
            inference_params={
                k: float(v) if k == "temperature" else int(v)
                for k, v in inf.items()
            },
        )

        if response.outputs:
            output = response.outputs[0]
            if hasattr(output, "data") and hasattr(output.data, "text"):
                return output.data.text.raw or ""

        return "(LLM returned empty response)"
