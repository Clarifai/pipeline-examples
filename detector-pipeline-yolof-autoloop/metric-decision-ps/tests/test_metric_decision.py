"""Unit tests for the metric-decision pipeline step.

Run: python -m pytest detector-pipeline-yolof-autoloop/metric-decision-ps/tests/test_metric_decision.py -v
"""

import json
import os
import sys
import tempfile

import pytest

# Add model.py to path using the same __import__ trick as pipeline_step.py
# (directory "1" is not a valid Python package name)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models"
    ),
)
_mod = __import__("model.1.model", fromlist=[""])
MetricDecision = _mod.MetricDecision


@pytest.fixture
def decider():
    return MetricDecision()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_eval_json(tmp_dir, metrics):
    """Write a mock eval_results.json and return its path."""
    path = os.path.join(tmp_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump({"metrics": metrics}, f)
    return path


def _read_output(key):
    """Read an Argo output param written to /tmp/."""
    with open(f"/tmp/{key}") as f:
        return f.read()


# ─── Deploy decisions ──────────────────────────────────────────


class TestDeploy:
    def test_detection_above_threshold(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.65, "AP50": 0.80})
        decider.decide(eval_results_json=path, task_type="detection", metric_threshold=0.50)
        assert _read_output("decision") == "deploy"
        assert _read_output("metric_name") == "AP"

    def test_exact_threshold_maximize(self, decider, tmp_dir):
        """Boundary: value == threshold with maximize → deploy (>=)."""
        path = _write_eval_json(tmp_dir, {"AP": 0.50})
        decider.decide(eval_results_json=path, task_type="detection", metric_threshold=0.50)
        assert _read_output("decision") == "deploy"


# ─── Retrain decisions ─────────────────────────────────────────


class TestRetrain:
    def test_below_threshold(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.30})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("next_iteration") == "2"

    def test_retrain_with_overfitting(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.30, "train_loss": 0.5, "eval_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=True,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "true"

    def test_retrain_no_overfitting_flag(self, decider, tmp_dir):
        """Overfitting detection disabled → is_overfitting stays false."""
        path = _write_eval_json(tmp_dir, {"AP": 0.30, "train_loss": 0.5, "eval_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=False,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "false"


# ─── Stop decisions ────────────────────────────────────────────


class TestStop:
    def test_max_iterations_reached(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.30})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=3,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "stop"

    def test_early_stop_plateau(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.31})
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"AP": 0.30}, "decision": "retrain", "reason": ""}
        ])
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=history,
            early_stop_min_delta=0.05,
        )
        assert _read_output("decision") == "stop"


# ─── Metric resolution ─────────────────────────────────────────


class TestMetricResolution:
    def test_auto_detection(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.65})
        decider.decide(eval_results_json=path, task_type="detection")
        assert _read_output("metric_name") == "AP"

    def test_explicit_override(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP50": 0.80, "AP": 0.40})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            primary_metric="AP50",
            metric_direction="maximize",
            metric_threshold=0.70,
        )
        assert _read_output("metric_name") == "AP50"
        assert _read_output("decision") == "deploy"

    def test_loss_pattern_auto_minimize(self, decider, tmp_dir):
        """primary_metric containing 'loss' → auto direction = minimize."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 0.8})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            primary_metric="eval_loss",
            metric_threshold=1.0,
        )
        # 0.8 <= 1.0 with minimize → deploy
        assert _read_output("decision") == "deploy"


# ─── History accumulation ──────────────────────────────────────


class TestHistory:
    def test_accumulates_entries(self, decider, tmp_dir):
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"AP": 0.20}, "decision": "retrain", "reason": ""},
            {"iteration": 2, "hyperparams": {}, "metrics": {"AP": 0.30}, "decision": "retrain", "reason": ""},
        ])
        path = _write_eval_json(tmp_dir, {"AP": 0.35})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=3,
            max_retrain_iterations=5,
            hp_history=history,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 3
        assert updated[-1]["iteration"] == 3
        assert updated[-1]["decision"] == "retrain"

    def test_empty_history(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.30})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            early_stop_min_delta=0.05,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 1
        assert _read_output("decision") == "retrain"


# ─── Test: Output files written correctly ──────────────────────


class TestOutputFiles:
    def test_all_output_files_written(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"AP": 0.65})
        result = decider.decide(eval_results_json=path, task_type="detection", metric_threshold=0.50)
        expected_files = [
            "decision", "metric_value", "metric_name",
            "hp_history", "is_overfitting",
            "current_iteration", "next_iteration",
            "decision_output.json",
        ]
        for fname in expected_files:
            assert os.path.exists(f"/tmp/{fname}"), f"Missing output: /tmp/{fname}"

        # Verify decision_output.json is valid JSON
        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "deploy"
        assert full["task_type"] == "detection"
        assert full["direction"] == "maximize"


# ─── Test: Argparse parser ─────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--task_type", "detection",
            "--metric_threshold", "0.5",
        ])
        assert args.eval_results_json == "/tmp/test.json"
        assert args.task_type == "detection"
        assert args.metric_threshold == 0.5
        assert args.primary_metric == "auto"

    def test_bool_parsing(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--overfitting_detection", "true",
        ])
        assert args.overfitting_detection is True

        args2 = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--overfitting_detection", "false",
        ])
        assert args2.overfitting_detection is False
