"""Unit tests for the metric-decision pipeline step (classification task).

Run: python -m pytest classifier-pipeline-resnet-autoloop/metric-decision-ps/tests/test_metric_decision.py -v
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
    def test_accuracy_above_threshold(self, decider, tmp_dir):
        """accuracy/top1 0.90 >= threshold 0.85 → deploy."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.90, "accuracy/top5": 0.98})
        decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        assert _read_output("decision") == "deploy"
        assert _read_output("metric_name") == "accuracy/top1"

    def test_exact_threshold(self, decider, tmp_dir):
        """Boundary: accuracy == threshold → deploy (>=)."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.85})
        decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        assert _read_output("decision") == "deploy"

    def test_high_accuracy(self, decider, tmp_dir):
        """Very high accuracy well above threshold → deploy."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.97})
        decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        assert _read_output("decision") == "deploy"


# ─── Retrain decisions ─────────────────────────────────────────


class TestRetrain:
    def test_below_threshold(self, decider, tmp_dir):
        """accuracy 0.60 < threshold 0.85 → retrain."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=1,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("next_iteration") == "2"

    def test_retrain_with_overfitting(self, decider, tmp_dir):
        """train_loss << eval_loss with overfitting_detection=True → retrain + is_overfitting."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60, "train_loss": 0.1, "eval_loss": 0.8})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=True,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "true"

    def test_retrain_no_overfitting_flag(self, decider, tmp_dir):
        """Overfitting detection disabled → is_overfitting stays false."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60, "train_loss": 0.1, "eval_loss": 0.8})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=False,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "false"

    def test_no_overfitting_when_losses_close(self, decider, tmp_dir):
        """train_loss NOT << eval_loss → no overfitting even with detection on."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60, "train_loss": 0.5, "eval_loss": 0.6})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=True,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "false"


# ─── Stop decisions ────────────────────────────────────────────


class TestStop:
    def test_max_iterations_reached(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.70})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=3,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "stop"

    def test_early_stop_plateau(self, decider, tmp_dir):
        """Accuracy barely improved (0.70 → 0.71, delta=0.01) but min_delta=0.03 → stop."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.71})
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"accuracy/top1": 0.70}, "decision": "retrain", "reason": ""}
        ])
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=history,
            early_stop_min_delta=0.03,
        )
        assert _read_output("decision") == "stop"

    def test_no_early_stop_when_improving(self, decider, tmp_dir):
        """Accuracy improved enough (0.70 → 0.78, delta=0.08 > min_delta=0.03) → retrain."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.78})
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"accuracy/top1": 0.70}, "decision": "retrain", "reason": ""}
        ])
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=history,
            early_stop_min_delta=0.03,
        )
        assert _read_output("decision") == "retrain"


# ─── Metric resolution ────────────────────────────────────────


class TestMetricResolution:
    def test_auto_resolves_accuracy_top1(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.90})
        decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        assert _read_output("metric_name") == "accuracy/top1"

    def test_auto_direction_is_maximize(self, decider, tmp_dir):
        """accuracy 0.60 < threshold 0.85 → retrain (because direction=maximize, lower is worse)."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60})
        decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        assert _read_output("decision") == "retrain"

    def test_explicit_top5_metric(self, decider, tmp_dir):
        """Override to use accuracy/top5."""
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.70, "accuracy/top5": 0.95})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            primary_metric="accuracy/top5",
            metric_direction="maximize",
            metric_threshold=0.90,
        )
        assert _read_output("metric_name") == "accuracy/top5"
        assert _read_output("decision") == "deploy"  # 0.95 >= 0.90


# ─── History accumulation ─────────────────────────────────────


class TestHistory:
    def test_accumulates_entries(self, decider, tmp_dir):
        history = json.dumps([
            {"iteration": 1, "hyperparams": {"per_item_lrate": 0.00002}, "metrics": {"accuracy/top1": 0.50}, "decision": "retrain", "reason": ""},
            {"iteration": 2, "hyperparams": {"per_item_lrate": 0.00001}, "metrics": {"accuracy/top1": 0.65}, "decision": "retrain", "reason": ""},
        ])
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.75})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            current_iteration=3,
            max_retrain_iterations=5,
            hp_history=history,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 3
        assert updated[-1]["iteration"] == 3
        assert updated[-1]["decision"] == "retrain"

    def test_empty_history(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.60})
        decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.85,
            early_stop_min_delta=0.03,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 1
        assert _read_output("decision") == "retrain"


# ─── Output files ─────────────────────────────────────────────


class TestOutputFiles:
    def test_all_output_files_written(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.90})
        result = decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        expected_files = [
            "decision", "metric_value", "metric_name",
            "hp_history", "is_overfitting",
            "current_iteration", "next_iteration",
            "decision_output.json",
        ]
        for fname in expected_files:
            assert os.path.exists(f"/tmp/{fname}"), f"Missing output: /tmp/{fname}"

        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "deploy"
        assert full["task_type"] == "classification"
        assert full["direction"] == "maximize"


# ─── Argparse parser ──────────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--task_type", "classification",
            "--metric_threshold", "0.85",
        ])
        assert args.eval_results_json == "/tmp/test.json"
        assert args.task_type == "classification"
        assert args.metric_threshold == 0.85
        assert args.primary_metric == "auto"

    def test_bool_parsing(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--overfitting_detection", "true",
        ])
        assert args.overfitting_detection is True
