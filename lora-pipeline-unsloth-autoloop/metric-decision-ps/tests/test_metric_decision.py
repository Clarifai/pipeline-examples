"""Unit tests for the metric-decision pipeline step (LLM fine-tuning task).

Run: python -m pytest lora-pipeline-unsloth-autoloop/metric-decision-ps/tests/test_metric_decision.py -v
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
    def test_eval_loss_below_threshold(self, decider, tmp_dir):
        """eval_loss 1.2 <= threshold 1.5 with minimize → deploy."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.2})
        decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
        assert _read_output("decision") == "deploy"
        assert _read_output("metric_name") == "eval_loss"

    def test_exact_threshold_minimize(self, decider, tmp_dir):
        """Boundary: eval_loss == threshold with minimize → deploy (<=)."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.5})
        decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
        assert _read_output("decision") == "deploy"

    def test_very_low_loss(self, decider, tmp_dir):
        """Very low loss well under threshold → deploy."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 0.3, "train_loss": 0.25})
        decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
        assert _read_output("decision") == "deploy"


# ─── Retrain decisions ─────────────────────────────────────────


class TestRetrain:
    def test_above_threshold_minimize(self, decider, tmp_dir):
        """eval_loss 2.5 > threshold 1.5 with minimize → retrain."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.5})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("next_iteration") == "2"

    def test_retrain_with_overfitting(self, decider, tmp_dir):
        """train_loss << eval_loss with overfitting_detection=True → retrain + is_overfitting."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.5, "train_loss": 0.5})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=True,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "true"

    def test_retrain_no_overfitting_flag(self, decider, tmp_dir):
        """Overfitting detection disabled → is_overfitting stays false even with diverging losses."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.5, "train_loss": 0.5})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=False,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "false"

    def test_no_overfitting_when_losses_close(self, decider, tmp_dir):
        """train_loss NOT << eval_loss → no overfitting even with detection on."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.5, "train_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
            overfitting_detection=True,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("is_overfitting") == "false"


# ─── Stop decisions ────────────────────────────────────────────


class TestStop:
    def test_max_iterations_reached(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=3,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "stop"

    def test_early_stop_plateau(self, decider, tmp_dir):
        """Loss barely improved (2.0 → 1.95, delta=0.05) but min_delta=0.1 → stop."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.95})
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"eval_loss": 2.0}, "decision": "retrain", "reason": ""}
        ])
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=history,
            early_stop_min_delta=0.1,
        )
        assert _read_output("decision") == "stop"

    def test_no_early_stop_when_improving(self, decider, tmp_dir):
        """Loss improved enough (2.0 → 1.7, delta=0.3 > min_delta=0.1) → retrain, not stop."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.7})
        history = json.dumps([
            {"iteration": 1, "hyperparams": {}, "metrics": {"eval_loss": 2.0}, "decision": "retrain", "reason": ""}
        ])
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=history,
            early_stop_min_delta=0.1,
        )
        assert _read_output("decision") == "retrain"


# ─── Metric resolution ────────────────────────────────────────


class TestMetricResolution:
    def test_auto_resolves_eval_loss(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.0})
        decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=2.0)
        assert _read_output("metric_name") == "eval_loss"
        assert _read_output("decision") == "deploy"

    def test_auto_direction_is_minimize(self, decider, tmp_dir):
        """eval_loss 2.0 > threshold 1.5 → retrain (because direction=minimize, higher is worse)."""
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.0})
        decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
        assert _read_output("decision") == "retrain"

    def test_explicit_perplexity_metric(self, decider, tmp_dir):
        """Override to use perplexity — 'perplexity' pattern auto-resolves to minimize."""
        path = _write_eval_json(tmp_dir, {"perplexity": 5.0, "eval_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            primary_metric="perplexity",
            metric_threshold=8.0,
        )
        assert _read_output("metric_name") == "perplexity"
        assert _read_output("decision") == "deploy"  # 5.0 <= 8.0


# ─── History accumulation ─────────────────────────────────────


class TestHistory:
    def test_accumulates_entries(self, decider, tmp_dir):
        history = json.dumps([
            {"iteration": 1, "hyperparams": {"learning_rate": 0.0002}, "metrics": {"eval_loss": 2.5}, "decision": "retrain", "reason": ""},
            {"iteration": 2, "hyperparams": {"learning_rate": 0.0001}, "metrics": {"eval_loss": 2.0}, "decision": "retrain", "reason": ""},
        ])
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.8})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            current_iteration=3,
            max_retrain_iterations=5,
            hp_history=history,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 3
        assert updated[-1]["iteration"] == 3
        assert updated[-1]["decision"] == "retrain"

    def test_empty_history(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"eval_loss": 2.0})
        decider.decide(
            eval_results_json=path,
            task_type="llm_finetune",
            metric_threshold=1.5,
            early_stop_min_delta=0.1,
        )
        updated = json.loads(_read_output("hp_history"))
        assert len(updated) == 1
        assert _read_output("decision") == "retrain"


# ─── Output files ─────────────────────────────────────────────


class TestOutputFiles:
    def test_all_output_files_written(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.0})
        result = decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
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
        assert full["task_type"] == "llm_finetune"
        assert full["direction"] == "minimize"


# ─── Argparse parser ──────────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--task_type", "llm_finetune",
            "--metric_threshold", "1.5",
        ])
        assert args.eval_results_json == "/tmp/test.json"
        assert args.task_type == "llm_finetune"
        assert args.metric_threshold == 1.5
        assert args.primary_metric == "auto"

    def test_bool_parsing(self):
        parser = MetricDecision.to_pipeline_parser()
        args = parser.parse_args([
            "--eval_results_json", "/tmp/test.json",
            "--overfitting_detection", "true",
        ])
        assert args.overfitting_detection is True
