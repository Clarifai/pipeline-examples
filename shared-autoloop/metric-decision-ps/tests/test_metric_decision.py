"""Unit tests for the metric-decision pipeline step (all task types).

Covers: detection, classification, llm_finetune

Run: python -m pytest shared-autoloop/metric-decision-ps/tests/test_metric_decision.py -v
"""

import importlib.util
import json
import os
import sys
import tempfile

import pytest

# Load model.py via importlib to avoid sys.modules collision with hp-adjust
_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models", "model", "1", "model.py"
)
_spec = importlib.util.spec_from_file_location("metric_decision_model", _model_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
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


# ═══════════════════════════════════════════════════════════════
# DETECTION TASK
# ═══════════════════════════════════════════════════════════════


class TestDeployDetection:
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


class TestRetrainDetection:
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


class TestStopDetection:
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


class TestMetricResolutionDetection:
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
        assert _read_output("decision") == "deploy"


class TestHistoryDetection:
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


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION TASK
# ═══════════════════════════════════════════════════════════════


class TestDeployClassification:
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


class TestRetrainClassification:
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


class TestStopClassification:
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


class TestMetricResolutionClassification:
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
        assert _read_output("decision") == "deploy"


class TestHistoryClassification:
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


# ═══════════════════════════════════════════════════════════════
# LLM FINE-TUNING TASK
# ═══════════════════════════════════════════════════════════════


class TestDeployLLM:
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


class TestRetrainLLM:
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


class TestStopLLM:
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


class TestMetricResolutionLLM:
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
        assert _read_output("decision") == "deploy"


class TestHistoryLLM:
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


# ═══════════════════════════════════════════════════════════════
# SHARED: Output files & Parser
# ═══════════════════════════════════════════════════════════════


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

        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "deploy"
        assert full["task_type"] == "detection"
        assert full["direction"] == "maximize"

    def test_classification_output(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"accuracy/top1": 0.90})
        result = decider.decide(eval_results_json=path, task_type="classification", metric_threshold=0.85)
        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "deploy"
        assert full["task_type"] == "classification"
        assert full["direction"] == "maximize"

    def test_llm_output(self, decider, tmp_dir):
        path = _write_eval_json(tmp_dir, {"eval_loss": 1.0})
        result = decider.decide(eval_results_json=path, task_type="llm_finetune", metric_threshold=1.5)
        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "deploy"
        assert full["task_type"] == "llm_finetune"
        assert full["direction"] == "minimize"


# ═══════════════════════════════════════════════════════════════
# MISSING METRIC KEY
# ═══════════════════════════════════════════════════════════════


class TestMissingMetric:
    def test_missing_metric_retrain(self, decider, tmp_dir):
        """When expected metric key is absent, decision is retrain (not at max iterations)."""
        path = _write_eval_json(tmp_dir, {"unrelated_metric": 0.9})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "retrain"
        assert _read_output("metric_value") == "N/A"
        assert _read_output("metric_name") == "AP"

    def test_missing_metric_stop_at_max_iterations(self, decider, tmp_dir):
        """When metric key is absent at max iterations, decision is stop."""
        path = _write_eval_json(tmp_dir, {"unrelated_metric": 0.9})
        decider.decide(
            eval_results_json=path,
            task_type="detection",
            metric_threshold=0.50,
            current_iteration=3,
            max_retrain_iterations=3,
        )
        assert _read_output("decision") == "stop"
        assert _read_output("metric_value") == "N/A"

    def test_missing_metric_output_json(self, decider, tmp_dir):
        """Full output JSON is well-formed when metric is missing."""
        path = _write_eval_json(tmp_dir, {"other": 0.5})
        result = decider.decide(
            eval_results_json=path,
            task_type="classification",
            metric_threshold=0.80,
            current_iteration=1,
            max_retrain_iterations=3,
        )
        with open(result) as f:
            full = json.load(f)
        assert full["decision"] == "retrain"
        assert full["metric_value"] == "N/A"
        assert full["metric_name"] == "accuracy/top1"
        assert full["direction"] == "maximize"


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
