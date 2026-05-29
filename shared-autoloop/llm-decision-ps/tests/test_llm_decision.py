"""Tests for the LLM decision step."""

import importlib
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest

# ── Import the module under test ──
_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1", "models", "model", "1"
)
sys.path.insert(0, _MODEL_DIR)

# Use importlib to avoid sys.modules collisions with other test suites
_spec = importlib.util.spec_from_file_location(
    "llm_decision_model", os.path.join(_MODEL_DIR, "model.py")
)
model_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_mod)

LLMDecision = model_mod.LLMDecision
_validate_llm_response = model_mod._validate_llm_response
_parse_llm_response = model_mod._parse_llm_response
_clamp_hyperparams = model_mod._clamp_hyperparams
_resolve_search_space = model_mod._resolve_search_space

_prompt_spec = importlib.util.spec_from_file_location(
    "llm_decision_prompts", os.path.join(_MODEL_DIR, "prompts.py")
)
prompts_mod = importlib.util.module_from_spec(_prompt_spec)
_prompt_spec.loader.exec_module(prompts_mod)

build_decision_prompt = prompts_mod.build_decision_prompt

_fallback_spec = importlib.util.spec_from_file_location(
    "llm_decision_fallback", os.path.join(_MODEL_DIR, "fallback.py")
)
fallback_mod = importlib.util.module_from_spec(_fallback_spec)
_fallback_spec.loader.exec_module(fallback_mod)

fallback_decide = fallback_mod.fallback_decide


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def eval_results_file():
    """Create a temp eval results JSON file."""
    def _make(metrics):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"metrics": metrics}, f)
        f.close()
        return f.name
    return _make


@pytest.fixture
def detection_metrics():
    return {"AP": 0.32, "AP50": 0.55, "AP75": 0.28}


@pytest.fixture
def classification_metrics():
    return {"accuracy/top1": 0.78, "accuracy/top5": 0.95}


@pytest.fixture
def llm_metrics():
    return {"eval_loss": 1.8, "train_loss": 0.5}


# ═══════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION TESTS
# ═══════════════════════════════════════════════════════════


class TestPromptConstruction:
    def test_includes_task_description(self):
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="Custom food detector for mobile app",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"per_item_lrate": 0.001875},
            search_space={"per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}},
            eval_metrics={"AP": 0.32},
            current_metric_value=0.32,
        )
        assert "Custom food detector for mobile app" in prompt

    def test_includes_metric_info(self):
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=[{"iteration": 1, "hyperparams": {}, "metrics": {"AP": 0.25}, "decision": "retrain", "reason": "below threshold"}],
            current_hyperparams={"per_item_lrate": 0.0009375},
            search_space={},
            eval_metrics={"AP": 0.32, "AP50": 0.55},
            current_metric_value=0.32,
        )
        assert "AP" in prompt
        assert "maximize" in prompt
        assert "0.50" in prompt or "0.5" in prompt
        assert "0.32" in prompt

    def test_includes_iteration_budget(self):
        prompt = build_decision_prompt(
            task_type="classification",
            task_description="",
            primary_metric="accuracy/top1",
            metric_direction="maximize",
            metric_threshold=0.85,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=[],
            current_hyperparams={},
            search_space={},
            eval_metrics={"accuracy/top1": 0.78},
            current_metric_value=0.78,
        )
        assert "2 of 5" in prompt
        assert "3" in prompt  # iterations remaining

    def test_includes_search_space(self):
        space = {
            "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "frozen_stages": {"type": "choice", "values": [0, 1]},
        }
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={},
            search_space=space,
            eval_metrics={"AP": 0.32},
            current_metric_value=0.32,
        )
        assert "per_item_lrate" in prompt
        assert "frozen_stages" in prompt
        assert "log_uniform" in prompt

    def test_includes_full_history(self):
        history = [
            {"iteration": 1, "hyperparams": {"lr": 0.01}, "metrics": {"AP": 0.20}, "decision": "retrain", "reason": "below"},
            {"iteration": 2, "hyperparams": {"lr": 0.005}, "metrics": {"AP": 0.30}, "decision": "retrain", "reason": "below"},
        ]
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=3,
            max_retrain_iterations=5,
            hp_history=history,
            current_hyperparams={"lr": 0.0025},
            search_space={},
            eval_metrics={"AP": 0.38},
            current_metric_value=0.38,
        )
        assert "Iteration 1" in prompt
        assert "Iteration 2" in prompt
        assert "0.20" in prompt or "0.2" in prompt

    def test_empty_history_message(self):
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={},
            search_space={},
            eval_metrics={"AP": 0.32},
            current_metric_value=0.32,
        )
        assert "first training run" in prompt.lower()


# ═══════════════════════════════════════════════════════════
# LLM RESPONSE PARSING TESTS
# ═══════════════════════════════════════════════════════════


class TestResponseParsing:
    def test_valid_json(self):
        text = '{"decision": "deploy", "reasoning": "Metric is good", "confidence": 0.9, "is_overfitting": false, "next_hyperparams": {}}'
        result = _parse_llm_response(text)
        assert result["decision"] == "deploy"

    def test_markdown_wrapped_json(self):
        text = '```json\n{"decision": "stop", "reasoning": "Plateau", "confidence": 0.8, "is_overfitting": false, "next_hyperparams": {}}\n```'
        result = _parse_llm_response(text)
        assert result["decision"] == "stop"

    def test_code_fence_no_language(self):
        text = '```\n{"decision": "retrain", "reasoning": "Can improve", "confidence": 0.7, "is_overfitting": false, "next_hyperparams": {"lr": 0.001}}\n```'
        result = _parse_llm_response(text)
        assert result["decision"] == "retrain"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("This is not JSON at all")

    def test_whitespace_handling(self):
        text = '  \n  {"decision": "deploy", "reasoning": "Good", "confidence": 0.95, "is_overfitting": false, "next_hyperparams": {}}  \n  '
        result = _parse_llm_response(text)
        assert result["decision"] == "deploy"


# ═══════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════


class TestValidation:
    def test_valid_deploy(self):
        resp = {"decision": "deploy", "reasoning": "Threshold met", "confidence": 0.95, "is_overfitting": False, "next_hyperparams": {}}
        errors = _validate_llm_response(resp, {})
        assert errors == []

    def test_valid_retrain(self):
        resp = {"decision": "retrain", "reasoning": "Can improve", "confidence": 0.7, "is_overfitting": False, "next_hyperparams": {"lr": 0.001}}
        errors = _validate_llm_response(resp, {})
        assert errors == []

    def test_invalid_decision_value(self):
        resp = {"decision": "continue", "reasoning": "Hmm", "confidence": 0.5, "is_overfitting": False, "next_hyperparams": {}}
        errors = _validate_llm_response(resp, {})
        assert any("Invalid decision" in e for e in errors)

    def test_missing_reasoning(self):
        resp = {"decision": "deploy", "reasoning": "", "confidence": 0.9, "is_overfitting": False, "next_hyperparams": {}}
        errors = _validate_llm_response(resp, {})
        assert any("reasoning" in e.lower() for e in errors)

    def test_confidence_out_of_range(self):
        resp = {"decision": "deploy", "reasoning": "Good", "confidence": 1.5, "is_overfitting": False, "next_hyperparams": {}}
        errors = _validate_llm_response(resp, {})
        assert any("Confidence" in e for e in errors)

    def test_retrain_without_hyperparams(self):
        resp = {"decision": "retrain", "reasoning": "Needs work", "confidence": 0.6, "is_overfitting": False, "next_hyperparams": {}}
        errors = _validate_llm_response(resp, {})
        assert any("next_hyperparams" in e for e in errors)

    def test_not_a_dict(self):
        errors = _validate_llm_response("not a dict", {})
        assert any("not a JSON object" in e for e in errors)


# ═══════════════════════════════════════════════════════════
# HP CLAMPING TESTS
# ═══════════════════════════════════════════════════════════


class TestHPClamping:
    def test_clamp_to_range(self):
        space = {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}}
        result = _clamp_hyperparams({"lr": 0.05}, space)
        assert result["lr"] == 1e-2

    def test_clamp_below_range(self):
        space = {"lr": {"type": "uniform", "low": 0.001, "high": 0.1}}
        result = _clamp_hyperparams({"lr": 0.0001}, space)
        assert result["lr"] == 0.001

    def test_snap_to_choice(self):
        space = {"lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]}}
        result = _clamp_hyperparams({"lora_r": 20}, space)
        assert result["lora_r"] == 16

    def test_passthrough_unknown_keys(self):
        space = {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}}
        result = _clamp_hyperparams({"lr": 0.001, "batch_size": 32}, space)
        assert result["batch_size"] == 32
        assert result["lr"] == 0.001

    def test_value_in_range_unchanged(self):
        space = {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}}
        result = _clamp_hyperparams({"lr": 0.001}, space)
        assert result["lr"] == 0.001

    def test_discrete_uniform_rounding(self):
        space = {"epochs": {"type": "discrete_uniform", "low": 6, "high": 24, "step": 6}}
        result = _clamp_hyperparams({"epochs": 10}, space)
        assert result["epochs"] == 12


# ═══════════════════════════════════════════════════════════
# FALLBACK TESTS
# ═══════════════════════════════════════════════════════════


class TestFallback:
    def test_deploy_detection(self, eval_results_file):
        metrics = {"AP": 0.65, "AP50": 0.80}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="detection",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"per_item_lrate": 0.001875},
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            seed=42,
        )
        assert result["decision"] == "deploy"
        assert "[FALLBACK]" in result["reasoning"]

    def test_retrain_detection(self, eval_results_file):
        metrics = {"AP": 0.32, "AP50": 0.55}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="detection",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"per_item_lrate": 0.001875, "frozen_stages": 1},
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            seed=42,
        )
        assert result["decision"] == "retrain"
        assert result["next_hyperparams"] != {}

    def test_stop_max_iterations(self, eval_results_file):
        metrics = {"AP": 0.32}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="detection",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=0.50,
            current_iteration=3,
            max_retrain_iterations=3,
            hp_history=[{"iteration": 1, "hyperparams": {}, "metrics": {"AP": 0.25}, "decision": "retrain"}],
            current_hyperparams={"per_item_lrate": 0.001875},
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            seed=42,
        )
        assert result["decision"] == "stop"

    def test_deploy_llm_finetuned(self):
        metrics = {"eval_loss": 1.2, "train_loss": 0.8}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="llm_finetune",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"learning_rate": 2e-4},
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            seed=42,
        )
        assert result["decision"] == "deploy"

    def test_retrain_classification(self):
        metrics = {"accuracy/top1": 0.72}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="classification",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=0.85,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"per_item_lrate": 1.95e-5, "weight_decay": 0.01},
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=False,
            seed=42,
        )
        assert result["decision"] == "retrain"
        assert result["next_hyperparams"] != {}

    def test_overfitting_detected(self):
        metrics = {"eval_loss": 2.0, "train_loss": 0.3}
        result = fallback_decide(
            eval_results={"metrics": metrics},
            task_type="llm_finetune",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 16},
            early_stop_min_delta=0.0,
            overfitting_detection=True,
            tuning_strategy="schedule",
            search_space="auto",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            seed=42,
        )
        assert result["decision"] == "retrain"
        assert result["is_overfitting"] is True


# ═══════════════════════════════════════════════════════════
# END-TO-END TESTS (MOCKED LLM)
# ═══════════════════════════════════════════════════════════


class TestEndToEndLLMPath:
    """Test the full decide() method with a mocked LLM call."""

    def _run_decide(self, eval_metrics, llm_response_dict, **kwargs):
        """Helper: write eval file, mock LLM, run decide()."""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"metrics": eval_metrics}, f)
        f.close()

        defaults = {
            "eval_results_json": f.name,
            "task_type": "detection",
            "task_description": "",
            "primary_metric": "auto",
            "metric_direction": "auto",
            "metric_threshold": 0.50,
            "current_iteration": 1,
            "max_retrain_iterations": 3,
            "hp_history": "[]",
            "current_hyperparams": '{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            "search_space": "auto",
            "tuning_strategy": "schedule",
            "lr_decay_factor": 0.5,
            "unfreeze_on_retry": True,
            "early_stop_min_delta": 0.0,
            "overfitting_detection": False,
            "llm_model_url": "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b",
            "llm_temperature": 0.1,
            "llm_max_retries": 3,
            "seed": 42,
        }
        defaults.update(kwargs)

        with patch.object(model_mod, '_call_llm', return_value=json.dumps(llm_response_dict)):
            result_path = LLMDecision().decide(**defaults)

        # Read outputs
        outputs = {}
        for key in ["decision", "hyperparams_json", "reasoning", "metric_value",
                    "metric_name", "hp_history", "is_overfitting", "strategy_metadata"]:
            path = os.path.join("/tmp", key)
            if os.path.exists(path):
                with open(path) as fout:
                    outputs[key] = fout.read()

        os.unlink(f.name)
        return outputs

    def test_llm_deploy_decision(self):
        llm_resp = {
            "decision": "deploy",
            "reasoning": "AP of 0.55 exceeds the 0.50 threshold. Model is ready.",
            "confidence": 0.95,
            "is_overfitting": False,
            "next_hyperparams": {},
        }
        outputs = self._run_decide({"AP": 0.55, "AP50": 0.80}, llm_resp)
        assert outputs["decision"] == "deploy"
        assert "llm" in outputs["strategy_metadata"]

    def test_llm_retrain_decision(self):
        llm_resp = {
            "decision": "retrain",
            "reasoning": "AP is improving but hasn't reached threshold. Reducing LR should help.",
            "confidence": 0.75,
            "is_overfitting": False,
            "next_hyperparams": {"per_item_lrate": 0.0009375, "frozen_stages": 0},
        }
        outputs = self._run_decide({"AP": 0.38, "AP50": 0.60}, llm_resp)
        assert outputs["decision"] == "retrain"
        hp = json.loads(outputs["hyperparams_json"])
        assert hp["per_item_lrate"] == 0.0009375

    def test_llm_stop_decision(self):
        llm_resp = {
            "decision": "stop",
            "reasoning": "Metrics are oscillating with no clear improvement trend. Stopping.",
            "confidence": 0.8,
            "is_overfitting": False,
            "next_hyperparams": {},
        }
        outputs = self._run_decide({"AP": 0.32}, llm_resp)
        assert outputs["decision"] == "stop"

    def test_llm_hp_clamping(self):
        llm_resp = {
            "decision": "retrain",
            "reasoning": "Trying aggressive LR",
            "confidence": 0.6,
            "is_overfitting": False,
            "next_hyperparams": {"per_item_lrate": 0.05, "frozen_stages": 0},
        }
        outputs = self._run_decide({"AP": 0.32}, llm_resp)
        hp = json.loads(outputs["hyperparams_json"])
        # Should be clamped to 1e-2 (search space max)
        assert hp["per_item_lrate"] == pytest.approx(1e-2)

    def test_llm_failure_triggers_fallback(self):
        """When LLM returns invalid response, fallback should be used."""
        with patch.object(model_mod, '_call_llm', return_value="This is not JSON"):
            f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump({"metrics": {"AP": 0.32}}, f)
            f.close()

            LLMDecision().decide(
                eval_results_json=f.name,
                task_type="detection",
                task_description="",
                primary_metric="auto",
                metric_direction="auto",
                metric_threshold=0.50,
                current_iteration=1,
                max_retrain_iterations=3,
                hp_history="[]",
                current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
                search_space="auto",
                tuning_strategy="schedule",
                lr_decay_factor=0.5,
                unfreeze_on_retry=True,
                early_stop_min_delta=0.0,
                overfitting_detection=False,
                llm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-oss-120b",
                llm_temperature=0.1,
                llm_max_retries=2,
                seed=42,
            )

            with open("/tmp/decision") as df:
                decision = df.read()
            with open("/tmp/strategy_metadata") as mf:
                meta = json.loads(mf.read())

            assert decision == "retrain"
            assert meta["method"] == "fallback"
            os.unlink(f.name)

    def test_no_llm_url_uses_fallback(self):
        """When llm_model_url is empty, fallback runs directly."""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"metrics": {"AP": 0.65}}, f)
        f.close()

        LLMDecision().decide(
            eval_results_json=f.name,
            task_type="detection",
            task_description="",
            primary_metric="auto",
            metric_direction="auto",
            metric_threshold=0.50,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history="[]",
            current_hyperparams='{"per_item_lrate": 0.001875}',
            search_space="auto",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
            early_stop_min_delta=0.0,
            overfitting_detection=False,
            llm_model_url="",
            llm_temperature=0.1,
            llm_max_retries=3,
            seed=42,
        )

        with open("/tmp/decision") as df:
            decision = df.read()
        with open("/tmp/strategy_metadata") as mf:
            meta = json.loads(mf.read())

        assert decision == "deploy"
        assert meta["method"] == "fallback"
        os.unlink(f.name)

    def test_history_accumulation(self):
        """History should be updated with current iteration."""
        prior = [{"iteration": 1, "hyperparams": {"lr": 0.01}, "metrics": {"AP": 0.25}, "decision": "retrain", "reason": "below"}]
        llm_resp = {
            "decision": "retrain",
            "reasoning": "Improving steadily",
            "confidence": 0.7,
            "is_overfitting": False,
            "next_hyperparams": {"per_item_lrate": 0.0005, "frozen_stages": 0},
        }
        outputs = self._run_decide(
            {"AP": 0.38},
            llm_resp,
            current_iteration=2,
            hp_history=json.dumps(prior),
        )
        history = json.loads(outputs["hp_history"])
        assert len(history) == 2
        assert history[1]["iteration"] == 2
        assert history[1]["method"] == "llm"
        assert "Improving steadily" in history[1]["reasoning"]


# ═══════════════════════════════════════════════════════════
# SEARCH SPACE RESOLUTION TESTS
# ═══════════════════════════════════════════════════════════


class TestSearchSpaceResolution:
    def test_auto_detection(self):
        space = _resolve_search_space("auto", "detection")
        assert "per_item_lrate" in space
        assert "frozen_stages" in space

    def test_auto_classification(self):
        space = _resolve_search_space("auto", "classification")
        assert "per_item_lrate" in space
        assert "weight_decay" in space

    def test_auto_llm(self):
        space = _resolve_search_space("auto", "llm_finetune")
        assert "learning_rate" in space
        assert "lora_r" in space

    def test_custom_json(self):
        custom = '{"my_param": {"type": "uniform", "low": 0, "high": 1}}'
        space = _resolve_search_space(custom, "detection")
        assert "my_param" in space
        assert space["my_param"]["type"] == "uniform"


# ═══════════════════════════════════════════════════════════
# RETRY LOGIC TESTS
# ═══════════════════════════════════════════════════════════


class TestRetryLogic:
    def test_succeeds_on_second_attempt(self):
        """LLM fails once then succeeds."""
        invalid = "not json"
        valid = json.dumps({
            "decision": "deploy",
            "reasoning": "Threshold met on retry",
            "confidence": 0.9,
            "is_overfitting": False,
            "next_hyperparams": {},
        })

        call_count = {"n": 0}
        def mock_call(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return invalid
            return valid

        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"metrics": {"AP": 0.55}}, f)
        f.close()

        with patch.object(model_mod, '_call_llm', side_effect=mock_call):
            LLMDecision().decide(
                eval_results_json=f.name,
                task_type="detection",
                task_description="",
                primary_metric="auto",
                metric_direction="auto",
                metric_threshold=0.50,
                current_iteration=1,
                max_retrain_iterations=3,
                hp_history="[]",
                current_hyperparams='{}',
                search_space="auto",
                tuning_strategy="schedule",
                lr_decay_factor=0.5,
                unfreeze_on_retry=True,
                early_stop_min_delta=0.0,
                overfitting_detection=False,
                llm_model_url="https://clarifai.com/openai/chat-completion/models/gpt-oss-120b",
                llm_temperature=0.1,
                llm_max_retries=3,
                seed=42,
            )

        with open("/tmp/decision") as df:
            assert df.read() == "deploy"
        with open("/tmp/strategy_metadata") as mf:
            meta = json.loads(mf.read())
            assert meta["method"] == "llm"
            assert meta["attempts"] == 2
        os.unlink(f.name)
