"""Integration tests for LLM decision step — calls a real Clarifai LLM.

Run with:
    pytest shared-autoloop/llm-decision-ps/tests/test_llm_integration.py -v -s

Requires:
    - CLARIFAI_PAT environment variable set
    - Network access to clarifai.com
"""

import importlib.util
import json
import os
import sys
import pytest

# ── Import model code via importlib ──
_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "1", "models", "model", "1")
)

_model_spec = importlib.util.spec_from_file_location(
    "llm_decision_model_integ", os.path.join(_MODEL_DIR, "model.py")
)
model_mod = importlib.util.module_from_spec(_model_spec)
sys.modules["llm_decision_model_integ"] = model_mod
_model_spec.loader.exec_module(model_mod)

_call_llm = model_mod._call_llm
_parse_llm_response = model_mod._parse_llm_response
_validate_llm_response = model_mod._validate_llm_response

_prompt_spec = importlib.util.spec_from_file_location(
    "llm_decision_prompts_integ", os.path.join(_MODEL_DIR, "prompts.py")
)
prompts_mod = importlib.util.module_from_spec(_prompt_spec)
_prompt_spec.loader.exec_module(prompts_mod)

SYSTEM_PROMPT = prompts_mod.SYSTEM_PROMPT
build_decision_prompt = prompts_mod.build_decision_prompt

# ── Config ──
LLM_MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"
LLM_TEMPERATURE = 0.1

# Skip unless explicitly opted in
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_LLM_TESTS"),
    reason="Set RUN_LIVE_LLM_TESTS=1 (and CLARIFAI_PAT) to run live LLM integration tests",
)


def _call_and_parse(user_prompt):
    """Call the LLM and parse + validate the response."""
    raw = _call_llm(
        model_url=LLM_MODEL_URL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=LLM_TEMPERATURE,
    )
    parsed = _parse_llm_response(raw)
    errors = _validate_llm_response(parsed, search_space={})
    return raw, parsed, errors


# ═══════════════════════════════════════════════════════════
# DEPLOY: metric clearly above threshold
# ═══════════════════════════════════════════════════════════


class TestLLMDeploy:

    def test_detection_above_threshold(self):
        """AP=0.72 vs threshold=0.50 -> should deploy."""
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="Object detection model for traffic signs",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=[
                {"iteration": 1, "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
                 "metrics": {"AP": 0.45, "AP50": 0.68}, "decision": "retrain", "reason": "below threshold"},
            ],
            current_hyperparams={"per_item_lrate": 0.0009375, "frozen_stages": 0},
            search_space={
                "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
                "frozen_stages": {"type": "choice", "values": [0, 1]},
            },
            eval_metrics={"AP": 0.72, "AP50": 0.88, "AP75": 0.65},
            current_metric_value=0.72,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[deploy] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "deploy", f"Expected deploy, got: {parsed['decision']}"
        assert parsed["confidence"] >= 0.7, f"Low confidence for clear deploy: {parsed['confidence']}"
        assert len(parsed["reasoning"]) > 10

    def test_classification_meets_threshold(self):
        """accuracy/top1=0.85 vs threshold=0.85 -> should deploy."""
        prompt = build_decision_prompt(
            task_type="classification",
            task_description="Image classifier for product categories",
            primary_metric="accuracy/top1",
            metric_direction="maximize",
            metric_threshold=0.85,
            current_iteration=3,
            max_retrain_iterations=5,
            hp_history=[
                {"iteration": 1, "hyperparams": {"per_item_lrate": 0.001}, "metrics": {"accuracy/top1": 0.72},
                 "decision": "retrain", "reason": "below threshold"},
                {"iteration": 2, "hyperparams": {"per_item_lrate": 0.0005}, "metrics": {"accuracy/top1": 0.80},
                 "decision": "retrain", "reason": "below threshold"},
            ],
            current_hyperparams={"per_item_lrate": 0.00025, "weight_decay": 0.01},
            search_space={
                "per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
                "weight_decay": {"type": "log_uniform", "low": 0.001, "high": 0.1},
            },
            eval_metrics={"accuracy/top1": 0.85, "accuracy/top5": 0.97},
            current_metric_value=0.85,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[deploy exact] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "deploy"


# ═══════════════════════════════════════════════════════════
# RETRAIN: below threshold with budget remaining
# ═══════════════════════════════════════════════════════════


class TestLLMRetrain:

    def test_detection_improving_below_threshold(self):
        """AP=0.38 vs threshold=0.50, iteration 2/5 -> should retrain with valid HPs."""
        search_space = {
            "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "frozen_stages": {"type": "choice", "values": [0, 1]},
        }
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="Detect defects on assembly line",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.50,
            current_iteration=2,
            max_retrain_iterations=5,
            hp_history=[
                {"iteration": 1, "hyperparams": {"per_item_lrate": 0.001875, "frozen_stages": 1},
                 "metrics": {"AP": 0.28, "AP50": 0.45}, "decision": "retrain", "reason": "below threshold"},
            ],
            current_hyperparams={"per_item_lrate": 0.001875, "frozen_stages": 1},
            search_space=search_space,
            eval_metrics={"AP": 0.38, "AP50": 0.58, "AP75": 0.30},
            current_metric_value=0.38,
        )

        current_hps = {"per_item_lrate": 0.001875, "frozen_stages": 1}
        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[retrain detection] Raw:\n{raw}")
        errors = _validate_llm_response(parsed, search_space)
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "retrain", f"Expected retrain, got: {parsed['decision']}"
        hps = parsed["next_hyperparams"]
        assert len(hps) > 0, "retrain must suggest new hyperparameters"
        # Bounds check
        if "per_item_lrate" in hps:
            assert 1e-5 <= hps["per_item_lrate"] <= 1e-2, f"LR out of range: {hps['per_item_lrate']}"
        if "frozen_stages" in hps:
            assert hps["frozen_stages"] in [0, 1], f"frozen_stages invalid: {hps['frozen_stages']}"
        # HPs must differ from current (no point retraining with same params)
        assert hps != current_hps, f"LLM suggested identical HPs to current: {hps}"
        # Should not repeat already-tried HP combos from history
        tried_hps = [{"per_item_lrate": 0.001875, "frozen_stages": 1}]
        assert hps not in tried_hps, f"LLM suggested previously tried HPs: {hps}"

    def test_llm_finetune_high_eval_loss(self):
        """eval_loss=2.1 vs threshold=1.5, iteration 1/4 -> should retrain."""
        search_space = {
            "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
            "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
            "lora_alpha": {"type": "choice", "values": [8, 16, 32, 64, 128]},
        }
        prompt = build_decision_prompt(
            task_type="llm_finetune",
            task_description="Fine-tune LLM for customer support summarization",
            primary_metric="eval_loss",
            metric_direction="minimize",
            metric_threshold=1.5,
            current_iteration=1,
            max_retrain_iterations=4,
            hp_history=[],
            current_hyperparams={"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 32},
            search_space=search_space,
            eval_metrics={"eval_loss": 2.1, "train_loss": 1.2},
            current_metric_value=2.1,
        )

        current_hps = {"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 32}
        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[retrain LLM] Raw:\n{raw}")
        errors = _validate_llm_response(parsed, search_space)
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "retrain"
        hps = parsed["next_hyperparams"]
        assert len(hps) > 0
        # Bounds check
        if "learning_rate" in hps:
            assert 1e-5 <= hps["learning_rate"] <= 5e-4
        if "lora_r" in hps:
            assert hps["lora_r"] in [8, 16, 32, 64, 128]
        if "lora_alpha" in hps:
            assert hps["lora_alpha"] in [8, 16, 32, 64, 128]
        # HPs must differ from current
        assert hps != current_hps, f"LLM suggested identical HPs to current: {hps}"


# ═══════════════════════════════════════════════════════════
# STOP: plateaued / budget exhausted
# ═══════════════════════════════════════════════════════════


class TestLLMStop:

    def test_plateau_at_max_iterations(self):
        """AP stuck ~0.35 for 5 iters, now at 5/5 -> should stop."""
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="Detect rare wildlife in camera traps",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.60,
            current_iteration=5,
            max_retrain_iterations=5,
            hp_history=[
                {"iteration": 1, "hyperparams": {"per_item_lrate": 0.001875}, "metrics": {"AP": 0.30},
                 "decision": "retrain", "reason": "below threshold"},
                {"iteration": 2, "hyperparams": {"per_item_lrate": 0.0009375}, "metrics": {"AP": 0.33},
                 "decision": "retrain", "reason": "below threshold"},
                {"iteration": 3, "hyperparams": {"per_item_lrate": 0.0004687}, "metrics": {"AP": 0.34},
                 "decision": "retrain", "reason": "below threshold"},
                {"iteration": 4, "hyperparams": {"per_item_lrate": 0.0002343}, "metrics": {"AP": 0.35},
                 "decision": "retrain", "reason": "below threshold"},
            ],
            current_hyperparams={"per_item_lrate": 0.0001171},
            search_space={"per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}},
            eval_metrics={"AP": 0.35, "AP50": 0.55},
            current_metric_value=0.35,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[stop plateau] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "stop", f"Expected stop, got: {parsed['decision']}"

    def test_minimize_no_improvement(self):
        """eval_loss stuck at 2.0+ for 3 iterations, at 3/3 -> should stop."""
        prompt = build_decision_prompt(
            task_type="llm_finetune",
            task_description="Fine-tune for code generation",
            primary_metric="eval_loss",
            metric_direction="minimize",
            metric_threshold=1.0,
            current_iteration=3,
            max_retrain_iterations=3,
            hp_history=[
                {"iteration": 1, "hyperparams": {"learning_rate": 2e-4, "lora_r": 16},
                 "metrics": {"eval_loss": 2.5}, "decision": "retrain", "reason": "above threshold"},
                {"iteration": 2, "hyperparams": {"learning_rate": 1e-4, "lora_r": 32},
                 "metrics": {"eval_loss": 2.3}, "decision": "retrain", "reason": "above threshold"},
            ],
            current_hyperparams={"learning_rate": 5e-5, "lora_r": 64},
            search_space={
                "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
                "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
            },
            eval_metrics={"eval_loss": 2.1, "train_loss": 0.8},
            current_metric_value=2.1,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[stop budget] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert parsed["decision"] == "stop"


# ═══════════════════════════════════════════════════════════
# FORMAT: verify JSON schema compliance
# ═══════════════════════════════════════════════════════════


class TestLLMResponseFormat:

    def test_has_all_required_fields(self):
        """Response must include decision, reasoning, confidence, is_overfitting, next_hyperparams."""
        prompt = build_decision_prompt(
            task_type="classification",
            task_description="Classify satellite imagery",
            primary_metric="accuracy/top1",
            metric_direction="maximize",
            metric_threshold=0.90,
            current_iteration=1,
            max_retrain_iterations=3,
            hp_history=[],
            current_hyperparams={"per_item_lrate": 0.001},
            search_space={"per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3}},
            eval_metrics={"accuracy/top1": 0.65, "accuracy/top5": 0.88},
            current_metric_value=0.65,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[format] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert "decision" in parsed
        assert "reasoning" in parsed
        assert "confidence" in parsed
        assert "is_overfitting" in parsed
        assert "next_hyperparams" in parsed
        assert isinstance(parsed["reasoning"], str)
        assert isinstance(parsed["confidence"], (int, float))
        assert isinstance(parsed["is_overfitting"], bool)
        assert isinstance(parsed["next_hyperparams"], dict)

    def test_json_parseable(self):
        """LLM output is valid JSON (handles code fences)."""
        prompt = build_decision_prompt(
            task_type="detection",
            task_description="Detect PPE compliance",
            primary_metric="AP",
            metric_direction="maximize",
            metric_threshold=0.70,
            current_iteration=2,
            max_retrain_iterations=4,
            hp_history=[
                {"iteration": 1, "hyperparams": {"per_item_lrate": 0.005}, "metrics": {"AP": 0.55},
                 "decision": "retrain", "reason": "below threshold"},
            ],
            current_hyperparams={"per_item_lrate": 0.0025},
            search_space={"per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}},
            eval_metrics={"AP": 0.62, "AP50": 0.80},
            current_metric_value=0.62,
        )

        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[json] Raw:\n{raw}")
        assert isinstance(parsed, dict)
        assert parsed["decision"] in ("deploy", "retrain", "stop")


# ═══════════════════════════════════════════════════════════
# OVERFITTING: train_loss << eval_loss
# ═══════════════════════════════════════════════════════════


class TestLLMOverfitting:

    def test_detects_overfitting(self):
        """train_loss=0.3, eval_loss=2.5 -> should flag is_overfitting=True."""
        prompt = build_decision_prompt(
            task_type="llm_finetune",
            task_description="Fine-tune for entity extraction",
            primary_metric="eval_loss",
            metric_direction="minimize",
            metric_threshold=1.0,
            current_iteration=2,
            max_retrain_iterations=4,
            hp_history=[
                {"iteration": 1, "hyperparams": {"learning_rate": 3e-4, "lora_r": 64},
                 "metrics": {"eval_loss": 2.0, "train_loss": 0.8}, "decision": "retrain",
                 "reason": "above threshold"},
            ],
            current_hyperparams={"learning_rate": 3e-4, "lora_r": 64},
            search_space={
                "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
                "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
            },
            eval_metrics={"eval_loss": 2.5, "train_loss": 0.3},
            current_metric_value=2.5,
        )

        current_hps = {"learning_rate": 3e-4, "lora_r": 64}
        raw, parsed, errors = _call_and_parse(prompt)
        print(f"\n[overfitting] Raw:\n{raw}")
        assert not errors, f"Validation errors: {errors}"
        assert parsed["is_overfitting"] is True, f"Expected overfitting=True, got: {parsed['is_overfitting']}"
        reasoning_lower = parsed["reasoning"].lower()
        assert "overfit" in reasoning_lower or "diverge" in reasoning_lower or "generali" in reasoning_lower
        # If retrain, LLM should suggest regularization (lower LR or smaller lora_r)
        if parsed["decision"] == "retrain":
            hps = parsed["next_hyperparams"]
            assert hps != current_hps, f"LLM suggested identical HPs despite overfitting: {hps}"
            lr_reduced = hps.get("learning_rate", 3e-4) < 3e-4
            rank_reduced = hps.get("lora_r", 64) < 64
            assert lr_reduced or rank_reduced, (
                f"Expected LLM to reduce LR or lora_r for overfitting, got: {hps}"
            )
