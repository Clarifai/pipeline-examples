import json
import inspect
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import SYSTEM_PROMPT, build_decision_prompt
from fallback import fallback_decide, _resolve_metric, METRIC_DEFAULTS

logging.basicConfig(level=logging.INFO)

# Search space defaults (same as hp-adjust-ps strategies.py)
SEARCH_SPACE_DEFAULTS = {
    "detection": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "frozen_stages": {"type": "choice", "values": [0, 1]},
    },
    "classification": {
        "per_item_lrate": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
        "weight_decay": {"type": "log_uniform", "low": 0.001, "high": 0.1},
    },
    "llm_finetune": {
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
        "lora_r": {"type": "choice", "values": [8, 16, 32, 64, 128]},
        "lora_alpha": {"type": "choice", "values": [8, 16, 32, 64, 128]},
    },
}


def _resolve_search_space(search_space_str, task_type):
    """Resolve search space from string input."""
    if search_space_str == "auto":
        return SEARCH_SPACE_DEFAULTS.get(task_type, {})
    return json.loads(search_space_str)


def _clamp_hyperparams(next_hps, search_space):
    """Clamp HP values to search space bounds."""
    clamped = next_hps.copy()
    for key, value in list(clamped.items()):
        if key not in search_space:
            continue
        spec = search_space[key]
        if isinstance(spec, list):
            # Grid list: snap to nearest
            if value not in spec:
                clamped[key] = min(spec, key=lambda x: abs(x - value))
        elif isinstance(spec, dict):
            dist_type = spec.get("type", "")
            if dist_type in ("log_uniform", "uniform", "int_log_uniform"):
                low, high = spec["low"], spec["high"]
                clamped[key] = max(low, min(high, value))
                if dist_type in ("int_log_uniform", "discrete_uniform"):
                    clamped[key] = round(clamped[key])
            elif dist_type == "discrete_uniform":
                low, high = spec["low"], spec["high"]
                step = spec.get("step", 1)
                clamped[key] = max(low, min(high, value))
                clamped[key] = round((clamped[key] - low) / step) * step + low
            elif dist_type == "choice":
                values = spec["values"]
                if value not in values:
                    if isinstance(value, (int, float)):
                        clamped[key] = min(values, key=lambda x: abs(x - value))
                    else:
                        clamped[key] = values[0]
    return clamped


def _validate_llm_response(response, search_space):
    """Validate the parsed LLM response. Returns list of error strings."""
    errors = []

    if not isinstance(response, dict):
        return ["Response is not a JSON object"]

    decision = response.get("decision")
    if decision not in ("deploy", "retrain", "stop"):
        errors.append(f"Invalid decision: '{decision}' (expected deploy/retrain/stop)")

    if not response.get("reasoning", "").strip():
        errors.append("Missing or empty reasoning")

    confidence = response.get("confidence", -1)
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        errors.append(f"Confidence out of range: {confidence}")

    if decision == "retrain":
        next_hps = response.get("next_hyperparams")
        if not isinstance(next_hps, dict) or not next_hps:
            errors.append("decision=retrain requires non-empty next_hyperparams")

    return errors


def _parse_llm_response(raw_text):
    """Parse JSON from LLM response text, handling common formatting issues."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


def _call_llm(model_url, system_prompt, user_prompt, temperature):
    """Call a Clarifai-hosted LLM model. Returns raw text response."""
    from clarifai.client.user import User

    # Parse model URL to extract user_id, app_id, model_id
    # URL format: https://clarifai.com/{user_id}/{app_id}/models/{model_id}
    parts = model_url.rstrip("/").split("/")
    # Find 'models' in URL path
    model_idx = None
    for i, part in enumerate(parts):
        if part == "models":
            model_idx = i
            break

    if model_idx is None or model_idx < 2:
        raise ValueError(f"Invalid model URL format: {model_url}")

    model_user_id = parts[model_idx - 2]
    model_app_id = parts[model_idx - 1]
    model_id = parts[model_idx + 1]

    # Use the User class to get model and predict
    user = User(user_id=model_user_id)
    model = user.app(app_id=model_app_id).model(model_id=model_id)

    response = model.predict_by_bytes(
        input_bytes=user_prompt.encode("utf-8"),
        input_type="text",
        inference_params={
            "temperature": temperature,
            "max_tokens": 1024,
            "system_prompt": system_prompt,
        },
    )

    output_text = response.outputs[0].data.text.raw
    return output_text


class LLMDecision:

    @staticmethod
    def _get_argparse_type(param_annotation):
        if param_annotation == int:
            return int
        elif param_annotation == float:
            return float
        elif param_annotation == str:
            return str
        elif param_annotation == bool:
            return lambda x: str(x).lower() == 'true'
        else:
            return str

    @classmethod
    def to_pipeline_parser(cls):
        import argparse
        parser = argparse.ArgumentParser(description="LLM-based decision + HP adjustment for retrain loop")
        sig = inspect.signature(cls.decide)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def decide(
        self,
        eval_results_json: str,
        task_type: str = "detection",
        task_description: str = "",
        primary_metric: str = "auto",
        metric_direction: str = "auto",
        metric_threshold: float = 0.50,
        current_iteration: int = 1,
        max_retrain_iterations: int = 3,
        hp_history: str = "[]",
        current_hyperparams: str = "{}",
        search_space: str = "auto",
        tuning_strategy: str = "schedule",
        lr_decay_factor: float = 0.5,
        unfreeze_on_retry: bool = True,
        early_stop_min_delta: float = 0.0,
        overfitting_detection: bool = False,
        llm_model_url: str = "",
        llm_temperature: float = 0.1,
        llm_max_retries: int = 3,
        seed: int = 42,
    ) -> str:
        """LLM-based decision + HP generation with fallback to hardcoded logic.

        Returns path to /tmp/decision_output.json.
        """
        # ── Parse inputs ──
        with open(eval_results_json, 'r') as f:
            eval_results = json.load(f)
        history = json.loads(hp_history)
        current_hps = json.loads(current_hyperparams)
        metrics = eval_results.get("metrics", {})
        resolved_space = _resolve_search_space(search_space, task_type)

        # Resolve metric name and direction
        metric_name, direction = _resolve_metric(primary_metric, metric_direction, task_type)
        current_value = metrics.get(metric_name)

        logging.info(
            f"[Iteration {current_iteration}] {metric_name} = {current_value} "
            f"(threshold: {metric_threshold}, direction: {direction})"
        )

        # ── Attempt LLM path ──
        llm_result = None
        metadata = {}
        fallback_reason = ""

        if llm_model_url:
            user_prompt = build_decision_prompt(
                task_type=task_type,
                task_description=task_description,
                primary_metric=metric_name,
                metric_direction=direction,
                metric_threshold=metric_threshold,
                current_iteration=current_iteration,
                max_retrain_iterations=max_retrain_iterations,
                hp_history=history,
                current_hyperparams=current_hps,
                search_space=resolved_space,
                eval_metrics=metrics,
                current_metric_value=current_value,
            )

            last_errors = []
            for attempt in range(1, llm_max_retries + 1):
                try:
                    start_time = time.time()
                    raw_response = _call_llm(
                        model_url=llm_model_url,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=llm_temperature,
                    )
                    latency_ms = int((time.time() - start_time) * 1000)

                    parsed = _parse_llm_response(raw_response)
                    errors = _validate_llm_response(parsed, resolved_space)

                    if errors:
                        last_errors = errors
                        logging.warning(
                            f"[LLM attempt {attempt}/{llm_max_retries}] "
                            f"Validation errors: {errors}"
                        )
                        continue

                    # Valid response — clamp HPs if retrain
                    if parsed["decision"] == "retrain" and parsed.get("next_hyperparams"):
                        parsed["next_hyperparams"] = _clamp_hyperparams(
                            parsed["next_hyperparams"], resolved_space
                        )

                    llm_result = parsed
                    metadata = {
                        "method": "llm",
                        "model_url": llm_model_url,
                        "temperature": llm_temperature,
                        "attempts": attempt,
                        "latency_ms": latency_ms,
                        "confidence": parsed.get("confidence", 0.0),
                    }
                    logging.info(
                        f"[LLM] Decision: {parsed['decision']} "
                        f"(attempt {attempt}, {latency_ms}ms)"
                    )
                    break

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    last_errors = [f"{type(e).__name__}: {e}"]
                    logging.warning(
                        f"[LLM attempt {attempt}/{llm_max_retries}] "
                        f"Parse error: {e}"
                    )
                except Exception as e:
                    error_str = str(e)
                    # Non-retryable HTTP errors
                    if any(code in error_str for code in ["401", "403", "404"]):
                        last_errors = [f"Non-retryable error: {e}"]
                        logging.error(f"[LLM] Non-retryable error: {e}")
                        break
                    last_errors = [f"{type(e).__name__}: {e}"]
                    logging.warning(
                        f"[LLM attempt {attempt}/{llm_max_retries}] "
                        f"Error: {e}"
                    )

            if llm_result is None:
                fallback_reason = f"{llm_max_retries} retries exhausted: {last_errors}"
                logging.info(f"[LLM] All retries failed, using fallback. Reason: {fallback_reason}")
        else:
            fallback_reason = "llm_model_url not configured"
            logging.info("[LLM] No model URL configured, using fallback directly.")

        # ── Fallback path ──
        if llm_result is None:
            fb = fallback_decide(
                eval_results=eval_results,
                task_type=task_type,
                primary_metric=primary_metric,
                metric_direction=metric_direction,
                metric_threshold=metric_threshold,
                current_iteration=current_iteration,
                max_retrain_iterations=max_retrain_iterations,
                hp_history=history,
                current_hyperparams=current_hps,
                early_stop_min_delta=early_stop_min_delta,
                overfitting_detection=overfitting_detection,
                tuning_strategy=tuning_strategy,
                search_space=search_space,
                lr_decay_factor=lr_decay_factor,
                unfreeze_on_retry=unfreeze_on_retry,
                seed=seed,
                fallback_reason=fallback_reason,
            )
            decision = fb["decision"]
            reasoning = fb["reasoning"]
            confidence = fb["confidence"]
            is_overfit = fb["is_overfitting"]
            next_hps = fb["next_hyperparams"]
            metric_name = fb["metric_name"]
            current_value = fb["metric_value"]
            metadata = fb["metadata"]
        else:
            decision = llm_result["decision"]
            reasoning = llm_result["reasoning"]
            confidence = llm_result.get("confidence", 0.0)
            is_overfit = llm_result.get("is_overfitting", False)
            next_hps = llm_result.get("next_hyperparams", {})

        # ── Build reason string (short version for history) ──
        if current_value is not None:
            reason = f"{metric_name} {current_value:.4f} vs threshold {metric_threshold}"
        else:
            reason = f"Metric '{metric_name}' not found in eval results"

        # ── Update history ──
        history.append({
            "iteration": current_iteration,
            "hyperparams": current_hps,
            "metrics": metrics,
            "decision": decision,
            "reason": reason,
            "reasoning": reasoning,
            "method": metadata.get("method", "unknown"),
        })

        # ── Write Argo output parameters ──
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)

        outputs = {
            "decision": decision,
            "hyperparams_json": json.dumps(next_hps),
            "reasoning": reasoning,
            "metric_value": f"{current_value:.6f}" if current_value is not None else "N/A",
            "metric_name": metric_name,
            "hp_history": json.dumps(history),
            "is_overfitting": str(is_overfit).lower(),
            "current_iteration": str(current_iteration),
            "next_iteration": str(current_iteration + 1),
            "strategy_metadata": json.dumps(metadata),
        }
        for key, value in outputs.items():
            with open(os.path.join(output_dir, key), 'w') as f:
                f.write(value)

        # Full record for debugging
        full_output = {
            **outputs,
            "threshold": metric_threshold,
            "direction": direction,
            "task_type": task_type,
            "history_length": len(history),
            "confidence": confidence,
        }
        output_path = os.path.join(output_dir, "decision_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"Decision: {decision} | Method: {metadata.get('method')} | Reasoning: {reasoning}")
        return output_path
