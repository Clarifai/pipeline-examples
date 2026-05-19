"""Fallback logic: runs existing metric-decision + hp-adjust when LLM is unavailable."""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)

# Import the existing shared-autoloop step implementations
# 5 levels up from llm-decision-ps/1/models/model/1/ -> shared-autoloop/
_SHARED_AUTOLOOP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)

METRIC_DEFAULTS = {
    "detection": {"metric": "AP", "direction": "maximize"},
    "classification": {"metric": "accuracy/top1", "direction": "maximize"},
    "llm_finetune": {"metric": "eval_loss", "direction": "minimize"},
}

MINIMIZE_PATTERNS = ["loss", "perplexity", "error", "cer", "wer"]


def _resolve_metric(primary_metric, metric_direction, task_type):
    """Resolve 'auto' metric name and direction from task_type."""
    if primary_metric == "auto":
        if task_type not in METRIC_DEFAULTS:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Allowed values: {list(METRIC_DEFAULTS.keys())}"
            )
        primary_metric = METRIC_DEFAULTS[task_type]["metric"]
    if metric_direction == "auto":
        if any(p in primary_metric.lower() for p in MINIMIZE_PATTERNS):
            metric_direction = "minimize"
        else:
            metric_direction = METRIC_DEFAULTS.get(
                task_type, {}
            ).get("direction", "maximize")
    return primary_metric, metric_direction


def _metric_passes(value, threshold, direction):
    """Check if metric meets quality gate."""
    if direction == "maximize":
        return value >= threshold
    elif direction == "minimize":
        return value <= threshold
    else:
        raise ValueError(
            f"Invalid metric_direction '{direction}'. Must be 'maximize' or 'minimize'."
        )


def _should_early_stop(hp_history, current_value, primary_metric, direction, min_delta):
    """Check for plateau between latest two iterations."""
    if min_delta <= 0 or len(hp_history) < 1:
        return False
    prev_value = hp_history[-1]["metrics"].get(primary_metric)
    if prev_value is None:
        return False
    if direction == "maximize":
        improvement = current_value - prev_value
    else:
        improvement = prev_value - current_value
    return improvement < min_delta


def _detect_overfitting(eval_results, overfitting_detection):
    """Check for train_loss << eval_loss divergence."""
    if not overfitting_detection:
        return False
    metrics = eval_results.get("metrics", {})
    train_loss = metrics.get("train_loss")
    eval_loss = metrics.get("eval_loss")
    if train_loss is None or eval_loss is None:
        return False
    return train_loss < eval_loss * 0.5


def _run_hp_adjust(current_hps, task_type, tuning_strategy, search_space,
                   lr_decay_factor, unfreeze_on_retry, is_overfitting, hp_history, seed):
    """Run HP adjustment logic inline (equivalent to hp-adjust-ps)."""
    # In Docker container, strategies.py is co-located; in local dev, use sibling path
    local_dir = os.path.dirname(os.path.abspath(__file__))
    strategies_path = os.path.join(
        _SHARED_AUTOLOOP_DIR, "hp-adjust-ps", "1", "models", "model", "1"
    )
    for path in (local_dir, strategies_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    from strategies import (
        schedule_select, grid_select, random_select,
        apply_overfit_corrections, resolve_search_space,
    )

    resolved_space = resolve_search_space(search_space, task_type, tuning_strategy)
    iteration = len(hp_history) + 1

    if tuning_strategy == "schedule":
        new_hps = schedule_select(
            current_hps=current_hps,
            task_type=task_type,
            iteration=iteration,
            lr_decay=lr_decay_factor,
            unfreeze=unfreeze_on_retry,
        )
    else:
        strategy_fn = grid_select if tuning_strategy == "grid" else random_select
        new_hps = strategy_fn(
            current_hps=current_hps,
            search_space=resolved_space,
            hp_history=hp_history,
            seed=seed,
        )

    # Apply overfitting corrections
    overfit_applied = False
    if is_overfitting:
        pre_overfit = new_hps.copy()
        new_hps = apply_overfit_corrections(new_hps, task_type)
        overfit_applied = any(new_hps.get(k) != pre_overfit.get(k) for k in new_hps)

    return new_hps, resolved_space, overfit_applied


def fallback_decide(
    eval_results,
    task_type,
    primary_metric,
    metric_direction,
    metric_threshold,
    current_iteration,
    max_retrain_iterations,
    hp_history,
    current_hyperparams,
    early_stop_min_delta,
    overfitting_detection,
    tuning_strategy,
    search_space,
    lr_decay_factor,
    unfreeze_on_retry,
    seed,
    fallback_reason="",
):
    """Run hardcoded metric-decision + hp-adjust logic as fallback.

    Returns a dict matching the LLM response schema:
    {decision, reasoning, confidence, is_overfitting, next_hyperparams}
    Plus metadata for strategy_metadata output.
    """
    metrics = eval_results.get("metrics", {})

    # Resolve metric
    metric_name, direction = _resolve_metric(primary_metric, metric_direction, task_type)
    current_value = metrics.get(metric_name)

    # Decision logic
    reason = ""
    is_overfit = False

    if current_value is None:
        if current_iteration >= max_retrain_iterations:
            decision = "stop"
            reason = f"Metric '{metric_name}' missing from eval results after {current_iteration} iterations"
        else:
            decision = "retrain"
            reason = f"Metric '{metric_name}' missing from eval results"

    elif _metric_passes(current_value, metric_threshold, direction):
        decision = "deploy"
        reason = f"{metric_name} {current_value:.4f} meets threshold {metric_threshold}"

    elif early_stop_min_delta > 0 and _should_early_stop(
        hp_history, current_value, metric_name, direction, early_stop_min_delta
    ):
        decision = "stop"
        reason = f"Plateau: improvement below {early_stop_min_delta}"

    elif current_iteration >= max_retrain_iterations:
        decision = "stop"
        reason = (
            f"{metric_name} {current_value:.4f} below threshold {metric_threshold} "
            f"after {current_iteration} iterations"
        )

    else:
        decision = "retrain"
        reason = f"{metric_name} {current_value:.4f} below threshold {metric_threshold}"
        is_overfit = _detect_overfitting(eval_results, overfitting_detection)
        if is_overfit:
            reason += " (overfitting detected)"

    # HP adjustment (only if retrain)
    next_hps = {}
    overfit_applied = False
    if decision == "retrain":
        next_hps, resolved_space, overfit_applied = _run_hp_adjust(
            current_hps=current_hyperparams,
            task_type=task_type,
            tuning_strategy=tuning_strategy,
            search_space=search_space if isinstance(search_space, str) else json.dumps(search_space),
            lr_decay_factor=lr_decay_factor,
            unfreeze_on_retry=unfreeze_on_retry,
            is_overfitting=is_overfit,
            hp_history=hp_history,
            seed=seed,
        )

    reasoning = f"[FALLBACK] {reason}"
    if fallback_reason:
        reasoning += f" (LLM unavailable: {fallback_reason})"

    metadata = {
        "method": "fallback",
        "fallback_reason": fallback_reason,
        "fallback_strategy": tuning_strategy,
        "fallback_task_type": task_type,
        "overfit_adjustments": overfit_applied,
    }

    logging.info(f"[Fallback] Decision: {decision} | Reason: {reason}")

    return {
        "decision": decision,
        "reasoning": reasoning,
        "confidence": 1.0,
        "is_overfitting": is_overfit,
        "next_hyperparams": next_hps,
        "metric_name": metric_name,
        "metric_value": current_value,
        "metadata": metadata,
    }
