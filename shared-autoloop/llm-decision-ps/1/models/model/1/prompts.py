"""Prompt templates for the LLM decision step."""

import json

SYSTEM_PROMPT = """You are an expert ML training loop controller. Your job is to analyze model evaluation metrics and decide whether to:
1. **deploy** the model (metrics meet the quality threshold)
2. **retrain** the model with adjusted hyperparameters (metrics are improving but haven't met threshold)
3. **stop** training (metrics have plateaued, budget exhausted, or no further improvement is likely)

You MUST respond with a single JSON object matching the schema provided. Do not include any other text outside the JSON."""


def _format_search_space(search_space):
    """Format search space for prompt display."""
    if not search_space:
        return "No search space constraints defined."
    lines = []
    for param, spec in search_space.items():
        if isinstance(spec, dict):
            stype = spec.get("type", "unknown")
            if stype in ("log_uniform", "uniform"):
                lines.append(f"  - {param}: {stype} [{spec.get('low')}, {spec.get('high')}]")
            elif stype == "choice":
                lines.append(f"  - {param}: {stype} {spec.get('values')}")
            elif stype == "discrete_uniform":
                lines.append(f"  - {param}: {stype} [{spec.get('low')}, {spec.get('high')}] step={spec.get('step')}")
            else:
                lines.append(f"  - {param}: {json.dumps(spec)}")
        else:
            lines.append(f"  - {param}: {spec}")
    return "\n".join(lines)


def _format_history(hp_history):
    """Format training history for prompt display."""
    if not hp_history:
        return "This is the first training run — no prior history available."
    lines = []
    for entry in hp_history:
        iteration = entry.get("iteration", "?")
        hps = entry.get("hyperparams", {})
        metrics = entry.get("metrics", {})
        decision = entry.get("decision", "?")
        reason = entry.get("reason", "")
        lines.append(f"  Iteration {iteration}: decision={decision}")
        if hps:
            lines.append(f"    Hyperparams: {json.dumps(hps)}")
        if metrics:
            metric_strs = [f"{k}={v}" for k, v in metrics.items()]
            lines.append(f"    Metrics: {', '.join(metric_strs)}")
        if reason:
            lines.append(f"    Reason: {reason}")
    return "\n".join(lines)


def build_decision_prompt(
    task_type,
    task_description,
    primary_metric,
    metric_direction,
    metric_threshold,
    current_iteration,
    max_retrain_iterations,
    hp_history,
    current_hyperparams,
    search_space,
    eval_metrics,
    current_metric_value,
):
    """Build the user prompt for the LLM decision call.

    Args:
        task_type: "detection", "classification", or "llm_finetune"
        task_description: Human-readable description of the task
        primary_metric: Name of the primary metric to optimize
        metric_direction: "maximize" or "minimize"
        metric_threshold: Target threshold for deployment
        current_iteration: Current iteration number (1-based)
        max_retrain_iterations: Maximum iterations allowed
        hp_history: List of prior iteration records
        current_hyperparams: Current hyperparameter values
        search_space: Hyperparameter search space definitions
        eval_metrics: All evaluation metrics from current run
        current_metric_value: Value of the primary metric
    """
    iterations_remaining = max_retrain_iterations - current_iteration

    sections = []

    # Task context
    sections.append("## Task Context")
    sections.append(f"Task type: {task_type}")
    if task_description:
        sections.append(f"Description: {task_description}")

    # Metric info
    sections.append("\n## Quality Gate")
    sections.append(f"Primary metric: {primary_metric} (direction: {metric_direction})")
    sections.append(f"Threshold for deployment: {metric_threshold}")
    sections.append(f"Current {primary_metric} value: {current_metric_value}")

    # All eval metrics
    if eval_metrics:
        sections.append("\nAll evaluation metrics:")
        for k, v in eval_metrics.items():
            sections.append(f"  - {k}: {v}")

    # Iteration budget
    sections.append("\n## Iteration Budget")
    sections.append(f"Current iteration: {current_iteration} of {max_retrain_iterations}")
    sections.append(f"Iterations remaining: {iterations_remaining}")
    if iterations_remaining <= 1:
        sections.append("WARNING: Very few iterations remain. Consider stopping if improvement is unlikely.")

    # Current hyperparams
    sections.append("\n## Current Hyperparameters")
    if current_hyperparams:
        sections.append(json.dumps(current_hyperparams, indent=2))
    else:
        sections.append("No hyperparameters set yet.")

    # Search space
    sections.append("\n## Hyperparameter Search Space")
    sections.append(_format_search_space(search_space))

    # Training history
    sections.append("\n## Training History")
    sections.append(_format_history(hp_history))

    # Output schema
    sections.append("\n## Required Output")
    sections.append("""Respond with a JSON object matching this schema:
{
  "decision": "deploy" | "retrain" | "stop",
  "reasoning": "<brief explanation of your decision>",
  "confidence": <float 0.0-1.0>,
  "is_overfitting": <boolean>,
  "next_hyperparams": {<new hyperparameter values if decision is retrain, else {}>}
}

Rules:
- If the primary metric meets/exceeds the threshold, decide "deploy"
- If retrain, you MUST provide next_hyperparams within the search space bounds
- If little improvement is possible or budget is exhausted, decide "stop"
- Confidence should reflect how certain you are about the decision""")

    return "\n".join(sections)
