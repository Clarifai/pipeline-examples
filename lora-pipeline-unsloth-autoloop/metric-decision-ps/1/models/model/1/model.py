import json
import inspect
import logging
import os

logging.basicConfig(level=logging.INFO)

METRIC_DEFAULTS = {
    "detection": {"metric": "AP", "direction": "maximize"},
    "classification": {"metric": "accuracy/top1", "direction": "maximize"},
    "llm_finetune": {"metric": "eval_loss", "direction": "minimize"},
}

MINIMIZE_PATTERNS = ["loss", "perplexity", "error", "cer", "wer"]


class MetricDecision:

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
        parser = argparse.ArgumentParser(description="Metric-based conditional branching")
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
        primary_metric: str = "auto",
        metric_direction: str = "auto",
        metric_threshold: float = 0.50,
        current_iteration: int = 1,
        max_retrain_iterations: int = 3,
        hp_history: str = "[]",
        current_hyperparams: str = "{}",
        early_stop_min_delta: float = 0.0,
        overfitting_detection: bool = False,
    ) -> str:
        """Compare eval metrics against threshold and output a routing decision.

        Returns path to /tmp/decision_output.json.
        """
        # ── Parse inputs ──
        with open(eval_results_json, 'r') as f:
            eval_results = json.load(f)
        history = json.loads(hp_history)
        current_hps = json.loads(current_hyperparams)
        metrics = eval_results.get("metrics", {})

        # ── Resolve metric ──
        metric_name, direction = self._resolve_metric(
            primary_metric, metric_direction, task_type
        )
        current_value = metrics.get(metric_name, 0.0)
        logging.info(
            f"[Iteration {current_iteration}] {metric_name} = {current_value:.4f} "
            f"(threshold: {metric_threshold}, direction: {direction})"
        )

        # ── Decision logic ──
        reason = ""
        is_overfit = False

        if self._metric_passes(current_value, metric_threshold, direction):
            decision = "deploy"
            reason = f"{metric_name} {current_value:.4f} meets threshold {metric_threshold}"

        elif early_stop_min_delta > 0 and self._should_early_stop(
            history, current_value, metric_name, direction, early_stop_min_delta
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
            is_overfit = self._detect_overfitting(
                eval_results, task_type, overfitting_detection
            )
            if is_overfit:
                reason += " (overfitting detected)"

        # ── Update history ──
        history.append({
            "iteration": current_iteration,
            "hyperparams": current_hps,
            "metrics": metrics,
            "decision": decision,
            "reason": reason,
        })

        # ── Write Argo output parameters ──
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)

        outputs = {
            "decision": decision,
            "metric_value": f"{current_value:.6f}",
            "metric_name": metric_name,
            "hp_history": json.dumps(history),
            "is_overfitting": str(is_overfit).lower(),
            "current_iteration": str(current_iteration),
            "next_iteration": str(current_iteration + 1),
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
        }
        output_path = os.path.join(output_dir, "decision_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"Decision: {decision} | Reason: {reason}")
        return output_path

    @staticmethod
    def _resolve_metric(primary_metric, metric_direction, task_type):
        if primary_metric == "auto":
            primary_metric = METRIC_DEFAULTS[task_type]["metric"]
        if metric_direction == "auto":
            if any(p in primary_metric.lower() for p in MINIMIZE_PATTERNS):
                metric_direction = "minimize"
            else:
                metric_direction = METRIC_DEFAULTS.get(
                    task_type, {}
                ).get("direction", "maximize")
        return primary_metric, metric_direction

    @staticmethod
    def _metric_passes(value, threshold, direction):
        if direction == "maximize":
            return value >= threshold
        return value <= threshold

    @staticmethod
    def _should_early_stop(hp_history, current_value, primary_metric, direction, min_delta):
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

    @staticmethod
    def _detect_overfitting(eval_results, task_type, overfitting_detection):
        if not overfitting_detection:
            return False
        metrics = eval_results.get("metrics", {})
        train_loss = metrics.get("train_loss")
        eval_loss = metrics.get("eval_loss")
        if train_loss is None or eval_loss is None:
            return False
        return train_loss < eval_loss * 0.5
