import json
import inspect
import logging
import os
import sys

# Ensure strategies.py (co-located with model.py) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import (
    schedule_select,
    grid_select,
    random_select,
    apply_overfit_corrections,
    resolve_search_space,
)

logging.basicConfig(level=logging.INFO)


STRATEGY_DISPATCH = {
    "schedule": schedule_select,
    "grid": grid_select,
    "random": random_select,
}


class HPAdjustment:

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
        parser = argparse.ArgumentParser(description="Hyperparameter adjustment for retrain loop")
        sig = inspect.signature(cls.adjust)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            if param.default != inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}", type=arg_type, default=param.default)
            else:
                parser.add_argument(f"--{param_name}", type=arg_type, required=True)
        return parser

    def adjust(
        self,
        hp_history: str = "[]",
        current_hyperparams: str = "{}",
        task_type: str = "detection",
        tuning_strategy: str = "schedule",
        search_space: str = "auto",
        lr_decay_factor: float = 0.5,
        unfreeze_on_retry: bool = True,
        is_overfitting: bool = False,
        seed: int = 42,
    ) -> str:
        """Generate adjusted hyperparameters for the next training iteration.

        Returns path to /tmp/hp_output.json.
        """
        # ── Parse inputs ──
        history = json.loads(hp_history)
        current_hps = json.loads(current_hyperparams)
        resolved_space = resolve_search_space(search_space, task_type, tuning_strategy)
        iteration = len(history) + 1

        logging.info(
            f"[HP Adjust] Strategy: {tuning_strategy}, task: {task_type}, "
            f"iteration: {iteration}, overfit: {is_overfitting}"
        )

        # ── Select strategy and generate new HPs ──
        strategy_fn = STRATEGY_DISPATCH.get(tuning_strategy)
        if strategy_fn is None:
            raise ValueError(
                f"Unknown tuning_strategy: {tuning_strategy}. "
                f"Valid options: {list(STRATEGY_DISPATCH.keys())}"
            )

        if tuning_strategy == "schedule":
            new_hps = strategy_fn(
                current_hps=current_hps,
                task_type=task_type,
                iteration=iteration,
                lr_decay=lr_decay_factor,
                unfreeze=unfreeze_on_retry,
            )
        else:  # grid or random
            new_hps = strategy_fn(
                current_hps=current_hps,
                search_space=resolved_space,
                hp_history=history,
                seed=seed,
            )

        # ── Track changes before overfit corrections ──
        changes = {
            k: {"from": current_hps.get(k), "to": v}
            for k, v in new_hps.items()
            if current_hps.get(k) != v
        }

        # ── Apply overfitting corrections ──
        overfit_applied = False
        if is_overfitting:
            pre_overfit = new_hps.copy()
            new_hps = apply_overfit_corrections(new_hps, task_type)
            overfit_applied = any(new_hps.get(k) != pre_overfit.get(k) for k in new_hps)

        # ── Build metadata ──
        metadata = {
            "strategy": tuning_strategy,
            "task_type": task_type,
            "iteration": iteration,
            "changes": changes,
            "overfit_adjustments": overfit_applied,
            "search_space_used": resolved_space if tuning_strategy != "schedule" else None,
            "seed": seed if tuning_strategy in ("grid", "random") else None,
        }

        # ── Write outputs ──
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)

        outputs = {
            "hyperparams_json": json.dumps(new_hps),
            "strategy_metadata": json.dumps(metadata),
        }
        for key, value in outputs.items():
            with open(os.path.join(output_dir, key), 'w') as f:
                f.write(value)

        # Full record for debugging
        full_output = {
            "input_hyperparams": current_hps,
            "output_hyperparams": new_hps,
            "metadata": metadata,
        }
        output_path = os.path.join(output_dir, "hp_output.json")
        with open(output_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logging.info(f"[HP Adjust] Changes: {changes}")
        if overfit_applied:
            logging.info(f"[HP Adjust] Overfit corrections applied for {task_type}")

        return output_path
