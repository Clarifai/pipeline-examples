import itertools
import json
import math
import random


# ═══════════════════════════════════════════════════════════
# SCHEDULE STRATEGY
# ═══════════════════════════════════════════════════════════

def schedule_select(current_hps, task_type, iteration, lr_decay, unfreeze):
    dispatch = {
        "detection": _schedule_detection,
        "classification": _schedule_classification,
        "llm_finetune": _schedule_llm,
    }
    fn = dispatch.get(task_type, _schedule_detection)
    return fn(current_hps, iteration, lr_decay, unfreeze)


def _schedule_detection(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["per_item_lrate"] = current_hps.get("per_item_lrate", 0.001875) * lr_decay
    if unfreeze:
        frozen = new_hps.get("frozen_stages", 1)
        if frozen > 0:
            new_hps["frozen_stages"] = frozen - 1
    return new_hps


def _schedule_classification(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["per_item_lrate"] = current_hps.get("per_item_lrate", 1.95e-5) * lr_decay
    current_wd = new_hps.get("weight_decay", 0.01)
    new_hps["weight_decay"] = min(current_wd * 1.5, 0.1)
    return new_hps


def _schedule_llm(current_hps, iteration, lr_decay, unfreeze):
    new_hps = current_hps.copy()
    new_hps["learning_rate"] = current_hps.get("learning_rate", 2e-4) * lr_decay
    current_r = new_hps.get("lora_r", 16)
    if current_r < 128:
        new_hps["lora_r"] = min(current_r * 2, 128)
        new_hps["lora_alpha"] = new_hps["lora_r"]
    return new_hps


# ═══════════════════════════════════════════════════════════
# GRID STRATEGY
# ═══════════════════════════════════════════════════════════

GRID_DEFAULTS = {
    "detection": {
        "per_item_lrate": [0.001875, 0.0009375, 0.003],
        "frozen_stages": [0, 1],
    },
    "classification": {
        "per_item_lrate": [1.95e-5, 1e-5, 5e-5],
        "weight_decay": [0.01, 0.05],
    },
    "llm_finetune": {
        "learning_rate": [2e-4, 1e-4, 5e-5],
        "lora_r": [16, 32, 64],
    },
}


def grid_select(current_hps, search_space, hp_history, seed):
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    tried = set()
    for entry in hp_history:
        combo_key = tuple(entry["hyperparams"].get(k) for k in keys)
        tried.add(combo_key)

    remaining = [
        combo for combo in all_combos
        if tuple(combo.get(k) for k in keys) not in tried
    ]

    if not remaining:
        remaining = all_combos

    new_hps = current_hps.copy()
    new_hps.update(remaining[0])
    return new_hps


# ═══════════════════════════════════════════════════════════
# RANDOM STRATEGY
# ═══════════════════════════════════════════════════════════

RANDOM_DEFAULTS = {
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


def _sample_hp(spec, rng):
    dist_type = spec["type"]
    if dist_type == "log_uniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    elif dist_type == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    elif dist_type == "discrete_uniform":
        steps = list(range(spec["low"], spec["high"] + 1, spec.get("step", 1)))
        return rng.choice(steps)
    elif dist_type == "choice":
        return rng.choice(spec["values"])
    elif dist_type == "int_log_uniform":
        return round(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def random_select(current_hps, search_space, hp_history, seed):
    iteration = len(hp_history) + 1
    rng = random.Random(seed + iteration)
    new_hps = current_hps.copy()
    for key, spec in search_space.items():
        new_hps[key] = _sample_hp(spec, rng)
    return new_hps


# ═══════════════════════════════════════════════════════════
# OVERFITTING CORRECTIONS
# ═══════════════════════════════════════════════════════════

def apply_overfit_corrections(new_hps, task_type):
    corrected = new_hps.copy()
    if task_type == "detection":
        current_epochs = corrected.get("num_epochs", 12)
        corrected["num_epochs"] = max(current_epochs // 2, 4)
    elif task_type == "classification":
        current_wd = corrected.get("weight_decay", 0.01)
        corrected["weight_decay"] = min(current_wd * 3, 0.1)
        current_epochs = corrected.get("num_epochs", 200)
        corrected["num_epochs"] = max(current_epochs // 2, 20)
    elif task_type == "llm_finetune":
        current_r = corrected.get("lora_r", 16)
        corrected["lora_r"] = max(current_r // 2, 4)
        corrected["lora_alpha"] = corrected["lora_r"]
        current_wd = corrected.get("weight_decay", 0.0)
        corrected["weight_decay"] = max(current_wd, 0.01)
    return corrected


# ═══════════════════════════════════════════════════════════
# SEARCH SPACE RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_search_space(search_space, task_type, tuning_strategy):
    if search_space != "auto":
        return json.loads(search_space)
    if tuning_strategy == "grid":
        return GRID_DEFAULTS.get(task_type, {})
    elif tuning_strategy == "random":
        return RANDOM_DEFAULTS.get(task_type, {})
    return {}
