"""Unit tests for the hp-adjust pipeline step (all task types).

Covers: detection, classification, llm_finetune

Run: python -m pytest shared-autoloop/hp-adjust-ps/tests/test_hp_adjust.py -v
"""

import importlib.util
import json
import os
import sys

import pytest

# Load model.py via importlib to avoid sys.modules collision with metric-decision
_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models", "model", "1", "model.py"
)
_spec = importlib.util.spec_from_file_location("hp_adjust_model", _model_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HPAdjustment = _mod.HPAdjustment

# Also import strategies directly for isolated tests
_strategies_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models", "model", "1", "strategies.py"
)
_spec_s = importlib.util.spec_from_file_location("hp_adjust_strategies", _strategies_path)
strategies = importlib.util.module_from_spec(_spec_s)
_spec_s.loader.exec_module(strategies)


@pytest.fixture
def adjuster():
    return HPAdjustment()


def _read_output(key):
    """Read an Argo output param written to /tmp/."""
    with open(f"/tmp/{key}") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════
# DETECTION TASK
# ═══════════════════════════════════════════════════════════════


class TestScheduleDetection:
    def test_lr_decay(self, adjuster):
        """LR should be halved with default lr_decay_factor=0.5."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(0.0009375)

    def test_unfreeze_on_retry(self, adjuster):
        """frozen_stages should decrease by 1 when unfreeze_on_retry=True."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 2}',
            task_type="detection",
            tuning_strategy="schedule",
            unfreeze_on_retry=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == 1

    def test_unfreeze_stops_at_zero(self, adjuster):
        """frozen_stages should not go below 0."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 0}',
            task_type="detection",
            tuning_strategy="schedule",
            unfreeze_on_retry=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == 0

    def test_no_unfreeze_when_disabled(self, adjuster):
        """frozen_stages should stay unchanged when unfreeze_on_retry=False."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="schedule",
            unfreeze_on_retry=False,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == 1

    def test_successive_decays(self, adjuster):
        """Two schedule iterations: LR decays twice, frozen_stages reaches 0."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
        )
        iter1_hps = json.loads(_read_output("hyperparams_json"))
        assert iter1_hps["per_item_lrate"] == pytest.approx(0.0009375)
        assert iter1_hps["frozen_stages"] == 0

        adjuster.adjust(
            current_hyperparams=json.dumps(iter1_hps),
            hp_history='[{"iteration": 1, "hyperparams": {}, "metrics": {}, "decision": "retrain", "reason": ""}]',
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
        )
        iter2_hps = json.loads(_read_output("hyperparams_json"))
        assert iter2_hps["per_item_lrate"] == pytest.approx(0.00046875)
        assert iter2_hps["frozen_stages"] == 0

    def test_custom_decay_factor(self, adjuster):
        """LR decay with non-default factor (0.3)."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875}',
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.3,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(0.001875 * 0.3)

    def test_empty_current_hps_uses_defaults(self, adjuster):
        """Empty current_hyperparams → uses default per_item_lrate and frozen_stages."""
        adjuster.adjust(
            current_hyperparams="{}",
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(0.001875 * 0.5)
        assert new_hps["frozen_stages"] == 0


class TestGridDetection:
    def test_first_grid_combo(self, adjuster):
        """First iteration picks the first untried grid combination."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == 0
        assert new_hps["per_item_lrate"] == 0.001875

    def test_skips_tried_combos(self, adjuster):
        """Should skip combinations already in hp_history."""
        history = json.dumps([
            {"iteration": 1, "hyperparams": {"frozen_stages": 0, "per_item_lrate": 0.001875}, "metrics": {}, "decision": "retrain", "reason": ""}
        ])
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            hp_history=history,
            task_type="detection",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        combo = (new_hps["frozen_stages"], new_hps["per_item_lrate"])
        assert combo != (0, 0.001875)

    def test_custom_search_space(self, adjuster):
        """Custom grid search space."""
        space = json.dumps({
            "per_item_lrate": [0.001, 0.002],
            "frozen_stages": [0],
        })
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="grid",
            search_space=space,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == 0
        assert new_hps["per_item_lrate"] in [0.001, 0.002]

    def test_grid_wraps_around(self, adjuster):
        """When all combos tried, restarts from beginning."""
        import itertools
        default_grid = strategies.GRID_DEFAULTS["detection"]
        keys = sorted(default_grid.keys())
        all_combos = list(itertools.product(*[default_grid[k] for k in keys]))
        history = []
        for i, combo in enumerate(all_combos):
            hp = dict(zip(keys, combo))
            history.append({"iteration": i + 1, "hyperparams": hp, "metrics": {}, "decision": "retrain", "reason": ""})

        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            hp_history=json.dumps(history),
            task_type="detection",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["frozen_stages"] == all_combos[0][0]
        assert new_hps["per_item_lrate"] == all_combos[0][1]


class TestRandomDetection:
    def test_random_produces_valid_hps(self, adjuster):
        """Random strategy produces per_item_lrate in range and valid frozen_stages."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="random",
            search_space="auto",
            seed=42,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert 1e-5 <= new_hps["per_item_lrate"] <= 1e-2
        assert new_hps["frozen_stages"] in [0, 1]

    def test_random_is_seeded(self, adjuster):
        """Same seed + same history length → same output."""
        for _ in range(2):
            adjuster.adjust(
                current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
                task_type="detection",
                tuning_strategy="random",
                search_space="auto",
                seed=42,
            )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="random",
            search_space="auto",
            seed=42,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1 == result2

    def test_different_seeds_differ(self, adjuster):
        """Different seeds → (very likely) different outputs."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="random",
            search_space="auto",
            seed=42,
        )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="random",
            search_space="auto",
            seed=999,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1["per_item_lrate"] != result2["per_item_lrate"]


class TestOverfitDetection:
    def test_overfit_halves_epochs(self, adjuster):
        """is_overfitting=True → num_epochs halved for detection."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1, "num_epochs": 100}',
            task_type="detection",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["num_epochs"] == 50

    def test_overfit_epoch_floor(self, adjuster):
        """Overfit epoch reduction should not go below 4."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "num_epochs": 6}',
            task_type="detection",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["num_epochs"] == 4

    def test_no_overfit_no_epoch_change(self, adjuster):
        """is_overfitting=False → num_epochs untouched by overfit corrections."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "num_epochs": 100}',
            task_type="detection",
            tuning_strategy="schedule",
            is_overfitting=False,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["num_epochs"] == 100

    def test_overfit_metadata(self, adjuster):
        """Metadata should reflect that overfit corrections were applied."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "num_epochs": 100}',
            task_type="detection",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert metadata["overfit_adjustments"] is True


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION TASK
# ═══════════════════════════════════════════════════════════════


class TestScheduleClassification:
    def test_lr_decay(self, adjuster):
        """per_item_lrate should be multiplied by lr_decay_factor."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(1.95e-5 * 0.5)

    def test_weight_decay_increases(self, adjuster):
        """weight_decay should increase by 1.5x each iteration."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["weight_decay"] == pytest.approx(0.015)

    def test_weight_decay_caps_at_01(self, adjuster):
        """weight_decay should not exceed 0.1."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.08}',
            task_type="classification",
            tuning_strategy="schedule",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["weight_decay"] == pytest.approx(0.1)

    def test_successive_iterations(self, adjuster):
        """Three iterations: LR decays, weight_decay grows each time."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter1 = json.loads(_read_output("hyperparams_json"))
        assert iter1["per_item_lrate"] == pytest.approx(1.95e-5 * 0.5)
        assert iter1["weight_decay"] == pytest.approx(0.015)

        adjuster.adjust(
            current_hyperparams=json.dumps(iter1),
            hp_history='[{"iteration": 1, "hyperparams": {}, "metrics": {}, "decision": "retrain", "reason": ""}]',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter2 = json.loads(_read_output("hyperparams_json"))
        assert iter2["per_item_lrate"] == pytest.approx(1.95e-5 * 0.25)
        assert iter2["weight_decay"] == pytest.approx(0.0225)

        adjuster.adjust(
            current_hyperparams=json.dumps(iter2),
            hp_history='[{}, {}]',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter3 = json.loads(_read_output("hyperparams_json"))
        assert iter3["per_item_lrate"] == pytest.approx(1.95e-5 * 0.125)
        assert iter3["weight_decay"] == pytest.approx(0.03375)

    def test_empty_hps_uses_defaults(self, adjuster):
        """Empty current_hyperparams → uses default per_item_lrate=1.95e-5, weight_decay=0.01."""
        adjuster.adjust(
            current_hyperparams="{}",
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(1.95e-5 * 0.5)
        assert new_hps["weight_decay"] == pytest.approx(0.015)

    def test_custom_decay_factor(self, adjuster):
        """Custom lr_decay_factor=0.3."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5}',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.3,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] == pytest.approx(1.95e-5 * 0.3)


class TestGridClassification:
    def test_first_grid_combo(self, adjuster):
        """First iteration picks the first grid combination."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] in [1.95e-5, 1e-5, 5e-5]
        assert new_hps["weight_decay"] in [0.01, 0.05]

    def test_skips_tried_combos(self, adjuster):
        """Should skip combinations already in hp_history."""
        import itertools
        default_grid = strategies.GRID_DEFAULTS["classification"]
        keys = sorted(default_grid.keys())
        first_combo = dict(zip(keys, next(itertools.product(*[default_grid[k] for k in keys]))))

        history = json.dumps([
            {"iteration": 1, "hyperparams": first_combo, "metrics": {}, "decision": "retrain", "reason": ""}
        ])
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            hp_history=history,
            task_type="classification",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        result_combo = {k: new_hps[k] for k in keys}
        assert result_combo != first_combo

    def test_custom_search_space(self, adjuster):
        """Custom grid search space for classification."""
        space = json.dumps({
            "per_item_lrate": [1e-5, 3e-5],
            "weight_decay": [0.02, 0.08],
        })
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="grid",
            search_space=space,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] in [1e-5, 3e-5]
        assert new_hps["weight_decay"] in [0.02, 0.08]

    def test_exhausts_all_combos(self, adjuster):
        """After exhausting all combos, should wrap around."""
        import itertools
        default_grid = strategies.GRID_DEFAULTS["classification"]
        keys = sorted(default_grid.keys())
        all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*[default_grid[k] for k in keys])]
        history = json.dumps([
            {"iteration": i + 1, "hyperparams": combo, "metrics": {}, "decision": "retrain", "reason": ""}
            for i, combo in enumerate(all_combos)
        ])
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            hp_history=history,
            task_type="classification",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["per_item_lrate"] in [1.95e-5, 1e-5, 5e-5]
        assert new_hps["weight_decay"] in [0.01, 0.05]


class TestRandomClassification:
    def test_random_produces_valid_hps(self, adjuster):
        """Random strategy produces per_item_lrate and weight_decay in range."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="random",
            search_space="auto",
            seed=42,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert 1e-6 <= new_hps["per_item_lrate"] <= 1e-3
        assert 0.001 <= new_hps["weight_decay"] <= 0.1

    def test_random_is_seeded(self, adjuster):
        """Same seed + same history length → same output."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="random",
            seed=42,
        )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="random",
            seed=42,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1 == result2

    def test_different_seeds_differ(self, adjuster):
        """Different seeds → different outputs."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="random",
            seed=42,
        )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="random",
            seed=999,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1["per_item_lrate"] != result2["per_item_lrate"]


class TestOverfitClassification:
    def test_overfit_triples_weight_decay(self, adjuster):
        """is_overfitting=True → weight_decay tripled (capped at 0.1)."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        # Schedule sets WD to 0.015, then overfit triples → 0.045
        assert new_hps["weight_decay"] == pytest.approx(0.045)

    def test_overfit_weight_decay_cap(self, adjuster):
        """Overfit weight_decay should not exceed 0.1."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.05}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["weight_decay"] == pytest.approx(0.1)

    def test_overfit_halves_epochs(self, adjuster):
        """is_overfitting=True → num_epochs halved (floor 20)."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "num_epochs": 200}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["num_epochs"] == 100

    def test_overfit_epoch_floor(self, adjuster):
        """num_epochs should not go below 20."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "num_epochs": 30}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["num_epochs"] == 20

    def test_no_overfit_no_epoch_change(self, adjuster):
        """is_overfitting=False → no overfit corrections applied."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01, "num_epochs": 200}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=False,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["weight_decay"] == pytest.approx(0.015)
        assert new_hps["num_epochs"] == 200

    def test_overfit_metadata(self, adjuster):
        """Metadata should reflect that overfit corrections were applied."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5}',
            task_type="classification",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert metadata["overfit_adjustments"] is True


# ═══════════════════════════════════════════════════════════════
# LLM FINE-TUNING TASK
# ═══════════════════════════════════════════════════════════════


class TestScheduleLLM:
    def test_lr_decay(self, adjuster):
        """Learning rate should be halved with default lr_decay_factor=0.5."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["learning_rate"] == pytest.approx(0.0001)

    def test_lora_rank_doubles(self, adjuster):
        """LoRA rank should double on each schedule iteration."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["lora_r"] == 32
        assert new_hps["lora_alpha"] == 32

    def test_lora_rank_caps_at_128(self, adjuster):
        """LoRA rank should not exceed 128."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 128, "lora_alpha": 128}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["lora_r"] == 128
        assert new_hps["lora_alpha"] == 128

    def test_alpha_follows_rank(self, adjuster):
        """lora_alpha should always equal lora_r after schedule adjustment."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 32, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["lora_alpha"] == new_hps["lora_r"]

    def test_successive_iterations(self, adjuster):
        """Three iterations: LR decays, rank doubles each time up to cap."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter1 = json.loads(_read_output("hyperparams_json"))
        assert iter1["learning_rate"] == pytest.approx(1e-4)
        assert iter1["lora_r"] == 32

        adjuster.adjust(
            current_hyperparams=json.dumps(iter1),
            hp_history='[{"iteration": 1, "hyperparams": {}, "metrics": {}, "decision": "retrain", "reason": ""}]',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter2 = json.loads(_read_output("hyperparams_json"))
        assert iter2["learning_rate"] == pytest.approx(5e-5)
        assert iter2["lora_r"] == 64

        adjuster.adjust(
            current_hyperparams=json.dumps(iter2),
            hp_history='[{}, {}]',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter3 = json.loads(_read_output("hyperparams_json"))
        assert iter3["learning_rate"] == pytest.approx(2.5e-5)
        assert iter3["lora_r"] == 128

    def test_empty_hps_uses_defaults(self, adjuster):
        """Empty current_hyperparams → uses default learning_rate=2e-4, lora_r=16."""
        adjuster.adjust(
            current_hyperparams="{}",
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["learning_rate"] == pytest.approx(1e-4)
        assert new_hps["lora_r"] == 32
        assert new_hps["lora_alpha"] == 32

    def test_custom_decay_factor(self, adjuster):
        """Custom lr_decay_factor=0.3."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.3,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["learning_rate"] == pytest.approx(0.0002 * 0.3)


class TestGridLLM:
    def test_first_grid_combo(self, adjuster):
        """First iteration picks the first grid combination."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["learning_rate"] in [2e-4, 1e-4, 5e-5]
        assert new_hps["lora_r"] in [16, 32, 64]

    def test_skips_tried_combos(self, adjuster):
        """Should skip combinations already in hp_history."""
        import itertools
        default_grid = strategies.GRID_DEFAULTS["llm_finetune"]
        keys = sorted(default_grid.keys())
        first_combo = dict(zip(keys, next(itertools.product(*[default_grid[k] for k in keys]))))

        history = json.dumps([
            {"iteration": 1, "hyperparams": first_combo, "metrics": {}, "decision": "retrain", "reason": ""}
        ])
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            hp_history=history,
            task_type="llm_finetune",
            tuning_strategy="grid",
            search_space="auto",
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        result_combo = {k: new_hps[k] for k in keys}
        assert result_combo != first_combo

    def test_custom_search_space(self, adjuster):
        """Custom grid search space for LoRA."""
        space = json.dumps({
            "learning_rate": [1e-4, 5e-5],
            "lora_r": [32, 64],
        })
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="grid",
            search_space=space,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["learning_rate"] in [1e-4, 5e-5]
        assert new_hps["lora_r"] in [32, 64]


class TestRandomLLM:
    def test_random_produces_valid_hps(self, adjuster):
        """Random strategy produces learning_rate in range and valid lora_r."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="random",
            search_space="auto",
            seed=42,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert 1e-5 <= new_hps["learning_rate"] <= 5e-4
        assert new_hps["lora_r"] in [8, 16, 32, 64, 128]
        assert new_hps["lora_alpha"] in [8, 16, 32, 64, 128]

    def test_random_is_seeded(self, adjuster):
        """Same seed + same history length → same output."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="random",
            seed=42,
        )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="random",
            seed=42,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1 == result2

    def test_different_seeds_differ(self, adjuster):
        """Different seeds → different outputs."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="random",
            seed=42,
        )
        result1 = json.loads(_read_output("hyperparams_json"))

        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="random",
            seed=999,
        )
        result2 = json.loads(_read_output("hyperparams_json"))
        assert result1["learning_rate"] != result2["learning_rate"]


class TestOverfitLLM:
    def test_overfit_halves_lora_rank(self, adjuster):
        """is_overfitting=True → lora_r halved, lora_alpha follows."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 32, "lora_alpha": 32}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        # Schedule doubles rank to 64, then overfit halves it to 32
        assert new_hps["lora_r"] == 32
        assert new_hps["lora_alpha"] == 32

    def test_overfit_lora_rank_floor(self, adjuster):
        """Overfit lora_r reduction should not go below 4."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 4, "lora_alpha": 4}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        # Schedule doubles 4→8, overfit halves 8→4
        assert new_hps["lora_r"] == 4
        assert new_hps["lora_alpha"] == 4

    def test_overfit_adds_weight_decay(self, adjuster):
        """Overfitting should set weight_decay to at least 0.01."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "weight_decay": 0.0}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["weight_decay"] >= 0.01

    def test_no_overfit_no_rank_change(self, adjuster):
        """is_overfitting=False → rank follows schedule (doubles), no overfit correction."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            is_overfitting=False,
        )
        new_hps = json.loads(_read_output("hyperparams_json"))
        assert new_hps["lora_r"] == 32

    def test_overfit_metadata(self, adjuster):
        """Metadata should reflect that overfit corrections were applied."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            is_overfitting=True,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert metadata["overfit_adjustments"] is True


# ═══════════════════════════════════════════════════════════════
# SHARED: Output files & Parser
# ═══════════════════════════════════════════════════════════════


class TestOutputFiles:
    def test_all_output_files_written(self, adjuster):
        result = adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="schedule",
        )
        expected = ["hyperparams_json", "strategy_metadata", "hp_output.json"]
        for fname in expected:
            assert os.path.exists(f"/tmp/{fname}"), f"Missing output: /tmp/{fname}"

        with open(result) as f:
            full = json.load(f)
        assert "input_hyperparams" in full
        assert "output_hyperparams" in full
        assert "metadata" in full
        assert full["metadata"]["strategy"] == "schedule"
        assert full["metadata"]["task_type"] == "detection"

    def test_metadata_tracks_changes(self, adjuster):
        """Changes dict should show from/to for modified params."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 0.001875, "frozen_stages": 1}',
            task_type="detection",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
            unfreeze_on_retry=True,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert "per_item_lrate" in metadata["changes"]
        assert metadata["changes"]["per_item_lrate"]["from"] == 0.001875
        assert metadata["changes"]["per_item_lrate"]["to"] == pytest.approx(0.0009375)


class TestParser:
    def test_parser_builds(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args([
            "--task_type", "detection",
            "--tuning_strategy", "schedule",
            "--lr_decay_factor", "0.5",
        ])
        assert args.task_type == "detection"
        assert args.tuning_strategy == "schedule"
        assert args.lr_decay_factor == 0.5

    def test_bool_parsing(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args(["--unfreeze_on_retry", "true", "--is_overfitting", "false"])
        assert args.unfreeze_on_retry is True
        assert args.is_overfitting is False
