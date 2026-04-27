"""Unit tests for the hp-adjust pipeline step (classification task).

Run: python -m pytest classifier-pipeline-resnet-autoloop/hp-adjust-ps/tests/test_hp_adjust.py -v
"""

import json
import os
import sys

import pytest

# Add model.py to path using the same __import__ trick as pipeline_step.py
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models"
    ),
)
_mod = __import__("model.1.model", fromlist=[""])
HPAdjustment = _mod.HPAdjustment

# Also import strategies directly for isolated tests
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "1", "models", "model", "1"
    ),
)
import strategies


@pytest.fixture
def adjuster():
    return HPAdjustment()


def _read_output(key):
    with open(f"/tmp/{key}") as f:
        return f.read()


# ─── Schedule strategy ─────────────────────────────────────────


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
        # Iteration 1
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter1 = json.loads(_read_output("hyperparams_json"))
        assert iter1["per_item_lrate"] == pytest.approx(1.95e-5 * 0.5)
        assert iter1["weight_decay"] == pytest.approx(0.015)

        # Iteration 2
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

        # Iteration 3
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


# ─── Grid strategy ─────────────────────────────────────────────


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
        default_grid = strategies.GRID_DEFAULTS["classification"]
        keys = sorted(default_grid.keys())
        import itertools
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
        # 3 lr * 2 wd = 6 combos total; mark all 6 as tried
        default_grid = strategies.GRID_DEFAULTS["classification"]
        keys = sorted(default_grid.keys())
        import itertools
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
        # Wraps around to first combo
        assert new_hps["per_item_lrate"] in [1.95e-5, 1e-5, 5e-5]
        assert new_hps["weight_decay"] in [0.01, 0.05]


# ─── Random strategy ──────────────────────────────────────────


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


# ─── Overfitting corrections ──────────────────────────────────


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
        # Schedule: 0.05*1.5=0.075, overfit: 0.075*3=0.225 → capped at 0.1
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
        # No overfit: weight_decay only from schedule (0.015), epochs untouched
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


# ─── Output files ─────────────────────────────────────────────


class TestOutputFiles:
    def test_all_output_files_written(self, adjuster):
        result = adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
        )
        expected = ["hyperparams_json", "strategy_metadata", "hp_output.json"]
        for fname in expected:
            assert os.path.exists(f"/tmp/{fname}"), f"Missing output: /tmp/{fname}"

        with open(result) as f:
            full = json.load(f)
        assert full["metadata"]["strategy"] == "schedule"
        assert full["metadata"]["task_type"] == "classification"

    def test_metadata_tracks_changes(self, adjuster):
        """Changes dict should show from/to for modified params."""
        adjuster.adjust(
            current_hyperparams='{"per_item_lrate": 1.95e-5, "weight_decay": 0.01}',
            task_type="classification",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert "per_item_lrate" in metadata["changes"]
        assert metadata["changes"]["per_item_lrate"]["from"] == pytest.approx(1.95e-5)
        assert metadata["changes"]["per_item_lrate"]["to"] == pytest.approx(1.95e-5 * 0.5)


# ─── Argparse parser ──────────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args([
            "--task_type", "classification",
            "--tuning_strategy", "schedule",
            "--lr_decay_factor", "0.5",
        ])
        assert args.task_type == "classification"
        assert args.tuning_strategy == "schedule"

    def test_bool_parsing(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args(["--is_overfitting", "true", "--unfreeze_on_retry", "false"])
        assert args.is_overfitting is True
        assert args.unfreeze_on_retry is False
