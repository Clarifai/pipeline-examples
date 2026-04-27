"""Unit tests for the hp-adjust pipeline step (detection task).

Run: python -m pytest detector-pipeline-yolof-autoloop/hp-adjust-ps/tests/test_hp_adjust.py -v
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

# Also import strategies directly for isolated strategy tests
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
    """Read an Argo output param written to /tmp/."""
    with open(f"/tmp/{key}") as f:
        return f.read()


# ─── Schedule strategy ─────────────────────────────────────────


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
        # Iteration 1
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

        # Iteration 2 — feed output back as input
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


# ─── Grid strategy ─────────────────────────────────────────────


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
        # Default grid: frozen_stages=[0,1], per_item_lrate=[0.001875,0.0009375,0.003]
        # First combo (sorted keys): frozen_stages=0, per_item_lrate=0.001875
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
        # Should NOT be the first combo again
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
        # Default detection grid: 2 frozen_stages × 3 LR = 6 combos
        default_grid = strategies.GRID_DEFAULTS["detection"]
        keys = sorted(default_grid.keys())
        import itertools
        all_combos = list(itertools.product(*[default_grid[k] for k in keys]))
        # Mark all as tried
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
        # Should wrap around to the first combo
        assert new_hps["frozen_stages"] == all_combos[0][0]
        assert new_hps["per_item_lrate"] == all_combos[0][1]


# ─── Random strategy ──────────────────────────────────────────


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


# ─── Overfitting corrections ──────────────────────────────────


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


# ─── Output files ─────────────────────────────────────────────


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


# ─── Argparse parser ──────────────────────────────────────────


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
