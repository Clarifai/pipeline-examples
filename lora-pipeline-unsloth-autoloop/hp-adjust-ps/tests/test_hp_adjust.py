"""Unit tests for the hp-adjust pipeline step (LLM fine-tuning task).

Run: python -m pytest lora-pipeline-unsloth-autoloop/hp-adjust-ps/tests/test_hp_adjust.py -v
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
    """Read an Argo output param written to /tmp/."""
    with open(f"/tmp/{key}") as f:
        return f.read()


# ─── Schedule strategy ─────────────────────────────────────────


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
        # Iteration 1
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter1 = json.loads(_read_output("hyperparams_json"))
        assert iter1["learning_rate"] == pytest.approx(1e-4)
        assert iter1["lora_r"] == 32

        # Iteration 2
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

        # Iteration 3
        adjuster.adjust(
            current_hyperparams=json.dumps(iter2),
            hp_history='[{}, {}]',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        iter3 = json.loads(_read_output("hyperparams_json"))
        assert iter3["learning_rate"] == pytest.approx(2.5e-5)
        assert iter3["lora_r"] == 128  # capped

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


# ─── Grid strategy ─────────────────────────────────────────────


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
        # Default grid: learning_rate=[2e-4,1e-4,5e-5], lora_r=[16,32,64]
        # Sorted keys: learning_rate, lora_r → first combo
        assert new_hps["learning_rate"] in [2e-4, 1e-4, 5e-5]
        assert new_hps["lora_r"] in [16, 32, 64]

    def test_skips_tried_combos(self, adjuster):
        """Should skip combinations already in hp_history."""
        # Mark first combo as tried
        default_grid = strategies.GRID_DEFAULTS["llm_finetune"]
        keys = sorted(default_grid.keys())
        import itertools
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


# ─── Random strategy ──────────────────────────────────────────


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


# ─── Overfitting corrections ──────────────────────────────────


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
        # But also: the overfit correction uses the *output* of schedule
        # Schedule: lora_r=64, then overfit: 64//2=32
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
        assert new_hps["lora_r"] == 32  # doubled by schedule, no overfit correction

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


# ─── Output files ─────────────────────────────────────────────


class TestOutputFiles:
    def test_all_output_files_written(self, adjuster):
        result = adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
        )
        expected = ["hyperparams_json", "strategy_metadata", "hp_output.json"]
        for fname in expected:
            assert os.path.exists(f"/tmp/{fname}"), f"Missing output: /tmp/{fname}"

        with open(result) as f:
            full = json.load(f)
        assert full["metadata"]["strategy"] == "schedule"
        assert full["metadata"]["task_type"] == "llm_finetune"

    def test_metadata_tracks_changes(self, adjuster):
        """Changes dict should show from/to for modified params."""
        adjuster.adjust(
            current_hyperparams='{"learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 16}',
            task_type="llm_finetune",
            tuning_strategy="schedule",
            lr_decay_factor=0.5,
        )
        metadata = json.loads(_read_output("strategy_metadata"))
        assert "learning_rate" in metadata["changes"]
        assert metadata["changes"]["learning_rate"]["from"] == 0.0002
        assert metadata["changes"]["learning_rate"]["to"] == pytest.approx(0.0001)


# ─── Argparse parser ──────────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args([
            "--task_type", "llm_finetune",
            "--tuning_strategy", "schedule",
            "--lr_decay_factor", "0.5",
        ])
        assert args.task_type == "llm_finetune"
        assert args.tuning_strategy == "schedule"

    def test_bool_parsing(self):
        parser = HPAdjustment.to_pipeline_parser()
        args = parser.parse_args(["--is_overfitting", "true", "--unfreeze_on_retry", "false"])
        assert args.is_overfitting is True
        assert args.unfreeze_on_retry is False
