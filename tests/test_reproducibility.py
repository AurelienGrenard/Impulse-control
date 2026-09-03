import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from impulse_control.reproducibility import (
    PUBLISHED_EVALUATION_BATCH_SIZE,
    PUBLISHED_EVALUATION_PATHS,
    PUBLISHED_EVALUATION_SEED,
    PUBLISHED_MIN_REL_IMPULSE,
    policy_evaluation_batches,
    published_horizons,
    published_training_parameters,
    sample_mean_std,
    seed_everything,
    value_diagnostic_seed,
    write_run_manifest,
)


ROOT = Path(__file__).parents[1]


def test_seed_everything_repeats_random_streams():
    """Check that the public seed helper resets every used random stream."""
    seed_everything(1234)
    first = (random.random(), np.random.rand(), torch.rand(4))
    seed_everything(1234)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_run_manifest_records_reproduction_context(tmp_path):
    """Check that each new checkpoint receives an auditable sidecar manifest."""
    target = write_run_manifest(
        str(tmp_path / "example.pt"),
        application="dividend",
        mode="unlimited",
        dimension=6,
        seed=1234,
        device="cpu",
        smoke_test=True,
        randomized_candidates=16,
        horizons=(1.0,),
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["seed"] == 1234
    assert payload["dimension"] == 6
    assert payload["network"]["activation"] == "leaky_relu"
    assert payload["network"]["negative_slope"] == 0.01
    assert payload["randomized_candidates"] == 16
    assert payload["candidate_batch_size"] == 8
    assert payload["min_rel_impulse"] == PUBLISHED_MIN_REL_IMPULSE
    assert payload["policy_evaluation"] == {
        "paths": 2,
        "batch_size": 2,
        "seed": PUBLISHED_EVALUATION_SEED,
        "standard_deviation_ddof": 1,
    }
    assert payload["software"]["pytorch"]


def test_published_policy_evaluation_protocol():
    """Check the archived batching, seeds, and sample-standard-deviation rule."""
    batches = list(policy_evaluation_batches())
    assert len(batches) == 32
    assert batches[0] == (PUBLISHED_EVALUATION_BATCH_SIZE, PUBLISHED_EVALUATION_SEED)
    assert batches[-1] == (8, PUBLISHED_EVALUATION_SEED + 31)
    assert sum(size for size, _ in batches) == PUBLISHED_EVALUATION_PATHS
    assert value_diagnostic_seed(40) == PUBLISHED_EVALUATION_SEED + 100_040
    mean, std = sample_mean_std(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0
    assert np.isclose(std, 1.0)


def test_documented_numerical_protocol_matches_public_constants():
    """Check that the machine-readable configuration matches the implementation."""
    payload = json.loads(
        (ROOT / "reproducibility" / "training-configurations.json").read_text()
    )
    search = payload["intervention_search"]
    assert search["candidate_batch_size"] == 512
    assert search["relative_sparsification_threshold"] == PUBLISHED_MIN_REL_IMPULSE
    learned = payload["policy_evaluation"]["learned_policy"]
    assert learned["d1_paths"] == PUBLISHED_EVALUATION_PATHS
    assert learned["d6_paths"] == PUBLISHED_EVALUATION_PATHS
    assert learned["batch_size"] == PUBLISHED_EVALUATION_BATCH_SIZE
    assert learned["base_seed"] == PUBLISHED_EVALUATION_SEED
    assert learned["standard_deviation_ddof"] == 1


def test_published_unlimited_schedule_matches_archived_components():
    """Check the profiles used by the selected unlimited maturities."""
    dividend_t5 = published_training_parameters("dividend", "unlimited", 6, 5)
    dividend_t10 = published_training_parameters("dividend", "unlimited", 6, 10)
    harvesting_t5 = published_training_parameters("harvesting", "unlimited", 6, 5)
    harvesting_t10 = published_training_parameters("harvesting", "unlimited", 6, 10)
    for profile in (dividend_t5, dividend_t10, harvesting_t5, harvesting_t10):
        assert profile["design_states"] == 15_000
        assert profile["rollouts_per_state"] == 8
        assert profile["randomized_candidates"] == 6_000
        assert profile["candidate_batch_size"] == 512
        assert profile["min_rel_impulse"] == PUBLISHED_MIN_REL_IMPULSE
        assert profile["transfer_steps"] == 500
        assert profile["transfer_lr"] == 5e-4
    assert published_horizons("unlimited", 6) == (5.0, 10.0, 20.0, 40.0)
    assert published_horizons("unlimited", 1) == (5.0, 10.0, 20.0, 40.0)
    d1 = published_training_parameters("dividend", "unlimited", 1, 40)
    assert d1["design_states"] == 120_000
    assert d1["rollouts_per_state"] == 1
    assert d1["randomized_candidates"] == 6_000
    assert d1["transfer_steps"] == 500
    assert d1["transfer_lr"] == 5e-4


def test_figure_map_covers_every_manuscript_panel():
    """Check the article-to-artifact map and all referenced files."""
    with (ROOT / "reproducibility" / "figure-map.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["article_figure"]) for row in rows} == set(range(1, 5))
    assert len({row["output"] for row in rows}) == 16
    for row in rows:
        assert (ROOT / row["checkpoint"]).is_file()
        assert (ROOT / row["output"]).is_file()


def test_seeded_dividend_training_cli_smoke(tmp_path, monkeypatch):
    """Run the seeded dividend CLI through training, saving, and manifest creation."""
    from impulse_control import train_dividend

    outputs = [tmp_path / "dividend_a.pt", tmp_path / "dividend_b.pt"]
    for output in outputs:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_dividend",
                "--mode",
                "limited",
                "--dimension",
                "1",
                "--device",
                "cpu",
                "--seed",
                "1234",
                "--smoke-test",
                "--output",
                str(output),
            ],
        )
        train_dividend.main()
        assert output.is_file()
        manifest = json.loads(output.with_suffix(".json").read_text())
        assert manifest["seed"] == 1234
        assert manifest["network"]["activation"] == "leaky_relu"

    first, second = [torch.load(path, map_location="cpu", weights_only=False) for path in outputs]

    def assert_equal(left, right):
        """Compare serialized scientific content independently of ZIP metadata."""
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_equal(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_equal(left_item, right_item)
        else:
            assert left == right

    assert_equal(first, second)


def test_seeded_harvesting_training_cli_smoke(tmp_path, monkeypatch):
    """Run the seeded harvesting CLI through training, saving, and manifest creation."""
    from impulse_control import train_harvesting

    output = tmp_path / "harvesting.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_harvesting",
            "--mode",
            "limited",
            "--dimension",
            "1",
            "--device",
            "cpu",
            "--seed",
            "1234",
            "--smoke-test",
            "--output",
            str(output),
        ],
    )
    train_harvesting.main()
    assert output.is_file()
    assert json.loads(output.with_suffix(".json").read_text())["seed"] == 1234


def test_harvesting_unlimited_smoke_manifest_matches_checkpoint(tmp_path, monkeypatch):
    """Check that the unlimited smoke manifest records the effective batch size."""
    from impulse_control import train_harvesting
    from impulse_control.saving import load_all_results_unlimited

    output = tmp_path / "harvesting_unlimited.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_harvesting",
            "--mode",
            "unlimited",
            "--dimension",
            "1",
            "--device",
            "cpu",
            "--seed",
            "1234",
            "--smoke-test",
            "--output",
            str(output),
        ],
    )
    train_harvesting.main()

    result = load_all_results_unlimited(str(output), map_location="cpu")[0]
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert result["cfg"].opt.n_global_batch == 8
    assert manifest["candidate_batch_size"] == result["cfg"].opt.n_global_batch
