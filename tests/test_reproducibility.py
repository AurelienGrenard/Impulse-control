import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from impulse_control.reproducibility import (
    published_horizons,
    published_training_parameters,
    seed_everything,
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
        mode="limited",
        dimension=4,
        seed=1234,
        device="cpu",
        smoke_test=True,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["seed"] == 1234
    assert payload["dimension"] == 4
    assert payload["network"]["activation"] == "leaky_relu"
    assert payload["network"]["negative_slope"] == 0.01
    assert payload["randomized_candidates"] == 5_000
    assert payload["software"]["pytorch"]


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
