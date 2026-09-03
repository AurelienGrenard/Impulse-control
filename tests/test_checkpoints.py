from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from impulse_control.plotting import plot_limited_summary, plot_unlimited_summary
from impulse_control.saving import load_all_results_unlimited, load_results_bundle
from impulse_control.reproducibility import published_horizons


RUNS = Path(__file__).parents[1] / "runs"


@pytest.mark.parametrize("application", ["dividend", "harvesting"])
@pytest.mark.parametrize("dimension", [1, 4])
def test_load_limited_checkpoints(application, dimension):
    """Check that every bounded checkpoint loads and renders."""
    path = RUNS / f"{application}_limited_d{dimension}.pt"
    bundle = load_results_bundle(str(path), map_location="cpu")
    assert [result["max_impulses"] for result in bundle["bounded_results"]] == [
        1,
        2,
        3,
        4,
    ]
    assert bundle["unconstrained"] is not None
    figure = plot_limited_summary(bundle, show=False)
    assert figure.axes
    assert tuple(figure.get_size_inches()) == (8.0, 5.0)
    assert figure.axes[0].get_title() == ""
    assert all(result["cfg"].net.activation == "leaky_relu" for result in bundle["bounded_results"])


@pytest.mark.parametrize("application", ["dividend", "harvesting"])
@pytest.mark.parametrize("dimension", [1, 6])
def test_load_unlimited_checkpoints(application, dimension):
    """Check that every unconstrained checkpoint loads and renders."""
    path = RUNS / f"{application}_unlimited_d{dimension}.pt"
    results = load_all_results_unlimited(str(path), map_location="cpu")
    assert results
    assert all(result["qhats"] for result in results)
    assert [result["T"] for result in results] == list(
        published_horizons("unlimited", dimension)
    )
    assert all(result.get("mc_n_NN") == 1_000 for result in results)
    assert all(result["cfg"].net.activation == "leaky_relu" for result in results)
    figure = plot_unlimited_summary(results, show=False)
    assert figure.axes
    assert tuple(figure.get_size_inches()) == (8.0, 5.0)
    assert figure.axes[0].get_title() == ""
    assert figure.axes[0].get_lines()[0].get_label() == "Closed-form"
    assert len(figure.axes[0].get_lines()) == len(results) + 1
