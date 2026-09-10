"""Regenerate every figure file used by the paper from supplied checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impulse_control.plotting import (
    plot_limited_paths,
    plot_limited_summary,
    plot_unlimited_policy_consistency,
    plot_unlimited_summary,
)
from impulse_control.saving import load_all_results_unlimited, load_results_bundle


EXPERIMENTS = (
    "dividend_limited_d1",
    "dividend_limited_d4",
    "harvesting_limited_d1",
    "dividend_unlimited_d1",
    "harvesting_unlimited_d1",
    "dividend_unlimited_d6",
    "harvesting_unlimited_d6",
)
LIMITED_VALUE_EXPERIMENTS = {"dividend_limited_d1", "harvesting_limited_d1"}
LIMITED_PATH_EXPERIMENTS = {
    "dividend_limited_d1",
    "dividend_limited_d4",
    "harvesting_limited_d1",
}


def policy_comparison_rows(application: str, dimension: int):
    """Load the common-path annual-grid policy evaluation used in the paper."""
    path = ROOT / "reproducibility" / "policy-evaluation.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row
        for row in rows
        if row["problem"] == application and int(row["d"]) == dimension
    ]


def reproduce_one(stem: str, output_dir: Path) -> None:
    """Regenerate the figures associated with one checkpoint."""
    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    application, mode, dimension_text = stem.split("_")
    dimension = int(dimension_text[1:])
    checkpoint = ROOT / "runs" / f"{stem}.pt"
    if mode == "limited":
        bundle = load_results_bundle(str(checkpoint), map_location="cpu")
        count = 0
        if stem in LIMITED_VALUE_EXPERIMENTS:
            plot_limited_summary(
                bundle,
                output=str(output_dir / f"{stem}_value_functions.png"),
                show=False,
            )
            plt.close("all")
            count += 1
        if stem in LIMITED_PATH_EXPERIMENTS:
            seed = 124 if stem == "dividend_limited_d1" else 123
            figures = plot_limited_paths(
                bundle,
                seed=seed,
                output_prefix=str(output_dir / stem),
                show=False,
            )
            count += len(figures)
    else:
        results = load_all_results_unlimited(str(checkpoint), map_location="cpu")
        plot_unlimited_summary(
            results,
            output=str(output_dir / f"{stem}_value_functions.png"),
            show=False,
        )
        plt.close("all")
        plot_unlimited_policy_consistency(
            results,
            comparison_rows=policy_comparison_rows(application, dimension),
            output=str(output_dir / f"{stem}_policy_consistency.png"),
            show=False,
        )
        count = 2
    plt.close("all")
    print(f"generated {application} {mode} d={dimension}: {count} figure(s)")


def reproduce(output_dir: Path) -> None:
    """Regenerate all figures while releasing each checkpoint before the next."""
    for stem in EXPERIMENTS:
        reproduce_one(stem, output_dir)
        gc.collect()


def main() -> None:
    """Parse the output directory and regenerate the paper figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args()
    if args.experiment:
        reproduce_one(args.experiment, args.output_dir.resolve())
    else:
        reproduce(args.output_dir.resolve())


if __name__ == "__main__":
    main()
