"""Rebuild the modular sources and result notebooks from the original experiments."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Impulse-control-main"
TARGET = ROOT / "Impulse-control-project"
PACKAGE = TARGET / "impulse_control"


def cell(path: Path, index: int) -> str:
    """Build a notebook cell."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join(notebook["cells"][index]["source"])


def without_definitions(source: str, names: set[str]) -> str:
    """Remove selected top-level definitions from extracted source."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    removed: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            start = node.lineno
            if node.decorator_list:
                start = min(item.lineno for item in node.decorator_list)
            removed.update(range(start - 1, node.end_lineno))
    return "".join(line for index, line in enumerate(lines) if index not in removed)


def module_header(exact_module: str) -> str:
    """Build a generated module header."""
    return f'''"""Numerical algorithms extracted from the original experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .common import OptConfig, TimeGridConfig
from .{exact_module} import *

Tensor = torch.Tensor

'''


def write_sources() -> None:
    """Generate source modules from the original notebooks."""
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "__init__.py").write_text(
        '"""Neural regression Monte Carlo for impulse control."""\n',
        encoding="utf-8",
    )

    pairs = {
        "dividend": SOURCE / "Dividend" / "Dividends(limited impulses, d=4).ipynb",
        "harvesting": SOURCE / "Harvesting" / "Harvesting(limited impulses, d=4).ipynb",
    }
    for name, notebook in pairs.items():
        exact = cell(notebook, 3)
        (PACKAGE / f"exact_{name}.py").write_text(
            module_header("common") + exact,
            encoding="utf-8",
        )

        numerical = "\n\n".join(cell(notebook, i) for i in (5, 7, 9, 11, 13))
        numerical = without_definitions(numerical, {"OptConfig", "TimeGridConfig"})
        (PACKAGE / f"{name}.py").write_text(
            module_header(f"exact_{name}") + numerical,
            encoding="utf-8",
        )


def notebook(title: str, application: str, model: str, limited: bool) -> dict:
    """Build a compact result notebook."""
    result_key = "bounded_results" if limited else None
    figure_prefix = Path(model).stem
    command = (
        f"python -m impulse_control.train_{application} "
        f"--mode {'limited' if limited else 'unlimited'} "
        f"--dimension <DIMENSION> --output runs/{application}_"
        f"{'limited' if limited else 'unlimited'}_d<DIMENSION>.pt"
    )
    path_seed = 124 if model == "dividend_limited_d1.pt" else 123
    load = (
        "from impulse_control.saving import "
        + ("load_results_bundle" if limited else "load_all_results_unlimited")
        + "\n\n"
        + (
            f'bundle = load_results_bundle("runs/{model}", map_location="cpu")\n'
            f'all_results_loaded = bundle["{result_key}"]'
            if limited
            else f'all_results_loaded = load_all_results_unlimited("runs/{model}", map_location="cpu")'
        )
    )
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n\n",
                "Load a trained impulse-control model and inspect its numerical results.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Train your own model\n\n",
                "Run the following command from the project root:\n\n",
                f"```bash\n{command}\n```\n\n",
                "The training script saves the checkpoint specified by `--output`. The supplied pretrained models can be loaded directly without retraining.",
            ],
        },
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": load.splitlines(keepends=True)},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Value functions\n\n",
                (
                    "Compare each finite impulse budget with the unconstrained value function."
                    if limited
                    else "Compare finite-horizon neural values with the closed-form infinite-horizon solution."
                ),
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                (
                    "from impulse_control.plotting import plot_limited_summary\n\n"
                    f'_ = plot_limited_summary(bundle, output="figures/{figure_prefix}_value_functions.png")'
                )
                if limited
                else (
                    "from impulse_control.plotting import plot_unlimited_summary\n\n"
                    f'_ = plot_unlimited_summary(all_results_loaded, output="figures/{figure_prefix}_value_functions.png")'
                )
            ).splitlines(keepends=True),
        },
    ]
    if limited:
        cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Controlled paths\n\n",
                        "Compare impulse budgets on the same Brownian realization. "
                        "Vertical jumps mark interventions.",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": (
                        "from impulse_control.plotting import plot_limited_paths\n\n"
                        f'_ = plot_limited_paths(bundle, seed={path_seed}, '
                        f'output_prefix="figures/{figure_prefix}")'
                    ).splitlines(keepends=True),
                },
            ]
        )
    else:
        cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Policy consistency\n\n",
                        "Compare Monte Carlo scores under the learned and exact band policies "
                        "across horizons. Error bars show standard errors.",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": (
                        "from impulse_control.plotting import plot_unlimited_policy_consistency\n\n"
                        f'_ = plot_unlimited_policy_consistency(all_results_loaded, '
                        f'output="figures/{figure_prefix}_policy_consistency.png")'
                    ).splitlines(keepends=True),
                },
            ]
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebooks() -> None:
    """Generate every compact result notebook."""
    folder = TARGET / "notebooks"
    folder.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "dividend_limited.ipynb",
        "dividend_unlimited.ipynb",
        "harvesting_limited.ipynb",
        "harvesting_unlimited.ipynb",
    ):
        legacy_path = folder / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    specs = [
        ("dividend_limited_d1.ipynb", "Dividend — limited impulses ($d=1$)", "dividend", "dividend_limited_d1.pt", True),
        ("dividend_limited_d4.ipynb", "Dividend — limited impulses ($d=4$)", "dividend", "dividend_limited_d4.pt", True),
        ("dividend_unlimited_d1.ipynb", "Dividend — unlimited impulses ($d=1$)", "dividend", "dividend_unlimited_d1.pt", False),
        ("dividend_unlimited_d6.ipynb", "Dividend — unlimited impulses ($d=6$)", "dividend", "dividend_unlimited_d6.pt", False),
        ("harvesting_limited_d1.ipynb", "Harvesting — limited impulses ($d=1$)", "harvesting", "harvesting_limited_d1.pt", True),
        ("harvesting_limited_d4.ipynb", "Harvesting — limited impulses ($d=4$)", "harvesting", "harvesting_limited_d4.pt", True),
        ("harvesting_unlimited_d1.ipynb", "Harvesting — unlimited impulses ($d=1$)", "harvesting", "harvesting_unlimited_d1.pt", False),
        ("harvesting_unlimited_d6.ipynb", "Harvesting — unlimited impulses ($d=6$)", "harvesting", "harvesting_unlimited_d6.pt", False),
    ]
    for filename, title, app, model, limited in specs:
        (folder / filename).write_text(
            json.dumps(notebook(title, app, model, limited), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _indent(source: str, spaces: int = 4) -> str:
    """Indent generated source text."""
    prefix = " " * spaces
    return "".join(prefix + line if line.strip() else line for line in source.splitlines(keepends=True))


def write_trainers() -> None:
    """Generate the dividend and harvesting training entry points."""
    for application, folder, noun in (
        ("dividend", "Dividend", "Dividends"),
        ("harvesting", "Harvesting", "Harvesting"),
    ):
        limited_nb = SOURCE / folder / f"{noun}(limited impulses, d=4).ipynb"
        unlimited_nb = SOURCE / folder / f"{noun}(unlimited impulses, d=6).ipynb"
        limited = cell(limited_nb, 17)
        unlimited = cell(unlimited_nb, 17)
        for source_name in ("limited", "unlimited"):
            source = limited if source_name == "limited" else unlimited
            source = source.replace(
                'device = "cuda:0" if torch.cuda.is_available() else "cpu"',
                "device = requested_device",
                1,
            )
            source = source.replace("state_dim = 4", "state_dim = dimension", 1)
            source = source.replace("state_dim = 6", "state_dim = dimension", 1)
            source = source.replace("d = 6", "d = dimension", 1)
            source = source.replace("d_state = 6", "d_state = dimension", 1)
            smoke_overrides = """
if smoke_test:
    steps = 2
    transfer_steps = 1
    batch_size = 32
    n_global = 16
    n_global_batch = 8
    N_k = 16
    M_k = 2
"""
            marker = "M_k = 8"
            marker_line = next(line for line in source.splitlines(keepends=True) if line.lstrip().startswith(marker))
            source = source.replace(marker_line, marker_line + smoke_overrides, 1)
            if source_name == "limited":
                source = source.replace(
                    smoke_overrides,
                    smoke_overrides + "    max_imp_list = [1]\n    n_sim_eval = 2\n    dt_fine_eval = 0.1\n",
                    1,
                )
            else:
                source = source.replace(
                    smoke_overrides,
                    smoke_overrides + "    T_list = [1.0]\n    n_sim_eval = 2\n    dt_fine_eval = 0.1\n",
                    1,
                )
            if source_name == "limited":
                source += (
                    f'\n\nsave_results_bundle(output, bounded_results=all_results, '
                    "unconstrained=res_inf_nd)\n"
                )
            else:
                source += "\n\nsave_all_results_unlimited(output, all_results)\n"
            if source_name == "limited":
                limited = source
            else:
                unlimited = source

        text = f'''"""Command-line training entry point for the {application} example."""

from __future__ import annotations

import argparse
import torch

from .{application} import *
from .saving import save_all_results_unlimited, save_results_bundle


def train_limited(dimension: int, output: str, requested_device: str, smoke_test: bool = False) -> None:
{_indent(limited)}


def train_unlimited(dimension: int, output: str, requested_device: str, smoke_test: bool = False) -> None:
{_indent(unlimited)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("limited", "unlimited"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run the same pipeline with tiny validation sizes.")
    args = parser.parse_args()
    trainer = train_limited if args.mode == "limited" else train_unlimited
    trainer(args.dimension, args.output, args.device, args.smoke_test)


if __name__ == "__main__":
    main()
'''
        (PACKAGE / f"train_{application}.py").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    write_sources()
    write_notebooks()
    write_trainers()
