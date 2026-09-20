"""Reproducibility helpers used by the training entry points."""

from __future__ import annotations

from collections.abc import Iterator
import json
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy
import torch


PUBLISHED_MIN_REL_IMPULSE = None
PUBLISHED_EVALUATION_PATHS = 1_000
PUBLISHED_EVALUATION_BATCH_SIZE = 32
PUBLISHED_EVALUATION_SEED = 20_260_924
PUBLISHED_TRAINING_SEED = 2_345
PUBLISHED_SOURCE_HORIZON = 10.0
PUBLISHED_REPORTING_HORIZONS = (2.0, 4.0, 6.0, 8.0, 10.0)


def published_horizons(mode: str, dimension: int) -> tuple[float, ...]:
    """Return the maturities reported from the published checkpoints."""
    if mode == "limited":
        return (PUBLISHED_SOURCE_HORIZON,)
    if mode != "unlimited":
        raise ValueError(f"Unsupported mode: {mode}")
    return PUBLISHED_REPORTING_HORIZONS


def published_training_horizons(mode: str, dimension: int) -> tuple[float, ...]:
    """Return the horizons that require an independent backward recursion."""
    if mode not in {"limited", "unlimited"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return (PUBLISHED_SOURCE_HORIZON,)


def published_training_parameters(
    application: str,
    mode: str,
    dimension: int,
    horizon: float | None = None,
) -> dict[str, int | float | None]:
    """Return the numerical settings used by a published checkpoint component."""
    if application not in {"dividend", "harvesting"}:
        raise ValueError(f"Unsupported application: {application}")
    if mode not in {"limited", "unlimited"}:
        raise ValueError(f"Unsupported mode: {mode}")
    profiles = {
        1: (100_000, 12_500),
        4: (200_000, 50_000),
        6: (350_000, 75_000),
    }
    if dimension not in profiles:
        raise ValueError(f"No published profile for dimension {dimension}.")
    design_states, randomized_candidates = profiles[dimension]
    parameters: dict[str, int | float | None] = {
        "design_states": design_states,
        "rollouts_per_state": 1,
        "randomized_candidates": randomized_candidates,
        "candidate_batch_size": 512,
        "min_rel_impulse": PUBLISHED_MIN_REL_IMPULSE,
        "coordinate_masks": (1 << dimension) - 1,
        "transfer_steps": 500,
        "transfer_lr": 5e-4,
    }
    return parameters


def value_diagnostic_seed(horizon: float) -> int:
    """Return the archived seed used to evaluate a value function."""
    return PUBLISHED_EVALUATION_SEED + 100_000 + int(horizon)


def policy_evaluation_batches(
    n_paths: int = PUBLISHED_EVALUATION_PATHS,
    batch_size: int = PUBLISHED_EVALUATION_BATCH_SIZE,
    base_seed: int = PUBLISHED_EVALUATION_SEED,
) -> Iterator[tuple[int, int]]:
    """Yield the batch size and seed used by each policy-evaluation batch."""
    if n_paths <= 0 or batch_size <= 0:
        raise ValueError("Policy-evaluation path and batch counts must be positive.")
    for start in range(0, n_paths, batch_size):
        yield min(batch_size, n_paths - start), base_seed + start // batch_size


def sample_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return the mean and sample standard deviation used in the figures."""
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError("At least one policy score is required.")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def seed_everything(seed: int) -> None:
    """Initialize every random-number generator used by the experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_run_manifest(
    checkpoint: str,
    *,
    application: str,
    mode: str,
    dimension: int,
    seed: int,
    device: str,
    smoke_test: bool,
    activation: str = "leaky_relu",
    negative_slope: float = 0.01,
    randomized_candidates: int = 5_000,
    horizons: tuple[float, ...] | None = None,
) -> Path:
    """Write the seed, command parameters, and software versions beside a checkpoint."""
    target = Path(checkpoint).with_suffix(".json")
    payload: dict[str, Any] = {
        "application": application,
        "mode": mode,
        "dimension": dimension,
        "seed": seed,
        "device": device,
        "smoke_test": smoke_test,
        "network": {
            "depth": 3,
            "width": 128,
            "activation": activation,
            "negative_slope": negative_slope,
        },
        "randomized_candidates": randomized_candidates,
        "candidate_batch_size": 8 if smoke_test else 512,
        "min_rel_impulse": PUBLISHED_MIN_REL_IMPULSE,
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    if mode == "unlimited":
        schedule = horizons or published_training_horizons(mode, dimension)
        payload["policy_evaluation"] = {
            "paths": 2 if smoke_test else PUBLISHED_EVALUATION_PATHS,
            "batch_size": 2 if smoke_test else PUBLISHED_EVALUATION_BATCH_SIZE,
            "seed": PUBLISHED_EVALUATION_SEED,
            "standard_deviation_ddof": 1,
        }
        payload["reported_horizons"] = list(PUBLISHED_REPORTING_HORIZONS)
        payload["horizon_parameters"] = {
            f"T={horizon:g}": published_training_parameters(
                application, mode, dimension, horizon
            )
            for horizon in schedule
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
