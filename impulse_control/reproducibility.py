"""Reproducibility helpers used by the training entry points."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy
import torch


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
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
