"""Build the reported maturities from one time-homogeneous T=10 recursion."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

import numpy as np
import torch

from .common import TimeGridConfig
from .reproducibility import (
    PUBLISHED_EVALUATION_BATCH_SIZE,
    PUBLISHED_EVALUATION_PATHS,
    PUBLISHED_EVALUATION_SEED,
    PUBLISHED_REPORTING_HORIZONS,
    policy_evaluation_batches,
    sample_mean_std,
    seed_everything,
    value_diagnostic_seed,
)


def build_reported_results(
    source: dict[str, Any],
    *,
    module: ModuleType,
    application: str,
    x_max: float,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    """Reuse the last continuation networks for each reported maturity."""
    cfg_source = source["cfg"]
    source_steps = int(cfg_source.time.K)
    horizons = (1.0,) if smoke_test else PUBLISHED_REPORTING_HORIZONS
    n_paths = 2 if smoke_test else PUBLISHED_EVALUATION_PATHS
    batch_size = 2 if smoke_test else PUBLISHED_EVALUATION_BATCH_SIZE
    dt_fine = 0.1 if smoke_test else 2e-3
    score_key = "rewards" if application == "dividend" else "costs"
    results: list[dict[str, Any]] = []

    for horizon in horizons:
        steps = int(horizon)
        if steps > source_steps:
            raise ValueError("A reported horizon cannot exceed the trained horizon.")
        cfg = replace(
            cfg_source,
            time=TimeGridConfig(T=float(horizon), K=steps, dt_fine=1e-2),
            horizon=steps,
        )
        start = source_steps - steps
        qhats = source["qhats"][start:source_steps] + [None]
        x = np.linspace(0.0, x_max, 401)
        diagonal = torch.as_tensor(x, device=cfg.device, dtype=cfg.dtype)[:, None]
        diagonal = diagonal.expand(-1, int(getattr(cfg, application).state_dim))

        seed_everything(value_diagnostic_seed(horizon))
        with torch.inference_mode():
            values = module.vhat_unconstrained(
                k=0,
                x=diagonal,
                qhats=qhats,
                problem=source["problem"],
                opt_cfg=cfg.opt,
                t_grid=cfg.t_grid,
            ).detach().cpu().numpy()

        score_batches = []
        dimension = int(getattr(cfg, application).state_dim)
        for current_batch, batch_seed in policy_evaluation_batches(
            n_paths, batch_size, PUBLISHED_EVALUATION_SEED
        ):
            noise = module.precompute_euler_noise(
                t_grid=cfg.t_grid,
                n_sim=current_batch,
                state_dim=dimension,
                dt_fine=dt_fine,
                seed=batch_seed,
                device=torch.device(cfg.device),
                dtype=cfg.dtype,
            )
            initial = torch.ones(
                (current_batch, dimension), device=cfg.device, dtype=cfg.dtype
            )
            seed_everything(batch_seed)
            simulation = module.simulate_controlled_paths(
                problem=source["problem"],
                stepper=module.EulerStepper(),
                policy=qhats,
                t_grid=cfg.t_grid,
                x0=initial,
                opt_cfg=cfg.opt,
                n_sim=current_batch,
                n_display=0,
                dt_fine=dt_fine,
                noise_pack=noise,
            )
            score_batches.append(simulation[score_key].detach().cpu().numpy())
        mean, std = sample_mean_std(np.concatenate(score_batches))

        x0_value = float(np.interp(1.0, x, np.asarray(values).reshape(-1)))
        results.append(
            {
                "T": float(horizon),
                "cfg": cfg,
                "problem": source["problem"],
                "t_grid": cfg.t_grid,
                "qhats": qhats,
                "x": x,
                "V": values,
                "V0_1": x0_value,
                "mc_mean_NN": mean,
                "mc_std_NN": std,
                "mc_n_NN": n_paths,
                "mc_seed_NN": PUBLISHED_EVALUATION_SEED,
                "mc_batch_size_NN": batch_size,
                "source_horizon": float(cfg_source.time.T),
                "source_date_indices": list(range(start, source_steps)),
            }
        )
    return results
