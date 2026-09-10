"""Evaluate learned and band policies on the same annual decision grid."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch

from impulse_control.exact_dividend import solve_psi_dividend
from impulse_control.exact_harvesting import solve_psi_general
from impulse_control.reproducibility import policy_evaluation_batches, seed_everything
from impulse_control.saving import load_all_results_unlimited


def _scalar(value) -> float:
    """Read the first component of a scalar or componentwise parameter."""
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(value)


def _summary(values: np.ndarray) -> tuple[float, float]:
    """Return the sample mean and its 95% Monte Carlo half-width."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, half_width


@torch.inference_mode()
def _dividend_annual_band_batch(result, noise_pack: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the dividend band only at the checkpoint decision dates."""
    cfg = result["cfg"]
    params = cfg.dividend
    device = noise_pack["noise"].device
    dtype = noise_pack["noise"].dtype
    dimension = int(params.state_dim)
    mu = _scalar(params.mu)
    sigma = _scalar(params.sigma)
    rho = _scalar(params.rho)
    lam = _scalar(params.lam)
    fixed_cost = _scalar(params.c)
    _, _, band = solve_psi_dividend(
        mu=mu,
        sigma=sigma,
        rho=rho,
        lam=lam,
        c=fixed_cost,
    )
    x_hat = float(band["x_hat"])
    x_star = float(band["x_star"])

    grid = result["t_grid"]
    segment_counts = noise_pack["seg_n_list"].tolist()
    noise = noise_pack["noise"]
    batch_size = noise.shape[0]
    state = torch.ones((batch_size, dimension), device=device, dtype=dtype)
    alive = torch.ones_like(state, dtype=torch.bool)
    rewards = torch.zeros(batch_size, device=device, dtype=dtype)
    offset = 0

    for k, segment_count in enumerate(segment_counts):
        t0 = float(grid[k].item())
        t1 = float(grid[k + 1].item())
        dt = (t1 - t0) / segment_count

        active = alive & (state >= x_star)
        impulse = torch.where(
            active,
            torch.clamp((state - x_hat - fixed_cost) / (1.0 + lam), min=0.0),
            torch.zeros_like(state),
        )
        rewards += math.exp(-rho * t0) * impulse.sum(dim=1)
        state = torch.where(
            active,
            torch.as_tensor(x_hat, device=device, dtype=dtype),
            state,
        )

        increments = (
            mu * dt
            + sigma
            * math.sqrt(dt)
            * noise[:, offset : offset + segment_count]
        )
        free_path = state[:, None, :] + torch.cumsum(increments, dim=1)
        alive_path = torch.cumprod((free_path > 0).to(torch.int8), dim=1).bool()
        alive_path &= alive[:, None, :]
        state = torch.where(
            alive_path[:, -1],
            free_path[:, -1],
            torch.zeros_like(state),
        )
        alive = alive_path[:, -1]
        offset += segment_count

    terminal_impulse = torch.clamp(
        (state - fixed_cost) / (1.0 + lam),
        min=0.0,
    )
    rewards += math.exp(-rho * float(grid[-1].item())) * terminal_impulse.sum(dim=1)
    return rewards


@torch.inference_mode()
def _harvesting_annual_band_batch(result, noise_pack: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the harvesting band only at the checkpoint decision dates."""
    cfg = result["cfg"]
    params = cfg.harvesting
    device = noise_pack["noise"].device
    dtype = noise_pack["noise"].dtype
    dimension = int(params.state_dim)
    mu = _scalar(params.mu)
    sigma = _scalar(params.sigma)
    rho = _scalar(params.rho)
    alpha = _scalar(params.alpha)
    target = _scalar(params.x0)
    lam = _scalar(params.lam)
    fixed_cost = _scalar(params.c)
    _, _, band = solve_psi_general(
        mu=mu,
        sigma=sigma,
        rho=rho,
        alpha=alpha,
        x0=target,
        lam=lam,
        c=fixed_cost,
    )
    x_hat = float(band["x_hat"])
    x_star = float(band["x_star"])

    grid = result["t_grid"]
    segment_counts = noise_pack["seg_n_list"].tolist()
    noise = noise_pack["noise"]
    batch_size = noise.shape[0]
    state = torch.ones((batch_size, dimension), device=device, dtype=dtype)
    costs = torch.zeros(batch_size, device=device, dtype=dtype)
    offset = 0

    for k, segment_count in enumerate(segment_counts):
        t0 = float(grid[k].item())
        t1 = float(grid[k + 1].item())
        dt = (t1 - t0) / segment_count

        active = state >= x_star
        impulse = torch.where(
            active,
            torch.clamp(state - x_hat, min=0.0),
            torch.zeros_like(state),
        )
        impulse_cost = active.to(dtype) * (fixed_cost + lam * impulse)
        costs += math.exp(-rho * t0) * impulse_cost.sum(dim=1)
        state = torch.where(
            active,
            torch.as_tensor(x_hat, device=device, dtype=dtype),
            state,
        )

        increments = (
            mu * dt
            + sigma
            * math.sqrt(dt)
            * noise[:, offset : offset + segment_count]
        )
        previous_increments = torch.cat(
            [
                torch.zeros(
                    (batch_size, 1, dimension),
                    device=device,
                    dtype=dtype,
                ),
                torch.cumsum(increments[:, :-1], dim=1),
            ],
            dim=1,
        )
        states_before_step = state[:, None, :] + previous_increments
        times = t0 + torch.arange(segment_count, device=device, dtype=dtype) * dt
        discounts = torch.exp(-rho * times)
        running_costs = alpha * (states_before_step - target).square().sum(dim=2)
        costs += (running_costs * discounts[None, :]).sum(dim=1) * dt
        state += increments.sum(dim=1)
        offset += segment_count

    return costs


def _application_api(application: str):
    """Return the simulation functions and score key for one example."""
    if application == "dividend":
        from impulse_control.dividend import (
            EulerStepper,
            precompute_euler_noise,
            simulate_controlled_paths,
        )

        return (
            EulerStepper,
            precompute_euler_noise,
            simulate_controlled_paths,
            _dividend_annual_band_batch,
            "rewards",
        )

    from impulse_control.harvesting import (
        EulerStepper,
        precompute_euler_noise,
        simulate_controlled_paths,
    )

    return (
        EulerStepper,
        precompute_euler_noise,
        simulate_controlled_paths,
        _harvesting_annual_band_batch,
        "costs",
    )


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    application: str,
    dimension: int,
    device: str = "cuda",
    dt_fine: float = 2e-3,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Evaluate both policies on common paths and return paired statistics."""
    device = str(torch.device(device))
    results = load_all_results_unlimited(str(checkpoint), map_location=device)
    (
        stepper_type,
        precompute_noise,
        simulate_learned,
        simulate_band,
        score_key,
    ) = _application_api(application)
    rows = []

    for result in results:
        horizon = float(result["T"])
        n_paths = int(result.get("mc_n_NN", 1_000))
        batch_size = int(result.get("mc_batch_size_NN", 32))
        base_seed = int(result.get("mc_seed_NN", 20_260_830))
        learned_parts = []
        band_parts = []
        started = time.perf_counter()

        for batch_number, (batch_n, seed) in enumerate(
            policy_evaluation_batches(n_paths, batch_size, base_seed),
            start=1,
        ):
            noise_pack = precompute_noise(
                t_grid=result["t_grid"],
                n_sim=batch_n,
                state_dim=dimension,
                dt_fine=dt_fine,
                seed=seed,
                device=torch.device(device),
                dtype=result["cfg"].dtype,
            )
            initial_state = torch.ones(
                (batch_n, dimension),
                device=device,
                dtype=result["cfg"].dtype,
            )
            seed_everything(seed)
            learned = simulate_learned(
                problem=result["problem"],
                stepper=stepper_type(),
                policy=result["qhats"],
                t_grid=result["t_grid"],
                x0=initial_state,
                opt_cfg=result["cfg"].opt,
                n_sim=batch_n,
                n_display=0,
                dt_fine=dt_fine,
                noise_pack=noise_pack,
            )[score_key]
            band = simulate_band(result, noise_pack)
            learned_parts.append(learned.detach().cpu().numpy())
            band_parts.append(band.detach().cpu().numpy())

            if batch_number % 8 == 0 or sum(part.size for part in learned_parts) == n_paths:
                completed = sum(part.size for part in learned_parts)
                progress(
                    f"{application} d={dimension}, T={horizon:g}: "
                    f"{completed}/{n_paths} paths"
                )

        learned_scores = np.concatenate(learned_parts)
        band_scores = np.concatenate(band_parts)
        learned_mean, learned_half_width = _summary(learned_scores)
        band_mean, band_half_width = _summary(band_scores)
        if application == "dividend":
            advantage = learned_scores - band_scores
        else:
            advantage = band_scores - learned_scores
        advantage_mean, advantage_half_width = _summary(advantage)
        archived_mean = float(result["mc_mean_NN"])
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "problem": application,
                "d": dimension,
                "T": int(horizon),
                "learned": learned_mean,
                "learned_ci95": learned_half_width,
                "annual_band": band_mean,
                "annual_band_ci95": band_half_width,
                "learned_advantage": advantage_mean,
                "advantage_ci95": advantage_half_width,
                "archived_learned": archived_mean,
                "rerun_minus_archived": learned_mean - archived_mean,
                "n_paths": n_paths,
                "elapsed_seconds": elapsed,
            }
        )
        progress(
            f"completed T={horizon:g} in {elapsed:.1f}s; "
            f"learned advantage={advantage_mean:.4f} ± {advantage_half_width:.4f}"
        )

    del results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def evaluate_all(
    root: str | Path,
    *,
    device: str | None = None,
    dt_fine: float = 2e-3,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Run the four unlimited experiments used in Figures 1 and 4."""
    root = Path(root)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = []
    for application, dimension in (
        ("dividend", 1),
        ("harvesting", 1),
        ("dividend", 6),
        ("harvesting", 6),
    ):
        checkpoint = root / "runs" / f"{application}_unlimited_d{dimension}.pt"
        progress(f"\nLoading {checkpoint.name} on {device}")
        frames.append(
            evaluate_checkpoint(
                checkpoint,
                application=application,
                dimension=dimension,
                device=device,
                dt_fine=dt_fine,
                progress=progress,
            )
        )
    return pd.concat(frames, ignore_index=True)
