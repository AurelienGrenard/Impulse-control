"""Closed-form reference solution and band-policy simulation for harvesting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .common import OptConfig, TimeGridConfig
from .common import *

Tensor = torch.Tensor

torch.set_default_dtype(torch.float64)
# ------------------------- helpers: coefficients -------------------------

def _coefficients(mu: float, sigma: float, rho: float, alpha: float, x0: float):
    """Compute A, B, C, r1 for the general case."""
    assert rho > 0.0 and sigma > 0.0 and alpha > 0.0, "Require rho>0, sigma>0, alpha>0."
    A = alpha / rho
    B = (2.0 * alpha / (rho * rho)) * (mu - rho * x0)
    C = (sigma * sigma * A + mu * B + alpha * x0 * x0) / rho
    disc = mu * mu + 2.0 * rho * sigma * sigma
    r1 = (-mu + math.sqrt(disc)) / (sigma * sigma)  # positive root
    return A, B, C, r1

# ------------------------- inner solver: d given x_hat -------------------------

def _solve_d_given_xhat(x_hat: float, A: float, B: float, r1: float, lam: float, c: float,
                        tol: float = 1e-12, itmax: int = 200) -> float:
    """Recover the positive band width by monotone bisection."""
    const = (2.0 / r1) - (B - lam) / A

    def T(d: float) -> float:
        """Evaluate the inner scalar root-finding residual."""
        return 2.0 * x_hat + d - const - (c / (A * d))

    # Bracket: as d -> 0+, T(d) -> -∞ ; as d -> +∞, T(d) -> +∞
    dL = 1e-14
    dR = 1.0
    while T(dR) < 0.0:
        dR *= 2.0
        if dR > 1e12:
            raise RuntimeError("Failed to bracket d. Check parameters (A, B, r1, c, lambda).")

    # Bisection
    for _ in range(itmax):
        dm = 0.5 * (dL + dR)
        val = T(dm)
        if abs(val) < tol or abs(dR - dL) < tol:
            return dm
        if val < 0.0:
            dL = dm
        else:
            dR = dm
    return 0.5 * (dL + dR)

# ------------------------- outer residual in x_hat -------------------------

def _outer_residual(x_hat: float, A: float, B: float, r1: float, lam: float, c: float) -> float:
    """
    Outer residual: (Agen) after eliminating d via (Bgen).
      (2A x_hat + B - λ) e^{-r1 x_hat} = (2A(x_hat+d) + B - λ) e^{-r1 (x_hat + d)}
    Returns lhs - rhs.
    """
    d = _solve_d_given_xhat(x_hat, A, B, r1, lam, c, tol=1e-13)
    x_star = x_hat + d
    lhs = (2.0 * A * x_hat + B - lam) * math.exp(-r1 * x_hat)
    rhs = (2.0 * A * x_star + B - lam) * math.exp(-r1 * x_star)
    return lhs - rhs

# ------------------------- core solver -------------------------

def _solve_band_general(mu: float, sigma: float, rho: float, alpha: float, x0: float,
                        lam: float, c: float, tol: float = 1e-12, itmax: int = 200):
    """
    Solve for (x_hat, x_star, a) in the general diffusion case (no jumps).
    Uses nested bisection: d|x_hat via (Bgen), then x_hat via (Agen).
    """
    assert c > 0.0 and lam >= 0.0, "Require c>0 and λ>=0."
    A, B, C, r1 = _coefficients(mu, sigma, rho, alpha, x0)

    # Bracket x_hat for outer residual (Agen)
    xL = 1e-12
    xR = max(1e-3, 0.5 * lam * (rho if rho > 0 else 1.0) + math.sqrt(abs(c) / (alpha if alpha > 0 else 1.0)) + abs(x0))
    fL = _outer_residual(xL, A, B, r1, lam, c)
    fR = _outer_residual(xR, A, B, r1, lam, c)

    expand = 0
    while fL * fR > 0.0 and expand < 60:
        xR *= 2.0
        fR = _outer_residual(xR, A, B, r1, lam, c)
        expand += 1
    if fL * fR > 0.0:
        raise RuntimeError("Failed to bracket x_hat; try different parameters (ρ,λ,c) or initial bounds.")

    # Bisection on x_hat
    for _ in range(itmax):
        xm = 0.5 * (xL + xR)
        fm = _outer_residual(xm, A, B, r1, lam, c)
        if abs(fm) < tol or abs(xR - xL) < tol:
            x_hat = xm
            break
        if fL * fm <= 0.0:
            xR, fR = xm, fm
        else:
            xL, fL = xm, fm
    else:
        x_hat = 0.5 * (xL + xR)

    # Recover d, x_star, a
    d = _solve_d_given_xhat(x_hat, A, B, r1, lam, c, tol=1e-13)
    x_star = x_hat + d
    a = ((2.0 * A * x_star + B - lam) / r1) * math.exp(-r1 * x_star)

    return x_hat, x_star, a, (A, B, C, r1)

# ------------------------- public API -------------------------

def solve_psi_general(mu: float, sigma: float, rho: float, alpha: float, x0: float,
                      lam: float, c: float):
    """
    Build ψ, ζ* for the general diffusion case (no jumps).
    Returns:
        psi (callable: torch.Tensor -> torch.Tensor),
        zeta_star (callable: torch.Tensor -> torch.Tensor),
        params (dict)
    """
    x_hat, x_star, a, (A, B, C, r1) = _solve_band_general(mu, sigma, rho, alpha, x0, lam, c)
    # Precompute ψ0(x_hat)
    psi0_xhat = (A * x_hat * x_hat + B * x_hat + C) - a * math.exp(r1 * x_hat)

    def psi(x: torch.Tensor) -> torch.Tensor:
        """Piecewise ψ: inaction vs action."""
        x = x.to(dtype=torch.get_default_dtype())
        psi_p = A * x * x + B * x + C
        val_inact = psi_p - a * torch.exp(torch.tensor(r1) * x)
        val_act = torch.full_like(x, psi0_xhat) + c + lam * (x - x_hat)
        return torch.where(x < x_star, val_inact, val_act)

    def zeta_star(x: torch.Tensor) -> torch.Tensor:
        """Optimal impulse size: jump down to x_hat."""
        x = x.to(dtype=torch.get_default_dtype())
        return torch.clamp(x - x_hat, min=0.0)

    params = {
        "x_hat": x_hat,
        "x_star": x_star,
        "a": a,
        "r1": r1,
        "A": A, "B": B, "C": C,
        "mu": mu, "sigma": sigma, "rho": rho, "alpha": alpha, "x0": x0,
        "lam": lam, "c": c,
    }
    return psi, zeta_star, params

# ------------------------- plotting utility -------------------------

def plot_psi_general(mu: float, sigma: float, rho: float, alpha: float, x0: float,
                     lam: float, c: float,
                     x_min: float = 0.0, x_max: float = 5.0, n_points: int = 1201,
                     show: bool = True, save_path: str | None = None):
    """
    Plot ψ(x) over [x_min, x_max] with vertical markers at x_hat and x_star.
    Uses matplotlib (no seaborn); single figure; no explicit colors.
    """
    psi, zeta, p = solve_psi_general(mu, sigma, rho, alpha, x0, lam, c)
    print(f"psi(1) = {psi(torch.ones(1))}")
    xs = torch.linspace(x_min, x_max, n_points, dtype=torch.get_default_dtype())
    ys = psi(xs)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(xs.cpu().numpy(), ys.detach().cpu().numpy(), label=r"$\psi(x)$")
    plt.axvline(p["x_hat"], linestyle="--", linewidth=1.0, label=r"$\hat{x}$")
    plt.axvline(p["x_star"], linestyle=":", linewidth=1.0, label=r"$x^\ast$")

    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title("Band-policy value function ψ (general μ, σ, ρ, α, x0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    return {k: p[k] for k in ("x_hat", "x_star", "a", "r1", "A", "B", "C")}


# -*- coding: utf-8 -*-
import torch
from typing import Optional, Dict

@torch.no_grad()
def simulate_band_policy_nd(
    mu: float,
    sigma: float,
    rho: float,
    alpha: float,
    x0: float,
    lam: float,
    c_f: float,
    d: int,                         # state dimension
    x_init: float = 1.0,
    dt: float = 1e-3,
    T_max: float = 5.0,
    n_paths: int = 5,
    n_impulses_max: int = 2000,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Simulate the exact harvesting band policy in every state coordinate."""
    from math import ceil

    # Solve 1D band-policy once (gives thresholds, coefficients etc.)
    psi_1d, zeta_star_1d, params_1d = solve_psi_general(
        mu=mu,
        sigma=sigma,
        rho=rho,
        alpha=alpha,
        x0=x0,
        lam=lam,
        c=c_f,
    )
    x_hat  = float(params_1d["x_hat"])
    x_star = float(params_1d["x_star"])

    # Device / dtype
    dev   = torch.device(device)
    dtype = torch.get_default_dtype()

    mu_t    = torch.tensor(mu,    dtype=dtype, device=dev)
    sigma_t = torch.tensor(sigma, dtype=dtype, device=dev)
    rho_t   = torch.tensor(rho,   dtype=dtype, device=dev)
    alpha_t = torch.tensor(alpha, dtype=dtype, device=dev)
    x0_t    = torch.tensor(x0,    dtype=dtype, device=dev)
    lam_t   = torch.tensor(lam,   dtype=dtype, device=dev)
    c_t     = torch.tensor(c_f,   dtype=dtype, device=dev)
    x_hat_t  = torch.tensor(x_hat,  dtype=dtype, device=dev)
    x_star_t = torch.tensor(x_star, dtype=dtype, device=dev)

    dt_t      = torch.tensor(dt, dtype=dtype, device=dev)
    sqrt_dt   = torch.sqrt(dt_t)
    max_steps = max(1, int(T_max / dt))

    # RNG
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))

    # Storage: trajectories and impulses in R^d
    X_hist = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)
    impulse_mask  = torch.zeros((n_paths, max_steps + 1, d), dtype=torch.bool, device=dev)
    impulse_state = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)
    impulse_zeta  = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)

    cost_per_path = torch.zeros(n_paths, dtype=dtype, device=dev)
    running_cost  = torch.zeros(n_paths, dtype=dtype, device=dev)

    # Init X in R^d with same scalar x_init on each coordinate
    X = torch.full((n_paths, d), float(x_init), dtype=dtype, device=dev)
    X_hist[:, 0, :] = X

    # Count total number of impulses per path (sum over coords and time)
    impulses_count = torch.zeros(n_paths, dtype=torch.int32, device=dev)

    # Discount factor
    disc_t    = torch.ones((), dtype=dtype, device=dev)     # scalar
    disc_step = torch.exp(-rho_t * dt_t)                    # scalar

    # Time loop
    for step in range(1, max_steps + 1):

        # (A) PRE-diffusion impulses (product (s,S)-policy per coordinate)
        # zeta_ij = X_ij - x_hat if X_ij >= x_star, else 0
        # Enforce the global impulse cap on each path.
        if (impulses_count < n_impulses_max).any():
            # Compute candidate zetas for all coords
            zeta_all = X - x_hat_t                 # [n_paths, d]
            mask_coord = (X >= x_star_t)          # [n_paths, d]
            # Apply global cap: if path already at cap, forbid new impulses
            active_paths = (impulses_count < n_impulses_max).view(-1, 1)
            mask = mask_coord & active_paths      # [n_paths, d]

            if mask.any():
                # Clip zeta to non-negative only where mask is True
                zeta = torch.where(mask, torch.clamp(zeta_all, min=0.0), torch.zeros_like(zeta_all))

                # Per-coordinate impulse cost, discounted at current time
                # For each coord where mask_ij=True:
                #   cost_ij = c_f + lam * zeta_ij
                # and zero elsewhere
                impulse_cost_ij = mask.to(dtype) * (c_t + lam_t * zeta)  # [n_paths, d]
                cost_per_path += disc_t * impulse_cost_ij.sum(dim=1)    # [n_paths]

                # Apply impulses on coordinates where mask=True
                X = torch.where(mask, x_hat_t, X)

                # Update counts: each coord impulse increments per path
                impulses_count += mask.sum(dim=1).to(impulses_count.dtype)

                impulse_mask[:, step, :]  = mask
                impulse_state[:, step, :] = X
                impulse_zeta[:, step, :]  = zeta

        # (B) Diffusion step in R^d
        dW = torch.randn((n_paths, d), dtype=dtype, device=dev, generator=gen)
        X = X + mu_t * dt_t + sigma_t * sqrt_dt * dW

        # (C) Running cost: sum_i alpha (X_i - x0)^2
        rc_ij = alpha_t * (X - x0_t) ** 2               # [n_paths, d]
        rc = rc_ij.sum(dim=1)                           # [n_paths]
        running_cost += disc_t * rc * dt_t

        # (D) Store and update discount
        X_hist[:, step, :] = X
        disc_t = disc_t * disc_step

        # Optional early stop: all paths reached impulse cap
        if (impulses_count >= n_impulses_max).all():
            break

    cost_per_path += running_cost

    return {
        "COST_total_per_path": cost_per_path,           # [n_paths]
        "COST_mean": cost_per_path.mean(),
        "X_hist": X_hist,                               # [n_paths, max_steps+1, d]
        "impulse_mask": impulse_mask,                   # [n_paths, max_steps+1, d]
        "impulse_state": impulse_state,                 # [n_paths, max_steps+1, d]
        "impulse_zeta": impulse_zeta,                   # [n_paths, max_steps+1, d]
        "x_hat": torch.tensor(x_hat, dtype=dtype, device=dev),
        "x_star": torch.tensor(x_star, dtype=dtype, device=dev),
        "params_1d": params_1d,
        "dt": torch.tensor(dt, dtype=dtype, device=dev),
        "T_max": torch.tensor(T_max, dtype=dtype, device=dev),
        "impulses_count": impulses_count,               # [n_paths]
        "d": torch.tensor(d, dtype=torch.int64, device=dev),
    }
