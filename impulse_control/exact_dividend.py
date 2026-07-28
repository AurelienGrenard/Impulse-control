"""Closed-form reference solution and band-policy simulation for dividends."""

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

# ------------------------- helpers: characteristic roots -------------------------

def _roots_r(mu: float, sigma: float, rho: float):
    """
    Solve (1/2) sigma^2 r^2 + mu r - rho = 0.
    Returns r1 > 0 and r2 < 0.
    """
    assert sigma > 0.0 and rho > 0.0, "Require sigma > 0 and rho > 0."
    disc = mu * mu + 2.0 * rho * sigma * sigma
    sqrt_disc = math.sqrt(disc)
    r1 = (-mu + sqrt_disc) / (sigma * sigma)
    r2 = (-mu - sqrt_disc) / (sigma * sigma)
    return r1, r2


def _x_tilde(r1: float, r2: float) -> float:
    """
    Inflection point of psi'(x):
        x_tilde = 2 (log|r2| - log r1) / (r1 - r2)
    """
    return 2.0 * (math.log(abs(r2)) - math.log(r1)) / (r1 - r2)


def _g_of_x(x: float, r1: float, r2: float) -> float:
    """
    g(x) = r1 e^{r1 x} - r2 e^{r2 x}
    """
    return r1 * math.exp(r1 * x) - r2 * math.exp(r2 * x)


def _h_of_x(x: float, r1: float, r2: float) -> float:
    """
    h(x) = e^{r1 x} - e^{r2 x}
    """
    return math.exp(r1 * x) - math.exp(r2 * x)


# ------------------------- inner solver: x_hat given x_star -------------------------

def _find_xhat_given_xstar(
    x_star: float,
    r1: float,
    r2: float,
    tol: float = 1e-12,
    itmax: int = 200,
) -> float:
    """
    Solve g(x_hat) = g(x_star) on the interval (0, x_tilde),
    assuming x_star > x_tilde.
    """
    xt = _x_tilde(r1, r2)
    g_star = _g_of_x(x_star, r1, r2)

    def F(x: float) -> float:
        """Evaluate the scalar root-finding residual."""
        return _g_of_x(x, r1, r2) - g_star

    xL = 1e-14
    xR = xt - 1e-14

    fL = F(xL)
    fR = F(xR)

    if fL * fR > 0.0:
        raise RuntimeError("Failed to bracket x_hat in (0, x_tilde).")

    for _ in range(itmax):
        xm = 0.5 * (xL + xR)
        fm = F(xm)

        if abs(fm) < tol or abs(xR - xL) < tol:
            return xm

        if fL * fm <= 0.0:
            xR, fR = xm, fm
        else:
            xL, fL = xm, fm

    return 0.5 * (xL + xR)


# ------------------------- outer residual in x_star -------------------------

def _outer_residual(x_star: float, r1: float, r2: float, lam: float, c: float) -> float:
    """Evaluate value matching after eliminating the lower threshold."""
    xt = _x_tilde(r1, r2)

    if x_star <= xt:
        return float("inf")

    g_star = _g_of_x(x_star, r1, r2)
    x_hat = _find_xhat_given_xstar(x_star, r1, r2, tol=1e-13)
    A = 1.0 / ((1.0 + lam) * g_star)

    lhs = A * (_h_of_x(x_star, r1, r2) - _h_of_x(x_hat, r1, r2))
    rhs = (x_star - x_hat - c) / (1.0 + lam)

    return lhs - rhs


# ------------------------- core solver -------------------------

def _solve_band_dividend(
    mu: float,
    sigma: float,
    rho: float,
    lam: float,
    c: float,
    tol: float = 1e-12,
    itmax: int = 200,
):
    """
    Solve for (x_hat, x_star, A) in the dividend problem.
    Uses nested bisection:
        x_hat | x_star from g(x_hat) = g(x_star),
        then x_star from the value-matching residual.
    """
    assert c > 0.0 and lam >= 0.0, "Require c > 0 and lambda >= 0."

    r1, r2 = _roots_r(mu, sigma, rho)
    xt = _x_tilde(r1, r2)

    def F(x_star: float) -> float:
        """Evaluate the scalar root-finding residual."""
        return _outer_residual(x_star, r1, r2, lam, c)

    xL = xt + 1e-8
    xR = max(xt + 0.25, xt + 1.0)

    fL = F(xL)
    fR = F(xR)

    expand = 0
    while fL * fR > 0.0 and expand < 60:
        xR *= 2.0
        fR = F(xR)
        expand += 1

    if fL * fR > 0.0:
        raise RuntimeError("Failed to bracket x_star.")

    for _ in range(itmax):
        xm = 0.5 * (xL + xR)
        fm = F(xm)

        if abs(fm) < tol or abs(xR - xL) < tol:
            x_star = xm
            break

        if fL * fm <= 0.0:
            xR, fR = xm, fm
        else:
            xL, fL = xm, fm
    else:
        x_star = 0.5 * (xL + xR)

    x_hat = _find_xhat_given_xstar(x_star, r1, r2, tol=1e-13)
    g_star = _g_of_x(x_star, r1, r2)
    A = 1.0 / ((1.0 + lam) * g_star)

    return x_hat, x_star, A, (r1, r2)


# ------------------------- public API -------------------------

def solve_psi_dividend(
    mu: float,
    sigma: float,
    rho: float,
    lam: float,
    c: float,
):
    """
    Build ψ, ζ* for the dividend problem.

    Returns:
        psi (callable: torch.Tensor -> torch.Tensor),
        zeta_star (callable: torch.Tensor -> torch.Tensor),
        params (dict)
    """
    x_hat, x_star, A, (r1, r2) = _solve_band_dividend(mu, sigma, rho, lam, c)

    h_hat = _h_of_x(x_hat, r1, r2)
    psi_hat = A * h_hat

    def psi(x: torch.Tensor) -> torch.Tensor:
        """
        Piecewise value function: inaction vs action.
        """
        x = x.to(dtype=torch.get_default_dtype())

        r1_t = x.new_tensor(r1)
        r2_t = x.new_tensor(r2)
        A_t = x.new_tensor(A)
        x_hat_t = x.new_tensor(x_hat)
        x_star_t = x.new_tensor(x_star)
        c_t = x.new_tensor(c)
        lam_t = x.new_tensor(lam)
        psi_hat_t = x.new_tensor(psi_hat)

        val_inact = A_t * (torch.exp(r1_t * x) - torch.exp(r2_t * x))
        val_act = psi_hat_t + (x - x_hat_t - c_t) / (1.0 + lam_t)

        return torch.where(x < x_star_t, val_inact, val_act)

    def zeta_star(x: torch.Tensor) -> torch.Tensor:
        """
        Optimal dividend size sending the state back to x_hat:
            zeta*(x) = max((x - x_hat - c) / (1 + lambda), 0)
        """
        x = x.to(dtype=torch.get_default_dtype())

        x_hat_t = x.new_tensor(x_hat)
        c_t = x.new_tensor(c)
        lam_t = x.new_tensor(lam)

        return torch.clamp((x - x_hat_t - c_t) / (1.0 + lam_t), min=0.0)

    params = {
        "x_hat": x_hat,
        "x_star": x_star,
        "A": A,
        "r1": r1,
        "r2": r2,
        "mu": mu,
        "sigma": sigma,
        "rho": rho,
        "lam": lam,
        "c": c,
    }

    return psi, zeta_star, params

@torch.no_grad()
def simulate_dividend_policy_nd(
    mu: float,
    sigma: float,
    rho: float,
    lam: float,
    c_f: float,
    d: int,
    x_init: float = 1.0,
    dt: float = 1e-3,
    T_max: float = 5.0,
    n_paths: int = 5,
    n_impulses_max: int = 2000,
    seed: Optional[int] = None,
    device: str = "cuda",
    absorb_at_zero: bool = True,
) -> Dict[str, torch.Tensor]:
    """Simulate the exact dividend band policy in every state coordinate."""
    psi_1d, zeta_star_1d, params_1d = solve_psi_dividend(
        mu=mu,
        sigma=sigma,
        rho=rho,
        lam=lam,
        c=c_f,
    )

    x_hat = float(params_1d["x_hat"])
    x_star = float(params_1d["x_star"])

    dev = torch.device(device)
    dtype = torch.get_default_dtype()

    mu_t = torch.tensor(mu, dtype=dtype, device=dev)
    sigma_t = torch.tensor(sigma, dtype=dtype, device=dev)
    rho_t = torch.tensor(rho, dtype=dtype, device=dev)
    lam_t = torch.tensor(lam, dtype=dtype, device=dev)
    c_t = torch.tensor(c_f, dtype=dtype, device=dev)

    x_hat_t = torch.tensor(x_hat, dtype=dtype, device=dev)
    x_star_t = torch.tensor(x_star, dtype=dtype, device=dev)

    dt_t = torch.tensor(dt, dtype=dtype, device=dev)
    sqrt_dt = torch.sqrt(dt_t)
    max_steps = max(1, int(T_max / dt))

    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))

    X_hist = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)
    alive_hist = torch.zeros((n_paths, max_steps + 1, d), dtype=torch.bool, device=dev)

    impulse_mask = torch.zeros((n_paths, max_steps + 1, d), dtype=torch.bool, device=dev)
    impulse_state = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)
    impulse_zeta = torch.zeros((n_paths, max_steps + 1, d), dtype=dtype, device=dev)

    reward_per_path = torch.zeros(n_paths, dtype=dtype, device=dev)

    X = torch.full((n_paths, d), float(x_init), dtype=dtype, device=dev)
    alive = X > 0.0

    X_hist[:, 0, :] = X
    alive_hist[:, 0, :] = alive

    impulses_count = torch.zeros(n_paths, dtype=torch.int32, device=dev)

    disc_t = torch.ones((), dtype=dtype, device=dev)
    disc_step = torch.exp(-rho_t * dt_t)

    for step in range(1, max_steps + 1):

        if (impulses_count < n_impulses_max).any():
            active_paths = (impulses_count < n_impulses_max).view(-1, 1)

            if absorb_at_zero:
                eligible = alive & active_paths & (X >= x_star_t)
            else:
                eligible = active_paths & (X >= x_star_t)

            if eligible.any():
                zeta_all = (X - x_hat_t - c_t) / (1.0 + lam_t)
                zeta = torch.where(
                    eligible,
                    torch.clamp(zeta_all, min=0.0),
                    torch.zeros_like(X),
                )

                reward_per_path += disc_t * zeta.sum(dim=1)

                X = torch.where(eligible, x_hat_t, X)
                impulses_count += eligible.sum(dim=1).to(impulses_count.dtype)

                impulse_mask[:, step, :] = eligible
                impulse_state[:, step, :] = X
                impulse_zeta[:, step, :] = zeta

        dW = torch.randn((n_paths, d), dtype=dtype, device=dev, generator=gen)

        if absorb_at_zero:
            X_next = X + mu_t * dt_t + sigma_t * sqrt_dt * dW
            X = torch.where(alive, X_next, X)
            alive = alive & (X > 0.0)
            X = torch.where(alive, X, torch.zeros_like(X))
        else:
            X = X + mu_t * dt_t + sigma_t * sqrt_dt * dW

        X_hist[:, step, :] = X
        alive_hist[:, step, :] = alive if absorb_at_zero else torch.ones_like(alive)

        disc_t = disc_t * disc_step

        if (impulses_count >= n_impulses_max).all():
            break

    # Terminal liquidation at time T_max
    zeta_terminal = torch.clamp((X - c_t) / (1.0 + lam_t), min=0.0)
    terminal_reward_per_path = disc_t * zeta_terminal.sum(dim=1)
    reward_per_path += terminal_reward_per_path

    return {
        "REWARD_total_per_path": reward_per_path,
        "REWARD_mean": reward_per_path.mean(),
        "REWARD_terminal_per_path": terminal_reward_per_path,
        "X_hist": X_hist,
        "alive_hist": alive_hist,
        "impulse_mask": impulse_mask,
        "impulse_state": impulse_state,
        "impulse_zeta": impulse_zeta,
        "zeta_terminal": zeta_terminal,
        "x_hat": torch.tensor(x_hat, dtype=dtype, device=dev),
        "x_star": torch.tensor(x_star, dtype=dtype, device=dev),
        "params_1d": params_1d,
        "dt": torch.tensor(dt, dtype=dtype, device=dev),
        "T_max": torch.tensor(T_max, dtype=dtype, device=dev),
        "impulses_count": impulses_count,
        "d": torch.tensor(d, dtype=torch.int64, device=dev),
    }
