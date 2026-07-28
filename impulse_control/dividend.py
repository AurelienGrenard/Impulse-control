"""Neural impulse-control algorithms for the dividend problem."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .common import OptConfig, TimeGridConfig
from .exact_dividend import *

Tensor = torch.Tensor

Tensor = torch.Tensor

# Global experiment config
@dataclass
class NetConfig:
    """Configure the neural continuation-value regressor."""
    depth: int = 3
    width: int = 128

    # Activation
    activation: str = "leaky_relu"     # default activation
    negative_slope: float = 0.01       # used for LeakyReLU
    softplus_beta: float = 1.0         # used only if activation="softplus"

    weight_decay: float = 0.0
    lr: float = 1e-3

    # Default "full training" budget (used when not using transfer)
    steps: int = 20_000
    batch_size: int = 1024

    use_custom_init: bool = True

    # Optional exact projection so that Q(0) = 0
    use_zero_projection: bool = False

    # Transfer learning
    use_transfer_learning: bool = True
    transfer_steps: int = 400
    transfer_lr: Optional[float] = None
    freeze_backbone: bool = False
    reset_last_layer: bool = False

@dataclass
class MCConfig:
    """Configure Monte Carlo rollout batching."""
    M_k: int = 1              # number of replications per design point (variance reduction)
    chunk_size_x: int = 8192  # chunking over design points (GPU memory control)




@dataclass
class DividendParams:
    """Store one-dimensional dividend parameters."""
    mu: float = 1.0
    sigma: float = 0.50
    rho: float = 0.05
    lam: float = 0.20   # proportional intervention cost
    c: float = 0.50   # fixed cost

@dataclass
class DividendParamsND:
    """Store dividend parameters for every state coordinate."""

    mu:    Optional[Tensor] = None
    sigma: Optional[Tensor] = None
    lam:   Optional[Tensor] = None
    c:     Optional[Tensor] = None

    # Single discount rate for the whole problem
    rho: Optional[float] = None

    d: int = 2  # default ND dimension

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """
        If tensors are not provided, instantiate them by replicating
        the default 1D DividendParams over 'd' coordinates.
        """
        base = DividendParams()  # uses default 1D values

        def rep(val: float) -> Tensor:
            """Replicate a scalar parameter across all coordinates."""
            return torch.full((self.d,), float(val),
                              device=self.device, dtype=self.dtype)

        if self.mu    is None: self.mu    = rep(base.mu)
        if self.sigma is None: self.sigma = rep(base.sigma)
        if self.lam   is None: self.lam   = rep(base.lam)
        if self.c     is None: self.c     = rep(base.c)
        if self.rho   is None: self.rho   = base.rho

    @property
    def state_dim(self) -> int:
        """Return the number of state coordinates."""
        return self.d



# Design configuration in R^d with shared time grid


@dataclass
class DesignConfig:
    """Configure state sampling domains and dynamic design bounds."""

    # Default scalar bounds, used if per-coordinate bounds are not given
    x_min: float = 0.0
    x_max: float = 8.0

    # Optional per-coordinate bounds: length must match d when provided
    x_min_vec: Optional[List[float]] = None
    x_max_vec: Optional[List[float]] = None

    # Scalar bounds used for one-dimensional and diagonal plots.
    x_min_plot: float = 0.0
    x_max_plot: float = 2.0

    N_k: int = 2_000
    scheme: str = "uniform"

    # Dynamic cone parameters
    cone_width: float = 4.0
    use_cone: bool = False

    # Dynamic design ranges per time step: tensors of shape [K+1, d]
    x_min_k: Optional[Tensor] = None
    x_max_k: Optional[Tensor] = None

    def _resolve_bounds(
        self,
        d: int,
        device: str,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return per-coordinate base bounds x_min_base, x_max_base as tensors
        of shape [d], using x_min_vec/x_max_vec if provided, otherwise
        replicating scalar x_min/x_max.
        """
        if self.x_min_vec is not None:
            if len(self.x_min_vec) != d:
                raise ValueError(
                    f"x_min_vec length {len(self.x_min_vec)} != dimension d={d}"
                )
            x_min_base = torch.tensor(self.x_min_vec, device=device, dtype=dtype)
        else:
            x_min_base = torch.full(
                (d,),
                self.x_min,
                device=device,
                dtype=dtype,
            )

        if self.x_max_vec is not None:
            if len(self.x_max_vec) != d:
                raise ValueError(
                    f"x_max_vec length {len(self.x_max_vec)} != dimension d={d}"
                )
            x_max_base = torch.tensor(self.x_max_vec, device=device, dtype=dtype)
        else:
            x_max_base = torch.full(
                (d,),
                self.x_max,
                device=device,
                dtype=dtype,
            )

        return x_min_base, x_max_base

    def build_dynamic_ranges(
        self,
        Tgrid: Tensor,
        params_nd: "DividendParamsND",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build time-dependent multidimensional state-sampling bounds."""
        # Shared time grid
        t = Tgrid.to(device=device, dtype=dtype)  # shape [K+1]
        Kp1 = t.shape[0]

        # Dimension is taken from the ND parameters
        d = params_nd.state_dim

        # Base bounds per coordinate: shape [d]
        x_min_base, x_max_base = self._resolve_bounds(d=d, device=device, dtype=dtype)

        if not self.use_cone:
            # Static hyper-rectangle, same for all time steps
            self.x_min_k = x_min_base.unsqueeze(0).expand(Kp1, d).contiguous()
            self.x_max_k = x_max_base.unsqueeze(0).expand(Kp1, d).contiguous()
            return

        # Dynamic cones, fully vectorized
        # params_nd.mu, params_nd.sigma are [d]
        mu_vec    = params_nd.mu.to(device=device, dtype=dtype)      # [d]
        sigma_vec = params_nd.sigma.to(device=device, dtype=dtype)   # [d]

        # Shapes for broadcasting:
        #   t_col:       [K+1, 1]
        #   sqrt_t_col:  [K+1, 1]
        #   mu_row:      [1, d]
        #   sigma_row:   [1, d]
        t_col = t.view(Kp1, 1)
        sqrt_t_col = torch.sqrt(t.clamp_min(0.0)).view(Kp1, 1)

        mu_row = mu_vec.view(1, d)
        sigma_row = sigma_vec.view(1, d)

        x_min_base_row = x_min_base.view(1, d)
        x_max_base_row = x_max_base.view(1, d)

        # Broadcasted computations, result [K+1, d]
        self.x_min_k = (
            x_min_base_row
            + t_col * mu_row
            - self.cone_width * sqrt_t_col * sigma_row
        )

        self.x_max_k = (
            x_max_base_row
            + t_col * mu_row
            + self.cone_width * sqrt_t_col * sigma_row
        )


@dataclass
class ExperimentConfig:
    """Collect all numerical and financial configuration blocks."""
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    time: TimeGridConfig = field(default_factory=TimeGridConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    opt: OptConfig = field(default_factory=OptConfig)
    net: NetConfig = field(default_factory=NetConfig)

    # ND dividend parameters: dimension is given by dividend.d
    dividend: DividendParamsND = field(default_factory=DividendParamsND)

    # Design config, dimension is inferred from dividend inside build_dynamic_ranges
    design: DesignConfig = field(default_factory=DesignConfig)

    # maximum number of impulses allowed
    # -1 means "unconstrained" (unlimited impulses)
    #  0 means "no impulses allowed"
    #  N >= 1 means "at most N impulses" over the whole horizon
    max_impulses: int = -1

    # Optional finite horizon in number of decision steps for rollouts
    horizon: Optional[int] = None

    verbose: bool = True

    def __post_init__(self):
        """
        Build the shared time grid and the dynamic design ranges.
        The state dimension is taken from dividend.d (via dividend.state_dim),
        and DesignConfig does not store any dimension itself.
        """
        # Shared time grid t_0,...,t_K on [0, T]
        self.t_grid = torch.linspace(
            0.0,
            self.time.T,
            self.time.K + 1,
            device=self.device,
            dtype=self.dtype,
        )

        # Build dynamic design ranges in R^d, where d = dividend.state_dim
        self.design.build_dynamic_ranges(
            Tgrid=self.t_grid,
            params_nd=self.dividend,
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def is_unconstrained(self) -> bool:
        """
        Convenience flag to check whether the control
        is unconstrained in the number of impulses.
        """
        return self.max_impulses < 0

# Problem interface + dividend problem
class DividendProblemND:
    """Define the multidimensional dividend dynamics, rewards, and impulses."""

    def __init__(
        self,
        params: DividendParamsND,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        # ND parameters as tensors of shape [d]
        """Initialize the object from the supplied configuration."""
        self.mu_vec = params.mu.to(device=device, dtype=dtype)         # [d]
        self.sigma_vec = params.sigma.to(device=device, dtype=dtype)   # [d]
        self.lam_vec = params.lam.to(device=device, dtype=dtype)       # [d]
        self.c_vec = params.c.to(device=device, dtype=dtype)           # [d]

        # Single discount rate
        self.rho = float(params.rho)

        self.device = device
        self.dtype = dtype

        self.state_dim = params.state_dim   # d
        self.action_dim = self.state_dim    # one impulse component per state component

    # Drift and volatility
    def mu(self, t: float, x: Tensor) -> Tensor:
        """Return the drift vector."""
        x2d = x.view(x.shape[0], -1)                    # [B, d]
        mu_row = self.mu_vec.view(1, self.state_dim)    # [1, d]
        return torch.ones_like(x2d) * mu_row            # [B, d]

    def sigma(self, t: float, x: Tensor) -> Tensor:
        """Return the volatility vector."""
        x2d = x.view(x.shape[0], -1)                        # [B, d]
        sigma_row = self.sigma_vec.view(1, self.state_dim)  # [1, d]
        return torch.ones_like(x2d) * sigma_row             # [B, d]

    # Rewards (UNdiscounted)
    def running_reward(self, t: float, x: Tensor) -> Tensor:
        """Evaluate the instantaneous dividend reward."""
        x2d = x.view(x.shape[0], -1)  # [B, d]
        return torch.zeros((x2d.shape[0],), device=x2d.device, dtype=x2d.dtype)

    def terminal_payoff(self, x: Tensor) -> Tensor:
        """Return the undiscounted coordinatewise liquidation payoff."""
        x2d = x.view(x.shape[0], -1)                        # [B, d]
        lam_row = self.lam_vec.view(1, self.state_dim)      # [1, d]
        c_row = self.c_vec.view(1, self.state_dim)          # [1, d]

        xi_max = torch.clamp((x2d - c_row) / (1.0 + lam_row), min=0.0)  # [B, d]
        return xi_max.sum(dim=1)  # [B]

    def discount(self, t: float) -> float:
        """
        Scalar discount factor used in rollouts:
            exp(-rho * t).
        """
        return math.exp(-self.rho * t)

    # Impulses
    def xi_bounds(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return feasible componentwise dividend impulses."""
        x2d = x.view(x.shape[0], -1)                        # [B, d]
        lam_row = self.lam_vec.view(1, self.state_dim)      # [1, d]
        c_row = self.c_vec.view(1, self.state_dim)          # [1, d]

        xi_min = torch.zeros_like(x2d)
        xi_max = torch.clamp((x2d - c_row) / (1.0 + lam_row), min=0.0)

        return xi_min, xi_max

    def apply_impulse(self, x: Tensor, xi: Tensor) -> Tensor:
        """Apply a dividend impulse to the current state."""
        x2d = x.view(x.shape[0], -1)                       # [B, d]
        xi2d = xi.view(x.shape[0], -1)                     # [B, d]

        lam_row = self.lam_vec.view(1, self.state_dim)     # [1, d]
        c_row = self.c_vec.view(1, self.state_dim)         # [1, d]

        active = (xi2d > 0.0).to(xi2d.dtype)               # [B, d]

        x_post = x2d - (1.0 + lam_row) * xi2d - active * c_row
        x_post = torch.clamp(x_post, min=0.0)

        return x_post

    def impulse_reward(self, t: float, xi: Tensor) -> Tensor:
        """
        Undiscounted impulse reward:
            reward(xi) = sum_i xi_i.

        Inputs:
            xi : [B, d]

        Output:
            reward : [B]
        """
        xi2d = xi.view(xi.shape[0], -1)   # [B, d]
        return xi2d.sum(dim=1)            # [B]


# Diffusion stepper (Euler)
class EulerStepper:
    """Advance the uncontrolled state with Euler discretization."""
    def step(
        self,
        problem: DividendProblemND,
        t: float,
        x: Tensor,
        dt: float,
        noise: Optional[Tensor] = None
    ) -> Tensor:
        """Advance the state by one Euler step."""
        if noise is None:
            noise = torch.randn_like(x)

        alive = (x > 0.0)  # [B, d] bool
        if not torch.any(alive):
            return x

        mu = problem.mu(t, x)
        sig = problem.sigma(t, x)

        x_next = x + mu * dt + sig * math.sqrt(dt) * noise
        x_next = torch.clamp(x_next, min=0.0)

        # Absorbing at 0 coordinatewise: once dead, stay dead
        x_next = torch.where(alive, x_next, torch.zeros_like(x_next))
        return x_next


def diffuse_interval(
    stepper: EulerStepper,
    problem: DividendProblemND,
    x: Tensor,
    t_start: float,
    t_end: float,
    dt_fine: Optional[float],
) -> Tensor:
    """
    Diffuse x from t_start to t_end using sub-steps of size dt_fine.
    Ensures exact landing at t_end.
    """
    if dt_fine is None:
        return stepper.step(problem, t_start, x, t_end - t_start)

    T = t_end - t_start
    n = max(1, math.ceil(T / dt_fine))
    dt = T / n

    t = t_start
    for _ in range(n):
        x = stepper.step(problem, t, x, dt)
        t += dt
    return x

# regressor for Q_hat

class QNet(nn.Module):
    """Approximate a continuation value with an optional zero-state projection."""

    def __init__(self, d_in: int, cfg: NetConfig):
        """Initialize the object from the supplied configuration."""
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in

        layers: List[nn.Module] = []
        din = d_in

        def make_activation():
            """Construct the configured activation layer."""
            if cfg.activation == "leaky_relu":
                return nn.LeakyReLU(cfg.negative_slope)
            elif cfg.activation == "softplus":
                return nn.Softplus(beta=cfg.softplus_beta)
            else:
                raise ValueError(f"Unknown activation {cfg.activation}")

        for _ in range(cfg.depth):
            layers.append(nn.Linear(din, cfg.width))
            layers.append(make_activation())
            din = cfg.width

        layers.append(nn.Linear(din, 1))
        self.net = nn.Sequential(*layers)

        if self.cfg.use_zero_projection:
            self.register_buffer("x_origin", torch.zeros(1, d_in))

        if self.cfg.use_custom_init:
            self.reset_parameters()

        if self.cfg.use_zero_projection:
            self.forward = self._forward_projected
        else:
            self.forward = self._forward_raw

    def reset_parameters(self):
        """Reset all trainable network layers."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=self.cfg.negative_slope)
                nn.init.zeros_(m.bias)

    def _forward_raw(self, x: Tensor) -> Tensor:
        """Evaluate the unconstrained network output."""
        return self.net(x).view(-1)

    def _forward_projected(self, x: Tensor) -> Tensor:
        """Evaluate the output with the zero-state projection."""
        out = self.net(x).view(-1)
        out_zero = self.net(self.x_origin).view(-1)
        return out - out_zero

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the continuation-value approximation."""
        raise RuntimeError("forward should have been rebound in __init__.")


def fit_qnet(
    model,
    X: Tensor,
    y: Tensor,
    cfg: NetConfig,
    verbose: bool = False,
    *,
    is_transfer: bool = False,
) -> None:
    """Fit a continuation network by MSE, optionally using transfer learning."""
    # -------- choose budgets / lr --------
    steps = cfg.transfer_steps if is_transfer else cfg.steps
    lr = (cfg.transfer_lr if (is_transfer and cfg.transfer_lr is not None) else cfg.lr)

    # -------- optionally freeze backbone --------
    if is_transfer and cfg.freeze_backbone:
        # Train only the last Linear layer
        for p in model.parameters():
            p.requires_grad_(False)
        # Enable grads on last Linear
        last_linear = None
        for m in reversed(list(model.net)):
            if isinstance(m, torch.nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("Could not find a Linear layer in model.net to unfreeze.")
        last_linear.weight.requires_grad_(True)
        last_linear.bias.requires_grad_(True)

    # -------- optionally reset last layer --------
    if is_transfer and cfg.reset_last_layer:
        last_linear = None
        for m in reversed(list(model.net)):
            if isinstance(m, torch.nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("Could not find a Linear layer in model.net to reset.")
        torch.nn.init.kaiming_uniform_(last_linear.weight, a=0.0)
        torch.nn.init.zeros_(last_linear.bias)

    # Build optimizer on trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)

    N = X.shape[0]
    bs = min(cfg.batch_size, N)

    for it in range(steps):
        idx = torch.randint(0, N, (bs,), device=X.device)
        pred = model(X[idx])
        loss = torch.mean((pred - y[idx]) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (it % 200 == 0 or it == steps - 1):
            tag = "transfer" if is_transfer else "full"
            print(f"[{tag}] fit step {it:5d} | loss={loss.item():.6e}")

    # Restore gradients for subsequent fits.
    if is_transfer and cfg.freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(True)



# Argmax on xi via random search
@torch.no_grad()
def argmax_intervention_value(
    problem,
    qhat: Callable[[Tensor], Tensor],
    t: float,
    x: Tensor,
    opt_cfg: OptConfig,
) -> Tuple[Tensor, Tensor]:
    """Maximize the intervention value over randomized multidimensional impulses."""
    # Ensure 2D shape
    x2d = x.view(x.shape[0], -1)  # [B, d]
    B, d = x2d.shape

    # Bounds on impulses: [B, d]
    xi_min_all, xi_max_all = problem.xi_bounds(x2d)

    # Feasibility: at least one coordinate admits a strictly positive impulse
    feasible = (xi_max_all > 0.0).any(dim=1)  # [B]

    # Default outputs: infeasible => -inf, xi=0
    m_val = torch.full(
        (B,),
        float("-inf"),
        device=x.device,
        dtype=x.dtype,
    )
    xi_star = torch.zeros(
        (B, d),
        device=x.device,
        dtype=x.dtype,
    )

    if not torch.any(feasible):
        return m_val, xi_star

    # Work only on feasible states
    idx = torch.nonzero(feasible).view(-1)   # [B_pos]
    x_pos = x2d[idx]                         # [B_pos, d]
    xi_min = xi_min_all[idx]                 # [B_pos, d]
    xi_max = xi_max_all[idx]                 # [B_pos, d]
    Bp = x_pos.shape[0]

    # Global random search (batched over n_global)
    nG = opt_cfg.n_global
    nG_batch = opt_cfg.n_global_batch or nG

    v_best = torch.full(
        (Bp,),
        float("-inf"),
        device=x.device,
        dtype=x.dtype,
    )  # [B_pos]

    xi_best = torch.zeros(
        (Bp, d),
        device=x.device,
        dtype=x.dtype,
    )  # [B_pos, d]

    xi_min_exp = xi_min[:, None, :]              # [B_pos, 1, d]
    span = (xi_max - xi_min)[:, None, :]         # [B_pos, 1, d]

    start = 0
    while start < nG:
        this_batch = min(nG_batch, nG - start)

        # u: [B_pos, this_batch, d] ~ U[0,1]
        u = torch.rand(
            (Bp, this_batch, d),
            device=x.device,
            dtype=x.dtype,
        )

        # Candidates in [xi_min, xi_max]
        xi_g_batch = xi_min_exp + span * u       # [B_pos, this_batch, d]

        # Replicate x_pos for each candidate
        x_rep = x_pos[:, None, :].expand(Bp, this_batch, d).reshape(-1, d)
        xi_rep = xi_g_batch.reshape(-1, d)       # [B_pos*this_batch, d]

        # Post-impulse state
        x_post = problem.apply_impulse(x_rep, xi_rep)  # [B_pos*this_batch, d]

        # Objective: Q(x_post) + impulse_reward(t, xi)
        val_batch = (
            qhat(x_post).view(Bp, this_batch)
            + problem.impulse_reward(t, xi_rep).view(Bp, this_batch)
        )  # [B_pos, this_batch]

        # Best candidate within this batch
        j_batch = torch.argmax(val_batch, dim=1)  # [B_pos]
        v_batch_best = val_batch[torch.arange(Bp, device=x.device), j_batch]       # [B_pos]
        xi_batch_best = xi_g_batch[torch.arange(Bp, device=x.device), j_batch, :]  # [B_pos, d]

        # Compare with global best so far
        better = v_batch_best > v_best
        v_best = torch.where(better, v_batch_best, v_best)
        xi_best = torch.where(better[:, None], xi_batch_best, xi_best)

        start += this_batch

    # Coordinate-wise sparsification
    x_post_best = problem.apply_impulse(x_pos, xi_best)   # [B_pos, d]

    # Relative move per coordinate: |x_post - x| / |x|
    x_abs = x_pos.abs().clamp_min(1e-6)                   # [B_pos, d]
    delta = (x_post_best - x_pos).abs()                   # [B_pos, d]
    rel_move = delta / x_abs                              # [B_pos, d]

    thresh = opt_cfg.min_rel_impulse
    mask_keep = (rel_move >= thresh)                      # [B_pos, d] in {0,1}

    xi_sparse = xi_best * mask_keep.to(xi_best.dtype)     # [B_pos, d]

    # Evaluate objective with sparsified xi
    x_post_sparse = problem.apply_impulse(x_pos, xi_sparse)   # [B_pos, d]
    val_sparse = qhat(x_post_sparse) + problem.impulse_reward(t, xi_sparse)  # [B_pos]

    # Keep the better between dense xi_best and sparse xi_sparse
    better_sparse = val_sparse > v_best
    v_best = torch.where(better_sparse, val_sparse, v_best)
    xi_best = torch.where(better_sparse[:, None], xi_sparse, xi_best)

    # Scatter back into full tensors
    m_val[idx] = v_best
    xi_star[idx] = xi_best

    return m_val, xi_star

@torch.no_grad()
def vhat_unconstrained(
    k: int,
    x: Tensor,
    qhats: List[Optional[QNet]],
    problem: DividendProblemND,
    opt_cfg: OptConfig,
    t_grid: Tensor,
) -> Tensor:
    """
    ND unconstrained dividend impulse control:
      V_k(x) = max( Q_k(x), M_k(x) ).
    """
    K = len(qhats) - 1

    if k == K:
        return problem.terminal_payoff(x)

    qk = qhats[k]
    if qk is None:
        raise RuntimeError(f"qhats[{k}] is None in vhat_unconstrained.")

    q_val = qk(x)
    t_k = float(t_grid[k].item())

    m_val, _ = argmax_intervention_value(
        problem=problem,
        qhat=lambda x_in: qk(x_in),
        t=t_k,
        x=x,
        opt_cfg=opt_cfg,
    )

    return torch.maximum(q_val, m_val)


@torch.no_grad()
def vhat_bounded(
    k: int,
    n: int,
    x: Tensor,
    qhats: List[List[Optional[QNet]]],
    problem: DividendProblemND,
    opt_cfg: OptConfig,
    t_grid: Tensor,
) -> Tensor:
    """
    ND bounded-impulse dividend version.

    With at most n impulses remaining at time step k:
      V_k^n(x) = max( Q_k^n(x), M_k^n(x) ),
    where the intervention branch uses one impulse immediately and then
    continues with n-1 impulses remaining.
    """
    K = len(qhats[0]) - 1

    if k == K:
        return problem.terminal_payoff(x)

    if n <= 0:
        qk0 = qhats[0][k]
        if qk0 is None:
            raise RuntimeError("qhats[0][k] is None at n=0.")
        return qk0(x)

    qk_n = qhats[n][k]
    if qk_n is None:
        raise RuntimeError("qhats[n][k] is None.")

    qk_nm1 = qhats[n - 1][k]
    if qk_nm1 is None:
        raise RuntimeError("qhats[n-1][k] is None.")

    t_k = float(t_grid[k].item())

    def qhat_nm1(x_in: Tensor) -> Tensor:
        """Evaluate the continuation network with one fewer impulse."""
        return qk_nm1(x_in)

    q_val = qk_n(x)

    m_val, _ = argmax_intervention_value(
        problem=problem,
        qhat=qhat_nm1,
        t=t_k,
        x=x,
        opt_cfg=opt_cfg,
    )

    return torch.maximum(q_val, m_val)


# Rollout: compute J_{k,K}(x) under a given future policy
@torch.no_grad()
def rollout_reward_to_go_unconstrained(
    problem: DividendProblemND,
    stepper: EulerStepper,
    t_grid: Tensor,
    k0: int,
    x0: Tensor,                       # [B, d] (or [B] for d=1)
    qhats: List[Optional[QNet]],      # length K+1, qhats[K] may be None
    opt_cfg: OptConfig,
    mc_cfg: MCConfig,
    horizon: Optional[int] = None,
    dt_fine: Optional[float] = None,
) -> Tensor:
    """Estimate unconstrained reward-to-go targets by Monte Carlo rollout."""
    B = x0.shape[0]
    K = t_grid.numel() - 1
    M = mc_cfg.M_k

    k_end = K if horizon is None else min(K, k0 + int(horizon))
    out = torch.zeros((B,), device=x0.device, dtype=x0.dtype)

    def qhat_or_zero(k: int, x_in: Tensor) -> Tensor:
        """Evaluate a continuation network or return the terminal value."""
        qk = qhats[k]
        if qk is None and k < len(qhats) - 1:
            raise RuntimeError(
                f"qhat[{k}] is None but k < K; missing regression or inconsistent qhats."
            )
        if qk is None:
            return torch.zeros((x_in.shape[0],), device=x_in.device, dtype=x_in.dtype)
        return qk(x_in)

    for _ in range(M):
        x = x0.clone()
        total = torch.zeros((B,), device=x0.device, dtype=x0.dtype)

        for k in range(k0, k_end):
            t_k = float(t_grid[k].item())
            t_kp1 = float(t_grid[k + 1].item())

            # Diffusion over [t_k, t_{k+1}]
            x = diffuse_interval(
                stepper=stepper,
                problem=problem,
                x=x,
                t_start=t_k,
                t_end=t_kp1,
                dt_fine=dt_fine,
            )

            # POST-diffusion decision at t_{k+1}, except at k_end
            kp = k + 1
            if kp < k_end:
                # A path is active if at least one coordinate can still pay a positive dividend
                xi_min, xi_max = problem.xi_bounds(x)              # [B, d], [B, d]
                active = (xi_max > 0.0).any(dim=1)                 # [B]

                if torch.any(active):
                    idx = torch.nonzero(active).view(-1)
                    x_sub = x[idx]                                 # [B_sub, d]

                    qhat_fn = lambda x_in, kk=kp: qhat_or_zero(kk, x_in)
                    Q_vals = qhat_fn(x_sub)                        # [B_sub]

                    M_vals, xi_star = argmax_intervention_value(
                        problem=problem,
                        qhat=qhat_fn,
                        t=t_kp1,
                        x=x_sub,
                        opt_cfg=opt_cfg,
                    )

                    do = (M_vals > Q_vals)
                    if torch.any(do):
                        idx_do = idx[do]
                        xi_do = xi_star[do]                        # [B_do, d]

                        total[idx_do] += (
                            problem.discount(t_kp1)
                            * problem.impulse_reward(t_kp1, xi_do)
                        )
                        x[idx_do] = problem.apply_impulse(x[idx_do], xi_do)

        # Residual value at k_end
        t_end = float(t_grid[k_end].item())
        V_residual = vhat_unconstrained(
            k=k_end,
            x=x,
            qhats=qhats,
            problem=problem,
            opt_cfg=opt_cfg,
            t_grid=t_grid,
        )
        total += problem.discount(t_end) * V_residual

        out += total

    return out / float(M)


@torch.no_grad()
def rollout_reward_to_go_bounded(
    problem: DividendProblemND,
    stepper: EulerStepper,
    t_grid: Tensor,
    k0: int,
    x0: Tensor,                          # [B, d]
    qhats: List[List[Optional[QNet]]],   # qhats[n][k] = Q_k^{(n)}(·)
    n0: int,                             # impulses remaining at time k0
    max_impulses: int,                   # global maximum allowed
    opt_cfg: OptConfig,
    mc_cfg: MCConfig,
    horizon: Optional[int] = None,
    dt_fine: Optional[float] = None,
) -> Tensor:
    """Estimate bounded reward-to-go targets by Monte Carlo rollout."""
    if max_impulses < 0:
        raise ValueError("rollout_reward_to_go_bounded requires max_impulses >= 0.")

    if not (0 <= n0 <= max_impulses):
        raise ValueError(
            f"n0 must be in [0, max_impulses], got n0={n0}, max_impulses={max_impulses}."
        )

    B = x0.shape[0]
    K = t_grid.numel() - 1
    M = mc_cfg.M_k

    k_end = K if horizon is None else min(K, k0 + int(horizon))
    out = torch.zeros((B,), device=x0.device, dtype=x0.dtype)

    for _ in range(M):
        x = x0.clone()
        total = torch.zeros((B,), device=x0.device, dtype=x0.dtype)

        # n_rem = impulses remaining along each path
        n_rem = torch.full(
            (B,),
            n0,
            device=x0.device,
            dtype=torch.long,
        )

        for k in range(k0, k_end):
            t_k = float(t_grid[k].item())
            t_kp1 = float(t_grid[k + 1].item())

            # Diffusion over [t_k, t_{k+1}]
            x = diffuse_interval(
                stepper=stepper,
                problem=problem,
                x=x,
                t_start=t_k,
                t_end=t_kp1,
                dt_fine=dt_fine,
            )

            # POST-diffusion decision at t_{k+1}, except at k_end
            kp = k + 1
            if kp < k_end:
                # Feasibility from xi_bounds, not merely x > 0
                _, xi_max = problem.xi_bounds(x)                   # [B, d]
                can_act = (xi_max > 0.0).any(dim=1)                # [B]

                # Need both: feasible positive impulse and at least one impulse left
                active = can_act & (n_rem > 0)
                if not torch.any(active):
                    continue

                n_min = int(n_rem[active].min().item())
                n_max = int(n_rem[active].max().item())

                for n_avail in range(n_min, n_max + 1):
                    mask = active & (n_rem == n_avail)
                    if not torch.any(mask):
                        continue

                    idx = torch.nonzero(mask).view(-1)
                    x_sub = x[idx]                                  # [B_sub, d]

                    # Continuation Q_{kp}^{(n_avail)}(x)
                    qk_cont = qhats[n_avail][kp]
                    if qk_cont is None:
                        raise RuntimeError(
                            f"qhats[{n_avail}][{kp}] is None in rollout_reward_to_go_bounded."
                        )
                    Q_vals = qk_cont(x_sub)                         # [B_sub]

                    # Post-impulse continuation uses Q_{kp}^{(n_avail - 1)}
                    qk_next = qhats[n_avail - 1][kp]
                    if qk_next is None:
                        raise RuntimeError(
                            f"qhats[{n_avail - 1}][{kp}] is None in rollout_reward_to_go_bounded."
                        )

                    def qhat_post(x_in: Tensor, qnet=qk_next) -> Tensor:
                        """Evaluate continuation after a candidate intervention."""
                        return qnet(x_in)

                    M_vals, xi_star = argmax_intervention_value(
                        problem=problem,
                        qhat=qhat_post,
                        t=t_kp1,
                        x=x_sub,
                        opt_cfg=opt_cfg,
                    )

                    do = (M_vals > Q_vals)
                    if torch.any(do):
                        idx_do = idx[do]
                        xi_do = xi_star[do]                         # [B_do, d]

                        total[idx_do] += (
                            problem.discount(t_kp1)
                            * problem.impulse_reward(t_kp1, xi_do)
                        )
                        x[idx_do] = problem.apply_impulse(x[idx_do], xi_do)
                        n_rem[idx_do] = n_avail - 1

        # Residual value at k_end with n_rem impulses left
        t_end = float(t_grid[k_end].item())
        V_residual = torch.zeros((B,), device=x0.device, dtype=x0.dtype)

        for n_avail in range(0, max_impulses + 1):
            mask = (n_rem == n_avail)
            if not torch.any(mask):
                continue

            idx = torch.nonzero(mask).view(-1)
            x_sub = x[idx]

            V_residual[idx] = vhat_bounded(
                k=k_end,
                n=n_avail,
                x=x_sub,
                qhats=qhats,
                problem=problem,
                opt_cfg=opt_cfg,
                t_grid=t_grid,
            )

        total += problem.discount(t_end) * V_residual
        out += total

    return out / float(M)


# Design builder (ND)
def sample_D_k(
    design_cfg: DesignConfig,
    k: int,
    device: str,
    dtype: torch.dtype,
) -> Tensor:
    """
    Generate N_k design points uniformly over the hyper-rectangle
    [x_min_k[k, :], x_max_k[k, :]] in R^d.

    Returns a tensor of shape [N_k, d].
    """
    if design_cfg.x_min_k is None or design_cfg.x_max_k is None:
        raise RuntimeError(
            "DesignConfig.x_min_k and x_max_k must be built before calling sample_D_k."
        )

    # x_min_k[k] and x_max_k[k] are [d]
    xmin = design_cfg.x_min_k[k].to(device=device, dtype=dtype)  # [d]
    xmax = design_cfg.x_max_k[k].to(device=device, dtype=dtype)  # [d]

    d = xmin.shape[0]

    # Uniform sampling in the hyper-rectangle
    # u: [N_k, d] with entries in [0,1]
    u = torch.rand((design_cfg.N_k, d), device=device, dtype=dtype)

    # Broadcast xmin, xmax to [N_k, d]
    xmin_row = xmin.view(1, d)           # [1, d]
    span_row = (xmax - xmin).view(1, d)  # [1, d]

    Xk = xmin_row + span_row * u         # [N_k, d]
    return Xk


# Backward training loop (unconstrained, ND)
def train_dividend_unconstrained(
    cfg: ExperimentConfig,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train the unconstrained dividend continuation networks backward in time."""
    device, dtype = cfg.device, cfg.dtype

    # ND problem and stepper
    problem = DividendProblemND(cfg.dividend, device=device, dtype=dtype)
    stepper = EulerStepper()

    # Time grid
    t_grid = cfg.t_grid
    K = t_grid.numel() - 1

    # Ensure dynamic design ranges are built
    if cfg.design.x_min_k is None or cfg.design.x_max_k is None:
        cfg.design.build_dynamic_ranges(
            Tgrid=t_grid,
            params_nd=cfg.dividend,
            device=device,
            dtype=dtype,
        )

    # State dimension d
    d_in = cfg.dividend.state_dim

    # Q_hat_k approximators, k = 0..K
    qhats: List[Optional[QNet]] = [None for _ in range(K + 1)]

    # Backward induction: k = K-1, ..., 0
    for k in reversed(range(K)):
        if cfg.verbose:
            print(f"\n[Unconstrained dividend ND] Backward step k={k}/{K-1}")

        # Design sampling at time t_k
        # Xk: [N_k, d]
        Xk = sample_D_k(cfg.design, k=k, device=device, dtype=dtype)

        # Monte Carlo estimate of J_k(x) under current future policy
        y_chunks: List[Tensor] = []
        for start in range(0, Xk.shape[0], cfg.mc.chunk_size_x):
            Xb = Xk[start:start + cfg.mc.chunk_size_x]  # [B_chunk, d]

            yb = rollout_reward_to_go_unconstrained(
                problem=problem,
                stepper=stepper,
                t_grid=t_grid,
                k0=k,
                x0=Xb,
                qhats=qhats,
                opt_cfg=cfg.opt,
                mc_cfg=cfg.mc,
                horizon=cfg.horizon,
                dt_fine=cfg.time.dt_fine,
            )
            y_chunks.append(yb)

        y = torch.cat(y_chunks, dim=0)  # [N_k]
        t_k = float(t_grid[k].item())

        # Local undiscounted regression target
        y_local = y / problem.discount(t_k)

        # Build Q_k net, warm start from Q_{k+1} if enabled and available
        use_transfer = bool(cfg.net.use_transfer_learning and (qhats[k + 1] is not None))

        if use_transfer:
            qnet = QNet(d_in=d_in, cfg=cfg.net).to(device=device, dtype=dtype)
            qnet.load_state_dict(qhats[k + 1].state_dict())
            fit_qnet(qnet, Xk, y_local, cfg.net, verbose=False, is_transfer=True)
        else:
            qnet = QNet(d_in=d_in, cfg=cfg.net).to(device=device, dtype=dtype)
            fit_qnet(qnet, Xk, y_local, cfg.net, verbose=False, is_transfer=False)

        qhats[k] = qnet

        # Optional plotting on a diagonal slice if d >= 2
        if verbose:
            plot_step_functions_unconstrained(
                k=k,
                cfg=cfg,
                qhats=qhats,
                problem=problem,
                t_grid=t_grid,
                opt_cfg=cfg.opt,
                n_points=800,
            )

    return {
        "config": cfg,
        "problem": problem,
        "t_grid": t_grid,
        "qhats": qhats,
    }


def train_dividend_bounded(
    cfg: ExperimentConfig,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Bounded impulse version of the RMC training for the ND dividend problem.

    Assumes:
      cfg.max_impulses >= 0
      qhats_bounded[n][k] approximates Q_k^{(n)}(x) for n = 0..max_imp, k = 0..K
      rollout_reward_to_go_bounded computes J_k^{(n)}(x) under the current future policy.
    """
    device, dtype = cfg.device, cfg.dtype

    if not hasattr(cfg, "max_impulses"):
        raise AttributeError("cfg.max_impulses must be defined for the bounded training.")
    if cfg.max_impulses < 0:
        raise ValueError("train_dividend_bounded requires cfg.max_impulses >= 0.")

    max_imp = cfg.max_impulses

    # ND problem and stepper
    problem = DividendProblemND(cfg.dividend, device=device, dtype=dtype)
    stepper = EulerStepper()

    # Time grid
    t_grid = cfg.t_grid
    K = t_grid.numel() - 1

    # Ensure dynamic design ranges are built
    if cfg.design.x_min_k is None or cfg.design.x_max_k is None:
        cfg.design.build_dynamic_ranges(
            Tgrid=t_grid,
            params_nd=cfg.dividend,
            device=device,
            dtype=dtype,
        )

    # State dimension
    d_in = cfg.dividend.state_dim

    # qhats_bounded[n][k] ~ Q_k^{(n)}(x)
    qhats_bounded: List[List[Optional[QNet]]] = [
        [None for _ in range(K + 1)] for _ in range(max_imp + 1)
    ]

    if cfg.verbose:
        print(f"\n[Bounded dividend ND] Training with at most {max_imp} impulses.")

    def pick_warm_start(n: int, k: int) -> Optional[QNet]:
        """
        Choose the closest previously trained network to warm start Q_k^{(n)}.

        Priority:
          1) Q_{k+1}^{(n)}
          2) Q_k^{(n-1)}
          3) Q_{k+1}^{(n-1)}
        """
        if cfg.net.use_transfer_learning:
            if (k + 1) <= K and qhats_bounded[n][k + 1] is not None:
                return qhats_bounded[n][k + 1]
            if n > 0 and qhats_bounded[n - 1][k] is not None:
                return qhats_bounded[n - 1][k]
            if n > 0 and (k + 1) <= K and qhats_bounded[n - 1][k + 1] is not None:
                return qhats_bounded[n - 1][k + 1]
        return None

    # Backward induction in time: k = K-1, ..., 0
    for k in reversed(range(K)):
        if cfg.verbose:
            print(f"\n[Bounded dividend ND] Backward step k={k}/{K-1}")

        # Design sampling at time t_k
        Xk = sample_D_k(cfg.design, k=k, device=device, dtype=dtype)  # [N_k, d]

        # Compute J_k^{(n)}(x) and regress Q_k^{(n)}
        n_max_k = min(max_imp, K - k)
        for n in range(0, n_max_k + 1):
            if cfg.verbose:
                print(f"  [Bounded dividend ND]  n = {n} impulses remaining")

            y_chunks_n: List[Tensor] = []
            for start in range(0, Xk.shape[0], cfg.mc.chunk_size_x):
                Xb = Xk[start:start + cfg.mc.chunk_size_x]  # [B_chunk, d]

                yb = rollout_reward_to_go_bounded(
                    problem=problem,
                    stepper=stepper,
                    t_grid=t_grid,
                    k0=k,
                    x0=Xb,
                    qhats=qhats_bounded,
                    n0=n,
                    max_impulses=max_imp,
                    opt_cfg=cfg.opt,
                    mc_cfg=cfg.mc,
                    horizon=cfg.horizon,
                    dt_fine=cfg.time.dt_fine,
                )
                y_chunks_n.append(yb)

            y_n = torch.cat(y_chunks_n, dim=0)  # [N_k]
            t_k = float(t_grid[k].item())
            y_local_n = y_n / problem.discount(t_k)

            # Warm start selection
            warm = pick_warm_start(n=n, k=k)
            is_transfer = (warm is not None)

            qnet_n = QNet(d_in=d_in, cfg=cfg.net).to(device=device, dtype=dtype)

            if is_transfer:
                qnet_n.load_state_dict(warm.state_dict())
                fit_qnet(qnet_n, Xk, y_local_n, cfg.net, verbose=False, is_transfer=True)
            else:
                fit_qnet(qnet_n, Xk, y_local_n, cfg.net, verbose=False, is_transfer=False)

            qhats_bounded[n][k] = qnet_n

        # For n > n_max_k, reuse the last available network at this time k
        for n in range(n_max_k + 1, max_imp + 1):
            qhats_bounded[n][k] = qhats_bounded[n_max_k][k]

        # Optional plotting on a diagonal slice if d >= 2
        if verbose:
            n_plot = max_imp
            plot_step_functions_bounded(
                k=k,
                n=n_plot,
                cfg=cfg,
                qhats=qhats_bounded,
                problem=problem,
                t_grid=t_grid,
                opt_cfg=cfg.opt,
                n_points=800,
            )

    return {
        "config": cfg,
        "problem": problem,
        "t_grid": t_grid,
        "qhats": qhats_bounded,
        "max_impulses": max_imp,
    }


@torch.no_grad()
def plot_step_functions_unconstrained(
    k: int,
    cfg: ExperimentConfig,
    qhats: List[Optional[QNet]],
    problem: DividendProblemND,
    t_grid: Tensor,
    opt_cfg: OptConfig,
    n_points: int = 500,
):
    """Plot unconstrained dividend step functions on a diagonal state slice."""
    import matplotlib.pyplot as plt

    device = cfg.device
    dtype = cfg.dtype
    d = cfg.dividend.state_dim

    # If Q_k not learned yet, skip
    if qhats[k] is None:
        print(f"[plot_step_functions_unconstrained] qhat[{k}] is None, skipping.")
        return

    qhat_k = qhats[k]

    # Build 1D grid in x, then lift to R^d if needed
    xmin = float(cfg.design.x_min_plot)
    xmax = float(cfg.design.x_max_plot)
    x_1d = torch.linspace(xmin, xmax, n_points, device=device, dtype=dtype).view(-1, 1)  # [n_points, 1]

    if d == 1:
        X_grid = x_1d  # [n_points, 1]
    else:
        # ND case: restrict to the diagonal (x, x, ..., x) in R^d
        X_grid = x_1d.expand(-1, d).contiguous()  # [n_points, d]

    # Q_hat_k(x): continuation value
    Q_vals = qhat_k(X_grid)  # [n_points]

    # M_hat_k(x): intervention value = max_xi { Q(x_post) + impulse_reward(xi) }
    t_k = float(t_grid[k].item())
    M_vals, _ = argmax_intervention_value(
        problem=problem,
        qhat=lambda x_in: qhat_k(x_in),
        t=t_k,
        x=X_grid,
        opt_cfg=opt_cfg,
    )

    # V_hat_k(x): max(Q_k, M_k) for a reward problem
    V_vals = torch.maximum(Q_vals, M_vals)

    # Plot
    plt.figure(figsize=(7, 4))
    xx = x_1d.view(-1).cpu().numpy()

    plt.plot(
        xx,
        Q_vals.detach().cpu().numpy(),
        label=rf"$\hat{{Q}}_{{{k}}}(x)$",
        lw=2,
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        xx,
        M_vals.detach().cpu().numpy(),
        label=rf"$\hat{{M}}_{{{k}}}(x)$",
        lw=2,
        linestyle=":",
        alpha=0.7,
    )
    plt.plot(
        xx,
        V_vals.detach().cpu().numpy(),
        label=rf"$\hat{{V}}_{{{k}}}(x)$",
        lw=2,
        linestyle="-",
        alpha=0.7,
    )

    if d == 1:
        plt.title(f"Reward functions at step k={k} (1D)")
        plt.xlabel("x")
    else:
        plt.title(f"Reward functions at step k={k} along diagonal x→(x,...,x), d={d}")
        plt.xlabel("x (diagonal slice)")

    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_step_functions_bounded(
    k: int,
    n: int,
    cfg: ExperimentConfig,
    qhats: List[List[Optional[QNet]]],
    problem: DividendProblemND,
    t_grid: Tensor,
    opt_cfg: OptConfig,
    n_points: int = 500,
):
    """Plot bounded dividend step functions on a diagonal state slice."""
    import matplotlib.pyplot as plt

    device = cfg.device
    dtype = cfg.dtype
    d = cfg.dividend.state_dim

    if qhats[n][k] is None:
        print(f"[plot_step_functions_bounded] qhat[{n}][{k}] is None, skipping.")
        return

    qhat_kn = qhats[n][k]

    # Build 1D grid
    xmin = float(cfg.design.x_min_plot)
    xmax = float(cfg.design.x_max_plot)
    x_1d = torch.linspace(xmin, xmax, n_points, device=device, dtype=dtype).view(-1, 1)  # [n_points, 1]

    if d == 1:
        X_grid = x_1d  # [n_points, 1]
    else:
        # Diagonal slice (x, ..., x)
        X_grid = x_1d.expand(-1, d).contiguous()  # [n_points, d]

    # Q_k^{(n)}(x): continuation value with n impulses remaining
    Q_vals = qhat_kn(X_grid)  # [n_points]

    # M_k^{(n)}(x): intervention value with n impulses remaining
    t_k = float(t_grid[k].item())

    if n > 0:
        qhat_knm1 = qhats[n - 1][k]
        if qhat_knm1 is None:
            raise RuntimeError(
                f"qhat[{n-1}][{k}] is None in plot_step_functions_bounded."
            )

        def continuation_n_minus_1(x_in: Tensor, qnet=qhat_knm1) -> Tensor:
            """Evaluate continuation with one fewer available impulse."""
            return qnet(x_in)

        M_vals, _ = argmax_intervention_value(
            problem=problem,
            qhat=continuation_n_minus_1,
            t=t_k,
            x=X_grid,
            opt_cfg=opt_cfg,
        )
    else:
        # n = 0: no intervention allowed, set M to -inf so V = Q
        M_vals = torch.full_like(Q_vals, float("-inf"))

    # V_k^{(n)}(x) = max(Q_k^{(n)}(x), M_k^{(n)}(x))
    V_vals = torch.maximum(Q_vals, M_vals)

    # Plot
    plt.figure(figsize=(7, 4))
    xx = x_1d.view(-1).cpu().numpy()

    plt.plot(
        xx,
        Q_vals.detach().cpu().numpy(),
        label=fr"$\hat{{Q}}_{{{k}}}^{{({n})}}(x)$",
        lw=2,
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        xx,
        M_vals.detach().cpu().numpy(),
        label=fr"$\hat{{M}}_{{{k}}}^{{({n})}}(x)$",
        lw=2,
        linestyle=":",
        alpha=0.7,
    )
    plt.plot(
        xx,
        V_vals.detach().cpu().numpy(),
        label=fr"$\hat{{V}}_{{{k}}}^{{({n})}}(x)$",
        lw=2,
        linestyle="-",
        alpha=0.7,
    )

    if d == 1:
        plt.title(f"Reward functions at step k={k}, n={n} (1D)")
        plt.xlabel("x")
    else:
        plt.title(f"Reward functions at step k={k}, n={n} along diagonal x→(x,...,x), d={d}")
        plt.xlabel("x (diagonal slice)")

    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _build_fine_grid_and_segcounts(t_grid: Tensor, dt_fine: Optional[float]) -> tuple[Tensor, List[int]]:
    """
    Build the fine time grid and the number of Euler substeps per coarse interval.
    """
    device, dtype = t_grid.device, t_grid.dtype
    K = t_grid.numel() - 1

    t_fine_list = [float(t_grid[0].item())]
    seg_n_list: List[int] = []

    for k in range(K):
        t0 = float(t_grid[k].item())
        t1 = float(t_grid[k + 1].item())
        Tseg = t1 - t0

        if dt_fine is None:
            nseg = 1
        else:
            nseg = max(1, math.ceil(Tseg / dt_fine))

        seg_n_list.append(nseg)
        dt_seg = Tseg / nseg
        for j in range(1, nseg + 1):
            t_fine_list.append(t0 + j * dt_seg)

    t_fine = torch.tensor(t_fine_list, device=device, dtype=dtype)
    return t_fine, seg_n_list


def precompute_euler_noise(
    *,
    t_grid: Tensor,
    n_sim: int,
    state_dim: int,
    dt_fine: Optional[float],
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Tensor]:
    """Generate reusable Euler noise on the fine simulation grid."""
    device = device if device is not None else t_grid.device
    dtype  = dtype  if dtype  is not None else t_grid.dtype

    t_fine, seg_n_list = _build_fine_grid_and_segcounts(t_grid, dt_fine)
    L = t_fine.numel()
    n_steps = L - 1

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    noise = torch.randn((n_sim, n_steps, state_dim), device=device, dtype=dtype, generator=g)

    return {
        "t_fine": t_fine,
        "noise": noise,
        "seg_n_list": torch.tensor(seg_n_list, device="cpu", dtype=torch.long),
    }


@torch.no_grad()
def simulate_controlled_paths(
    problem: "DividendProblemND",
    stepper: EulerStepper,
    policy: List[Optional[QNet]],
    t_grid: Tensor,
    x0: Tensor,                 # [B0, d]
    opt_cfg: OptConfig,
    n_sim: int = 1024,
    n_display: int = 32,
    dt_fine: Optional[float] = 1e-3,
    *,
    noise_pack: Optional[Dict[str, Tensor]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Simulate paths under the learned unconstrained policy."""
    device = x0.device
    dtype = x0.dtype

    K = t_grid.numel() - 1
    n_sim = min(n_sim, x0.shape[0])
    x = x0[:n_sim].clone()      # [n_sim, d]
    d = x.shape[1]

    if noise_pack is None and seed is not None:
        noise_pack = precompute_euler_noise(
            t_grid=t_grid,
            n_sim=n_sim,
            state_dim=d,
            dt_fine=dt_fine,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    if noise_pack is not None:
        t_fine = noise_pack["t_fine"].to(device=device, dtype=dtype)
        seg_n_list = noise_pack["seg_n_list"].tolist()
        noise_all = noise_pack["noise"].to(device=device, dtype=dtype)   # [n_sim, L-1, d]
    else:
        t_fine, seg_n_list = _build_fine_grid_and_segcounts(t_grid, dt_fine)
        noise_all = None

    L = t_fine.numel()

    paths_fine = torch.zeros((n_sim, L, d), device=device, dtype=dtype)
    impulse_mask = torch.zeros((n_sim, L, d), device=device, dtype=torch.bool)
    impulse_xi = torch.zeros((n_sim, L, d), device=device, dtype=dtype)
    rewards = torch.zeros((n_sim,), device=device, dtype=dtype)

    paths_fine[:, 0, :] = x

    idx_fine = 0
    idx_step = 0

    for k in range(K):
        t_k = float(t_grid[k].item())
        nseg = int(seg_n_list[k])
        dt_seg = (float(t_grid[k + 1].item()) - t_k) / nseg

        # PRE-diffusion decision at t_k
        if policy[k] is not None:
            qhat_fn = lambda x_in, qk=policy[k]: qk(x_in)

            # Feasibility is handled inside argmax_intervention_value
            m_val, xi_star = argmax_intervention_value(
                problem=problem,
                qhat=qhat_fn,
                t=t_k,
                x=x,
                opt_cfg=opt_cfg,
            )

            q_val = qhat_fn(x)
            do = (m_val > q_val)

            if torch.any(do):
                xi_do = xi_star[do]                      # [B_do, d]
                rewards[do] += (
                    problem.discount(t_k)
                    * problem.impulse_reward(t_k, xi_do)
                )
                x[do] = problem.apply_impulse(x[do], xi_do)

                impulse_mask[do, idx_fine, :] = (xi_do > 0.0)
                impulse_xi[do, idx_fine, :] = xi_do

        paths_fine[:, idx_fine, :] = x

        # Diffusion on fine grid between t_k and t_{k+1}
        t_cur = t_k
        for _ in range(nseg):
            if noise_all is None:
                x = stepper.step(problem, t_cur, x, dt_seg, noise=None)
            else:
                eps = noise_all[:, idx_step, :].view(n_sim, d)
                x = stepper.step(problem, t_cur, x, dt_seg, noise=eps)

            t_cur += dt_seg
            idx_fine += 1
            idx_step += 1
            paths_fine[:, idx_fine, :] = x

    # Terminal payoff at T
    tK = float(t_grid[-1].item())
    terminal_reward = problem.terminal_payoff(x)   # [n_sim]
    rewards += problem.discount(tK) * terminal_reward

    n_display = min(n_display, n_sim)
    idx_sel = torch.arange(n_display, device=device)

    return {
        "t_fine": t_fine,
        "paths_fine": paths_fine[idx_sel],
        "impulse_mask": impulse_mask[idx_sel],
        "impulse_xi": impulse_xi[idx_sel],
        "rewards": rewards,
        "rewards_mean": rewards.mean(),
        "rewards_std": rewards.std(),
        "noise_pack": noise_pack if noise_pack is not None else None,
    }


@torch.no_grad()
def simulate_controlled_paths_bounded(
    problem: "DividendProblemND",
    stepper: EulerStepper,
    policy_bounded: List[List[Optional[QNet]]],
    t_grid: Tensor,
    x0: Tensor,          # [B0, d]
    opt_cfg: OptConfig,
    n0: int,
    max_impulses: int,
    n_sim: int = 1024,
    n_display: int = 32,
    dt_fine: Optional[float] = 1e-3,
    *,
    noise_pack: Optional[Dict[str, Tensor]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Simulate bounded-policy paths with optional common random numbers."""
    if max_impulses < 0:
        raise ValueError("simulate_controlled_paths_bounded requires max_impulses >= 0.")
    if not (0 <= n0 <= max_impulses):
        raise ValueError(
            f"n0 must be in [0, max_impulses], got n0={n0}, max_impulses={max_impulses}."
        )

    device = x0.device
    dtype = x0.dtype

    K = t_grid.numel() - 1
    n_sim = min(n_sim, x0.shape[0])
    x = x0[:n_sim].clone()      # [n_sim, d]
    d = x.shape[1]

    # Impulses remaining per path
    n_rem = torch.full((n_sim,), n0, device=device, dtype=torch.long)

    if noise_pack is None and seed is not None:
        noise_pack = precompute_euler_noise(
            t_grid=t_grid,
            n_sim=n_sim,
            state_dim=d,
            dt_fine=dt_fine,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    if noise_pack is not None:
        t_fine = noise_pack["t_fine"].to(device=device, dtype=dtype)
        seg_n_list = noise_pack["seg_n_list"].tolist()
        noise_all = noise_pack["noise"].to(device=device, dtype=dtype)   # [n_sim, L-1, d]
    else:
        t_fine, seg_n_list = _build_fine_grid_and_segcounts(t_grid, dt_fine)
        noise_all = None

    L = t_fine.numel()

    paths_fine = torch.zeros((n_sim, L, d), device=device, dtype=dtype)
    impulse_mask = torch.zeros((n_sim, L, d), device=device, dtype=torch.bool)
    impulse_xi = torch.zeros((n_sim, L, d), device=device, dtype=dtype)
    rewards = torch.zeros((n_sim,), device=device, dtype=dtype)

    paths_fine[:, 0, :] = x

    idx_fine = 0
    idx_step = 0

    for k in range(K):
        t_k = float(t_grid[k].item())
        nseg = int(seg_n_list[k])
        dt_seg = (float(t_grid[k + 1].item()) - t_k) / nseg

        # PRE-diffusion decision at t_k, grouped by n_rem
        active = (n_rem > 0)
        if torch.any(active):
            n_min = int(n_rem[active].min().item())
            n_max = int(n_rem[active].max().item())

            for n_avail in range(n_min, n_max + 1):
                mask = active & (n_rem == n_avail)
                if not torch.any(mask):
                    continue

                idx_paths = torch.nonzero(mask).view(-1)
                x_sub = x[idx_paths]                         # [B_sub, d]

                qk_cont = policy_bounded[n_avail][k]
                if qk_cont is None:
                    raise RuntimeError(f"policy_bounded[{n_avail}][{k}] is None.")
                Q_vals = qk_cont(x_sub)

                qk_next = policy_bounded[n_avail - 1][k]
                if qk_next is None:
                    raise RuntimeError(f"policy_bounded[{n_avail - 1}][{k}] is None.")

                def qhat_post(x_in: Tensor, qnet=qk_next) -> Tensor:
                    """Evaluate continuation after a candidate intervention."""
                    return qnet(x_in)

                M_vals, xi_star = argmax_intervention_value(
                    problem=problem,
                    qhat=qhat_post,
                    t=t_k,
                    x=x_sub,
                    opt_cfg=opt_cfg,
                )

                do = (M_vals > Q_vals)
                if torch.any(do):
                    idx_do = idx_paths[do]
                    xi_do = xi_star[do]                     # [B_do, d]

                    rewards[idx_do] += (
                        problem.discount(t_k)
                        * problem.impulse_reward(t_k, xi_do)
                    )
                    x[idx_do] = problem.apply_impulse(x[idx_do], xi_do)
                    n_rem[idx_do] = n_avail - 1

                    impulse_mask[idx_do, idx_fine, :] = (xi_do > 0.0)
                    impulse_xi[idx_do, idx_fine, :] = xi_do

        paths_fine[:, idx_fine, :] = x

        # Diffusion on fine grid between t_k and t_{k+1}
        t_cur = t_k
        for _ in range(nseg):
            if noise_all is None:
                x = stepper.step(problem, t_cur, x, dt_seg, noise=None)
            else:
                eps = noise_all[:, idx_step, :].view(n_sim, d)
                x = stepper.step(problem, t_cur, x, dt_seg, noise=eps)

            t_cur += dt_seg
            idx_fine += 1
            idx_step += 1
            paths_fine[:, idx_fine, :] = x

    # Terminal payoff at T
    tK = float(t_grid[-1].item())
    terminal_reward = problem.terminal_payoff(x)   # [n_sim]
    rewards += problem.discount(tK) * terminal_reward

    n_display = min(n_display, n_sim)
    idx_sel = torch.arange(n_display, device=device)

    return {
        "t_fine": t_fine,
        "paths_fine": paths_fine[idx_sel],
        "impulse_mask": impulse_mask[idx_sel],
        "impulse_xi": impulse_xi[idx_sel],
        "rewards": rewards,
        "rewards_mean": rewards.mean(),
        "rewards_std": rewards.std(),
        "n_rem_final": n_rem,
        "noise_pack": noise_pack if noise_pack is not None else None,
    }


def plot_band_vs_nn_paths_nd(
    band_out: dict,
    nn_out: dict,
    n_display_per_fig: int = 3,
    n_total_to_plot: int = 9,
    figsize_per_dim: float = 2.5,
):
    """Compare exact band-policy and learned-policy paths by coordinate."""
    X_band = band_out["X_hist"]          # [n_paths_band, L_band, d]
    dt_band = float(band_out["dt"].item())
    n_paths_band, L_band, d = X_band.shape

    paths_nn = nn_out["paths_fine"]      # [n_paths_nn, L_nn, d]
    t_nn     = nn_out["t_fine"]          # [L_nn]
    n_paths_nn, L_nn, d_nn = paths_nn.shape

    if d != d_nn:
        raise ValueError(f"Dimension mismatch: band d={d}, nn d={d_nn}")

    # Align horizons by truncating to the minimum length
    L = min(L_band, L_nn)
    X_band = X_band[:, :L, :]                 # [n_paths_band, L, d]
    paths_nn = paths_nn[:, :L, :]             # [n_paths_nn, L, d]
    t = t_nn[:L].detach().cpu().numpy()       # [L]

    # Limit display to available paths.
    max_available = min(n_paths_band, n_paths_nn)
    if n_total_to_plot is None:
        n_total_to_plot = max_available
    else:
        n_total_to_plot = min(n_total_to_plot, max_available)

    if n_total_to_plot <= 0:
        print("[plot_band_vs_nn_paths_nd] Nothing to display (no paths).")
        return

    # Clip n_display_per_fig
    n_display_per_fig = max(1, n_display_per_fig)

    # Color maps for band and NN trajectories
    cmap_band = plt.get_cmap("Blues")
    cmap_nn   = plt.get_cmap("Reds")

    # Loop over chunks of size n_display_per_fig
    start = 0
    fig_idx = 0
    while start < n_total_to_plot:
        end = min(start + n_display_per_fig, n_total_to_plot)
        m = end - start  # number of paths in this figure

        fig_idx += 1

        # Create one subplot per dimension
        fig, axes = plt.subplots(
            nrows=d,
            ncols=1,
            figsize=(12, d * figsize_per_dim),
            sharex=True,
        )
        if d == 1:
            axes = [axes]

        for j in range(d):
            ax = axes[j]

            for local_i, global_i in enumerate(range(start, end)):
                # Colors for this path in this figure
                color_band = cmap_band((local_i + 1) / (m + 1))
                color_nn   = cmap_nn((local_i + 1) / (m + 1))

                # Band policy
                ax.plot(
                    t,
                    X_band[global_i, :, j].detach().cpu().numpy(),
                    linewidth=1.4,
                    alpha=0.9,
                    color=color_band,
                    label="band" if (j == 0 and local_i == 0 and start == 0) else None,
                )

                # NN policy
                ax.plot(
                    t,
                    paths_nn[global_i, :, j].detach().cpu().numpy(),
                    linewidth=1.4,
                    alpha=0.9,
                    linestyle="--",
                    color=color_nn,
                    label="NN" if (j == 0 and local_i == 0 and start == 0) else None,
                )
            if d==1 :
                ax.set_ylabel(f"$X_t$")
            else :
                ax.set_ylabel(f"$X_t^{{({j+1})}}$")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time t")
        axes[0].set_title(
            f"ND trajectories band (blue) vs NN (red) "
            f"[paths {start} to {end-1}]"
        )

        # Legend only once (on first figure, first subplot)
        if start == 0:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc="best")

        plt.tight_layout()
        plt.show()

        start = end

def plot_bounded_paths_nd(
    nn_out: dict,
    max_imp_traj: int,
    n_display_per_fig: int = 3,
    n_total_to_plot: int = 9,
    figsize_per_dim: float = 2.5,
):
    """Plot bounded learned-policy paths by coordinate."""
    paths_nn = nn_out["paths_fine"]      # [n_paths, L, d]
    t_nn     = nn_out["t_fine"]          # [L]
    n_paths, L, d = paths_nn.shape

    # Time grid for plotting
    t = t_nn.detach().cpu().numpy()

    # Limit display to available paths.
    max_available = n_paths
    if n_total_to_plot is None:
        n_total_to_plot = max_available
    else:
        n_total_to_plot = min(n_total_to_plot, max_available)

    if n_total_to_plot <= 0:
        print("[plot_bounded_paths_nd] Nothing to display (no paths).")
        return

    n_display_per_fig = max(1, n_display_per_fig)

    cmap_nn = plt.get_cmap("Reds")

    start = 0
    fig_idx = 0
    while start < n_total_to_plot:
        end = min(start + n_display_per_fig, n_total_to_plot)
        m = end - start  # number of paths in this figure
        fig_idx += 1

        # Fixed-size figure, independent of d
        fig, axes = plt.subplots(
            d, 1,
            figsize=(12, d*figsize_per_dim),
            sharex=True,
        )
        if d == 1:
            axes = [axes]

        for j in range(d):
            ax = axes[j]

            for local_i, global_i in enumerate(range(start, end)):
                color_nn = cmap_nn((local_i + 1) / (m + 1))

                ax.plot(
                    t,
                    paths_nn[global_i, :, j].detach().cpu().numpy(),
                    linewidth=1.6,
                    alpha=0.9,
                    linestyle="-",
                    color=color_nn,
                    label="NN bounded" if (j == 0 and local_i == 0 and start == 0) else None,
                )
            if d==1 :
                ax.set_ylabel(f"$X_t$")
            else :
                ax.set_ylabel(f"$X_t^{{({j+1})}}$")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time t")
        axes[0].set_title(
            f"ND bounded trajectories (NN policy, max_impulses = {max_imp_traj}) "
            f"[paths {start} to {end-1}]"
        )

        # Legend only once
        if start == 0:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc="best")

        plt.tight_layout()
        plt.show()

        start = end

def plot_bounded_paths_nd_dividend(
    nn_out: dict,
    max_imp_traj: int,
    n_display_per_fig: int = 3,
    n_total_to_plot: int = 9,
    figsize_per_dim: float = 2.5,
):
    """
    Plot ND trajectories under a bounded-impulse NN dividend policy.

    nn_out:
        dict from simulate_controlled_paths_bounded, with keys:
            - "paths_fine": [n_paths, L, d]
            - "t_fine": [L]
    """
    paths_nn = nn_out["paths_fine"]      # [n_paths, L, d]
    t_nn = nn_out["t_fine"]              # [L]
    n_paths, L, d = paths_nn.shape

    t = t_nn.detach().cpu().numpy()

    max_available = n_paths
    if n_total_to_plot is None:
        n_total_to_plot = max_available
    else:
        n_total_to_plot = min(n_total_to_plot, max_available)

    if n_total_to_plot <= 0:
        print("[plot_bounded_paths_nd_dividend] Nothing to display.")
        return

    n_display_per_fig = max(1, n_display_per_fig)
    cmap_nn = plt.get_cmap("Reds")

    start = 0
    fig_idx = 0
    while start < n_total_to_plot:
        end = min(start + n_display_per_fig, n_total_to_plot)
        m = end - start
        fig_idx += 1

        fig, axes = plt.subplots(
            d,
            1,
            figsize=(12, d * figsize_per_dim),
            sharex=True,
        )
        if d == 1:
            axes = [axes]

        for j in range(d):
            ax = axes[j]

            for local_i, global_i in enumerate(range(start, end)):
                color_nn = cmap_nn((local_i + 1) / (m + 1))

                ax.plot(
                    t,
                    paths_nn[global_i, :, j].detach().cpu().numpy(),
                    linewidth=1.6,
                    alpha=0.9,
                    linestyle="-",
                    color=color_nn,
                    label="NN bounded policy" if (j == 0 and local_i == 0 and start == 0) else None,
                )

            if d == 1:
                ax.set_ylabel(r"$X_t$")
            else:
                ax.set_ylabel(rf"$X_t^{{({j+1})}}$")

            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time t")
        axes[0].set_title(
            f"ND bounded dividend trajectories under NN policy "
            f"(max_impulses = {max_imp_traj}) [paths {start} to {end-1}]"
        )

        if start == 0:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc="best")

        plt.tight_layout()
        plt.show()

        start = end
