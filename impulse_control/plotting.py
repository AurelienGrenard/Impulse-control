"""Plot value functions, policy scores, and controlled trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from .reproducibility import seed_everything


# Grayscale tones and dash patterns remain distinct in print.
INK = "#111111"
GRAYS = ("#2B2B2B", "#555555", "#777777", "#969696", "#B0B0B0", "#686868")
LINESTYLES = (
    "-",
    (0, (7, 2.5)),
    (0, (2, 2)),
    (0, (7, 2, 1.5, 2)),
    (0, (1, 1.5)),
    (0, (4, 1.5, 1, 1.5)),
)
FIGURE_SIZE = (8.0, 5.0)
ARTICLE_TEXT_SIZE = 20


def use_white_style() -> None:
    """Apply the shared white-background figure style."""
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import InlineBackend

        if get_ipython() is not None:
            InlineBackend.instance().print_figure_kwargs = {"bbox_inches": None}
    except ImportError:
        pass
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D4D4D4",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": ARTICLE_TEXT_SIZE,
            "axes.labelsize": ARTICLE_TEXT_SIZE,
            "axes.labelpad": 7,
            "xtick.labelsize": ARTICLE_TEXT_SIZE,
            "ytick.labelsize": ARTICLE_TEXT_SIZE,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "axes.linewidth": 0.9,
            "legend.fontsize": 16,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#C8C8C8",
            "legend.framealpha": 0.94,
            "legend.borderpad": 0.45,
            "legend.labelspacing": 0.45,
            "legend.handlelength": 2.5,
            "lines.linewidth": 1.9,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.04,
        }
    )


def _finish(fig, *, output: Optional[str], show: bool):
    """Finalize, save, and optionally display a figure."""
    for ax in fig.axes:
        ax.minorticks_on()
        ax.tick_params(which="both", width=0.9)
        ax.grid(which="minor", visible=False)
    fig.tight_layout(pad=0.65)
    if output:
        target = Path(output)
        if not target.is_absolute():
            target = Path(__file__).resolve().parents[1] / target
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
    if show:
        plt.show()
    return fig


def plot_limited_summary(
    bundle: Dict[str, Any],
    *,
    output: Optional[str] = None,
    show: bool = True,
):
    """Compare every bounded value function with the unlimited benchmark."""
    use_white_style()
    results = sorted(bundle["bounded_results"], key=lambda item: item["max_impulses"])
    unlimited = bundle["unconstrained"]
    if not results or unlimited is None:
        raise ValueError("The bundle must contain bounded and unconstrained results.")

    x_min = max(float(np.min(result["x"])) for result in results)
    x_max = min(float(np.max(result["x"])) for result in results)
    x = np.linspace(x_min, x_max, 700)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        x,
        np.interp(x, unlimited["x_np_inf"], np.asarray(unlimited["V_np_inf"]).reshape(-1)),
        color=INK,
        linewidth=2.3,
        linestyle=(0, (9, 3)),
        label=r"$\hat V^{(\infty)}(0,x)$",
        zorder=5,
    )
    for index, result in enumerate(results):
        ax.plot(
            x,
            np.interp(x, result["x"], np.asarray(result["V"]).reshape(-1)),
            color=GRAYS[index % len(GRAYS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=rf"$\hat V^{{({result['max_impulses']})}}(0,x)$",
        )
    ax.set(xlabel=r"Diagonal state $x$", ylabel=r"Estimated value $\hat V(0,x)$")
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0,
        handlelength=2.4,
    )
    ax.margins(x=0)
    return _finish(fig, output=output, show=show)


def plot_unlimited_summary(
    results: Iterable[Dict[str, Any]],
    *,
    output: Optional[str] = None,
    show: bool = True,
):
    """Compare unlimited neural value functions with the exact solution."""
    use_white_style()
    ordered = sorted(results, key=lambda item: item["T"])
    if not ordered:
        raise ValueError("At least one unlimited result is required.")

    cfg = ordered[0]["cfg"]
    if hasattr(cfg, "dividend"):
        from .exact_dividend import solve_psi_dividend

        params = cfg.dividend
        exact_1d, _, _ = solve_psi_dividend(
            mu=float(params.mu[0].item()),
            sigma=float(params.sigma[0].item()),
            rho=float(params.rho),
            lam=float(params.lam[0].item()),
            c=float(params.c[0].item()),
        )
    else:
        from .exact_harvesting import solve_psi_general

        params = cfg.harvesting
        exact_1d, _, _ = solve_psi_general(
            mu=float(params.mu[0].item()),
            sigma=float(params.sigma[0].item()),
            rho=float(params.rho),
            alpha=float(params.alpha[0].item()),
            x0=float(params.x0[0].item()),
            lam=float(params.lam[0].item()),
            c=float(params.c[0].item()),
        )

    dimension = int(params.state_dim)
    x_min = max(float(np.min(result["x"])) for result in ordered)
    x_max = min(float(np.max(result["x"])) for result in ordered)
    x = np.linspace(x_min, x_max, 700)
    exact = (
        dimension
        * exact_1d(torch.as_tensor(x, dtype=torch.float64)).detach().cpu().numpy()
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        x,
        exact,
        color=INK,
        linewidth=2.35,
        linestyle="-",
        label="Closed-form",
        zorder=5,
    )
    for index, result in enumerate(ordered):
        ax.plot(
            x,
            np.interp(x, result["x"], np.asarray(result["V"]).reshape(-1)),
            color=GRAYS[index % len(GRAYS)],
            linestyle=LINESTYLES[(index + 1) % len(LINESTYLES)],
            label=rf"$T={result['T']:g}$",
        )
    ax.set(xlabel=r"Diagonal state $x$", ylabel=r"Estimated value $\hat V(0,x)$")
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0,
        handlelength=2.4,
    )
    ax.margins(x=0)
    return _finish(fig, output=output, show=show)


def plot_unlimited_policy_consistency(
    results: Iterable[Dict[str, Any]],
    *,
    comparison_rows: Optional[Iterable[Dict[str, Any]]] = None,
    output: Optional[str] = None,
    show: bool = True,
):
    """Compare learned-policy and band-policy Monte Carlo scores."""
    use_white_style()
    ordered = sorted(results, key=lambda item: item["T"])
    if not ordered:
        raise ValueError("At least one unlimited result is required.")

    cfg = ordered[0]["cfg"]
    evaluation_counts = {int(result.get("mc_n_NN", 500)) for result in ordered}
    if len(evaluation_counts) != 1:
        raise ValueError("All maturities must use the same policy-evaluation size.")
    n_sim_band = evaluation_counts.pop()
    if hasattr(cfg, "dividend"):
        from .exact_dividend import simulate_dividend_policy_nd

        params = cfg.dividend
        score_key = "REWARD_total_per_path"
        ylabel = "Reward"

        def simulate_band(horizon: float):
            """Simulate the exact band policy for one horizon."""
            return simulate_dividend_policy_nd(
                mu=float(params.mu[0].item()),
                sigma=float(params.sigma[0].item()),
                rho=float(params.rho),
                lam=float(params.lam[0].item()),
                c_f=float(params.c[0].item()),
                d=int(params.state_dim),
                x_init=1.0,
                dt=2e-3,
                T_max=horizon,
                n_paths=n_sim_band,
                n_impulses_max=2000,
                seed=1234,
                device=cfg.device,
                absorb_at_zero=True,
            )

    else:
        from .exact_harvesting import simulate_band_policy_nd

        params = cfg.harvesting
        score_key = "COST_total_per_path"
        ylabel = "Cost"

        def simulate_band(horizon: float):
            """Simulate the exact band policy for one horizon."""
            return simulate_band_policy_nd(
                mu=float(params.mu[0].item()),
                sigma=float(params.sigma[0].item()),
                rho=float(params.rho),
                alpha=float(params.alpha[0].item()),
                x0=float(params.x0[0].item()),
                lam=float(params.lam[0].item()),
                c_f=float(params.c[0].item()),
                d=int(params.state_dim),
                x_init=1.0,
                dt=2e-3,
                T_max=horizon,
                n_paths=n_sim_band,
                n_impulses_max=2000,
                seed=1234,
                device=cfg.device,
            )

    horizons = np.asarray([result["T"] for result in ordered], dtype=float)
    if comparison_rows is None:
        band_means = []
        band_stds = []
        for result in ordered:
            scores = simulate_band(float(result["T"]))[score_key].detach().cpu().numpy()
            band_means.append(float(scores.mean()))
            band_stds.append(float(scores.std()))
        learned_means = np.asarray(
            [result["mc_mean_NN"] for result in ordered], dtype=float
        )
        learned_counts = np.asarray(
            [result.get("mc_n_NN", 500) for result in ordered], dtype=float
        )
        learned_stds = np.asarray(
            [result["mc_std_NN"] for result in ordered], dtype=float
        )
        learned_errors = 1.96 * learned_stds / np.sqrt(learned_counts)
        band_means_np = np.asarray(band_means)
        band_errors = 1.96 * np.asarray(band_stds) / np.sqrt(n_sim_band)
    else:
        rows = sorted(comparison_rows, key=lambda item: float(item["T"]))
        row_horizons = np.asarray([float(row["T"]) for row in rows])
        if len(rows) != len(ordered) or not np.array_equal(row_horizons, horizons):
            raise ValueError("Comparison rows must match the checkpoint horizons.")
        learned_means = np.asarray([float(row["learned"]) for row in rows])
        learned_errors = np.asarray([float(row["learned_ci95"]) for row in rows])
        band_means_np = np.asarray([float(row["annual_band"]) for row in rows])
        band_errors = np.asarray([float(row["annual_band_ci95"]) for row in rows])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.errorbar(
        horizons,
        learned_means,
        yerr=learned_errors,
        fmt="x-",
        markersize=8.0,
        markeredgewidth=1.8,
        linewidth=1.65,
        capsize=3,
        capthick=1.1,
        color=INK,
        zorder=4,
        label="Learned policy",
    )
    ax.errorbar(
        horizons,
        band_means_np,
        yerr=band_errors,
        fmt="o--",
        markersize=9.0,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=1.5,
        linewidth=1.65,
        capsize=3,
        capthick=1.1,
        color="#5F5F5F",
        zorder=3,
        label="Band policy",
    )
    ax.set(xlabel=r"Horizon $T$", ylabel=ylabel)
    ax.legend(
        ncol=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0,
    )
    ax.margins(x=0.05)
    return _finish(fig, output=output, show=show)


def plot_limited_paths(
    bundle: Dict[str, Any],
    *,
    seed: int = 123,
    dt_fine: float = 2e-3,
    output_prefix: Optional[str] = None,
    show: bool = True,
):
    """Plot shared-Brownian controlled paths without exposing bundle internals."""
    seed_everything(seed)
    use_white_style()
    unlimited = bundle["unconstrained"]
    results = sorted(bundle["bounded_results"], key=lambda item: item["max_impulses"])
    cfg = unlimited["config"]
    application = "dividend" if hasattr(cfg, "dividend") else "harvesting"
    module = __import__(f"impulse_control.{application}", fromlist=["EulerStepper"])
    params = getattr(cfg, application)
    dimension = int(params.state_dim)
    x0 = torch.ones((1, dimension), device=cfg.device, dtype=cfg.dtype)
    noise = module.precompute_euler_noise(
        t_grid=unlimited["t_grid"],
        n_sim=1,
        state_dim=dimension,
        dt_fine=dt_fine,
        seed=seed,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    simulations = [
        (
            r"$n=\infty$",
            module.simulate_controlled_paths(
                problem=unlimited["problem"],
                stepper=module.EulerStepper(),
                policy=unlimited["qhats"],
                t_grid=unlimited["t_grid"],
                x0=x0,
                opt_cfg=cfg.opt,
                n_sim=1,
                n_display=1,
                dt_fine=dt_fine,
                noise_pack=noise,
            ),
        )
    ]
    for result in results:
        n = int(result["max_impulses"])
        simulations.append(
            (
                f"$n={n}$",
                module.simulate_controlled_paths_bounded(
                    problem=result["problem"],
                    stepper=module.EulerStepper(),
                    policy_bounded=result["qhats"],
                    t_grid=result["t_grid"],
                    x0=x0,
                    opt_cfg=result["cfg"].opt,
                    n0=n,
                    max_impulses=n,
                    n_sim=1,
                    n_display=1,
                    dt_fine=dt_fine,
                    noise_pack=noise,
                ),
            )
        )
    time = noise["t_fine"].cpu().numpy()
    figures = []
    for coordinate in range(dimension):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for index, (label, simulation) in enumerate(simulations):
            path = simulation["paths_fine"][0, :, coordinate].detach().cpu().numpy()
            ax.plot(
                time,
                path,
                color=(INK, *GRAYS)[index % (len(GRAYS) + 1)],
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                linewidth=1.15,
                label=label,
                alpha=0.92,
            )
        ax.set(
            xlabel="Time",
            ylabel=rf"$X_t^{{({coordinate + 1})}}$" if dimension > 1 else r"$X_t$",
        )
        ax.legend(
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            borderaxespad=0,
        )
        ax.margins(x=0)
        output = (
            f"{output_prefix}_path_coordinate_{coordinate + 1}.png"
            if output_prefix
            else None
        )
        figures.append(_finish(fig, output=output, show=show))
    return figures
