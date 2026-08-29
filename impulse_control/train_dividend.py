"""Command-line training entry point for the dividend example."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dividend import *
from .reproducibility import (
    published_training_parameters,
    seed_everything,
    write_run_manifest,
)
from .saving import (
    load_all_results_unlimited,
    load_results_bundle,
    save_all_results_unlimited,
    save_results_bundle,
)


def _validate_resume_result(
    result: dict,
    dimension: int,
    activation: str,
    negative_slope: float,
) -> None:
    """Reject a checkpoint that does not match the requested experiment."""
    cfg = result.get("cfg", result.get("config"))
    if cfg is None or cfg.dividend.state_dim != dimension:
        raise RuntimeError("Resume checkpoint has a different dividend dimension.")
    if cfg.net.activation != activation:
        raise RuntimeError("Resume checkpoint has a different network activation.")
    if activation == "leaky_relu" and cfg.net.negative_slope != negative_slope:
        raise RuntimeError("Resume checkpoint has a different LeakyReLU slope.")


def train_limited(
    dimension: int,
    output: str,
    requested_device: str,
    smoke_test: bool = False,
    activation: str = "leaky_relu",
    negative_slope: float = 0.01,
    progress: bool = False,
) -> None:
    # Device, dimension and global parameters

    """Train and save the bounded experiments for one dimension."""
    device = requested_device   # Computation device
    state_dim = dimension                                               # State dimension
    T_fixed = 25.0                                              # Fixed maturity
    max_imp_list = [1, 2, 3, 4, 5]                              # Tested impulse budgets
    n_sim_eval = 25                                             # MC paths for policy evaluation
    dt_fine_eval = 2e-3                                         # Fine Euler step for evaluation
    all_results = []                                            # Container for all experiment results

    # Hyperparameters

    steps = 20_000                                              # Full training iterations
    transfer_steps = 100                                        # Transfer learning iterations
    batch_size = 8192                                           # Training batch size
    n_global = 5_000                                            # Random impulse candidates
    n_global_batch = 512                                        # Candidate batch size
    min_rel_impulse = 0.4                                       # Sparsification threshold
    N_k = 100_000 if state_dim == 1 else 12_500                 # Regression states per date
    M_k = 1 if state_dim == 1 else 8                            # Rollouts averaged per state

    if smoke_test:
        steps = 2
        transfer_steps = 1
        batch_size = 32
        n_global = 16
        n_global_batch = 8
        N_k = 16
        M_k = 2
        max_imp_list = [1]
        n_sim_eval = 2
        dt_fine_eval = 0.1
    x_min = 0.0                                                 # Design domain lower bound
    x_max = 8.0                                                 # Design domain upper bound


    # Unconstrained dividend ND (infinite impulses)
    print(f"\n=== Dividend ND (d = {state_dim}), UNCONSTRAINED (T = {T_fixed}) ===")

    cfg_inf = ExperimentConfig(
        device=device,
        dtype=torch.float32,
        verbose=progress,
        max_impulses=-1,

        time=TimeGridConfig(
            T=T_fixed,
            K=int(T_fixed),
            dt_fine=1e-2,
        ),

        mc=MCConfig(
            M_k=M_k,
        ),

        opt=OptConfig(
            n_global=n_global,
            n_global_batch=n_global_batch,
            min_rel_impulse=min_rel_impulse,
        ),

        net=NetConfig(
            activation=activation,
            negative_slope=negative_slope,
            steps=steps,
            transfer_steps=transfer_steps,
            batch_size=batch_size,
            use_zero_projection=True,
        ),

        dividend=DividendParamsND(
            d=state_dim,
            device=device,
            dtype=torch.float32,
        ),

        design=DesignConfig(
            N_k=N_k,
            x_min=0.0,
            x_max=8.0,
        ),
    )

    output_path = Path(output)
    loaded_unconstrained = None
    if output_path.is_file():
        loaded_bundle = load_results_bundle(output, map_location=device)
        loaded_unconstrained = loaded_bundle["unconstrained"]
        all_results = loaded_bundle["bounded_results"]
        if loaded_unconstrained is None:
            raise RuntimeError("Resume checkpoint is missing its unconstrained result.")
        _validate_resume_result(loaded_unconstrained, state_dim, activation, negative_slope)
        for result in all_results:
            _validate_resume_result(result, state_dim, activation, negative_slope)

    res_inf_nd = loaded_unconstrained or train_dividend_unconstrained(cfg_inf, verbose=False)

    problem_inf = res_inf_nd["problem"]
    t_grid_inf  = res_inf_nd["t_grid"]
    qhats_inf   = res_inf_nd["qhats"]

    # Evaluate the unconstrained value at the initial state.
    x_min_plot_inf = cfg_inf.design.x_min_plot
    x_max_plot_inf = cfg_inf.design.x_max_plot
    n_x_inf = 400

    x_grid_1d_inf = torch.linspace(
        x_min_plot_inf,
        x_max_plot_inf,
        n_x_inf,
        device=cfg_inf.device,
        dtype=cfg_inf.dtype,
    ).view(-1, 1)

    if state_dim == 1:
        X_grid_inf = x_grid_1d_inf
    else:
        X_grid_inf = x_grid_1d_inf.expand(-1, state_dim)

    with torch.no_grad():

        V_vals_inf = vhat_unconstrained(
            k=0,
            x=X_grid_inf,
            qhats=qhats_inf,
            problem=problem_inf,
            opt_cfg=cfg_inf.opt,
            t_grid=t_grid_inf,
        )

        x0_tensor_inf = torch.full(
            (1, state_dim),
            1.0,
            device=cfg_inf.device,
            dtype=cfg_inf.dtype,
        )

        V0_1_inf = vhat_unconstrained(
            k=0,
            x=x0_tensor_inf,
            qhats=qhats_inf,
            problem=problem_inf,
            opt_cfg=cfg_inf.opt,
            t_grid=t_grid_inf,
        ).item()

    x_np_inf = x_grid_1d_inf.view(-1).cpu().numpy()
    V_np_inf = V_vals_inf.cpu().numpy()

    print(
        f"UNCONSTRAINED | "
        f"V_hat^infinity(0,(1,...,1)) = {V0_1_inf:10.6f}"
    )

    # Useful for plots later
    res_inf_nd["x_np_inf"] = x_np_inf
    res_inf_nd["V_np_inf"] = V_np_inf
    res_inf_nd["V0_1_inf"] = V0_1_inf
    save_results_bundle(output, bounded_results=all_results, unconstrained=res_inf_nd)


    # Bounded dividend ND
    completed_budgets = {int(result["max_impulses"]) for result in all_results}
    for n_imp in max_imp_list:

        if n_imp in completed_budgets:
            print(f"Skipping completed dividend budget n={n_imp}.")
            continue

        print(
            f"\n=== Dividend ND (d = {state_dim}), "
            f"training BOUNDED with max_impulses = {n_imp} (T = {T_fixed}) ==="
        )

        cfg_n = ExperimentConfig(
            device=device,
            dtype=torch.float32,
            verbose=progress,
            max_impulses=n_imp,

            time=TimeGridConfig(
                T=T_fixed,
                K=int(T_fixed),
                dt_fine=1e-2,
            ),

            mc=MCConfig(
                M_k=M_k,
            ),

            opt=OptConfig(
                n_global=n_global,
                n_global_batch=n_global_batch,
                min_rel_impulse=min_rel_impulse,
            ),

            net=NetConfig(
                activation=activation,
                negative_slope=negative_slope,
                steps=steps,
                transfer_steps=transfer_steps,
                batch_size=batch_size,
                use_zero_projection=True,
            ),

            dividend=DividendParamsND(
                d=state_dim,
                device=device,
                dtype=torch.float32,
            ),

            design=DesignConfig(
                N_k=N_k,
                x_min=0.0,
                x_max=8.0,
            ),
        )

        res_n = train_dividend_bounded(cfg_n, verbose=False)

        problem_n = res_n["problem"]
        t_grid_n  = res_n["t_grid"]
        qhats_n   = res_n["qhats"]

        d_n = cfg_n.dividend.state_dim

        x_min_plot = cfg_n.design.x_min_plot
        x_max_plot = cfg_n.design.x_max_plot

        n_x = 400
        x_grid_1d = torch.linspace(
            x_min_plot,
            x_max_plot,
            n_x,
            device=cfg_n.device,
            dtype=cfg_n.dtype,
        ).view(-1, 1)

        if d_n == 1:
            X_grid_n = x_grid_1d
        else:
            X_grid_n = x_grid_1d.expand(-1, d_n)

        with torch.no_grad():

            V_vals = vhat_bounded(
                k=0,
                n=n_imp,
                x=X_grid_n,
                qhats=qhats_n,
                problem=problem_n,
                opt_cfg=cfg_n.opt,
                t_grid=t_grid_n,
            )

            x0_tensor_n = torch.full(
                (1, d_n),
                1.0,
                device=cfg_n.device,
                dtype=cfg_n.dtype,
            )

            V0_1 = vhat_bounded(
                k=0,
                n=n_imp,
                x=x0_tensor_n,
                qhats=qhats_n,
                problem=problem_n,
                opt_cfg=cfg_n.opt,
                t_grid=t_grid_n,
            ).item()

        x_np = x_grid_1d.view(-1).cpu().numpy()
        V_np = V_vals.cpu().numpy()

        # Monte Carlo evaluation
        x0_paths_n = torch.full(
            (n_sim_eval, d_n),
            1.0,
            device=cfg_n.device,
            dtype=cfg_n.dtype,
        )

        sim_NN = simulate_controlled_paths_bounded(
            problem=problem_n,
            stepper=EulerStepper(),
            policy_bounded=qhats_n,
            t_grid=t_grid_n,
            x0=x0_paths_n,
            opt_cfg=cfg_n.opt,
            n0=n_imp,
            max_impulses=n_imp,
            n_sim=n_sim_eval,
            n_display=3,
            dt_fine=dt_fine_eval,
        )

        vals = sim_NN["rewards"].detach().cpu().numpy()
        mc_mean_NN = float(vals.mean())
        mc_std_NN  = float(vals.std())

        print(
            f"max_impulses = {n_imp:2d} | "
            f"V_hat^{n_imp}(0,(1,...,1)) = {V0_1:10.6f} | "
            f"MC NN reward = {mc_mean_NN:10.6f} ± {mc_std_NN:8.6f}"
        )

        all_results.append(
            {
                "max_impulses": n_imp,
                "cfg": cfg_n,
                "problem": problem_n,
                "t_grid": t_grid_n,
                "qhats": qhats_n,
                "x": x_np,
                "V": V_np,
                "V0_1": V0_1,
                "mc_mean_NN": mc_mean_NN,
                "mc_std_NN": mc_std_NN,
                "mc_n_NN": n_sim_eval,
            }
        )

        save_results_bundle(output, bounded_results=all_results, unconstrained=res_inf_nd)

    save_results_bundle(output, bounded_results=all_results, unconstrained=res_inf_nd)



def train_unlimited(
    dimension: int,
    output: str,
    requested_device: str,
    smoke_test: bool = False,
    activation: str = "leaky_relu",
    negative_slope: float = 0.01,
    progress: bool = False,
) -> None:
    # Device, dimension and global parameters

    """Train and save the unconstrained experiments for one dimension."""
    device = requested_device   # Computation device
    d_state = dimension                                                 # State dimension
    T_list = [5.0, 10.0, 25.0, 50.0, 100.0]                     # Tested maturities
    n_sim_eval = 500                                            # MC paths for policy evaluation
    dt_fine_eval = 2e-3                                         # Fine Euler step for evaluation
    all_results = []                                            # Container for all experiment results

    # Hyperparameters

    steps = 20_000                                              # Full training iterations
    transfer_steps = 100                                        # Transfer learning iterations
    batch_size = 8192                                           # Training batch size
    n_global = 5_000                                            # Random impulse candidates
    n_global_batch = 512                                        # Candidate batch size
    min_rel_impulse = 0.4                                       # Sparsification threshold
    N_k = 100_000 if d_state == 1 else 12_500                   # Regression states per date
    M_k = 1 if d_state == 1 else 8                              # Rollouts averaged per state

    if smoke_test:
        steps = 2
        transfer_steps = 1
        batch_size = 32
        n_global = 16
        n_global_batch = 8
        N_k = 16
        M_k = 2
        T_list = [1.0]
        n_sim_eval = 2
        dt_fine_eval = 0.1
    x_min = 0.0                                                 # Design domain lower bound
    x_max = 8.0                                                 # Design domain upper bound

    output_path = Path(output)
    if output_path.is_file():
        all_results = load_all_results_unlimited(output, map_location=device)
        for result in all_results:
            _validate_resume_result(result, d_state, activation, negative_slope)
    completed_horizons = {float(result["T"]) for result in all_results}


    for T_val in T_list:

        if T_val in completed_horizons:
            print(f"Skipping completed dividend horizon T={T_val:g}.")
            continue

        print(f"\n=== Dividend ND, training for T = {T_val} (d = {d_state}) ===")

        horizon_N_k = N_k
        horizon_M_k = M_k
        horizon_n_global = n_global
        horizon_transfer_steps = transfer_steps
        horizon_transfer_lr = None
        if not smoke_test:
            published = published_training_parameters(
                "dividend", "unlimited", d_state, T_val
            )
            horizon_N_k = int(published["design_states"])
            horizon_M_k = int(published["rollouts_per_state"])
            horizon_n_global = int(published["randomized_candidates"])
            horizon_transfer_steps = int(published["transfer_steps"])
            horizon_transfer_lr = published["transfer_lr"]

        # Build experiment configuration
        cfg_T = ExperimentConfig(
            device=device,
            dtype=torch.float32,
            verbose=progress,

            time=TimeGridConfig(
                T=T_val,
                K=int(T_val),
                dt_fine=1e-2,
            ),

            mc=MCConfig(
                M_k=horizon_M_k,
                chunk_size_x=8192,
            ),

            opt=OptConfig(
                n_global=horizon_n_global,
                n_global_batch=n_global_batch,
                min_rel_impulse=min_rel_impulse,
            ),

            net=NetConfig(
                depth=3,
                width=128,
                activation=activation,
                negative_slope=negative_slope,
                steps=steps,
                transfer_steps=horizon_transfer_steps,
                transfer_lr=horizon_transfer_lr,
                batch_size=batch_size,
                use_zero_projection=True,
            ),

            dividend=DividendParamsND(
                d=d_state,
                device=device,
                dtype=torch.float32,
            ),

            design=DesignConfig(
                x_min=x_min,
                x_max=x_max,
                x_min_plot=0.0,
                x_max_plot=2.0,
                N_k=horizon_N_k,
            ),
        )

        # Backward RMC training (unconstrained ND)
        res_T = train_dividend_unconstrained(cfg_T, verbose=False)

        problem_T = res_T["problem"]
        t_grid_T  = res_T["t_grid"]
        qhats_T   = res_T["qhats"]

        # Evaluate V_hat(0,x) on diagonal grid
        x_min_plot = cfg_T.design.x_min_plot
        x_max_plot = cfg_T.design.x_max_plot

        n_x = 400

        x_grid_1d = torch.linspace(
            x_min_plot,
            x_max_plot,
            n_x,
            device=cfg_T.device,
            dtype=cfg_T.dtype,
        ).view(-1, 1)

        if d_state == 1:
            x_grid = x_grid_1d
        else:
            x_grid = x_grid_1d.expand(-1, d_state)

        with torch.no_grad():

            V_vals = vhat_unconstrained(
                k=0,
                x=x_grid,
                qhats=qhats_T,
                problem=problem_T,
                opt_cfg=cfg_T.opt,
                t_grid=t_grid_T,
            )

            x0_tensor = torch.full(
                (1, d_state),
                1.0,
                device=cfg_T.device,
                dtype=cfg_T.dtype,
            )

            V0_1 = vhat_unconstrained(
                k=0,
                x=x0_tensor,
                qhats=qhats_T,
                problem=problem_T,
                opt_cfg=cfg_T.opt,
                t_grid=t_grid_T,
            ).item()

        x_np = x_grid_1d.view(-1).cpu().numpy()
        V_np = V_vals.cpu().numpy()

        # Monte Carlo evaluation of learned NN policy
        x0_paths = torch.full(
            (n_sim_eval, d_state),
            1.0,
            device=cfg_T.device,
            dtype=cfg_T.dtype,
        )

        sim_NN = simulate_controlled_paths(
            problem=problem_T,
            stepper=EulerStepper(),
            policy=qhats_T,
            t_grid=t_grid_T,
            x0=x0_paths,
            opt_cfg=cfg_T.opt,
            n_sim=n_sim_eval,
            n_display=3,
            dt_fine=dt_fine_eval,
        )

        rewards_NN = sim_NN["rewards"].detach().cpu().numpy()

        mc_mean_NN = float(rewards_NN.mean())
        mc_std_NN  = float(rewards_NN.std())

        print(
            f"T = {T_val:6.1f} | "
            f"V_hat(0,1) = {V0_1:10.6f} | "
            f"MC NN reward = {mc_mean_NN:10.6f} ± {mc_std_NN:8.6f}"
        )

        # Store results
        all_results.append(
            {
                "T": T_val,
                "cfg": cfg_T,
                "problem": problem_T,
                "t_grid": t_grid_T,
                "qhats": qhats_T,
                "x": x_np,
                "V": V_np,
                "V0_1": V0_1,
                "mc_mean_NN": mc_mean_NN,
                "mc_std_NN": mc_std_NN,
                "mc_n_NN": n_sim_eval,
            }
        )

        save_all_results_unlimited(output, all_results)

    save_all_results_unlimited(output, all_results)



def main() -> None:
    """Parse command-line arguments and launch the selected training mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("limited", "unlimited"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--activation",
        choices=("leaky_relu", "softplus"),
        default="leaky_relu",
        help="Continuation-network activation (default: leaky_relu).",
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.01,
        help="LeakyReLU negative slope (default: 0.01).",
    )
    parser.add_argument("--progress", action="store_true", help="Report progress and ETA after every date.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the same pipeline with tiny validation sizes.")
    args = parser.parse_args()
    seed_everything(args.seed)
    trainer = train_limited if args.mode == "limited" else train_unlimited
    trainer(
        args.dimension,
        args.output,
        args.device,
        args.smoke_test,
        args.activation,
        args.negative_slope,
        args.progress,
    )
    write_run_manifest(
        args.output,
        application="dividend",
        mode=args.mode,
        dimension=args.dimension,
        seed=args.seed,
        device=args.device,
        smoke_test=args.smoke_test,
        activation=args.activation,
        negative_slope=args.negative_slope,
        randomized_candidates=16 if args.smoke_test else 5_000,
    )


if __name__ == "__main__":
    main()
