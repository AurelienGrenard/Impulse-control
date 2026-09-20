# Neural Regression and Randomized Optimization for Impulse Control

This repository contains the code, pretrained checkpoints, and post-processing
scripts for the numerical experiments in *Neural Regression and Randomized
Optimization for Impulse Control*.

Authors: Lokman Abbas Turki, Aurélien Grenard, Idris Kharroubi, Qinghua Li, and
Antonio Ocello.

## Contents

```text
impulse_control/          models, algorithms, persistence, and plotting
training_notebook/       one GPU training notebook per canonical checkpoint
runs/                    eight pretrained checkpoints stored with Git LFS
figures/                 the 16 panels used in the manuscript
reproducibility/         checksums, configurations, provenance, and figure map
tests/                   deterministic, checkpoint, and plotting tests
tools/                   verification, evaluation, and figure commands
environment.yml          pinned reference environment
```

Every neural network receives the full state in dimension `d`. At each decision
date, the intervention step samples full `d`-dimensional candidate vectors,
selects the best vector, evaluates its `2^d-1` nonempty coordinate masks, and
retains the best masked action. The multidimensional problem is not replaced by
a sum of trained one-dimensional problems.

## Obtain the artifact

The checkpoints use Git LFS and total approximately 84 MiB.

```bash
git lfs install
git clone --branch sisc-article-v1.0.5 --depth 1 \
  https://github.com/AurelienGrenard/Impulse-control.git
cd Impulse-control
git lfs pull
sha256sum --check reproducibility/checkpoints.sha256
```

## Environment

```bash
conda env create -f environment.yml
conda activate impulse-control-sisc
python -m pip install --no-deps -e .
```

The reference environment pins Python 3.11.9, PyTorch 2.3.1, CUDA 12.1,
NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.8.4. Checkpoint validation and
plots that do not simulate learned policies run on CPU. Full training and the
controlled-path panels require a CUDA-capable GPU.

## Reproduce the results

Verify the distributed artifact and run the tests:

```bash
python tools/verify_artifacts.py
python -m pytest -q
```

Regenerate all 16 manuscript panels:

```bash
MPLBACKEND=Agg python tools/reproduce_figures.py --device cuda:0
```

The exact panel-to-checkpoint correspondence is recorded in
[`reproducibility/figure-map.csv`](reproducibility/figure-map.csv). To recompute
the common-path learned-versus-band comparisons before plotting, run:

```bash
python tools/evaluate_policy_comparison.py --device cuda:0
```

| Figure | Numerical diagnostics |
|---|---|
| Fig. 1 | Value functions for both problems in `d=1` and `d=6` |
| Fig. 2 | Finite-budget values and paths for both problems in `d=1` |
| Fig. 3 | Finite-budget dividend paths in `d=4` |
| Fig. 4 | Learned and grid-restricted band policies in `d=1` and `d=6` |

## Published configuration

The componentwise model parameters are:

- dividends: `mu=0.5`, `sigma=0.3`, `rho=0.2`, `lambda=0.2`, `c=0.05`;
- harvesting: `mu=0.25`, `sigma=0.25`, `rho=0.2`, `alpha=1`, `x0=1`,
  `lambda=0.7`, `c=0.05`.

All continuation networks have three hidden layers of width 128, LeakyReLU
activation with negative slope `0.01`, and one scalar output. Initial fits use
20,000 Adam steps with learning rate `1e-3` and batch size 8,192. Transfer fits
use 500 Adam steps with learning rate `5e-4`. All experiments use one rollout
per design state and seed 2345.

The dimension-dependent allocations are:

| Experiment | Design states `N_k` | Randomized candidates |
|---|---:|---:|
| `d=1` | 100,000 | 12,500 |
| finite budget, `d=4` | 200,000 | 50,000 |
| no finite budget, `d=6` | 350,000 | 75,000 |

Training uses the annual decision grid `t_k=k`, `k=0,...,10`, a training Euler
step of `1e-2`, and one backward recursion with `T=K=10`. For the
time-homogeneous unlimited experiments, the plots at `T in {2,4,6,8,10}` retain
the last `T` continuation networks and shift the time origin to zero. The
finite-budget experiments use budgets `1,...,4` at `T=10`.

Policy comparisons use 1,000 common Brownian paths in batches of 32, base seed
20260924, and Euler step `2e-3`. The stationary band rule is truncated at the
reported maturity and applied only at the same annual dates as the learned
policy. Aggregate statistics are archived in
[`reproducibility/policy-evaluation.csv`](reproducibility/policy-evaluation.csv).

The complete machine-readable protocol is in
[`reproducibility/training-configurations.json`](reproducibility/training-configurations.json),
and checkpoint provenance is in
[`reproducibility/checkpoint-sources.json`](reproducibility/checkpoint-sources.json).

## Retrain

The eight notebooks under [`training_notebook/`](training_notebook/) each write
one checkpoint to `retrained_runs/`, stream progress and an ETA, and record a
SHA-256 manifest. They default to `cuda:0` and can resume an interrupted run.

The same entry points can be called directly. For example:

```bash
python -m impulse_control.train_dividend \
  --mode limited --dimension 4 --seed 2345 \
  --output retrained_runs/dividend_limited_d4.pt --progress

python -m impulse_control.train_harvesting \
  --mode unlimited --dimension 6 --seed 2345 \
  --output retrained_runs/harvesting_unlimited_d6.pt --progress
```

Without `--horizons`, unlimited training performs the published `T=10`
recursion and constructs the five reported maturities from its continuation
networks. Pass `--smoke-test --device cpu` to exercise the complete pipeline
with tiny validation sizes. Fixed seeds reproduce the protocol, although GPU
weights need not be byte-identical across architectures and software stacks.

## Integrity and supplementary archive

```bash
python -m compileall -q impulse_control tools
python tools/verify_artifacts.py
python tools/create_sisc_archive.py
```

The archive command rejects Git LFS pointers and creates a materialized ZIP
with an internal SHA-256 manifest. The generated file under `dist/` is ignored
by Git and can be uploaded as supplementary material.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication
metadata and the DOI can be added after acceptance.
