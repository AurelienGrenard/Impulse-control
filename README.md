# Neural Regression and Randomized Optimization for Impulse Control

This repository contains the code, pretrained checkpoints, and post-processing
scripts required to reproduce the numerical experiments in *Neural Regression
and Randomized Optimization for Impulse Control*.

Authors: Lokman Abbas Turki, Aurélien Grenard, Idris Kharroubi, Qinghua Li, and
Antonio Ocello.

## Artifact contents

```text
impulse_control/          models, training algorithms, persistence, and plotting
training_notebook/       one GPU training notebook per published checkpoint
runs/                    eight pretrained LeakyReLU checkpoints stored with Git LFS
figures/                 the 16 PNG panels used in the manuscript
reproducibility/         checksums, configurations, provenance, and figure map
tests/                   deterministic, checkpoint, and plotting tests
tools/                   artifact verification and figure reproduction commands
environment.yml          pinned reference environment
```

Every network receives the full state in dimension `d`. The randomized search
samples vector-valued impulses in dimension `d` and retains the best candidate.
It then zeros coordinates whose relative state displacement is below `0.4` and
keeps the better of the original and sparsified candidates. The
multidimensional problem is never replaced by a sum of trained one-dimensional
solutions.

## Obtain the artifact

The checkpoints use Git LFS and total approximately 236 MiB.

```bash
git lfs install
git clone --branch sisc-article-v1.0.1 --depth 1 \
  https://github.com/AurelienGrenard/Impulse-control.git
cd Impulse-control
git lfs pull
sha256sum --check reproducibility/checkpoints.sha256
```

The supplementary ZIP submitted with the paper must contain materialized
checkpoint bytes, not Git LFS pointer files.

## Reference environment

```bash
conda env create -f environment.yml
conda activate impulse-control-sisc
python -m pip install --no-deps -e .
```

The pinned environment uses Python 3.11.9, PyTorch 2.3.1, CUDA 12.1,
NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.8.4. Checkpoint loading,
validation, and figure generation run on CPU. Full training requires a
CUDA-capable GPU.

## Reproduce the paper figures

```bash
python tools/verify_artifacts.py
python -m pytest -q
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES='' python tools/reproduce_figures.py
```

The last command regenerates the 16 manuscript panels from `runs/` without
retraining. The exact correspondence is recorded in
[`reproducibility/figure-map.csv`](reproducibility/figure-map.csv).

| Figure | Numerical diagnostics |
|---|---|
| Fig. 1 | Unlimited dividend and harvesting problems, `d=1` |
| Fig. 2 | Finite-budget values and controlled paths for both problems, `d=1` |
| Fig. 3 | Dividend finite-budget paths, `d=4` |
| Fig. 4 | Unlimited dividend and harvesting problems, `d=6` |

## Published numerical configuration

The model parameters are:

- dividend: `mu=1`, `sigma=0.5`, `rho=0.05`, `lambda=0.2`, `c=0.5`;
- harvesting: `mu=0.25`, `sigma=0.25`, `rho=0.05`, `alpha=1`, `x0=1`,
  `lambda=0.7`, `c=0.7`.

All continuation networks have three hidden layers of width 128, LeakyReLU
activation with negative slope `0.01`, and one scalar output. Initial fits use
20,000 Adam steps with learning rate `1e-3` and batch size 8,192. Randomized
impulse candidates are processed in batches of 512. The relative displacement
threshold used to construct the single sparsified candidate is `0.4`.

For limited experiments:

- `(N_k,M_k)=(100000,1)` in `d=1` and `(12500,8)` in `d=4`;
- 5,000 randomized impulse candidates;
- 100 Adam steps per date under transfer learning, with learning rate `1e-3`;
- seed 1234.

For unlimited experiments in both `d=1` and `d=6`:

- `(N_k,M_k)=(120000,1)` in `d=1` and `(15000,8)` in `d=6`, hence
  120,000 rollouts per date;
- 6,000 randomized impulse candidates;
- 500 Adam steps per date under transfer learning, with learning rate `5e-4`;
- horizons `T in {5,10,20,40}`.

Limited experiments use `T=25` and budgets `1,...,4` for both problems. In
every experiment, `K=T`, the training Euler step is `1e-2`, and the evaluation
Euler step is `2e-3`.

The selected unlimited bundles combine independently trained maturities. Their
exact seeds and component checksums are recorded in
[`reproducibility/d1-checkpoint-sources.json`](reproducibility/d1-checkpoint-sources.json)
and
[`reproducibility/d6-checkpoint-sources.json`](reproducibility/d6-checkpoint-sources.json).
All other settings are serialized in each checkpoint and summarized in
[`reproducibility/training-configurations.json`](reproducibility/training-configurations.json).

Unlimited-policy scores use 1,000 paths in batches of 32, with base seed
`20260830` incremented once per batch. Their reported standard deviations use
the sample convention (`ddof=1`). Band-policy benchmarks use seed `1234` and
the same sample size. Figure error bars are 95% Monte Carlo confidence
intervals.

## Retrain the experiments

The eight notebooks under [`training_notebook/`](training_notebook/) each
produce one canonical `.pt` file in `retrained_runs/`, display progress and an
ETA, and can resume completed components. The unlimited notebooks train the
four selected maturity/seed pairs independently and then assemble the final
checkpoint.

The same entry points can be called directly. For example:

```bash
python -m impulse_control.train_dividend \
  --mode limited --dimension 1 --seed 1234 \
  --output reproduced_runs/dividend_limited_d1.pt --progress

python -m impulse_control.train_dividend \
  --mode unlimited --dimension 6 --horizons 5 --seed 3456 \
  --output reproduced_runs/components/dividend_d6_seed3456_T005.pt --progress
```

Omitting `--horizons` uses the published maturity schedule for the requested
dimension. Pass `--smoke-test --device cpu` to exercise the complete pipeline
with tiny validation sizes. Smoke-test outputs are not paper results.

Fixed seeds make the experiment protocol reproducible, but independently
retrained GPU weights need not be byte-identical across GPU architectures and
software stacks. The supplied checkpoints are the immutable artifacts used to
produce the committed figures.

## Validation and integrity

```bash
python -m compileall -q impulse_control tools
python tools/verify_artifacts.py
python -m pytest -q
```

`reproducibility/checkpoints.sha256` identifies the exact pretrained files.
Create a materialized supplementary snapshot with:

```bash
python tools/create_sisc_archive.py
```

The command rejects Git LFS pointers and writes an internal SHA-256 manifest.
The generated ZIP under `dist/` is ignored by Git and is intended for direct
upload to the journal or an archival repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication
metadata and the DOI should be added after acceptance.
