# Neural Regression and Randomized Optimization for Impulse Control

This repository contains the code, pretrained checkpoints, and post-processing
notebooks required to reproduce the numerical results in the paper *Neural
Regression and Randomized Optimization for Impulse Control*.

Authors: Lokman Abbas Turki, Aurélien Grenard, Idris Kharroubi, Qinghua Li, and
Antonio Ocello.

The numerical methods are implemented for the dividend problem in Section 6 and
the harvesting problem in Appendix C. The repository is dedicated to the
reproducibility material for this paper.

## Artifact contents

```text
impulse_control/
  common.py              shared optimization and time-grid configuration
  exact_dividend.py      closed-form dividend benchmark and band policy
  exact_harvesting.py    closed-form harvesting benchmark and band policy
  dividend.py            dividend model and numerical algorithms
  harvesting.py          harvesting model and numerical algorithms
  saving.py              portable checkpoint serialization
  plotting.py            grayscale paper figures
  reproducibility.py     seeds and run manifests
  train_dividend.py      dividend training command
  train_harvesting.py    harvesting training command
notebooks/               executed presentation notebooks
training_notebook/       eight independent GPU training notebooks
runs/                    eight pretrained checkpoints stored with Git LFS
figures/                 22 generated PNG files used in the paper panels
reproducibility/         checksums and figure-to-artifact correspondence
tests/                   checkpoint and plotting smoke tests
tools/                   figure reproduction and artifact verification commands
environment.yml          pinned reference software environment
```

The neural networks take the full state in dimension `d` as input. The
randomized impulse search is also performed in dimension `d`; the implementation
does not replace a multidimensional problem by a sum of one-dimensional neural
solutions.

## Training notebooks

The [`training_notebook/`](training_notebook/) directory contains one notebook
for each published checkpoint. Every notebook defaults to `cuda:0`, streams
progress and ETA information, and writes resumable results to
`retrained_runs/`. The device can be changed in one configuration cell when a
different local GPU is required.

## Obtain the artifact

Git LFS is required because the pretrained checkpoints total approximately
316 MiB.

```bash
git lfs install
git clone https://github.com/AurelienGrenard/Impulse-control.git
cd Impulse-control
git lfs pull
sha256sum --check reproducibility/checkpoints.sha256
```

The SISC supplementary ZIP must contain the materialized `.pt` files rather
than Git LFS pointer files.

## Reference environment

Create the pinned environment and install this package from the checkout:

```bash
conda env create -f environment.yml
conda activate impulse-control-sisc
python -m pip install --no-deps -e .
```

The reference environment uses Python 3.11.9, PyTorch 2.3.1, and CUDA 12.1.
Loading checkpoints, running tests, and regenerating figures work on CPU.
Complete training is intended for a CUDA-capable GPU and is substantially more
expensive. Floating-point roundoff can vary across GPU models and software
stacks even when the random seed is fixed.

## Reproduce every paper figure

First verify the supplied artifact:

```bash
python tools/verify_artifacts.py
pytest -q
```

Then regenerate all 22 PNG files from the supplied checkpoints:

```bash
MPLBACKEND=Agg python tools/reproduce_figures.py
```

This command overwrites the files in `figures/` without retraining. The eight
executed notebooks display the same results and may be opened directly:

```bash
jupyter lab notebooks/
```

The complete correspondence between article panels, checkpoints, notebooks,
and PNG files is recorded in
[`reproducibility/figure-map.csv`](reproducibility/figure-map.csv). There are no
numerical tables in the manuscript.

| Article figure | Experiment | Artifact |
|---|---|---|
| Fig. 1 | Dividend, unlimited, `d=1` | `dividend_unlimited_d1` |
| Fig. 2 | Dividend, limited, `d=1` | `dividend_limited_d1` |
| Fig. 3 | Dividend, limited paths, `d=4` | `dividend_limited_d4` |
| Fig. 4 | Dividend, unlimited, `d=6` | `dividend_unlimited_d6` |
| Fig. 5 | Harvesting, unlimited, `d=1` | `harvesting_unlimited_d1` |
| Fig. 6 | Harvesting, limited, `d=1` | `harvesting_limited_d1` |
| Fig. 7 | Harvesting, limited paths, `d=4` | `harvesting_limited_d4` |
| Fig. 8 | Harvesting, unlimited, `d=6` | `harvesting_unlimited_d6` |

The left panel of Fig. 8 is generated in dimension `d=6`.

## Numerical parameters

All financial and numerical parameters are stored in the training entry points
and serialized in each checkpoint. The principal settings are:

- dividend: `mu=1`, `sigma=0.5`, `rho=0.05`, `lambda=0.2`, `c=0.5`;
- harvesting: `mu=0.25`, `sigma=0.25`, `rho=0.05`, `alpha=1`, `x0=1`,
  `lambda=0.7`, `c=0.7`;
- network: three hidden layers of width 128 with LeakyReLU activation and
  negative slope `0.01`;
- training: 20,000 iterations and batch size 8,192; dimension `d=1` uses
  100,000 design states with one rollout per state, while dimensions `d=4` and
  `d=6` use 12,500 states with eight rollouts per state;
- randomized impulse search: 5,000 candidates in every experiment;
- intervention grid: `K=T`, with `T=25` for limited experiments and
  `T in {5, 10, 25, 50, 100}` for unlimited experiments;
- limited budgets: `n in {1,...,5}` for dividends and `n in {1,...,4}` for
  harvesting.

The source entry points remain the authoritative specification, including
application-specific evaluation sample sizes and transfer-learning settings.

## Retrain the published experiments

Training commands use seed `1234` by default. The seed, selected device, and
software versions are written to a JSON manifest beside every new checkpoint.
Use a new output directory unless replacement of the supplied checkpoints is
intentional.

```bash
python -m impulse_control.train_dividend --mode unlimited --dimension 1 --seed 1234 --output reproduced_runs/dividend_unlimited_d1.pt
python -m impulse_control.train_dividend --mode limited   --dimension 1 --seed 1234 --output reproduced_runs/dividend_limited_d1.pt
python -m impulse_control.train_dividend --mode limited   --dimension 4 --seed 1234 --output reproduced_runs/dividend_limited_d4.pt
python -m impulse_control.train_dividend --mode unlimited --dimension 6 --seed 1234 --output reproduced_runs/dividend_unlimited_d6.pt

python -m impulse_control.train_harvesting --mode unlimited --dimension 1 --seed 1234 --output reproduced_runs/harvesting_unlimited_d1.pt
python -m impulse_control.train_harvesting --mode limited   --dimension 1 --seed 1234 --output reproduced_runs/harvesting_limited_d1.pt
python -m impulse_control.train_harvesting --mode limited   --dimension 4 --seed 1234 --output reproduced_runs/harvesting_limited_d4.pt
python -m impulse_control.train_harvesting --mode unlimited --dimension 6 --seed 1234 --output reproduced_runs/harvesting_unlimited_d6.pt
```

Pass `--device cpu` for CPU execution. Pass `--smoke-test` to exercise the same
pipeline with tiny training and simulation budgets; smoke-test outputs are not
paper results.

Because training is stochastic and GPU arithmetic is platform-dependent,
retrained networks are expected to reproduce the reported curves and policy
scores up to Monte Carlo variability and floating-point roundoff, not to have
byte-identical weights. The supplied checkpoints are the immutable numerical
artifacts used to draw the committed figures.

## Validation and integrity

```bash
python -m compileall -q impulse_control tools
python tools/verify_artifacts.py
pytest -q
```

`reproducibility/checkpoints.sha256` identifies the exact pretrained files.
Before submission, a tagged release and a materialized ZIP snapshot should be
deposited in the SISC Supplementary Materials or an archival repository such as
Zenodo.

Create the materialized supplementary snapshot with:

```bash
python tools/create_sisc_archive.py
```

The command rejects unresolved Git LFS pointers and writes an internal checksum
manifest. Its default output under `dist/` is ignored by Git and is intended for
direct upload to the journal submission system or Zenodo.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication
metadata and the DOI should be added after acceptance.
