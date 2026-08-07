# Impulse Control — modular implementation

This directory is a clean, notebook-light reorganization of the original
`Impulse-control-main` experiments accompanying *Neural Regression and
Randomized Optimization for Impulse Control*.

The financial definitions, Monte Carlo rollouts, randomized optimizers,
training loops, network architectures, time grids and exact solutions are
unchanged. Numerical function bodies were migrated verbatim from the published
notebooks; `tools/build_refactored_project.py` records that migration.

## Structure

```text
impulse_control/
  common.py              shared optimization and time-grid configuration
  exact_dividend.py      exact dividend solution
  exact_harvesting.py    exact harvesting solution
  dividend.py            dividend problem and numerical algorithms
  harvesting.py          harvesting problem and numerical algorithms
  saving.py              common checkpoint save/load API
  plotting.py            common white-background figure style and public plots
  train_dividend.py      dividend training CLI
  train_harvesting.py    harvesting training CLI
notebooks/
  dividend_limited_d1.ipynb
  dividend_limited_d4.ipynb
  dividend_unlimited_d1.ipynb
  dividend_unlimited_d6.ipynb
  harvesting_limited_d1.ipynb
  harvesting_limited_d4.ipynb
  harvesting_unlimited_d1.ipynb
  harvesting_unlimited_d6.ipynb
runs/                    supplied pretrained checkpoints
tests/
```

The notebooks are presentation layers only: one loading cell and one plotting
cell. They do not contain numerical or drawing implementation.

## Environment

Python 3.11 and the same dependency versions as the original experiments are
recommended:

```bash
conda create -n ImpulseControl python=3.11
conda activate ImpulseControl
conda install pytorch=2.3.1 pytorch-cuda=12.1 matplotlib numpy scipy jupyter -c pytorch -c nvidia
```

Run commands from this directory. For a source checkout without installation:

```bash
export PYTHONPATH="$PWD"
```

The pretrained checkpoints are tracked with Git LFS:

```bash
git lfs install
git lfs pull
```

## Use a supplied model

Limited experiments intentionally need only these two lines:

```python
from impulse_control.saving import load_results_bundle

bundle = load_results_bundle("runs/dividend_limited_d4.pt", map_location="cpu")
all_results_loaded = bundle["bounded_results"]
```

The unconstrained benchmark remains available inside `bundle`, but plotting
functions consume it internally:

```python
from impulse_control.plotting import plot_limited_summary, plot_limited_paths

plot_limited_summary(bundle, output="figures/dividend_values.png")
plot_limited_paths(bundle, output_prefix="figures/dividend_limited_d4")
```

Unlimited checkpoints use:

```python
from impulse_control.saving import load_all_results_unlimited

all_results_loaded = load_all_results_unlimited(
    "runs/dividend_unlimited_d6.pt",
    map_location="cpu",
)
```

## Train and save a model

The CLI keeps the notebook hyperparameters and algorithms. Select the
application, mode, dimension and output checkpoint explicitly:

```bash
python -m impulse_control.train_dividend \
  --mode limited \
  --dimension 4 \
  --output runs/dividend_limited_d4.pt

python -m impulse_control.train_harvesting \
  --mode unlimited \
  --dimension 6 \
  --output runs/harvesting_unlimited_d6.pt
```

Use another output name when preserving a supplied checkpoint. CUDA is selected
when available; pass `--device cpu` to force CPU execution.

## Validation

```bash
python -m compileall -q impulse_control
pytest -q
```

The smoke tests load every supplied checkpoint, rebuild all networks and render
the summary figures with a non-interactive backend.
