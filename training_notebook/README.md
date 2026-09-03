# Training notebooks

This directory contains one notebook for each published checkpoint:

- dividend and harvesting;
- limited problems in dimensions 1 and 4;
- unlimited problems in dimensions 1 and 6.

## Environment

Create the environment from the repository root and start Jupyter from the
same checkout:

```bash
conda env create -f environment.yml
conda activate impulse-control-sisc
python -m pip install --no-deps -e .
jupyter notebook
```

Each notebook defaults to `cuda:0`. Change the `DEVICE` cell when another GPU
must be selected.

## Outputs

Each notebook writes one canonical checkpoint to `retrained_runs/`, streams
training progress and an ETA, and records logs and SHA-256 manifests. Existing
completed work is reused.

The unlimited notebooks use `T in {5,10,20,40}`, 120,000 rollouts per date,
6,000 randomized candidates, and LeakyReLU with negative slope 0.01. Each
maturity is trained independently with its recorded seed, saved under
`retrained_runs/components/`, and then assembled into the canonical `.pt`
file. Its learned policy is evaluated on 1,000 paths in seeded batches of 32.
Limited notebooks use budgets `1,...,4` for both problems.

The exact source schedules are documented in
`reproducibility/d1-checkpoint-sources.json` and
`reproducibility/d6-checkpoint-sources.json`. Temporary training outputs are
ignored by Git; only the curated checkpoints under `runs/` are published.
