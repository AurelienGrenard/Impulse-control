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
conda activate impulse-control-repro
python -m pip install --no-deps -e .
jupyter notebook
```

Each notebook defaults to `cuda:0`. Change the `DEVICE` cell when another GPU
must be selected.

## Outputs

Each notebook writes one canonical checkpoint to `retrained_runs/`, streams
training progress and an ETA, and records logs and SHA-256 manifests. Existing
completed work is reused.

All notebooks use seed 2345 and train one backward recursion with `T=K=10`.
The unlimited notebooks then report `T in {2,4,6,8,10}` by retaining the final
continuation networks of that recursion. Limited notebooks use budgets
`1,...,4`. The dimension-dependent state and candidate counts are printed by
the preflight cell before training starts.

The exact protocol is documented in
`reproducibility/training-configurations.json`, and the published artifact
provenance is recorded in `reproducibility/checkpoint-sources.json`. Temporary
training outputs are ignored by Git; only curated checkpoints under `runs/`
are published.
