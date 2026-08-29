# Training notebooks

Each notebook trains one published checkpoint on `cuda:0`. The notebooks use
the same public training commands documented in the project README; they only
add live logs, progress information, resumable output, and SHA-256 reporting.
Before training, they display the horizon-specific values of $N_k$, $M_k$,
the randomized-search size, and the transfer-learning schedule.

## Environment

From the repository root, create the reference environment and start Jupyter:

```bash
conda env create -f environment.yml
conda activate impulse-control-sisc
python -m pip install --no-deps -e .
jupyter lab training_notebook/
```

An existing compatible environment can instead install the checkout with the
same `pip` command. Git is not required while a notebook is running.

## Outputs

Every notebook writes one checkpoint to `retrained_runs/`, with the canonical
name shown in its title. A JSON run manifest is written beside the checkpoint,
and the notebook execution log is stored under `retrained_runs/logs/`.

The figures used by the manuscript are copied to `training_notebook/figures/`
for convenient replacement of the corresponding Overleaf directory.

If execution is interrupted, run the same cell again. Completed maturities or
impulse budgets stored in the checkpoint are validated and skipped. Do not run
two notebooks targeting the same checkpoint simultaneously.

To use a different GPU on a multi-GPU machine, change the `DEVICE` value in the
configuration cell from `cuda:0` to the required local index.
