# Reproducibility records

- `checkpoints.sha256` identifies the eight materialized Git LFS checkpoints.
- `checkpoint-sources.json` records the training source and construction of
  each canonical checkpoint.
- `training-configurations.json` contains the complete numerical protocol.
- `policy-evaluation.csv` stores the common-path policy comparisons.
- `figure-map.csv` maps each manuscript panel to its input artifacts.

The unlimited checkpoints contain the five reported maturities
`T in {2,4,6,8,10}`. They share one source recursion at `T=10`; a shorter
maturity retains the corresponding final continuation networks and shifts the
time origin to zero.

The band benchmark uses stationary thresholds, is truncated at each finite
maturity, and is exercised only on the annual grid available to the learned
policy. Recompute its paired statistics with
`python tools/evaluate_policy_comparison.py --device cuda:0`.

Run `git lfs pull` before checking hashes or building the supplementary ZIP.
Newly trained checkpoints receive a same-basename JSON manifest containing the
seed, command settings, software versions, and reporting schedule.
