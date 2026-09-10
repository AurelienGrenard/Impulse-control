# Reproducibility records

`figure-map.csv` maps every manuscript panel to its checkpoint and generated
PNG file. `checkpoints.sha256` identifies the exact pretrained
artifacts used for the committed figures.

`training-configurations.json` records the complete published schedules.
`d1-checkpoint-sources.json` and `d6-checkpoint-sources.json` record the seed
and SHA-256 digest of every independently trained maturity assembled into the
unlimited bundles. They also record the batched, seeded policy-evaluation
protocol used to compute the statistics stored in those bundles.
`policy-evaluation.csv` contains the common-path comparison used in the policy
panels. The band rule uses its stationary thresholds, is truncated at the
finite maturity, and is applied only at the annual decision dates available to
the learned policy. Recompute it with
`python tools/evaluate_policy_comparison.py --device cuda`.

The checksums apply to materialized Git LFS files. Run `git lfs pull` before
checking them. A SISC supplementary archive must likewise include the complete
checkpoint contents, not pointer files.

Each checkpoint serializes its numerical configuration. Newly trained checkpoints receive a
same-basename JSON manifest containing the explicit seed, device, software
versions, and horizon schedule.
