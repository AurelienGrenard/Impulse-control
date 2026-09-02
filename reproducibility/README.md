# Reproducibility records

`figure-map.csv` maps every manuscript panel to its checkpoint and generated
PNG file. `checkpoints.sha256` identifies the exact pretrained
artifacts used for the committed figures.

`training-configurations.json` records the complete published schedules.
`d6-checkpoint-sources.json` records the seed and SHA-256 digest of every
independently trained maturity assembled into the two unlimited `d=6` bundles.

The checksums apply to materialized Git LFS files. Run `git lfs pull` before
checking them. A SISC supplementary archive must likewise include the complete
checkpoint contents, not pointer files.

Each checkpoint serializes its numerical configuration. Newly trained checkpoints receive a
same-basename JSON manifest containing the explicit seed, device, software
versions, and horizon schedule.
