# Reproducibility records

`figure-map.csv` maps every manuscript panel to its checkpoint and generated
PNG file. `checkpoints.sha256` identifies the exact pretrained
artifacts used for the committed figures.

The checksums apply to materialized Git LFS files. Run `git lfs pull` before
checking them. A SISC supplementary archive must likewise include the complete
checkpoint contents, not pointer files.

Each checkpoint serializes its numerical configuration. The common seed and
the few horizon-specific `d=6` settings are also listed in
`training-configurations.json`. Newly trained checkpoints receive a
same-basename JSON manifest containing the explicit seed, device, software
versions, and horizon schedule.
