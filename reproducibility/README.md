# Reproducibility records

`figure-map.csv` maps every manuscript panel to its checkpoint, notebook, and
generated PNG file. `checkpoints.sha256` identifies the exact pretrained
artifacts used for the committed figures.

The checksums apply to materialized Git LFS files. Run `git lfs pull` before
checking them. A SISC supplementary archive must likewise include the complete
checkpoint contents, not pointer files.

The existing checkpoints predate the seeded command-line manifests. They remain
the authoritative artifacts for the committed paper figures. Newly trained
checkpoints receive a same-basename JSON manifest containing the explicit seed,
device, and software versions.
