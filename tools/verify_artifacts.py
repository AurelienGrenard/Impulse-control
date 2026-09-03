"""Verify the integrity and completeness of the SISC reproducibility artifact."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impulse_control.saving import load_all_results_unlimited, load_results_bundle
from impulse_control.reproducibility import (
    PUBLISHED_EVALUATION_PATHS,
    PUBLISHED_EVALUATION_SEED,
    PUBLISHED_MIN_REL_IMPULSE,
    published_horizons,
)


def sha256(path: Path) -> str:
    """Compute a file checksum without loading the complete checkpoint in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Check hashes, figure mappings, and checkpoint readability."""
    checksum_file = ROOT / "reproducibility" / "checkpoints.sha256"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative}")
        print(f"checksum ok: {relative}")

    with (ROOT / "reproducibility" / "figure-map.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    outputs = {row["output"] for row in rows}
    if len(outputs) != 16:
        raise RuntimeError(f"Expected 16 manuscript panels, found {len(outputs)}")
    for row in rows:
        for key in ("checkpoint", "output"):
            if not (ROOT / row[key]).is_file():
                raise FileNotFoundError(row[key])
    print(f"figure map ok: {len(outputs)} manuscript panels")

    for application in ("dividend", "harvesting"):
        for dimension in (1, 4):
            path = ROOT / "runs" / f"{application}_limited_d{dimension}.pt"
            bundle = load_results_bundle(str(path), map_location="cpu")
            budgets = [
                int(result["max_impulses"])
                for result in bundle["bounded_results"]
            ]
            if budgets != [1, 2, 3, 4] or bundle["unconstrained"] is None:
                raise RuntimeError(f"Incomplete checkpoint: {path.name}")
            configs = [result["cfg"] for result in bundle["bounded_results"]]
            configs.append(bundle["unconstrained"]["config"])
            if any(cfg.net.activation != "leaky_relu" for cfg in configs):
                raise RuntimeError(f"Unexpected activation: {path.name}")
            if any(float(cfg.opt.min_rel_impulse) != PUBLISHED_MIN_REL_IMPULSE for cfg in configs):
                raise RuntimeError(f"Unexpected sparsification threshold: {path.name}")
            print(f"checkpoint loads: {path.name}")
        for dimension in (1, 6):
            path = ROOT / "runs" / f"{application}_unlimited_d{dimension}.pt"
            results = load_all_results_unlimited(str(path), map_location="cpu")
            expected = list(published_horizons("unlimited", dimension))
            if [float(result["T"]) for result in results] != expected:
                raise RuntimeError(f"Unexpected horizons: {path.name}")
            if any(result["cfg"].net.activation != "leaky_relu" for result in results):
                raise RuntimeError(f"Unexpected activation: {path.name}")
            for result in results:
                cfg = result["cfg"]
                if int(cfg.design.N_k) * int(cfg.mc.M_k) != 120_000:
                    raise RuntimeError(f"Unexpected rollout budget: {path.name}")
                if int(cfg.opt.n_global) != 6_000:
                    raise RuntimeError(f"Unexpected candidate count: {path.name}")
                if int(cfg.opt.n_global_batch) != 512:
                    raise RuntimeError(f"Unexpected candidate batch size: {path.name}")
                if float(cfg.opt.min_rel_impulse) != PUBLISHED_MIN_REL_IMPULSE:
                    raise RuntimeError(f"Unexpected sparsification threshold: {path.name}")
                if result.get("mc_n_NN") != PUBLISHED_EVALUATION_PATHS:
                    raise RuntimeError(f"Unexpected evaluation path count: {path.name}")
                if result.get("mc_seed_NN") != PUBLISHED_EVALUATION_SEED:
                    raise RuntimeError(f"Unexpected evaluation seed: {path.name}")
            print(f"checkpoint loads: {path.name}")

    print(f"artifact ok: 8 checkpoints and {len(outputs)} manuscript panels")


if __name__ == "__main__":
    main()
