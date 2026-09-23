"""Verify the integrity and completeness of the manuscript companion artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impulse_control.reproducibility import (
    PUBLISHED_EVALUATION_PATHS,
    PUBLISHED_EVALUATION_SEED,
    PUBLISHED_REPORTING_HORIZONS,
    PUBLISHED_SOURCE_HORIZON,
    published_training_parameters,
)
from impulse_control.saving import load_all_results_unlimited, load_results_bundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(value)


def _validate_config(cfg, application: str, dimension: int, horizon: float) -> None:
    if not math.isclose(float(cfg.time.T), horizon) or int(cfg.time.K) != int(horizon):
        raise RuntimeError("Unexpected time grid.")
    if cfg.net.activation != "leaky_relu" or not math.isclose(
        float(cfg.net.negative_slope), 0.01
    ):
        raise RuntimeError("Unexpected network activation.")
    if int(cfg.net.transfer_steps) != 500 or not math.isclose(
        float(cfg.net.transfer_lr), 5e-4
    ):
        raise RuntimeError("Unexpected transfer-learning schedule.")
    profile = published_training_parameters(application, "unlimited", dimension)
    if (
        int(cfg.design.N_k) != profile["design_states"]
        or int(cfg.mc.M_k) != profile["rollouts_per_state"]
        or int(cfg.opt.n_global) != profile["randomized_candidates"]
        or int(cfg.opt.n_global_batch) != profile["candidate_batch_size"]
        or cfg.opt.min_rel_impulse is not None
    ):
        raise RuntimeError("Unexpected randomized-search configuration.")

    params = getattr(cfg, application)
    expected = {
        "dividend": {"mu": 0.5, "sigma": 0.3, "rho": 0.2, "lam": 0.2, "c": 0.05},
        "harvesting": {
            "mu": 0.25,
            "sigma": 0.25,
            "rho": 0.2,
            "alpha": 1.0,
            "x0": 1.0,
            "lam": 0.7,
            "c": 0.05,
        },
    }[application]
    if int(params.state_dim) != dimension:
        raise RuntimeError("Unexpected state dimension.")
    for name, target in expected.items():
        if not math.isclose(_scalar(getattr(params, name)), target, abs_tol=1e-6):
            raise RuntimeError(f"Unexpected {application} parameter: {name}.")


def main() -> None:
    checksum_file = ROOT / "reproducibility" / "checkpoints.sha256"
    checksum_rows = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative}")
        checksum_rows.append((path.name, actual))
        print(f"checksum ok: {relative}")
    if len(checksum_rows) != 8:
        raise RuntimeError("Expected eight checkpoint checksums.")

    provenance = json.loads(
        (ROOT / "reproducibility" / "checkpoint-sources.json").read_text()
    )
    for name, digest in checksum_rows:
        if provenance["checkpoints"][name]["published_sha256"] != digest:
            raise RuntimeError(f"Provenance mismatch for {name}.")

    with (ROOT / "reproducibility" / "figure-map.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        figure_rows = list(csv.DictReader(stream))
    outputs = {row["output"] for row in figure_rows}
    if len(outputs) != 16:
        raise RuntimeError(f"Expected 16 manuscript panels, found {len(outputs)}")
    for row in figure_rows:
        for key in ("checkpoint", "output"):
            if not (ROOT / row[key]).is_file():
                raise FileNotFoundError(row[key])
        if row.get("statistics") and not (ROOT / row["statistics"]).is_file():
            raise FileNotFoundError(row["statistics"])
    print(f"figure map ok: {len(outputs)} manuscript panels")

    with (ROOT / "reproducibility" / "policy-evaluation.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        policy_rows = list(csv.DictReader(stream))
    expected_policy_rows = {
        (application, dimension, int(horizon))
        for application in ("dividend", "harvesting")
        for dimension in (1, 6)
        for horizon in PUBLISHED_REPORTING_HORIZONS
    }
    actual_policy_rows = {
        (row["problem"], int(row["d"]), int(row["T"])) for row in policy_rows
    }
    if actual_policy_rows != expected_policy_rows:
        raise RuntimeError("Incomplete policy-evaluation table.")
    if any(int(row["n_paths"]) != PUBLISHED_EVALUATION_PATHS for row in policy_rows):
        raise RuntimeError("Unexpected policy-evaluation path count.")
    print(f"policy evaluation ok: {len(policy_rows)} common-path comparisons")

    for application in ("dividend", "harvesting"):
        for dimension in (1, 4):
            path = ROOT / "runs" / f"{application}_limited_d{dimension}.pt"
            bundle = load_results_bundle(str(path), map_location="cpu")
            budgets = [int(item["max_impulses"]) for item in bundle["bounded_results"]]
            if budgets != [1, 2, 3, 4] or bundle["unconstrained"] is None:
                raise RuntimeError(f"Incomplete checkpoint: {path.name}")
            configs = [bundle["unconstrained"]["config"]] + [
                item["cfg"] for item in bundle["bounded_results"]
            ]
            for cfg in configs:
                _validate_config(
                    cfg, application, dimension, PUBLISHED_SOURCE_HORIZON
                )
            print(f"checkpoint loads: {path.name}")

        for dimension in (1, 6):
            path = ROOT / "runs" / f"{application}_unlimited_d{dimension}.pt"
            results = load_all_results_unlimited(str(path), map_location="cpu")
            if [float(item["T"]) for item in results] != list(PUBLISHED_REPORTING_HORIZONS):
                raise RuntimeError(f"Unexpected horizons: {path.name}")
            for result in results:
                _validate_config(result["cfg"], application, dimension, float(result["T"]))
                if len(result["qhats"]) != int(result["T"]) + 1:
                    raise RuntimeError(f"Unexpected continuation slice: {path.name}")
                if result.get("mc_n_NN") != PUBLISHED_EVALUATION_PATHS:
                    raise RuntimeError(f"Unexpected evaluation count: {path.name}")
                if result.get("mc_seed_NN") != PUBLISHED_EVALUATION_SEED:
                    raise RuntimeError(f"Unexpected evaluation seed: {path.name}")
            print(f"checkpoint loads: {path.name}")

    print(f"artifact ok: 8 checkpoints and {len(outputs)} manuscript panels")


if __name__ == "__main__":
    main()
