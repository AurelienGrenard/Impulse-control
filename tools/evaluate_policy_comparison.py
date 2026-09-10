"""Recompute the common-path policy comparisons used in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impulse_control.policy_evaluation import evaluate_all


def main() -> None:
    """Evaluate all policy pairs and write their aggregate statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dt-fine", type=float, default=2e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility" / "policy-evaluation.csv",
    )
    args = parser.parse_args()

    results = evaluate_all(ROOT, device=args.device, dt_fine=args.dt_fine)
    columns = [
        "problem",
        "d",
        "T",
        "learned",
        "learned_ci95",
        "annual_band",
        "annual_band_ci95",
        "learned_advantage",
        "advantage_ci95",
        "n_paths",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results[columns].to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
