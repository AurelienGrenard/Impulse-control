"""Small helpers shared by the eight one-checkpoint training notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "retrained_runs"
DEFAULT_DEVICE = "cuda:0"
SEED = 1234
ACTIVATION = "leaky_relu"
NEGATIVE_SLOPE = 0.01


@dataclass(frozen=True)
class TrainingTask:
    """Describe one published checkpoint."""

    application: str
    mode: str
    dimension: int

    def __post_init__(self) -> None:
        if self.application not in {"dividend", "harvesting"}:
            raise ValueError(f"Unsupported application: {self.application}")
        if self.mode not in {"limited", "unlimited"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        valid_dimensions = {1, 4} if self.mode == "limited" else {1, 6}
        if self.dimension not in valid_dimensions:
            raise ValueError(
                f"Dimension {self.dimension} is not published for mode {self.mode}."
            )

    @property
    def stem(self) -> str:
        """Return the canonical checkpoint name without its extension."""
        return f"{self.application}_{self.mode}_d{self.dimension}"

    @property
    def output(self) -> Path:
        """Return the destination used for collaborative retraining."""
        return OUTPUT_ROOT / f"{self.stem}.pt"

    @property
    def module(self) -> str:
        """Return the public training module for this application."""
        return f"impulse_control.train_{self.application}"


def _utc_now() -> str:
    """Return a compact UTC timestamp for the execution manifest."""
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    """Compute the digest used to identify a trained checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Persist a human-readable execution manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _command(task: TrainingTask, device: str) -> list[str]:
    """Build the exact reproducible command executed by a notebook."""
    return [
        sys.executable,
        "-u",
        "-m",
        task.module,
        "--mode",
        task.mode,
        "--dimension",
        str(task.dimension),
        "--seed",
        str(SEED),
        "--activation",
        ACTIVATION,
        "--negative-slope",
        str(NEGATIVE_SLOPE),
        "--device",
        device,
        "--output",
        str(task.output),
        "--progress",
    ]


def preflight(task: TrainingTask, device: str = DEFAULT_DEVICE) -> None:
    """Display the fixed setup and fail early when CUDA is unavailable."""
    if not (REPOSITORY_ROOT / "impulse_control").is_dir():
        raise RuntimeError(f"Invalid repository root: {REPOSITORY_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Jupyter kernel.")
    if not device.startswith("cuda:"):
        raise ValueError("Training notebooks require a CUDA device such as cuda:0.")
    device_index = int(device.split(":", maxsplit=1)[1])
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{device} is unavailable; PyTorch exposes {torch.cuda.device_count()} GPU(s)."
        )

    state = "resume/check" if task.output.is_file() else "new"
    print(f"Repository: {REPOSITORY_ROOT}")
    print(f"Task: {task.stem} [{state}]")
    print(f"Output: {task.output}")
    print(f"Device: {device} — {torch.cuda.get_device_name(device_index)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"Seed: {SEED}")
    print(f"Network: 3 x 128, {ACTIVATION}, negative slope {NEGATIVE_SLOPE}")
    print("Randomized impulse candidates: 5000")


def run_training(task: TrainingTask, device: str = DEFAULT_DEVICE) -> Path:
    """Run one training command while streaming output to the notebook and a log."""
    preflight(task, device)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "logs" / f"{task.stem}.log"
    manifest_path = OUTPUT_ROOT / "manifests" / f"{task.stem}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = _command(task, device)
    manifest: dict[str, object] = {
        "task": task.stem,
        "status": "running",
        "started_utc": _utc_now(),
        "command": command,
        "output": str(task.output),
        "log": str(log_path),
        "seed": SEED,
        "device": device,
        "activation": ACTIVATION,
        "negative_slope": NEGATIVE_SLOPE,
        "randomized_candidates": 5_000,
    }
    _write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(SEED)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["MPLCONFIGDIR"] = str(OUTPUT_ROOT / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print("\nCommand:", " ".join(command))
    print(f"Live log: {log_path}\n", flush=True)

    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()

    elapsed = time.perf_counter() - started
    manifest["finished_utc"] = _utc_now()
    manifest["elapsed_seconds"] = elapsed
    manifest["return_code"] = return_code

    if return_code != 0:
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"Training failed for {task.stem}. Last log lines:\n" + "\n".join(tail)
        )
    if not task.output.is_file():
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise FileNotFoundError(f"Training ended without creating {task.output}.")

    digest = _sha256(task.output)
    manifest["status"] = "complete"
    manifest["checkpoint_bytes"] = task.output.stat().st_size
    manifest["checkpoint_sha256"] = digest
    _write_json(manifest_path, manifest)

    print(f"\nCompleted in {elapsed / 3600:.2f} h")
    print(f"Checkpoint: {task.output}")
    print(f"SHA-256: {digest}")
    print(f"Execution manifest: {manifest_path}")
    return task.output
