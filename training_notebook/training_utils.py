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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from impulse_control.reproducibility import (
    published_horizons,
    published_training_parameters,
)
from impulse_control.saving import load_all_results_unlimited, save_all_results_unlimited


OUTPUT_ROOT = REPOSITORY_ROOT / "retrained_runs"
DEFAULT_DEVICE = "cuda:0"
SEED = 1234
ACTIVATION = "leaky_relu"
NEGATIVE_SLOPE = 0.01
PUBLISHED_D6_SEEDS = {
    "dividend": {5.0: 3456, 10.0: 1234, 20.0: 2345, 40.0: 2345},
    "harvesting": {5.0: 1234, 10.0: 3456, 20.0: 2345, 40.0: 2345},
}


def _published_unlimited_seeds(task: "TrainingTask") -> dict[float, int]:
    """Return the component seed used for each published maturity."""
    if task.dimension == 1:
        return {
            horizon: SEED
            for horizon in published_horizons(task.mode, task.dimension)
        }
    return PUBLISHED_D6_SEEDS[task.application]


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


def _command(
    task: TrainingTask,
    device: str,
    *,
    seed: int = SEED,
    horizon: float | None = None,
    output: Path | None = None,
) -> list[str]:
    """Build the exact reproducible command executed by a notebook."""
    command = [
        sys.executable,
        "-u",
        "-m",
        task.module,
        "--mode",
        task.mode,
        "--dimension",
        str(task.dimension),
        "--seed",
        str(seed),
        "--activation",
        ACTIVATION,
        "--negative-slope",
        str(NEGATIVE_SLOPE),
        "--device",
        device,
        "--output",
        str(output or task.output),
        "--progress",
    ]
    if horizon is not None:
        command.extend(("--horizons", f"{horizon:g}"))
    return command


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
    print(f"Device: {device} - {torch.cuda.get_device_name(device_index)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"Seed: {SEED}")
    print(f"Network: 3 x 128, {ACTIVATION}, negative slope {NEGATIVE_SLOPE}")
    if task.mode == "unlimited":
        print("Published horizon schedule:")
        for horizon in published_horizons(task.mode, task.dimension):
            settings = published_training_parameters(
                task.application, task.mode, task.dimension, horizon
            )
            seed = _published_unlimited_seeds(task)[horizon]
            print(
                f"  T={horizon:g}, seed={seed}: N_k={settings['design_states']}, "
                f"M_k={settings['rollouts_per_state']}, "
                f"candidates={settings['randomized_candidates']}, "
                f"Adam steps after transfer={settings['transfer_steps']}"
            )
    else:
        settings = published_training_parameters(
            task.application, task.mode, task.dimension
        )
        print(
            f"N_k={settings['design_states']}, M_k={settings['rollouts_per_state']}, "
            f"randomized candidates={settings['randomized_candidates']}"
        )


def _run_published_unlimited(task: TrainingTask, device: str) -> Path:
    """Train the selected maturity components and assemble one checkpoint."""
    schedule = _published_unlimited_seeds(task)
    manifest_path = OUTPUT_ROOT / "manifests" / f"{task.stem}.json"
    if task.output.is_file():
        existing = load_all_results_unlimited(str(task.output), map_location="cpu")
        if [float(result["T"]) for result in existing] != list(schedule):
            raise RuntimeError(
                f"Existing {task.output} does not use the published d=6 maturities."
            )
        if any(result["cfg"].net.activation != ACTIVATION for result in existing):
            raise RuntimeError(f"Existing {task.output} does not use {ACTIVATION}.")
        print(f"Published checkpoint already complete: {task.output}")
        return task.output

    component_root = OUTPUT_ROOT / "components" / task.stem
    log_root = OUTPUT_ROOT / "logs" / task.stem
    component_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, object]] = []
    started = time.perf_counter()

    for horizon, seed in schedule.items():
        component = component_root / f"seed{seed}_T{int(horizon):03d}.pt"
        command = _command(
            task,
            device,
            seed=seed,
            horizon=horizon,
            output=component,
        )
        log_path = log_root / f"seed{seed}_T{int(horizon):03d}.log"
        if not component.is_file():
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = str(seed)
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["PYTHONIOENCODING"] = "utf-8"
            environment["PYTHONUTF8"] = "1"
            environment["MPLCONFIGDIR"] = str(OUTPUT_ROOT / ".matplotlib")
            Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            print(f"\nTraining T={horizon:g}, seed={seed}")
            print("Command:", " ".join(command), flush=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"Training failed for T={horizon:g}; inspect {log_path}."
                )
        else:
            print(f"Reusing component: {component}")

        loaded = load_all_results_unlimited(str(component), map_location="cpu")
        if len(loaded) != 1 or float(loaded[0]["T"]) != horizon:
            raise RuntimeError(f"Invalid component checkpoint: {component}")
        components.append(
            {
                "T": horizon,
                "seed": seed,
                "path": str(component),
                "sha256": _sha256(component),
                "command": command,
            }
        )

    results = [
        load_all_results_unlimited(str(item["path"]), map_location="cpu")[0]
        for item in components
    ]
    save_all_results_unlimited(str(task.output), results)
    elapsed = time.perf_counter() - started
    manifest = {
        "task": task.stem,
        "status": "complete",
        "finished_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "output": str(task.output),
        "checkpoint_bytes": task.output.stat().st_size,
        "checkpoint_sha256": _sha256(task.output),
        "activation": ACTIVATION,
        "negative_slope": NEGATIVE_SLOPE,
        "device": device,
        "components": components,
    }
    _write_json(manifest_path, manifest)
    print(f"\nAssembled checkpoint: {task.output}")
    print(f"SHA-256: {manifest['checkpoint_sha256']}")
    return task.output


def run_training(task: TrainingTask, device: str = DEFAULT_DEVICE) -> Path:
    """Run one training command while streaming output to the notebook and a log."""
    preflight(task, device)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if task.mode == "unlimited":
        return _run_published_unlimited(task, device)
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
        "training_schedule": (
            {
                f"T={horizon:g}": published_training_parameters(
                    task.application, task.mode, task.dimension, horizon
                )
                for horizon in published_horizons(task.mode, task.dimension)
            }
            if task.mode == "unlimited"
            else published_training_parameters(
                task.application, task.mode, task.dimension
            )
        ),
    }
    _write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(SEED)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
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
            encoding="utf-8",
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
