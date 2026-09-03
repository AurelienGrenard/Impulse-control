"""Common, refactor-safe persistence for both numerical examples."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


Tensor = torch.Tensor


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """Write a checkpoint without exposing a partially written target file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(target)


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve project-relative checkpoint paths from scripts or notebooks."""
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    project_candidate = Path(__file__).resolve().parents[1] / candidate
    return project_candidate if project_candidate.exists() else candidate


def _application(path: str | os.PathLike[str], payload: Dict[str, Any]) -> str:
    """Identify the financial problem stored in a result."""
    name = Path(path).name.lower()
    if "dividend" in name:
        return "dividend"
    if "harvesting" in name:
        return "harvesting"
    cfg = None
    if payload.get("bounded_results"):
        cfg = payload["bounded_results"][0].get("cfg_dict")
    elif payload.get("results"):
        cfg = payload["results"][0].get("cfg_dict")
    if cfg and "dividend" in cfg:
        return "dividend"
    if cfg and "harvesting" in cfg:
        return "harvesting"
    raise ValueError("Application impossible à déduire du checkpoint.")


def _api(application: str):
    """Import the numerical API for a financial problem."""
    if application == "dividend":
        from . import dividend as module
        return module, "dividend", module.DividendParamsND, module.DividendProblemND
    from . import harvesting as module
    return module, "harvesting", module.HarvestingParamsND, module.HarvestingProblemND


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Serialize a PyTorch dtype."""
    return str(dtype).replace("torch.", "")


def _str_to_dtype(value: str) -> torch.dtype:
    """Restore a serialized PyTorch dtype."""
    if hasattr(torch, value):
        return getattr(torch, value)
    raise ValueError(f"Unknown dtype string: {value}")


def _tensor_to_cpu(value: Any) -> Any:
    """Detach a tensor and move it to CPU memory."""
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _serialize_qhats(qhats: Any) -> Dict[str, Any]:
    """Serialize continuation-network state dictionaries."""
    def state(q):
        """Serialize one continuation network."""
        return None if q is None else {key: value.detach().cpu() for key, value in q.state_dict().items()}

    if isinstance(qhats, list) and (not qhats or not isinstance(qhats[0], list)):
        return {"type": "unconstrained", "state_dicts": [state(q) for q in qhats]}
    if isinstance(qhats, list) and qhats and isinstance(qhats[0], list):
        return {"type": "bounded", "state_dicts": [[state(q) for q in row] for row in qhats]}
    raise TypeError("qhats must be a list or a list of lists.")


def _infer_input_dimension(state_dict: Dict[str, Tensor]) -> int:
    """Infer the network input dimension from saved weights."""
    weights = [(key, value) for key, value in state_dict.items() if value.ndim == 2 and key.endswith("weight")]
    if not weights:
        raise RuntimeError("No linear weight found in state_dict.")
    return int(next((value for key, value in weights if key.endswith("net.0.weight")), weights[0][1]).shape[1])


def _rebuild_cfg(cfg_dict: Dict[str, Any], application: str, map_location: str):
    """Rebuild an experiment configuration on the target device."""
    module, parameter_name, parameter_type, _ = _api(application)
    dtype = _str_to_dtype(cfg_dict.get("dtype", "float32"))

    def maybe_tensor(value):
        """Restore tensor-valued configuration fields."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=map_location, dtype=dtype)
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, device=map_location, dtype=dtype)
        return value

    parameter_dict = dict(cfg_dict[parameter_name])
    parameter_fields = ("mu", "sigma", "lam", "c") if application == "dividend" else (
        "mu", "sigma", "rho", "x0", "alpha", "lam", "c"
    )
    for name in parameter_fields:
        if name in parameter_dict:
            parameter_dict[name] = maybe_tensor(parameter_dict[name])
    parameter_dict.update(device=map_location, dtype=dtype)

    design_dict = dict(cfg_dict["design"])
    for name in ("x_min_k", "x_max_k"):
        if name in design_dict:
            design_dict[name] = maybe_tensor(design_dict[name])

    arguments = {
        "device": map_location,
        "dtype": dtype,
        "time": module.TimeGridConfig(**cfg_dict["time"]),
        "mc": module.MCConfig(**cfg_dict["mc"]),
        "opt": module.OptConfig(**cfg_dict["opt"]),
        "net": module.NetConfig(**cfg_dict["net"]),
        parameter_name: parameter_type(**parameter_dict),
        "design": module.DesignConfig(**design_dict),
        "max_impulses": int(cfg_dict.get("max_impulses", -1)),
        "horizon": cfg_dict.get("horizon"),
        "verbose": bool(cfg_dict.get("verbose", False)),
    }
    return module.ExperimentConfig(**arguments)


def _rebuild_result(item: Dict[str, Any], application: str, map_location: str) -> Dict[str, Any]:
    """Rebuild one saved result and its continuation networks."""
    module, _, _, problem_type = _api(application)
    cfg = _rebuild_cfg(item["cfg_dict"], application, map_location)
    parameter = cfg.dividend if application == "dividend" else cfg.harvesting
    problem = problem_type(parameter, device=cfg.device, dtype=cfg.dtype)

    def network(state_dict):
        """Rebuild one continuation network from saved weights."""
        if state_dict is None:
            return None
        net = module.QNet(_infer_input_dimension(state_dict), cfg.net).to(map_location, dtype=cfg.dtype)
        net.load_state_dict(state_dict)
        net.eval()
        return net

    pack = item["qhats_pack"]
    if pack["type"] == "bounded":
        qhats = [[network(state) for state in row] for row in pack["state_dicts"]]
    else:
        qhats = [network(state) for state in pack["state_dicts"]]
    grid = item.get("t_grid", cfg.t_grid).to(map_location, dtype=cfg.dtype)
    result = {"cfg": cfg, "problem": problem, "t_grid": grid, "qhats": qhats}
    for key in (
        "T",
        "max_impulses",
        "V0_1",
        "mc_mean_NN",
        "mc_std_NN",
        "mc_n_NN",
        "mc_seed_NN",
        "mc_batch_size_NN",
        "x",
        "V",
    ):
        if key in item:
            result[key] = item[key]
    return result


def load_results_bundle(path: str, *, map_location: str = "cpu") -> Dict[str, Any]:
    """Load bounded and unconstrained results from one checkpoint."""
    path = str(_resolve_path(path))
    payload = torch.load(path, map_location=map_location, weights_only=False)
    application = _application(path, payload)
    bounded = [_rebuild_result(item, application, map_location) for item in payload["bounded_results"]]
    output: Dict[str, Any] = {"bounded_results": bounded, "unconstrained": None}
    if payload.get("unconstrained") is not None:
        item = dict(payload["unconstrained"])
        rebuilt = _rebuild_result(item, application, map_location)
        output["unconstrained"] = {
            "config": rebuilt["cfg"],
            "problem": rebuilt["problem"],
            "t_grid": rebuilt["t_grid"],
            "qhats": rebuilt["qhats"],
            "x_np_inf": item.get("x_np_inf"),
            "V_np_inf": item.get("V_np_inf"),
            "V0_1_inf": item.get("V0_1_inf"),
        }
    return output


def load_all_results_unlimited(path: str, *, map_location: str = "cpu") -> List[Dict[str, Any]]:
    """Load all horizons from an unconstrained checkpoint."""
    path = str(_resolve_path(path))
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("mode") != "unconstrained_multiT":
        raise RuntimeError(f"Unexpected file mode: {payload.get('mode')}")
    application = _application(path, payload)
    results = [_rebuild_result(item, application, map_location) for item in payload["results"]]
    results.sort(key=lambda result: result["T"])
    return results


def _pack_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one in-memory result into a portable payload."""
    cfg = result.get("cfg", result.get("config"))
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else cfg
    cfg_dict["dtype"] = _dtype_to_str(cfg.dtype)
    cfg_dict["device"] = str(cfg.device)
    packed = {
        "cfg_dict": cfg_dict,
        "t_grid": _tensor_to_cpu(result["t_grid"]),
        "qhats_pack": _serialize_qhats(result["qhats"]),
    }
    for key in (
        "T",
        "max_impulses",
        "V0_1",
        "mc_mean_NN",
        "mc_std_NN",
        "mc_n_NN",
        "mc_seed_NN",
        "mc_batch_size_NN",
        "x",
        "V",
    ):
        if key in result:
            packed[key] = _tensor_to_cpu(result[key])
    return packed


def save_results_bundle(
    path: str,
    *,
    bounded_results: List[Dict[str, Any]],
    unconstrained: Optional[Dict[str, Any]] = None,
) -> None:
    """Save bounded and unconstrained results in one checkpoint."""
    payload: Dict[str, Any] = {
        "version": 2,
        "bounded_results": [_pack_result(result) for result in bounded_results],
        "unconstrained": None,
    }
    if unconstrained is not None:
        payload["unconstrained"] = _pack_result(unconstrained) | {
            "x_np_inf": unconstrained.get("x_np_inf"),
            "V_np_inf": unconstrained.get("V_np_inf"),
            "V0_1_inf": unconstrained.get("V0_1_inf"),
        }
    _atomic_torch_save(payload, path)


def save_all_results_unlimited(path: str, all_results: List[Dict[str, Any]]) -> None:
    """Save all unconstrained horizons in one checkpoint."""
    payload = {"version": 3, "mode": "unconstrained_multiT", "results": [_pack_result(r) for r in all_results]}
    _atomic_torch_save(payload, path)
