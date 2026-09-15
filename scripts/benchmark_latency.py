#!/usr/bin/env python3
"""Benchmark frozen patch and target-free full-field deployment latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402
from urbantales_ml.config import load_config  # noqa: E402
from urbantales_ml.data import (  # noqa: E402
    ablate_model_input_channels,
    build_input_patch,
    load_geometry,
)
from urbantales_ml.inference import predict_full_case_inputs_only  # noqa: E402
from urbantales_ml.protocol import sha256_file, validate_frozen_protocol  # noqa: E402
from urbantales_ml.runner import _build_model, seed_everything  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_latency_model_lock(
    lock_path: Path,
    protocol_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    lock_path = lock_path.expanduser().resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("model_and_hyperparameter_selection") != "closed":
        raise ValueError("Latency benchmark requires the closed final primary model lock")
    if _sha256(protocol_path.expanduser().resolve()) != lock["evaluation_protocol"]["sha256"]:
        raise ValueError("Latency evaluation protocol hash differs from the model lock")
    if _sha256(config_path.expanduser().resolve()) != lock["selected_config"]["sha256"]:
        raise ValueError("Latency config hash differs from the model lock")
    reference_seed = int(lock["latency_reference_seed"])
    matches = [
        row for row in lock["selected_checkpoints"] if int(row["seed"]) == reference_seed
    ]
    if len(matches) != 1:
        raise ValueError("Latency reference seed must identify exactly one checkpoint")
    if _sha256(checkpoint_path.expanduser().resolve()) != matches[0]["sha256"]:
        raise ValueError("Latency checkpoint hash differs from the locked reference checkpoint")
    return lock


def _verify_latency_execution_protocol(
    execution_protocol_path: Path,
    evaluation_protocol_path: Path,
    model_lock_path: Path,
) -> dict[str, Any]:
    execution_protocol_path = execution_protocol_path.expanduser().resolve()
    execution = yaml.safe_load(execution_protocol_path.read_text(encoding="utf-8"))
    if execution.get("status") != "frozen":
        raise ValueError("Latency execution protocol must be frozen")
    references = {
        "evaluation_protocol": evaluation_protocol_path.expanduser().resolve(),
        "model_lock": model_lock_path.expanduser().resolve(),
        "implementation": Path(__file__).resolve(),
    }
    for key, path in references.items():
        if _sha256(path) != execution[key]["sha256"]:
            raise ValueError(f"Latency {key} hash differs from execution protocol")
    return execution


def _exclusive_gpu_processes() -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    foreign = sorted({pid for pid in pids if pid != os.getpid()})
    if foreign:
        raise RuntimeError(f"Frozen latency protocol requires an exclusive GPU; foreign PIDs: {foreign}")
    return sorted(set(pids))


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _timed_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/evaluation/frozen_protocol_v1.yaml",
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=ROOT / "configs/evaluation/final_primary_model_lock_v1.json",
    )
    parser.add_argument(
        "--execution-protocol",
        type=Path,
        default=ROOT / "configs/evaluation/latency_execution_protocol_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors = validate_frozen_protocol(ROOT, args.protocol)
    if errors:
        raise ValueError(f"Frozen evaluation protocol failed validation: {errors}")
    latency_execution = _verify_latency_execution_protocol(
        args.execution_protocol,
        args.protocol,
        args.model_lock,
    )
    protocol = load_config(args.protocol)
    config = load_config(args.config)
    model_lock = _verify_latency_model_lock(
        args.model_lock,
        args.protocol,
        args.config,
        args.checkpoint,
    )
    latency = protocol["latency"]
    patch_protocol = latency["patch_model_only"]
    field_protocol = latency["full_field_deployment"]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Latency output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Frozen latency protocol requires CUDA")
    actual_gpu = torch.cuda.get_device_name(device)
    if actual_gpu != latency["hardware"]["gpu"]:
        raise RuntimeError(
            f"Frozen protocol requires {latency['hardware']['gpu']}, found {actual_gpu}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("tasks", ())) != tuple(config["data"]["targets"]):
        raise ValueError("Checkpoint targets do not match latency config")
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    gpu_compute_pids = _exclusive_gpu_processes()
    amp = bool(config["training"].get("amp", True))
    root = Path(config["data"]["root"]).expanduser().resolve()
    catalog = UrbanTalesCatalog(root)
    sentinel_ids = list(field_protocol["sentinel_case_ids"])
    cases = catalog.subset(sentinel_ids)
    data = config["data"]

    first_case = cases[0]
    first_geometry = load_geometry(root, first_case)
    raw_pixels = max(8, int(round(data["patch_size_m"] / first_case.dx_m)))
    source_ny, source_nx = first_geometry["topo"].shape
    patch = build_input_patch(
        first_case,
        first_geometry,
        y0=(source_ny - raw_pixels) // 2,
        x0=(source_nx - raw_pixels) // 2,
        patch_size_m=data["patch_size_m"],
        output_pixels=data["output_pixels"],
        height_scale_m=data.get("height_scale_m", 50.0),
        sdf_scale_m=data.get("sdf_scale_m", 64.0),
        u_tau_reference_m_s=data.get("u_tau_reference_m_s", 0.21),
    )
    patch = ablate_model_input_channels(patch, data.get("zero_model_input_channels"))
    patch_results: dict[str, Any] = {}
    for batch_size in patch_protocol["batch_sizes"]:
        inputs = patch.unsqueeze(0).expand(int(batch_size), -1, -1, -1).to(device)
        for _ in range(int(patch_protocol["warmup_repeats"])):
            _timed_forward(model, inputs, device, amp)
        torch.cuda.reset_peak_memory_stats(device)
        values = [
            _timed_forward(model, inputs, device, amp)
            for _ in range(int(patch_protocol["measured_repeats"]))
        ]
        patch_results[str(batch_size)] = {
            "seconds": _statistics(values),
            "raw_seconds": values,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }

    field_results: list[dict[str, Any]] = []
    for case in cases:
        common = {
            "patch_size_m": float(data["patch_size_m"]),
            "output_pixels": int(data["output_pixels"]),
            "height_scale_m": float(data.get("height_scale_m", 50.0)),
            "sdf_scale_m": float(data.get("sdf_scale_m", 64.0)),
            "u_tau_reference_m_s": float(data.get("u_tau_reference_m_s", 0.21)),
            "stride_pixels": int(protocol["field_reconstruction"]["stride_pixels"]),
            "batch_size": int(field_protocol["batch_size"]),
            "amp": amp,
            "zero_model_input_channels": data.get("zero_model_input_channels"),
        }
        cold = predict_full_case_inputs_only(model, root, case, device, **common)
        geometry = load_geometry(root, case)
        for _ in range(int(field_protocol["warmup_repeats_per_case"])):
            predict_full_case_inputs_only(
                model, root, case, device, geometry=geometry, **common
            )
        torch.cuda.reset_peak_memory_stats(device)
        repeats = [
            predict_full_case_inputs_only(
                model, root, case, device, geometry=geometry, **common
            )
            for _ in range(int(field_protocol["measured_repeats_per_case"]))
        ]
        phase_names = repeats[0]["timing_seconds"]
        phase_statistics = {
            phase: _statistics([row["timing_seconds"][phase] for row in repeats])
            for phase in phase_names
        }
        field_results.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "source_dx_m": case.dx_m,
                "output_shape": cold["output_shape"],
                "tile_count": cold["tile_count"],
                "cold_timing_seconds": cold["timing_seconds"],
                "warm_timing_statistics_seconds": phase_statistics,
                "warm_raw_timing_seconds": [row["timing_seconds"] for row in repeats],
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )

    result = {
        "status": "ok",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model": config["model"]["name"],
        "model_lock": str(args.model_lock.expanduser().resolve()),
        "model_lock_sha256": _sha256(args.model_lock.expanduser().resolve()),
        "latency_execution_protocol": str(args.execution_protocol.expanduser().resolve()),
        "latency_execution_protocol_sha256": _sha256(args.execution_protocol.expanduser().resolve()),
        "latency_reference_seed": int(model_lock["latency_reference_seed"]),
        "exclusive_gpu_compute_pids_at_start": gpu_compute_pids,
        "target_fields_loaded": False,
        "patch_model_only": patch_results,
        "full_field_deployment": field_results,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": actual_gpu,
            "device": str(device),
        },
        "notes": [
            f"Execution policy: {latency_execution['protocol_id']}",
            "Cold timing includes topography text parsing but may benefit from OS page cache.",
            (
                "Warm full-field timing uses preloaded geometry and excludes target NetCDF "
                "and metrics."
            ),
            "CFD speedup must use total_online, never GPU_forward alone.",
        ],
    }
    (args.output / "latency.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
