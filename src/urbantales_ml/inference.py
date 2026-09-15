"""Periodic overlap-add inference for complete UrbanTALES horizontal fields."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .catalog import CaseRecord
from .data import (
    UrbanTalesPatchDataset,
    ablate_model_input_channels,
    build_input_patch,
    load_geometry,
)


def _tile_origins(length: int, stride: int) -> list[int]:
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    return list(range(0, length, stride))


def _blend_window(size: int) -> np.ndarray:
    window_1d = torch.hann_window(size, periodic=False).clamp_min(1e-3)
    return torch.outer(window_1d, window_1d).numpy().astype(np.float32, copy=False)


def _periodic_add_shared_weight(
    value_sum: np.ndarray,
    weight_sum: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    """Add a channel-first tile, including when a tile wraps more than once."""
    _, tile_y, tile_x = values.shape
    rows = (np.arange(tile_y) + y0) % value_sum.shape[-2]
    cols = (np.arange(tile_x) + x0) % value_sum.shape[-1]
    flat_indices = (rows[:, None] * value_sum.shape[-1] + cols[None, :]).reshape(-1)
    flat_weights = weights.reshape(-1)
    np.add.at(weight_sum.reshape(-1), flat_indices, flat_weights)
    for channel in range(value_sum.shape[0]):
        np.add.at(
            value_sum[channel].reshape(-1),
            flat_indices,
            (values[channel] * weights).reshape(-1),
        )


def _periodic_add_channel_weights(
    value_sum: np.ndarray,
    weight_sum: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    _, tile_y, tile_x = values.shape
    rows = (np.arange(tile_y) + y0) % value_sum.shape[-2]
    cols = (np.arange(tile_x) + x0) % value_sum.shape[-1]
    flat_indices = (rows[:, None] * value_sum.shape[-1] + cols[None, :]).reshape(-1)
    for channel in range(value_sum.shape[0]):
        flat_weights = weights[channel].reshape(-1)
        np.add.at(weight_sum[channel].reshape(-1), flat_indices, flat_weights)
        np.add.at(
            value_sum[channel].reshape(-1),
            flat_indices,
            (values[channel] * weights[channel]).reshape(-1),
        )


@torch.inference_mode()
def predict_full_case(
    model: torch.nn.Module,
    dataset: UrbanTalesPatchDataset,
    case_index: int,
    device: torch.device,
    *,
    stride_pixels: int | None = None,
    batch_size: int = 4,
    amp: bool = True,
    zero_model_input_channels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Predict and reconstruct one complete periodic field at model resolution.

    The returned target is reconstructed with the identical overlap-add path,
    which makes seam diagnostics and 0.5 m-to-1 m comparisons explicit. Source
    resolutions must divide the model pixel spacing exactly (true for the 0.5 m
    and 1.0 m cases in this release and the default 1.0 m model grid).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    case = dataset.cases[case_index]
    loaded = dataset._load(case)
    model_pixel_size_m = dataset.patch_size_m / dataset.output_pixels
    native_pixels_per_model_pixel = model_pixel_size_m / case.dx_m
    rounded_ratio = round(native_pixels_per_model_pixel)
    if rounded_ratio < 1 or abs(native_pixels_per_model_pixel - rounded_ratio) > 1e-6:
        raise ValueError(
            f"{case.case_id}: model pixel size {model_pixel_size_m} m is not an "
            f"integer multiple of source dx {case.dx_m} m"
        )
    source_ny, source_nx = loaded["topo"].shape
    output_ny = max(1, int(round(source_ny / rounded_ratio)))
    output_nx = max(1, int(round(source_nx / rounded_ratio)))
    patch_pixels = dataset.output_pixels
    stride = int(stride_pixels or max(1, patch_pixels // 2))
    if stride > patch_pixels:
        raise ValueError("stride_pixels cannot exceed the patch width")
    origins = [
        (y0, x0)
        for y0 in _tile_origins(output_ny, stride)
        for x0 in _tile_origins(output_nx, stride)
    ]
    window = _blend_window(patch_pixels)
    task_count = len(dataset.targets)
    input_count = len(dataset.input_channels)
    prediction_sum = np.zeros((task_count, output_ny, output_nx), dtype=np.float32)
    prediction_weight = np.zeros((output_ny, output_nx), dtype=np.float32)
    scale_sum: np.ndarray | None = None
    scale_weight: np.ndarray | None = None
    input_sum = np.zeros((input_count, output_ny, output_nx), dtype=np.float32)
    input_weight = np.zeros((output_ny, output_nx), dtype=np.float32)
    target_sum = np.zeros((task_count, output_ny, output_nx), dtype=np.float32)
    target_weight = np.zeros_like(target_sum)

    model.eval()
    total_start = time.perf_counter()
    forward_seconds = 0.0
    for batch_start in range(0, len(origins), batch_size):
        batch_origins = origins[batch_start : batch_start + batch_size]
        samples = []
        for offset, (output_y0, output_x0) in enumerate(batch_origins):
            samples.append(
                dataset.sample_at(
                    case_index,
                    output_y0 * rounded_ratio,
                    output_x0 * rounded_ratio,
                    patch_index=batch_start + offset,
                )
            )
        inputs = torch.stack([sample["input"] for sample in samples]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(ablate_model_input_channels(inputs, zero_model_input_channels))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - forward_start

        means = output["mean"].float().cpu().numpy()
        log_scales = output.get("log_scale")
        scales = log_scales.float().exp().cpu().numpy() if log_scales is not None else None
        if scales is not None and scale_sum is None:
            scale_sum = np.zeros_like(prediction_sum)
            scale_weight = np.zeros_like(prediction_weight)
        for index, (output_y0, output_x0) in enumerate(batch_origins):
            sample_input = np.asarray(samples[index]["input"], dtype=np.float32)
            sample_target = np.asarray(samples[index]["target"], dtype=np.float32)
            sample_mask = np.asarray(samples[index]["mask"], dtype=bool)
            _periodic_add_shared_weight(
                prediction_sum,
                prediction_weight,
                means[index],
                window,
                output_y0,
                output_x0,
            )
            _periodic_add_shared_weight(
                input_sum,
                input_weight,
                sample_input,
                window,
                output_y0,
                output_x0,
            )
            target_weights = window[None, :, :] * sample_mask
            _periodic_add_channel_weights(
                target_sum,
                target_weight,
                sample_target,
                target_weights,
                output_y0,
                output_x0,
            )
            if scales is not None and scale_sum is not None and scale_weight is not None:
                # The learned Laplace scale, not its logarithm, is blended.
                _periodic_add_shared_weight(
                    scale_sum,
                    scale_weight,
                    scales[index],
                    window,
                    output_y0,
                    output_x0,
                )

    total_seconds = time.perf_counter() - total_start
    safe_prediction_weight = np.maximum(prediction_weight, 1e-12)
    safe_input_weight = np.maximum(input_weight, 1e-12)
    prediction = prediction_sum / safe_prediction_weight[None, :, :]
    reconstructed_input = input_sum / safe_input_weight[None, :, :]
    valid = target_weight > 0
    target = target_sum / np.maximum(target_weight, 1e-12)
    log_scale = None
    if scale_sum is not None and scale_weight is not None:
        log_scale = np.log(
            np.maximum(
                scale_sum / np.maximum(scale_weight[None, :, :], 1e-12),
                1e-12,
            )
        )
    return {
        "mean": torch.from_numpy(prediction).unsqueeze(0),
        "log_scale": torch.from_numpy(log_scale).unsqueeze(0) if log_scale is not None else None,
        "target": torch.from_numpy(target).unsqueeze(0),
        "mask": torch.from_numpy(valid).unsqueeze(0),
        "input": torch.from_numpy(reconstructed_input).unsqueeze(0),
        "case_id": case.case_id,
        "u_tau_m_s": torch.tensor([case.u_tau_m_s], dtype=torch.float32),
        "source_dx_m": case.dx_m,
        "model_pixel_size_m": model_pixel_size_m,
        "output_shape": [output_ny, output_nx],
        "tile_count": len(origins),
        "stride_pixels": stride,
        "forward_seconds": forward_seconds,
        "end_to_end_seconds": total_seconds,
    }


@torch.inference_mode()
def predict_full_case_inputs_only(
    model: torch.nn.Module,
    root: str | Path,
    case: CaseRecord,
    device: torch.device,
    *,
    patch_size_m: float,
    output_pixels: int,
    height_scale_m: float,
    sdf_scale_m: float,
    u_tau_reference_m_s: float,
    stride_pixels: int,
    batch_size: int,
    amp: bool = True,
    geometry: dict[str, np.ndarray] | None = None,
    zero_model_input_channels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Deployment path that never opens target NetCDF or computes metrics."""
    if batch_size <= 0 or stride_pixels <= 0:
        raise ValueError("batch_size and stride_pixels must be positive")
    if stride_pixels > output_pixels:
        raise ValueError("stride_pixels cannot exceed output_pixels")
    total_start = time.perf_counter()
    geometry_load_start = time.perf_counter()
    if geometry is None:
        geometry = load_geometry(Path(root).resolve(), case)
    geometry_file_load_seconds = time.perf_counter() - geometry_load_start

    model_pixel_size_m = float(patch_size_m) / int(output_pixels)
    native_pixels_per_model_pixel = model_pixel_size_m / case.dx_m
    rounded_ratio = round(native_pixels_per_model_pixel)
    if rounded_ratio < 1 or abs(native_pixels_per_model_pixel - rounded_ratio) > 1e-6:
        raise ValueError(
            f"{case.case_id}: model pixel size {model_pixel_size_m} m is not an "
            f"integer multiple of source dx {case.dx_m} m"
        )
    source_ny, source_nx = geometry["topo"].shape
    output_ny = max(1, int(round(source_ny / rounded_ratio)))
    output_nx = max(1, int(round(source_nx / rounded_ratio)))
    origins = [
        (y0, x0)
        for y0 in _tile_origins(output_ny, stride_pixels)
        for x0 in _tile_origins(output_nx, stride_pixels)
    ]
    window = _blend_window(output_pixels)
    prediction_sum: np.ndarray | None = None
    prediction_weight = np.zeros((output_ny, output_nx), dtype=np.float32)
    scale_sum: np.ndarray | None = None
    scale_weight: np.ndarray | None = None
    input_preprocessing_seconds = 0.0
    host_to_device_seconds = 0.0
    forward_seconds = 0.0
    device_to_host_seconds = 0.0
    stitching_seconds = 0.0

    model.eval()
    for batch_start in range(0, len(origins), batch_size):
        batch_origins = origins[batch_start : batch_start + batch_size]
        phase_start = time.perf_counter()
        inputs_cpu = torch.stack(
            [
                build_input_patch(
                    case,
                    geometry,
                    y0=output_y0 * rounded_ratio,
                    x0=output_x0 * rounded_ratio,
                    patch_size_m=patch_size_m,
                    output_pixels=output_pixels,
                    height_scale_m=height_scale_m,
                    sdf_scale_m=sdf_scale_m,
                    u_tau_reference_m_s=u_tau_reference_m_s,
                )
                for output_y0, output_x0 in batch_origins
            ]
        )
        inputs_cpu = ablate_model_input_channels(inputs_cpu, zero_model_input_channels)
        input_preprocessing_seconds += time.perf_counter() - phase_start

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_start = time.perf_counter()
        inputs = inputs_cpu.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        host_to_device_seconds += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        means = output["mean"].float().cpu().numpy()
        log_scales = output.get("log_scale")
        scales = log_scales.float().exp().cpu().numpy() if log_scales is not None else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        device_to_host_seconds += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if prediction_sum is None:
            prediction_sum = np.zeros(
                (means.shape[1], output_ny, output_nx), dtype=np.float32
            )
        if scales is not None and scale_sum is None:
            scale_sum = np.zeros_like(prediction_sum)
            scale_weight = np.zeros_like(prediction_weight)
        for index, (output_y0, output_x0) in enumerate(batch_origins):
            _periodic_add_shared_weight(
                prediction_sum,
                prediction_weight,
                means[index],
                window,
                output_y0,
                output_x0,
            )
            if scales is not None and scale_sum is not None and scale_weight is not None:
                _periodic_add_shared_weight(
                    scale_sum,
                    scale_weight,
                    scales[index],
                    window,
                    output_y0,
                    output_x0,
                )
        stitching_seconds += time.perf_counter() - phase_start

    if prediction_sum is None:
        raise RuntimeError(f"{case.case_id}: no inference tiles were generated")
    phase_start = time.perf_counter()
    prediction = prediction_sum / np.maximum(prediction_weight[None, :, :], 1e-12)
    log_scale = None
    if scale_sum is not None and scale_weight is not None:
        log_scale = np.log(
            np.maximum(
                scale_sum / np.maximum(scale_weight[None, :, :], 1e-12),
                1e-12,
            )
        )
    stitching_seconds += time.perf_counter() - phase_start
    return {
        "mean": torch.from_numpy(prediction).unsqueeze(0),
        "log_scale": torch.from_numpy(log_scale).unsqueeze(0) if log_scale is not None else None,
        "case_id": case.case_id,
        "source_dx_m": case.dx_m,
        "model_pixel_size_m": model_pixel_size_m,
        "output_shape": [output_ny, output_nx],
        "tile_count": len(origins),
        "stride_pixels": stride_pixels,
        "timing_seconds": {
            "geometry_file_load": geometry_file_load_seconds,
            "input_preprocessing": input_preprocessing_seconds,
            "host_to_device": host_to_device_seconds,
            "GPU_forward": forward_seconds,
            "device_to_host": device_to_host_seconds,
            "overlap_add_stitching": stitching_seconds,
            "total_online": time.perf_counter() - total_start,
        },
    }
