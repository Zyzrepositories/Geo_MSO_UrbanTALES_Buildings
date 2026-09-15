"""Mask-aware evaluation metrics in model and physical units.

The UrbanTALES release used here contains pedestrian-height horizontal slices.
Consequently ``du/dx + dv/dy`` is reported only as a horizontal-divergence
diagnostic; without ``dw/dz`` it is not an incompressible-flow residual.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


VELOCITY_TARGETS = {"uped", "vped", "Uped"}
QUADRATIC_TARGETS = {"TKEped", "Tuwped"}


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    selected = values[mask]
    return selected if selected.numel() else None


def _add_error_metrics(
    result: dict[str, float],
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    error = prediction - target
    values = _masked_values(error, mask)
    truth = _masked_values(target, mask)
    if values is None or truth is None:
        return
    result[f"{prefix}/mae"] = float(values.abs().mean())
    result[f"{prefix}/rmse"] = float(values.square().mean().sqrt())
    result[f"{prefix}/relative_l2"] = float(
        values.square().sum().sqrt() / truth.square().sum().sqrt().clamp_min(1e-12)
    )
    selected_prediction = prediction[mask]
    centered_prediction = selected_prediction - selected_prediction.mean()
    centered_target = truth - truth.mean()
    denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    )
    if float(denominator) > 1e-12:
        result[f"{prefix}/pearson_r"] = float(
            (centered_prediction * centered_target).sum() / denominator
        )


def _physical_scale(
    task_name: str, u_tau_m_s: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    factor = u_tau_m_s.to(device=reference.device, dtype=reference.dtype)
    if task_name in QUADRATIC_TARGETS:
        factor = factor.square()
    elif task_name not in VELOCITY_TARGETS:
        raise KeyError(f"Unknown physical scale for {task_name}")
    return factor[:, None, None]


def _high_gradient_region(
    speed: torch.Tensor,
    valid: torch.Tensor,
    pixel_size_m: float,
    quantile: float,
) -> torch.Tensor:
    """Select the highest-gradient valid pixels independently per sample."""
    region = torch.zeros_like(valid)
    joint_valid = (
        valid[:, :-1, :-1]
        & valid[:, 1:, :-1]
        & valid[:, :-1, 1:]
    )
    dx = (speed[:, :-1, 1:] - speed[:, :-1, :-1]) / pixel_size_m
    dy = (speed[:, 1:, :-1] - speed[:, :-1, :-1]) / pixel_size_m
    magnitude = torch.sqrt(dx.square() + dy.square())
    for batch_index in range(speed.shape[0]):
        sample_values = magnitude[batch_index][joint_valid[batch_index]]
        if sample_values.numel():
            threshold = torch.quantile(sample_values.float(), quantile).to(magnitude.dtype)
            selected = joint_valid[batch_index] & (magnitude[batch_index] >= threshold)
            region[batch_index, :-1, :-1] = selected
    return region


def _add_vector_diagnostics(
    result: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    task_names: Sequence[str],
    u_tau_m_s: torch.Tensor,
    pixel_size_m: float,
    minimum_direction_speed_m_s: float,
) -> None:
    indices = {name: index for index, name in enumerate(task_names)}
    if not {"uped", "vped"}.issubset(indices):
        return
    u_index, v_index = indices["uped"], indices["vped"]
    velocity_scale = u_tau_m_s.to(prediction)[:, None, None]
    pred_u = prediction[:, u_index] * velocity_scale
    pred_v = prediction[:, v_index] * velocity_scale
    true_u = target[:, u_index] * velocity_scale
    true_v = target[:, v_index] * velocity_scale
    velocity_valid = mask[:, u_index] & mask[:, v_index]

    vector_error = torch.sqrt((pred_u - true_u).square() + (pred_v - true_v).square())
    values = _masked_values(vector_error, velocity_valid)
    if values is not None:
        result["vector/error_magnitude_mae_m_s"] = float(values.mean())
        result["vector/error_magnitude_rmse_m_s"] = float(values.square().mean().sqrt())

    true_speed = torch.sqrt(true_u.square() + true_v.square())
    direction_valid = velocity_valid & (true_speed >= minimum_direction_speed_m_s)
    if direction_valid.any():
        angle_delta = torch.atan2(pred_v, pred_u) - torch.atan2(true_v, true_u)
        wrapped = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta)).abs()
        result["vector/direction_mae_deg"] = float(
            torch.rad2deg(wrapped[direction_valid]).mean()
        )

    if "Uped" in indices:
        speed_index = indices["Uped"]
        speed_valid = velocity_valid & mask[:, speed_index]
        predicted_speed = prediction[:, speed_index] * velocity_scale
        target_speed = target[:, speed_index] * velocity_scale
        pred_consistency = (predicted_speed - torch.sqrt(pred_u.square() + pred_v.square())).abs()
        target_consistency = (target_speed - true_speed).abs()
        if speed_valid.any():
            result["consistency/predicted_Uped_vs_uv_mae_m_s"] = float(
                pred_consistency[speed_valid].mean()
            )
            result["consistency/target_Uped_vs_uv_mae_m_s"] = float(
                target_consistency[speed_valid].mean()
            )

    # Forward differences share the lower-left valid stencil. This quantity is
    # not a full mass-conservation residual because w and dw/dz are unavailable.
    divergence_valid = (
        velocity_valid[:, :-1, :-1]
        & velocity_valid[:, :-1, 1:]
        & velocity_valid[:, 1:, :-1]
    )
    pred_divergence = (
        (pred_u[:, :-1, 1:] - pred_u[:, :-1, :-1])
        + (pred_v[:, 1:, :-1] - pred_v[:, :-1, :-1])
    ) / pixel_size_m
    true_divergence = (
        (true_u[:, :-1, 1:] - true_u[:, :-1, :-1])
        + (true_v[:, 1:, :-1] - true_v[:, :-1, :-1])
    ) / pixel_size_m
    if divergence_valid.any():
        result["horizontal_divergence/pred_abs_mean_s_inv"] = float(
            pred_divergence[divergence_valid].abs().mean()
        )
        result["horizontal_divergence/target_abs_mean_s_inv"] = float(
            true_divergence[divergence_valid].abs().mean()
        )
        result["horizontal_divergence/error_mae_s_inv"] = float(
            (pred_divergence - true_divergence)[divergence_valid].abs().mean()
        )


def _add_uncertainty_metrics(
    result: dict[str, float],
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_scale: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Evaluate the Laplace scale used by the training likelihood."""
    absolute_error = (prediction - target).abs()
    laplace_scale = log_scale.exp()
    valid_error = _masked_values(absolute_error, mask)
    valid_scale = _masked_values(laplace_scale, mask)
    if valid_error is None or valid_scale is None:
        return
    result[f"{prefix}/laplace_nll"] = float(
        (valid_error / valid_scale.clamp_min(1e-12) + valid_scale.log()).mean()
    )
    for probability in (0.5, 0.9):
        half_width = -math.log1p(-probability) * valid_scale
        coverage = float((valid_error <= half_width).float().mean())
        label = int(probability * 100)
        result[f"{prefix}/coverage_{label}"] = coverage
        result[f"{prefix}/coverage_{label}_absolute_error"] = abs(coverage - probability)


def batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    task_names: Sequence[str],
    *,
    u_tau_m_s: torch.Tensor | None = None,
    input_tensor: torch.Tensor | None = None,
    log_scale: torch.Tensor | None = None,
    pixel_size_m: float = 1.0,
    sdf_scale_m: float = 64.0,
    near_building_distance_m: float = 8.0,
    high_gradient_quantile: float = 0.9,
    minimum_direction_speed_m_s: float = 0.1,
) -> dict[str, float]:
    """Return global and optional physical/regional diagnostics for one batch.

    The first four arguments preserve the original dimensionless-only API.
    Physical metrics are enabled by passing the per-sample friction velocities.
    ``input_tensor`` is the six-channel model input; its normalized SDF channel
    is used to define the air-side building-neighbourhood region.
    """
    prediction = prediction.float()
    target = target.float()
    mask = mask.bool()
    result: dict[str, float] = {}
    for index, name in enumerate(task_names):
        _add_error_metrics(
            result,
            name,
            prediction[:, index],
            target[:, index],
            mask[:, index],
        )

    if u_tau_m_s is None:
        return result
    if u_tau_m_s.ndim != 1 or u_tau_m_s.shape[0] != prediction.shape[0]:
        raise ValueError("u_tau_m_s must have one value per batch sample")
    if pixel_size_m <= 0 or sdf_scale_m <= 0:
        raise ValueError("pixel_size_m and sdf_scale_m must be positive")
    if not 0.0 <= high_gradient_quantile <= 1.0:
        raise ValueError("high_gradient_quantile must be in [0, 1]")

    indices = {name: index for index, name in enumerate(task_names)}
    high_gradient_region = None
    if "Uped" in indices:
        speed_index = indices["Uped"]
        speed_scale = _physical_scale("Uped", u_tau_m_s, prediction)
        speed = target[:, speed_index] * speed_scale
        high_gradient_region = _high_gradient_region(
            speed,
            mask[:, speed_index],
            pixel_size_m,
            high_gradient_quantile,
        )

    near_building_region = None
    if input_tensor is not None:
        if input_tensor.ndim != 4 or input_tensor.shape[1] < 3:
            raise ValueError("input_tensor must include height, occupancy and SDF channels")
        sdf_m = input_tensor[:, 2].to(prediction).float() * sdf_scale_m
        near_building_region = (sdf_m > 0.0) & (sdf_m <= near_building_distance_m)

    for index, name in enumerate(task_names):
        scale = _physical_scale(name, u_tau_m_s, prediction)
        pred_physical = prediction[:, index] * scale
        target_physical = target[:, index] * scale
        unit = "m_s" if name in VELOCITY_TARGETS else "m2_s2"
        _add_error_metrics(
            result,
            f"{name}/physical_{unit}",
            pred_physical,
            target_physical,
            mask[:, index],
        )
        if near_building_region is not None:
            _add_error_metrics(
                result,
                f"{name}/near_building_{unit}",
                pred_physical,
                target_physical,
                mask[:, index] & near_building_region,
            )
        if high_gradient_region is not None:
            _add_error_metrics(
                result,
                f"{name}/high_gradient_{unit}",
                pred_physical,
                target_physical,
                mask[:, index] & high_gradient_region,
            )
        if log_scale is not None:
            log_scale_physical = log_scale[:, index].float() + scale.log()
            _add_uncertainty_metrics(
                result,
                f"{name}/uncertainty_physical_{unit}",
                pred_physical,
                target_physical,
                log_scale_physical,
                mask[:, index],
            )

    _add_vector_diagnostics(
        result,
        prediction,
        target,
        mask,
        task_names,
        u_tau_m_s,
        pixel_size_m,
        minimum_direction_speed_m_s,
    )
    return result
