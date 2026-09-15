"""Mask-aware, dimensionless multi-task objectives."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_gradient_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    dx_mask = mask[..., :, 1:] & mask[..., :, :-1]
    dy_mask = mask[..., 1:, :] & mask[..., :-1, :]
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (
        _masked_mean((pred_dx - true_dx).abs(), dx_mask)
        + _masked_mean((pred_dy - true_dy).abs(), dy_mask)
    )


class MaskedMultiTaskLoss(nn.Module):
    def __init__(
        self,
        task_names: Sequence[str],
        *,
        task_weights: Sequence[float] | None = None,
        gradient_weight: float = 0.0,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        weights = task_weights or [1.0] * len(self.task_names)
        if len(weights) != len(self.task_names):
            raise ValueError("task_weights must match task_names")
        normalized = torch.tensor(weights, dtype=torch.float32)
        normalized = normalized / normalized.sum()
        self.register_buffer("task_weights", normalized)
        self.gradient_weight = float(gradient_weight)
        self.use_uncertainty = bool(use_uncertainty)

    def forward(
        self,
        output: dict[str, torch.Tensor | None],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = output["mean"]
        log_scale = output.get("log_scale")
        channel_losses = []
        details: dict[str, float] = {}
        for index, name in enumerate(self.task_names):
            error = (prediction[:, index] - target[:, index]).abs()
            channel_mask = mask[:, index]
            if self.use_uncertainty and log_scale is not None:
                scale = log_scale[:, index]
                data_loss = _masked_mean(torch.exp(-scale) * error + scale, channel_mask)
            else:
                data_loss = _masked_mean(error, channel_mask)
            gradient = _masked_gradient_l1(
                prediction[:, index], target[:, index], channel_mask
            )
            channel_loss = data_loss + self.gradient_weight * gradient
            channel_losses.append(channel_loss)
            details[f"{name}/data"] = float(data_loss.detach())
            details[f"{name}/gradient"] = float(gradient.detach())
        stacked = torch.stack(channel_losses)
        total = (stacked * self.task_weights).sum()
        details["total"] = float(total.detach())
        return total, details

