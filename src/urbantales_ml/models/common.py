"""Shared multi-task output heads."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        channels: int,
        task_names: Sequence[str],
        predict_uncertainty: bool,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        self.mean = nn.ModuleDict(
            {name: nn.Conv2d(channels, 1, 1) for name in self.task_names}
        )
        self.log_scale = (
            nn.ModuleDict({name: nn.Conv2d(channels, 1, 1) for name in self.task_names})
            if predict_uncertainty
            else None
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | None]:
        mean = torch.cat([self.mean[name](features) for name in self.task_names], dim=1)
        log_scale = None
        if self.log_scale is not None:
            log_scale = torch.cat(
                [self.log_scale[name](features) for name in self.task_names], dim=1
            ).clamp(-7.0, 5.0)
        return {"mean": mean, "log_scale": log_scale}

