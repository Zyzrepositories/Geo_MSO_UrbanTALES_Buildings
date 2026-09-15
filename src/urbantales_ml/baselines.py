"""Transparent non-neural baselines for UrbanTALES field prediction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F


class ConstantFieldModel(torch.nn.Module):
    """Broadcast one constant per target over every requested patch."""

    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.register_buffer("values", values.float().reshape(1, -1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        mean = self.values.expand(inputs.shape[0], -1, inputs.shape[-2], inputs.shape[-1])
        return {"mean": mean, "log_scale": None}


def geometry_condition_features(
    inputs: torch.Tensor, embedding_pixels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return low-resolution geometry and scalar boundary-condition features."""
    if inputs.ndim != 4 or inputs.shape[1] < 6:
        raise ValueError("Expected [batch, >=6, height, width] UrbanTALES inputs")
    if embedding_pixels <= 0:
        raise ValueError("embedding_pixels must be positive")
    geometry = F.adaptive_avg_pool2d(inputs[:, :3].float(), embedding_pixels).flatten(1)
    conditions = inputs[:, 3:6].float().mean(dim=(-2, -1))
    return geometry, conditions


class PatchNearestNeighborModel(torch.nn.Module):
    """Copy the target of the closest training patch in geometry/BC feature space.

    Distance is the mean squared difference between pooled height, occupancy and
    SDF, plus ``boundary_weight`` times the mean squared difference between wind
    cosine, wind sine and normalized friction velocity. The target bank must be
    built exclusively from the training partition.
    """

    def __init__(
        self,
        bank_geometry: torch.Tensor,
        bank_conditions: torch.Tensor,
        bank_targets: torch.Tensor,
        donor_case_ids: Sequence[str],
        *,
        embedding_pixels: int,
        boundary_weight: float = 1.0,
    ):
        super().__init__()
        if bank_geometry.ndim != 2 or bank_conditions.ndim != 2:
            raise ValueError("Nearest-neighbor feature banks must be matrices")
        if bank_targets.ndim != 4:
            raise ValueError("Nearest-neighbor target bank must be [N,C,H,W]")
        bank_size = bank_geometry.shape[0]
        if bank_size == 0 or any(
            size != bank_size
            for size in (bank_conditions.shape[0], bank_targets.shape[0], len(donor_case_ids))
        ):
            raise ValueError("Nearest-neighbor bank dimensions are inconsistent")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative")
        self.register_buffer("bank_geometry", bank_geometry.float())
        self.register_buffer("bank_conditions", bank_conditions.float())
        self.register_buffer("bank_targets", bank_targets.float())
        self.donor_case_ids = tuple(donor_case_ids)
        self.embedding_pixels = int(embedding_pixels)
        self.boundary_weight = float(boundary_weight)
        self._matched_indices: list[int] = []

    def reset_matches(self) -> None:
        self._matched_indices.clear()

    def match_counts(self) -> dict[str, int]:
        counts = Counter(self.donor_case_ids[index] for index in self._matched_indices)
        return dict(sorted(counts.items()))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        geometry, conditions = geometry_condition_features(inputs, self.embedding_pixels)
        geometry_distance = (geometry[:, None] - self.bank_geometry[None]).square().mean(-1)
        condition_distance = (
            (conditions[:, None] - self.bank_conditions[None]).square().mean(-1)
        )
        nearest = (geometry_distance + self.boundary_weight * condition_distance).argmin(-1)
        self._matched_indices.extend(nearest.detach().cpu().tolist())
        return {"mean": self.bank_targets[nearest], "log_scale": None}
