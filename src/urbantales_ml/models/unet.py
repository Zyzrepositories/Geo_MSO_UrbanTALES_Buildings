"""Maintainable U-Net baseline with multi-task and uncertainty heads."""

from __future__ import annotations

import torch
from torch import nn

from .common import MultiTaskHeads


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([self.up(inputs), skip], dim=1))


class MultiTaskUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        task_names: list[str] | tuple[str, ...],
        *,
        base_channels: int = 32,
        depth: int = 4,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.task_names = tuple(task_names)
        self.predict_uncertainty = bool(predict_uncertainty)
        widths = [base_channels * (2**level) for level in range(depth)]
        self.stem = ResidualBlock(in_channels, widths[0])
        self.downs = nn.ModuleList(
            DownBlock(widths[level], widths[level + 1]) for level in range(depth - 1)
        )
        self.bottleneck = DownBlock(widths[-1], widths[-1] * 2)
        decoder_in = widths[-1] * 2
        ups = []
        for skip_channels in reversed(widths):
            ups.append(UpBlock(decoder_in, skip_channels, skip_channels))
            decoder_in = skip_channels
        self.ups = nn.ModuleList(ups)
        self.heads = MultiTaskHeads(widths[0], self.task_names, self.predict_uncertainty)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        skips = [self.stem(inputs)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        features = self.bottleneck(skips[-1])
        for up, skip in zip(self.ups, reversed(skips)):
            features = up(features, skip)
        return self.heads(features)
