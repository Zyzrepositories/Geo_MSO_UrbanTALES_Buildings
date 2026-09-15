"""Compact CNN and Fourier-neural-operator baselines."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .common import MultiTaskHeads
from .unet import ResidualBlock, _groups


class LightCNN(nn.Module):
    """Small dilated CNN with no encoder-decoder hierarchy."""

    def __init__(
        self,
        in_channels: int,
        task_names: Sequence[str],
        *,
        width: int = 32,
        layers: int = 6,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv2d(in_channels, width, 3, padding=1), nn.SiLU()]
        for index in range(layers):
            dilation = (1, 2, 4)[index % 3]
            blocks.extend(
                [
                    nn.Conv2d(
                        width,
                        width,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=width,
                        bias=False,
                    ),
                    nn.GroupNorm(_groups(width), width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 1),
                    nn.SiLU(),
                ]
            )
        self.body = nn.Sequential(*blocks)
        self.heads = MultiTaskHeads(width, task_names, predict_uncertainty)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return self.heads(self.body(inputs))


class SpectralConv2d(nn.Module):
    """Conventional dense Fourier layer used only for the FNO baseline."""

    def __init__(self, channels: int, modes_y: int, modes_x: int):
        super().__init__()
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 1.0 / max(channels, 1)
        shape = (channels, channels, self.modes_y, self.modes_x)
        component_scale = scale / math.sqrt(2.0)
        self.weight_top_real = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_top_imag = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_bottom_real = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_bottom_imag = nn.Parameter(component_scale * torch.randn(*shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # FFTs stay in float32 under AMP; half-precision FFT support is more
        # restrictive and can differ across CUDA/PyTorch builds.
        spectrum = torch.fft.rfft2(inputs.float(), norm="ortho")
        output = torch.zeros_like(spectrum)
        my = min(self.modes_y, spectrum.shape[-2] // 2)
        mx = min(self.modes_x, spectrum.shape[-1])
        if my and mx:
            weight_top = torch.complex(self.weight_top_real, self.weight_top_imag)
            weight_bottom = torch.complex(self.weight_bottom_real, self.weight_bottom_imag)
            output[:, :, :my, :mx] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, :my, :mx], weight_top[:, :, :my, :mx]
            )
            output[:, :, -my:, :mx] = torch.einsum(
                "bixy,ioxy->boxy",
                spectrum[:, :, -my:, :mx],
                weight_bottom[:, :, :my, :mx],
            )
        return torch.fft.irfft2(output, s=inputs.shape[-2:], norm="ortho").to(inputs.dtype)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(_groups(width), width)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.spectral(inputs) + self.local(inputs)))


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        task_names: Sequence[str],
        *,
        width: int = 48,
        layers: int = 4,
        modes_y: int = 16,
        modes_x: int = 16,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.Sequential(
            *(FNOBlock(width, modes_y, modes_x) for _ in range(layers))
        )
        self.project = ResidualBlock(width, width)
        self.heads = MultiTaskHeads(width, task_names, predict_uncertainty)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return self.heads(self.project(self.blocks(self.lift(inputs))))
