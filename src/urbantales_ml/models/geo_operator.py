"""Scale-conditioned multi-scale local/spectral operator (proposed model)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .common import MultiTaskHeads
from .unet import DownBlock, ResidualBlock, UpBlock, _groups


class DepthwiseSpectralConv2d(nn.Module):
    """Memory-bounded Fourier mixing followed by learned channel mixing."""

    def __init__(self, channels: int, modes_y: int, modes_x: int):
        super().__init__()
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = channels**-0.5
        shape = (channels, self.modes_y, self.modes_x)
        # Store spectral coefficients as real parameters. PyTorch's CUDA AMP
        # GradScaler cannot unscale ComplexFloat optimizer parameters, whereas
        # gradients propagate normally through torch.complex(real, imag).
        component_scale = scale / math.sqrt(2.0)
        self.weight_top_real = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_top_imag = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_bottom_real = nn.Parameter(component_scale * torch.randn(*shape))
        self.weight_bottom_imag = nn.Parameter(component_scale * torch.randn(*shape))
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Keep FFT numerics portable across CPU tests and CUDA AMP execution.
        spectrum = torch.fft.rfft2(inputs.float(), norm="ortho")
        output = torch.zeros_like(spectrum)
        my = min(self.modes_y, spectrum.shape[-2] // 2)
        mx = min(self.modes_x, spectrum.shape[-1])
        if my and mx:
            weight_top = torch.complex(self.weight_top_real, self.weight_top_imag)
            weight_bottom = torch.complex(self.weight_bottom_real, self.weight_bottom_imag)
            output[:, :, :my, :mx] = (
                spectrum[:, :, :my, :mx] * weight_top[:, :my, :mx]
            )
            output[:, :, -my:, :mx] = (
                spectrum[:, :, -my:, :mx] * weight_bottom[:, :my, :mx]
            )
        spatial = torch.fft.irfft2(output, s=inputs.shape[-2:], norm="ortho")
        return self.mix(spatial.to(inputs.dtype))


class ConditionedLocalSpectralBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        modes: int,
        *,
        use_film: bool,
    ):
        super().__init__()
        self.spectral = DepthwiseSpectralConv2d(channels, modes, modes)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.film = nn.Linear(condition_dim, 2 * channels) if use_film else None
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, 1),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.norm(self.spectral(inputs) + self.local(inputs))
        if self.film is not None:
            gamma, beta = self.film(condition).chunk(2, dim=1)
            features = features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        features = self.activation(features)
        return inputs + self.mlp(features)


class GeoMultiScaleOperator(nn.Module):
    """Local geometry encoder plus Fourier mixing at the deepest physical scales.

    Wind cosine/sine and normalized friction velocity are taken from the last
    three spatially constant input channels and used as FiLM conditions.  The
    factorized depthwise spectral path avoids the quadratic channel cost of a
    dense high-width FNO bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        task_names: Sequence[str],
        *,
        base_channels: int = 32,
        depth: int = 4,
        operator_levels: int = 2,
        modes: int = 12,
        use_film: bool = True,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if not 1 <= operator_levels <= depth + 1:
            raise ValueError("operator_levels must be in [1, depth + 1]")
        widths = [base_channels * (2**level) for level in range(depth)]
        self.stem = ResidualBlock(in_channels, widths[0])
        self.downs = nn.ModuleList(
            DownBlock(widths[level], widths[level + 1]) for level in range(depth - 1)
        )
        self.bottleneck_down = DownBlock(widths[-1], widths[-1] * 2)
        feature_widths = [*widths, widths[-1] * 2]
        active_indices = set(range(len(feature_widths) - operator_levels, len(feature_widths)))
        self.operator_blocks = nn.ModuleDict(
            {
                str(index): ConditionedLocalSpectralBlock(
                    channels,
                    condition_dim=3,
                    modes=max(4, modes // (2 ** max(0, len(feature_widths) - 1 - index))),
                    use_film=use_film,
                )
                for index, channels in enumerate(feature_widths)
                if index in active_indices
            }
        )
        decoder_in = feature_widths[-1]
        self.ups = nn.ModuleList()
        for skip_channels in reversed(widths):
            self.ups.append(UpBlock(decoder_in, skip_channels, skip_channels))
            decoder_in = skip_channels
        self.heads = MultiTaskHeads(widths[0], task_names, predict_uncertainty)

    def _operator(
        self, index: int, features: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        key = str(index)
        return self.operator_blocks[key](features, condition) if key in self.operator_blocks else features

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if inputs.shape[1] < 6:
            raise ValueError("Expected height, occupancy, SDF, cos, sin and u_tau channels")
        condition = inputs[:, -3:].mean(dim=(-2, -1))
        skips = [self._operator(0, self.stem(inputs), condition)]
        for index, down in enumerate(self.downs, start=1):
            skips.append(self._operator(index, down(skips[-1]), condition))
        features = self._operator(len(skips), self.bottleneck_down(skips[-1]), condition)
        for up, skip in zip(self.ups, reversed(skips)):
            features = up(features, skip)
        return self.heads(features)
