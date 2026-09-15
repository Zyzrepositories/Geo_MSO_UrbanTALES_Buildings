"""PyTorch input pipeline for scale-consistent UrbanTALES patches."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset as TorchDataset

from .catalog import CaseRecord


DEFAULT_TARGETS = ("uped", "vped", "Uped", "TKEped")
INPUT_CHANNELS = ("height", "occupancy", "sdf", "wind_cos", "wind_sin", "u_tau")
VELOCITY_TARGETS = {"uped", "vped", "Uped"}
QUADRATIC_TARGETS = {"TKEped", "Tuwped"}


def scale_target(values: np.ndarray, target: str, u_tau: float) -> np.ndarray:
    """Convert a dimensional pedestrian field to friction-velocity units."""
    if target in VELOCITY_TARGETS:
        return values / u_tau
    if target in QUADRATIC_TARGETS:
        return values / (u_tau**2)
    raise KeyError(f"Unknown target scale for {target}")


def unscale_target(values: np.ndarray, target: str, u_tau: float) -> np.ndarray:
    if target in VELOCITY_TARGETS:
        return values * u_tau
    if target in QUADRATIC_TARGETS:
        return values * (u_tau**2)
    raise KeyError(f"Unknown target scale for {target}")


def ablate_model_input_channels(
    inputs: torch.Tensor, channel_names: Sequence[str] | None
) -> torch.Tensor:
    """Zero selected model channels without altering inputs used for region metrics."""
    names = tuple(channel_names or ())
    if not names:
        return inputs
    unknown = sorted(set(names) - set(INPUT_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown model input channels for ablation: {unknown}")
    if inputs.ndim < 3 or inputs.shape[-3] != len(INPUT_CHANNELS):
        raise ValueError(
            f"Expected channel dimension {-3} to have size {len(INPUT_CHANNELS)}, "
            f"found shape {tuple(inputs.shape)}"
        )
    result = inputs.clone()
    indices = [INPUT_CHANNELS.index(name) for name in names]
    result[..., indices, :, :] = 0.0
    return result


def periodic_truncated_sdf_patch(
    occupied: np.ndarray,
    y0: int,
    x0: int,
    size: int,
    dx_m: float,
    truncation_m: float,
) -> np.ndarray:
    """Return an exact-within-truncation periodic SDF for one patch.

    Computing an EDT over a 3x3 tiling of a complete realistic case can consume
    hundreds of MB per data-loader worker.  A halo at least as wide as the SDF
    truncation gives the same clipped values while bounding memory by patch size.
    Positive values denote air and negative values denote buildings.
    """
    halo = max(1, int(math.ceil(truncation_m / dx_m)))
    extended = _periodic_patch(occupied, y0 - halo, x0 - halo, size + 2 * halo)
    outside = distance_transform_edt(~extended, sampling=(dx_m, dx_m))
    inside = distance_transform_edt(extended, sampling=(dx_m, dx_m))
    signed = outside - inside
    return np.clip(
        signed[halo : halo + size, halo : halo + size],
        -truncation_m,
        truncation_m,
    ).astype(np.float32, copy=False)


def _read_masked(variable) -> tuple[np.ndarray, np.ndarray]:
    values = np.ma.asarray(variable[:]).squeeze()
    raw = np.asarray(values.filled(np.nan), dtype=np.float32)
    valid = ~np.ma.getmaskarray(values) & np.isfinite(raw)
    return raw, valid


def load_geometry(root: Path, case: CaseRecord) -> dict[str, np.ndarray]:
    """Load only geometry required for deployment-time inference."""
    topo = np.atleast_2d(np.loadtxt(root / case.topo, dtype=np.float32))
    if topo.ndim != 2 or not np.isfinite(topo).all():
        raise ValueError(f"{case.case_id}: invalid topography")
    return {"topo": topo, "occupied": topo > 0}


def load_case(root: Path, case: CaseRecord, targets: Sequence[str]) -> dict[str, np.ndarray]:
    geometry = load_geometry(root, case)
    topo = geometry["topo"]
    fields: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    with Dataset(root / case.ped_nc, "r") as dataset:
        for target in targets:
            if target not in dataset.variables:
                raise KeyError(f"{case.case_id}: target {target} missing")
            fields[target], masks[target] = _read_masked(dataset.variables[target])
    shapes = {topo.shape, *(value.shape for value in fields.values())}
    if len(shapes) != 1:
        raise ValueError(f"{case.case_id}: inconsistent shapes {sorted(shapes)}")
    return {
        **geometry,
        **{f"field:{name}": fields[name] for name in targets},
        **{f"mask:{name}": masks[name] for name in targets},
    }


def _periodic_patch(array: np.ndarray, y0: int, x0: int, size: int) -> np.ndarray:
    rows = (np.arange(size) + y0) % array.shape[-2]
    cols = (np.arange(size) + x0) % array.shape[-1]
    return array[np.ix_(rows, cols)]


def _resize(array: np.ndarray, size: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    kwargs = {"size": (size, size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0, 0]


def _resize_masked(
    values: np.ndarray, valid: np.ndarray, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted = _resize(np.where(valid, values, 0.0), size, "bilinear")
    coverage = _resize(valid.astype(np.float32), size, "bilinear")
    output = weighted / coverage.clamp_min(1e-6)
    output_valid = coverage >= 0.999
    return output, output_valid


def build_input_patch(
    case: CaseRecord,
    geometry: dict[str, np.ndarray],
    *,
    y0: int,
    x0: int,
    patch_size_m: float,
    output_pixels: int,
    height_scale_m: float,
    sdf_scale_m: float,
    u_tau_reference_m_s: float,
) -> torch.Tensor:
    """Build the six deployment input channels without opening target NetCDF."""
    raw_pixels = max(8, int(round(patch_size_m / case.dx_m)))
    topo = _periodic_patch(geometry["topo"], y0, x0, raw_pixels)
    occupied = _periodic_patch(geometry["occupied"], y0, x0, raw_pixels)
    sdf = periodic_truncated_sdf_patch(
        geometry["occupied"], y0, x0, raw_pixels, case.dx_m, sdf_scale_m
    )
    height = _resize(topo / height_scale_m, output_pixels, "bilinear")
    occupancy = _resize(occupied.astype(np.float32), output_pixels, "nearest")
    sdf_tensor = _resize(
        np.clip(sdf / sdf_scale_m, -1.0, 1.0), output_pixels, "bilinear"
    )
    constants = [
        torch.full_like(height, case.wind_cos),
        torch.full_like(height, case.wind_sin),
        torch.full_like(height, case.u_tau_m_s / u_tau_reference_m_s),
    ]
    return torch.stack([height, occupancy, sdf_tensor, *constants])


class UrbanTalesPatchDataset(TorchDataset):
    """Sample fixed-physical-size periodic patches and resample to a model grid."""

    input_channels = INPUT_CHANNELS

    def __init__(
        self,
        root: str | Path,
        cases: Sequence[CaseRecord],
        *,
        targets: Sequence[str] = DEFAULT_TARGETS,
        patch_size_m: float = 256.0,
        output_pixels: int = 256,
        patches_per_case: int = 4,
        seed: int = 20260904,
        random_patches: bool = True,
        height_scale_m: float = 50.0,
        sdf_scale_m: float = 64.0,
        u_tau_reference_m_s: float = 0.21,
        max_cache_cases: int = 2,
    ):
        self.root = Path(root).resolve()
        self.cases = list(cases)
        self.targets = tuple(targets)
        self.patch_size_m = float(patch_size_m)
        self.output_pixels = int(output_pixels)
        self.patches_per_case = int(patches_per_case)
        self.seed = int(seed)
        self.random_patches = bool(random_patches)
        self.height_scale_m = float(height_scale_m)
        self.sdf_scale_m = float(sdf_scale_m)
        self.u_tau_reference_m_s = float(u_tau_reference_m_s)
        self.max_cache_cases = int(max_cache_cases)
        self.epoch = 0
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        if not self.cases:
            raise ValueError("Dataset requires at least one case")
        if self.output_pixels <= 0 or self.patches_per_case <= 0:
            raise ValueError("output_pixels and patches_per_case must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cases) * self.patches_per_case

    def _load(self, case: CaseRecord) -> dict[str, np.ndarray]:
        if case.case_id in self._cache:
            self._cache.move_to_end(case.case_id)
            return self._cache[case.case_id]
        loaded = load_case(self.root, case, self.targets)
        self._cache[case.case_id] = loaded
        while len(self._cache) > self.max_cache_cases:
            self._cache.popitem(last=False)
        return loaded

    def _build_sample(
        self,
        case: CaseRecord,
        loaded: dict[str, np.ndarray],
        *,
        y0: int,
        x0: int,
        patch_index: int,
    ) -> dict[str, object]:
        raw_pixels = max(8, int(round(self.patch_size_m / case.dx_m)))
        inputs = build_input_patch(
            case,
            loaded,
            y0=y0,
            x0=x0,
            patch_size_m=self.patch_size_m,
            output_pixels=self.output_pixels,
            height_scale_m=self.height_scale_m,
            sdf_scale_m=self.sdf_scale_m,
            u_tau_reference_m_s=self.u_tau_reference_m_s,
        )

        target_tensors = []
        mask_tensors = []
        for target in self.targets:
            values = scale_target(loaded[f"field:{target}"], target, case.u_tau_m_s)
            valid = loaded[f"mask:{target}"]
            values_patch = _periodic_patch(values, y0, x0, raw_pixels)
            valid_patch = _periodic_patch(valid, y0, x0, raw_pixels)
            resized, resized_valid = _resize_masked(
                values_patch, valid_patch, self.output_pixels
            )
            target_tensors.append(resized)
            mask_tensors.append(resized_valid)

        return {
            "input": inputs,
            "target": torch.stack(target_tensors),
            "mask": torch.stack(mask_tensors),
            "case_id": case.case_id,
            "u_tau_m_s": torch.tensor(case.u_tau_m_s, dtype=torch.float32),
            "dx_m": torch.tensor(case.dx_m, dtype=torch.float32),
            "raw_patch_pixels": raw_pixels,
            "patch_index": patch_index,
            "patch_origin_yx": (y0, x0),
        }

    def sample_at(
        self, case_index: int, y0: int, x0: int, *, patch_index: int = 0
    ) -> dict[str, object]:
        """Build a patch at an explicit native-grid origin for tiled inference."""
        case = self.cases[case_index]
        return self._build_sample(
            case,
            self._load(case),
            y0=int(y0),
            x0=int(x0),
            patch_index=int(patch_index),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        case_index = index // self.patches_per_case
        patch_index = index % self.patches_per_case
        case = self.cases[case_index]
        loaded = self._load(case)
        raw_pixels = max(8, int(round(self.patch_size_m / case.dx_m)))
        ny, nx = loaded["topo"].shape
        if self.random_patches:
            sequence = np.random.SeedSequence([self.seed, self.epoch, case_index, patch_index])
            rng = np.random.default_rng(sequence)
            y0 = int(rng.integers(0, ny))
            x0 = int(rng.integers(0, nx))
        elif self.patches_per_case == 1:
            y0 = (ny - raw_pixels) // 2
            x0 = (nx - raw_pixels) // 2
        else:
            # Fixed low-discrepancy patch centres give deterministic spatial
            # coverage. Reusing the geometric centre here would silently make
            # every evaluation patch for a case identical.
            y_fraction = (patch_index + 0.5) / self.patches_per_case
            x_fraction = (
                patch_index * ((5.0**0.5 - 1.0) / 2.0)
                + 0.5 / self.patches_per_case
            ) % 1.0
            y0 = int(round(y_fraction * ny)) - raw_pixels // 2
            x0 = int(round(x_fraction * nx)) - raw_pixels // 2

        return self._build_sample(
            case,
            loaded,
            y0=y0,
            x0=x0,
            patch_index=patch_index,
        )
