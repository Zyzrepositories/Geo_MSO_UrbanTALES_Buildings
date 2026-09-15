#!/usr/bin/env python
"""Dependency-light real-data smoke test before any model training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402


TARGET_POWERS = {"uped": 1, "vped": 1, "Uped": 1, "TKEped": 2}


def inspect_case(root: Path, case) -> dict:
    topo = np.atleast_2d(np.loadtxt(root / case.topo, dtype=np.float32))
    result = {
        "case_id": case.case_id,
        "metadata_name": case.metadata_name,
        "family": case.family,
        "shape": list(topo.shape),
        "dx_m": case.dx_m,
        "u_tau_m_s": case.u_tau_m_s,
        "wind_label": case.wind_label,
        "wind_angle_deg": case.wind_angle_deg,
        "occupied_fraction": float(np.mean(topo > 0)),
        "targets": {},
    }
    with Dataset(root / case.ped_nc, "r") as dataset:
        for name, power in TARGET_POWERS.items():
            values = np.ma.asarray(dataset.variables[name][:]).squeeze()
            raw = np.asarray(values.filled(np.nan), dtype=np.float32)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(raw)
            if raw.shape != topo.shape:
                raise AssertionError(f"{case.case_id}/{name}: {raw.shape} != {topo.shape}")
            normalized = raw[valid] / (case.u_tau_m_s**power)
            restored = normalized * (case.u_tau_m_s**power)
            if not np.allclose(restored, raw[valid], rtol=2e-6, atol=2e-7):
                raise AssertionError(f"{case.case_id}/{name}: scale round trip failed")
            result["targets"][name] = {
                "valid_fraction": float(np.mean(valid)),
                "raw_min": float(np.min(raw[valid])),
                "raw_max": float(np.max(raw[valid])),
                "normalized_mean": float(np.mean(normalized)),
                "normalized_std": float(np.std(normalized)),
            }
    if case.wind_label.casefold() == "flux":
        expected = math.degrees(math.atan2(-case.dpdy, -case.dpdx)) % 360.0
        if not math.isclose(expected, case.wind_angle_deg, abs_tol=1e-9):
            raise AssertionError(f"{case.case_id}: FLUX direction mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/data_smoke/summary.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    catalog = UrbanTalesCatalog(root)
    # One ordinary idealized case, one ordinary realistic case, and one FLUX case.
    chosen = [
        next(case for case in catalog.cases if case.family == "idealized" and not case.is_val_resolution),
        next(
            case
            for case in catalog.cases
            if case.family == "realistic" and case.wind_label.casefold() != "flux"
        ),
        next(case for case in catalog.cases if case.wind_label.casefold() == "flux"),
    ]
    summary = {
        "status": "ok",
        "purpose": "real_file_io_and_scale_round_trip_only_no_training",
        "cases": [inspect_case(root, case) for case in chosen],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

