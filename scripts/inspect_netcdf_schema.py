#!/usr/bin/env python3
"""Print dimensions, variables, and attributes for UrbanTALES NetCDF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_file(path: Path, include_stats: bool) -> dict:
    result: dict = {"path": str(path), "dimensions": {}, "attributes": {}, "variables": {}}
    with Dataset(path, "r") as dataset:
        result["data_model"] = dataset.data_model
        result["disk_format"] = dataset.disk_format
        result["attributes"] = {
            name: jsonable(dataset.getncattr(name)) for name in dataset.ncattrs()
        }
        result["dimensions"] = {
            name: {"size": len(dim), "unlimited": dim.isunlimited()}
            for name, dim in dataset.dimensions.items()
        }
        for name, variable in dataset.variables.items():
            item = {
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "attributes": {
                    attr: jsonable(variable.getncattr(attr)) for attr in variable.ncattrs()
                },
                "chunking": jsonable(variable.chunking()),
                "filters": jsonable(variable.filters()),
            }
            if include_stats:
                values = np.ma.asarray(variable[:])
                compressed = values.compressed()
                finite = compressed[np.isfinite(compressed)] if compressed.size else compressed
                item["statistics"] = {
                    "elements": int(values.size),
                    "masked": int(np.ma.count_masked(values)),
                    "nonfinite_unmasked": int(compressed.size - finite.size),
                    "min": float(np.min(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                    "mean": float(np.mean(finite)) if finite.size else None,
                }
            result["variables"][name] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--stats", action="store_true", help="Read arrays and calculate basic statistics.")
    args = parser.parse_args()
    print(json.dumps([inspect_file(path, args.stats) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
