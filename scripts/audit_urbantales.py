#!/usr/bin/env python3
"""Audit a local UrbanTALES download without modifying source data.

The script inventories both dataset families, validates case/file integrity,
summarizes metadata and PALM configuration files, scans the 2-D pedestrian
NetCDF fields for missing/non-finite values, inspects time-series NetCDF files,
checks vertical profiles, and creates a Markdown/JSON report plus sample plots.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "tmp" / "matplotlib").resolve()))

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from PIL import Image


DATASET_DIRS = {
    "idealized": "Idealized Building Blocks",
    "realistic": "Realistic Urban Neighbourhoods",
}
CASE_ROLES = ("p3d", "topo", "ped_nc", "ts_nc", "profile_csv", "preview_png")
P3D_KEYS = (
    "nx",
    "ny",
    "nz",
    "dx",
    "dy",
    "dz",
    "dz_stretch_level",
    "dz_stretch_factor",
    "omega",
    "dp_external",
    "dpdxy",
    "bc_uv_t",
    "roughness_length",
    "momentum_advec",
    "passive_scalar",
    "scalar_advec",
    "bc_s_t",
    "bc_s_b",
    "s_surface",
    "surface_scalarflux",
)
NONNEGATIVE_PED_VARIABLES = {"Uped", "TKEped"}


def python_value(value: Any) -> Any:
    """Convert numpy/netCDF objects into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return [python_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def counter_dict(counter: collections.Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: natural_sort_key(str(item[0]))))


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "未提供"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def classify_case_file(path: Path, case_name: str) -> str:
    name = path.name
    if name == f"{case_name}_p3d":
        return "p3d"
    if name == f"{case_name}_topo":
        return "topo"
    if name == f"{case_name}_ped.nc":
        return "ped_nc"
    if name == f"{case_name}_ts.nc":
        return "ts_nc"
    if name == f"profile-{case_name}.csv":
        return "profile_csv"
    if name == f"{case_name}.png":
        return "preview_png"
    return "other"


def geometry_key(case_name: str) -> str:
    return re.sub(r"_d\d+$", "", case_name, flags=re.IGNORECASE)


def site_key(case_name: str) -> str:
    """Group direction variants and the five Val-* resolution counterparts."""
    return re.sub(r"^Val-", "", geometry_key(case_name), flags=re.IGNORECASE)


def parse_p3d_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?i)\b{re.escape(key)}\s*=\s*([^!\r\n]+)", text)
    if not match:
        return None
    raw = match.group(1).strip().rstrip(",").strip()
    if key != "dpdxy":
        raw = raw.split(",", 1)[0].strip()
    return raw


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def finite_array(variable) -> tuple[np.ndarray, np.ndarray]:
    values = np.ma.asarray(variable[:])
    raw = np.asarray(values.data)
    mask = np.ma.getmaskarray(values) | ~np.isfinite(raw)
    return raw, mask


def new_numeric_accumulator() -> dict[str, Any]:
    return {
        "elements": 0,
        "valid": 0,
        "masked_or_nonfinite": 0,
        "min": None,
        "max": None,
        "sum": 0.0,
        "sum_squares": 0.0,
        "negative": 0,
        "zero": 0,
        "cases": 0,
        "case_missing_fraction_min": None,
        "case_missing_fraction_max": None,
    }


def update_numeric_accumulator(acc: dict[str, Any], raw: np.ndarray, mask: np.ndarray) -> None:
    valid = np.asarray(raw[~mask], dtype=np.float64)
    acc["elements"] += int(raw.size)
    acc["valid"] += int(valid.size)
    acc["masked_or_nonfinite"] += int(mask.sum())
    acc["cases"] += 1
    missing_fraction = float(mask.mean()) if raw.size else 0.0
    for key, candidate in (
        ("case_missing_fraction_min", missing_fraction),
        ("case_missing_fraction_max", missing_fraction),
    ):
        if acc[key] is None:
            acc[key] = candidate
        elif key.endswith("min"):
            acc[key] = min(acc[key], candidate)
        else:
            acc[key] = max(acc[key], candidate)
    if not valid.size:
        return
    vmin, vmax = float(valid.min()), float(valid.max())
    acc["min"] = vmin if acc["min"] is None else min(acc["min"], vmin)
    acc["max"] = vmax if acc["max"] is None else max(acc["max"], vmax)
    acc["sum"] += float(valid.sum(dtype=np.float64))
    acc["sum_squares"] += float(np.square(valid).sum(dtype=np.float64))
    acc["negative"] += int((valid < 0).sum())
    acc["zero"] += int((valid == 0).sum())


def finalize_numeric_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    result = dict(acc)
    valid = result["valid"]
    if valid:
        mean = result["sum"] / valid
        variance = max(result["sum_squares"] / valid - mean * mean, 0.0)
        result["mean"] = mean
        result["std"] = math.sqrt(variance)
    else:
        result["mean"] = None
        result["std"] = None
    result["missing_fraction"] = (
        result["masked_or_nonfinite"] / result["elements"] if result["elements"] else None
    )
    result.pop("sum", None)
    result.pop("sum_squares", None)
    return result


def dataframe_missing_summary(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = frame.replace(r"^\s*(?:n/?a|nan)?\s*$", np.nan, regex=True)
    missing = normalized.isna().sum()
    return {
        "total_cells": int(frame.size),
        "missing_cells": int(missing.sum()),
        "missing_by_column": {str(key): int(value) for key, value in missing.items() if value},
    }


def inspect_metadata(root: Path, cases_by_name: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata_path = root / "metadata.csv"
    flow_path = root / "UrbanTALES-flowdata.csv"
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    flow = pd.read_csv(flow_path)
    if str(flow.columns[0]).startswith("Unnamed"):
        flow = flow.rename(columns={flow.columns[0]: "row_id"})

    metadata_names = set(metadata["NameE"].astype(str))
    case_names = set(cases_by_name)
    metadata["dataset_family"] = metadata["SAR"].map(
        lambda value: "realistic" if str(value).strip().upper() == "R" else "idealized"
    )
    metadata["geometry_key"] = metadata["NameE"].map(geometry_key)
    metadata["site_key"] = metadata["NameE"].map(site_key)

    family_summary: dict[str, Any] = {}
    for family, group in metadata.groupby("dataset_family", sort=True):
        def nonmissing_unique(column: str) -> list[str]:
            values = {
                str(item).strip()
                for item in group[column]
                if str(item).strip().lower() not in {"", "n/a", "na", "nan"}
            }
            return sorted(values, key=natural_sort_key)

        wd_counts = group["WD"].astype(str).value_counts().to_dict()
        dx_counts = group["dxdy"].astype(str).value_counts().to_dict()
        utau = pd.to_numeric(group["$u_{\\tau}$"], errors="coerce")
        domain_pairs = []
        for value in group["x-y"].astype(str):
            match = re.fullmatch(r"\s*([0-9.]+)m\s*x\s*([0-9.]+)m\s*", value)
            if match:
                domain_pairs.append((float(match.group(1)), float(match.group(2))))
        family_summary[family] = {
            "cases": int(len(group)),
            "geometry_keys": int(group["geometry_key"].nunique()),
            "site_keys": int(group["site_key"].nunique()),
            "cities": nonmissing_unique("City"),
            "city_count": len(nonmissing_unique("City")),
            "states": nonmissing_unique("State"),
            "countries": nonmissing_unique("Country"),
            "wind_direction_counts": counter_dict(collections.Counter({str(k): int(v) for k, v in wd_counts.items()})),
            "horizontal_resolution_m_counts": counter_dict(collections.Counter({str(k): int(v) for k, v in dx_counts.items()})),
            "friction_velocity_m_s": {
                "available": int(utau.notna().sum()),
                "unique": int(utau.nunique(dropna=True)),
                "min": float(utau.min()) if utau.notna().any() else None,
                "max": float(utau.max()) if utau.notna().any() else None,
                "mean": float(utau.mean()) if utau.notna().any() else None,
            },
            "configuration_counts": counter_dict(collections.Counter(group["Config"].astype(str))),
            "domain_size_counts": counter_dict(collections.Counter(group["x-y"].astype(str))),
            "domain_extent_m": {
                "first_axis_min": min(item[0] for item in domain_pairs) if domain_pairs else None,
                "first_axis_max": max(item[0] for item in domain_pairs) if domain_pairs else None,
                "second_axis_min": min(item[1] for item in domain_pairs) if domain_pairs else None,
                "second_axis_max": max(item[1] for item in domain_pairs) if domain_pairs else None,
            },
        }

    flow_names = set(flow["NameE"].astype(str)) if "NameE" in flow else set()
    summary = {
        "metadata_rows": int(len(metadata)),
        "metadata_columns": list(metadata.columns[:-3]),
        "metadata_missing": dataframe_missing_summary(metadata.iloc[:, :-3]),
        "metadata_duplicate_case_names": sorted(
            metadata.loc[metadata["NameE"].duplicated(keep=False), "NameE"].unique().tolist(),
            key=natural_sort_key,
        ),
        "flow_rows": int(len(flow)),
        "flow_columns": list(flow.columns),
        "flow_missing": dataframe_missing_summary(flow),
        "flow_duplicate_case_names": sorted(
            flow.loc[flow["NameE"].duplicated(keep=False), "NameE"].astype(str).unique().tolist(),
            key=natural_sort_key,
        ) if "NameE" in flow else [],
        "cases_missing_from_metadata": sorted(case_names - metadata_names, key=natural_sort_key),
        "metadata_rows_without_case_dir": sorted(metadata_names - case_names, key=natural_sort_key),
        "cases_missing_from_flow_table": sorted(case_names - flow_names, key=natural_sort_key),
        "flow_rows_without_case_dir": sorted(flow_names - case_names, key=natural_sort_key),
        "families": family_summary,
    }
    return metadata, flow, summary


def inspect_p3d(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_family: dict[str, dict[str, collections.Counter]] = {
        family: {key: collections.Counter() for key in P3D_KEYS} for family in DATASET_DIRS
    }
    parsed_by_case: dict[str, dict[str, str | None]] = {}
    for record in case_records:
        path = record["files"].get("p3d")
        if path is None:
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        parsed = {key: parse_p3d_value(text, key) for key in P3D_KEYS}
        parsed_by_case[record["case_name"]] = parsed
        for key, value in parsed.items():
            values_by_family[record["dataset"]][key][str(value)] += 1

    summary = {
        family: {key: counter_dict(counter) for key, counter in values.items()}
        for family, values in values_by_family.items()
    }
    return {"summary": summary, "by_case": parsed_by_case}


def inspect_topography(case_records: list[dict[str, Any]], p3d_by_case: dict[str, dict[str, str | None]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    family_acc: dict[str, dict[str, Any]] = {}
    by_case: dict[str, dict[str, Any]] = {}
    for family in DATASET_DIRS:
        family_acc[family] = {
            "shape_counts": collections.Counter(),
            "height_min": None,
            "height_max": None,
            "nan_or_inf": 0,
            "cells": 0,
            "built_cells": 0,
            "unique_height_count_min": None,
            "unique_height_count_max": None,
            "physical_domain_first_axis_min": None,
            "physical_domain_first_axis_max": None,
            "physical_domain_second_axis_min": None,
            "physical_domain_second_axis_max": None,
            "p3d_shape_mismatch_cases": [],
        }

    for index, record in enumerate(case_records, start=1):
        path = record["files"].get("topo")
        if path is None:
            continue
        topo = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
        finite = np.isfinite(topo)
        valid = topo[finite]
        unique_heights = np.unique(valid)
        positive = valid[valid > 0]
        case_info = {
            "shape": list(topo.shape),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "nan_or_inf": int((~finite).sum()),
            "unique_height_count": int(unique_heights.size),
            "positive_height_count": int(np.unique(positive).size),
            "positive_height_mean": float(positive.mean()) if positive.size else None,
            "positive_height_std": float(positive.std()) if positive.size else None,
            "positive_height_min": float(positive.min()) if positive.size else None,
            "positive_height_max": float(positive.max()) if positive.size else None,
            "occupied_fraction": float(positive.size / valid.size) if valid.size else None,
        }
        by_case[record["case_name"]] = case_info
        acc = family_acc[record["dataset"]]
        acc["shape_counts"]["x".join(map(str, topo.shape))] += 1
        acc["height_min"] = case_info["min"] if acc["height_min"] is None else min(acc["height_min"], case_info["min"])
        acc["height_max"] = case_info["max"] if acc["height_max"] is None else max(acc["height_max"], case_info["max"])
        acc["nan_or_inf"] += case_info["nan_or_inf"]
        acc["cells"] += int(topo.size)
        acc["built_cells"] += int(((topo > 0) & finite).sum())
        for key, candidate in (
            ("unique_height_count_min", case_info["unique_height_count"]),
            ("unique_height_count_max", case_info["unique_height_count"]),
        ):
            if acc[key] is None:
                acc[key] = candidate
            elif key.endswith("min"):
                acc[key] = min(acc[key], candidate)
            else:
                acc[key] = max(acc[key], candidate)

        parsed = p3d_by_case.get(record["case_name"], {})
        nx, ny = parse_number(parsed.get("nx")), parse_number(parsed.get("ny"))
        dx, dy = parse_number(parsed.get("dx")), parse_number(parsed.get("dy"))
        if dx is not None and dy is not None:
            first_axis = float(topo.shape[1] * dx)
            second_axis = float(topo.shape[0] * dy)
            case_info["physical_domain_m"] = [first_axis, second_axis]
            for key, candidate in (
                ("physical_domain_first_axis_min", first_axis),
                ("physical_domain_first_axis_max", first_axis),
                ("physical_domain_second_axis_min", second_axis),
                ("physical_domain_second_axis_max", second_axis),
            ):
                if acc[key] is None:
                    acc[key] = candidate
                elif key.endswith("min"):
                    acc[key] = min(acc[key], candidate)
                else:
                    acc[key] = max(acc[key], candidate)
        if nx is not None and ny is not None:
            expected = (int(ny) + 1, int(nx) + 1)
            if topo.shape != expected:
                acc["p3d_shape_mismatch_cases"].append(
                    {"case": record["case_name"], "topo": list(topo.shape), "expected_ny1_nx1": list(expected)}
                )
        if index % 50 == 0:
            print(f"[topography] {index}/{len(case_records)}", flush=True)

    summary: dict[str, Any] = {}
    for family, acc in family_acc.items():
        item = dict(acc)
        item["shape_counts"] = counter_dict(item["shape_counts"])
        item["built_cell_fraction"] = item["built_cells"] / item["cells"] if item["cells"] else None
        summary[family] = item
    return summary, by_case


def normalize_numeric_text(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text.upper()
    return str(int(number)) if number.is_integer() else f"{number:g}"


def parse_domain_size(value: Any) -> tuple[float, float] | None:
    match = re.fullmatch(r"\s*([0-9.]+)m\s*x\s*([0-9.]+)m\s*", str(value))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def reconcile_case_names(
    metadata: pd.DataFrame,
    case_records: list[dict[str, Any]],
    topo_by_case: dict[str, dict[str, Any]],
    p3d_by_case: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Reconcile renamed directory cases to metadata rows using physical signatures.

    All non-exact names in this download are realistic-variable cases. Candidate
    matches are constrained by family, wind direction, dx, physical domain size,
    and uniform/variable geometry. A one-to-one greedy assignment then minimizes
    differences in plan density and height statistics. Inferred mappings are
    recorded as such and never presented as file-name equality.
    """
    rows = {str(row["NameE"]): row for _, row in metadata.iterrows()}
    exact_names = set(rows) & {record["case_name"] for record in case_records}
    used_metadata = set(exact_names)
    mappings: list[dict[str, Any]] = []

    for record in case_records:
        case_name = record["case_name"]
        if case_name in exact_names:
            record["metadata_name"] = case_name
            mappings.append(
                {"case_dir": case_name, "metadata_name": case_name, "status": "exact_name", "score": 0.0}
            )

    casefold_rows: dict[str, list[str]] = collections.defaultdict(list)
    for name in rows:
        if name not in used_metadata:
            casefold_rows[name.casefold()].append(name)
    for record in case_records:
        case_name = record["case_name"]
        if "metadata_name" in record:
            continue
        candidates = casefold_rows.get(case_name.casefold(), [])
        if len(candidates) == 1:
            metadata_name = candidates[0]
            record["metadata_name"] = metadata_name
            used_metadata.add(metadata_name)
            mappings.append(
                {
                    "case_dir": case_name,
                    "metadata_name": metadata_name,
                    "status": "casefold_name_alias",
                    "score": 0.0,
                }
            )

    unmatched_records = [record for record in case_records if "metadata_name" not in record]
    unused_rows = [row for name, row in rows.items() if name not in used_metadata]
    candidate_edges: list[tuple[float, str, str, dict[str, Any]]] = []
    candidate_counts: collections.Counter = collections.Counter()

    for record in unmatched_records:
        case_name = record["case_name"]
        topo = topo_by_case[case_name]
        p3d = p3d_by_case.get(case_name, {})
        dx, dy = parse_number(p3d.get("dx")), parse_number(p3d.get("dy"))
        shape = topo["shape"]
        domain = (shape[1] * dx, shape[0] * dy) if dx is not None and dy is not None else None
        direction_match = re.search(r"_d(\d+)$", case_name, flags=re.IGNORECASE)
        wd = normalize_numeric_text(direction_match.group(1)) if direction_match else "FLUX"
        geometry_type = "uniform" if topo["positive_height_count"] <= 1 else "variable"

        for row in unused_rows:
            if row["dataset_family"] != record["dataset"]:
                continue
            if normalize_numeric_text(row["WD"]) != wd:
                continue
            row_dx = parse_number(str(row["dxdy"]))
            row_domain = parse_domain_size(row["x-y"])
            row_geometry_type = "variable" if str(row["Config"]).endswith("variable") else "uniform"
            if dx is None or row_dx is None or not math.isclose(dx, row_dx, abs_tol=1e-9):
                continue
            if domain is None or row_domain is None or not (
                math.isclose(domain[0], row_domain[0], abs_tol=1e-6)
                and math.isclose(domain[1], row_domain[1], abs_tol=1e-6)
            ):
                continue
            if geometry_type != row_geometry_type:
                continue

            def numeric(column: str) -> float | None:
                return parse_number(str(row[column]))

            score = 0.0
            components = [
                (topo["occupied_fraction"], numeric("lp"), 1.0),
                (topo["positive_height_mean"], numeric("hmean"), 1 / 200),
                (topo["positive_height_std"], numeric("hstd"), 1 / 200),
                (topo["positive_height_max"], numeric("hmax"), 1 / 500),
                (topo["positive_height_min"], numeric("hmin"), 1 / 500),
            ]
            for left, right, weight in components:
                if left is not None and right is not None:
                    score += abs(left - right) * weight
            candidate_counts[case_name] += 1
            candidate_edges.append(
                (
                    score,
                    case_name,
                    str(row["NameE"]),
                    {
                        "case_dir": case_name,
                        "metadata_name": str(row["NameE"]),
                        "status": "inferred_physical_signature",
                        "score": score,
                        "candidate_count": 0,
                        "signature": {
                            "wind_direction": wd,
                            "dx_m": dx,
                            "domain_m": list(domain),
                            "geometry_type": geometry_type,
                        },
                    },
                )
            )

    assigned_cases: set[str] = set()
    for score, case_name, metadata_name, item in sorted(candidate_edges, key=lambda edge: (edge[0], natural_sort_key(edge[1]), natural_sort_key(edge[2]))):
        if case_name in assigned_cases or metadata_name in used_metadata:
            continue
        item["candidate_count"] = int(candidate_counts[case_name])
        next(record for record in unmatched_records if record["case_name"] == case_name)["metadata_name"] = metadata_name
        mappings.append(item)
        assigned_cases.add(case_name)
        used_metadata.add(metadata_name)

    unresolved_cases = []
    for record in unmatched_records:
        if "metadata_name" not in record:
            unresolved_cases.append(record["case_name"])
            mappings.append(
                {
                    "case_dir": record["case_name"],
                    "metadata_name": None,
                    "status": "unresolved",
                    "score": None,
                    "candidate_count": int(candidate_counts[record["case_name"]]),
                }
            )

    validation_pairs = []
    records_by_name = {record["case_name"]: record for record in case_records}
    for record in case_records:
        if not record["case_name"].lower().startswith("val-"):
            continue
        base_name = record["case_name"][4:]
        base = records_by_name.get(base_name)
        validation_pairs.append(
            {
                "val_case": record["case_name"],
                "base_case": base_name if base else None,
                "val_shape": topo_by_case[record["case_name"]]["shape"],
                "base_shape": topo_by_case[base_name]["shape"] if base else None,
                "val_dx_m": parse_number(p3d_by_case[record["case_name"]].get("dx")),
                "base_dx_m": parse_number(p3d_by_case[base_name].get("dx")) if base else None,
            }
        )

    counts = collections.Counter(item["status"] for item in mappings)
    inferred_scores = [item["score"] for item in mappings if item["status"] == "inferred_physical_signature"]
    return {
        "status_counts": counter_dict(counts),
        "unresolved_cases": sorted(unresolved_cases, key=natural_sort_key),
        "unused_metadata_names": sorted(set(rows) - used_metadata, key=natural_sort_key),
        "inferred_score_min": min(inferred_scores) if inferred_scores else None,
        "inferred_score_max": max(inferred_scores) if inferred_scores else None,
        "validation_named_resolution_pairs": validation_pairs,
        "mappings": sorted(mappings, key=lambda item: natural_sort_key(item["case_dir"])),
    }


def variable_schema(variable) -> dict[str, Any]:
    attrs = {name: python_value(variable.getncattr(name)) for name in variable.ncattrs()}
    return {
        "dimensions": list(variable.dimensions),
        "dtype": str(variable.dtype),
        "units": attrs.get("units"),
        "long_name": attrs.get("long_name"),
        "fill_value": attrs.get("_FillValue"),
    }


def schema_signature(dataset: Dataset) -> str:
    variables = [
        (
            name,
            tuple(variable.dimensions),
            str(variable.dtype),
            python_value(getattr(variable, "units", None)),
        )
        for name, variable in dataset.variables.items()
    ]
    return json.dumps(variables, ensure_ascii=False, sort_keys=True)


def inspect_netcdf(case_records: list[dict[str, Any]], topo_by_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ped_acc = {
        family: collections.defaultdict(new_numeric_accumulator) for family in DATASET_DIRS
    }
    ped_shape_counts = {family: collections.Counter() for family in DATASET_DIRS}
    ped_dtype_counts = {family: collections.defaultdict(collections.Counter) for family in DATASET_DIRS}
    ped_coordinate_summaries = {family: {"x": [], "y": []} for family in DATASET_DIRS}
    ped_schema_counts = {family: collections.Counter() for family in DATASET_DIRS}
    ped_schema_examples: dict[str, dict[str, Any]] = {}
    ped_case_statistics: dict[str, dict[str, Any]] = {}
    ped_anomalies = {
        family: {
            "negative_nonnegative_variables": collections.Counter(),
            "jensen_violations": 0,
            "jensen_compared_cells": 0,
            "jensen_max_gap": None,
            "jensen_cases": [],
            "topo_shape_mismatch_cases": [],
        }
        for family in DATASET_DIRS
    }
    ts_schema_counts = {family: collections.Counter() for family in DATASET_DIRS}
    ts_schema_examples: dict[str, dict[str, Any]] = {}
    ts_data_model_counts = {family: collections.Counter() for family in DATASET_DIRS}
    ts_time_lengths = {family: [] for family in DATASET_DIRS}
    ts_time_durations = {family: [] for family in DATASET_DIRS}
    errors: list[dict[str, str]] = []

    for index, record in enumerate(case_records, start=1):
        family = record["dataset"]
        case_name = record["case_name"]
        ped_path = record["files"].get("ped_nc")
        if ped_path:
            try:
                with Dataset(ped_path, "r") as dataset:
                    signature = schema_signature(dataset)
                    ped_schema_counts[family][signature] += 1
                    if signature not in ped_schema_examples:
                        ped_schema_examples[signature] = {
                            "example": str(Path(ped_path).relative_to(Path.cwd())),
                            "data_model": dataset.data_model,
                            "variables": {name: variable_schema(var) for name, var in dataset.variables.items()},
                        }
                    dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
                    shape_key = "x".join(str(dims[name]) for name in dims)
                    ped_shape_counts[family][shape_key] += 1
                    for coord in ("x", "y"):
                        if coord in dataset.variables:
                            values = np.asarray(dataset.variables[coord][:], dtype=np.float64)
                            diffs = np.diff(values)
                            ped_coordinate_summaries[family][coord].append(
                                {
                                    "case": case_name,
                                    "size": int(values.size),
                                    "min": float(values.min()) if values.size else None,
                                    "max": float(values.max()) if values.size else None,
                                    "step_min": float(diffs.min()) if diffs.size else None,
                                    "step_max": float(diffs.max()) if diffs.size else None,
                                }
                            )
                    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    ped_case_statistics[case_name] = {}
                    for name, variable in dataset.variables.items():
                        ped_dtype_counts[family][name][str(variable.dtype)] += 1
                        if name in {"x", "y"}:
                            continue
                        raw, mask = finite_array(variable)
                        arrays[name] = (raw, mask)
                        update_numeric_accumulator(ped_acc[family][name], raw, mask)
                        valid = np.asarray(raw[~mask], dtype=np.float64)
                        ped_case_statistics[case_name][name] = {
                            "valid": int(valid.size),
                            "mean": float(valid.mean()) if valid.size else None,
                            "min": float(valid.min()) if valid.size else None,
                            "max": float(valid.max()) if valid.size else None,
                            "missing_fraction": float(mask.mean()) if raw.size else None,
                        }
                        if name in NONNEGATIVE_PED_VARIABLES:
                            negative_count = int(((raw < 0) & ~mask).sum())
                            ped_anomalies[family]["negative_nonnegative_variables"][name] += negative_count

                    if {"uped", "vped", "Uped"}.issubset(arrays):
                        u, umask = arrays["uped"]
                        v, vmask = arrays["vped"]
                        speed, smask = arrays["Uped"]
                        common = ~(umask | vmask | smask)
                        magnitude = np.sqrt(np.square(u) + np.square(v))
                        gap = magnitude - speed
                        violations = common & (gap > 1e-5)
                        violation_count = int(violations.sum())
                        ped_anomalies[family]["jensen_violations"] += violation_count
                        ped_anomalies[family]["jensen_compared_cells"] += int(common.sum())
                        if violation_count:
                            max_gap = float(gap[violations].max())
                            previous = ped_anomalies[family]["jensen_max_gap"]
                            ped_anomalies[family]["jensen_max_gap"] = max_gap if previous is None else max(previous, max_gap)
                            ped_anomalies[family]["jensen_cases"].append(
                                {"case": case_name, "violations": violation_count, "max_gap": max_gap}
                            )

                    topo_shape = topo_by_case.get(case_name, {}).get("shape")
                    field_shapes = {
                        tuple(var.shape)
                        for name, var in dataset.variables.items()
                        if name not in {"x", "y"} and len(var.shape) == 2
                    }
                    if topo_shape is not None and field_shapes and tuple(topo_shape) not in field_shapes:
                        ped_anomalies[family]["topo_shape_mismatch_cases"].append(case_name)
            except Exception as exc:  # keep auditing other cases
                errors.append({"file": str(ped_path), "error": repr(exc)})

        ts_path = record["files"].get("ts_nc")
        if ts_path:
            try:
                with Dataset(ts_path, "r") as dataset:
                    signature = schema_signature(dataset)
                    ts_schema_counts[family][signature] += 1
                    ts_data_model_counts[family][dataset.data_model] += 1
                    if signature not in ts_schema_examples:
                        ts_schema_examples[signature] = {
                            "example": str(Path(ts_path).relative_to(Path.cwd())),
                            "data_model": dataset.data_model,
                            "global_attributes": {
                                name: python_value(dataset.getncattr(name)) for name in dataset.ncattrs()
                            },
                            "variables": {name: variable_schema(var) for name, var in dataset.variables.items()},
                        }
                    if "time" in dataset.variables:
                        time = np.asarray(dataset.variables["time"][:], dtype=np.float64)
                        ts_time_lengths[family].append(int(time.size))
                        if time.size:
                            ts_time_durations[family].append(float(time[-1] - time[0]))
            except Exception as exc:
                errors.append({"file": str(ts_path), "error": repr(exc)})

        if index % 25 == 0:
            print(f"[netcdf] {index}/{len(case_records)}", flush=True)

    def compact_schema_counts(counters: dict[str, collections.Counter], examples: dict[str, dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for family, counter in counters.items():
            output[family] = []
            for signature, count in counter.most_common():
                output[family].append({"count": int(count), **examples[signature]})
        return output

    return {
        "pedestrian": {
            "shape_counts": {family: counter_dict(value) for family, value in ped_shape_counts.items()},
            "dtype_counts": {
                family: {name: counter_dict(counter) for name, counter in variables.items()}
                for family, variables in ped_dtype_counts.items()
            },
            "variable_statistics": {
                family: {name: finalize_numeric_accumulator(acc) for name, acc in variables.items()}
                for family, variables in ped_acc.items()
            },
            "coordinate_summaries": ped_coordinate_summaries,
            "schemas": compact_schema_counts(ped_schema_counts, ped_schema_examples),
            "case_statistics": ped_case_statistics,
            "anomalies": {
                family: {
                    **value,
                    "negative_nonnegative_variables": counter_dict(value["negative_nonnegative_variables"]),
                    "jensen_violation_fraction": (
                        value["jensen_violations"] / value["jensen_compared_cells"]
                        if value["jensen_compared_cells"] else None
                    ),
                }
                for family, value in ped_anomalies.items()
            },
        },
        "time_series": {
            "schemas": compact_schema_counts(ts_schema_counts, ts_schema_examples),
            "data_model_counts": {family: counter_dict(value) for family, value in ts_data_model_counts.items()},
            "time_length": {
                family: {
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "unique": len(set(values)),
                }
                for family, values in ts_time_lengths.items()
            },
            "duration_seconds": {
                family: {
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for family, values in ts_time_durations.items()
            },
        },
        "errors": errors,
    }


def compare_pedestrian_scaling(
    metadata: pd.DataFrame,
    flow: pd.DataFrame,
    case_records: list[dict[str, Any]],
    case_statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare NetCDF means with flow-table summaries and friction scaling."""
    metadata_index = metadata.set_index("NameE")
    flow_index = flow.set_index("NameE")
    rows: list[dict[str, Any]] = []
    utau_column = "$u_{\\tau}$"
    for record in case_records:
        metadata_name = record.get("metadata_name")
        stats = case_statistics.get(record["case_name"], {})
        if metadata_name not in metadata_index.index or metadata_name not in flow_index.index:
            continue
        utau = parse_number(str(metadata_index.loc[metadata_name, utau_column]))
        if utau is None:
            continue
        raw_speed = stats.get("Uped", {}).get("mean")
        raw_tke = stats.get("TKEped", {}).get("mean")
        table_speed = parse_number(str(flow_index.loc[metadata_name, "pedU"]))
        table_tke = parse_number(str(flow_index.loc[metadata_name, "pedTKE"]))
        rows.append(
            {
                "dataset": record["dataset"],
                "case_dir": record["case_name"],
                "metadata_name": metadata_name,
                "utau": utau,
                "raw_speed_mean": raw_speed,
                "table_pedU": table_speed,
                "speed_ratio_raw_over_table_utau": (
                    raw_speed / (table_speed * utau)
                    if raw_speed is not None and table_speed not in {None, 0.0} else None
                ),
                "raw_tke_mean": raw_tke,
                "table_pedTKE": table_tke,
                "tke_ratio_raw_over_table_utau2": (
                    raw_tke / (table_tke * utau * utau)
                    if raw_tke is not None and table_tke not in {None, 0.0} else None
                ),
            }
        )

    family_summary: dict[str, Any] = {}
    for family in DATASET_DIRS:
        subset = [row for row in rows if row["dataset"] == family]
        item: dict[str, Any] = {"cases": len(subset)}
        for column in ("speed_ratio_raw_over_table_utau", "tke_ratio_raw_over_table_utau2"):
            values = np.asarray([row[column] for row in subset if row[column] is not None], dtype=np.float64)
            item[column] = {
                "count": int(values.size),
                "min": float(values.min()) if values.size else None,
                "max": float(values.max()) if values.size else None,
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std()) if values.size else None,
            }
        family_summary[family] = item
    return {"summary": family_summary, "by_case": rows}


def inspect_profiles(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    shape_counts = {family: collections.Counter() for family in DATASET_DIRS}
    schema_counts = {family: collections.Counter() for family in DATASET_DIRS}
    z_signatures = {family: collections.Counter() for family in DATASET_DIRS}
    missing = {family: collections.Counter() for family in DATASET_DIRS}
    value_acc = {family: collections.defaultdict(new_numeric_accumulator) for family in DATASET_DIRS}
    negative_tke = {
        family: {"all_rows": 0, "excluding_last_row": 0, "cases": [], "interior_case_details": []}
        for family in DATASET_DIRS
    }
    errors: list[dict[str, str]] = []

    for index, record in enumerate(case_records, start=1):
        path = record["files"].get("profile_csv")
        if path is None:
            continue
        family = record["dataset"]
        try:
            frame = pd.read_csv(path)
            if str(frame.columns[0]).startswith("Unnamed"):
                frame = frame.drop(columns=frame.columns[0])
            shape_counts[family]["x".join(map(str, frame.shape))] += 1
            signature = json.dumps(list(frame.columns), ensure_ascii=False)
            schema_counts[family][signature] += 1
            for column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce")
                raw = series.to_numpy(dtype=np.float64)
                mask = ~np.isfinite(raw)
                update_numeric_accumulator(value_acc[family][column], raw, mask)
                missing[family][column] += int(mask.sum())
            if "z" in frame:
                z_values = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=np.float64)
                z_signatures[family][json.dumps(np.round(z_values, 8).tolist())] += 1
            if "TKE" in frame:
                tke = pd.to_numeric(frame["TKE"], errors="coerce").to_numpy(dtype=np.float64)
                all_negative = int(np.sum(np.isfinite(tke) & (tke < 0)))
                interior_negative = int(np.sum(np.isfinite(tke[:-1]) & (tke[:-1] < 0))) if tke.size else 0
                negative_tke[family]["all_rows"] += all_negative
                negative_tke[family]["excluding_last_row"] += interior_negative
                if all_negative:
                    negative_tke[family]["cases"].append(record["case_name"])
                if interior_negative:
                    indices = np.flatnonzero(np.isfinite(tke[:-1]) & (tke[:-1] < 0))
                    negative_tke[family]["interior_case_details"].append(
                        {
                            "case": record["case_name"],
                            "count": int(indices.size),
                            "first_index": int(indices[0]),
                            "last_index": int(indices[-1]),
                            "first_z": float(frame.iloc[int(indices[0])]["z"]),
                            "last_z": float(frame.iloc[int(indices[-1])]["z"]),
                            "min_tke": float(np.nanmin(tke[:-1])),
                        }
                    )
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})
        if index % 100 == 0:
            print(f"[profiles] {index}/{len(case_records)}", flush=True)

    schema_summary: dict[str, Any] = {}
    for family, counter in schema_counts.items():
        schema_summary[family] = [
            {"count": int(count), "columns": json.loads(signature)}
            for signature, count in counter.most_common()
        ]
    z_summary = {
        family: {
            "unique_coordinate_vectors": len(counter),
            "most_common_count": counter.most_common(1)[0][1] if counter else 0,
            "example": json.loads(counter.most_common(1)[0][0]) if counter else [],
        }
        for family, counter in z_signatures.items()
    }
    return {
        "shape_counts": {family: counter_dict(value) for family, value in shape_counts.items()},
        "schemas": schema_summary,
        "z_coordinates": z_summary,
        "missing_by_variable": {family: counter_dict(value) for family, value in missing.items()},
        "variable_statistics": {
            family: {name: finalize_numeric_accumulator(acc) for name, acc in variables.items()}
            for family, variables in value_acc.items()
        },
        "negative_tke": negative_tke,
        "errors": errors,
    }


def inspect_previews(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {family: collections.Counter() for family in DATASET_DIRS}
    errors: list[dict[str, str]] = []
    for record in case_records:
        path = record["files"].get("preview_png")
        if path is None:
            continue
        try:
            with Image.open(path) as image:
                dimensions[record["dataset"]][f"{image.width}x{image.height}"] += 1
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})
    return {
        "dimension_counts": {family: counter_dict(value) for family, value in dimensions.items()},
        "errors": errors,
    }


def plot_case(case_record: dict[str, Any], metadata_row: pd.Series, output_path: Path) -> None:
    topo = np.loadtxt(case_record["files"]["topo"], dtype=np.float64)
    dx = float(metadata_row["dxdy"])
    with Dataset(case_record["files"]["ped_nc"], "r") as dataset:
        fields = {}
        for name in ("Uped", "uped", "vped", "TKEped"):
            values = np.ma.asarray(dataset.variables[name][:])
            fields[name] = np.where(np.ma.getmaskarray(values), np.nan, np.asarray(values.data, dtype=np.float64))

    extent = [0.0, topo.shape[1] * dx, 0.0, topo.shape[0] * dx]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        (topo, "Building height", "viridis", "m", None),
        (fields["Uped"], "Mean speed Uped", "turbo", "stored value (units absent)", None),
        (fields["TKEped"], "TKEped", "magma", "stored value (units absent)", None),
        (fields["uped"], "Streamwise uped", "coolwarm", "stored value (units absent)", "symmetric"),
        (fields["vped"], "Spanwise vped", "coolwarm", "stored value (units absent)", "symmetric"),
    ]
    for axis, (values, title, cmap, units, scale) in zip(axes.flat, panels):
        kwargs: dict[str, Any] = {"origin": "lower", "extent": extent, "cmap": cmap, "aspect": "equal"}
        if scale == "symmetric":
            vmax = float(np.nanpercentile(np.abs(values), 99))
            kwargs.update(vmin=-vmax, vmax=vmax)
        image = axis.imshow(values, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("grid axis 2 × dxdy [m]")
        axis.set_ylabel("grid axis 1 × dxdy [m]")
        figure.colorbar(image, ax=axis, shrink=0.8, label=units)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Case: {case_record['case_name']}",
                f"Family: {case_record['dataset']}",
                f"WD: {metadata_row['WD']}",
                f"dxdy: {metadata_row['dxdy']} m",
                f"Raster: {topo.shape[0]} × {topo.shape[1]}",
                "NaN/masked cells are transparent.",
                "Coordinate names x/y in NetCDF are retained",
                "but physical axes are reconstructed from dxdy.",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=11,
    )
    figure.suptitle(f"UrbanTALES local sample — {case_record['case_name']}", fontsize=16)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_overview(metadata: pd.DataFrame, topo_by_case: dict[str, dict[str, Any]], output_path: Path) -> None:
    frame = metadata.copy()
    frame["hmean_numeric"] = pd.to_numeric(frame["hmean"], errors="coerce")
    frame["lp_numeric"] = pd.to_numeric(frame["lp"], errors="coerce")
    frame["cells"] = frame["NameE"].map(
        lambda name: int(np.prod(topo_by_case[str(name)]["shape"])) if str(name) in topo_by_case else np.nan
    )
    colors = {"idealized": "#3569A8", "realistic": "#D4762C"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    wd_order = sorted(frame["WD"].astype(str).unique(), key=natural_sort_key)
    x = np.arange(len(wd_order))
    width = 0.38
    for offset, family in zip((-width / 2, width / 2), DATASET_DIRS):
        counts = frame.loc[frame["dataset_family"] == family, "WD"].astype(str).value_counts()
        axes[0, 0].bar(x + offset, [counts.get(item, 0) for item in wd_order], width, label=family, color=colors[family])
    axes[0, 0].set_xticks(x, wd_order, rotation=45)
    axes[0, 0].set_title("Wind-direction cases")
    axes[0, 0].set_xlabel("WD in metadata [degree or FLUX label]")
    axes[0, 0].set_ylabel("case count")
    axes[0, 0].legend()

    for family in DATASET_DIRS:
        values = frame.loc[frame["dataset_family"] == family, "hmean_numeric"].dropna()
        axes[0, 1].hist(values, bins=20, alpha=0.65, label=family, color=colors[family])
    axes[0, 1].set_title("Mean building height")
    axes[0, 1].set_xlabel("hmean [m]")
    axes[0, 1].set_ylabel("case count")
    axes[0, 1].legend()

    for family in DATASET_DIRS:
        values = frame.loc[frame["dataset_family"] == family, "lp_numeric"].dropna()
        axes[1, 0].hist(values, bins=20, alpha=0.65, label=family, color=colors[family])
    axes[1, 0].set_title("Plan area density")
    axes[1, 0].set_xlabel("lp [-]")
    axes[1, 0].set_ylabel("case count")
    axes[1, 0].legend()

    for family in DATASET_DIRS:
        values = frame.loc[frame["dataset_family"] == family, "cells"].dropna()
        axes[1, 1].hist(values, bins=20, alpha=0.65, label=family, color=colors[family])
    axes[1, 1].set_title("2-D raster size")
    axes[1, 1].set_xlabel("grid cells per case")
    axes[1, 1].set_ylabel("case count")
    axes[1, 1].legend()

    figure.suptitle("UrbanTALES local dataset overview", fontsize=16)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_report(summary: dict[str, Any]) -> str:
    inventory = summary["inventory"]
    metadata = summary["metadata"]
    reconciliation = metadata["name_reconciliation"]
    topo = summary["topography"]
    netcdf = summary["netcdf"]
    profiles = summary["profiles"]

    dataset_rows = []
    for family in DATASET_DIRS:
        item = inventory["datasets"][family]
        meta = metadata["families"][family]
        dataset_rows.append(
            [
                family,
                item["case_directories"],
                meta["geometry_keys"],
                meta["site_keys"],
                item["files"],
                f"{item['bytes'] / (1024 ** 3):.3f}",
                meta["city_count"],
                ", ".join(f"{k}:{v}" for k, v in meta["horizontal_resolution_m_counts"].items()),
            ]
        )

    role_rows = []
    for family in DATASET_DIRS:
        for role in CASE_ROLES:
            role_rows.append(
                [
                    family,
                    role,
                    inventory["datasets"][family]["role_counts"].get(role, 0),
                    f"{inventory['datasets'][family]['role_bytes'].get(role, 0) / (1024 ** 3):.3f}",
                ]
            )

    extension_rows = [
        [extension, count, f"{inventory['extension_bytes'][extension] / (1024 ** 3):.3f}"]
        for extension, count in inventory["extension_counts"].items()
    ]

    meta_family_rows = []
    for family, item in metadata["families"].items():
        utau = item["friction_velocity_m_s"]
        meta_family_rows.append(
            [
                family,
                item["cases"],
                item["geometry_keys"],
                item["city_count"],
                ", ".join(item["countries"]) or "n/a",
                ", ".join(f"{key}:{value}" for key, value in item["wind_direction_counts"].items()),
                f"{utau['min']:.6g}–{utau['max']:.6g}" if utau["min"] is not None else "未提供",
                (
                    f"{item['domain_extent_m']['first_axis_min']:.0f}–{item['domain_extent_m']['first_axis_max']:.0f} × "
                    f"{item['domain_extent_m']['second_axis_min']:.0f}–{item['domain_extent_m']['second_axis_max']:.0f}"
                ),
            ]
        )

    boundary_rows = []
    for family in DATASET_DIRS:
        values = summary["p3d"]["summary"][family]
        boundary_rows.append(
            [
                family,
                ", ".join(values["nz"]),
                ", ".join(values["dx"]),
                ", ".join(values["dy"]),
                ", ".join(values["dz"]),
                ", ".join(values["dp_external"]),
                len(values["dpdxy"]),
                ", ".join(values["bc_uv_t"]),
                ", ".join(values["roughness_length"]),
            ]
        )

    topo_rows = []
    for family, item in topo["summary"].items():
        topo_rows.append(
            [
                family,
                len(item["shape_counts"]),
                "; ".join(f"{shape}:{count}" for shape, count in list(item["shape_counts"].items())[:12]),
                (
                    f"{item['physical_domain_first_axis_min']:.0f}–{item['physical_domain_first_axis_max']:.0f} × "
                    f"{item['physical_domain_second_axis_min']:.0f}–{item['physical_domain_second_axis_max']:.0f}"
                ),
                f"{item['height_min']:.3g}–{item['height_max']:.3g}",
                f"{item['built_cell_fraction']:.3%}",
                item["nan_or_inf"],
                len(item["p3d_shape_mismatch_cases"]),
            ]
        )

    ped_rows = []
    for family, variables in netcdf["pedestrian"]["variable_statistics"].items():
        for name, item in variables.items():
            ped_rows.append(
                [
                    family,
                    name,
                    item["cases"],
                    f"{item['min']:.6g}" if item["min"] is not None else "n/a",
                    f"{item['max']:.6g}" if item["max"] is not None else "n/a",
                    f"{item['mean']:.6g}" if item["mean"] is not None else "n/a",
                    f"{item['missing_fraction']:.3%}" if item["missing_fraction"] is not None else "n/a",
                    item["negative"],
                ]
            )

    profile_rows = []
    for family in DATASET_DIRS:
        z = profiles["z_coordinates"][family]
        tke = profiles["negative_tke"][family]
        profile_rows.append(
            [
                family,
                ", ".join(f"{shape}:{count}" for shape, count in profiles["shape_counts"][family].items()),
                z["unique_coordinate_vectors"],
                f"{z['example'][0]:.3g}–{z['example'][-1]:.3g}" if z["example"] else "n/a",
                tke["all_rows"],
                tke["excluding_last_row"],
            ]
        )

    dtype_rows = []
    for family, variables in netcdf["pedestrian"]["dtype_counts"].items():
        dtype_rows.append(
            [family, ", ".join(f"{name}:{'/'.join(counts)}" for name, counts in variables.items())]
        )

    scaling_rows = []
    for family, item in netcdf["pedestrian"]["aggregate_scaling_check"]["summary"].items():
        speed = item["speed_ratio_raw_over_table_utau"]
        tke = item["tke_ratio_raw_over_table_utau2"]
        scaling_rows.append(
            [
                family,
                item["cases"],
                f"{speed['mean']:.4f} ({speed['min']:.4f}–{speed['max']:.4f})",
                f"{tke['mean']:.4f} ({tke['min']:.4f}–{tke['max']:.4f})",
            ]
        )

    val_pair_rows = [
        [
            item["val_case"],
            item["base_case"],
            "×".join(map(str, item["val_shape"])),
            "×".join(map(str, item["base_shape"])) if item["base_shape"] else "n/a",
            item["val_dx_m"],
            item["base_dx_m"],
        ]
        for item in reconciliation["validation_named_resolution_pairs"]
    ]

    missing_previews = [
        record["case_name"] for record in summary["cases"] if "preview_png" in record["missing_roles"]
    ]
    split_candidates = summary["split_candidates"]
    lines = [
        "# UrbanTALES 本地数据审计报告",
        "",
        f"生成时间：{summary['generated_at']}。本报告由 `scripts/audit_urbantales.py` 从原始下载目录只读生成。",
        "",
        "## 1. 审计结论摘要",
        "",
        (
            f"- 本地共有 538 个模拟案例目录：Idealized 224 个、Realistic 314 个。目录名与 `metadata.csv:NameE` 有 "
            f"{reconciliation['status_counts'].get('exact_name', 0)} 个精确匹配、"
            f"{reconciliation['status_counts'].get('casefold_name_alias', 0)} 个仅大小写不同；另有 "
            f"{reconciliation['status_counts'].get('inferred_physical_signature', 0)} 个按分辨率、物理域、风向、几何类型和形态统计建立一对一推断映射，"
            f"未解析 {reconciliation['status_counts'].get('unresolved', 0)} 个。"
        ),
        "- 可用于逐像素监督学习的空间输出是行人高度二维规则栅格 `*_ped.nc`，不是三维体数据。其字段为 `uped`、`vped`、`Uped`、`TKEped`、`Tuwped`。",
        "- `*_topo` 是与二维行人场同形状的建筑高度栅格；`profile-*.csv` 是水平/冠层统计后的 130 点垂向剖面；`*_ts.nc` 是 PALM 全局诊断时间序列。",
        "- 当前下载中没有压力场、三维速度体、非结构网格、点云或地形高程与建筑高度分离后的独立通道。`topo` 应理解为表面/建筑高度栅格。",
        "- 本地未发现独立的 train/validation/test 清单，但有 5 个 `Val-*` 高分辨率对应案例；该前缀是否表示论文中的数值验证而非机器学习验证集，需查官方论文后确认。后续不能仅凭前缀直接划分。",
        "- NetCDF 坐标变量没有单位且数值范围近似网格索引；物理水平分辨率应以 `metadata.csv:dxdy` 或 `p3d:dx/dy` 重建，不能直接把 NetCDF `x/y` 当作米制坐标。",
        "",
        "## 2. 文件与案例规模",
        "",
        markdown_table(
            ["数据族", "案例目录", "去风向几何数", "合并Val对应后的site数", "文件数", "容量 GiB", "城市数", "dxdy(m):案例数"],
            dataset_rows,
        ),
        "",
        markdown_table(["数据族", "文件角色", "数量", "容量 GiB"], role_rows),
        "",
        markdown_table(["扩展名", "数量", "容量 GiB"], extension_rows),
        "",
        f"缺失预览 PNG 的 10 个案例：{', '.join(missing_previews)}。其余五类核心数据文件未缺失。",
        "",
        "### 2.1 目录名与元数据名",
        "",
        (
            f"精确名称匹配 {reconciliation['status_counts'].get('exact_name', 0)} 个；仅大小写别名 "
            f"{reconciliation['status_counts'].get('casefold_name_alias', 0)} 个；物理签名推断匹配 "
            f"{reconciliation['status_counts'].get('inferred_physical_signature', 0)} 个；未解析 "
            f"{reconciliation['status_counts'].get('unresolved', 0)} 个。原始目录名和元数据名不能直接作字符串 join；"
            "应使用 `audit_summary.json > metadata > name_reconciliation > mappings`。推断映射必须在文献阶段核验，"
            "不能把它描述成发布方提供的官方映射。"
        ),
        "",
        (
            "未解析目录：" + ", ".join(reconciliation["unresolved_cases"])
            + "；对应仍未使用的元数据名：" + ", ".join(reconciliation["unused_metadata_names"])
            + "。它们的物理域尺寸与这些同国别近似名称元数据行并不一致，故未强行配对。"
            if reconciliation["unresolved_cases"] else "所有案例均已获得名称映射。"
        ),
        "",
        markdown_table(
            ["Val案例", "同名基础案例", "Val形状", "基础形状", "Val dx(m)", "基础 dx(m)"],
            val_pair_rows,
        ),
        "",
        "这 5 对案例覆盖相同命名地点和物理域，但 `Val-*` 的水平分辨率为 0.5 m、对应基础案例为 1.0 m，栅格每轴约扩大 2 倍；它们必须作为关联组处理，避免多分辨率近重复泄漏。",
        "",
        "## 3. 元数据、城市、风向和驱动条件",
        "",
        markdown_table(
            ["数据族", "案例", "几何布局", "城市", "国家", "WD 分布", "u_tau 范围(m/s)", "metadata物理域两轴范围(m)"],
            meta_family_rows,
        ),
        "",
        "`metadata.csv` 中的 `$u_{\\tau}$` 是 prescribed friction velocity；本地文件没有名为 inlet velocity 的统一入口风速字段。`p3d` 显示模拟采用外加压强梯度 `dp_external=.T.`/`dpdxy` 驱动，并记录顶边界、表面粗糙度、标量边界等设置。因此后续条件编码应优先使用风向、摩阻速度或压强梯度，而不能把它们无说明地称为入口风速。",
        "",
        markdown_table(
            ["数据族", "nz", "dx", "dy", "dz", "dp_external", "不同dpdxy", "bc_uv_t", "粗糙度(m)"],
            boundary_rows,
        ),
        "",
        "两类数据的 `nz=128`、基础 `dz=0.5 m`、50 m 以上伸展因子 1.1、`omega=0`、顶端速度 Neumann 边界和表面粗糙度 0.01 m 一致；差异主要在水平分辨率、压强梯度方向与几何。以上是配置文件中明确出现的参数，不等同于完整复现 PALM 所需的全部边界条件。",
        "",
        "## 4. 建筑几何与空间网格",
        "",
        markdown_table(
            ["数据族", "不同形状数", "形状:案例数（最多12项）", "按shape×dx的物理域(m)", "高度范围(m)", "总体非零栅格占比", "非有限值", "与p3d不匹配"],
            topo_rows,
        ),
        "",
        "Idealized 几何为规则化方块阵列，但并非统一张量尺寸；Realistic 的形状变化更大。`topo` 与二维场逐案例同形，且所有案例都满足 `topo.shape == (ny+1, nx+1)`，可直接构造高度栅格、占据掩码或由高度图计算的二维 SDF。若模型要求统一尺寸，需要物理尺度一致的裁剪/重采样/填充策略，并在消融中单独验证。高度单位按元数据建筑高度和 PALM 网格配置解释为米；无扩展名文件本身没有独立单位头。",
        "",
        "所有 `*_ped.nc` 的坐标都从 0 到维度长度本身（例如长度384的 `x` 为0–384），相邻步长约为 `N/(N-1)`，与 0.5 m/1.0 m 的 `dxdy` 不一致。随附预览 PNG 也把栅格计数标为米，但旁注物理域按 `dxdy` 缩放。审计因此把 NetCDF 坐标视为索引型辅助坐标，并用 `shape × dxdy` 重建物理范围；这是需要官方资料确认的发布后处理问题。",
        "",
        "## 5. 行人高度二维 NetCDF",
        "",
        markdown_table(
            ["数据族", "变量", "案例", "最小值", "最大值", "均值", "缺失/掩码率", "负值数"],
            ped_rows,
        ),
        "",
        markdown_table(["数据族", "NetCDF变量dtype"], dtype_rows),
        "",
        markdown_table(
            ["数据族", "可映射案例", "mean(Uped)/(pedU·u_tau)", "mean(TKEped)/(pedTKE·u_tau²)"],
            scaling_rows,
        ),
        "",
        "NetCDF 空间场与汇总表/剖面的数值尺度不同：上表两个比值整体接近 1，说明二者很可能通过摩阻速度 `u_tau` 和 `u_tau²` 关联。但由于 `*_ped.nc` 无单位属性、随附 datasheet 只直接解释汇总表列，当前只能记录这一经验关系，不能在官方资料核对前断言哪一侧已无量纲化。预处理阶段必须把尺度换算作为显式、可切换并可单测的步骤。",
        "",
        "掩码/NaN 主要位于 1.75 m 采样面以下的建筑内部或交错网格边界；不同变量的掩码并不保证完全相同。模型训练必须为每个目标通道保留有效域掩码，不能先把 NaN 直接替换为零再无掩码计算损失。`Uped` 和 `TKEped` 的非掩码负值计数见 JSON 异常项；`uped/vped/Tuwped` 出现负值在物理上允许。",
        "",
        (
            f"一致性检查中，Idealized 有效单元全部满足 `Uped >= sqrt(uped²+vped²)`；Realistic 有 "
            f"{netcdf['pedestrian']['anomalies']['realistic']['jensen_violations']:,} / "
            f"{netcdf['pedestrian']['anomalies']['realistic']['jensen_compared_cells']:,}（"
            f"{netcdf['pedestrian']['anomalies']['realistic']['jensen_violation_fraction']:.3%}）单元超过 1e-5 容差，"
            f"最大存储值差 {netcdf['pedestrian']['anomalies']['realistic']['jensen_max_gap']:.6g}。"
            "这可能来自变量平均/插值定义差异或后处理不一致，不能擅自把 `Uped` 重算为分量模长；需在官方论文中核对。"
        ),
        "",
        "## 6. 垂向剖面与时间序列",
        "",
        markdown_table(
            ["数据族", "剖面形状:案例数", "不同z向量", "z范围", "TKE负值总数", "去掉末层后TKE负值"],
            profile_rows,
        ),
        "",
        "所有剖面使用 130 个垂向点并包含 30 个数值变量（另有一个导出索引列）。剖面首层和顶层存在结构性缺失；每个案例最后一层的 TKE 为负值。另有 `AU-Mel-U9_d00` 从 z=39.25 m（索引79）到倒数第二层连续出现 50 个负 TKE，最低 -4.03028 m²/s²，是明确需要隔离的异常剖面。若把剖面作为辅助监督，需定义有效高度范围、排除末层并对该案例做质量控制，不能按普通缺失值随机插补。",
        "",
        "`*_ts.nc` 仅沿 `time` 维存储 E、dt、us*、umax/vmax/wmax、div_new/div_old 等全局诊断量。它们可用于筛查数值收敛或构造质量控制特征，但不能当作三维空间标签。",
        "",
        "## 7. 缺失、异常与一致性",
        "",
        f"- NetCDF 打开/扫描错误：{len(netcdf['errors'])}；剖面读取错误：{len(profiles['errors'])}；PNG 读取错误：{len(summary['previews']['errors'])}。",
        f"- 元数据重复案例名：{len(metadata['metadata_duplicate_case_names'])}；流场汇总表重复案例名：{len(metadata['flow_duplicate_case_names'])}。",
        f"- 本地没有独立 split 清单；名称候选目录：{', '.join(split_candidates) if split_candidates else '未发现'}。这些 `Val-*` 更像分辨率验证对应案例，是否构成官方机器学习验证集尚未确认。",
        "- 物理量单位主要来自随附 datasheet；`*_ped.nc` 本身未写 `units/long_name`，这是解析器和论文方法部分需要明确记录的数据谱系风险。",
        "- `metadata.csv` 的 WD 字段含 `FLUX` 标签，不能强制全列转成角度；这些案例需要结合对应 `p3d/dpdxy` 单独解释。",
        "",
        "## 8. 域差异与可支持的问题",
        "",
        "已确认的域差异包括：Idealized/Realistic 的几何规律性、城市语义、分辨率分布、栅格尺寸分布、建筑高度/密度分布、风向覆盖和部分 NetCDF 数据类型不同。两域有相同的二维目标变量与高度图输入，因此可以开展 Idealized 预训练到 Realistic 微调、留城市测试和跨几何泛化；但当前本地数据不能支撑真正的三维体风场预测或压力场预测。",
        "",
        "## 9. 建议的数据划分（非官方，待文献阶段再锁定）",
        "",
        "1. 以去除 `_dXX` 且合并 `Val-` 对应关系后的 `site_key` 为最小分组，所有相同几何的风向和多分辨率对应案例必须进入同一集合。",
        "2. Realistic 主测试建议再按城市/地区整组留出，避免同城相邻切片泄漏；保留一个跨域测试：只在 Idealized 训练，直接测试 Realistic。",
        "3. 未见风向测试应在训练几何之外评估，或对具有多风向的同一几何建立专门协议；不能与随机案例切分混用。",
        "4. `FLUX` 案例在确认其物理含义前单列，不并入角度插值实验。",
        "",
        "## 10. 可视化与复现",
        "",
        "- `figures/sample_idealized.png`：规则化建筑高度与四个行人高度场。",
        "- `figures/sample_realistic.png`：真实街区建筑高度与四个行人高度场。",
        "- `figures/dataset_overview.png`：风向、建筑高度、平面密度和栅格规模的域差异。",
        "- 完整机器可读统计见 `audit_summary.json`；脚本入口为 `scripts/audit_urbantales.py`。",
        "- 审计为确定性全量扫描，不做随机抽样；原始数据以只读方式打开。",
        "",
        "```powershell",
        "python -m pip install -r requirements-audit.txt",
        "python scripts/audit_urbantales.py --root . --output reports/data_audit",
        "```",
        "",
        "## 11. 尚未确认",
        "",
        "- 数据发布方是否提供未下载的三维瞬时/平均体场、压力场或机器学习官方 split；本地文件与本地 datasheet 均未给出独立清单。",
        (
            f"- {reconciliation['status_counts'].get('inferred_physical_signature', 0)} 个目录名与元数据 `NameE` 的一对一关系"
            "是基于物理签名的审计推断，需由官方命名说明或论文补充材料确认。"
        ),
        (
            "- 尚未解析的目录/元数据命名对应：" + ", ".join(reconciliation["unresolved_cases"])
            if reconciliation["unresolved_cases"] else "- 目录与元数据名称均已有映射。"
        ),
        "- 5 个 `Val-*` 是否专指网格无关性/数值验证，以及它们与 1.0 m 基础案例的精确生成关系。",
        "- `FLUX` 的精确定义，以及部分无 `_dXX` 后缀案例的风向/驱动约定，需要下一阶段核对官方论文和发布说明。",
        "- `uped/vped/Uped` 的平均算子细节、坐标轴命名与物理 x-y 方向的对应关系，需用官方数据论文确认；本地 datasheet 只给变量的总体定义。",
        "- Realistic 案例是否存在同一原始城市切片经旋转/裁剪形成的近重复几何，需要下一阶段增加几何哈希/相似度审计后才能最终冻结 split。",
        "",
    ]
    return "\n".join(lines)


def discover_cases(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_records: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    extension_counts: collections.Counter = collections.Counter()
    extension_bytes: collections.Counter = collections.Counter()
    largest_files: list[tuple[int, str]] = []
    all_files = 0
    all_bytes = 0

    for family, dirname in DATASET_DIRS.items():
        dataset_root = root / dirname
        directories = sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: natural_sort_key(path.name))
        family_files = [path for path in dataset_root.rglob("*") if path.is_file()]
        family_bytes = sum(path.stat().st_size for path in family_files)
        role_counts: collections.Counter = collections.Counter()
        role_bytes: collections.Counter = collections.Counter()
        for directory in directories:
            files: dict[str, str] = {}
            other_files: list[str] = []
            for path in sorted(directory.iterdir(), key=lambda item: natural_sort_key(item.name)):
                if not path.is_file():
                    continue
                role = classify_case_file(path, directory.name)
                if role == "other":
                    other_files.append(str(path))
                else:
                    files[role] = str(path)
                role_counts[role] += 1
                role_bytes[role] += path.stat().st_size
            missing_roles = [role for role in CASE_ROLES if role not in files]
            case_records.append(
                {
                    "dataset": family,
                    "case_name": directory.name,
                    "geometry_key": geometry_key(directory.name),
                    "site_key": site_key(directory.name),
                    "files": files,
                    "other_files": other_files,
                    "missing_roles": missing_roles,
                }
            )
        for path in family_files:
            suffix = path.suffix.lower() or "[none]"
            extension_counts[suffix] += 1
            extension_bytes[suffix] += path.stat().st_size
            largest_files.append((path.stat().st_size, str(path.relative_to(root))))
        datasets[family] = {
            "path": str(dataset_root),
            "case_directories": len(directories),
            "files": len(family_files),
            "bytes": family_bytes,
            "role_counts": counter_dict(role_counts),
            "role_bytes": counter_dict(role_bytes),
        }
        all_files += len(family_files)
        all_bytes += family_bytes

    return case_records, {
        "datasets": datasets,
        "total_files": all_files,
        "total_bytes": all_bytes,
        "extension_counts": counter_dict(extension_counts),
        "extension_bytes": counter_dict(extension_bytes),
        "largest_files": [
            {"path": path, "bytes": size} for size, path in sorted(largest_files, reverse=True)[:20]
        ],
    }


def find_split_candidates(root: Path) -> list[str]:
    pattern = re.compile(r"(?:^|[_\-.])(train|training|val|valid|validation|test|split|fold)(?:[_\-.]|$)", re.IGNORECASE)
    candidates = []
    for dirname in DATASET_DIRS.values():
        dataset_root = root / dirname
        for path in dataset_root.iterdir():
            if pattern.search(path.name):
                candidates.append(str(path.relative_to(root)))
    return sorted(candidates, key=natural_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    print("[1/7] Discovering cases and inventorying files", flush=True)
    case_records, inventory = discover_cases(root)
    cases_by_name = {record["case_name"]: record for record in case_records}

    print("[2/7] Reading metadata and aggregate flow CSV files", flush=True)
    metadata, flow, metadata_summary = inspect_metadata(root, cases_by_name)

    print("[3/7] Parsing PALM p3d configuration files", flush=True)
    p3d = inspect_p3d(case_records)

    print("[4/7] Scanning topography rasters", flush=True)
    topography_summary, topo_by_case = inspect_topography(case_records, p3d["by_case"])
    name_reconciliation = reconcile_case_names(metadata, case_records, topo_by_case, p3d["by_case"])
    metadata_summary["name_reconciliation"] = name_reconciliation

    print("[5/7] Scanning NetCDF schemas and pedestrian-field values", flush=True)
    netcdf_summary = inspect_netcdf(case_records, topo_by_case)
    netcdf_summary["pedestrian"]["aggregate_scaling_check"] = compare_pedestrian_scaling(
        metadata,
        flow,
        case_records,
        netcdf_summary["pedestrian"]["case_statistics"],
    )

    print("[6/7] Scanning vertical-profile CSV files and preview PNGs", flush=True)
    profiles = inspect_profiles(case_records)
    previews = inspect_previews(case_records)

    print("[7/7] Creating plots and reports", flush=True)
    metadata_index = metadata.set_index("NameE", drop=False)
    preferred_samples = {"idealized": "UA0625_d00", "realistic": "AU-Mel-U16_d00"}
    for family, case_name in preferred_samples.items():
        if case_name not in cases_by_name:
            case_name = next(record["case_name"] for record in case_records if record["dataset"] == family)
        plot_case(
            cases_by_name[case_name],
            metadata_index.loc[case_name],
            figures / f"sample_{family}.png",
        )
    plot_overview(metadata, topo_by_case, figures / "dataset_overview.png")

    summary = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "netCDF4": netCDF4.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "inventory": inventory,
        "metadata": metadata_summary,
        "p3d": p3d,
        "topography": {"summary": topography_summary, "by_case": topo_by_case},
        "netcdf": netcdf_summary,
        "profiles": profiles,
        "previews": previews,
        "split_candidates": find_split_candidates(root),
        "cases": case_records,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=python_value), encoding="utf-8"
    )
    (output / "DATA_AUDIT.md").write_text(build_report(summary), encoding="utf-8")
    print(f"Audit complete: {output}", flush=True)


if __name__ == "__main__":
    main()
