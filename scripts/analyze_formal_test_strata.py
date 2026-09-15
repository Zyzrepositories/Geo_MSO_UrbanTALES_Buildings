#!/usr/bin/env python3
"""Post-hoc explanatory analysis of locked formal-test case metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


GROUPS = {
    "main_geometry": {
        "scope": "geometry_grouped",
        "model": "Geo-MSO",
        "runs": [
            "formal_test_main_geometry_seed20260904_v1",
            "formal_test_main_geometry_seed20260905_v1",
            "formal_test_main_geometry_seed20260906_v1",
        ],
    },
    "unet_geometry": {
        "scope": "geometry_grouped",
        "model": "U-Net",
        "runs": [
            "formal_test_unet_geometry_seed20260904_v1",
            "formal_test_unet_geometry_seed20260905_v1",
            "formal_test_unet_geometry_seed20260906_v1",
        ],
    },
    "fno_geometry": {
        "scope": "geometry_grouped",
        "model": "FNO",
        "runs": [
            "formal_test_fno_geometry_seed20260904_v1",
            "formal_test_fno_geometry_seed20260905_v1",
            "formal_test_fno_geometry_seed20260906_v1",
        ],
    },
    "main_city": {
        "scope": "city_grouped",
        "model": "Geo-MSO",
        "runs": [
            "formal_test_main_city_seed20260904_v1",
            "formal_test_main_city_seed20260905_v1",
            "formal_test_main_city_seed20260906_v1",
        ],
    },
    "main_wind": {
        "scope": "wind_held_out",
        "model": "Geo-MSO",
        "runs": [
            "formal_test_main_wind_seed20260904_v1",
            "formal_test_main_wind_seed20260905_v1",
            "formal_test_main_wind_seed20260906_v1",
        ],
    },
}

METRICS = {
    "Uped_mae_m_s": "Uped/physical_m_s/mae",
    "Uped_rmse_m_s": "Uped/physical_m_s/rmse",
    "Uped_relative_l2": "Uped/physical_m_s/relative_l2",
    "TKEped_mae_m2_s2": "TKEped/physical_m2_s2/mae",
    "vector_error_mae_m_s": "vector/error_magnitude_mae_m_s",
    "Uped_near_building_mae_m_s": "Uped/near_building_m_s/mae",
    "Uped_high_gradient_mae_m_s": "Uped/high_gradient_m_s/mae",
    "horizontal_divergence_error_mae_s_inv": "horizontal_divergence/error_mae_s_inv",
    "Uped_coverage_90": "Uped/uncertainty_physical_m_s/coverage_90",
}
MAIN_GROUPS = ("main_geometry", "main_city", "main_wind")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tile_bin(value: int) -> str:
    if value <= 4:
        return "01-04"
    if value <= 16:
        return "05-16"
    if value <= 64:
        return "17-64"
    return "65+"


def _finite(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=np.float64)
    return data[np.isfinite(data)]


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    data = _finite(values)
    if data.size == 0:
        return {
            "case_count": 0,
            "mean": None,
            "sample_std": None,
            "median": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "case_count": int(data.size),
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _load_seed_averaged(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for group, definition in GROUPS.items():
        per_seed = [
            {record["case_id"]: record for record in _read_jsonl(root / run / "case_metrics.jsonl")}
            for run in definition["runs"]
        ]
        case_sets = [set(records) for records in per_seed]
        if any(case_ids != case_sets[0] for case_ids in case_sets[1:]):
            raise ValueError(f"Seed case IDs are not aligned for {group}")
        group_index: dict[str, dict[str, Any]] = {}
        for case_id in sorted(case_sets[0]):
            records = [records[case_id] for records in per_seed]
            first = records[0]
            metadata = {
                "group": group,
                "scope": definition["scope"],
                "model": definition["model"],
                "case_id": case_id,
                "family": first.get("family"),
                "city": first.get("city"),
                "wind_angle_deg": first.get("wind_angle_deg"),
                "source_dx_m": first.get("source_dx_m"),
                "output_height": int(first["output_shape"][0]),
                "output_width": int(first["output_shape"][1]),
                "output_pixels": int(first["output_shape"][0] * first["output_shape"][1]),
                "tile_count": int(first["tile_count"]),
                "tile_bin": _tile_bin(int(first["tile_count"])),
            }
            metadata["family_wind_angle"] = (
                f"{metadata['family']}|{metadata['wind_angle_deg']}"
            )
            metadata["family_source_dx"] = (
                f"{metadata['family']}|{metadata['source_dx_m']}"
            )
            metadata["family_tile_bin"] = (
                f"{metadata['family']}|{metadata['tile_bin']}"
            )
            for key in (
                "family", "city", "wind_angle_deg", "source_dx_m", "output_shape", "tile_count"
            ):
                if any(record.get(key) != first.get(key) for record in records[1:]):
                    raise ValueError(f"Seed metadata mismatch for {group}/{case_id}/{key}")
            for output_name, source_name in METRICS.items():
                values = [float(record["metrics"][source_name]) for record in records]
                metadata[output_name] = float(np.mean(values))
                metadata[f"{output_name}_seed_sd"] = float(np.std(values, ddof=1))
            rows.append(metadata)
            group_index[case_id] = metadata
        index[group] = group_index
    return rows, index


def _strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    dimensions = (
        "family",
        "city",
        "wind_angle_deg",
        "source_dx_m",
        "tile_bin",
        "family_wind_angle",
        "family_source_dx",
        "family_tile_bin",
    )
    for group in MAIN_GROUPS:
        selected = [row for row in rows if row["group"] == group]
        for dimension in dimensions:
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                value = row.get(dimension)
                if value is None or value == "":
                    continue
                buckets[str(value)].append(row)
            for level, bucket in sorted(buckets.items()):
                for metric in METRICS:
                    result.append(
                        {
                            "group": group,
                            "scope": GROUPS[group]["scope"],
                            "dimension": dimension,
                            "level": level,
                            "metric": metric,
                            **_describe(row[metric] for row in bucket),
                        }
                    )
    return result


def _paired(index: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = index["main_geometry"]
    case_rows: list[dict[str, Any]] = []
    for candidate in ("unet_geometry", "fno_geometry"):
        if set(index[candidate]) != set(reference):
            raise ValueError(f"Geometry candidate cases not aligned: {candidate}")
        for case_id in sorted(reference):
            ref = reference[case_id]
            cand = index[candidate][case_id]
            row = {
                "candidate": candidate,
                "case_id": case_id,
                "family": ref["family"],
                "city": ref["city"],
                "wind_angle_deg": ref["wind_angle_deg"],
                "source_dx_m": ref["source_dx_m"],
                "tile_count": ref["tile_count"],
                "tile_bin": ref["tile_bin"],
            }
            for metric in METRICS:
                row[f"{metric}_candidate_minus_main"] = float(cand[metric] - ref[metric])
            case_rows.append(row)

    summary: list[dict[str, Any]] = []
    dimensions = ("overall", "family", "city", "wind_angle_deg", "source_dx_m", "tile_bin")
    for candidate in ("unet_geometry", "fno_geometry"):
        selected = [row for row in case_rows if row["candidate"] == candidate]
        for dimension in dimensions:
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if dimension == "overall":
                buckets["all"] = selected
            else:
                for row in selected:
                    value = row.get(dimension)
                    if value is None or value == "":
                        continue
                    buckets[str(value)].append(row)
            for level, bucket in sorted(buckets.items()):
                metric = "Uped_mae_m_s_candidate_minus_main"
                values = [float(row[metric]) for row in bucket]
                summary.append(
                    {
                        "candidate": candidate,
                        "dimension": dimension,
                        "level": level,
                        **_describe(values),
                        "main_lower_error_case_fraction": float(np.mean(np.asarray(values) > 0)),
                    }
                )
    return case_rows, summary


def _correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group in MAIN_GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        subsets = {"overall": group_rows}
        for family in sorted({str(row["family"]) for row in group_rows}):
            subsets[f"family={family}"] = [
                row for row in group_rows if str(row["family"]) == family
            ]
        for subset_name, selected in subsets.items():
            error = np.asarray([row["Uped_mae_m_s"] for row in selected], dtype=np.float64)
            for predictor in ("output_pixels", "tile_count", "source_dx_m"):
                values = np.asarray([row[predictor] for row in selected], dtype=np.float64)
                if len(selected) < 3 or np.unique(values).size < 2:
                    rho, p_value = math.nan, math.nan
                else:
                    statistic = spearmanr(values, error)
                    rho, p_value = float(statistic.statistic), float(statistic.pvalue)
                result.append(
                    {
                        "group": group,
                        "subset": subset_name,
                        "predictor": predictor,
                        "case_count": len(selected),
                        "spearman_rho": rho,
                        "exploratory_p_value": p_value,
                        "interpretation": (
                            "post-hoc descriptive; family-stratified rows reduce but do not "
                            "eliminate geometry/size/resolution confounding"
                        ),
                    }
                )
    return result


def _failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group in MAIN_GROUPS:
        selected = sorted(
            (row for row in rows if row["group"] == group),
            key=lambda row: row["Uped_mae_m_s"],
            reverse=True,
        )
        for rank, row in enumerate(selected[:10], start=1):
            result.append(
                {
                    "group": group,
                    "rank": rank,
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "city": row["city"],
                    "wind_angle_deg": row["wind_angle_deg"],
                    "source_dx_m": row["source_dx_m"],
                    "tile_count": row["tile_count"],
                    "Uped_mae_m_s": row["Uped_mae_m_s"],
                    "Uped_relative_l2": row["Uped_relative_l2"],
                    "Uped_high_gradient_mae_m_s": row["Uped_high_gradient_mae_m_s"],
                    "Uped_coverage_90": row["Uped_coverage_90"],
                }
            )
    return result


def _plots(
    output: Path,
    rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})
    labels = list(GROUPS)
    values = [[row["Uped_mae_m_s"] for row in rows if row["group"] == label] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(values, tick_labels=[label.replace("_", "\n") for label in labels], showfliers=False)
    ax.set_ylabel("Case-level Uped MAE (m/s), averaged over 3 seeds")
    ax.set_title("Locked formal-test error distributions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "uped_mae_by_group.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    paired_values = [
        [
            row["Uped_mae_m_s_candidate_minus_main"]
            for row in paired_rows
            if row["candidate"] == candidate
        ]
        for candidate in ("unet_geometry", "fno_geometry")
    ]
    parts = ax.violinplot(paired_values, positions=[1, 2], showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks([1, 2], ["U-Net − Geo-MSO", "FNO − Geo-MSO"])
    ax.set_ylabel("Per-case Uped MAE difference (m/s)")
    ax.set_title("Geometry test: seed-averaged paired case differences")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "geometry_paired_case_differences.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=False)
    for ax, group in zip(axes, MAIN_GROUPS):
        selected = [row for row in rows if row["group"] == group]
        x = np.asarray([row["output_pixels"] for row in selected], dtype=np.float64)
        y = np.asarray([row["Uped_mae_m_s"] for row in selected], dtype=np.float64)
        ax.scatter(x, y, s=18, alpha=0.65)
        ax.set_xscale("log")
        ax.set_title(group.replace("main_", ""))
        ax.set_xlabel("Output pixels (log scale)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Uped MAE (m/s)")
    fig.suptitle("Main model error versus full-field size")
    fig.tight_layout()
    fig.savefig(output / "main_uped_mae_vs_field_size.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5))
    for ax, group in zip(axes, MAIN_GROUPS):
        selected = [row for row in failures if row["group"] == group]
        selected.reverse()
        ax.barh([row["case_id"] for row in selected], [row["Uped_mae_m_s"] for row in selected])
        ax.set_title(group.replace("main_", ""))
        ax.set_xlabel("Uped MAE (m/s)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Top-10 locked formal-test failure cases per main-model scope")
    fig.tight_layout()
    fig.savefig(output / "main_top_failure_cases.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Analysis output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows, index = _load_seed_averaged(args.root)
    strata = _strata(rows)
    paired_rows, paired_summary = _paired(index)
    correlations = _correlations(rows)
    failures = _failures(rows)
    _write_csv(args.output / "per_case_seed_mean.csv", rows)
    _write_csv(args.output / "main_strata_summary.csv", strata)
    _write_csv(args.output / "geometry_paired_case_differences.csv", paired_rows)
    _write_csv(args.output / "geometry_paired_strata_summary.csv", paired_summary)
    _write_csv(args.output / "main_error_correlations.csv", correlations)
    _write_csv(args.output / "main_top_failure_cases.csv", failures)
    _plots(args.output, rows, paired_rows, failures)

    payload = {
        "status": "ok",
        "analysis_kind": "post_hoc_explanatory_only",
        "model_selection_or_tuning_permitted": False,
        "seed_aggregation": "each case averaged over the three fixed seeds before stratification",
        "group_case_counts": {
            group: sum(row["group"] == group for row in rows) for group in GROUPS
        },
        "outputs": sorted(path.name for path in args.output.iterdir()),
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
