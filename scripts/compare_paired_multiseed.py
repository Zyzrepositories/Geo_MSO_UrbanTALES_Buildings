#!/usr/bin/env python3
"""Compare ordered reference/ablation full-field results as paired seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "Uped_mae_m_s",
    "Uped_rmse_m_s",
    "Uped_relative_l2",
    "TKEped_mae_m2_s2",
    "uped_mae_m_s",
    "vped_mae_m_s",
    "vector_error_mae_m_s",
    "direction_mae_deg",
    "Uped_near_building_mae_m_s",
    "Uped_high_gradient_mae_m_s",
    "horizontal_divergence_error_mae_s_inv",
    "Uped_coverage_90",
)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    if len(reference) != len(ablation) or len(reference) != len(args.seeds):
        raise ValueError("reference, ablation, and seed counts must match")
    if any(int(row["case_count"]) != 81 for row in reference + ablation):
        raise ValueError("every paired full-field run must contain 81 validation cases")

    pairs = []
    for seed, ref_row, abl_row in zip(args.seeds, reference, ablation, strict=True):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            ref_value = ref_row.get(metric)
            abl_value = abl_row.get(metric)
            if ref_value is None or abl_value is None:
                metrics[metric] = None
                continue
            difference = float(abl_value) - float(ref_value)
            metrics[metric] = {
                "reference": float(ref_value),
                "ablation": float(abl_value),
                "paired_difference": difference,
                "paired_relative_percent": 100.0 * difference / max(abs(float(ref_value)), 1e-12),
            }
        pairs.append(
            {
                "seed": seed,
                "reference_run": ref_row["run"],
                "ablation_run": abl_row["run"],
                "metrics": metrics,
            }
        )

    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        available = [pair["metrics"][metric] for pair in pairs if pair["metrics"][metric] is not None]
        if not available:
            aggregate[metric] = None
            continue
        aggregate[metric] = {
            "reference": _stats([item["reference"] for item in available]),
            "ablation": _stats([item["ablation"] for item in available]),
            "paired_difference": _stats([item["paired_difference"] for item in available]),
            "paired_relative_percent": _stats(
                [item["paired_relative_percent"] for item in available]
            ),
            "ablation_better_seed_count": sum(
                item["paired_difference"] < 0.0 for item in available
            ),
        }

    result = {
        "seeds": args.seeds,
        "case_count_per_run": 81,
        "lower_is_better_for_all_reported_metrics": True,
        "pairs": pairs,
        "aggregate": aggregate,
        "test_targets_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "seed",
            "metric",
            "reference",
            "ablation",
            "paired_difference",
            "paired_relative_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            for metric, values in pair["metrics"].items():
                if values is None:
                    continue
                writer.writerow({"seed": pair["seed"], "metric": metric, **values})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

