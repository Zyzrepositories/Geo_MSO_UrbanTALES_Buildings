#!/usr/bin/env python3
"""Create absolute and relative complete-field deltas against one reference run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXCLUDED = {
    "run",
    "model",
    "protocol",
    "partition",
    "case_count",
    "total_tiles",
    "total_forward_seconds",
    "total_evaluation_pipeline_seconds",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-coverage", type=float, default=0.9)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if any(int(row["case_count"]) != 81 for row in rows):
        raise ValueError("all runs must contain 81 complete validation cases")
    if len({int(row["total_tiles"]) for row in rows}) != 1:
        raise ValueError("all runs must use the same number of reconstruction tiles")
    by_name = {row["run"]: row for row in rows}
    reference = by_name[args.reference_run]

    comparisons = []
    for row in rows:
        if row["run"] == args.reference_run:
            continue
        metrics = {}
        for metric, ref_value in reference.items():
            if metric in EXCLUDED or ref_value is None or row.get(metric) is None:
                continue
            value = float(row[metric])
            ref_value = float(ref_value)
            difference = value - ref_value
            metrics[metric] = {
                "reference": ref_value,
                "ablation": value,
                "difference": difference,
                "relative_percent": 100.0 * difference / max(abs(ref_value), 1e-12),
            }
        coverage = row.get("Uped_coverage_90")
        reference_coverage = reference.get("Uped_coverage_90")
        calibration = None
        if coverage is not None and reference_coverage is not None:
            calibration = {
                "reference_absolute_error": abs(float(reference_coverage) - args.nominal_coverage),
                "ablation_absolute_error": abs(float(coverage) - args.nominal_coverage),
            }
            calibration["absolute_error_difference"] = (
                calibration["ablation_absolute_error"]
                - calibration["reference_absolute_error"]
            )
        comparisons.append(
            {
                "run": row["run"],
                "metrics": metrics,
                "coverage_calibration": calibration,
            }
        )

    result = {
        "reference_run": args.reference_run,
        "case_count_per_run": 81,
        "total_tiles_per_run": int(reference["total_tiles"]),
        "lower_is_better_for_error_metrics": True,
        "comparisons": comparisons,
        "test_targets_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run", "metric", "reference", "ablation", "difference", "relative_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for metric, values in comparison["metrics"].items():
                writer.writerow({"run": comparison["run"], "metric": metric, **values})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

