#!/usr/bin/env python3
"""Create paired seed-level comparisons for complete-field validation/test tables."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t


DEFAULT_ERROR_METRICS = (
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
)


def _summary(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        critical = float(student_t.ppf(0.975, df=len(values) - 1))
        half_width = critical * sample_std / math.sqrt(len(values))
        interval = {
            "lower": mean - half_width,
            "upper": mean + half_width,
            "half_width": half_width,
            "degrees_of_freedom": len(values) - 1,
            "method": "two_sided_Student_t_over_paired_seed_values",
        }
    else:
        interval = None
    return {
        "mean": mean,
        "sample_std": sample_std,
        "confidence_95": interval,
        "values": values,
    }


def compare_rows(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    seeds: list[int],
    metrics: tuple[str, ...] = DEFAULT_ERROR_METRICS,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or len(reference) != len(seeds):
        raise ValueError("Reference, candidate, and seed counts must match")
    result: dict[str, Any] = {"seed_count": len(seeds), "seeds": seeds, "metrics": {}}
    for metric in metrics:
        ref = [float(row[metric]) for row in reference]
        cand = [float(row[metric]) for row in candidate]
        difference = [cand_value - ref_value for ref_value, cand_value in zip(ref, cand)]
        relative = [
            100.0 * delta / ref_value if ref_value != 0 else math.nan
            for delta, ref_value in zip(difference, ref)
        ]
        result["metrics"][metric] = {
            "reference": _summary(ref),
            "candidate": _summary(cand),
            "candidate_minus_reference": _summary(difference),
            "candidate_relative_to_reference_percent": _summary(relative),
            "reference_lower_error_seed_count": sum(a < b for a, b in zip(ref, cand)),
            "candidate_lower_error_seed_count": sum(b < a for a, b in zip(ref, cand)),
            "ties": sum(a == b for a, b in zip(ref, cand)),
        }
    if all("Uped_coverage_90" in row for row in reference + candidate):
        nominal = 0.90
        ref_error = [abs(float(row["Uped_coverage_90"]) - nominal) for row in reference]
        cand_error = [abs(float(row["Uped_coverage_90"]) - nominal) for row in candidate]
        difference = [b - a for a, b in zip(ref_error, cand_error)]
        result["metrics"]["Uped_coverage_90_absolute_calibration_error"] = {
            "reference": _summary(ref_error),
            "candidate": _summary(cand_error),
            "candidate_minus_reference": _summary(difference),
            "reference_lower_error_seed_count": sum(a < b for a, b in zip(ref_error, cand_error)),
            "candidate_lower_error_seed_count": sum(b < a for a, b in zip(ref_error, cand_error)),
            "ties": sum(a == b for a, b in zip(ref_error, cand_error)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="LABEL=PATH; repeat for multiple candidates",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    for item in args.candidate:
        if "=" not in item:
            raise ValueError("Candidate must use LABEL=PATH")
        label, raw_path = item.split("=", 1)
        rows = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        comparisons[label] = compare_rows(reference, rows, args.seeds)
    result = {
        "reference_path": str(args.reference.resolve()),
        "candidate_comparisons": comparisons,
        "interpretation": (
            "Error differences are candidate minus reference; positive values favour "
            "the reference. Coverage is compared by absolute error from nominal 0.90."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

