#!/usr/bin/env python3
"""Aggregate numeric metrics across aligned learning-curve/full-field seed rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t


def _aggregate(rows: list[dict[str, Any]], excluded: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(rows), "metrics": {}}
    common = set.intersection(*(set(row) for row in rows))
    for key in sorted(common - excluded):
        values = [row[key] for row in rows]
        numeric_values = [
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ]
        if not all(numeric_values):
            continue
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            continue
        mean = statistics.fmean(numeric)
        sample_std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        if len(numeric) > 1:
            critical = float(student_t.ppf(0.975, df=len(numeric) - 1))
            half_width = critical * sample_std / math.sqrt(len(numeric))
            confidence_95: dict[str, Any] | None = {
                "lower": mean - half_width,
                "upper": mean + half_width,
                "half_width": half_width,
                "method": "two_sided_Student_t_over_seed_means",
                "degrees_of_freedom": len(numeric) - 1,
            }
        else:
            confidence_95 = None
        result["metrics"][key] = {
            "mean": mean,
            "sample_std": sample_std,
            "confidence_95": confidence_95,
            "min": min(numeric),
            "max": max(numeric),
            "values": numeric,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves = json.loads(args.curves.read_text(encoding="utf-8"))
    full_fields = json.loads(args.full_fields.read_text(encoding="utf-8"))
    result = {
        "patch_validation": _aggregate(curves, {"curve"}),
        "complete_field_validation": _aggregate(full_fields, set()),
        "curve_runs": [row["run"] for row in curves],
        "complete_field_runs": [row["run"] for row in full_fields],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
