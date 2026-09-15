#!/usr/bin/env python3
"""Aggregate fixed-seed complete-field results with Student-t intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_seed_results import _aggregate


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
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.full_fields.read_text(encoding="utf-8"))
    if len(rows) != 3:
        raise ValueError("Formal aggregate requires exactly three fixed seeds")
    if {row["partition"] for row in rows} != {"test"}:
        raise ValueError("Formal aggregate accepts labelled test results only")
    result = {
        "partition": "test",
        "seed_count": 3,
        "runs": [row["run"] for row in rows],
        "complete_field_test": _aggregate(rows, EXCLUDED),
        "inference_totals": {
            "forward_seconds": sum(float(row["total_forward_seconds"]) for row in rows),
            "evaluation_pipeline_seconds": sum(
                float(row["total_evaluation_pipeline_seconds"]) for row in rows
            ),
        },
        "interpretation": (
            "95% intervals are descriptive two-sided Student-t intervals over three "
            "fixed seed-level macro-case means; no model selection is permitted on test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
