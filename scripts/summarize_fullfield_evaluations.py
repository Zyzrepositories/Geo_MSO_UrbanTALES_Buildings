#!/usr/bin/env python3
"""Create a compact table from complete-field evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = {
    "Uped_mae_m_s": "Uped/physical_m_s/mae",
    "Uped_rmse_m_s": "Uped/physical_m_s/rmse",
    "Uped_relative_l2": "Uped/physical_m_s/relative_l2",
    "TKEped_mae_m2_s2": "TKEped/physical_m2_s2/mae",
    "uped_mae_m_s": "uped/physical_m_s/mae",
    "vped_mae_m_s": "vped/physical_m_s/mae",
    "vector_error_mae_m_s": "vector/error_magnitude_mae_m_s",
    "direction_mae_deg": "vector/direction_mae_deg",
    "Uped_near_building_mae_m_s": "Uped/near_building_m_s/mae",
    "Uped_high_gradient_mae_m_s": "Uped/high_gradient_m_s/mae",
    "horizontal_divergence_error_mae_s_inv": "horizontal_divergence/error_mae_s_inv",
    "Uped_coverage_90": "Uped/uncertainty_physical_m_s/coverage_90",
}


def _row(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["macro_case_metrics"]
    row: dict[str, Any] = {
        "run": run_dir.name,
        "model": summary.get(
            "model_kind",
            (
                "checkpoint"
                if summary.get("checkpoint")
                else f"baseline_{summary.get('baseline')}"
            ),
        ),
        "protocol": summary["protocol"],
        "partition": summary["partition"],
        "case_count": summary["case_count"],
        "total_tiles": summary["total_tiles"],
        "total_forward_seconds": summary["total_forward_seconds"],
        "total_evaluation_pipeline_seconds": summary["total_evaluation_pipeline_seconds"],
    }
    row.update(
        {
            output: float(metrics[source]) if source in metrics else None
            for output, source in METRICS.items()
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [_row(args.root / run) for run in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
