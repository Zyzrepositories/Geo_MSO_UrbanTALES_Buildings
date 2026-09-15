#!/usr/bin/env python3
"""Apply the frozen validation-curve rule without consulting test targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rule: dict[str, Any] = yaml.safe_load(args.rule.read_text(encoding="utf-8"))
    if rule.get("status") != "frozen":
        raise ValueError("Convergence decision rule must be frozen")
    rows = json.loads(args.convergence.read_text(encoding="utf-8"))
    by_run = {row["run"]: row for row in rows}
    candidates = [by_run[name] for name in rule["candidate_runs"]]
    required = int(rule["required_epochs_per_candidate"])
    incomplete = [row["run"] for row in candidates if int(row["epoch_count"]) < required]
    if incomplete:
        raise ValueError(f"Candidate runs have fewer than {required} epochs: {incomplete}")

    ranked = sorted(candidates, key=lambda row: float(row["best_val_loss"]))
    selected = ranked[0]
    curve = selected["curve"]
    late_start = int(rule["late_window_start_epoch"])
    early_values = [float(point["val_loss"]) for point in curve if point["epoch"] < late_start]
    late = [point for point in curve if point["epoch"] >= late_start]
    if not early_values or len(late) < 2:
        raise ValueError("Learning curve does not cover both decision windows")
    early_best = min(early_values)
    late_best = min(float(point["val_loss"]) for point in late)
    relative_gain = max(0.0, (early_best - late_best) / max(abs(early_best), 1e-12))
    slope_points = curve[-min(10, len(curve)) :]
    recent_slope = float(
        np.polyfit(
            [point["epoch"] for point in slope_points],
            [point["val_loss"] for point in slope_points],
            1,
        )[0]
    )
    thresholds = rule["extend_selected_model_if"]
    extend = (
        relative_gain >= float(thresholds["late_best_relative_gain_at_least"])
        or recent_slope <= float(thresholds["or_last_10_epoch_slope_at_most"])
    )
    stop_epoch_key = (
        "selected_screening_stop_with_extension"
        if extend
        else "selected_screening_stop_without_extension"
    )
    result = {
        "rule_id": rule["rule_id"],
        "selected_run": selected["run"],
        "selected_model": selected["model"],
        "ranked_runs": [
            {
                "rank": rank,
                "run": row["run"],
                "model": row["model"],
                "best_epoch": row["best_epoch"],
                "best_val_loss": row["best_val_loss"],
            }
            for rank, row in enumerate(ranked, start=1)
        ],
        "selected_early_best_before_epoch_40": early_best,
        "selected_late_best_from_epoch_40": late_best,
        "selected_late_relative_gain": relative_gain,
        "selected_last_10_epoch_slope": recent_slope,
        "extend_selected_model": extend,
        "selected_screening_stop_epoch": int(rule[stop_epoch_key]),
        "formal_training_max_epochs": int(rule["formal_training_max_epochs"]),
        "formal_early_stopping_patience": int(rule["formal_early_stopping_patience"]),
        "test_targets_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
