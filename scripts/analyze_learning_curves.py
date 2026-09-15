#!/usr/bin/env python3
"""Summarize checkpoint-selection evidence from validation learning curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SELECTED_METRICS = {
    "Uped_mae_m_s": "Uped/physical_m_s/mae",
    "Uped_relative_l2": "Uped/physical_m_s/relative_l2",
    "TKEped_mae_m2_s2": "TKEped/physical_m2_s2/mae",
    "Uped_near_building_mae_m_s": "Uped/near_building_m_s/mae",
    "Uped_high_gradient_mae_m_s": "Uped/high_gradient_m_s/mae",
    "vector_error_mae_m_s": "vector/error_magnitude_mae_m_s",
    "direction_mae_deg": "vector/direction_mae_deg",
    "Uped_coverage_90": "Uped/uncertainty_physical_m_s/coverage_90",
}


def summarize_run(run_dir: Path, recent_window: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No learning-curve records in {run_dir}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise ValueError(f"Non-contiguous epochs in {run_dir}: {epochs}")
    values = np.asarray([row["val"]["loss"] for row in rows], dtype=np.float64)
    best_index = int(values.argmin())
    best = rows[best_index]
    window = min(int(recent_window), len(rows))
    recent_x = np.asarray(epochs[-window:], dtype=np.float64)
    recent_y = values[-window:]
    slope = float(np.polyfit(recent_x, recent_y, 1)[0]) if window > 1 else 0.0
    resolved_path = run_dir / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    data_config = resolved["data"]
    training_config = resolved["training"]
    optimization = summary.get("optimization", {})
    result: dict[str, Any] = {
        "run": run_dir.name,
        "model": resolved["model"]["name"],
        "seed": int(resolved["seed"]),
        "protocol": data_config["protocol"],
        "transfer_scope": data_config.get("transfer_scope"),
        "few_shot_percent": data_config.get("few_shot_percent"),
        "initialization": (
            "pretrained" if training_config.get("initial_checkpoint") else "scratch"
        ),
        "learning_rate": float(training_config["learning_rate"]),
        "train_case_count": len(summary.get("train_cases", [])) or None,
        "trainable_parameter_count": optimization.get("trainable_parameter_count"),
        "total_parameter_count": optimization.get("total_parameter_count"),
        "trainable_fraction": optimization.get("trainable_fraction"),
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
        "epoch_count": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val"]["loss"]),
        "last_val_loss": float(rows[-1]["val"]["loss"]),
        "best_train_loss": float(best["train"]["total"]),
        "last_train_loss": float(rows[-1]["train"]["total"]),
        "recent_window": window,
        "recent_val_loss_slope_per_epoch": slope,
        "recent_val_loss_change": float(recent_y[-1] - recent_y[0]),
        "best_within_last_five_epochs": int(best["epoch"]) >= epochs[-1] - 4,
        "total_epoch_compute_seconds": float(
            sum(row["epoch_seconds_excluding_checkpoint"] for row in rows)
        ),
    }
    for output_name, metric_name in SELECTED_METRICS.items():
        value = best["val"].get(metric_name)
        result[f"best_{output_name}"] = float(value) if value is not None else None
    result["curve"] = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train"]["total"]),
            "val_loss": float(row["val"]["loss"]),
            "Uped_mae_m_s": float(row["val"]["Uped/physical_m_s/mae"]),
            "TKEped_mae_m2_s2": float(row["val"]["TKEped/physical_m2_s2/mae"]),
        }
        for row in rows
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-window", type=int, default=5)
    args = parser.parse_args()
    results = [summarize_run(args.root / name, args.recent_window) for name in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{key: value for key, value in row.items() if key != "curve"} for row in results]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(csv_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
