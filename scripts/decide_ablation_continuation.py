#!/usr/bin/env python3
"""Apply the preregistered continuation rule at later ablation boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--expected-boundary", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    rule = protocol["continuation_rule"]
    chunk_epochs = int(rule["chunk_epochs"])
    hard_maximum = int(rule["hard_maximum_epochs"])
    boundary = int(args.expected_boundary)
    if boundary <= chunk_epochs or boundary % chunk_epochs:
        raise ValueError("expected boundary must be a later multiple of chunk_epochs")
    if boundary > hard_maximum:
        raise ValueError("expected boundary exceeds the hard maximum")

    allowed_runs = set(protocol["eligible_runs"])
    unknown = sorted(set(args.runs) - allowed_runs)
    if unknown:
        raise ValueError(f"runs are not eligible under the frozen protocol: {unknown}")

    decisions = []
    for run_name in args.runs:
        run_dir = args.root / run_name
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
        if not np.array_equal(epochs, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"{run_name}: epochs are not unique and contiguous from zero")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        stopped_early = bool(summary["stopped_early"])
        completed = int(epochs[-1]) + 1
        if completed > boundary:
            raise ValueError(f"{run_name}: run passed the expected boundary {boundary}")
        if completed < boundary and not stopped_early:
            raise ValueError(f"{run_name}: incomplete chunk without early stopping")

        values = np.asarray([row["val"]["loss"] for row in rows], dtype=np.float64)
        best_index = int(values.argmin())
        late_start = boundary - int(rule["late_window_epochs"])
        previous_values = values[epochs < late_start]
        late_values = values[(epochs >= late_start) & (epochs < boundary)]
        relative_gain = 0.0
        if previous_values.size and late_values.size:
            previous_best = float(previous_values.min())
            late_best = float(late_values.min())
            relative_gain = (previous_best - late_best) / max(abs(previous_best), 1e-12)
        slope = 0.0
        if late_values.size > 1:
            late_epochs = epochs[(epochs >= late_start) & (epochs < boundary)].astype(float)
            slope = float(np.polyfit(late_epochs, late_values, 1)[0])

        triggers = {
            "best_in_final_epochs": int(epochs[best_index])
            >= boundary - int(rule["best_within_final_epochs"]),
            "late_relative_gain": relative_gain
            >= float(rule["late_relative_gain_at_least"]),
            "late_slope": slope <= float(rule["late_slope_at_most"]),
        }
        at_hard_maximum = completed >= hard_maximum
        extend = (
            completed == boundary
            and not stopped_early
            and not at_hard_maximum
            and any(triggers.values())
        )
        decisions.append(
            {
                "run": run_name,
                "expected_boundary": boundary,
                "epoch_count": len(rows),
                "last_epoch": int(epochs[-1]),
                "best_epoch": int(epochs[best_index]),
                "best_validation_objective": float(values[best_index]),
                "stopped_early": stopped_early,
                "late_relative_gain": relative_gain,
                "last_window_slope_per_epoch": slope,
                "triggers": triggers,
                "decision": "extend" if extend else "stop",
            }
        )

    result = {
        "protocol_id": protocol["protocol_id"],
        "selection_partition": protocol["partition"],
        "expected_boundary": boundary,
        "decisions": decisions,
        "test_targets_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

