#!/usr/bin/env python3
"""Apply the frozen epoch-50 continuation rule to each ablation arm."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    rule = protocol["extension_after_epoch_50"]
    decisions = []
    for run_name in args.runs:
        run_dir = args.root / run_name
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        values = np.asarray([row["val"]["loss"] for row in rows], dtype=np.float64)
        epochs = np.asarray([row["epoch"] for row in rows], dtype=np.int64)
        best_index = int(values.argmin())
        early_values = values[epochs < 40]
        late_values = values[(epochs >= 40) & (epochs < 50)]
        relative_gain = 0.0
        if early_values.size and late_values.size:
            early_best = float(early_values.min())
            late_best = float(late_values.min())
            relative_gain = (early_best - late_best) / max(abs(early_best), 1e-12)
        window = min(10, len(rows))
        slope = (
            float(np.polyfit(epochs[-window:].astype(float), values[-window:], 1)[0])
            if window > 1
            else 0.0
        )
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stopped_early = bool(summary["stopped_early"])
        triggers = {
            "best_epoch": int(epochs[best_index])
            >= int(rule["extend_if_best_epoch_at_least"]),
            "late_relative_gain": relative_gain
            >= float(rule["or_epoch_40_to_49_relative_gain_at_least"]),
            "last_10_slope": slope <= float(rule["or_last_10_epoch_slope_at_most"]),
        }
        extend = not stopped_early and any(triggers.values())
        decisions.append(
            {
                "run": run_name,
                "epoch_count": len(rows),
                "last_epoch": int(epochs[-1]),
                "best_epoch": int(epochs[best_index]),
                "best_validation_objective": float(values[best_index]),
                "stopped_early": stopped_early,
                "late_relative_gain": relative_gain,
                "last_10_slope_per_epoch": slope,
                "triggers": triggers,
                "decision": "extend" if extend else "stop",
            }
        )
    result = {
        "protocol_id": protocol["protocol_id"],
        "selection_partition": protocol["partition"],
        "decisions": decisions,
        "test_targets_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

