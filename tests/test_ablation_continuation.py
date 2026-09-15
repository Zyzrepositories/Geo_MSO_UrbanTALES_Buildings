from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from scripts.decide_ablation_continuation import main


def _write_run(root: Path, name: str, epoch_count: int, stopped_early: bool) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    rows = []
    for epoch in range(epoch_count):
        loss = 1.0 - 0.002 * epoch
        rows.append({"epoch": epoch, "val": {"loss": loss}})
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"stopped_early": stopped_early}), encoding="utf-8"
    )


def _write_protocol(path: Path, runs: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "protocol_id": "test",
                "partition": "val",
                "eligible_runs": runs,
                "continuation_rule": {
                    "chunk_epochs": 50,
                    "hard_maximum_epochs": 200,
                    "late_window_epochs": 10,
                    "best_within_final_epochs": 5,
                    "late_relative_gain_at_least": 0.01,
                    "late_slope_at_most": -0.001,
                },
            }
        ),
        encoding="utf-8",
    )


def test_continuation_extends_at_boundary_and_stops_after_early_stop(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    _write_run(run_root, "complete", 100, False)
    _write_run(run_root, "early", 80, True)
    protocol = tmp_path / "protocol.yaml"
    output = tmp_path / "decision.json"
    _write_protocol(protocol, ["complete", "early"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decide_ablation_continuation.py",
            "--root",
            str(run_root),
            "--protocol",
            str(protocol),
            "--runs",
            "complete",
            "early",
            "--expected-boundary",
            "100",
            "--output",
            str(output),
        ],
    )

    main()

    decisions = {
        item["run"]: item
        for item in json.loads(output.read_text(encoding="utf-8"))["decisions"]
    }
    assert decisions["complete"]["decision"] == "extend"
    assert decisions["complete"]["triggers"]["best_in_final_epochs"] is True
    assert decisions["early"]["decision"] == "stop"
    assert decisions["early"]["stopped_early"] is True

