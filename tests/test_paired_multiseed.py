from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.compare_paired_multiseed import METRICS, main


def _rows(prefix: str, scale: float) -> list[dict]:
    return [
        {
            "run": f"{prefix}_{seed}",
            "case_count": 81,
            **{metric: scale * (index + 1) for metric in METRICS},
        }
        for index, seed in enumerate((1, 2, 3))
    ]


def test_paired_multiseed_reports_ordered_differences(tmp_path: Path, monkeypatch) -> None:
    reference = tmp_path / "reference.json"
    ablation = tmp_path / "ablation.json"
    output = tmp_path / "comparison.json"
    reference.write_text(json.dumps(_rows("reference", 1.0)), encoding="utf-8")
    ablation.write_text(json.dumps(_rows("ablation", 0.9)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_paired_multiseed.py",
            "--reference",
            str(reference),
            "--ablation",
            str(ablation),
            "--seeds",
            "1",
            "2",
            "3",
            "--output",
            str(output),
        ],
    )

    main()

    result = json.loads(output.read_text(encoding="utf-8"))
    metric = result["aggregate"]["Uped_mae_m_s"]
    assert metric["ablation_better_seed_count"] == 3
    assert abs(metric["paired_relative_percent"]["mean"] + 10.0) < 1e-12
    assert output.with_suffix(".csv").is_file()

