#!/usr/bin/env python3
"""Issue all releases before results are read, then run the fixed formal test suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/evaluation/formal_test_job_manifest_v1.yaml"
PROTOCOL = ROOT / "configs/evaluation/formal_test_suite_protocol_v1.yaml"
RUN_ROOT = Path(os.environ.get("URBANTALES_RUN_ROOT", str(ROOT / "runs"))).expanduser()
LOG_ROOT = Path(os.environ.get("URBANTALES_LOG_ROOT", str(RUN_ROOT / "logs"))).expanduser()
PROGRESS = RUN_ROOT / "formal_test_suite_progress_v1.json"
PYTHON = Path(sys.executable)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _run(command: list[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def _write_progress(payload: dict[str, Any]) -> None:
    PROGRESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        raise FileExistsError(f"Refusing to overwrite formal test progress: {PROGRESS}")
    subprocess.run(
        [
            str(PYTHON),
            "scripts/validate_formal_test_suite_protocol.py",
            "--verify-artifacts",
            "--require-clean",
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    fixed = manifest["fixed_execution"]
    jobs: list[dict[str, Any]] = manifest["jobs"]
    progress: dict[str, Any] = {
        "status": "releases_being_issued",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL),
        "manifest": str(MANIFEST),
        "job_count": len(jobs),
        "releases_issued": [],
        "jobs_completed": [],
    }
    _write_progress(progress)

    # Critical safeguard: every job is released before the first labelled target is read.
    for job in jobs:
        command = [
            str(PYTHON),
            "scripts/create_test_release.py",
            "--training-lock", str(_path(job["training_lock"])),
            "--convergence", str(Path(job["convergence"])),
            "--selected-run", job["selected_run"],
            "--config", str(_path(job["config"])),
            "--checkpoint", str(Path(job["checkpoint"])),
            "--evaluation-scope", job["scope"],
            "--evaluation-output", job["output"],
            "--stride-pixels", str(fixed["stride_pixels"]),
            "--batch-size", str(fixed["batch_size"]),
            "--output", str(_path(job["release"])),
        ]
        _run(command, LOG_ROOT / f"{job['id']}_release.log")
        progress["releases_issued"].append(job["id"])
        _write_progress(progress)

    progress["status"] = "evaluating"
    progress["all_releases_issued_before_first_test_read"] = True
    _write_progress(progress)
    for job in jobs:
        command = [
            str(PYTHON),
            "scripts/evaluate_full_fields.py",
            "--config", str(_path(job["config"])),
            "--checkpoint", str(Path(job["checkpoint"])),
            "--partition", "test",
            "--test-release", str(_path(job["release"])),
            "--output", job["output"],
            "--stride-pixels", str(fixed["stride_pixels"]),
            "--batch-size", str(fixed["batch_size"]),
        ]
        _run(command, LOG_ROOT / f"{job['id']}_evaluation.log")
        progress["jobs_completed"].append(job["id"])
        progress["last_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_progress(progress)

    for group in ("main_geometry", "unet_geometry", "fno_geometry", "main_city", "main_wind"):
        group_jobs = [job for job in jobs if job["group"] == group]
        run_names = [Path(job["output"]).name for job in group_jobs]
        table = RUN_ROOT / f"formal_test_{group}_multiseed_v1.csv"
        _run(
            [
                str(PYTHON), "scripts/summarize_fullfield_evaluations.py",
                "--root", str(RUN_ROOT), "--runs", *run_names, "--output", str(table),
            ],
            LOG_ROOT / f"formal_test_{group}_summary.log",
        )
        _run(
            [
                str(PYTHON), "scripts/aggregate_fullfield_seed_results.py",
                "--full-fields", str(table.with_suffix(".json")),
                "--output", str(RUN_ROOT / f"formal_test_{group}_multiseed_aggregate_v1.json"),
            ],
            LOG_ROOT / f"formal_test_{group}_aggregate.log",
        )

    _run(
        [
            str(PYTHON), "scripts/compare_multiseed_fullfield.py",
            "--reference", str(RUN_ROOT / "formal_test_main_geometry_multiseed_v1.json"),
            "--candidate", f"unet={RUN_ROOT / 'formal_test_unet_geometry_multiseed_v1.json'}",
            "--candidate", f"fno={RUN_ROOT / 'formal_test_fno_geometry_multiseed_v1.json'}",
            "--seeds", "20260904", "20260905", "20260906",
            "--output", str(RUN_ROOT / "formal_test_main_vs_baselines_paired_v1.json"),
        ],
        LOG_ROOT / "formal_test_main_vs_baselines_paired_v1.log",
    )
    _run(
        [str(PYTHON), "scripts/lock_formal_test_results.py"],
        LOG_ROOT / "formal_test_result_lock_v1.log",
    )
    progress["status"] = "complete"
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_progress(progress)


if __name__ == "__main__":
    main()
