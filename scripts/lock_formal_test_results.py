#!/usr/bin/env python3
"""Lock the complete, predeclared formal labelled-test suite outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/evaluation/formal_test_suite_protocol_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/evaluation/formal_test_result_lock_v1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite formal test result lock: {args.output}")

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    manifest_path = _project_path(protocol["job_manifest"]["path"])
    if _sha256(manifest_path) != protocol["job_manifest"]["sha256"]:
        raise ValueError("Formal test manifest hash mismatch")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = manifest["jobs"]
    records: list[dict[str, Any]] = []
    for job in jobs:
        release_path = _project_path(job["release"])
        marker_path = Path(str(release_path.resolve()) + ".used.json")
        output_path = Path(job["output"])
        summary_path = output_path / "summary.json"
        case_metrics_path = output_path / "case_metrics.jsonl"
        for path in (release_path, marker_path, summary_path, case_metrics_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing formal test artifact: {path}")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if release.get("status") != "locked" or summary.get("status") != "ok":
            raise ValueError(f"Incomplete formal test job: {job['id']}")
        if summary.get("partition") != "test" or summary.get("protocol") != job["scope"]:
            raise ValueError(f"Formal test summary scope mismatch: {job['id']}")
        if _sha256(release_path) != marker.get("release_sha256"):
            raise ValueError(f"Consumption marker mismatch: {job['id']}")
        if str(output_path.resolve()) != str(Path(marker["output_directory"]).resolve()):
            raise ValueError(f"Consumption output mismatch: {job['id']}")
        records.append(
            {
                "id": job["id"],
                "group": job["group"],
                "scope": job["scope"],
                "seed": job["seed"],
                "release_path": str(release_path),
                "release_sha256": _sha256(release_path),
                "consumption_marker_path": str(marker_path),
                "consumption_marker_sha256": _sha256(marker_path),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "case_metrics_path": str(case_metrics_path),
                "case_metrics_sha256": _sha256(case_metrics_path),
                "case_count": int(summary["case_count"]),
                "macro_case_metrics": summary["macro_case_metrics"],
            }
        )

    run_root = Path(
        os.environ.get("URBANTALES_RUN_ROOT", str(ROOT / "runs"))
    ).expanduser()
    result_files = {
        "main_geometry": run_root / "formal_test_main_geometry_multiseed_aggregate_v1.json",
        "unet_geometry": run_root / "formal_test_unet_geometry_multiseed_aggregate_v1.json",
        "fno_geometry": run_root / "formal_test_fno_geometry_multiseed_aggregate_v1.json",
        "main_city": run_root / "formal_test_main_city_multiseed_aggregate_v1.json",
        "main_wind": run_root / "formal_test_main_wind_multiseed_aggregate_v1.json",
        "paired_geometry": run_root / "formal_test_main_vs_baselines_paired_v1.json",
    }
    result_artifacts = {}
    for label, path in result_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing formal aggregate: {path}")
        result_artifacts[label] = {
            "path": str(path),
            "sha256": _sha256(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        }

    payload = {
        "schema_version": 1,
        "lock_id": "urbantales_formal_labelled_test_results_v1",
        "status": "locked",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_protocol_path": str(args.protocol.resolve()),
        "formal_test_protocol_sha256": _sha256(args.protocol),
        "job_manifest_path": str(manifest_path.resolve()),
        "job_manifest_sha256": _sha256(manifest_path),
        "job_count": len(records),
        "all_predeclared_releases_consumed_once": True,
        "model_selection_on_test": False,
        "jobs": records,
        "result_artifacts": result_artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
