#!/usr/bin/env python3
"""Validate the frozen one-checkpoint/one-output labelled-test execution protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evaluation/test_execution_protocol_v2.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    errors: list[str] = []
    protocol = _load(PROTOCOL)
    if protocol.get("schema_version") != 2 or protocol.get("status") != "frozen":
        errors.append("test execution protocol must be frozen schema version 2")

    for key in ("evaluation_protocol", "final_primary_model_lock"):
        reference = protocol[key]
        path = ROOT / reference["path"]
        digest = _sha256(path)
        if digest != reference["sha256"]:
            errors.append(f"{key} hash mismatch: {digest} != {reference['sha256']}")

    for key in ("release_creator", "complete_field_evaluator"):
        reference = protocol["implementation"][key]
        path = ROOT / reference["path"]
        digest = _sha256(path)
        if digest != reference["sha256"]:
            errors.append(f"{key} implementation hash mismatch: {digest} != {reference['sha256']}")

    evaluation = _load(ROOT / protocol["evaluation_protocol"]["path"])
    job = protocol["complete_field_accuracy"]
    if int(job["stride_pixels"]) != int(evaluation["field_reconstruction"]["stride_pixels"]):
        errors.append("test stride differs from frozen field reconstruction")
    if job.get("partition") != "test":
        errors.append("labelled accuracy job must use the test partition")
    if job.get("explicit_case_ids") != "prohibited" or job.get("max_cases") != "prohibited":
        errors.append("partial or explicit test selection must be prohibited")
    if not job.get("checkpoint_only"):
        errors.append("v2 release must be checkpoint-only")

    scopes = set(protocol["release"]["allowed_scopes"])
    if scopes != {"geometry_grouped", "city_grouped", "wind_held_out"}:
        errors.append(f"unexpected allowed scopes: {sorted(scopes)}")
    if protocol["selection_safeguards"].get("test_targets_read_during_selection") is not False:
        errors.append("test-target selection safeguard is not false")

    result = {"status": "ok" if not errors else "error", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
