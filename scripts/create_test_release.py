#!/usr/bin/env python3
"""Create the auditable post-selection artifact that unlocks labelled test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _load_mapping(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _continuity_signature(config: dict[str, Any]) -> str:
    ignored_training = {
        "resume_checkpoint",
        "initial_checkpoint",
        "max_epochs_this_run",
        "allow_existing_output",
        "num_workers",
        "pin_memory",
    }
    training = {
        key: value for key, value in config["training"].items() if key not in ignored_training
    }
    data = {
        key: value for key, value in config["data"].items() if key not in {"root", "max_cache_cases"}
    }
    payload = {
        "seed": config["seed"],
        "deterministic": config.get("deterministic", True),
        "data": data,
        "model": config["model"],
        "loss": config["loss"],
        "training": training,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _training_lock_authorizes(
    lock: dict[str, Any],
    *,
    scope: str,
    run: str,
    config_sha256: str,
    checkpoint_sha256: str,
) -> bool:
    lock_id = lock.get("lock_id")
    if lock_id == "urbantales_final_primary_model_v1":
        return (
            scope == "geometry_grouped"
            and lock.get("selected_config", {}).get("sha256") == config_sha256
            and any(
                row.get("run") == run and row.get("sha256") == checkpoint_sha256
                for row in lock.get("selected_checkpoints", [])
            )
        )
    if lock_id not in {
        "urbantales_geometry_multiseed_results_v1",
        "urbantales_auxiliary_generalization_results_v1",
        "urbantales_baseline_multiseed_results_v1",
    }:
        return False

    def matching_record(value: Any) -> bool:
        if isinstance(value, dict):
            if {
                "run",
                "config_sha256",
                "checkpoint_sha256",
            }.issubset(value):
                record_scope = value.get("protocol", "geometry_grouped")
                if (
                    record_scope == scope
                    and value["run"] == run
                    and value["config_sha256"] == config_sha256
                    and value["checkpoint_sha256"] == checkpoint_sha256
                ):
                    return True
            return any(matching_record(item) for item in value.values())
        if isinstance(value, list):
            return any(matching_record(item) for item in value)
        return False

    return matching_record(lock)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/evaluation/frozen_protocol_v1.yaml",
    )
    parser.add_argument(
        "--release-protocol",
        type=Path,
        default=ROOT / "configs/evaluation/test_execution_protocol_v2.yaml",
    )
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--selected-run", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-scope", required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--stride-pixels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite test release: {args.output}")
    protocol_path = args.protocol.expanduser().resolve()
    protocol: dict[str, Any] = _load_mapping(protocol_path)
    if protocol.get("status") != "frozen":
        raise ValueError("Evaluation protocol must be frozen before releasing the test set")
    release_protocol_path = args.release_protocol.expanduser().resolve()
    release_protocol = _load_mapping(release_protocol_path)
    if release_protocol.get("status") != "frozen" or release_protocol.get("schema_version") != 2:
        raise ValueError("Test execution protocol must be frozen schema version 2")
    creator = release_protocol["implementation"]["release_creator"]
    if _sha256(Path(__file__).resolve()) != creator["sha256"]:
        raise ValueError("Release creator implementation differs from the frozen test protocol")
    expected_evaluation = release_protocol["evaluation_protocol"]
    if _sha256(protocol_path) != expected_evaluation["sha256"]:
        raise ValueError("Evaluation protocol is not the one frozen by the test execution protocol")
    if args.evaluation_scope not in release_protocol["release"]["allowed_scopes"]:
        raise ValueError(f"Test scope is not allowed: {args.evaluation_scope}")
    job = release_protocol["complete_field_accuracy"]
    if args.stride_pixels != int(job["stride_pixels"]):
        raise ValueError("Requested stride differs from the frozen test job")
    if args.batch_size != int(job["batch_size"]):
        raise ValueError("Requested batch size differs from the frozen test job")
    if args.save_arrays != bool(job["save_arrays"]):
        raise ValueError("Requested array-saving setting differs from the frozen test job")

    training_lock_path = args.training_lock.expanduser().resolve()
    training_lock = _load_mapping(training_lock_path)
    lock_status = training_lock.get("status", training_lock.get("decision_status"))
    if lock_status not in release_protocol["release"]["allowed_training_lock_statuses"]:
        raise ValueError(f"Training lock is not frozen/locked: {training_lock_path}")

    config_path = args.config.expanduser().resolve()
    config = _load_mapping(config_path)
    data = config["data"]
    scope = (
        f"domain_transfer:{data['transfer_scope']}"
        if data["protocol"] == "domain_transfer"
        else str(data["protocol"])
    )
    if scope != args.evaluation_scope:
        raise ValueError(f"Config scope differs from requested scope: {scope}")
    convergence_path = args.convergence.expanduser().resolve()
    rows = json.loads(convergence_path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row["run"] == args.selected_run]
    if len(matches) != 1:
        raise ValueError(f"Selected run must appear exactly once: {args.selected_run}")
    evidence = matches[0]
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != int(evidence["best_epoch"]):
        raise ValueError(
            "Selected checkpoint epoch does not equal the validation-curve best epoch: "
            f"{checkpoint['epoch']} != {evidence['best_epoch']}"
        )
    checkpoint_val = float(checkpoint["validation"]["loss"])
    if abs(checkpoint_val - float(evidence["best_val_loss"])) > 1e-10:
        raise ValueError("Selected checkpoint validation loss does not match convergence evidence")
    if _continuity_signature(checkpoint["config"]) != _continuity_signature(config):
        raise ValueError("Checkpoint training trajectory differs from the release config")

    config_sha256 = _sha256(config_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    if not _training_lock_authorizes(
        training_lock,
        scope=scope,
        run=args.selected_run,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    ):
        raise ValueError("Training lock does not authorize this exact run/config/checkpoint")

    evaluation_output = args.evaluation_output.expanduser().resolve()
    if evaluation_output.exists():
        raise FileExistsError(f"Test output must not exist when a release is created: {evaluation_output}")

    split_relative = Path(protocol["artifacts"]["split_manifest"]["path"])
    split_path = ROOT / split_relative
    payload = {
        "schema_version": 2,
        "release_kind": release_protocol["release"]["release_kind"],
        "status": "locked",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_protocol_path": _project_path(release_protocol_path),
        "release_protocol_sha256": _sha256(release_protocol_path),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": _project_path(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "split_manifest_path": str(split_relative).replace("\\", "/"),
        "split_manifest_sha256": _sha256(split_path),
        "selection_rule": protocol["model_selection"]["checkpoint_rule"],
        "training_lock_path": _project_path(training_lock_path),
        "training_lock_sha256": _sha256(training_lock_path),
        "evaluation_scope": args.evaluation_scope,
        "evaluation_partition": "test",
        "selected_run": args.selected_run,
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_checkpoint_epoch": int(checkpoint["epoch"]),
        "selected_config_path": _project_path(config_path),
        "selected_config_sha256": config_sha256,
        "selected_model": checkpoint["config"]["model"],
        "selected_training": checkpoint["config"]["training"],
        "validation_evidence_path": str(convergence_path),
        "validation_evidence_sha256": _sha256(convergence_path),
        "validation_evidence": evidence,
        "evaluation_job": {
            "stride_pixels": args.stride_pixels,
            "batch_size": args.batch_size,
            "save_arrays": args.save_arrays,
            "explicit_case_ids": False,
            "max_cases": None,
            "output_directory": str(evaluation_output),
        },
        "consumption_marker": str(args.output.expanduser().resolve()) + ".used.json",
        "test_targets_read_during_selection": False,
    }
    expected_split = protocol["artifacts"]["split_manifest"]["sha256"]
    if payload["split_manifest_sha256"] != expected_split:
        raise ValueError("Split manifest hash differs from the frozen protocol")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
