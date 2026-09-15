"""Validation helpers for immutable experiment and evaluation protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(case_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(case_ids) + "\n").encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _check_partition(
    errors: list[str],
    name: str,
    actual_ids: Sequence[str],
    expected: dict[str, Any],
) -> None:
    if len(actual_ids) != int(expected["count"]):
        errors.append(f"{name}: expected {expected['count']} cases, got {len(actual_ids)}")
    actual_hash = ordered_ids_sha256(actual_ids)
    if actual_hash != expected["ordered_ids_sha256"]:
        errors.append(f"{name}: ordered case-ID hash changed to {actual_hash}")


def validate_frozen_protocol(root: Path, protocol_path: Path) -> list[str]:
    """Return all protocol-integrity errors without mutating any artifact."""
    root = root.resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    errors: list[str] = []
    if protocol.get("status") != "frozen":
        errors.append("Protocol status is not frozen")
    artifacts = protocol["artifacts"]
    for name, expected in artifacts.items():
        path = root / expected["path"]
        if not path.exists():
            errors.append(f"{name}: missing artifact {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            errors.append(f"{name}: file hash changed to {actual_hash}")
    split_path = root / artifacts["split_manifest"]["path"]
    if not split_path.exists():
        return errors
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    frozen = protocol["partitions"]
    for name in ("geometry_grouped", "city_grouped", "wind_held_out"):
        for partition in ("train", "val", "test"):
            _check_partition(
                errors,
                f"{name}.{partition}",
                manifest["protocols"][name][partition],
                frozen[name][partition],
            )
    transfer = manifest["protocols"]["domain_transfer"]
    for scope in ("source_idealized", "target_realistic"):
        for partition in ("val", "test"):
            _check_partition(
                errors,
                f"domain_transfer.{scope}.{partition}",
                transfer[scope][partition],
                frozen["domain_transfer"][scope][partition],
            )
    pairs = manifest["protocols"]["val_resolution"]["pairs"]
    expected_pairs = frozen["val_resolution"]
    if len(pairs) != int(expected_pairs["pair_count"]):
        errors.append("val_resolution: pair count changed")
    if canonical_json_sha256(pairs) != expected_pairs["canonical_pairs_sha256"]:
        errors.append("val_resolution: canonical pair hash changed")
    test_ids = set(manifest["protocols"]["geometry_grouped"]["test"])
    sentinels = protocol["latency"]["full_field_deployment"]["sentinel_case_ids"]
    outside = sorted(set(sentinels) - test_ids)
    if outside:
        errors.append(f"Latency sentinels outside geometry_grouped.test: {outside}")
    if len(sentinels) != len(set(sentinels)):
        errors.append("Latency sentinel list contains duplicates")
    return errors
