from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluate_full_fields as evaluator_module
from scripts.evaluate_full_fields import _verify_test_release


PRIMARY_CONFIG = {"data": {"protocol": "geometry_grouped"}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("status: frozen\n", encoding="utf-8")
    release_protocol = tmp_path / "test_protocol.yaml"
    release_protocol.write_text(
        "schema_version: 2\n"
        "status: frozen\n"
        "implementation:\n"
        "  complete_field_evaluator:\n"
        "    path: scripts/evaluate_full_fields.py\n"
        f"    sha256: {_sha(Path(evaluator_module.__file__).resolve())}\n",
        encoding="utf-8",
    )
    config = tmp_path / "model.yaml"
    config.write_text("data:\n  protocol: geometry_grouped\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    output = tmp_path / "test-output"
    release = tmp_path / "release.json"
    marker = tmp_path / "release.json.used.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "release_kind": "urbantales_labelled_test_job",
                "status": "locked",
                "release_protocol_path": str(release_protocol),
                "release_protocol_sha256": _sha(release_protocol),
                "protocol_path": str(protocol),
                "protocol_sha256": _sha(protocol),
                "evaluation_scope": "geometry_grouped",
                "evaluation_partition": "test",
                "selected_config_sha256": _sha(config),
                "selected_checkpoint_sha256": _sha(checkpoint),
                "evaluation_job": {
                    "stride_pixels": 128,
                    "batch_size": 4,
                    "save_arrays": False,
                    "output_directory": str(output),
                },
                "consumption_marker": str(marker),
            }
        ),
        encoding="utf-8",
    )
    return {
        "protocol": protocol,
        "release_protocol": release_protocol,
        "config": config,
        "checkpoint": checkpoint,
        "output": output,
        "release": release,
        "marker": marker,
    }


def _verify(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    arguments = {
        "config_path": paths["config"],
        "checkpoint_path": paths["checkpoint"],
        "output_path": paths["output"],
        "partition": "test",
        "stride_pixels": 128,
        "batch_size": 4,
        "save_arrays": False,
    }
    arguments.update(overrides)
    return _verify_test_release(paths["release"], PRIMARY_CONFIG, **arguments)


def test_labelled_test_evaluation_requires_release() -> None:
    with pytest.raises(ValueError, match="Labelled test evaluation is locked"):
        _verify_test_release(None, PRIMARY_CONFIG)


def test_release_binds_protocol_config_checkpoint_and_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assert _verify(paths)["status"] == "locked"

    paths["protocol"].write_text("status: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen protocol hash mismatch"):
        _verify(paths)


def test_release_rejects_wrong_scope_config_checkpoint_and_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    transfer_config = {
        "data": {"protocol": "domain_transfer", "transfer_scope": "target_realistic"}
    }
    with pytest.raises(ValueError, match="does not authorize evaluation scope"):
        _verify_test_release(
            paths["release"],
            transfer_config,
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            output_path=paths["output"],
            stride_pixels=128,
        )

    paths = _fixture(tmp_path / "config-case")
    wrong_config = tmp_path / "wrong.yaml"
    wrong_config.write_text("different: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config hash"):
        _verify(paths, config_path=wrong_config)

    wrong_checkpoint = tmp_path / "wrong.pt"
    wrong_checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        _verify(paths, checkpoint_path=wrong_checkpoint)

    with pytest.raises(ValueError, match="output directory"):
        _verify(paths, output_path=tmp_path / "other-output")


def test_release_rejects_partial_or_changed_job_and_is_one_time(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Partial or explicit"):
        _verify(paths, max_cases=1)
    with pytest.raises(ValueError, match="stride"):
        _verify(paths, stride_pixels=64)
    with pytest.raises(ValueError, match="baseline"):
        _verify(paths, baseline="zero", checkpoint_path=None)

    _verify(paths, consume=True)
    assert paths["marker"].exists()
    with pytest.raises(ValueError, match="already been consumed"):
        _verify(paths, consume=True)
