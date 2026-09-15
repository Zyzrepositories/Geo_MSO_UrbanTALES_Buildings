from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_latency import _verify_latency_model_lock


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    protocol = tmp_path / "protocol.yaml"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "best.pt"
    protocol.write_text("status: frozen\n", encoding="utf-8")
    config.write_text("model: locked\n", encoding="utf-8")
    checkpoint.write_bytes(b"locked checkpoint")
    lock = tmp_path / "model-lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "locked",
                "model_and_hyperparameter_selection": "closed",
                "evaluation_protocol": {"sha256": _sha(protocol)},
                "selected_config": {"sha256": _sha(config)},
                "latency_reference_seed": 7,
                "selected_checkpoints": [{"seed": 7, "sha256": _sha(checkpoint)}],
            }
        ),
        encoding="utf-8",
    )
    return {"lock": lock, "protocol": protocol, "config": config, "checkpoint": checkpoint}


def test_latency_inputs_are_bound_to_final_model_lock(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _verify_latency_model_lock(
        paths["lock"], paths["protocol"], paths["config"], paths["checkpoint"]
    )
    assert result["latency_reference_seed"] == 7


@pytest.mark.parametrize("name", ["protocol", "config", "checkpoint"])
def test_latency_lock_rejects_changed_input(tmp_path: Path, name: str) -> None:
    paths = _fixture(tmp_path)
    paths[name].write_bytes(paths[name].read_bytes() + b" changed")
    with pytest.raises(ValueError, match="hash differs"):
        _verify_latency_model_lock(
            paths["lock"], paths["protocol"], paths["config"], paths["checkpoint"]
        )
