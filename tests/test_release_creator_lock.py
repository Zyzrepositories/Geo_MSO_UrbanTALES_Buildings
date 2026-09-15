from __future__ import annotations

from scripts.create_test_release import _training_lock_authorizes


def test_final_primary_lock_authorizes_only_exact_geometry_artifact() -> None:
    lock = {
        "lock_id": "urbantales_final_primary_model_v1",
        "selected_config": {"sha256": "config-a"},
        "selected_checkpoints": [{"run": "run-a", "sha256": "checkpoint-a"}],
    }
    assert _training_lock_authorizes(
        lock,
        scope="geometry_grouped",
        run="run-a",
        config_sha256="config-a",
        checkpoint_sha256="checkpoint-a",
    )
    assert not _training_lock_authorizes(
        lock,
        scope="city_grouped",
        run="run-a",
        config_sha256="config-a",
        checkpoint_sha256="checkpoint-a",
    )


def test_auxiliary_lock_requires_exact_record() -> None:
    lock = {
        "lock_id": "urbantales_auxiliary_generalization_results_v1",
        "protocols": {
            "city_grouped": {
                "runs": [
                    {
                        "protocol": "city_grouped",
                        "run": "run-b",
                        "config_sha256": "config-b",
                        "checkpoint_sha256": "checkpoint-b",
                    }
                ]
            }
        },
    }
    assert _training_lock_authorizes(
        lock,
        scope="city_grouped",
        run="run-b",
        config_sha256="config-b",
        checkpoint_sha256="checkpoint-b",
    )
    assert not _training_lock_authorizes(
        lock,
        scope="city_grouped",
        run="run-b",
        config_sha256="config-b",
        checkpoint_sha256="tampered",
    )
