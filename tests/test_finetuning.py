from __future__ import annotations

import pytest

from urbantales_ml.models import GeoMultiScaleOperator
from urbantales_ml.runner import _build_optimizer


def _model() -> GeoMultiScaleOperator:
    return GeoMultiScaleOperator(
        6,
        ("uped", "vped", "Uped", "TKEped"),
        base_channels=8,
        depth=2,
        operator_levels=2,
        modes=4,
        predict_uncertainty=True,
    )


def test_head_and_film_only_freezes_everything_else() -> None:
    model = _model()
    _, report = _build_optimizer(
        model,
        {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "trainable_patterns": ["operator_blocks.*.film.*", "heads.*"],
        },
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith("heads.") or ".film." in name for name in trainable_names)
    assert report["trainable_parameter_count"] < report["total_parameter_count"]
    assert report["frozen_parameter_count"] > 0


def test_discriminative_groups_are_complete_and_use_requested_rates() -> None:
    model = _model()
    optimizer, report = _build_optimizer(
        model,
        {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "parameter_groups": [
                {
                    "name": "adaptation",
                    "patterns": ["operator_blocks.*.film.*", "heads.*"],
                    "learning_rate": 3e-4,
                },
                {"name": "backbone", "remaining": True, "learning_rate": 3e-5},
            ],
        },
    )
    assert report["trainable_fraction"] == 1.0
    assert sum(group["parameter_count"] for group in report["groups"]) == report[
        "total_parameter_count"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 3e-5]


def test_parameter_group_overlap_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="matches multiple groups"):
        _build_optimizer(
            model,
            {
                "learning_rate": 3e-4,
                "parameter_groups": [
                    {"name": "heads", "patterns": ["heads.*"]},
                    {"name": "all", "patterns": ["*"]},
                ],
            },
        )
