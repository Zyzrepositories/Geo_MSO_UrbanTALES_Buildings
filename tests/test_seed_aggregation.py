from __future__ import annotations

import math

import pytest

from scripts.aggregate_seed_results import _aggregate


def test_three_seed_aggregate_includes_student_t_confidence_interval() -> None:
    result = _aggregate(
        [{"run": "a", "metric": 1.0}, {"run": "b", "metric": 2.0}, {"run": "c", "metric": 3.0}],
        {"run"},
    )
    metric = result["metrics"]["metric"]
    expected_half_width = 4.302652729696142 * 1.0 / math.sqrt(3.0)
    assert metric["mean"] == pytest.approx(2.0)
    assert metric["sample_std"] == pytest.approx(1.0)
    assert metric["confidence_95"]["half_width"] == pytest.approx(expected_half_width)
    assert metric["confidence_95"]["degrees_of_freedom"] == 2


def test_one_seed_aggregate_has_no_confidence_interval() -> None:
    result = _aggregate([{"metric": 1.5}], set())
    assert result["metrics"]["metric"]["confidence_95"] is None
