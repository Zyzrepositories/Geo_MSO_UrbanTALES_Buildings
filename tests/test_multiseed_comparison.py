from __future__ import annotations

import pytest

from scripts.compare_multiseed_fullfield import compare_rows


def test_paired_comparison_preserves_seed_signs_and_calibration_meaning() -> None:
    reference = [
        {"Uped_mae_m_s": 1.0, "Uped_coverage_90": 0.80},
        {"Uped_mae_m_s": 2.0, "Uped_coverage_90": 0.85},
        {"Uped_mae_m_s": 3.0, "Uped_coverage_90": 0.88},
    ]
    candidate = [
        {"Uped_mae_m_s": 2.0, "Uped_coverage_90": 0.91},
        {"Uped_mae_m_s": 1.0, "Uped_coverage_90": 0.89},
        {"Uped_mae_m_s": 4.0, "Uped_coverage_90": 0.92},
    ]
    result = compare_rows(reference, candidate, [1, 2, 3], metrics=("Uped_mae_m_s",))
    error = result["metrics"]["Uped_mae_m_s"]
    assert error["candidate_minus_reference"]["values"] == [1.0, -1.0, 1.0]
    assert error["reference_lower_error_seed_count"] == 2
    calibration = result["metrics"]["Uped_coverage_90_absolute_calibration_error"]
    assert calibration["candidate"]["mean"] == pytest.approx((0.01 + 0.01 + 0.02) / 3)
    assert calibration["candidate_lower_error_seed_count"] == 2
    assert calibration["ties"] == 1


def test_paired_comparison_rejects_misaligned_seed_count() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        compare_rows([{"x": 1.0}], [{"x": 1.0}], [1, 2], metrics=("x",))
