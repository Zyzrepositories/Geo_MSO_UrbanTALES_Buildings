from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402
from urbantales_ml.protocol import validate_frozen_protocol  # noqa: E402
from urbantales_ml.splits import validate_manifest  # noqa: E402


class CatalogAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            ROOT / "metadata.csv",
            ROOT / "reports/data_audit/audit_summary.json",
            ROOT / "reports/literature_review/case_name_map.csv",
        )
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "UrbanTALES integration metadata are not present: " + ", ".join(missing)
            )
        cls.catalog = UrbanTalesCatalog(ROOT)
        cls.manifest = json.loads(
            (ROOT / "configs/splits/splits_v1.json").read_text(encoding="utf-8")
        )

    def test_catalog_has_all_cases(self) -> None:
        self.assertEqual(len(self.catalog.cases), 538)
        self.assertEqual(
            {case.family for case in self.catalog.cases}, {"idealized", "realistic"}
        )

    def test_flux_directions_are_numeric(self) -> None:
        flux = [case for case in self.catalog.cases if case.wind_label.casefold() == "flux"]
        self.assertEqual(len(flux), 10)
        self.assertTrue(all(0.0 <= case.wind_angle_deg < 360.0 for case in flux))
        self.assertTrue(all(case.dpdx != 0.0 or case.dpdy != 0.0 for case in flux))

    def test_manifest_is_leakage_checked(self) -> None:
        self.assertEqual(validate_manifest(self.catalog, self.manifest), [])

    def test_frozen_evaluation_protocol_matches_manifest(self) -> None:
        protocol = ROOT / "configs/evaluation/frozen_protocol_v1.yaml"
        self.assertEqual(validate_frozen_protocol(ROOT, protocol), [])


if __name__ == "__main__":
    unittest.main()
