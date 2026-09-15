#!/usr/bin/env python3
"""Validate a frozen split manifest against the current local catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402
from urbantales_ml.splits import validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs" / "splits" / "splits_v1.json"
    )
    args = parser.parse_args()
    catalog = UrbanTalesCatalog(args.root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(catalog, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"valid={args.manifest} cases={len(catalog.cases)}")


if __name__ == "__main__":
    main()
