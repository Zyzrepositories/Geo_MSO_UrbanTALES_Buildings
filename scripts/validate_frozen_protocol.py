#!/usr/bin/env python3
"""Verify that the frozen evaluation protocol still matches its source artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.protocol import validate_frozen_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/evaluation/frozen_protocol_v1.yaml",
    )
    args = parser.parse_args()
    errors = validate_frozen_protocol(ROOT, args.protocol)
    result = {"status": "ok" if not errors else "error", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
