#!/usr/bin/env python3
"""Generate the frozen UrbanTALES split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402
from urbantales_ml.splits import build_manifest, save_manifest, validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "configs" / "splits" / "splits_v1.json"
    )
    args = parser.parse_args()
    catalog = UrbanTalesCatalog(args.root)
    manifest = build_manifest(catalog, args.seed)
    errors = validate_manifest(catalog, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    save_manifest(manifest, args.output)
    print(f"wrote={args.output}")
    print(f"cases={len(catalog.cases)} seed={args.seed}")
    for name, protocol in manifest["protocols"].items():
        if all(part in protocol for part in ("train", "val", "test")):
            print(
                f"{name}: train={len(protocol['train'])} "
                f"val={len(protocol['val'])} test={len(protocol['test'])}"
            )


if __name__ == "__main__":
    main()

