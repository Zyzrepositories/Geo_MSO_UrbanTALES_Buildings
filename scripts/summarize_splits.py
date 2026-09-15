#!/usr/bin/env python
"""Write a human-readable summary of the frozen split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/splits/splits_v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiment_design/SPLITS_V1.md")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = UrbanTalesCatalog(root)
    by_id = {case.case_id: case for case in catalog.cases}

    def family_counts(case_ids: list[str]) -> str:
        counts = Counter(by_id[case_id].family for case_id in case_ids)
        return f"I={counts['idealized']}, R={counts['realistic']}"

    lines = [
        "# UrbanTALES split manifest v1",
        "",
        f"- Frozen seed: `{manifest['seed']}`",
        f"- Canonical cases: `{manifest['case_count']}`",
        "- `Val-*` cases are paired resolution probes, not the ML validation set.",
        "- All primary generalization results use group-disjoint partitions.",
        "",
        "## Flat protocols",
        "",
        "| Protocol | Train | Validation | Test | Grouping |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("iid_case_reference", "geometry_grouped", "city_grouped", "wind_held_out"):
        protocol = manifest["protocols"][name]
        cells = [
            f"{len(protocol[part])} ({family_counts(protocol[part])})" for part in ("train", "val", "test")
        ]
        lines.append(f"| `{name}` | {cells[0]} | {cells[1]} | {cells[2]} | {protocol.get('group_key')} |")

    transfer = manifest["protocols"]["domain_transfer"]
    source = transfer["source_idealized"]
    target = transfer["target_realistic"]
    lines.extend(
        [
            "",
            "## Domain-transfer protocol",
            "",
            f"- Idealized source: train/val/test = {len(source['train'])}/{len(source['val'])}/{len(source['test'])}; grouped by layout.",
            f"- Realistic target: train/val/test = {len(target['train'])}/{len(target['val'])}/{len(target['test'])}; grouped by city.",
            "- Nested realistic training subsets: "
            + ", ".join(
                f"{fraction}%={len(case_ids)} cases"
                for fraction, case_ids in target["few_shot_train_percent"].items()
            )
            + ".",
            "",
            "## Resolution protocol",
            "",
            f"- Paired `Val-*`/base cases: {len(manifest['protocols']['val_resolution']['pairs'])}.",
            "- Seen-geometry mode measures sensitivity to source resolution.",
            "- Unseen-geometry mode excludes both members of each pair before training.",
            "",
            "## Interpretation",
            "",
            "`iid_case_reference` is diagnostic only because different directions of one geometry can leak across sets. "
            "The primary accuracy table should use `geometry_grouped`; city, wind and transfer protocols answer distinct out-of-distribution questions.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

