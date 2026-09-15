"""Resolve UrbanTALES release identifiers and special-case semantics.

This script is deliberately read-only with respect to the downloaded dataset. It
writes small, machine-readable tables under reports/literature_review so later
data loaders do not have to repeat release-specific assumptions.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "data_audit" / "audit_summary.json"
OUT = ROOT / "reports" / "literature_review"

# Verified against the folder identifiers currently shown on the official
# UrbanTALES realistic-neighbourhood download page (checked 2026-09-03).
OFFICIAL_PORTAL_ALIASES = {
    "CN-BE-V2_d00": "CN-Bei-V2_d00",
    "CN-CD-V1_d00": "CN-Che-V1_d00",
    "CN-SH-V1_d00": "CN-Sha-V1_d00",
}

DPDXY_RE = re.compile(
    r"^\s*dpdxy\s*=\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)", re.MULTILINE
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    reconciliation = audit["metadata"]["name_reconciliation"]

    name_rows = []
    for row in reconciliation["mappings"]:
        case_dir = row["case_dir"]
        item = {
            "case_dir": case_dir,
            "metadata_name": row.get("metadata_name", ""),
            "status": row["status"],
            "score": row.get("score", ""),
        }
        if case_dir in OFFICIAL_PORTAL_ALIASES:
            item["metadata_name"] = OFFICIAL_PORTAL_ALIASES[case_dir]
            item["status"] = "official_portal_alias"
            item["score"] = ""
            item["source"] = "UrbanTALES official realistic download page"
        else:
            item["source"] = "local audit"
        name_rows.append(item)
    write_csv(
        OUT / "case_name_map.csv",
        ["case_dir", "metadata_name", "status", "score", "source"],
        name_rows,
    )

    val_rows = audit["metadata"]["name_reconciliation"][
        "validation_named_resolution_pairs"
    ]
    flattened_val_rows = [
        {
            "val_case": row["val_case"],
            "base_case": row["base_case"],
            "val_shape": "x".join(map(str, row["val_shape"])),
            "base_shape": "x".join(map(str, row["base_shape"])),
            "val_dx_m": row["val_dx_m"],
            "base_dx_m": row["base_dx_m"],
        }
        for row in val_rows
    ]
    write_csv(
        OUT / "val_resolution_pairs.csv",
        [
            "val_case",
            "base_case",
            "val_shape",
            "base_shape",
            "val_dx_m",
            "base_dx_m",
        ],
        flattened_val_rows,
    )

    with (ROOT / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        metadata = list(csv.DictReader(handle))
    flux_rows = []
    for row in metadata:
        if row["WD"].strip().upper() != "FLUX":
            continue
        case_id = row["NameE"]
        p3d_candidates = list(
            (ROOT / "Realistic Urban Neighbourhoods" / case_id).glob("*_p3d")
        )
        if len(p3d_candidates) != 1:
            raise RuntimeError(f"Expected one p3d for {case_id}, got {p3d_candidates}")
        text = p3d_candidates[0].read_text(encoding="utf-8", errors="replace")
        match = DPDXY_RE.search(text)
        if match is None:
            raise RuntimeError(f"No dpdxy found in {p3d_candidates[0]}")
        grad_x, grad_y = map(float, match.groups())
        # PALM flow is driven down the pressure gradient. The convention was
        # cross-checked against UrbanTALES d00/d45/d90 cases.
        angle = math.degrees(math.atan2(-grad_y, -grad_x)) % 360.0
        if math.isclose(angle, 360.0, abs_tol=1e-9):
            angle = 0.0
        flux_rows.append(
            {
                "case_id": case_id,
                "metadata_WD": row["WD"],
                "dpdx": f"{grad_x:.16g}",
                "dpdy": f"{grad_y:.16g}",
                "derived_flow_to_angle_deg_from_plus_x": f"{angle:.6g}",
                "angle_status": "derived_from_p3d_not_explicit_metadata",
            }
        )
    write_csv(
        OUT / "flux_forcing_directions.csv",
        [
            "case_id",
            "metadata_WD",
            "dpdx",
            "dpdy",
            "derived_flow_to_angle_deg_from_plus_x",
            "angle_status",
        ],
        flux_rows,
    )

    status_counts: dict[str, int] = {}
    for row in name_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "generated_from": str(AUDIT),
        "official_portal_checked_on": "2026-09-03",
        "case_name_mapping_status_counts": status_counts,
        "case_name_mapping_total": len(name_rows),
        "flux_case_count": len(flux_rows),
        "val_pair_count": len(flattened_val_rows),
        "interpretation_boundaries": {
            "FLUX": (
                "site/source category, not a numeric angle; numeric forcing is "
                "derived from each case p3d pressure-gradient vector"
            ),
            "Val_prefix": (
                "strongly supported as a 0.5 m versus 1.0 m grid-resolution "
                "validation pair; the publication does not explicitly define the token"
            ),
        },
    }
    (OUT / "release_semantics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
