"""Canonical, release-aware UrbanTALES case catalog."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DPDXY_RE = re.compile(
    r"^\s*dpdxy\s*=\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)", re.MULTILINE
)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    metadata_name: str
    mapping_status: str
    family: str
    geometry_key: str
    site_key: str
    city: str
    country: str
    config: str
    wind_label: str
    wind_angle_deg: float
    wind_cos: float
    wind_sin: float
    dx_m: float
    u_tau_m_s: float
    dpdx: float
    dpdy: float
    is_val_resolution: bool
    case_dir: str
    ped_nc: str
    topo: str
    p3d: str
    profile_csv: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: str | None, *, default: float = math.nan) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _normalized_angle(angle: float) -> float:
    value = angle % 360.0
    return 0.0 if math.isclose(value, 360.0, abs_tol=1e-9) else value


def _pressure_gradient(path: Path) -> tuple[float, float]:
    match = DPDXY_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ValueError(f"No dpdxy entry in {path}")
    return float(match.group(1)), float(match.group(2))


class UrbanTalesCatalog:
    """Join local case folders to official metadata without renaming raw data."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._cases = self._load()
        self._by_id = {case.case_id: case for case in self._cases}
        if len(self._by_id) != len(self._cases):
            raise ValueError("Duplicate case_id in catalog")

    @property
    def cases(self) -> tuple[CaseRecord, ...]:
        return tuple(self._cases)

    def get(self, case_id: str) -> CaseRecord:
        return self._by_id[case_id]

    def subset(self, case_ids: Iterable[str]) -> list[CaseRecord]:
        return [self.get(case_id) for case_id in case_ids]

    def _load(self) -> list[CaseRecord]:
        audit_path = self.root / "reports" / "data_audit" / "audit_summary.json"
        map_path = self.root / "reports" / "literature_review" / "case_name_map.csv"
        metadata_path = self.root / "metadata.csv"
        if not all(path.exists() for path in (audit_path, map_path, metadata_path)):
            raise FileNotFoundError("Run the data audit and release-semantics scripts first")

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_cases = {row["case_name"]: row for row in audit["cases"]}
        with map_path.open(newline="", encoding="utf-8-sig") as handle:
            mappings = {row["case_dir"]: row for row in csv.DictReader(handle)}
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            metadata_rows = list(csv.DictReader(handle))
        metadata = {row["NameE"]: row for row in metadata_rows}

        tau_key = next(key for key in metadata_rows[0] if "tau" in key.lower())
        records: list[CaseRecord] = []
        for case_id in sorted(audit_cases):
            audit_case = audit_cases[case_id]
            mapping = mappings[case_id]
            metadata_name = mapping["metadata_name"]
            if metadata_name not in metadata:
                raise KeyError(f"Metadata alias not found for {case_id}: {metadata_name}")
            row = metadata[metadata_name]
            family = audit_case["dataset"]
            family_dir = (
                "Idealized Building Blocks"
                if family == "idealized"
                else "Realistic Urban Neighbourhoods"
            )
            relative_case_dir = Path(family_dir) / case_id
            p3d = relative_case_dir / f"{case_id}_p3d"
            dpdx, dpdy = _pressure_gradient(self.root / p3d)
            wind_label = row["WD"].strip()
            try:
                angle = float(wind_label)
            except ValueError:
                # PALM cases are pressure-gradient driven, hence flow is along -grad(p).
                angle = math.degrees(math.atan2(-dpdy, -dpdx))
            angle = _normalized_angle(angle)
            radians = math.radians(angle)

            records.append(
                CaseRecord(
                    case_id=case_id,
                    metadata_name=metadata_name,
                    mapping_status=mapping["status"],
                    family=family,
                    geometry_key=audit_case["geometry_key"],
                    site_key=audit_case["site_key"],
                    city=row.get("City", "").strip(),
                    country=row.get("Country", "").strip(),
                    config=row.get("Config", "").strip(),
                    wind_label=wind_label,
                    wind_angle_deg=angle,
                    wind_cos=math.cos(radians),
                    wind_sin=math.sin(radians),
                    dx_m=float(row["dxdy"]),
                    u_tau_m_s=float(row[tau_key]),
                    dpdx=dpdx,
                    dpdy=dpdy,
                    is_val_resolution=case_id.startswith("Val-"),
                    case_dir=relative_case_dir.as_posix(),
                    ped_nc=(relative_case_dir / f"{case_id}_ped.nc").as_posix(),
                    topo=(relative_case_dir / f"{case_id}_topo").as_posix(),
                    p3d=p3d.as_posix(),
                    profile_csv=(relative_case_dir / f"profile-{case_id}.csv").as_posix(),
                )
            )
        return records

