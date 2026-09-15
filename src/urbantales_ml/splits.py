"""Deterministic, leakage-aware data split protocols."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .catalog import CaseRecord, UrbanTalesCatalog


PARTITIONS = ("train", "val", "test")


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _partition_groups(
    cases: Sequence[CaseRecord],
    group_key: Callable[[CaseRecord], str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        groups[group_key(case)].append(case)
    total = len(cases)
    targets = dict(zip(PARTITIONS, (ratio * total for ratio in ratios)))
    assigned = {name: [] for name in PARTITIONS}
    counts = {name: 0 for name in PARTITIONS}
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), _rank(seed, item[0])))
    for group, members in ordered:
        destination = max(
            PARTITIONS,
            key=lambda name: (
                (targets[name] - counts[name]) / max(targets[name], 1.0),
                _rank(seed, f"{group}:{name}"),
            ),
        )
        assigned[destination].extend(member.case_id for member in members)
        counts[destination] += len(members)
    return {name: sorted(values) for name, values in assigned.items()}


def _merge(parts: Iterable[dict[str, list[str]]]) -> dict[str, list[str]]:
    result = {name: [] for name in PARTITIONS}
    for part in parts:
        for name in PARTITIONS:
            result[name].extend(part[name])
    return {name: sorted(values) for name, values in result.items()}


def _iid_case_split(cases: Sequence[CaseRecord], seed: int) -> dict:
    ordered = sorted((case.case_id for case in cases), key=lambda value: _rank(seed, value))
    n_train = round(0.70 * len(ordered))
    n_val = round(0.15 * len(ordered))
    return {
        "purpose": "reference_only_quantify_random_case_split_optimism",
        "group_key": None,
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def _geometry_grouped(cases: Sequence[CaseRecord], seed: int) -> dict:
    parts = []
    for family in ("idealized", "realistic"):
        family_cases = [case for case in cases if case.family == family]
        parts.append(
            _partition_groups(
                family_cases,
                lambda case: f"{case.family}:{case.site_key.casefold()}",
                (0.70, 0.15, 0.15),
                seed + (0 if family == "idealized" else 1),
            )
        )
    result = _merge(parts)
    result.update(
        {
            "purpose": "primary_iid_geometry_generalization_without_site_leakage",
            "group_key": "family + site_key; Val/base share site_key",
        }
    )
    return result


def _city_grouped(cases: Sequence[CaseRecord], seed: int) -> dict:
    realistic = [case for case in cases if case.family == "realistic"]
    result = _partition_groups(
        realistic,
        lambda case: f"{case.country.casefold()}|{case.city.casefold()}",
        (0.70, 0.15, 0.15),
        seed,
    )
    result.update(
        {
            "purpose": "unseen_realistic_city_generalization",
            "group_key": "country + city",
            "excluded": sorted(case.case_id for case in cases if case.family == "idealized"),
        }
    )
    return result


def _wind_held_out(cases: Sequence[CaseRecord], seed: int) -> dict:
    eligible_cases = [case for case in cases if not case.is_val_resolution]
    by_site: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in eligible_cases:
        by_site[f"{case.family}:{case.site_key.casefold()}"].append(case)
    eligible_sites = {
        site: members
        for site, members in by_site.items()
        if len({round(case.wind_angle_deg, 6) for case in members}) >= 2
    }
    val_sites = {
        site
        for site in eligible_sites
        if int(_rank(seed, site)[:8], 16) % 5 == 0
    }
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    heldout_angles: dict[str, float] = {}
    for site, members in sorted(eligible_sites.items()):
        angles = sorted({round(case.wind_angle_deg, 6) for case in members})
        heldout = min(angles, key=lambda angle: _rank(seed, f"{site}:{angle}"))
        heldout_angles[site] = heldout
        for case in members:
            if round(case.wind_angle_deg, 6) == heldout:
                (val if site in val_sites else test).append(case.case_id)
            else:
                train.append(case.case_id)
    used = set(train) | set(val) | set(test)
    return {
        "purpose": "unseen_direction_on_seen_geometry",
        "group_key": "family + site_key",
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "heldout_angle_by_site": heldout_angles,
        "excluded_single_direction_or_val": sorted(
            case.case_id for case in cases if case.case_id not in used
        ),
    }


def _transfer(cases: Sequence[CaseRecord], seed: int) -> dict:
    idealized = [case for case in cases if case.family == "idealized"]
    realistic = [case for case in cases if case.family == "realistic"]
    source = _partition_groups(
        idealized,
        lambda case: case.site_key.casefold(),
        (0.80, 0.10, 0.10),
        seed,
    )
    target = _partition_groups(
        realistic,
        lambda case: f"{case.country.casefold()}|{case.city.casefold()}",
        (0.70, 0.15, 0.15),
        seed + 1,
    )
    realistic_by_id = {case.case_id: case for case in realistic}
    train_cities = sorted(
        {
            f"{realistic_by_id[case_id].country.casefold()}|"
            f"{realistic_by_id[case_id].city.casefold()}"
            for case_id in target["train"]
        },
        key=lambda city: _rank(seed + 2, city),
    )
    few_shot = {}
    for fraction in (10, 25, 50, 100):
        count = max(1, round(len(train_cities) * fraction / 100))
        selected = set(train_cities[:count])
        few_shot[str(fraction)] = sorted(
            case_id
            for case_id in target["train"]
            if (
                f"{realistic_by_id[case_id].country.casefold()}|"
                f"{realistic_by_id[case_id].city.casefold()}"
            )
            in selected
        )
    return {
        "purpose": "idealized_pretraining_to_few_shot_realistic_transfer",
        "source_idealized": source,
        "target_realistic": {
            **target,
            "few_shot_train_percent": few_shot,
            "group_key": "country + city",
        },
    }


def _resolution_pairs(root: Path) -> dict:
    path = root / "reports" / "literature_review" / "val_resolution_pairs.csv"
    import csv

    with path.open(newline="", encoding="utf-8-sig") as handle:
        pairs = list(csv.DictReader(handle))
    return {
        "purpose": "paired_1m_to_0.5m_resolution_evaluation_not_an_ml_validation_split",
        "pairs": pairs,
        "seen_geometry_mode": "train_on_base_1m_then_evaluate_paired_val_0.5m",
        "unseen_geometry_mode": "exclude_both_pair_members_from_training_then_evaluate_both",
    }


def build_manifest(catalog: UrbanTalesCatalog, seed: int = 20260904) -> dict:
    cases = list(catalog.cases)
    return {
        "schema_version": 1,
        "seed": seed,
        "case_count": len(cases),
        "notes": [
            "All paths are resolved through the canonical catalog.",
            "The iid_case_reference protocol is not the primary reported result.",
            "Val-* is a resolution study, never an implicit ML validation set.",
        ],
        "protocols": {
            "iid_case_reference": _iid_case_split(cases, seed),
            "geometry_grouped": _geometry_grouped(cases, seed + 10),
            "city_grouped": _city_grouped(cases, seed + 20),
            "wind_held_out": _wind_held_out(cases, seed + 30),
            "domain_transfer": _transfer(cases, seed + 40),
            "val_resolution": _resolution_pairs(catalog.root),
        },
    }


def validate_manifest(catalog: UrbanTalesCatalog, manifest: dict) -> list[str]:
    errors: list[str] = []
    known = {case.case_id for case in catalog.cases}
    by_id = {case.case_id: case for case in catalog.cases}
    for protocol_name in ("iid_case_reference", "geometry_grouped", "city_grouped"):
        protocol = manifest["protocols"][protocol_name]
        parts = [set(protocol[name]) for name in PARTITIONS]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            errors.append(f"{protocol_name}: case overlap")
        if not set.union(*parts) <= known:
            errors.append(f"{protocol_name}: unknown case")

    geometry = manifest["protocols"]["geometry_grouped"]
    geometry_groups = {
        name: {
            f"{by_id[case_id].family}:{by_id[case_id].site_key.casefold()}"
            for case_id in geometry[name]
        }
        for name in PARTITIONS
    }
    if any(
        geometry_groups[PARTITIONS[i]] & geometry_groups[PARTITIONS[j]]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        errors.append("geometry_grouped: site group leakage")

    city = manifest["protocols"]["city_grouped"]
    city_groups = {
        name: {
            f"{by_id[case_id].country.casefold()}|{by_id[case_id].city.casefold()}"
            for case_id in city[name]
        }
        for name in PARTITIONS
    }
    if any(
        city_groups[PARTITIONS[i]] & city_groups[PARTITIONS[j]]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        errors.append("city_grouped: city leakage")

    wind = manifest["protocols"]["wind_held_out"]
    train_sites = {
        f"{by_id[case_id].family}:{by_id[case_id].site_key.casefold()}"
        for case_id in wind["train"]
    }
    for name in ("val", "test"):
        for case_id in wind[name]:
            site = f"{by_id[case_id].family}:{by_id[case_id].site_key.casefold()}"
            if site not in train_sites:
                errors.append(f"wind_held_out: unseen site {site} in {name}")
            heldout = round(float(wind["heldout_angle_by_site"][site]), 6)
            if round(by_id[case_id].wind_angle_deg, 6) != heldout:
                errors.append(f"wind_held_out: {case_id} is not at held-out angle")
    for case_id in wind["train"]:
        site = f"{by_id[case_id].family}:{by_id[case_id].site_key.casefold()}"
        heldout = round(float(wind["heldout_angle_by_site"][site]), 6)
        if round(by_id[case_id].wind_angle_deg, 6) == heldout:
            errors.append(f"wind_held_out: held-out angle leaked into train for {site}")

    transfer = manifest["protocols"]["domain_transfer"]
    for scope, expected_family in (
        ("source_idealized", "idealized"),
        ("target_realistic", "realistic"),
    ):
        selected = transfer[scope]
        parts = [set(selected[name]) for name in PARTITIONS]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            errors.append(f"domain_transfer/{scope}: case overlap")
        if any(by_id[case_id].family != expected_family for ids in parts for case_id in ids):
            errors.append(f"domain_transfer/{scope}: wrong data family")
    source = transfer["source_idealized"]
    source_groups = {
        name: {by_id[case_id].site_key.casefold() for case_id in source[name]}
        for name in PARTITIONS
    }
    if any(
        source_groups[PARTITIONS[i]] & source_groups[PARTITIONS[j]]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        errors.append("domain_transfer/source_idealized: geometry leakage")
    target = transfer["target_realistic"]
    target_groups = {
        name: {
            f"{by_id[case_id].country.casefold()}|{by_id[case_id].city.casefold()}"
            for case_id in target[name]
        }
        for name in PARTITIONS
    }
    if any(
        target_groups[PARTITIONS[i]] & target_groups[PARTITIONS[j]]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        errors.append("domain_transfer/target_realistic: city leakage")
    previous: set[str] = set()
    target_train = set(target["train"])
    for fraction in ("10", "25", "50", "100"):
        current = set(target["few_shot_train_percent"][fraction])
        if not previous <= current or not current <= target_train:
            errors.append(f"domain_transfer: invalid nested few-shot subset {fraction}")
        previous = current
    if previous != target_train:
        errors.append("domain_transfer: 100 percent subset is not the full target train set")

    for pair in manifest["protocols"]["val_resolution"]["pairs"]:
        if pair["val_case"] not in known or pair["base_case"] not in known:
            errors.append(f"val_resolution: unknown pair {pair}")
    return errors


def save_manifest(manifest: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
