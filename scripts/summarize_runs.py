"""Create a compact, dependency-free comparison table from experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _row(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    validation = summary.get("last_validation", {})
    batch_size = int(summary.get("inference_batch_size", 0))
    batch_seconds = float(summary.get("inference_batch_seconds", 0.0))
    peak_bytes = summary.get("peak_gpu_memory_bytes")
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "protocol": config["data"]["protocol"],
        "train_cases": len(summary.get("train_cases", [])),
        "val_cases": len(summary.get("val_cases", [])),
        "validation_samples": summary.get("validation_sample_count"),
        "global_steps": summary.get("global_steps"),
        "parameters": summary.get("parameter_count"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "inference_batch_size": batch_size,
        "inference_batch_ms": 1000.0 * batch_seconds,
        "inference_ms_per_sample_equivalent": (
            1000.0 * batch_seconds / batch_size if batch_size else None
        ),
        "peak_gpu_gib": peak_bytes / (2**30) if peak_bytes is not None else None,
        "val_loss": validation.get("loss"),
        "uped_mae_m_s": validation.get("uped/physical_m_s/mae"),
        "vped_mae_m_s": validation.get("vped/physical_m_s/mae"),
        "Uped_mae_m_s": validation.get("Uped/physical_m_s/mae"),
        "TKEped_mae_m2_s2": validation.get("TKEped/physical_m2_s2/mae"),
        "Uped_near_building_mae_m_s": validation.get("Uped/near_building_m_s/mae"),
        "Uped_high_gradient_mae_m_s": validation.get("Uped/high_gradient_m_s/mae"),
        "direction_mae_deg": validation.get("vector/direction_mae_deg"),
        "horizontal_divergence_error_mae_s_inv": validation.get(
            "horizontal_divergence/error_mae_s_inv"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing run folders")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directory names")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    args = parser.parse_args()

    rows = [_row(args.root / name) for name in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
