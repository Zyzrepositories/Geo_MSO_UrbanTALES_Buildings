#!/usr/bin/env python3
"""Plot validation evidence used to select the formal UrbanTALES configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LABELS = {
    "multitask_unet": "U-Net",
    "fno2d": "FNO",
    "geo_multiscale_operator": "Geo-MSO (proposed)",
}

RUN_LABELS = {
    "screen5_main_v1": "Geo-MSO seed 20260904",
    "formal_main_seed20260905_v1": "Geo-MSO seed 20260905",
    "formal_main_seed20260906_v1": "Geo-MSO seed 20260906",
    "finetune_realistic_25_lr3e4_v1": "Full fine-tune (3e-4)",
    "scratch_realistic_25_lr3e4_v1": "Scratch (3e-4)",
    "finetune_realistic_25_headfilm_v1": "FiLM + heads only",
    "finetune_realistic_25_discriminative_v1": "Discriminative LR",
    "ablation_operator1_seed20260904_v1": "One operator scale",
    "ablation_no_sdf_seed20260904_v1": "No SDF channel",
    "ablation_no_film_seed20260904_v1": "No FiLM",
    "ablation_no_gradient_seed20260904_v1": "No gradient loss",
    "ablation_no_uncertainty_seed20260904_v1": "No uncertainty head/loss",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    specifications = (
        ("val", "loss", "Validation composite loss", axes[0, 0]),
        ("train", "total", "Training composite loss", axes[0, 1]),
        ("val", "Uped/physical_m_s/mae", "Validation $U_{ped}$ MAE (m/s)", axes[1, 0]),
        (
            "val",
            "TKEped/physical_m2_s2/mae",
            "Validation TKE$_{ped}$ MAE (m$^2$/s$^2$)",
            axes[1, 1],
        ),
    )
    for run_name in args.runs:
        run_dir = args.root / run_name
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        label = RUN_LABELS.get(
            run_name,
            LABELS.get(resolved["model"]["name"], resolved["model"]["name"]),
        )
        epochs = np.asarray([row["epoch"] for row in rows])
        for group, metric, title, axis in specifications:
            values = np.asarray([row[group][metric] for row in rows], dtype=np.float64)
            axis.plot(epochs, values, linewidth=1.8, label=label)
            if group == "val" and metric == "loss":
                best = int(values.argmin())
                axis.scatter(epochs[best], values[best], s=32, zorder=3)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
