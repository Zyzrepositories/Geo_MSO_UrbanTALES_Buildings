#!/usr/bin/env python3
"""Evaluate a checkpoint on overlap-added complete UrbanTALES fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbantales_ml.baselines import (  # noqa: E402
    ConstantFieldModel,
    PatchNearestNeighborModel,
    geometry_condition_features,
)
from urbantales_ml.catalog import UrbanTalesCatalog  # noqa: E402
from urbantales_ml.config import load_config  # noqa: E402
from urbantales_ml.data import (  # noqa: E402
    UrbanTalesPatchDataset,
    ablate_model_input_channels,
)
from urbantales_ml.inference import predict_full_case  # noqa: E402
from urbantales_ml.metrics import batch_metrics  # noqa: E402
from urbantales_ml.runner import _build_model, seed_everything  # noqa: E402


def _evaluation_scope(config: dict[str, Any]) -> str:
    data = config["data"]
    if data["protocol"] == "domain_transfer":
        return f"domain_transfer:{data['transfer_scope']}"
    return str(data["protocol"])


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_test_release(
    path: Path | None,
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    baseline: str | None = None,
    output_path: Path | None = None,
    partition: str = "test",
    stride_pixels: int | None = None,
    batch_size: int = 4,
    save_arrays: bool = False,
    case_ids: list[str] | None = None,
    max_cases: int | None = None,
    consume: bool = False,
) -> dict[str, Any]:
    """Require a post-selection lock artifact before any labelled test evaluation."""
    if path is None:
        raise ValueError(
            "Labelled test evaluation is locked. Pass --test-release only after "
            "validation-based model and hyperparameter selection is complete."
        )
    release_path = path.expanduser().resolve()
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "locked"
        or payload.get("schema_version") != 2
        or payload.get("release_kind") != "urbantales_labelled_test_job"
    ):
        raise ValueError(f"Test release is not locked: {release_path}")
    release_protocol_path = _resolve_recorded_path(payload["release_protocol_path"])
    if _sha256(release_protocol_path) != payload.get("release_protocol_sha256"):
        raise ValueError("Frozen test execution protocol hash mismatch")
    release_protocol = yaml.safe_load(release_protocol_path.read_text(encoding="utf-8"))
    if release_protocol.get("status") != "frozen":
        raise ValueError("Test execution protocol is no longer frozen")
    evaluator = release_protocol["implementation"]["complete_field_evaluator"]
    if _sha256(Path(__file__).resolve()) != evaluator["sha256"]:
        raise ValueError("Complete-field evaluator differs from the frozen test protocol")

    protocol_path = _resolve_recorded_path(payload["protocol_path"])
    digest = _sha256(protocol_path)
    if digest != payload.get("protocol_sha256"):
        raise ValueError(
            f"Frozen protocol hash mismatch: {digest} != {payload.get('protocol_sha256')}"
        )
    scope = _evaluation_scope(config)
    if scope != payload.get("evaluation_scope"):
        raise ValueError(f"Test release does not authorize evaluation scope: {scope}")
    if partition != "test" or payload.get("evaluation_partition") != "test":
        raise ValueError("Test release only authorizes the complete test partition")
    if config_path is None or _sha256(config_path.expanduser().resolve()) != payload.get("selected_config_sha256"):
        raise ValueError("Evaluation config hash does not match the test release")
    if baseline is not None or checkpoint_path is None:
        raise ValueError("This test release authorizes one checkpoint, not a baseline")
    if _sha256(checkpoint_path.expanduser().resolve()) != payload.get("selected_checkpoint_sha256"):
        raise ValueError("Checkpoint hash does not match the test release")

    job = payload["evaluation_job"]
    if stride_pixels != int(job["stride_pixels"]):
        raise ValueError("Evaluation stride does not match the test release")
    if int(batch_size) != int(job["batch_size"]):
        raise ValueError("Evaluation batch size does not match the test release")
    if bool(save_arrays) != bool(job["save_arrays"]):
        raise ValueError("Array-saving setting does not match the test release")
    if case_ids is not None or max_cases is not None:
        raise ValueError("Partial or explicit labelled-test selection is prohibited")
    if output_path is None or output_path.expanduser().resolve() != Path(job["output_directory"]).resolve():
        raise ValueError("Evaluation output directory does not match the test release")

    if consume:
        marker = Path(payload["consumption_marker"]).expanduser().resolve()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker_payload = {
            "release_sha256": _sha256(release_path),
            "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(output_path.expanduser().resolve()),
        }
        try:
            with marker.open("x", encoding="utf-8") as handle:
                json.dump(marker_payload, handle, ensure_ascii=False, indent=2)
        except FileExistsError as exc:
            raise ValueError(f"Test release has already been consumed: {marker}") from exc
    return payload


def _partition_ids(
    manifest: dict[str, Any], config: dict[str, Any], partition: str
) -> list[str]:
    data = config["data"]
    protocol = manifest["protocols"][data["protocol"]]
    if data["protocol"] != "domain_transfer":
        return list(protocol[partition])
    scope = data.get("transfer_scope")
    if scope not in {"source_idealized", "target_realistic"}:
        raise ValueError("domain_transfer evaluation requires a valid transfer_scope")
    selected = protocol[scope]
    if partition == "train" and scope == "target_realistic":
        fraction = str(data.get("few_shot_percent", 100))
        return list(selected["few_shot_train_percent"][fraction])
    return list(selected[partition])


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        for key, value in record["metrics"].items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def _fit_global_means(
    root: Path,
    catalog: UrbanTalesCatalog,
    case_ids: list[str],
    config: dict[str, Any],
    patches_per_case: int,
) -> torch.Tensor:
    data = config["data"]
    fit_dataset = UrbanTalesPatchDataset(
        root,
        catalog.subset(case_ids),
        targets=data["targets"],
        patch_size_m=data["patch_size_m"],
        output_pixels=data["output_pixels"],
        patches_per_case=patches_per_case,
        seed=config["seed"] + 3,
        random_patches=False,
        height_scale_m=data.get("height_scale_m", 50.0),
        sdf_scale_m=data.get("sdf_scale_m", 64.0),
        u_tau_reference_m_s=data.get("u_tau_reference_m_s", 0.21),
        max_cache_cases=data.get("max_cache_cases", 2),
    )
    sums = torch.zeros(len(data["targets"]), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for index in range(len(fit_dataset)):
        sample = fit_dataset[index]
        target = sample["target"].double()
        mask = sample["mask"].bool()
        sums += (target * mask).sum(dim=(-2, -1))
        counts += mask.sum(dim=(-2, -1))
    if (counts == 0).any():
        raise RuntimeError("At least one target has no valid fit pixels")
    return (sums / counts).float()


def _fit_nearest_neighbor_bank(
    root: Path,
    catalog: UrbanTalesCatalog,
    case_ids: list[str],
    config: dict[str, Any],
    patches_per_case: int,
    embedding_pixels: int,
    fill_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build a deterministic patch bank from training cases only."""
    data = config["data"]
    fit_dataset = UrbanTalesPatchDataset(
        root,
        catalog.subset(case_ids),
        targets=data["targets"],
        patch_size_m=data["patch_size_m"],
        output_pixels=data["output_pixels"],
        patches_per_case=patches_per_case,
        seed=config["seed"] + 3,
        random_patches=False,
        height_scale_m=data.get("height_scale_m", 50.0),
        sdf_scale_m=data.get("sdf_scale_m", 64.0),
        u_tau_reference_m_s=data.get("u_tau_reference_m_s", 0.21),
        max_cache_cases=data.get("max_cache_cases", 2),
    )
    geometries: list[torch.Tensor] = []
    conditions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    donor_case_ids: list[str] = []
    for index in range(len(fit_dataset)):
        sample = fit_dataset[index]
        geometry, condition = geometry_condition_features(
            sample["input"].unsqueeze(0), embedding_pixels
        )
        target = sample["target"].float()
        valid = sample["mask"].bool()
        target = torch.where(valid, target, fill_values[:, None, None])
        geometries.append(geometry.squeeze(0))
        conditions.append(condition.squeeze(0))
        targets.append(target)
        donor_case_ids.append(str(sample["case_id"]))
    return (
        torch.stack(geometries),
        torch.stack(conditions),
        torch.stack(targets),
        donor_case_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=Path)
    model_group.add_argument("--baseline", choices=("zero", "global_mean", "geometry_nn"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--test-release",
        type=Path,
        help="Required lock artifact for labelled test evaluation",
    )
    parser.add_argument("--case-ids", nargs="+", help="Explicit case IDs override the partition")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--fit-max-cases", type=int)
    parser.add_argument("--fit-patches-per-case", type=int, default=2)
    parser.add_argument("--nn-embedding-pixels", type=int, default=16)
    parser.add_argument("--nn-boundary-weight", type=float, default=1.0)
    parser.add_argument("--stride-pixels", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {args.output}")
    if args.partition == "test":
        _verify_test_release(
            args.test_release,
            config,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            baseline=args.baseline,
            output_path=args.output,
            partition=args.partition,
            stride_pixels=args.stride_pixels,
            batch_size=args.batch_size,
            save_arrays=args.save_arrays,
            case_ids=args.case_ids,
            max_cases=args.max_cases,
            consume=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    root = Path(config["data"]["root"]).expanduser().resolve()
    catalog = UrbanTalesCatalog(root)
    manifest_path = root / config["data"]["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_ids = list(args.case_ids) if args.case_ids else _partition_ids(
        manifest, config, args.partition
    )
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise ValueError("No cases selected")

    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    fitted_constants = None
    fit_case_ids: list[str] = []
    fit_patch_count = 0
    if args.checkpoint is not None:
        model = _build_model(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("tasks", ())) != tuple(config["data"]["targets"]):
            raise ValueError("Checkpoint targets do not match evaluation config")
        model.load_state_dict(checkpoint["model"], strict=True)
        model_kind = config["model"]["name"]
    else:
        if args.baseline in {"global_mean", "geometry_nn"}:
            fit_case_ids = _partition_ids(manifest, config, "train")
            if args.fit_max_cases is not None:
                fit_case_ids = fit_case_ids[: args.fit_max_cases]
            fitted_constants = _fit_global_means(
                root,
                catalog,
                fit_case_ids,
                config,
                args.fit_patches_per_case,
            )
        else:
            fitted_constants = torch.zeros(len(config["data"]["targets"]))
        if args.baseline == "geometry_nn":
            bank = _fit_nearest_neighbor_bank(
                root,
                catalog,
                fit_case_ids,
                config,
                args.fit_patches_per_case,
                args.nn_embedding_pixels,
                fitted_constants,
            )
            fit_patch_count = len(bank[-1])
            model = PatchNearestNeighborModel(
                *bank,
                embedding_pixels=args.nn_embedding_pixels,
                boundary_weight=args.nn_boundary_weight,
            ).to(device)
        else:
            model = ConstantFieldModel(fitted_constants).to(device)
        model_kind = f"baseline_{args.baseline}"
    model.eval()

    data = config["data"]
    dataset = UrbanTalesPatchDataset(
        root,
        catalog.subset(case_ids),
        targets=data["targets"],
        patch_size_m=data["patch_size_m"],
        output_pixels=data["output_pixels"],
        patches_per_case=1,
        seed=config["seed"] + 2,
        random_patches=False,
        height_scale_m=data.get("height_scale_m", 50.0),
        sdf_scale_m=data.get("sdf_scale_m", 64.0),
        u_tau_reference_m_s=data.get("u_tau_reference_m_s", 0.21),
        max_cache_cases=data.get("max_cache_cases", 2),
    )
    amp = bool(config["training"].get("amp", True))
    evaluation = config.get("evaluation", {})
    records: list[dict[str, Any]] = []
    arrays_dir = args.output / "arrays"
    if args.save_arrays:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    # One unmeasured patch warms up CUDA/cuDNN without reconstructing a field.
    warmup = dataset[0]["input"].unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        model(ablate_model_input_channels(warmup, data.get("zero_model_input_channels")))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for case_index, case in enumerate(dataset.cases):
        if isinstance(model, PatchNearestNeighborModel):
            model.reset_matches()
        output = predict_full_case(
            model,
            dataset,
            case_index,
            device,
            stride_pixels=args.stride_pixels,
            batch_size=args.batch_size,
            amp=amp,
            zero_model_input_channels=data.get("zero_model_input_channels"),
        )
        metrics = batch_metrics(
            output["mean"],
            output["target"],
            output["mask"],
            data["targets"],
            u_tau_m_s=output["u_tau_m_s"],
            input_tensor=output["input"],
            log_scale=output["log_scale"],
            pixel_size_m=output["model_pixel_size_m"],
            sdf_scale_m=float(data.get("sdf_scale_m", 64.0)),
            near_building_distance_m=float(
                evaluation.get("near_building_distance_m", 8.0)
            ),
            high_gradient_quantile=float(evaluation.get("high_gradient_quantile", 0.9)),
            minimum_direction_speed_m_s=float(
                evaluation.get("minimum_direction_speed_m_s", 0.1)
            ),
        )
        record = {
            "case_id": case.case_id,
            "family": case.family,
            "city": case.city,
            "wind_angle_deg": case.wind_angle_deg,
            "source_dx_m": output["source_dx_m"],
            "model_pixel_size_m": output["model_pixel_size_m"],
            "output_shape": output["output_shape"],
            "tile_count": output["tile_count"],
            "stride_pixels": output["stride_pixels"],
            "forward_seconds": output["forward_seconds"],
            "evaluation_pipeline_seconds": output["end_to_end_seconds"],
            "metrics": metrics,
        }
        if isinstance(model, PatchNearestNeighborModel):
            record["nearest_neighbor_match_counts"] = model.match_counts()
        records.append(record)
        if args.save_arrays:
            np.savez_compressed(
                arrays_dir / f"{case.case_id}.npz",
                prediction=output["mean"].squeeze(0).numpy(),
                target=output["target"].squeeze(0).numpy(),
                mask=output["mask"].squeeze(0).numpy(),
                input=output["input"].squeeze(0).numpy(),
                log_scale=(
                    output["log_scale"].squeeze(0).numpy()
                    if output["log_scale"] is not None
                    else np.empty((0,), dtype=np.float32)
                ),
            )

    (args.output / "case_metrics.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "model_kind": model_kind,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "baseline": args.baseline,
        "fitted_constants_dimensionless": (
            fitted_constants.tolist() if fitted_constants is not None else None
        ),
        "fit_case_count": len(fit_case_ids) if args.checkpoint is None else None,
        "fit_patches_per_case": (
            args.fit_patches_per_case
            if args.baseline in {"global_mean", "geometry_nn"}
            else None
        ),
        "fit_patch_count": fit_patch_count if args.baseline == "geometry_nn" else None,
        "nearest_neighbor": (
            {
                "embedding_pixels": args.nn_embedding_pixels,
                "boundary_weight": args.nn_boundary_weight,
                "distance": (
                    "mean_geometry_squared_error + boundary_weight * "
                    "mean_condition_squared_error"
                ),
                "features": ["height", "occupancy", "sdf", "wind_cos", "wind_sin", "u_tau"],
            }
            if args.baseline == "geometry_nn"
            else None
        ),
        "config": str(args.config.resolve()),
        "protocol": data["protocol"],
        "partition": "explicit" if args.case_ids else args.partition,
        "case_count": len(records),
        "case_ids": [record["case_id"] for record in records],
        "macro_case_metrics": _mean_metrics(records),
        "total_tiles": sum(record["tile_count"] for record in records),
        "total_forward_seconds": sum(record["forward_seconds"] for record in records),
        "total_evaluation_pipeline_seconds": sum(
            record["evaluation_pipeline_seconds"] for record in records
        ),
        "timing_note": (
            "forward_seconds excludes CPU patch preparation and transfer; "
            "evaluation_pipeline_seconds includes target reconstruction for scoring"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
