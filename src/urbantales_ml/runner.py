"""Config-driven training and smoke-test runner."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .catalog import CaseRecord, UrbanTalesCatalog
from .data import UrbanTalesPatchDataset, ablate_model_input_channels
from .losses import MaskedMultiTaskLoss
from .metrics import batch_metrics
from .models import FNO2d, GeoMultiScaleOperator, LightCNN, MultiTaskUNet


def _training_continuity_signature(config: dict[str, Any]) -> str:
    """Hash settings that must remain identical across a resumed trajectory."""
    ignored_training_keys = {
        "resume_checkpoint",
        "initial_checkpoint",
        "max_epochs_this_run",
        "allow_existing_output",
        "num_workers",
        "pin_memory",
    }
    training = {
        key: value
        for key, value in config["training"].items()
        if key not in ignored_training_keys
    }
    data = {
        key: value
        for key, value in config["data"].items()
        if key not in {"root", "max_cache_cases"}
    }
    payload = {
        "seed": config["seed"],
        "deterministic": config.get("deterministic", True),
        "data": data,
        "model": config["model"],
        "loss": config["loss"],
        "training": training,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _balanced_limit(
    case_ids: Sequence[str], catalog: UrbanTalesCatalog, limit: int | None
) -> list[str]:
    ordered = list(case_ids)
    if limit is None or limit <= 0 or len(ordered) <= limit:
        return ordered
    by_family: dict[str, list[str]] = {"idealized": [], "realistic": []}
    for case_id in ordered:
        by_family[catalog.get(case_id).family].append(case_id)
    selected = []
    cursor = {key: 0 for key in by_family}
    while len(selected) < limit:
        progressed = False
        for family in ("idealized", "realistic"):
            if cursor[family] < len(by_family[family]) and len(selected) < limit:
                selected.append(by_family[family][cursor[family]])
                cursor[family] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _make_dataset(
    root: Path,
    catalog: UrbanTalesCatalog,
    case_ids: Sequence[str],
    config: dict[str, Any],
    *,
    train: bool,
) -> UrbanTalesPatchDataset:
    data = config["data"]
    return UrbanTalesPatchDataset(
        root,
        catalog.subset(case_ids),
        targets=data["targets"],
        patch_size_m=data["patch_size_m"],
        output_pixels=data["output_pixels"],
        patches_per_case=(
            data["patches_per_case_train"] if train else data["patches_per_case_eval"]
        ),
        seed=config["seed"] + (0 if train else 1),
        random_patches=train,
        height_scale_m=data.get("height_scale_m", 50.0),
        sdf_scale_m=data.get("sdf_scale_m", 64.0),
        u_tau_reference_m_s=data.get("u_tau_reference_m_s", 0.21),
        max_cache_cases=data.get("max_cache_cases", 2),
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _select_partition(
    protocol: dict[str, Any], data: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Select flat or transfer-learning train/validation case lists."""
    if data["protocol"] != "domain_transfer":
        return list(protocol["train"]), list(protocol["val"])
    scope = data.get("transfer_scope")
    if scope == "source_idealized":
        selected = protocol[scope]
        return list(selected["train"]), list(selected["val"])
    if scope == "target_realistic":
        selected = protocol[scope]
        fraction = str(data.get("few_shot_percent", 100))
        try:
            train_ids = selected["few_shot_train_percent"][fraction]
        except KeyError as exc:
            choices = sorted(selected["few_shot_train_percent"])
            raise ValueError(f"few_shot_percent must be one of {choices}") from exc
        return list(train_ids), list(selected["val"])
    raise ValueError(
        "domain_transfer requires data.transfer_scope equal to "
        "source_idealized or target_realistic"
    )


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def _build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    shared = {
        "in_channels": len(UrbanTalesPatchDataset.input_channels),
        "task_names": config["data"]["targets"],
        "predict_uncertainty": model_cfg.get("predict_uncertainty", True),
    }
    name = model_cfg["name"]
    if name == "multitask_unet":
        return MultiTaskUNet(
            **shared,
            base_channels=model_cfg["base_channels"],
            depth=model_cfg["depth"],
        )
    if name == "light_cnn":
        return LightCNN(
            **shared,
            width=model_cfg.get("width", 32),
            layers=model_cfg.get("layers", 6),
        )
    if name == "fno2d":
        return FNO2d(
            **shared,
            width=model_cfg.get("width", 48),
            layers=model_cfg.get("layers", 4),
            modes_y=model_cfg.get("modes_y", 16),
            modes_x=model_cfg.get("modes_x", 16),
        )
    if name == "geo_multiscale_operator":
        return GeoMultiScaleOperator(
            **shared,
            base_channels=model_cfg.get("base_channels", 32),
            depth=model_cfg.get("depth", 4),
            operator_levels=model_cfg.get("operator_levels", 2),
            modes=model_cfg.get("modes", 12),
            use_film=model_cfg.get("use_film", True),
        )
    raise ValueError(f"Unknown model.name: {name}")


def _build_optimizer(
    model: torch.nn.Module, training_config: dict[str, Any]
) -> tuple[AdamW, dict[str, Any]]:
    """Apply auditable freezing and learning-rate groups, then build AdamW.

    ``trainable_patterns`` uses shell-style patterns against full parameter names.
    ``parameter_groups`` may contain explicit ``patterns`` groups and at most one
    ``remaining: true`` catch-all group. Explicit groups must not overlap.
    """
    named_parameters = list(model.named_parameters())
    trainable_patterns = training_config.get("trainable_patterns")
    if trainable_patterns is not None:
        if not isinstance(trainable_patterns, list) or not trainable_patterns:
            raise ValueError("training.trainable_patterns must be a non-empty list")
        pattern_hits = {pattern: 0 for pattern in trainable_patterns}
        for name, parameter in named_parameters:
            matches = [
                pattern
                for pattern in trainable_patterns
                if fnmatch.fnmatchcase(name, pattern)
            ]
            parameter.requires_grad = bool(matches)
            for pattern in matches:
                pattern_hits[pattern] += 1
        missing = [pattern for pattern, hits in pattern_hits.items() if hits == 0]
        if missing:
            raise ValueError(f"trainable_patterns matched no parameters: {missing}")

    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Fine-tuning configuration left no trainable parameters")

    base_lr = float(training_config["learning_rate"])
    base_weight_decay = float(training_config.get("weight_decay", 1e-4))
    group_configs = training_config.get("parameter_groups")
    optimizer_groups: list[dict[str, Any]] = []
    group_report: list[dict[str, Any]] = []
    if group_configs is None:
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in trainable],
                "lr": base_lr,
                "weight_decay": base_weight_decay,
            }
        )
        group_report.append(
            {
                "name": "default",
                "learning_rate": base_lr,
                "weight_decay": base_weight_decay,
                "parameter_tensors": len(trainable),
                "parameter_count": sum(parameter.numel() for _, parameter in trainable),
            }
        )
    else:
        if not isinstance(group_configs, list) or not group_configs:
            raise ValueError("training.parameter_groups must be a non-empty list")
        names = [str(group["name"]) for group in group_configs]
        if len(names) != len(set(names)):
            raise ValueError("training.parameter_groups names must be unique")
        remaining_groups = [group for group in group_configs if group.get("remaining", False)]
        if len(remaining_groups) > 1:
            raise ValueError("At most one parameter group may set remaining: true")

        assignments: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
            str(group["name"]): [] for group in group_configs
        }
        unmatched: list[tuple[str, torch.nn.Parameter]] = []
        explicit_groups = [group for group in group_configs if not group.get("remaining", False)]
        pattern_hits: dict[tuple[str, str], int] = {}
        for group in explicit_groups:
            patterns = group.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(
                    f"Parameter group {group['name']!r} requires a non-empty patterns list"
                )
            for pattern in patterns:
                pattern_hits[(str(group["name"]), pattern)] = 0
        for name, parameter in trainable:
            matching_groups = []
            for group in explicit_groups:
                matches = [
                    pattern
                    for pattern in group["patterns"]
                    if fnmatch.fnmatchcase(name, pattern)
                ]
                if matches:
                    matching_groups.append(group)
                    for pattern in matches:
                        pattern_hits[(str(group["name"]), pattern)] += 1
            if len(matching_groups) > 1:
                matched_names = [str(group["name"]) for group in matching_groups]
                raise ValueError(f"Parameter {name!r} matches multiple groups: {matched_names}")
            if matching_groups:
                assignments[str(matching_groups[0]["name"])].append((name, parameter))
            else:
                unmatched.append((name, parameter))
        missing_patterns = [
            f"{group_name}:{pattern}"
            for (group_name, pattern), hits in pattern_hits.items()
            if hits == 0
        ]
        if missing_patterns:
            raise ValueError(f"parameter_groups patterns matched no parameters: {missing_patterns}")
        if remaining_groups:
            assignments[str(remaining_groups[0]["name"])] = unmatched
            unmatched = []
        if unmatched:
            raise ValueError(
                "Trainable parameters were not assigned to an optimizer group: "
                + ", ".join(name for name, _ in unmatched[:8])
            )
        for group in group_configs:
            group_name = str(group["name"])
            members = assignments[group_name]
            if not members:
                raise ValueError(f"Parameter group {group_name!r} is empty")
            learning_rate = float(group.get("learning_rate", base_lr))
            weight_decay = float(group.get("weight_decay", base_weight_decay))
            optimizer_groups.append(
                {
                    "params": [parameter for _, parameter in members],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
            group_report.append(
                {
                    "name": group_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "parameter_tensors": len(members),
                    "parameter_count": sum(parameter.numel() for _, parameter in members),
                    "patterns": group.get("patterns"),
                    "remaining": bool(group.get("remaining", False)),
                }
            )

    optimizer = AdamW(optimizer_groups)
    total_count = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    report = {
        "total_parameter_count": total_count,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": total_count - trainable_count,
        "trainable_fraction": trainable_count / total_count,
        "trainable_patterns": trainable_patterns,
        "groups": group_report,
    }
    return optimizer, report


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: MaskedMultiTaskLoss,
    device: torch.device,
    task_names: Sequence[str],
    amp: bool,
    max_batches: int | None,
    data_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sample_records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
            ):
                model_inputs = ablate_model_input_channels(
                    inputs, data_config.get("zero_model_input_channels")
                )
                output = model(model_inputs)
            u_tau_m_s = batch["u_tau_m_s"].to(device, non_blocking=True)
            for sample_index in range(inputs.shape[0]):
                sample_output = {
                    key: value[sample_index : sample_index + 1].float()
                    if value is not None
                    else None
                    for key, value in output.items()
                }
                sample_target = target[sample_index : sample_index + 1].float()
                sample_mask = mask[sample_index : sample_index + 1]
                sample_loss, _ = loss_fn(sample_output, sample_target, sample_mask)
                metrics = batch_metrics(
                    sample_output["mean"],
                    sample_target,
                    sample_mask,
                    task_names,
                    u_tau_m_s=u_tau_m_s[sample_index : sample_index + 1],
                    input_tensor=inputs[sample_index : sample_index + 1],
                    log_scale=sample_output.get("log_scale"),
                    pixel_size_m=(
                        float(data_config["patch_size_m"])
                        / float(data_config["output_pixels"])
                    ),
                    sdf_scale_m=float(data_config.get("sdf_scale_m", 64.0)),
                    near_building_distance_m=float(
                        evaluation_config.get("near_building_distance_m", 8.0)
                    ),
                    high_gradient_quantile=float(
                        evaluation_config.get("high_gradient_quantile", 0.9)
                    ),
                    minimum_direction_speed_m_s=float(
                        evaluation_config.get("minimum_direction_speed_m_s", 0.1)
                    ),
                )
                metrics["loss"] = float(sample_loss)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
                sample_records.append(
                    {
                        "case_id": batch["case_id"][sample_index],
                        "patch_index": int(batch["patch_index"][sample_index]),
                        "u_tau_m_s": float(batch["u_tau_m_s"][sample_index]),
                        "source_dx_m": float(batch["dx_m"][sample_index]),
                        "metrics": metrics,
                    }
                )
    averages = {key: value / counts[key] for key, value in totals.items()}
    return averages, sample_records


def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed, bool(config.get("deterministic", True)))
    root = Path(config["data"]["root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    training_config = config["training"]
    initial_checkpoint = training_config.get("initial_checkpoint")
    resume_checkpoint = training_config.get("resume_checkpoint")
    if initial_checkpoint and resume_checkpoint:
        raise ValueError("initial_checkpoint and resume_checkpoint are mutually exclusive")
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not resume_checkpoint
        and not training_config.get("allow_existing_output", False)
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new output_dir or "
            "set training.resume_checkpoint; do not mix independent runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = UrbanTalesCatalog(root)
    split_path = root / config["data"]["split_manifest"]
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split = manifest["protocols"][config["data"]["protocol"]]
    split_train, split_val = _select_partition(split, config["data"])
    train_ids = _balanced_limit(split_train, catalog, config["data"].get("max_train_cases"))
    val_ids = _balanced_limit(split_val, catalog, config["data"].get("max_val_cases"))
    train_dataset = _make_dataset(root, catalog, train_ids, config, train=True)
    val_dataset = _make_dataset(root, catalog, val_ids, config, train=False)
    loader_options = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"].get("num_workers", 0),
        "pin_memory": config["training"].get("pin_memory", True),
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(seed + 101)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    device = _device(config["training"].get("device", "auto"))
    model = _build_model(config).to(device)
    if initial_checkpoint:
        checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("tasks", ())) != tuple(config["data"]["targets"]):
            raise ValueError("Initial checkpoint targets do not match current targets")
        model.load_state_dict(checkpoint["model"], strict=True)
    loss_cfg = config["loss"]
    loss_fn = MaskedMultiTaskLoss(
        config["data"]["targets"],
        task_weights=loss_cfg.get("task_weights"),
        gradient_weight=loss_cfg.get("gradient_weight", 0.0),
        use_uncertainty=loss_cfg.get("use_uncertainty", True),
    ).to(device)
    optimizer, optimization_report = _build_optimizer(model, training_config)
    amp = bool(config["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    max_steps = config["training"].get("max_steps_per_epoch")
    max_val_batches = config["training"].get("max_val_batches")
    total_epochs = int(config["training"]["epochs"])
    scheduler_name = str(training_config.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(training_config.get("min_learning_rate", 1e-6)),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError("training.scheduler must be 'none' or 'cosine'")
    early_stopping_patience = training_config.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or null")
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("tasks", ())) != tuple(config["data"]["targets"]):
            raise ValueError("Resume checkpoint targets do not match current targets")
        previous_config = checkpoint.get("config")
        if previous_config is None:
            raise ValueError("Resume checkpoint does not contain its resolved config")
        previous_signature = _training_continuity_signature(previous_config)
        current_signature = _training_continuity_signature(config)
        if previous_signature != current_signature:
            raise ValueError(
                "Resume checkpoint continuity signature does not match the current config: "
                f"{previous_signature} != {current_signature}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get("validation", {}).get("loss", float("inf")),
            )
        )
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if checkpoint.get("data_loader_generator_state") is not None:
            train_generator.set_state(checkpoint["data_loader_generator_state"])
        _restore_rng_state(checkpoint.get("rng_state"))
    if start_epoch >= total_epochs:
        raise ValueError(
            f"Resume checkpoint starts at epoch {start_epoch}, but training.epochs={total_epochs}"
        )
    max_epochs_this_run = training_config.get("max_epochs_this_run")
    if max_epochs_this_run is not None:
        max_epochs_this_run = int(max_epochs_this_run)
        if max_epochs_this_run <= 0:
            raise ValueError("max_epochs_this_run must be positive or null")
        run_end_epoch = min(total_epochs, start_epoch + max_epochs_this_run)
    else:
        run_end_epoch = total_epochs
    environment = _environment(device)
    environment_name = (
        "environment.json" if not resume_checkpoint else f"environment_resume_{start_epoch}.json"
    )
    config_name = (
        "resolved_config.json"
        if not resume_checkpoint
        else f"resolved_config_resume_{start_epoch}.json"
    )
    (output_dir / environment_name).write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / config_name).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()
    last_train: dict[str, float] = {}
    last_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, run_end_epoch):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        steps_since_update = 0
        train_totals: dict[str, float] = {}
        train_step_count = 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= int(max_steps):
                break
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
            ):
                model_inputs = ablate_model_input_channels(
                    inputs, config["data"].get("zero_model_input_channels")
                )
                output = model(model_inputs)
                loss, details = loss_fn(output, target, mask)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            steps_since_update += 1
            if (step + 1) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
            global_step += 1
            train_step_count += 1
            for key, value in details.items():
                train_totals[key] = train_totals.get(key, 0.0) + value
        if not train_step_count:
            raise RuntimeError("No training batches were processed")
        if steps_since_update:
            correction = accumulation / steps_since_update
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        last_train = {
            key: value / train_step_count for key, value in train_totals.items()
        }
        validation, validation_samples = _run_validation(
            model,
            val_loader,
            loss_fn,
            device,
            config["data"]["targets"],
            amp,
            int(max_val_batches) if max_val_batches is not None else None,
            config["data"],
            config.get("evaluation", {}),
        )
        latest_samples_path = output_dir / "validation_samples_latest.jsonl"
        latest_samples_path.write_text(
            "".join(
                json.dumps({"epoch": epoch, **sample}, ensure_ascii=False) + "\n"
                for sample in validation_samples
            ),
            encoding="utf-8",
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        learning_rates = {
            group["name"]: float(optimizer.param_groups[index]["lr"])
            for index, group in enumerate(optimization_report["groups"])
        }
        validation_loss = validation.get("loss", float("inf"))
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_steps": train_step_count,
            "learning_rate": learning_rate,
            "learning_rates": learning_rates,
            "epoch_seconds_excluding_checkpoint": epoch_seconds,
            "train": last_train,
            "val": validation,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if improved:
            (output_dir / "validation_samples_best.jsonl").write_text(
                latest_samples_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "tasks": config["data"]["targets"],
            "global_step": global_step,
            "epoch": epoch,
            "validation": validation,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "rng_state": _capture_rng_state(),
            "data_loader_generator_state": train_generator.get_state(),
            "training_continuity_sha256": _training_continuity_signature(config),
            "optimization": optimization_report,
        }
        _atomic_torch_save(checkpoint, output_dir / "last.pt")
        if improved:
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        last_epoch = epoch
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break

    elapsed = time.perf_counter() - train_start

    # Timed forward pass after one warm-up; synchronize only for CUDA timing.
    batch = next(iter(val_loader))
    inputs = batch["input"].to(device)
    with torch.inference_mode():
        model_inputs = ablate_model_input_channels(
            inputs, config["data"].get("zero_model_input_channels")
        )
        _ = model(model_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(model_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - start
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "ok",
        "train_cases": train_ids,
        "val_cases": val_ids,
        "epochs": total_epochs,
        "start_epoch": start_epoch,
        "last_epoch": last_epoch,
        "run_end_epoch_exclusive": run_end_epoch,
        "epochs_completed_this_run": last_epoch - start_epoch + 1,
        "stopped_early": stopped_early,
        "global_steps": global_step,
        "parameter_count": parameter_count,
        "optimization": optimization_report,
        "elapsed_seconds": elapsed,
        "inference_batch_seconds": inference_seconds,
        "inference_batch_size": int(inputs.shape[0]),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "last_train": last_train,
        "last_validation": validation,
        "validation_aggregation": "macro_mean_over_evaluation_patches",
        "validation_sample_count": len(validation_samples),
        "best_validation_loss": best_validation_loss,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "training_continuity_sha256": _training_continuity_signature(config),
        "environment": environment,
        "process_id": os.getpid(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
