"""Measure full-model forward/backward feasibility on a CUDA GPU.

This deliberately uses synthetic tensors: it validates the model, AMP path,
optimizer state, runtime, and peak memory before any large UrbanTALES transfer
or training job is started.  It is not an accuracy or data-pipeline test.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from urbantales_ml.models import GeoMultiScaleOperator


TASK_NAMES = ("uped", "vped", "Uped", "TKEped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--output-pixels", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/gpu_synthetic_smoke/summary.json"),
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_model() -> GeoMultiScaleOperator:
    return GeoMultiScaleOperator(
        in_channels=6,
        task_names=TASK_NAMES,
        base_channels=32,
        depth=4,
        operator_levels=2,
        modes=12,
        use_film=True,
        predict_uncertainty=True,
    )


def synchronize() -> None:
    torch.cuda.synchronize()


def one_batch(batch_size: int, pixels: int, warmup: int, repeats: int, amp: bool) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    inputs = torch.randn(batch_size, 6, pixels, pixels, device="cuda")
    # Keep the condition channels spatially constant, as in the real dataset.
    inputs[:, 3:] = torch.randn(batch_size, 3, 1, 1, device="cuda")
    targets = torch.randn(batch_size, len(TASK_NAMES), pixels, pixels, device="cuda")

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            prediction = model(inputs)
            loss = (prediction["mean"] - targets).square().mean()
            if prediction["log_scale"] is not None:
                loss = loss + 0.01 * prediction["log_scale"].square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    for _ in range(warmup):
        step()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings_ms: list[float] = []
    losses: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        losses.append(step())
        end.record()
        synchronize()
        timings_ms.append(float(start.elapsed_time(end)))

    result = {
        "status": "ok",
        "batch_size": batch_size,
        "output_pixels": pixels,
        "amp": amp,
        "train_step_ms_mean": sum(timings_ms) / len(timings_ms),
        "train_step_ms": timings_ms,
        "last_loss": losses[-1],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    del inputs, targets, optimizer, model, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    amp = not args.no_amp
    probe_model = make_model()
    parameters = sum(parameter.numel() for parameter in probe_model.parameters())
    del probe_model
    summary = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_parameters": parameters,
        "results": [],
    }
    for batch_size in batch_sizes:
        try:
            result = one_batch(
                batch_size, args.output_pixels, args.warmup, args.repeats, amp
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            result = {
                "status": "oom",
                "batch_size": batch_size,
                "output_pixels": args.output_pixels,
                "amp": amp,
                "error": str(error),
            }
        summary["results"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
