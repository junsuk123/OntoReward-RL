"""Measure the CPU/GPU crossover for the active R-GAT layer.

The claim that batching is what makes the GPU worth using is a measurement, not
an assumption, and it is machine-specific. This times one forward/backward pass
at representative batch sizes. It changes no hyperparameter and writes no model.

    python3 python/run_benchmark.py --batches 32 256 1024 4096
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from ..config import Config, default_config
from ..semantic import SemanticState, build_ontology_graph
from .model import build_potential

__all__ = ["benchmark"]


def _step(model, X: torch.Tensor, y: torch.Tensor, l2: float) -> None:
    model.zero_grad(set_to_none=True)
    pred = model(X)
    loss = ((pred - y) ** 2).mean() + l2 * (pred ** 2).mean()
    loss.backward()


def benchmark(batch_sizes: Sequence[int] = (32, 256, 1024, 4096),
              repetitions: int = 5, cfg: Config | None = None) -> list[dict[str, float]]:
    cfg = cfg or default_config("quick")
    sem = SemanticState(position_error=0.7, vertical_speed=-0.4, tilt=0.1,
                        angular_rate=0.2, wind_risk=0.3, marker_quality=0.8,
                        visual_stability=0.8, alignment=0.4,
                        attitude_stability=0.6, touchdown_safety=0.2,
                        pad_motion=0.35, battery_reserve=0.6)
    graph = build_ontology_graph(sem, cfg)
    rng = np.random.default_rng(cfg.seed)
    has_cuda = torch.cuda.is_available()
    if cfg.device.allow_tf32 and has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    rows: list[dict[str, float]] = []
    for batch in batch_sizes:
        base = np.repeat(graph.X.T[None], batch, axis=0)
        X_np = (base + 0.01 * rng.standard_normal(base.shape)).astype(np.float32)
        y_np = (0.5 * rng.standard_normal(batch)).astype(np.float32)
        row: dict[str, float] = {"batch": float(batch)}

        for device_name in (["cpu", "cuda"] if has_cuda else ["cpu"]):
            device = torch.device(device_name)
            model = build_potential(cfg, graph, device=device)
            X = torch.as_tensor(X_np, device=device)
            y = torch.as_tensor(y_np, device=device)
            _step(model, X, y, cfg.rgat.output_l2)       # warm up, and build the graph
            if device_name == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(repetitions):
                _step(model, X, y, cfg.rgat.output_l2)
            if device_name == "cuda":
                torch.cuda.synchronize()
            row[f"{device_name}_ms"] = 1e3 * (time.perf_counter() - started) / repetitions
        if "cuda_ms" in row:
            row["gpu_speedup"] = row["cpu_ms"] / max(row["cuda_ms"], 1e-9)
        rows.append(row)

    header = f"{'batch':>8} {'CPU ms':>10} {'GPU ms':>10} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        gpu = f"{row['cuda_ms']:10.2f}" if "cuda_ms" in row else f"{'-':>10}"
        speed = f"{row['gpu_speedup']:9.2f}x" if "gpu_speedup" in row else f"{'-':>10}"
        print(f"{int(row['batch']):>8} {row['cpu_ms']:10.2f} {gpu} {speed}")
    if any("gpu_speedup" in r for r in rows):
        crossover = next((int(r["batch"]) for r in rows if r.get("gpu_speedup", 0) > 1.0),
                         None)
        if crossover is None:
            print("\nThe GPU never wins at these sizes; leave cfg.device.rgat on 'auto'.")
        else:
            print(f"\nThe GPU wins from batch {crossover}; set "
                  f"cfg.device.min_batch_for_gpu to it if it differs from the default.")
    return rows
