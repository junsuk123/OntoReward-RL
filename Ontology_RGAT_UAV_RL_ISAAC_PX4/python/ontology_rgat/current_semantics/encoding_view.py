"""Live view of the graph encoders' internal state, for the study monitor (read-only).

For a training directory of a graph arm (Ontology R-GAT or the untyped GNN), the newest
checkpoint the run has written -- ``best.pt``, replaced whenever a validation round improves
it, or ``final.pt`` -- is flown for ONE deterministic probe episode on the calibrated local
plant, and every decision records what the encoder computed:

  * the attention ``alpha_ij`` of every edge at every hop (the softmax over the edges into
    each node, so the weights into one node sum to 1);
  * the node states ``h_i`` after the input projection and after each hop (their norms);
  * the 8-D LandingContext readout ``g_t`` the actor reads next to the raw observation;
  * what the policy saw (decoded pad estimate, vision and navigation age) and the action.

``trace_encoder`` repeats the encoder's own arithmetic step by step and asserts that its
readout equals ``GraphEncoder.forward`` on the same input, so the view cannot drift from the
model.  The probe environment (``PROBE_SEED``) is a monitor-only seed outside every
registered or audited range; the episode trains nothing, writes nothing, and its verdict is
not evidence for any comparison.  Results are cached per (run, probe, checkpoint mtime).
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import PROJECT_ROOT

GRAPH_ARMS = ("ontology_rgat_ppo", "untyped_gnn_ppo")
PROBE_SEED = 999_901            # monitor-only; outside every registered and audited range
PROBES = {"nominal": "composition_id", "blackout": "dropout_id"}
CONFIGS = ("spatial_view_contact_v1.yaml", "spatial_view_contact_v1_nav4_wmon.yaml",
           "spatial_view_contact_v1_nav4_kalman.yaml")
_CACHE: dict[tuple, dict[str, Any]] = {}
_LOCK = threading.Lock()


def trace_encoder(encoder: Any, nodes: torch.Tensor) -> dict[str, np.ndarray]:
    """alpha (hops, B, E), node-state norms (hops + 1, B, N) and g (B, D) of ``encoder`` on
    ``nodes``, by the encoder's own arithmetic (RelationalAttentionLayer.forward)."""
    from .models import NEGATIVE_SLOPE
    layer = encoder.layer
    source, target, relation = encoder.source, encoder.target, encoder.relation
    with torch.no_grad():
        h = torch.tanh(encoder.input(nodes))
        norms, alphas = [h.norm(dim=-1)], []
        batch, count, _ = h.shape
        for _ in range(encoder.hops):
            messages = torch.einsum("eoi,bei->beo", layer.weight[relation], h[:, source])
            score = F.leaky_relu(messages @ layer.att_src + (h @ layer.att_dst)[:, target]
                                 + layer.relation_bias[relation], NEGATIVE_SLOPE)
            peak = score.new_full((batch, count), -math.inf).scatter_reduce(
                1, target.expand(batch, -1), score, reduce="amax", include_self=True)
            weight = torch.exp(score - peak[:, target])
            total = weight.new_zeros((batch, count)).index_add(1, target, weight)
            alphas.append(weight / total[:, target])
            h = layer(h, source, target, relation)
            norms.append(h.norm(dim=-1))
        g = encoder.readout(h[:, encoder.readout_index])
        reference = encoder(nodes)
    if not torch.allclose(g, reference, atol=1e-6):
        raise RuntimeError("the traced readout differs from GraphEncoder.forward")
    return {"alpha": torch.stack(alphas).numpy(), "norm": torch.stack(norms).numpy(),
            "g": g.numpy()}


def config_for_root(root_name: str) -> Path | None:
    """The registered config whose run root is ``root_name`` (task root or navigation root)."""
    from .contracts import load_config
    from .experiment import config_run_root
    for name in CONFIGS:
        path = PROJECT_ROOT / "config" / "current_semantics" / name
        try:
            if config_run_root(load_config(path)).name == root_name:
                return path
        except (OSError, ValueError):
            continue
    return None


def newest_checkpoint(run_dir: Path) -> Path | None:
    found = [p for p in (run_dir / "best.pt", run_dir / "final.pt") if p.exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _load(arm: str, config: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    """The architecture is the arm's whatever the PPO budget or seeds it trained under, so
    the state dict loads strictly into a fresh model; for viewing only, never for a verdict."""
    from .models import CurrentActorCritic
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = CurrentActorCritic(arm, config, seed=0)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, dict(payload.get("extra") or {})


def probe_episode(model: Any, config: Any, probe: str) -> dict[str, Any]:
    """One deterministic local episode of ``model`` on the probe cell, decision by decision."""
    from . import development as dev
    from .backend import make_local_backend
    from .features import build_features
    from .models import tensors_from_frames
    from .observation import decode
    plan = dev.load_development(dev.default_plan_path(config), config)
    cell = plan.cell(PROBES[probe])
    factors, dropout = dev.cell_conditions(config, asdict(cell), plan.dropout_start_s)
    backend = make_local_backend(config, difficulty=1.0, randomization=None, scenario=cell.split,
                                 factors=factors, dropout=dropout)
    steps: list[dict[str, Any]] = []
    status = None
    try:
        observation, _ = backend.reset(PROBE_SEED)
        while True:
            raw, flat, nodes = tensors_from_frames([build_features(observation, config)])
            trace = trace_encoder(model.encoder, nodes)
            with torch.no_grad():
                out = model.act(raw, flat, nodes, deterministic=True)
            seen = decode(observation)
            step = backend.step(out["action"][0].double().numpy(),
                                latent=out["latent"][0].double().numpy())
            age = seen.vision_age_s
            steps.append({
                "t": round(float(step.elapsed_s), 2),
                "alpha": np.round(trace["alpha"][:, 0], 4).tolist(),
                "norm": np.round(trace["norm"][:, 0], 4).tolist(),
                "g": np.round(trace["g"][0], 4).tolist(),
                "action": np.round(out["action"][0].numpy(), 3).tolist(),
                "seen": {"valid": bool(seen.estimate_valid),
                         "ex": None if not seen.estimate_valid else round(seen.ex, 3),
                         "ey": None if not seen.estimate_valid else round(seen.ey, 3),
                         "h": None if not seen.estimate_valid else round(seen.h, 3),
                         "vision_age_s": None if not math.isfinite(age) else round(age, 2)}})
            observation = step.observation
            if step.done:
                status = step.status
                break
    finally:
        backend.close()
    return {"steps": steps, "status": status}


def view(results: Path, run_rel: str, probe: str = "nominal") -> dict[str, Any]:
    """The encoding view of one run (``run_rel`` relative to ``results``), cached until the
    run writes a newer checkpoint."""
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {tuple(PROBES)}")
    run_dir = (Path(results) / run_rel).resolve()
    if Path(results).resolve() not in run_dir.parents:
        raise ValueError("run outside the results folder")
    arm = run_dir.name if run_dir.name in GRAPH_ARMS else None
    if arm is None:
        raise ValueError(f"{run_rel} is not a graph-arm training directory {GRAPH_ARMS}")
    checkpoint = newest_checkpoint(run_dir)
    if checkpoint is None:
        return {"run": run_rel, "arm": arm, "checkpoint": None,
                "note": "no checkpoint yet: best.pt appears at the first validation at "
                        "difficulty 1.0"}
    key = (str(run_dir), probe, checkpoint.name, checkpoint.stat().st_mtime)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        from .contracts import load_config
        root = next((p for p in run_dir.parents if config_for_root(p.name) is not None), None)
        if root is None:
            raise ValueError(f"no registered config owns {run_rel}")
        config = load_config(config_for_root(root.name))
        model, extra = _load(arm, config, checkpoint)
        started = time.monotonic()
        episode = probe_episode(model, config, probe)
        spec = model.encoder.spec
        payload = {
            "run": run_rel, "arm": arm, "probe": probe, "probe_cell": PROBES[probe],
            "probe_seed": PROBE_SEED, "checkpoint": checkpoint.name,
            "checkpoint_update": extra.get("update"),
            "checkpoint_mtime": checkpoint.stat().st_mtime,
            "nodes": list(spec.nodes), "readout": spec.readout_node,
            "relations": list(spec.relations),
            "edges": [{"source": s, "target": t, "relation": r} for s, t, r in spec.edges],
            "hops": int(model.encoder.hops), "status": episode["status"],
            "decisions": len(episode["steps"]), "steps": episode["steps"],
            "compute_s": round(time.monotonic() - started, 2),
            "note": "monitor probe: one deterministic local episode on a monitor-only seed; "
                    "not evidence for any comparison"}
        for stale in [k for k in _CACHE if k[:2] == key[:2]]:
            del _CACHE[stale]
        _CACHE[key] = payload
        return payload
