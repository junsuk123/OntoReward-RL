# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Live view of the graph encoders' internal state, for the study monitor (read-only).

For a training directory of a graph arm -- the proposed ORL policy (orl.py), the registered
Ontology R-GAT or the untyped GNN -- the newest weights the run has written are flown for ONE
deterministic probe episode on the calibrated local plant: ``live.pt`` (the trainer's
snapshot every few updates, pilot.LIVE_EVERY), ``best.pt`` (replaced whenever a validation
round improves it) or ``final.pt``.  Every decision records what the encoder computed:

  * the attention ``alpha_ij`` of every edge at every hop (the softmax over the edges into
    each node, so the weights into one node sum to 1);
  * the node states ``h_i`` after the input projection and after each hop (their norms);
  * the readout: for the R-GAT arms the 8-D LandingContext ``g_t`` the actor reads next to
    the raw observation; for ORL the mirror-averaged decision outputs (mu_x, mu_y, mu_z and
    the vertical law's gate, hold and gated term), with the graph shown for the identity
    copy of the four mirror images;
  * what the policy saw (decoded pad estimate, vision and navigation age) and the action.

``trace_encoder`` / ``trace_orl`` repeat the encoder's own arithmetic step by step and assert
that the result equals ``GraphEncoder.forward`` / ``ORLEncoder.node_states`` on the same
input, so the view cannot drift from the model.  The probe environment (``PROBE_SEED``) is a monitor-only seed outside every
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
ORL_PREFIX = "orl_ppo"          # ORL arms are "orl_ppo:<hash>", their folders "orl_ppo_<hash>"
CHECKPOINT_NAMES = ("live.pt", "best.pt", "final.pt")
# ORL graph layout for the monitor's 660 x 445 drawing: the x plane on top, its mirror image
# (the y plane) below, the shared state concepts on the left, the decision nodes on the right.
ORL_LAYOUT = {"RelativeGeometry_x": (165, 40), "MotionState_x": (165, 92),
              "AttitudeState_x": (165, 144), "ApproachSituation_x": (345, 40),
              "ViewGeometry_x": (345, 100), "ContactCondition_x": (345, 160),
              "ObservationQuality": (60, 165), "VerticalState": (60, 222),
              "HeadingState": (60, 279),
              "AttitudeState_y": (165, 300), "MotionState_y": (165, 352),
              "RelativeGeometry_y": (165, 398), "ContactCondition_y": (345, 284),
              "ViewGeometry_y": (345, 342), "ApproachSituation_y": (345, 398),
              "LateralDecision_x": (515, 100), "DescentEligibility": (515, 220),
              "LateralDecision_y": (515, 344), "VerticalDecision": (615, 240)}
ORL_SHORT = {"ObservationQuality": "Q", "VerticalState": "V", "HeadingState": "H",
             "RelativeGeometry": "G", "MotionState": "M", "AttitudeState": "A",
             "ApproachSituation": "P", "ViewGeometry": "O", "ContactCondition": "C",
             "LateralDecision": "Lat", "DescentEligibility": "E", "VerticalDecision": "Z"}
ORL_OUTPUTS = ("μx", "μy", "μz", "gate", "hold", "gate·desc")


def arm_of(folder: str) -> str | None:
    """The arm a training folder holds, or None if it is not a graph arm."""
    if folder in GRAPH_ARMS:
        return folder
    if folder.startswith(ORL_PREFIX + "_"):
        return f"{ORL_PREFIX}:{folder[len(ORL_PREFIX) + 1:]}"
    return None


def _hops(layer: Any, source: torch.Tensor, target: torch.Tensor, relation: torch.Tensor,
          h: torch.Tensor, hops: int) -> tuple[list, list, torch.Tensor]:
    """Attention alpha (per hop) and node-state norms (initial + per hop) of ``hops`` rounds
    of RelationalAttentionLayer on ``h``; the update itself is the layer's own forward."""
    from .models import NEGATIVE_SLOPE
    norms, alphas = [h.norm(dim=-1)], []
    batch, count, _ = h.shape
    for _ in range(hops):
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
    return alphas, norms, h
PROBE_SEED = 999_901            # monitor-only; outside every registered and audited range
PROBES = {"nominal": "composition_id", "blackout": "dropout_id"}
CONFIGS = ("spatial_view_contact_v1.yaml",)   # the base contract ORL runs on
_CACHE: dict[tuple, dict[str, Any]] = {}
_LOCK = threading.Lock()


def trace_encoder(encoder: Any, nodes: torch.Tensor) -> dict[str, np.ndarray]:
    """alpha (hops, B, E), node-state norms (hops + 1, B, N) and g (B, D) of ``encoder`` on
    ``nodes``, by the encoder's own arithmetic (RelationalAttentionLayer.forward)."""
    with torch.no_grad():
        h = torch.tanh(encoder.input(nodes))
        alphas, norms, h = _hops(encoder.layer, encoder.source, encoder.target,
                                 encoder.relation, h, encoder.hops)
        g = encoder.readout(h[:, encoder.readout_index])
        reference = encoder(nodes)
    if not torch.allclose(g, reference, atol=1e-6):
        raise RuntimeError("the traced readout differs from GraphEncoder.forward")
    return {"alpha": torch.stack(alphas).numpy(), "norm": torch.stack(norms).numpy(),
            "g": g.numpy()}


def trace_orl(model: Any, flat44: torch.Tensor) -> dict[str, np.ndarray]:
    """The ORL encoder on the identity mirror copy, by its own arithmetic: alpha (hops, B, E)
    and node-state norms (hops + 1, B, N), asserted equal to ``ORLEncoder.node_states``; and
    the actor's mirror-averaged outputs (B, 6): mu_x, mu_y, mu_z, gate, hold, gated term."""
    from . import orl
    enc = model.encoder
    with torch.no_grad():
        batch = flat44.shape[0]
        states = {}
        for concept in orl.SHARED_ROLES:
            states[concept] = torch.tanh(enc.encoders[concept](
                flat44[:, getattr(enc, f"idx_{concept}")]))
        for concept in orl.PLANE_ROLES:
            for plane in orl.PLANES:
                inputs = flat44[:, getattr(enc, f"idx_{concept}_{plane}")] \
                    * getattr(enc, f"sgn_{concept}_{plane}")
                states[orl._node(concept, plane)] = torch.tanh(enc.encoders[concept](inputs))
        decision = torch.tanh(enc.decision)
        for plane in orl.PLANES:
            states[orl._node("LateralDecision", plane)] = decision[0].expand(batch, -1)
        states["DescentEligibility"] = decision[1].expand(batch, -1)
        states["VerticalDecision"] = decision[2].expand(batch, -1)
        h = torch.stack([states[n] for n in orl.NODES], dim=1)
        alphas, norms, h = _hops(enc.layer, enc.source, enc.target, enc.relation, h, orl.HOPS)
        reference = enc.node_states(flat44)
        terms = model.vertical_terms(flat44)
    if not torch.allclose(h, reference, atol=1e-6):
        raise RuntimeError("the traced node states differ from ORLEncoder.node_states")
    return {"alpha": torch.stack(alphas).numpy(), "norm": torch.stack(norms).numpy(),
            "terms": {k: v.numpy() for k, v in terms.items()}}


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
    found = [run_dir / name for name in CHECKPOINT_NAMES if (run_dir / name).exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _load(arm: str, config: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    """The architecture is the arm's whatever the PPO budget or seeds it trained under, so
    the state dict loads strictly into a fresh model; for viewing only, never for a verdict."""
    from .models import build_model
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(arm, config, seed=0)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, dict(payload.get("extra") or {})


def probe_episode(model: Any, config: Any, probe: str) -> dict[str, Any]:
    """One deterministic local episode of ``model`` on the probe cell, decision by decision."""
    orl_arm = str(getattr(model, "arm", "")).startswith(ORL_PREFIX)
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
            with torch.no_grad():
                out = model.act(raw, flat, nodes, deterministic=True)
            if orl_arm:
                trace = trace_orl(model, flat)
                t = trace["terms"]
                mean = out["latent"][0].numpy()   # deterministic: the latent is the mean
                readout = [mean[0], mean[1], mean[2], t["gate"][0], t["hold"][0], t["gated"][0]]
            else:
                trace = trace_encoder(model.encoder, nodes)
                readout = trace["g"][0]
            seen = decode(observation)
            step = backend.step(out["action"][0].double().numpy(),
                                latent=out["latent"][0].double().numpy())
            age = seen.vision_age_s
            steps.append({
                "t": round(float(step.elapsed_s), 2),
                "alpha": np.round(trace["alpha"][:, 0], 4).tolist(),
                "norm": np.round(trace["norm"][:, 0], 4).tolist(),
                "g": np.round(np.asarray(readout, dtype=float), 4).tolist(),
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


def _live_every() -> int:
    from .pilot import LIVE_EVERY
    return LIVE_EVERY


def _graph_fields(model: Any) -> dict[str, Any]:
    """What the page needs to draw this arm's graph and readout."""
    if str(model.arm).startswith(ORL_PREFIX):
        from . import orl
        spec = orl.GRAPHS[model.version]
        short = {}
        for node in spec.nodes:
            concept, _, plane = node.rpartition("_") if node.endswith(("_x", "_y")) \
                else (node, "", "")
            short[node] = ORL_SHORT[concept] + (plane if plane else "")
        return {"family": "orl", "version": model.version,
                "nodes": list(spec.nodes), "readout": spec.readout_node,
                "relations": list(spec.relations),
                "edges": [{"source": s, "target": t, "relation": r} for s, t, r in spec.edges],
                "hops": orl.HOPS, "state_dim": orl.EMBEDDING, "layout": ORL_LAYOUT,
                "short": short, "radius": 18,
                "columns": [[60, "공유 상태"], [165, "관측 개념"], [345, "착륙 상황"],
                            [560, "결정"]],
                "g_labels": list(ORL_OUTPUTS), "g_row_scale": True,
                "g_title": "결정 출력 (네 거울 복사본의 평균: μ는 tanh 전 평균, 수직 μz = hold + gate·descend)",
                "extra_note": "; the graph is the identity copy of the four mirror images, "
                              "the outputs are the mirror average the actor uses"}
    spec = model.encoder.spec
    return {"family": "rgat", "nodes": list(spec.nodes), "readout": spec.readout_node,
            "relations": list(spec.relations),
            "edges": [{"source": s, "target": t, "relation": r} for s, t, r in spec.edges],
            "hops": int(model.encoder.hops), "state_dim": int(model.encoder.dim),
            "g_labels": [f"g{i + 1}" for i in range(8)], "g_row_scale": False,
            "g_title": "LandingContext 판독 g_t (8차원, 현재 시점)", "extra_note": ""}


def view(results: Path, run_rel: str, probe: str = "nominal") -> dict[str, Any]:
    """The encoding view of one run (``run_rel`` relative to ``results``), cached until the
    run writes a newer checkpoint."""
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {tuple(PROBES)}")
    run_dir = (Path(results) / run_rel).resolve()
    if Path(results).resolve() not in run_dir.parents:
        raise ValueError("run outside the results folder")
    arm = arm_of(run_dir.name)
    if arm is None:
        raise ValueError(f"{run_rel} is not a graph-arm training directory "
                         f"({GRAPH_ARMS} or {ORL_PREFIX}_<hash>)")
    checkpoint = newest_checkpoint(run_dir)
    if checkpoint is None:
        return {"run": run_rel, "arm": arm, "checkpoint": None,
                "note": f"no weights yet: the trainer writes live.pt every "
                        f"{_live_every()} updates"}
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
        payload = _graph_fields(model)
        payload.update({
            "run": run_rel, "arm": arm, "probe": probe, "probe_cell": PROBES[probe],
            "probe_seed": PROBE_SEED, "checkpoint": checkpoint.name,
            "checkpoint_update": extra.get("update"),
            "checkpoint_mtime": checkpoint.stat().st_mtime,
            "checkpoint_difficulty": extra.get("difficulty"),
            "live": bool(extra.get("live")), "status": episode["status"],
            "decisions": len(episode["steps"]), "steps": episode["steps"],
            "compute_s": round(time.monotonic() - started, 2)})
        payload["note"] = ("monitor probe: one deterministic local episode on a monitor-only "
                           "seed; not evidence for any comparison" + payload.pop("extra_note"))
        for stale in [k for k in _CACHE if k[:2] == key[:2]]:
            del _CACHE[stale]
        _CACHE[key] = payload
        return payload
