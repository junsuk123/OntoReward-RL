"""Gates, separated report states and run bookkeeping of the current-observation path.

Nothing here trains.  The gate table (guide 10) is evidence, not a verdict to
be argued down: G0 and G1 are blocked by inputs absent from this machine, G2
and G3 are partial with their open items named, and no single PASSED stands
in for the separate report states of guide 12.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import errno
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np

from . import SCHEMA_ID
from .contracts import (ARMS, DESIGN_PARAMETER_TOTALS, NODE_ORDER, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, design_parameter_counts,
                        load_config)


REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "current_semantics_3d" / SCHEMA_ID
LOCK_NAME = "run.lock"
ARTIFACTS = MappingProxyType({"audit": "audit.json", "fixtures": "fixtures.json",
                              "smoke": "smoke.json", "coupling": "coupling_local.json",
                              "coupling_isaac": "coupling_isaac.json",
                              "learnability": "g4_pilot.json",
                              "isaac_eval": "isaac_eval.json",
                              "register_development": "development_registration.json",
                              "develop": "development.json",
                              "register_isaac_development":
                                  "isaac_development_registration.json",
                              "isaac_development": "isaac_development.json"})
GATES = ("G0", "G1", "G2", "G3", "G4")
# Development opens once G0-G4 are passed or waived (guide 10-11); confirmation opens
# only after a passed development gate and is not implemented yet.
DEVELOPMENT_STAGES = ("register-development", "develop")
REFUSED_STAGES = ("register-confirmation", "confirm")

_G0_EVIDENCE = (
    "3D baseline cab3dd2123c7eee3968a8c00753f2ba29d55fe71 pinned (contracts.BASELINE_3D_SHA)",
    "3D source/contract/runtime audit: docs/current_semantics/SOURCE_AUDIT_KO.md",
)
_G0_BLOCKING = (
    "the 2D current_semantics_v1 evidence bundle (registration.json, resolved config, "
    "selected checkpoints, validation history, gate results, episode CSVs) is absent on "
    "this machine",
    "commits 5853781 (local v1 delivery) and d438a7d (2D remote main) exist in no local "
    "git repository",
    "the synced 2D folder stops at 2026-10-27 12:27 KST (relational_residual_v7), before "
    "the 23:19 v1 registration",
)
_G1_BLOCKING = (
    "needs the frozen v1 development checkpoints (absent, see G0)",
    "the existing 2D branchRollouts.m perturbs a_x only and branches at UGV-acceleration "
    "times, not at the first geometric FOV-boundary approach",
    "struct-copy snapshots of the 2D env are exact only in perception mode "
    "(env.sensorStream is a RandStream handle)",
)
# User decision 2026-10-28 ("these two tasks are unnecessary"): the 2D-provenance
# gates and two G2 items are waived, not passed.  A waiver is recorded with its
# date and scope so no report can read it as evidence.
USER_WAIVERS = MappingProxyType({
    "G0": "waived by the user 2026-10-28: the 2D v1 evidence bundle will not be brought here",
    "G1": "waived by the user 2026-10-28: no frozen 2D failure diagnosis on v1 checkpoints",
    "G2": ("waived by the user 2026-10-28: the v1 2D 10D/Phi19/7-node fixture, and accepting "
           "navigation valid/age as constant by construction"),
})


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), *args], check=True,
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def provenance() -> dict[str, Any]:
    """Source and runtime identity of this invocation (read-only)."""
    import torch
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_head": None if head is None else head.strip(),
            "git_dirty": None if status is None else [line for line in status.splitlines()
                                                      if line.strip()],
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__}


# ------------------------------------------------------------- fixtures

def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def fixture_checks(config: CurrentSemanticsConfig) -> list[dict[str, Any]]:
    """Pure invariants of the registered contract, re-run on demand (G2 evidence)."""
    import torch
    from .features import FEATURE_NAMES, build_features, raw_from_flat, raw_from_nodes
    from .graph import TYPED_GRAPH, UNTYPED_GRAPH, reachable_within
    from .models import CurrentActorCritic, tensors_from_frames
    from .observation import CurrentPhysical, decode, encode

    rng = np.random.default_rng(20261028)
    vectors = []
    for index in range(64):
        ex, ey = rng.uniform(-4.0, 4.0, 2)
        physical = CurrentPhysical(
            ex=ex, ey=ey, h=rng.uniform(0.2, 6.0), vrx=rng.uniform(-2, 2),
            vry=rng.uniform(-2, 2), vgx=rng.uniform(-2, 2), vgy=rng.uniform(-2, 2),
            vz=rng.uniform(-1, 1), roll=rng.uniform(-0.4, 0.4), pitch=rng.uniform(-0.4, 0.4),
            yaw=rng.uniform(-math.pi, math.pi), p=rng.uniform(-1, 1), q=rng.uniform(-1, 1),
            r=rng.uniform(-1, 1), vision_updated=bool(index % 2), vision_age_s=index * 0.05,
            nav_valid=True, nav_age_s=0.0, estimate_valid=index % 16 != 0)
        vectors.append(encode(physical))
    checks = []
    round_trip = max(float(np.max(np.abs(encode(decode(v)) - v))) for v in vectors)
    checks.append(_check("observation_round_trip", round_trip <= 1e-12,
                         {"vectors": len(vectors), "max_abs_error": round_trip}))

    frames = [build_features(v, config) for v in vectors]
    recovered = all(np.array_equal(raw_from_flat(f.flat), f.raw)
                    and np.array_equal(raw_from_nodes(f.nodes), f.raw) for f in frames)
    checks.append(_check("phi44_layout", len(FEATURE_NAMES) == 44 and recovered,
                         {"features": len(FEATURE_NAMES), "raw21_recovered_by_name": recovered}))

    landing = NODE_ORDER[-1]
    reach = {kind: sorted(reachable_within(spec, landing, 2))
             for kind, spec in (("typed", TYPED_GRAPH), ("untyped", UNTYPED_GRAPH))}
    graph_ok = (len(TYPED_GRAPH.edges) == 22 and len(UNTYPED_GRAPH.edges) == 22
                and all(set(nodes) == set(NODE_ORDER) for nodes in reach.values())
                and (NODE_ORDER[5], NODE_ORDER[6]) not in {(s, t) for s, t, _ in TYPED_GRAPH.edges})
    checks.append(_check("graph_22_edges_two_hop_reach", graph_ok,
                         {"edges": len(TYPED_GRAPH.edges), "reach_within_2": reach}))

    design = design_parameter_counts(config.model)
    models = {arm: CurrentActorCritic(arm, config, seed=0) for arm in ARMS}
    measured = {arm: model.parameter_report() for arm, model in models.items()}
    checks.append(_check("parameter_budget",
                         all(measured[arm] == design[arm]
                             and measured[arm]["total"] == DESIGN_PARAMETER_TOTALS[arm]
                             for arm in ARMS),
                         {arm: measured[arm]["total"] for arm in ARMS}))

    reference = models[ARMS[0]].critic.state_dict()
    common = all(all(torch.equal(reference[key], model.critic.state_dict()[key])
                     for key in reference) for model in models.values())
    checks.append(_check("common_raw_critic", common,
                         "seed 0: every arm's critic is bit-identical and reads raw21 only"))

    raw, flat, nodes = tensors_from_frames(frames[:16])
    gradients = {}
    for arm in ("untyped_gnn_ppo", "ontology_rgat_ppo"):
        model = models[arm]
        model.zero_grad()
        model(raw, flat, nodes).square().sum().backward()
        weight = model.encoder.layer.weight.grad
        gradients[arm] = [float(weight[r].abs().sum()) for r in range(weight.shape[0])]
    checks.append(_check("encoder_gradient_every_relation",
                         all(all(g > 0.0 for g in values) for values in gradients.values()),
                         {arm: [f"{g:.3e}" for g in values] for arm, values in gradients.items()}))

    manifest = contract_manifest(config)
    hashes = ("observation_hash", "calibration_hash", "action_hash", "feature_hash",
              "graph_hash", "algorithm_hash")
    checks.append(_check("manifest_complete",
                         manifest["complete"] and all(manifest[key] for key in hashes),
                         {key: manifest[key] for key in hashes}))
    return checks


# ---------------------------------------------------------------- smoke

def run_smoke(config: CurrentSemanticsConfig, *, seed: int, steps: int) -> dict[str, Any]:
    """Connection/shape smoke on the calibrated local plant; never scientific.

    All four untrained arms read every decision (deterministic); the ontology
    arm acts through ``a_max*tanh(z)``.  Its landing behaviour means nothing.
    """
    import torch
    from .backend import make_local_backend, policy_action, require_training_backend
    from .features import build_features
    from .models import CurrentActorCritic, tensors_from_frames

    if steps < 1:
        raise ValueError("steps must be positive")
    require_training_backend("spatial_local_v12", allow_pending_g3=True)
    models = {arm: CurrentActorCritic(arm, config, seed=seed).eval() for arm in ARMS}
    backend = make_local_backend(config)
    finite, constrained, updated, rows = True, 0, 0, []
    termination = None
    try:
        observation, info = backend.reset(seed)
        for _ in range(steps):
            raw, flat, nodes = tensors_from_frames([build_features(observation, config)])
            with torch.no_grad():
                outputs = {arm: model.act(raw, flat, nodes, deterministic=True)
                           for arm, model in models.items()}
            finite &= all(bool(torch.isfinite(value).all())
                          for out in outputs.values() for value in out.values())
            latent = outputs["ontology_rgat_ppo"]["latent"][0].double().numpy()
            step = backend.step(policy_action(latent, config.action.maximum), latent=latent)
            finite &= bool(np.isfinite(step.observation).all())
            constrained += int(step.trace.adapter_constrained)
            updated += int(step.info["observation_provenance"]["frame_used"])
            rows.append({"decision": step.trace.decision_id,
                         "policy_accel": list(step.trace.policy_accel),
                         "adapter_limited_accel": list(step.trace.adapter_limited_accel),
                         "transition_s": step.transition_s})
            observation = step.observation
            if step.done:
                termination = step.info["termination"]
                break
    finally:
        backend.close()
    return {"schema_id": SCHEMA_ID, "stage": "smoke", "backend": "spatial_local_v12",
            "scientific_use": False, "seed": seed, "steps_requested": steps,
            "steps_run": len(rows), "all_finite": bool(finite),
            "adapter_constrained_steps": constrained, "frames_used": updated,
            "termination": termination, "arms": list(ARMS), "acting_arm": "ontology_rgat_ppo",
            "untrained": True, "trace": rows,
            "note": "connection and shape smoke only; untrained weights, no verdict"}


# ---------------------------------------------------------------- gates

def _read(root: Path | None, stage: str) -> dict[str, Any] | None:
    if root is None:
        return None
    path = Path(root) / ARTIFACTS[stage]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gate_table(config: CurrentSemanticsConfig, run_root: Path | None = None, *,
               fixtures: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """G0..G4 (guide 10) from what is on disk now."""
    from .backend import CALIBRATION_GAPS

    if fixtures is None:
        fixtures = fixture_checks(config)
    failed = [check["name"] for check in fixtures if not check["passed"]]
    g3_hash = contract_manifest(config)["g3_hash"]

    def half(stage: str, key: str) -> tuple[bool, bool]:
        report = _read(run_root, stage)
        current = bool(report) and (report.get("hashes") or {}).get("g3_hash") == g3_hash
        passed = bool(report and report.get(key) is True
                      and report.get("schema_id") == SCHEMA_ID and current)
        return passed, bool(report) and not current
    local_ok, local_stale = half("coupling", "local_checks_passed")
    isaac_ok, isaac_stale = half("coupling_isaac", "isaac_checks_passed")
    g3_open = [f"calibration gap: {name} ({gap.status.split(';')[0]})"
               for name, gap in CALIBRATION_GAPS.items()]
    if not isaac_ok:
        g3_open.append("Isaac half not passed: run --stage coupling --backend isaac "
                       "--allow-isaac on a user-approved owned Isaac/PX4 stack")
    for name, stale in (("coupling_local.json", local_stale),
                        ("coupling_isaac.json", isaac_stale)):
        if stale:
            g3_open.insert(0, f"{name} was produced under a different g3_hash")
    g3_status = ("passed" if local_ok and isaac_ok else
                 "partial" if local_ok or isaac_ok else "not_started")
    pilot = _read(run_root, "learnability")
    g4_current = bool(pilot) and pilot.get("hashes") == {
        key: contract_manifest(config)[key]
        for key in ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash", "g3_hash")}
    g4_status, g4_evidence, g4_blocking = "not_started", [], [
        "run ./run.sh current-semantics-3d --stage learnability (G4 pilot)"]
    if pilot and not g4_current:
        g4_blocking = ["g4_pilot.json was produced under a different contract or protocol"]
    elif pilot and not pilot.get("protocol_complete"):
        g4_blocking = ["g4_pilot.json ran a reduced budget; only the registered one judges G4"]
    elif pilot:
        verdicts, tests = pilot.get("verdicts", {}), pilot.get("tests", {})
        g4_evidence = [f"model seed {seed}: not tested ({tests[seed]['note']})"
                       if (tests.get(seed) or {}).get("checkpoint", "") is None else
                       f"model seed {seed}: deterministic {v['deterministic_success']}, "
                       f"sampled {v['sampled_success']} on the test seeds"
                       for seed, v in verdicts.items()]
        g4_status = "passed" if pilot.get("learnability_passed") is True else "failed"
        g4_blocking = [] if g4_status == "passed" else [
            "the registered criteria were not met: the common PPO does not learn this task "
            "under the registered protocol (guide 13: no representation conclusion)"]
    return {
        "G0": {"status": "waived", "evidence": list(_G0_EVIDENCE),
               "blocking": [], "waiver": USER_WAIVERS["G0"], "unresolved": list(_G0_BLOCKING)},
        "G1": {"status": "waived", "evidence": [], "blocking": [],
               "waiver": USER_WAIVERS["G1"], "unresolved": list(_G1_BLOCKING)},
        "G2": {"status": "blocked" if failed else "passed",
               "evidence": [f"fixture checks: {len(fixtures) - len(failed)}/{len(fixtures)} "
                            "passed", "capture-aligned observer (backend.CurrentObserver)",
                            "registered task contract (contracts.TaskContract)",
                            "Isaac 21D quality channels consistent (coupling_isaac.json)"],
               "blocking": [f"fixture failed: {name}" for name in failed],
               "waiver": USER_WAIVERS["G2"]},
        "G3": {"status": g3_status,
               "evidence": ([] + (["local coupling checks passed (coupling_local.json)"]
                                  if local_ok else [])
                            + (["Isaac coupling checks passed (coupling_isaac.json)"]
                               if isaac_ok else [])),
               "blocking": g3_open},
        "G4": {"status": g4_status, "evidence": g4_evidence, "blocking": g4_blocking},
    }


def isaac_eval_state(config: CurrentSemanticsConfig, run_root: Path | None) -> dict[str, Any]:
    """The Isaac evaluation of the selected checkpoints, read only under the current hashes."""
    from .isaac_eval import HASH_KEYS
    report = _read(run_root, "isaac_eval") if run_root is not None else None
    if not report:
        return {"value": None, "reason": "the selected checkpoints have not been flown in Isaac "
                                         "(--stage isaac-eval --allow-isaac)"}
    manifest = contract_manifest(config)
    if report.get("hashes") != {key: manifest[key] for key in HASH_KEYS}:
        return {"value": None, "reason": "isaac_eval.json was flown under a different contract"}
    if not report.get("protocol_complete"):
        return {"value": None, "reason": f"Isaac evaluation incomplete: {report.get('stopped')}"}
    rates = {key: r["deterministic"]["success"] for key, r in report["results"].items()}
    passed = bool(report.get("isaac_eval_passed"))
    return {"value": passed,
            "reason": (f"deterministic Isaac success per model seed {rates} against "
                       f">= {report['criterion']['deterministic_success_min']}, every cleanup "
                       f"{'confirmed' if passed else 'checked'}")}


def report_states(gates: dict[str, dict[str, Any]] | None = None,
                  isaac: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Guide 12's separated states, derived from the gates; never one overall PASSED."""
    gates = gates or {}
    status = lambda gate: (gates.get(gate) or {}).get("status")
    not_run = "no development procedure has been registered or run"
    accepted = status("G2") == "passed" and status("G3") == "passed"
    learn = status("G4")
    return {
        "procedure_complete": {"value": None, "reason": not_run},
        "implementation_acceptance": {
            "value": accepted,
            "reason": ("G2 passed with user waivers and G3 passed (local + Isaac halves)"
                       if accepted else "G2 and G3 must both pass")},
        "learnability_passed": {
            "value": {"passed": True, "failed": False}.get(learn),
            "reason": {"passed": "G4 pilot met the registered criteria",
                       "failed": "G4 pilot missed the registered criteria"}.get(
                learn, "G4 pilot not run under the registered protocol")},
        "development_gate_passed": {"value": None, "reason": not_run},
        "confirmation_opened": {"value": False, "reason": "development gate not run"},
        "scientific_goal_complete": {"value": False, "reason": "no comparison has been run"},
        "isaac_integration_passed": isaac or {
            "value": None, "reason": "the selected checkpoints have not been flown in Isaac"},
    }


def refusal_reasons(gates: dict[str, dict[str, Any]]) -> list[str]:
    return [f"{gate} {gates[gate]['status']}: " + "; ".join(gates[gate]["blocking"][:2])
            for gate in GATES if gates[gate]["status"] not in ("passed", "waived")]


# ---------------------------------------------------------------- plan

def resolved_plan(config_path: Path, stage: str, run_root: Path, *, dry_run: bool,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a stage would read, write and refuse; printed by --dry-run."""
    config = load_config(config_path)
    fixtures = fixture_checks(config)
    gates = gate_table(config, run_root, fixtures=fixtures)
    refused = stage in REFUSED_STAGES or (stage in DEVELOPMENT_STAGES
                                           and bool(refusal_reasons(gates)))
    artifact = ("coupling_isaac" if stage == "coupling" and (extra or {}).get("backend") == "isaac"
                else stage.replace("-", "_"))
    will_write = [] if refused or stage == "report" or artifact not in ARTIFACTS \
        else [str(Path(run_root) / ARTIFACTS[artifact])]
    if (extra or {}).get("pairs"):  # isaac-trace: a timestamped diagnostic folder only
        will_write = [str(extra["out_dir"])]
    if artifact in ("isaac_eval", "isaac_development") and will_write:
        will_write += [str(Path(run_root) / artifact / "episodes.jsonl"),
                       str(Path(run_root) / artifact / "isaac_stack")]
    return {"schema_id": SCHEMA_ID, "stage": stage, "dry_run": dry_run,
            "config_path": str(config_path), "run_root": str(run_root),
            "will_write": will_write, "refused": refused,
            "refusal_reasons": (refusal_reasons(gates) or (
                ["confirmation opens only after a passed development gate (not implemented)"]
                if stage in REFUSED_STAGES else [])) if refused else [],
            "manifest": contract_manifest(config), "gates": gates,
            "report_states": report_states(gates), "provenance": provenance(),
            **(extra or {})}


# ----------------------------------------------------------------- I/O

def write_json(path: Path, payload: Any) -> None:
    """Atomic write (tmp + replace) so a reader never sees half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
                         + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def owned(root: Path) -> Iterator[Path]:
    """Single writer per run root; a lock from a dead pid is cleared.

    The pattern of tools/run_full_pipeline.py: two writers on one root would
    interleave artifacts.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        owner = lock.read_text().strip()
        pid = int(owner.split()[0]) if owner[:1].isdigit() else -1
        alive = pid > 0
        if alive:
            try:
                os.kill(pid, 0)
            except OSError:
                alive = False
        if alive:
            raise SystemExit(f"another run owns {root} (pid {pid}); wait for it or stop it")
        lock.unlink()
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, f"{os.getpid()} {dt.datetime.now().isoformat()}\n".encode())
    os.close(handle)
    try:
        yield root
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()
