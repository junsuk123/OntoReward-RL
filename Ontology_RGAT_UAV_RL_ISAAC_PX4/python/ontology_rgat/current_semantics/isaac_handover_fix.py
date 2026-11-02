"""The handover-kick fix, flown (protocol isaac-handover-fix/1, status doc 16).

isaac-development/1 wrote the seeded handover kick in ONE physics tick; the VN-100 model
clipped it at 16 g and PX4 SITL's int16 FIFO wrapped +16 g to -16 g, so on 16 of its 30
environments PX4's EKF started with the wrong-sign own velocity
(isaac_sim/handover_perturbation.py).  This protocol re-flies identical cases of that
flight -- same environment seeds, conditions, registered checkpoints and deterministic
mode -- after the fix, and pairs every verdict with the pre-fix Isaac verdict and the
local reference already recorded for the same case.

The cases follow a rule fixed before flying, not the outcome: the first N wrapped
environments of the reference flight in seed order, environment i meeting all four arms
of training seed ``training_seeds[i % 5]``.  /2 (2026-11-01, user "3. after edit 2, go")
flies the rest: every reference environment /1 did not, in seed order, with the same
rotation, so after both every environment of isaac-development/1 has flown post-fix
once per arm and every development model six times.  ``load_plan`` recomputes the rule;
``register`` binds the cases, their recorded verdicts, the checkpoints and the Isaac
source of the fix before anything flies; ``summarize`` judges the registered criteria.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import SCHEMA_ID
from . import development as dev
from . import isaac_development as idev
from .contracts import PROJECT_ROOT, load_config

PROTOCOL = "isaac-handover-fix/1"
PROTOCOLS = (PROTOCOL, "isaac-handover-fix/2")
DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "isaac_handover_fix_v1.yaml"
REGISTRATION = "isaac_handover_fix_registration.json"
RESULT = "isaac_handover_fix.json"
OUT_DIR = "isaac_handover_fix"


def files(protocol: str) -> tuple[str, str, str]:
    """(registration, result, output directory) of one protocol; /1 keeps its names."""
    if protocol == PROTOCOL:
        return REGISTRATION, RESULT, OUT_DIR
    n = protocol.rsplit("/", 1)[1]
    return (f"isaac_handover_fix_{n}_registration.json", f"isaac_handover_fix_{n}.json",
            f"isaac_handover_fix_{n}")
# The Isaac source whose behaviour this flight measures; a change after registration
# would fly something else.
FIX_SOURCES = ("isaac_sim/handover_perturbation.py", "isaac_sim/landing_world.py",
               "isaac_sim/vn100_imu.py", "isaac_sim/sensor_profiles.py")
GRAVITY = 9.80665
CRITERIA = ("first_decision_sign_min_m_s", "first_decision_median_error_max_m_s",
            "handover_tilt_window_s", "handover_tilt_max", "success_min")


@dataclass(frozen=True)
class FixPlan:
    environments: int
    criteria: dict[str, float]
    mode: str = "deterministic"
    protocol: str = PROTOCOL
    reference_protocol: str = idev.ISAAC_DEVELOPMENT_PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "reference_protocol": self.reference_protocol,
                "mode": self.mode, "environments": self.environments,
                "criteria": dict(self.criteria)}


def imu_limits() -> tuple[float, float]:
    """(physics_dt, accelerometer range) of the deployment profile Isaac flies."""
    from ..spatial.runtime_contract import deployment_profile
    data = deployment_profile("spatial-reference/12")["resolved_scientific_configuration"]
    accel = ((data.get("imu") or {}).get("accelerometer") or {})
    return (float(data["isaac"]["physics_dt"]),
            float(accel.get("measurement_range_m_s2", 16.0 * GRAVITY)))


def kick(seed: int) -> np.ndarray:
    from ..benchmarks.randomization import sample_domain_randomization
    return np.asarray(sample_domain_randomization(int(seed)).initial_velocity_m_s, dtype=float)


def kick_wrapped(seed: int, physics_dt: float, accel_range: float) -> bool:
    """Did the one-tick kick reach PX4 inverted?  A hovering vehicle's specific force is
    +g up; the step adds k/dt; PX4 receives FRD, and +range wrapped to -range."""
    specific_force_frd = (kick(seed) / physics_dt + (0.0, 0.0, GRAVITY)) * (1.0, -1.0, -1.0)
    return bool(np.any(specific_force_frd >= accel_range))


def load_plan(path: str | Path) -> FixPlan:
    data = dev._section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                        ("protocol", "reference_protocol", "mode", "environments", "criteria"),
                        "isaac_handover_fix")
    if data["protocol"] not in PROTOCOLS:
        raise ValueError(f"isaac_handover_fix.protocol must be one of {PROTOCOLS}")
    if data["reference_protocol"] != idev.ISAAC_DEVELOPMENT_PROTOCOL:
        raise ValueError("isaac_handover_fix re-flies isaac-development/1 cases only")
    if data["mode"] != "deterministic":
        raise ValueError("isaac_handover_fix.mode must be deterministic, as the reference")
    count = data["environments"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("isaac_handover_fix.environments must be a positive integer")
    criteria = dev._section(data["criteria"], CRITERIA, "isaac_handover_fix.criteria")
    for name, value in criteria.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f"isaac_handover_fix.criteria.{name} must be a finite number >= 0")
    return FixPlan(environments=count, criteria={k: float(v) for k, v in criteria.items()},
                   protocol=data["protocol"])


def select_cases(plan: FixPlan, reference: idev.IsaacDevelopmentPlan,
                 dev_plan: dev.DevelopmentPlan) -> list[tuple[str, int, str]]:
    """The registered rule: (condition, environment seed, model), environment-major."""
    physics_dt, accel_range = imu_limits()
    condition = {seed: cell for cell, seeds in reference.conditions for seed in seeds}
    if plan.protocol == PROTOCOL:
        wrapped = [s for s in sorted(condition) if kick_wrapped(s, physics_dt, accel_range)]
        if len(wrapped) < plan.environments:
            raise ValueError(f"the reference flight has only {len(wrapped)} wrapped "
                             "environments")
        chosen = wrapped[:plan.environments]
    else:  # /2: every environment /1 did not fly, in seed order
        first = {s for _, s, _ in select_cases(load_plan(DEFAULT_PATH), reference, dev_plan)}
        chosen = [s for s in sorted(condition) if s not in first]
        if len(chosen) != plan.environments:
            raise ValueError(f"{plan.protocol} flies the {len(chosen)} environments /1 did "
                             f"not; the plan says {plan.environments}")
    seeds = dev_plan.training_seeds
    return [(condition[s], s, f"{seeds[i % len(seeds)]}/{arm}")
            for i, s in enumerate(chosen) for arm in dev_plan.arms]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: _sha256(PROJECT_ROOT / name) for name in FIX_SOURCES}


def plan_hash(plan: FixPlan, cases: list[tuple[str, int, str]], reference_hash: str) -> str:
    from ..direct_policy.contracts import digest
    return digest({"plan": plan.to_dict(), "cases": [list(c) for c in cases],
                   "reference_plan_hash": reference_hash})


def _recorded(root: Path) -> tuple[dict, dict]:
    """Pre-fix Isaac rows and the paired local reference of the reference flight."""
    out = root / idev.OUT_DIR
    read = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    key = lambda r: (r["condition"], int(r["seed"]), r["model"])
    return ({key(r): r for r in read(out / "episodes.jsonl")},
            {key(r): r for r in read(out / idev.LOCAL_REFERENCE)})


def prepare(config_path: Path, plan_path: Path, root: Path):
    """Everything the stage needs, or a refusal reason: (problem, context)."""
    config = load_config(config_path)
    dev_plan = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    reference = idev.load_plan(idev.DEFAULT_PATH, config, dev_plan)
    problem = idev.check_registration(config, reference, dev_plan, root)
    if problem:
        return f"reference flight: {problem}", None
    plan = load_plan(plan_path)
    cases = select_cases(plan, reference, dev_plan)
    before, local = _recorded(root)
    missing = [c for c in cases if c not in before or c not in local]
    if missing:
        return f"the reference flight has no Isaac or local verdict for {missing[:3]}", None
    ref_registration = json.loads((root / idev.REGISTRATION).read_text(encoding="utf-8"))
    return None, {"config": config, "dev_plan": dev_plan, "reference": reference,
                  "plan": plan, "cases": cases, "before": before, "local": local,
                  "plan_hash": plan_hash(plan, cases, ref_registration["plan_hash"]),
                  "reference_registration": ref_registration}


def registration(ctx: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    before, local = ctx["before"], ctx["local"]
    checkpoints = ctx["reference_registration"]["checkpoints"]
    return {"schema_id": SCHEMA_ID, "stage": "isaac-handover-fix", "status": "REGISTERED",
            "protocol": ctx["plan"].protocol, "plan_hash": ctx["plan_hash"],
            "plan_file": str(plan_path),
            "plan": ctx["plan"].to_dict(),
            "reference_plan_hash": ctx["reference_registration"]["plan_hash"],
            "cases": [{"condition": c, "seed": s, "model": k,
                       "kick_body_m_s": kick(s).tolist(),
                       "before_status": before[(c, s, k)]["status"],
                       "local_status": local[(c, s, k)]["status"]}
                      for c, s, k in ctx["cases"]],
            "checkpoints": {k: checkpoints[k] for k in sorted({k for _, _, k in ctx["cases"]})},
            "fix_sources": source_hashes(),
            "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": "registered before any case flew after the fix; criteria in the plan"}


def check_registration(ctx: dict[str, Any], root: Path) -> str | None:
    path = root / files(ctx["plan"].protocol)[0]
    if not path.exists():
        return None
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("plan_hash") != ctx["plan_hash"]:
        return f"{path} registers another plan; archive it first"
    if recorded.get("fix_sources") != source_hashes():
        return "the Isaac source of the fix changed after registration"
    return None


def first_decision(row: dict[str, Any], trace_first: dict[str, Any]) -> dict[str, Any]:
    """PX4's own-velocity change over the first decision against the yaw-rotated kick."""
    k = kick(row["seed"])
    yaw = float(row["scenario"]["entry_yaw_enu_rad"])
    c, s = math.cos(yaw), math.sin(yaw)
    expected = np.array([c * k[0] - s * k[1], s * k[0] + c * k[1], k[2]])
    measured = np.asarray(trace_first["measured"], dtype=float) * trace_first["measured_over_s"]
    return {"expected": expected.tolist(), "measured": measured.tolist(),
            "abs_error": np.abs(measured - expected).tolist()}


def summarize(plan: FixPlan, cases: list[tuple[str, int, str]], rows: list[dict[str, Any]],
              before: dict, local: dict, traces: Path) -> dict[str, Any]:
    crit = plan.criteria
    flown = {(r["condition"], int(r["seed"]), r["model"]): r for r in rows
             if r.get("status") is not None}
    ok = lambda status: status == "SUCCESS"
    tilt = lambda r: (r["status"] == "SAFETY_ENVELOPE_VIOLATION"
                      and r["elapsed_s"] <= crit["handover_tilt_window_s"])
    errors, sign_right, per_case = [], 0, []
    for case in cases:
        row = flown.get(case)
        entry = {"condition": case[0], "seed": case[1], "model": case[2],
                 "before": before[case]["status"], "local": local[case]["status"],
                 "after": row["status"] if row else None}
        if row:
            name = f"{case[0]}_seed{case[1]}_{case[2].replace('/', '_')}.jsonl"
            with (traces / name).open(encoding="utf-8") as handle:
                fd = first_decision(row, json.loads(handle.readline()))
            big = np.abs(fd["expected"]) >= crit["first_decision_sign_min_m_s"]
            right = bool(np.all(np.sign(np.asarray(fd["measured"])[big])
                                == np.sign(np.asarray(fd["expected"])[big])))
            sign_right += right
            errors += fd["abs_error"]
            entry.update(first_decision=fd, sign_right=right, handover_tilt=tilt(row),
                         max_tilt_deg=row.get("max_tilt_deg"), elapsed_s=row.get("elapsed_s"))
        per_case.append(entry)
    n = len(flown)
    complete = n == len(cases)
    after_ok = sum(ok(r["status"]) for r in flown.values())
    tilts = sum(tilt(r) for r in flown.values())
    median_error = float(np.median(errors)) if errors else None
    pairs = lambda a, b: {"both": sum(ok(x[a]) and ok(x[b]) for x in per_case if x["after"]),
                          f"{a}_only": sum(ok(x[a]) and not ok(x[b]) for x in per_case if x["after"]),
                          f"{b}_only": sum(ok(x[b]) and not ok(x[a]) for x in per_case if x["after"]),
                          "neither": sum(not ok(x[a]) and not ok(x[b])
                                         for x in per_case if x["after"])}
    checks = {
        "first_decision_sign": {"right": sign_right, "of": n, "passed": complete and sign_right == n},
        "first_decision_median_error": {
            "value_m_s": median_error,
            "max_m_s": crit["first_decision_median_error_max_m_s"],
            "passed": complete and median_error is not None
            and median_error <= crit["first_decision_median_error_max_m_s"]},
        "handover_tilt": {"count": tilts, "max": crit["handover_tilt_max"],
                          "passed": complete and tilts <= crit["handover_tilt_max"]},
        "success": {"count": after_ok, "min": crit["success_min"],
                    "passed": complete and after_ok >= crit["success_min"]}}
    by_arm = {}
    for arm in sorted({k.split("/")[1] for _, _, k in cases}):
        mine = [x for x in per_case if x["model"].endswith("/" + arm) and x["after"]]
        by_arm[arm] = {"episodes": len(mine), "after": sum(ok(x["after"]) for x in mine),
                       "before": sum(ok(x["before"]) for x in mine),
                       "local": sum(ok(x["local"]) for x in mine)}
    count = lambda key: {s: sum(x[key] == s for x in per_case if x["after"])
                         for s in sorted({x[key] for x in per_case if x["after"]})}
    return {"episodes": n, "planned": len(cases), "complete": complete,
            "success": {"after": after_ok,
                        "before": sum(ok(before[c]["status"]) for c in flown),
                        "local": sum(ok(local[c]["status"]) for c in flown)},
            "handover_tilt": {"after": tilts, "before": sum(tilt(before[c]) for c in flown)},
            "statuses": {"after": count("after"), "before": count("before"),
                         "local": count("local")},
            "paired_after_vs_before": pairs("after", "before"),
            "paired_after_vs_local": pairs("after", "local"),
            "by_arm": by_arm, "checks": checks,
            "fix_verified": complete and all(c["passed"] for c in checks.values()),
            "cases": per_case}
