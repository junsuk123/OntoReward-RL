"""orl-development/1-/4: ORL as a paired arm of development/1 (status doc 20).

The question is the CNN-vs-MLP one: with the same phi44 input and the same parameter
budget, does putting the ontology into the policy's structure (orl.py) beat the
unstructured semantic MLP?  ORL trains on development/1's training seeds, training
episodes, validation and evaluation environments (common random numbers; nothing is
selected on them) and is compared, paired by training seed, with the stored
development/1 models.  The rule is fixed here and hashed into the registration before
any ORL development model trains; registration also requires a passed ORL G4
learnability pilot under the same ORL hash.  Local plant only.

Each protocol binds one ORL version (``PROTOCOLS``) and writes under its own directory
(``OUTS``): orl-development/1 is orl/1 in ``<root>/orl``, orl-development/2 is orl/2 in
``<root>/orl2``.  orl/1's gate turned out inert (status doc 20.4), so /2 adds the GATE
PROBE (``GATE_PROBE``).  It flies each selected checkpoint through the training blackout
and measures whether the gate closes when the policy's own estimate says the pad has not
been seen.  Registration of /2 requires the pilot checkpoints' gates to be active.
orl/2's gate learned but its closed branch still sank (status doc 20.6), so /3 (orl/3, in
``<root>/orl3``) uses the second probe (``GATE_PROBES["orl/3"]``): the same drop AND a
blind sink no faster than ``max_blind_sink_m_s`` by the policy's own vz.  It also reports,
never decisively, the final checkpoint of each pilot seed.
orl/3's gate stayed half open when blind (status doc 20.8), so /4 (orl/4, in ``<root>/orl4``)
replaces the learned closing by the task's view rule (view_rule.py) and therefore compares
against a CONTROL carrying the same rule (``CONTROLS``: rule_control.semantic_mlp_rule_ppo,
trained here in ``<root>/orl4/control``), not against the plain semantic MLP.  Both arms run
the G4 pilot and probe 2 before registration, and both are trained on every training seed.
Either arm can be trained by its own process (``part``), each under its own lock.
orl-development/5 (orl/5 in ``<root>/orl5``, status doc 20.12) is /4's design with
view-rule/2 in BOTH arms: the closed rule tracks the task field's lost-view climb (and,
with ``view_rule.ALIGNMENT_EDGE``, also closes outside the field's alignment cone).  Its
control is semantic-mlp-rule/2 (``rule_control.CONTROL_ARM_V2``) in ``<root>/orl5/control``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import json
import multiprocessing
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import yaml

from . import SCHEMA_ID
from .contracts import PROJECT_ROOT, CurrentSemanticsConfig, contract_manifest, load_config
from .orl import ARMS, CURRENT_VERSION, HASHES, spec_dict, version_of
from .rule_control import (CONTROL_ARM, CONTROL_ARM_V2, CONTROL_HASHES, control_version,
                           spec_dict as control_spec_dict)

PROTOCOLS = MappingProxyType({"orl-development/1": "orl/1", "orl-development/2": "orl/2",
                              "orl-development/3": "orl/3", "orl-development/4": "orl/4",
                              "orl-development/5": "orl/5"})
BASE_PROTOCOL = "development/1"
OUTS = MappingProxyType({"orl/1": "orl", "orl/2": "orl2", "orl/3": "orl3", "orl/4": "orl4",
                        "orl/5": "orl5"})
# The rule controls (status doc 20.9, 20.12): trained and compared here, in <out>/CONTROL_DIR.
CONTROLS = MappingProxyType({"orl/4": CONTROL_ARM, "orl/5": CONTROL_ARM_V2})
CONTROL_DIR = "control"
PARTS = ("orl", "control")
DEFAULT_PLAN_PATHS = MappingProxyType({
    version: PROJECT_ROOT / "config" / "current_semantics"
    / f"orl_development_v{version.split('/')[1]}.yaml" for version in OUTS})
DEFAULT_PLAN_PATH = DEFAULT_PLAN_PATHS[CURRENT_VERSION]
REGISTRATION = "orl_development_registration.json"
RESULT = "orl_development.json"
LEARNABILITY = "orl_learnability.json"
STAGES = ("register-orl-development", "orl-develop")
PRIMARY = "semantic_mlp_ppo"
SECONDARY = ("raw_ppo", "untyped_gnn_ppo", "ontology_rgat_ppo")
RULE = (
    "Primary: mean SUCCESS over the ten development/1 cells; per training seed, delta = ORL "
    "minus the stored development/1 semantic_mlp_ppo (same phi44 input, same budget, no "
    "structure). ORL is called BETTER only if the two-sided 95 % t-interval of delta over the "
    "five training seeds lies above 0, WORSE only if it lies below 0, otherwise 'not "
    "established'. Every ORL model is trained, evaluated and reported; none is dropped. A "
    "model with no checkpoint at difficulty 1.0 scores 0 in every cell, as in development/1. "
    "Secondary, reported and never decisive: ORL minus raw_ppo, untyped_gnn_ppo and "
    "ontology_rgat_ppo on the same metric, and ID mean, four-OOD mean, composition_ood and "
    "OOD unsafe for every comparison; validation learnability of each ORL model.")
# The gate probe (orl/2, status doc 20.5).  Probe seeds 999101-999124 were never used by
# any registered family, audited range or diagnostic (contracts.DIAGNOSTIC_SEED_RANGES).
# The blackout is the TRAINING blackout (g4.training_randomization.vision_blackout), forced
# on, so the probe is in distribution.  Decisions are classed by the policy's OWN decoded
# observation: estimate valid and estimated height above ``high_m``, and a vision age of at
# most ``fresh_age_s`` (fresh) or above ``blind_age_s`` (blind; the age at which the
# reward-/4 lost-view field and the 20.4 blind-brake overlay start).
GATE_PROBE = MappingProxyType({
    "protocol": "orl-gate-probe/1", "seeds": tuple(range(999101, 999125)),
    "split": "in_distribution", "difficulty": 1.0, "deterministic": True,
    "blackout": "the training vision_blackout with probability 1", "high_m": 0.5,
    "fresh_age_s": 0.2, "blind_age_s": 0.5, "min_decisions": 50, "min_gate_drop": 0.2})
GATE_RULE = (
    "A checkpoint's gate is ACTIVE when, over the probe episodes, at least min_decisions "
    "fresh and min_decisions blind decisions occurred above high_m and the mean gate "
    "sigmoid(gate(E)) on fresh decisions exceeds the mean on blind ones by at least "
    "min_gate_drop. orl-development/2 registers only after every selected pilot checkpoint "
    "passed G4 AND has an active gate.")
RULE_V2 = RULE + (
    " orl-development/2 additionally probes the gate of every development model "
    "(GATE_PROBE, same seeds). The verdict is reported with the number of models whose "
    "gate is active; if any is inert the verdict is not read as a test of the gate.")
# The second probe (orl/3, status doc 20.7).  The same seeds, blackout and classes as the
# first.  ``max_blind_sink_m_s`` adds the closed gate's PURPOSE: on blind decisions the
# policy's own estimate of its vertical velocity (decoded drone_vz, up positive) averages a
# sink of at most that speed.  orl/2's pilot read -0.17..-0.20 (seed 0) and -0.25..-0.32
# (seed 1) there (status doc 20.6).  ``report_final``: each pilot seed's final.pt is probed
# and reported as well, never decisively (its best.pt can predate the gate, 20.6).
GATE_PROBE_V2 = MappingProxyType({**GATE_PROBE, "protocol": "orl-gate-probe/2",
                                  "max_blind_sink_m_s": 0.10, "report_final": True})
GATE_RULE_V2 = GATE_RULE + (
    " Probe 2 (orl/3) additionally requires the mean decoded own vertical velocity on the "
    "blind decisions to be at least -max_blind_sink_m_s (a closed gate must stop the "
    "descent, not only the gate value move). The final checkpoint of each pilot seed is "
    "probed and reported; only the selected (best) checkpoints decide registration.")
GATE_RULE_V4 = GATE_RULE_V2 + (
    " orl/4 (status doc 20.9): the same probe also runs on the rule control "
    "(semantic_mlp_rule_ppo), whose gate is the rule's open fraction 1 - w; "
    "orl-development/4 registers only after BOTH arms' pilots passed G4 and every selected "
    "checkpoint of both arms is active.")
GATE_RULE_V5 = GATE_RULE_V2 + (
    " orl/5 (status doc 20.12): the same probe also runs on its rule control "
    "(semantic-mlp-rule/2), whose gate is the rule's open fraction 1 - W; orl-development/5 "
    "registers only after BOTH arms' pilots passed G4 and every selected checkpoint of both "
    "arms is active.")
GATE_PROBES = MappingProxyType({"orl/2": GATE_PROBE, "orl/3": GATE_PROBE_V2,
                                "orl/4": GATE_PROBE_V2, "orl/5": GATE_PROBE_V2})
GATE_RULES = MappingProxyType({"orl/2": GATE_RULE, "orl/3": GATE_RULE_V2,
                               "orl/4": GATE_RULE_V4, "orl/5": GATE_RULE_V5})
RULE_V3 = RULE + (
    " orl-development/3 additionally probes the gate of every development model with "
    "probe 2 (GATE_PROBES['orl/3'], same seeds). The verdict is reported with the number "
    "of models whose gate is active; if any is inert the verdict is not read as a test of "
    "the gate.")
RULE_V4 = (
    "Primary: mean SUCCESS over the ten development/1 cells; per training seed, delta = orl/4 "
    "minus the rule control semantic_mlp_rule_ppo trained here on the same training seeds "
    "(the registered semantic MLP with the SAME view rule: same phi44 input, same rule, no "
    "ontology structure). orl/4 is BETTER only if the two-sided 95 % t-interval of delta over "
    "the five training seeds lies above 0, WORSE only if it lies below 0, otherwise 'not "
    "established'. Every model of both arms is trained, evaluated and reported; none is "
    "dropped. A model with no checkpoint at difficulty 1.0 scores 0 in every cell, as in "
    "development/1. Secondary, reported and never decisive: orl/4 minus the stored "
    "development/1 semantic_mlp_ppo, raw_ppo, untyped_gnn_ppo and ontology_rgat_ppo; the rule "
    "control minus the stored semantic_mlp_ppo (what the rule alone does to the MLP); ID mean, "
    "four-OOD mean, composition_ood and OOD unsafe for every comparison; validation "
    "learnability of each model. Every development model of both arms is probed with probe 2 "
    "(GATE_PROBES['orl/4'], same seeds); the verdict reports how many are active, and if any "
    "is inert it is not read as a test of the rule.")
RULE_V5 = (
    "Primary: mean SUCCESS over the ten development/1 cells; per training seed, delta = orl/5 "
    "minus the rule control semantic-mlp-rule/2 trained here on the same training seeds "
    "(the registered semantic MLP with the SAME view-rule/2: same phi44 input, same rule, no "
    "ontology structure). orl/5 is BETTER only if the two-sided 95 % t-interval of delta over "
    "the five training seeds lies above 0, WORSE only if it lies below 0, otherwise 'not "
    "established'. Every model of both arms is trained, evaluated and reported; none is "
    "dropped. A model with no checkpoint at difficulty 1.0 scores 0 in every cell, as in "
    "development/1. Secondary, reported and never decisive: orl/5 minus the stored "
    "development/1 semantic_mlp_ppo, raw_ppo, untyped_gnn_ppo and ontology_rgat_ppo; the rule "
    "control minus the stored semantic_mlp_ppo; ID mean, four-OOD mean, composition_ood and "
    "OOD unsafe for every comparison; validation learnability of each model. Every "
    "development model of both arms is probed with probe 2 (GATE_PROBES['orl/5'], same "
    "seeds); the verdict reports how many are active, and if any is inert it is not read as "
    "a test of the rule.")
RULES = MappingProxyType({"orl-development/1": RULE, "orl-development/2": RULE_V2,
                          "orl-development/3": RULE_V3, "orl-development/4": RULE_V4,
                          "orl-development/5": RULE_V5})


@dataclass(frozen=True)
class ORLPlan:
    protocol: str
    base_protocol: str
    arm: str
    workers: int

    @property
    def version(self) -> str:
        return PROTOCOLS[self.protocol]


def plan_version(path: str | Path) -> str:
    """The ORL version a plan file's protocol binds (before any other check)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    protocol = data.get("protocol") if isinstance(data, dict) else None
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}, got {protocol!r}")
    return PROTOCOLS[protocol]


def load_plan(path: str | Path, config: CurrentSemanticsConfig):
    from .development import DEFAULT_DEVELOPMENT_PATH, DEVELOPMENT_TASK, load_development
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = set(ORLPlan.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"ORL development plan needs exactly the keys {sorted(keys)}")
    if data["protocol"] not in PROTOCOLS or data["base_protocol"] != BASE_PROTOCOL:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)} on {BASE_PROTOCOL}")
    protocol = data["protocol"]
    arm = ARMS[PROTOCOLS[protocol]]
    if data["arm"] != arm:
        raise ValueError(f"plan arm {data['arm']!r} is not this code's ORL arm {arm} "
                         f"({PROTOCOLS[protocol]})")
    if DEVELOPMENT_TASK[BASE_PROTOCOL] != config.task.version:
        raise ValueError(f"{protocol} extends {BASE_PROTOCOL}, a "
                         f"{DEVELOPMENT_TASK[BASE_PROTOCOL]} plan; this config is "
                         f"{config.task.version}")
    workers = data["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive int")
    plan = ORLPlan(protocol, BASE_PROTOCOL, arm, workers)
    return plan, load_development(DEFAULT_DEVELOPMENT_PATH, config)


def probe_version(arm: str) -> str:
    """The ORL version whose gate probe an arm runs: its own, or that of the ORL version a
    rule control belongs to."""
    for version, control in CONTROLS.items():
        if arm == control:
            return version
    return version_of(arm)


def control_hash(version: str) -> str:
    return CONTROL_HASHES[control_version(CONTROLS[version])]


def control_spec(version: str) -> dict[str, Any]:
    return control_spec_dict(control_version(CONTROLS[version]))


def _probe_spec(version: str = "orl/2") -> dict[str, Any]:
    return {**{k: list(v) if isinstance(v, tuple) else v
               for k, v in GATE_PROBES[version].items()}, "rule": GATE_RULES[version]}


def probe_hash(version: str = "orl/2") -> str:
    from ..direct_policy.contracts import digest
    return digest(_probe_spec(version))


def plan_hash(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    from .development import plan_hash as development_hash
    record = {"plan": asdict(plan), "orl_hash": HASHES[plan.version],
              "rule": RULES[plan.protocol],
              "development_plan_hash": development_hash(development, config)}
    if plan.version != "orl/1":  # orl/1's registered hash predates the probe
        record["gate_probe"] = _probe_spec(plan.version)
    if plan.version in CONTROLS:
        record["control"] = {"arm": CONTROLS[plan.version], "hash": control_hash(plan.version)}
    return digest(record)


def out_dir(root: Path, version: str) -> Path:
    return root / OUTS[version]


def part_dir(root: Path, version: str, part: str = "orl") -> Path:
    """Where one arm of a protocol writes: the ORL arm in ``out_dir``, the rule control in
    ``out_dir / CONTROL_DIR``."""
    if part not in PARTS or (part == "control" and version not in CONTROLS):
        raise ValueError(f"{version} has no {part!r} part")
    return out_dir(root, version) if part == "orl" else out_dir(root, version) / CONTROL_DIR


def part_arm(version: str, part: str = "orl") -> str:
    return ARMS[version] if part == "orl" else CONTROLS[version]


def pilot_record(version: str, part: str = "orl") -> tuple[str, dict[str, Any]]:
    """(key, record) a learnability pilot stores to say which spec it trained."""
    if part == "orl":
        return "orl", {"version": version, "hash": HASHES[version], "spec": spec_dict(version)}
    return "control", {"version": version, "arm": CONTROLS[version],
                       "hash": control_hash(version), "spec": control_spec(version)}


def _dev_model(root: Path, training_seed: int, arm: str) -> Path:
    return root / "development" / f"seed{training_seed}" / arm


def _model_root(root: Path, training_seed: int, version: str, part: str = "orl") -> Path:
    return part_dir(root, version, part) / "development" / f"seed{training_seed}"


def blocker(config: CurrentSemanticsConfig, development, root: Path,
            version: str = CURRENT_VERSION) -> str | None:
    """None when development/1 is complete for every compared arm and the ORL G4 pilot
    passed under this ORL hash and contract (orl/2: with an active gate); otherwise why
    not."""
    from .development import REGISTRATION as DEV_REGISTRATION, plan_hash as development_hash
    path = root / DEV_REGISTRATION
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("plan_hash") \
            != development_hash(development, config):
        return f"{BASE_PROTOCOL} is not registered in {root} under this plan"
    for s in development.training_seeds:
        for arm in (PRIMARY, *SECONDARY):
            summary = _dev_model(root, s, arm) / "summary.json"
            if not summary.exists() or not json.loads(
                    summary.read_text(encoding="utf-8")).get("complete"):
                return f"{BASE_PROTOCOL} model {s}/{arm} is not complete"
    return pilot_blocker(config, root, version)


def pilot_blocker(config: CurrentSemanticsConfig, root: Path,
                  version: str = CURRENT_VERSION) -> str | None:
    """None when ``<root>/<OUTS[version]>/orl_learnability.json`` passed under this ORL
    spec and this contract (orl/2-/4: and the version's gate probe found every gate
    active), and, for a version with a rule control (orl/4), when the control's pilot in
    ``<out>/control`` did the same under the control's spec.
    ``pilot.run_pilot(arm=...)`` runs the registered G4 protocol with ``g4.arm`` replaced,
    so its ``g4_hash`` is the protocol's hash WITH the pilot's arm, not the Raw one."""
    for part in PARTS if version in CONTROLS else ("orl",):
        reason = _part_pilot_blocker(config, root, version, part)
        if reason:
            return reason
    return None


def _part_pilot_blocker(config: CurrentSemanticsConfig, root: Path, version: str,
                        part: str) -> str | None:
    arm, label = part_arm(version, part), ("ORL" if part == "orl" else "rule control")
    pilot = part_dir(root, version, part) / LEARNABILITY
    if not pilot.exists():
        return f"no {label} G4 learnability pilot ({pilot})"
    data = json.loads(pilot.read_text(encoding="utf-8"))
    manifest = contract_manifest(replace(config, g4=replace(config.g4, arm=arm)))
    key, record = pilot_record(version, part)
    if (data.get(key) or {}).get("hash") != record["hash"] or data.get("arm") != arm:
        return f"the {label} learnability pilot ran another {label} spec"
    if any(data.get("hashes", {}).get(k) != manifest[k]
           for k in ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash")):
        return f"the {label} learnability pilot ran under another contract"
    if not (data.get("protocol_complete") and data.get("learnability_passed")):
        return f"the {label} learnability pilot did not pass"
    if version != "orl/1":
        probe = data.get("gate_probe") or {}
        if probe.get("probe_hash") != probe_hash(version):
            return f"the {label} pilot's gate was not probed under this gate probe"
        if not probe.get("gate_active"):
            return (f"the {label} pilot's gate is inert (gate probe); a development run would "
                    "not test the gate")
    return None


def resolved(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> dict[str, Any]:
    from .development import CELLS
    g4, models = config.g4, len(development.training_seeds)
    extra = ({} if plan.version == "orl/1" else
             {"gate_probe": _probe_spec(plan.version), "probe_hash": probe_hash(plan.version)})
    arms = 1
    if plan.version in CONTROLS:
        arms = 2
        extra["control"] = {"arm": CONTROLS[plan.version], "hash": control_hash(plan.version),
                            "spec": control_spec(plan.version),
                            "out": f"{OUTS[plan.version]}/{CONTROL_DIR}"}
    return {"protocol": plan.protocol, "base_protocol": BASE_PROTOCOL, "arm": plan.arm,
            "orl_version": plan.version, "orl_hash": HASHES[plan.version],
            "orl_spec": spec_dict(plan.version), **extra,
            "plan_hash": plan_hash(plan, development, config),
            "training_seeds": list(development.training_seeds), "models": models,
            "updates": g4.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": arms * models * g4.updates * g4.episodes_per_update,
            "evaluation_episodes": arms * models * len(CELLS) * development.episodes_per_cell,
            "rule": RULES[plan.protocol],
            "primary_comparison": (CONTROLS[plan.version] if plan.version in CONTROLS
                                   else PRIMARY),
            "secondary": list(SECONDARY),
            "seeds": f"{BASE_PROTOCOL}'s training, validation and evaluation seeds, reused on "
                     "purpose (a paired arm); nothing is selected on them",
            "backend": "spatial_local_v12", "trains": True, "starts_isaac": False}


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    reason = blocker(config, development, root, plan.version)
    if reason:
        return 3, {"stage": STAGES[0], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    path = out_dir(root, plan.version) / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest_:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {"stage": STAGES[0], "status": "REFUSED",
                   "reason": f"{path.name} registers another plan; archive it first"}
    manifest = contract_manifest(config)
    return 0, {"schema_id": SCHEMA_ID, "stage": STAGES[0], "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_file": str(plan_path), **resolved(plan, development, config),
               "contract": {k: manifest[k] for k in ("algorithm_hash", "task_hash",
                                                     "feature_hash", "calibration_hash",
                                                     "g4_hash")}}


def check_registration(plan, development, config, root: Path) -> str | None:
    path = out_dir(root, plan.version) / REGISTRATION
    if not path.exists():
        return f"no {plan.protocol} registration: run --stage {STAGES[0]} first"
    if json.loads(path.read_text(encoding="utf-8")).get("plan_hash") != \
            plan_hash(plan, development, config):
        return "the plan, the ORL spec or the contract differs from the registration"
    return None


def judge(development, orl: dict[int, dict], stored: dict[str, dict[int, dict]],
          protocol: str = "orl-development/1",
          control: dict[int, dict] | None = None) -> dict[str, Any]:
    """``orl``: training seed -> ORL summary; ``stored``: arm -> training seed -> summary;
    ``control`` (orl-development/4): training seed -> rule-control summary, the primary
    comparison."""
    from .development import CELLS, FOUR_OOD, ID_CELLS, OOD_CELLS
    from .relation_control import _t_interval
    seeds = list(development.training_seeds)
    metrics = {"all_cells": (CELLS, "success"), "id_mean": (ID_CELLS, "success"),
               "four_ood_mean": (FOUR_OOD, "success"),
               "composition_ood": (("composition_ood",), "success"),
               "ood_unsafe": (OOD_CELLS, "unsafe")}

    def score(summary, cells, key):
        return float(np.mean([summary["cells"][c][key] for c in cells]))

    ruled = PROTOCOLS[protocol] in CONTROLS
    if ruled != (control is not None):
        raise ValueError(f"{protocol} {'needs' if ruled else 'has no'} rule-control summaries")
    out = {}
    for name, (cells, key) in metrics.items():
        mine = [score(orl[s], cells, key) for s in seeds]
        out[name] = {"orl": mine, **{arm: [score(stored[arm][s], cells, key) for s in seeds]
                                     for arm in (PRIMARY, *SECONDARY)}}
        for arm in (PRIMARY, *SECONDARY):
            out[name][f"orl_minus_{arm}"] = _t_interval(
                [a - b for a, b in zip(mine, out[name][arm])])
        if ruled:
            theirs = [score(control[s], cells, key) for s in seeds]
            out[name]["control"] = theirs
            out[name]["orl_minus_control"] = _t_interval(
                [a - b for a, b in zip(mine, theirs)])
            out[name][f"control_minus_{PRIMARY}"] = _t_interval(
                [a - b for a, b in zip(theirs, out[name][PRIMARY])])
    primary = out["all_cells"]["orl_minus_control" if ruled else f"orl_minus_{PRIMARY}"]
    low, high = primary["ci95"]
    word = "better" if low > 0 else "worse" if high < 0 else "not established"
    verdict = {"rule": RULES[protocol], "primary": primary,
               ("orl_vs_rule_control" if ruled else "orl_vs_semantic_mlp"): word,
               "metrics": out,
               "validation": {str(s): orl[s].get("validation") for s in seeds}}
    if ruled:
        verdict["control_validation"] = {str(s): control[s].get("validation") for s in seeds}
    if PROTOCOLS[protocol] != "orl/1":
        models = [orl[s] for s in seeds] + ([control[s] for s in seeds] if ruled else [])
        active = [m for m in models if (m.get("gate_probe") or {}).get("gate_active")]
        verdict["gate_active_models"] = f"{len(active)}/{len(models)}"
        verdict["reads_as_test_of_the_gate"] = len(active) == len(models)
    return verdict


def run(config_path: Path, plan_path: Path, root: Path, *, log: Callable[[str], None] = print,
        updates: int | None = None, episodes_per_cell: int | None = None,
        workers: int | None = None, part: str | None = None) -> tuple[int, dict[str, Any]]:
    """Train, select and evaluate ORL (and, on orl-development/4, the rule control) on every
    development/1 training seed; resumable.  ``part`` trains one arm only ("orl" or
    "control"), so the two can run as separate processes; the verdict is computed once
    every model of every arm has a summary, by whichever call finds them all."""
    from .development import CELLS, _status, _validation_of, arm_config, evaluate_model
    from .experiment import write_json
    from .models import build_model, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    reason = blocker(config, development, root, plan.version) or \
        check_registration(plan, development, config, root)
    if reason:
        return 3, {"stage": STAGES[1], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    if episodes_per_cell is not None:
        development = replace(development, episodes_per_cell=int(episodes_per_cell))
    complete = updates is None and episodes_per_cell is None
    version, protocol = plan.version, plan.protocol
    parts = PARTS if version in CONTROLS else ("orl",)
    if part is not None and part not in parts:
        return 3, {"stage": STAGES[1], "status": "REFUSED",
                   "reason": f"{protocol} has no {part!r} part (parts {list(parts)})"}
    results: dict[str, dict[int, dict]] = {p: {} for p in parts}
    started = time.monotonic()
    count = plan.workers if workers is None else int(workers)

    def stored_summary(p, s):
        path = _model_root(root, s, version, p) / "summary.json"
        if not path.exists():
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("plan_hash") != digest_ or summary.get("complete") != complete:
            raise RuntimeError(f"{path} belongs to another run; archive it")
        return summary

    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for p in parts:
            if part is not None and p != part:
                continue
            arm = part_arm(version, p)
            acfg = arm_config(config, development, arm)
            for seed_index, s in enumerate(development.training_seeds):
                model_root = _model_root(root, s, version, p)
                summary = stored_summary(p, s)
                if summary is not None:
                    results[p][s] = summary
                    log(f"{protocol} {p}: seed {s} already done")
                    continue
                log(f"{protocol} {p}: training seed {s}")
                _status(model_root, "training")
                trained = train_seed(acfg, config_path, s, model_root, pool, log=log,
                                     updates=updates)
                best = model_root / "best.pt"
                summary = {"plan_hash": digest_, "complete": complete, "arm": arm,
                           "training": trained,
                           "validation": _validation_of(model_root, trained["best_update"]),
                           "checkpoint": str(best) if best.exists() else None}
                if best.exists():
                    model = build_model(arm, acfg, seed=0)
                    load_checkpoint(best, model, manifest=contract_manifest(acfg))
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    log(f"{protocol} {p}: evaluating seed {s}")
                    summary["cells"] = evaluate_model(
                        pool, development, arm, state, seed_index,
                        progress=lambda name, index, root_=model_root: _status(
                            root_, "evaluating", cell=name, cell_index=index,
                            cells=len(development.cells)))
                else:
                    summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                            "timeout": 0.0, "statuses": {},
                                            "note": "no checkpoint reached difficulty 1.0"}
                                        for c in CELLS}
                if version != "orl/1":
                    summary["gate_probe"] = (probe_model(pool, arm, state) if best.exists()
                                             else {"gate_active": False,
                                                   "note": "no checkpoint reached difficulty "
                                                           "1.0"})
                    log(f"{protocol} {p}: seed {s} gate active "
                        f"{summary['gate_probe']['gate_active']}")
                write_json(model_root / "summary.json", summary)
                _status(model_root, "done")
                results[p][s] = summary
                log(f"{protocol} {p}: seed {s} all-cells success "
                    f"{np.mean([v['success'] for v in summary['cells'].values()]):.3f}")
    for p in parts:  # the other process's arm, if it has finished
        for s in development.training_seeds:
            if s not in results[p]:
                found = stored_summary(p, s)
                if found is not None:
                    results[p][s] = found
    finished = all(len(results[p]) == len(development.training_seeds) for p in parts)
    stored = {arm: {s: json.loads((_dev_model(root, s, arm) / "summary.json")
                                  .read_text(encoding="utf-8"))
                    for s in development.training_seeds} for arm in (PRIMARY, *SECONDARY)}
    verdict = None
    if complete and finished:
        verdict = judge(development, results["orl"], stored, protocol,
                        results.get("control") if version in CONTROLS else None)

    def brief(x):
        return ({k: v for k, v in x.items() if k not in ("cells", "gate_probe")}
                | {"gate_active": (x.get("gate_probe") or {}).get("gate_active")}
                | {"cells": {c: {k: v for k, v in cv.items() if k != "per_seed"}
                             for c, cv in x["cells"].items()}})
    payload = {"schema_id": SCHEMA_ID, "stage": STAGES[1], "protocol": protocol,
               "plan_hash": digest_, "orl_hash": HASHES[version], "procedure_complete": complete,
               "part": part, "all_parts_finished": finished, "verdict": verdict,
               "models": {str(s): brief(x) for s, x in results["orl"].items()},
               "wall_s": time.monotonic() - started,
               "note": "local plant only; paired with the stored development/1 models"}
    if version in CONTROLS:
        payload["control_hash"] = control_hash(version)
        payload["control_models"] = {str(s): brief(x) for s, x in results["control"].items()}
    return 0, payload


# ------------------------------------------------------------------ gate probe

def _probe_work(task: tuple) -> dict[str, Any]:
    """One probe episode (``GATE_PROBES`` of the arm's version) in a ``pilot._init_worker``
    process: the frozen policy through the forced training blackout, and per decision the
    mirror-averaged gate and gated term with the policy's own decoded estimate."""
    import torch
    from .backend import make_local_backend
    from .features import build_features
    from .models import tensors_from_frames
    from .observation import decode
    from .pilot import _WORKER, _model, fly_policy
    arm, state, seed = task
    spec = GATE_PROBES[probe_version(arm)]
    config = _WORKER["config"]
    blackout = replace(config.g4.training_randomization.vision_blackout, probability=1.0)
    model = _model(config, arm, state)
    backend = make_local_backend(config, difficulty=spec["difficulty"],
                                 randomization=None, scenario=spec["split"],
                                 dropout=blackout)
    try:
        try:
            episode = fly_policy(backend, model, config, seed=int(seed),
                                 deterministic=spec["deterministic"],
                                 generator=torch.Generator().manual_seed(int(seed) + 9_000_000))
        except RuntimeError as exc:  # pre-policy handover refusal: setup, not RL
            if backend.active:
                raise
            return {"seed": int(seed), "refused": str(exc)}
    finally:
        backend.close()
    observations = episode["observations"][:-1]
    _, flat, _ = tensors_from_frames([build_features(o, config) for o in observations])
    with torch.no_grad():
        terms = model.vertical_terms(flat)
    seen = [decode(o) for o in observations]
    return {"seed": int(seed), "refused": None, "status": episode["status"],
            "blackout_s": episode.get("evaluation_dropout_s"),
            "gate": terms["gate"].tolist(), "gated": terms["gated"].tolist(),
            "valid": [bool(x.estimate_valid) for x in seen],
            "vision_age_s": [float(x.vision_age_s) for x in seen],
            "h": [float(x.h) for x in seen], "vz": [float(x.vz) for x in seen]}


def summarize_probe(episodes: list[dict[str, Any]], version: str = "orl/2") -> dict[str, Any]:
    """``GATE_RULES[version]`` on one checkpoint's probe episodes; the own vertical velocity
    is reported whenever the episodes carry it and decides only under probe 2."""
    spec = GATE_PROBES[version]
    flown = [e for e in episodes if not e.get("refused")]
    cat = lambda key: np.concatenate([np.asarray(e[key], dtype=float) for e in flown]) \
        if flown else np.zeros(0)
    gate, gated, age, h = cat("gate"), cat("gated"), cat("vision_age_s"), cat("h")
    valid = cat("valid").astype(bool)
    high = valid & (h > spec["high_m"])
    fresh = high & (age <= spec["fresh_age_s"])
    blind = high & (age > spec["blind_age_s"])
    mean = lambda x, m: float(x[m].mean()) if m.any() else None
    out = {"episodes": len(flown), "refused": len(episodes) - len(flown),
           "statuses": {k: sum(e["status"] == k for e in flown)
                        for k in sorted({e["status"] for e in flown})},
           "decisions": int(gate.size), "fresh_decisions": int(fresh.sum()),
           "blind_decisions": int(blind.sum()),
           "gate_fresh": mean(gate, fresh), "gate_blind": mean(gate, blind),
           "gated_fresh": mean(gated, fresh), "gated_blind": mean(gated, blind),
           "gate_sd": float(gate.std()) if gate.size else None}
    if flown and all("vz" in e for e in flown):
        vz = cat("vz")
        out["vz_fresh"], out["vz_blind"] = mean(vz, fresh), mean(vz, blind)
    enough = min(out["fresh_decisions"], out["blind_decisions"]) >= spec["min_decisions"]
    out["gate_drop"] = (out["gate_fresh"] - out["gate_blind"]) if enough else None
    active = bool(enough and out["gate_drop"] >= spec["min_gate_drop"])
    if "max_blind_sink_m_s" in spec:
        out["blind_sink_ok"] = bool(enough and out.get("vz_blind") is not None
                                    and out["vz_blind"] >= -spec["max_blind_sink_m_s"])
        active = active and out["blind_sink_ok"]
    out["gate_active"] = active
    return out


def probe_model(pool: Any, arm: str, state: dict[str, Any]) -> dict[str, Any]:
    version = probe_version(arm)
    episodes = list(pool.map(_probe_work, [(arm, state, s)
                                           for s in GATE_PROBES[version]["seeds"]]))
    return {"probe_hash": probe_hash(version), **summarize_probe(episodes, version)}


def probe_checkpoints(config_path: Path, arm: str, checkpoints: dict[str, Path], *,
                      workers: int, log: Callable[[str], None] = print,
                      descriptive: dict[str, Path] | None = None) -> dict[str, Any]:
    """The arm version's gate probe on each selected checkpoint (model seed -> best.pt) of an
    ORL G4 pilot; ``gate_active`` only if every checkpoint exists and its gate is active.
    ``descriptive`` checkpoints (probe 2: each seed's final.pt) are probed and reported under
    ``descriptive_models`` and never enter ``gate_active``."""
    from .models import build_model, load_checkpoint
    from .pilot import _init_worker
    version = probe_version(arm)
    config = load_config(config_path)
    config = replace(config, g4=replace(config.g4, arm=arm))
    manifest = contract_manifest(config)
    groups: dict[str, dict[str, Any]] = {"models": {}, "descriptive_models": {}}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for group, paths in (("models", checkpoints), ("descriptive_models", descriptive or {})):
            for key, path in paths.items():
                if not Path(path).exists():
                    groups[group][key] = {"checkpoint": None, "gate_active": False}
                    continue
                model = build_model(arm, config, seed=0)
                meta = load_checkpoint(path, model, manifest=manifest)
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                found = {"checkpoint": str(path), "update": meta["extra"].get("update"),
                         **probe_model(pool, arm, state)}
                groups[group][key] = found
                log(f"gate probe {key}{' (descriptive)' if group != 'models' else ''}: gate "
                    f"fresh {found['gate_fresh']} blind {found['gate_blind']} drop "
                    f"{found['gate_drop']} vz blind {found.get('vz_blind')} active "
                    f"{found['gate_active']}")
    models = groups["models"]
    report = {"protocol": GATE_PROBES[version]["protocol"], "probe_hash": probe_hash(version),
              "spec": _probe_spec(version), "arm": arm, "models": models,
              "gate_active": bool(models) and all(m["gate_active"] for m in models.values())}
    if descriptive:
        report["descriptive_models"] = groups["descriptive_models"]
    return report
