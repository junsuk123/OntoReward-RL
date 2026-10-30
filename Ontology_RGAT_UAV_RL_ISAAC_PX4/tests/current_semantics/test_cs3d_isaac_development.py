"""Guide 11.6: task v5 on Isaac and the Isaac evaluation of the development models.

Nothing here starts Isaac: the world's half is tested through pad_motion.py (pure
python) and a fake IsaacBackend that acknowledges a reset the way landing_world.py
does; the evaluation logic runs on the local plant.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from isaac_sim.pad_motion import PadMotionConfig, PadTrajectory, is_benchmark_scenario
from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics import isaac_development as idev
from ontology_rgat.current_semantics.backend import (FLIGHT_SCHEMA, ISAAC_SCENARIO,
                                                     CurrentSpatialBackend, local_plant_config,
                                                     make_local_backend)
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.models import ARMS, CurrentActorCritic, save_checkpoint
from ontology_rgat.current_semantics.scenario import (PlantFactors, ScenarioLocalBackend,
                                                      check_isaac_reset, draw, isaac_scenario)
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import IsaacBackend
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.scenarios import (encode_current_omni, parse_current_omni,
                                             sample_spatial_scenario)
from ontology_rgat_px4.protocol import is_benchmark_scenario as wire_accepts

CONFIG = load_config()
DEV_PLAN = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, CONFIG)
SPEC = CONFIG.task.scenario


# ------------------------------------------------------------ the scenario name

def test_the_name_round_trips_exactly_and_every_hop_accepts_only_well_formed_names():
    drawn = draw(SPEC, 994113, "composition")
    name = encode_current_omni(drawn.heading_rad, drawn.bearing_rad, drawn.offset_radius_m)
    parsed = parse_current_omni(name)
    assert (parsed["heading_rad"], parsed["bearing_rad"], parsed["radius_m"]) == (
        drawn.heading_rad, drawn.bearing_rad, drawn.offset_radius_m)
    assert parsed["a2_m_s2"] is None
    assert wire_accepts(name) and is_benchmark_scenario(name)
    assert parse_current_omni("spatial_reference_cv_ca_cv") is None
    assert wire_accepts("spatial_reference_cv_ca_cv") and not wire_accepts("foo")
    for bad in (name + ":1", name.replace("nominal", "nan"),
                encode_current_omni(0.0, 0.0, 0.5).replace(":0.5:", ":-0.5:")):
        with pytest.raises(ValueError):
            parse_current_omni(bad)
    with pytest.raises(ValueError):
        encode_current_omni(4.0, 0.0, 0.5)  # outside [-pi, pi]


@pytest.mark.parametrize("seed, split, factors", [
    (994101, "in_distribution", PlantFactors()),
    (994117, "composition", PlantFactors()),
    (994125, "in_distribution", PlantFactors(pad_acceleration_m_s2=(0.6, 0.9))),
])
def test_isaac_deck_drives_exactly_what_the_local_plant_flies(seed, split, factors):
    """Same seed, same draw: the local task-v5 pad and Isaac's deck share position and
    velocity at every time (the paired-scenario guarantee, now in every direction)."""
    local = ScenarioLocalBackend(local_plant_config(CONFIG), scenario=SPEC, split=split,
                                 factors=factors)
    local.reset(seed)
    drawn, name = isaac_scenario(SPEC, seed, split, factors)
    assert drawn == local.episode_scenario
    profile = deployment_profile(FLIGHT_SCHEMA)["resolved_scientific_configuration"]
    deck = PadTrajectory(PadMotionConfig.from_mapping(profile))
    manifest = deck.reset(seed, 100.0, 1.0, name, hard_reset=True)
    assert manifest["current_omni"] == parse_current_omni(name)
    assert manifest["heading_rad"] == drawn.heading_rad == manifest["yaw_rad"]
    origin = np.asarray(manifest["position_enu_m"])
    shared = sample_spatial_scenario(seed)
    for t in (0.0, 0.3, shared.t1, shared.t1 + shared.t2, 12.0, 70.0):
        p, v = local.pad_state(t)
        p2, v2 = deck.pose(100.0 + t)
        np.testing.assert_allclose(p2 - origin, p, atol=1e-12)
        np.testing.assert_allclose(v2, v, atol=1e-12)
    assert manifest["spatial_motion"]["a2"] == local.a2


def test_the_noise_factor_never_reaches_isaac():
    with pytest.raises(ValueError, match="local plant only"):
        isaac_scenario(SPEC, 994101, "in_distribution", PlantFactors(optical_noise_scale=5.0))


# --------------------------------------------------- the acknowledgement check

def _ack(seed: int, name: str, *, yaw: float = 0.03, **tamper) -> dict:
    """What landing_world.py acknowledges for a current-omni reset."""
    profile = deployment_profile(FLIGHT_SCHEMA)["resolved_scientific_configuration"]
    deck = PadTrajectory(PadMotionConfig.from_mapping(profile)).reset(
        seed, 0.0, 1.0, name, hard_reset=True)
    omni = parse_current_omni(name)
    offset = [-omni["radius_m"] * math.cos(omni["bearing_rad"]),
              -omni["radius_m"] * math.sin(omni["bearing_rad"]), 2.2]
    detail = {"pad": deck, "entry_offset_pad_m": offset, "entry_yaw_enu_rad": yaw}
    for key, value in tamper.items():
        if key == "heading":
            detail["pad"] = {**deck, "heading_rad": value}
        elif key == "offset":
            detail["entry_offset_pad_m"] = value
    return detail


def test_the_ack_must_match_the_draw_in_heading_offset_a2_and_yaw():
    seed = 994121
    _, name = isaac_scenario(SPEC, seed, "in_distribution",
                             PlantFactors(pad_acceleration_m_s2=(0.6, 0.9)))
    record = check_isaac_reset(_ack(seed, name), seed, name)
    assert record["isaac_scenario"] == name and 0.6 <= record["pad_a2_m_s2"] <= 0.9
    for detail in (_ack(seed, name, heading=0.0), _ack(seed, name, offset=[0.01, 0.0, 2.2]),
                   _ack(seed, name, yaw=math.radians(30.0)), {"entry_offset_pad_m": [0, 0, 2]}):
        with pytest.raises(RuntimeError):
            check_isaac_reset(detail, seed, name)
    # The world's own /12 deck (a stale world that ignored the name) is refused too.
    _, other = isaac_scenario(SPEC, seed, "composition")
    with pytest.raises(RuntimeError):
        check_isaac_reset(_ack(seed, other), seed, name)


class FakeIsaac(IsaacBackend):
    """IsaacBackend's surface over the task-v5 local plant: ``reset`` applies the scenario
    NAME it was given (and checks it is the plant's own draw) and acknowledges it the
    way landing_world.py does."""
    name = "fake-isaac"

    def __init__(self, split: str, factors: PlantFactors = PlantFactors(), *, yaw=0.02):
        sha = CONFIG.calibration.source.sha256
        self.cfg = SpatialConfig(schema=FLIGHT_SCHEMA, isaac_profile_sha256=sha)
        self.profile_sha256, self.scenario, self.yaw = sha, ISAAC_SCENARIO, yaw
        self.plant = ScenarioLocalBackend(local_plant_config(CONFIG), scenario=SPEC,
                                          split=split, factors=factors)
        self.bridge = SimpleNamespace(last_reset_ack=None)
        self.names: list[str] = []

    def reset(self, seed):
        out = self.plant.reset(seed)
        omni = parse_current_omni(self.scenario)
        drawn = self.plant.episode_scenario
        assert (omni["heading_rad"], omni["bearing_rad"], omni["radius_m"]) == (
            drawn.heading_rad, drawn.bearing_rad, drawn.offset_radius_m)
        self.names.append(self.scenario)
        self.bridge.last_reset_ack = {"detail": _ack(seed, self.scenario, yaw=self.yaw)}
        return out

    def advance(self, command, callback):
        return self.plant.advance(command, callback)

    def finish(self):
        return True

    def close(self):
        pass


def test_the_isaac_wrapper_sends_the_draw_checks_the_ack_and_takes_a_dropout():
    flight = FakeIsaac("composition")
    cell = DEV_PLAN.cell("combined_id")
    factors, dropout = dev.cell_conditions(CONFIG, asdict(cell), DEV_PLAN.dropout_start_s)
    backend = CurrentSpatialBackend(flight, CONFIG, name="isaac_v12", scientific=False,
                                    split="composition", factors=factors, dropout=dropout)
    _, info = backend.reset(994111)
    assert flight.names == [isaac_scenario(SPEC, 994111, "composition")[1]]
    assert info["scenario"]["split"] == "composition"
    assert info["scenario"]["isaac_scenario"] == flight.names[0]
    assert info["evaluation_dropout_s"] is not None  # combined_id always drops 1-3 s
    # A second condition's wrapper may share the flight that now carries an omni name.
    CurrentSpatialBackend(flight, CONFIG, name="isaac_v12", scientific=False,
                          split="in_distribution")
    with pytest.raises(ValueError, match="must fly"):
        CurrentSpatialBackend(flight, CONFIG, name="isaac_v12", scientific=False)
    wrong = FakeIsaac("composition", yaw=math.radians(12.0))
    with pytest.raises(RuntimeError, match="entry yaw"):
        CurrentSpatialBackend(wrong, CONFIG, name="isaac_v12", scientific=False,
                              split="composition").reset(994111)


def test_split_and_factors_are_isaac_only():
    with pytest.raises(ValueError, match="Isaac flight"):
        CurrentSpatialBackend(make_local_backend(CONFIG).flight, CONFIG,
                              name="spatial_local_v12", scientific=False, split="composition")


# ------------------------------------------------------------------- the plan

def _plan_file(tmp_path, edit):
    data = yaml.safe_load(idev.DEFAULT_PATH.read_text(encoding="utf-8"))
    edit(data)
    out = tmp_path / "isaac_plan.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_registered_plan_flies_every_model_on_the_same_thirty_cases():
    plan = idev.load_plan(idev.DEFAULT_PATH, CONFIG, DEV_PLAN)
    keys = idev.model_keys(DEV_PLAN)
    assert len(keys) == 20 and [c for c, _ in plan.conditions] == [
        "composition_id", "composition_ood", "combined_id"]
    order = idev.episode_order(plan, keys)
    assert len(order) == 600 and len(set(order)) == 600
    # environment-major: all 20 models meet an environment before the next one starts
    assert {k for _, _, k in order[:20]} == set(keys) and len({s for _, s, _ in order[:20]}) == 1


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["conditions"].update(noise_ood=[994140]), "only the local plant"),
    (lambda d: d["conditions"].update(motion_ood=[994041]), "already used"),
    (lambda d: d["conditions"].update(motion_ood=[994031]), "already used"),
    (lambda d: d["conditions"].update(motion_ood=[992101]), "already used"),
    (lambda d: d["conditions"].update(motion_ood=[994101]), "distinct"),
    (lambda d: d["conditions"].update(nominal=[994140]), "not a development cell"),
    (lambda d: d.update(mode="sampled"), "deterministic"),
    (lambda d: d.update(development_protocol="development/2"), "development protocol"),
])
def test_plan_refuses_used_seeds_and_local_only_conditions(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        idev.load_plan(_plan_file(tmp_path, edit), CONFIG, DEV_PLAN)


def _passed_development(root: Path) -> None:
    manifest = contract_manifest(CONFIG)
    (root / "g4_pilot.json").write_text(json.dumps({
        "protocol_complete": True, "learnability_passed": True,
        "hashes": {k: manifest[k] for k in ("g4_hash", "algorithm_hash", "task_hash",
                                            "calibration_hash")}}))
    status, payload = dev.register(DEFAULT_CONFIG_PATH, dev.DEFAULT_DEVELOPMENT_PATH, root)
    assert status == 0
    (root / dev.REGISTRATION).write_text(json.dumps(payload))
    (root / dev.RESULT).write_text(json.dumps({
        "plan_hash": payload["plan_hash"], "procedure_complete": True,
        "verdict": {"development_gate_passed": True}}))


def _checkpoints(root: Path, seeds=None, plan=DEV_PLAN, model_seed=1) -> None:
    for key in idev.model_keys(plan):
        seed, arm = key.split("/")
        if seeds is not None and int(seed) not in seeds:
            continue
        acfg = dev.arm_config(CONFIG, plan, arm)
        idev.checkpoint_path(root, key).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(idev.checkpoint_path(root, key), CurrentActorCritic(arm, acfg, seed=model_seed),
                        manifest=contract_manifest(acfg), extra={"update": 1})


def test_register_needs_a_passed_development_and_binds_every_checkpoint(tmp_path):
    args = (DEFAULT_CONFIG_PATH, idev.DEFAULT_PATH, dev.DEFAULT_DEVELOPMENT_PATH, tmp_path)
    assert idev.register(*args)[0] == 3
    _passed_development(tmp_path)
    status, payload = idev.register(*args)
    assert status == 3 and "best.pt" in payload["reason"]
    _checkpoints(tmp_path)
    status, payload = idev.register(*args)
    assert status == 0 and payload["status"] == "REGISTERED" and payload["episodes"] == 600
    (tmp_path / idev.REGISTRATION).write_text(json.dumps(payload))
    plan = idev.load_plan(idev.DEFAULT_PATH, CONFIG, DEV_PLAN)
    assert idev.check_registration(CONFIG, plan, DEV_PLAN, tmp_path) is None
    assert idev.register(*args)[1]["status"] == "ALREADY_REGISTERED"
    # A retrained checkpoint is not the registered one.
    _checkpoints(tmp_path, seeds={12}, model_seed=2)
    assert "differs" in idev.check_registration(CONFIG, plan, DEV_PLAN, tmp_path)
    # Nor is a development stage that did not pass.
    _checkpoints(tmp_path)
    result = json.loads((tmp_path / dev.RESULT).read_text())
    result["verdict"]["development_gate_passed"] = False
    (tmp_path / dev.RESULT).write_text(json.dumps(result))
    assert "PASSED" in idev.check_registration(CONFIG, plan, DEV_PLAN, tmp_path)


def test_run_flies_every_case_resumes_and_pairs_with_the_local_reference(tmp_path):
    small = replace(DEV_PLAN, training_seeds=(10,))
    plan = idev.IsaacDevelopmentPlan(conditions=(("composition_ood", (994111,)),
                                                 ("combined_id", (994121,))))
    _checkpoints(tmp_path, plan=small)

    def make(cell):
        factors, dropout = dev.cell_conditions(CONFIG, asdict(cell), small.dropout_start_s)
        return make_local_backend(CONFIG, scenario=cell.split, factors=factors,
                                  dropout=dropout)

    out = tmp_path / "isaac_development"
    flown = idev.run(CONFIG, plan, small, tmp_path, make, out, log=lambda _: None,
                     trace=False)
    assert flown["complete"] and len(flown["rows"]) == 8
    assert {r["model"] for r in flown["rows"]} == {f"10/{a}" for a in ARMS}
    assert all(r["scenario"]["split"] == ("composition" if r["condition"] == "composition_ood"
                                          else "in_distribution") for r in flown["rows"])
    assert all(r["evaluation_dropout_s"] is not None for r in flown["rows"]
               if r["condition"] == "combined_id")
    again = idev.run(CONFIG, plan, small, tmp_path, make, out, log=lambda _: None,
                     trace=False, resume=True)
    assert again["complete"] and again["resumed_with"] == 8 and len(again["rows"]) == 8
    with pytest.raises(ValueError, match="already holds"):
        idev.run(CONFIG, plan, small, tmp_path, make, out, log=lambda _: None, trace=False)
    summary = idev.summarize(plan, small, flown["rows"], flown["rows"])
    paired = summary["paired_with_local"]
    assert paired["cases"] == 8 and paired["isaac_only_success"] == 0 \
        and paired["local_only_success"] == 0
