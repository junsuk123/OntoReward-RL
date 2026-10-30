"""Backend boundary of spatial_current_semantics_v1: observer stamps, adapter trace, registry.

Observer semantics are pinned on synthetic ``spatial.core.Measurement``
samples; the backend runs on the real spatial-reference/12 LocalBackend.
Nothing here constructs IsaacBackend or PX4Bridge, and make_isaac_backend is
only ever called without allow_isaac=True.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from ontology_rgat.current_semantics import backend
from ontology_rgat.current_semantics.backend import (
    BACKENDS, CALIBRATION_GAPS, FLIGHT_SCHEMA, CurrentObserver, CurrentSpatialBackend,
    LocalOpticalCadence, config_task_label, local_plant_config, make_isaac_backend,
    make_local_backend, policy_action, require_training_backend)
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, OBSERVATION_NAMES,
                                                       contract_manifest, load_config)
from ontology_rgat.current_semantics.observation import decode
from ontology_rgat.direct_policy.isaac_adapter import enu_to_ned
from ontology_rgat.spatial import environment
from ontology_rgat.spatial.core import Measurement, SpatialConfig
from ontology_rgat.spatial.environment import LocalBackend, Truth


PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
CONFIG = load_config()
AGE = OBSERVATION_NAMES.index("ugv_visionAge")
SEED = 990001


def sample(t, *, sample_id=0, frame=None, capture=None, own=(0.0, 0.0, 2.0),
           own_at_capture=None, velocity=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0),
           rates=(0.0, 0.0, 0.0)):
    """A Measurement; ``frame`` is the optical solve (UAV minus pad) captured at ``capture``."""
    seen = frame is not None
    return Measurement(
        time_s=float(t), own_position=np.array(own, float),
        own_velocity=np.array(velocity, float), quaternion=np.array(quaternion, float),
        angular_rate=np.array(rates, float),
        optical_position=np.array(frame, float) if seen else None,
        confidence=0.5 if seen else 0.0, sample_id=sample_id,
        optical_time_s=capture if seen else None,
        optical_own_position=(np.array(own if own_at_capture is None else own_at_capture, float)
                              if seen else None),
        optical_quaternion=np.array(quaternion, float) if seen else None)


def observer():
    obs = CurrentObserver()
    obs.reset("synthetic")
    return obs


# ------------------------------------------------------------------ observer

def test_uninitialised_estimate_is_the_age_sentinel_with_a_zeroed_estimate():
    obs = observer()
    obs.ingest(sample(0.0))
    vector, provenance = obs.vector()
    assert vector[AGE] == 1.0 and np.all(vector[:7] == 0.0)
    physical = decode(vector)
    assert not physical.estimate_valid and math.isinf(physical.vision_age_s)
    assert not physical.vision_updated
    assert provenance["estimate_initialized"] is False
    assert provenance["vision_age_s"] is None and provenance["track_std"] is None
    assert provenance["capture_stamp_s"] is None and provenance["sample_id"] is None
    json.dumps(provenance, allow_nan=False)


def test_age_runs_from_the_capture_stamp_and_the_pose_is_paired_at_capture():
    # Captured at 0.00 with the vehicle at x=0, received at 0.08 after it moved
    # to x=1.  Pairing the solve with the receive pose would put the pad at x=1.
    obs = observer()
    obs.ingest(sample(0.0))
    obs.vector()
    frame = dict(sample_id=1, frame=(0.0, 0.0, 2.0), capture=0.0, own=(1.0, 0.0, 2.0),
                 own_at_capture=(0.0, 0.0, 2.0))
    assert obs.ingest(sample(0.08, **frame)) is True
    assert obs.ingest(sample(0.10, **frame)) is False          # same sample id again
    vector, provenance = obs.vector()
    physical = decode(vector)
    assert physical.vision_updated and physical.estimate_valid
    assert physical.vision_age_s == pytest.approx(0.10, abs=1e-12)       # not 0.02
    assert provenance["capture_stamp_s"] == 0.0 and provenance["receive_stamp_s"] == 0.08
    assert provenance["decision_stamp_s"] == 0.10 and provenance["frames_since_decision"] == 1
    assert physical.ex == pytest.approx(-1.0, abs=1e-12) and physical.h == pytest.approx(2.0)

    obs.ingest(sample(0.20, **frame))                                    # repeated again
    vector, provenance = obs.vector()
    physical = decode(vector)
    assert not physical.vision_updated and provenance["frames_since_decision"] == 0
    assert physical.vision_age_s == pytest.approx(0.20, abs=1e-12)
    assert provenance["capture_stamp_s"] == 0.0


def test_out_of_order_and_stale_captures_do_not_refresh_age():
    obs = observer()
    pad = dict(frame=(0.0, 0.0, 2.0))
    obs.ingest(sample(0.10, sample_id=1, capture=0.05, **pad))
    obs.vector()
    # A new sample id carrying an OLDER capture (out of order) is not assimilated.
    assert obs.ingest(sample(0.20, sample_id=2, capture=0.02, **pad)) is False
    # A frame captured more than 0.5 s before its delivery is not assimilated.
    assert obs.ingest(sample(0.70, sample_id=3, capture=0.15, **pad)) is False
    vector, provenance = obs.vector()
    physical = decode(vector)
    assert not physical.vision_updated
    assert physical.vision_age_s == pytest.approx(0.70 - 0.05, abs=1e-12)
    assert provenance["sample_id"] == 1 and provenance["capture_stamp_s"] == 0.05
    # A fresh, in-order frame is.
    assert obs.ingest(sample(0.80, sample_id=4, capture=0.72, **pad)) is True
    assert decode(obs.vector()[0]).vision_age_s == pytest.approx(0.08, abs=1e-12)


def test_same_time_sample_defers_a_new_frame_to_the_next_clock_sample():
    obs = observer()
    pad = dict(frame=(0.0, 0.0, 2.0))
    obs.ingest(sample(0.10, sample_id=1, capture=0.05, **pad))
    # DDS can deliver a frame on a sample with the previous clock: not consumed...
    assert obs.ingest(sample(0.10, sample_id=2, capture=0.08, **pad)) is False
    # ...so the next increasing sample still carries and assimilates it.
    assert obs.ingest(sample(0.11, sample_id=2, capture=0.08, **pad)) is True
    vector, provenance = obs.vector()
    assert provenance["frames_since_decision"] == 2 and provenance["sample_id"] == 2
    assert decode(vector).vision_age_s == pytest.approx(0.03, abs=1e-12)
    with pytest.raises(ValueError, match="reversed"):
        obs.ingest(sample(0.05))


def test_attitude_and_body_rates_come_from_the_decision_sample():
    angles = (0.1, -0.2, 0.3)
    q = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
    obs = observer()
    obs.ingest(sample(0.0, quaternion=q, rates=(0.05, -0.04, 0.03)))
    physical = decode(obs.vector()[0])
    assert (physical.roll, physical.pitch, physical.yaw) == pytest.approx(angles, abs=1e-12)
    assert (physical.p, physical.q, physical.r) == pytest.approx((0.05, -0.04, 0.03), abs=1e-12)
    assert physical.nav_valid and physical.nav_age_s == 0.0


def test_observer_accepts_only_measurements_and_never_names_truth():
    obs = observer()
    with pytest.raises(RuntimeError):
        obs.vector()
    truth = Truth(np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(3), False)
    with pytest.raises(TypeError):
        obs.ingest(truth)
    assert list(inspect.signature(CurrentObserver.ingest).parameters) == ["self", "measurement"]
    tree = ast.parse(inspect.getsource(CurrentObserver))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    assert not [name for name in names if "truth" in name.lower()]


SHIFT = np.array([5.0, -5.0, 1.0])


def test_policy_observation_is_unchanged_when_the_flight_reports_different_truth():
    def run(perturb):
        local = make_local_backend(CONFIG)
        if perturb:
            original_reset, original_advance = local.flight.reset, local.flight.advance

            # Inside the task's envelope (60 m, 20 m, 21 deg), so the evaluator
            # keeps the episode running while every truth field differs.
            def shifted(truth):
                return replace(truth, relative_position=truth.relative_position + SHIFT,
                               relative_velocity=truth.relative_velocity - 7.0,
                               roll_pitch=truth.roll_pitch + 0.01,
                               angular_rate=truth.angular_rate * 3.0)
            local.flight.reset = lambda seed: (lambda m, t: (m, shifted(t)))(*original_reset(seed))
            local.flight.advance = lambda c, cb: shifted(original_advance(c, cb))
        vectors, truths = [local.reset(SEED)[0]], []
        for k in range(12):
            step = local.step(np.array([0.4, -0.3, 0.2 if k % 2 else -0.2]))
            vectors.append(step.observation)
            truths.append(step.info["evaluator_only"]["relative_position_own_minus_pad_m"])
        local.close()
        return np.stack(vectors), np.array(truths)
    clean, clean_truth = run(False)
    perturbed, perturbed_truth = run(True)
    assert clean.tobytes() == perturbed.tobytes()
    np.testing.assert_allclose(perturbed_truth - clean_truth, np.tile(SHIFT, (12, 1)))


# ------------------------------------------------------------------- backend

def test_local_backend_reset_reports_the_contract_and_the_registered_task():
    local = make_local_backend(CONFIG)
    vector, info = local.reset(SEED)
    assert vector.shape == (21,) and np.all(np.isfinite(vector))
    assert info["backend"] == "spatial_local_v12" and info["flight"] == LocalBackend.name
    assert info["scientific_use"] is False and info["difficulty"] == 1.0
    assert info["task"] == config_task_label(CONFIG)
    assert info["task"]["evaluator"] == "spatial.environment.Evaluator"
    assert info["handover_wait_decisions"] >= 0
    assert local.flight.cfg == local_plant_config(CONFIG)
    assert local.flight.cfg.touchdown_height == CONFIG.local_plant.contact_height_m
    assert set(info["evaluator_only"]) == {
        "relative_position_own_minus_pad_m", "relative_velocity_own_minus_pad_m_s",
        "roll_pitch_rad", "angular_rate_rad_s", "contact"}
    assert info["episode_id"].startswith("spatial_local_v12:990001:")
    assert local.controller.max_tilt == pytest.approx(math.radians(20.0))
    assert CONFIG.action.adapter_max_tilt_deg == 20.0
    json.dumps(info, allow_nan=False)
    local.close()


def test_step_traces_policy_limited_setpoint_and_measured_acceleration():
    local = make_local_backend(CONFIG)
    local.reset(SEED)
    before = local.observer.last_measurement.own_velocity.copy()
    step = local.step(np.array([0.5, -0.4, 0.3]))
    after = local.observer.last_measurement.own_velocity
    trace = step.trace
    assert trace.decision_id == 0 and trace.decision_stamp_s == 0.0
    assert trace.policy_accel == (0.5, -0.4, 0.3)
    assert trace.adapter_limited_accel == pytest.approx(trace.policy_accel, abs=1e-12)
    assert trace.adapter_constrained is False
    assert trace.adapter_change == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert trace.px4_setpoint_ned == tuple(enu_to_ned(trace.adapter_limited_accel))
    assert "not visible" in trace.px4_setpoint_note
    assert step.transition_s == pytest.approx(0.1) and trace.measured_over_s == step.transition_s
    np.testing.assert_allclose(trace.measured_accel, (after - before) / step.transition_s)
    assert step.info["action_trace"] == trace.to_dict()
    assert step.info["termination"] is None and not step.done and not step.contact

    # Joint tilt/thrust envelope at the 20 deg adapter limit: descending at full
    # authority leaves 9.807-2 m/s^2 of thrust, i.e. 2.84 m/s^2 of lateral room.
    step = local.step(np.array([2.5, 2.5, -2.0]))
    limited = np.array(step.trace.adapter_limited_accel)
    assert step.trace.adapter_constrained is True
    assert np.linalg.norm(limited[:2]) == pytest.approx(
        (9.80665 - 2.0) * math.tan(math.radians(20.0)), rel=1e-9)
    np.testing.assert_allclose(step.trace.adapter_change,
                               limited - np.array([2.5, 2.5, -2.0]), atol=1e-12)
    # Beyond the action maximum: clipped and flagged, never silently rescaled.
    step = local.step(np.array([3.0, 0.0, 0.0]))
    assert step.trace.adapter_constrained and step.trace.adapter_limited_accel[0] == 2.5
    local.close()


def test_latent_must_map_to_the_action_by_a_max_tanh_alone():
    local = make_local_backend(CONFIG)
    local.reset(SEED)
    latent = np.array([0.3, -1.2, 0.05])
    action = policy_action(latent, CONFIG.action.maximum)
    step = local.step(action, latent=latent)
    assert step.trace.latent_z == tuple(latent)
    with pytest.raises(ValueError, match="hidden transform"):
        local.step(action * 0.9, latent=latent)
    local.close()


def test_contact_and_horizon_end_the_episode_with_the_registered_verdict():
    local = make_local_backend(CONFIG)
    local.reset(SEED)
    for _ in range(60):
        step = local.step(np.array([0.0, 0.0, -2.0]))
        if step.done:
            break
        assert step.status == "RUNNING" and math.isfinite(step.reward)
    assert step.contact and step.done
    # A full-authority descent hits far above the 0.30 m/s touchdown limit; off the
    # 0.5 m pad the verdict is a missed pad instead (the moving pad's rule).
    truth = step.info["evaluator_only"]
    off_pad = math.hypot(*truth["relative_position_own_minus_pad_m"][:2]) > 0.5
    assert abs(truth["relative_velocity_own_minus_pad_m_s"][2]) > CONFIG.contact.z_speed_m_s
    expected = "MISSED_PAD_CONTACT" if off_pad else "UNSAFE_CONTACT"
    assert step.status == expected
    assert step.info["termination"] == {"reason": "truth_contact", "verdict": expected,
                                        "cleanup_confirmed": True}
    assert step.reward < -30.0 and step.info["evaluator_only"]["contact"] is True
    components = step.info["reward_components"]
    assert set(components) == {"potential", "shaping", "cost", "tilt", "terminal",
                               "reference_reward"}
    # task v4: the Evaluator's verdict and terminal table, the common dense reward.
    assert components["terminal"] == -40.0
    assert step.reward == pytest.approx(components["shaping"] + components["cost"]
                                        + components["tilt"] - 40.0)
    assert 0.0 <= step.info["lost_view_weight"] <= 1.0
    assert components["reference_reward"] < -30.0
    with pytest.raises(RuntimeError, match="reset required"):
        local.step(np.zeros(3))

    local.reset(SEED)
    local.evaluator.cfg = replace(local.evaluator.cfg, horizon=0.25)
    steps = [local.step(np.zeros(3)) for _ in range(3)]
    assert [s.done for s in steps] == [False, False, True]
    assert steps[-1].status == "TASK_TIMEOUT"
    assert steps[-1].info["termination"] == {"reason": "horizon", "verdict": "TASK_TIMEOUT",
                                             "cleanup_confirmed": True}
    assert not steps[-1].contact
    local.close()


def test_the_policy_action_is_costed_and_the_reward_telescopes():
    reward = CONFIG.task.reward
    local = make_local_backend(CONFIG)
    local.reset(SEED)
    quiet = local.step(np.zeros(3))
    busy = local.step(np.array([2.5, -2.5, 2.0]))
    q, b = quiet.info["reward_components"], busy.info["reward_components"]
    assert q["cost"] == pytest.approx(-reward.time_cost_per_s * quiet.transition_s)
    assert b["cost"] == pytest.approx(-(reward.time_cost_per_s + reward.control_cost_per_s)
                                      * busy.transition_s)
    # /4: no command-change term; a jump from the hover to full authority costs only
    # its control cost (exploration noise must not pay a per-decision tax).
    assert "command_change" not in b
    # The undiscounted potential difference telescopes over consecutive steps.
    assert b["shaping"] == pytest.approx(b["potential"] - q["potential"])
    assert q["terminal"] == b["terminal"] == 0.0
    local.close()


def test_cadence_tool_keeps_every_second_tick_and_drops_the_given_fraction():
    from ontology_rgat.current_semantics.contracts import OpticalCadence
    thinning = OpticalCadence(keep_every_nth_capture=2, extra_drop_probability=0.079)
    cadence = LocalOpticalCadence(thinning, 0.05)
    cadence.reset(990001)
    ticks = np.arange(0, 20000)
    kept = np.array([cadence.keeps(t * 0.05) for t in ticks])
    assert not kept[1::2].any() and kept[0]
    drop = 1.0 - kept[2::2].mean()
    p = thinning.extra_drop_probability
    assert abs(drop - p) < 4.0 * math.sqrt(p * (1 - p) / kept[2::2].size)
    again = LocalOpticalCadence(thinning, 0.05)
    again.reset(990001)
    assert [again.keeps(t * 0.05) for t in ticks[::-1]][::-1][2:] == kept.tolist()[2:]
    other = LocalOpticalCadence(thinning, 0.05)
    other.reset(990002)
    assert [other.keeps(t * 0.05) for t in ticks] != kept.tolist()


def test_registered_cadence_is_identity_until_isaac_measures_the_frame_rate():
    cadence = CONFIG.local_plant.optical_cadence
    assert (cadence.keep_every_nth_capture, cadence.extra_drop_probability) == (1, 0.0)
    local = make_local_backend(CONFIG)
    assert local.observer.cadence is None
    local.reset(SEED)
    frames = [local.step(np.zeros(3)).info["observation_provenance"]["frames_since_decision"]
              for _ in range(10)]
    local.close()
    assert max(frames) == 2  # the plant's own 20 Hz, untouched
    assert "rejected" in CALIBRATION_GAPS["optical_update_rate"].status


def test_step_refuses_malformed_actions_and_needs_a_reset():
    local = make_local_backend(CONFIG)
    with pytest.raises(RuntimeError):
        local.step(np.zeros(3))
    local.reset(SEED)
    for bad in (np.zeros(2), np.array([0.0, np.nan, 0.0]), np.zeros(24)):
        with pytest.raises(ValueError):
            local.step(bad)
    with pytest.raises(TypeError):
        local.reset(1.5)
    local.close()


def _variant(tmp_path, edit):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def test_backend_refuses_any_flight_but_the_registered_v12_plant(tmp_path):
    flight = LocalBackend(local_plant_config(CONFIG))
    with pytest.raises(RuntimeError, match="always visible"):
        CurrentSpatialBackend(flight, CONFIG, name="direct_local_toy", scientific=False)
    with pytest.raises(ValueError, match="unknown backend"):
        CurrentSpatialBackend(flight, CONFIG, name="reduced_3d", scientific=False)
    with pytest.raises(TypeError, match="IsaacBackend"):
        CurrentSpatialBackend(flight, CONFIG, name="isaac_v12", scientific=False)
    assert CurrentSpatialBackend(flight, CONFIG, name="spatial_local_v12",
                                 scientific=True).scientific is True
    for cfg in (SpatialConfig(schema="spatial-reference/11"),
                SpatialConfig(schema=FLIGHT_SCHEMA),
                replace(local_plant_config(CONFIG), camera_dt=0.1)):
        with pytest.raises(ValueError, match="registered"):
            CurrentSpatialBackend(LocalBackend(cfg), CONFIG, name="spatial_local_v12",
                                  scientific=False)
    rescaled = _variant(tmp_path, lambda d: d["action"].update(maximum_m_s2=[2.0, 2.5, 2.0]))
    with pytest.raises(ValueError, match="rescale"):
        CurrentSpatialBackend(flight, rescaled, name="spatial_local_v12", scientific=False)


def test_adapter_tilt_is_registered_with_the_action_and_hashed_there(tmp_path):
    base = contract_manifest(CONFIG)
    assert base["action"]["adapter_max_tilt_deg"] == 20.0
    variant = contract_manifest(_variant(tmp_path, lambda d: d["action"].update(
        adapter_max_tilt_deg=15.0)))
    assert variant["action_hash"] != base["action_hash"]
    assert variant["algorithm_hash"] != base["algorithm_hash"]
    assert variant["g3_hash"] == base["g3_hash"]
    assert variant["feature_hash"] == base["feature_hash"]
    for edit, match in ((lambda d: d["action"].update(adapter_max_tilt_deg=30.0), "exceed"),
                        (lambda d: d["action"].pop("adapter_max_tilt_deg"), "missing keys"),
                        (lambda d: d["action"].update(adapter_max_tilt_deg=True), "number")):
        with pytest.raises(ValueError, match=match):
            _variant(tmp_path, edit)


def test_isaac_backend_is_refused_without_allow_isaac(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("IsaacBackend must not be constructed")
    monkeypatch.setattr(environment.IsaacBackend, "__init__", forbidden)
    for allow in (False, None, 1, "yes"):
        with pytest.raises(RuntimeError, match="allow_isaac"):
            make_isaac_backend(CONFIG, allow_isaac=allow)
    with pytest.raises(RuntimeError, match="allow_isaac"):
        make_isaac_backend(CONFIG)


def test_local_episode_never_constructs_isaac_or_px4(monkeypatch):
    from ontology_rgat import bridge

    def forbidden(*args, **kwargs):
        raise AssertionError("a flight stack was constructed")
    monkeypatch.setattr(environment.IsaacBackend, "__init__", forbidden)
    monkeypatch.setattr(bridge.PX4Bridge, "__init__", forbidden)
    local = make_local_backend(CONFIG)
    local.reset(SEED)
    for _ in range(5):
        local.step(np.array([0.2, 0.1, 0.0]))
    local.close()


def test_local_backend_imports_no_isaac_ros_or_px4_runtime():
    code = (f"import sys; sys.path.insert(0, {str(PYTHON_DIR)!r})\n"
            "import numpy as np\n"
            "from ontology_rgat.current_semantics.backend import make_local_backend\n"
            "from ontology_rgat.current_semantics.contracts import load_config\n"
            "b = make_local_backend(load_config()); b.reset(990002)\n"
            "[b.step(np.zeros(3)) for _ in range(3)]; b.close()\n"
            "roots = ('rclpy', 'omni', 'isaacsim', 'pxr', 'carb', 'px4_msgs', 'rosidl')\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in roots))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, timeout=120)
    assert result.stdout.strip() == "[]"


# ------------------------------------------------------------------ registry

def test_registry_statuses_and_training_refusals():
    assert set(BACKENDS) == {"spatial_local_v12", "isaac_v12", "direct_local_toy"}
    assert BACKENDS["direct_local_toy"].status == "refused"
    assert BACKENDS["isaac_v12"].status == "requires_allow_isaac"
    assert BACKENDS["spatial_local_v12"].status == "g3_passed"
    with pytest.raises(RuntimeError, match="always visible"):
        require_training_backend("direct_local_toy", allow_pending_g3=True)
    with pytest.raises(RuntimeError, match="not a training backend"):
        require_training_backend("isaac_v12", allow_pending_g3=True)
    # Promoted on the 2026-10-28 G3 pass: training needs no pending-G3 escape any more.
    assert require_training_backend("spatial_local_v12") is BACKENDS["spatial_local_v12"]
    assert "coupling_isaac.json" in backend.G3_STATUS
    with pytest.raises(ValueError, match="unknown backend"):
        require_training_backend("reduced_3d", allow_pending_g3=True)


def test_calibration_gaps_register_the_audited_items_with_local_numbers_from_the_plant():
    audited = {"optical_update_rate", "touchdown_body_height", "entry_yaw", "board_yaw",
               "own_navigation", "optical_position_noise"}
    assert audited <= set(CALIBRATION_GAPS)
    assert set(BACKENDS["spatial_local_v12"].calibration_gaps) == set(CALIBRATION_GAPS)
    plant = SpatialConfig(schema=FLIGHT_SCHEMA)
    rate = CALIBRATION_GAPS["optical_update_rate"].numbers
    assert rate["local_camera_dt_s"] == plant.camera_dt
    assert rate["local_transport_s"] == plant.camera_transport_delay
    touchdown = CALIBRATION_GAPS["touchdown_body_height"].numbers
    assert touchdown["local_contact_body_height_m"] == plant.touchdown_height
    assert touchdown["isaac_first_contact_body_height_m"] == (0.0636, 0.0732)
    assert touchdown["isaac_median_first_contact_body_height_m"] \
        == CONFIG.local_plant.contact_height_m
    calibrated = {"touchdown_body_height", "handover_detection"}
    for name, gap in CALIBRATION_GAPS.items():
        prefix = "calibrated in the training plant" if name in calibrated else "open G3 item"
        assert gap.status.startswith(prefix), name
        assert gap.evidence
        json.dumps(gap.to_dict(), allow_nan=False)


def _mutable_module_values(module: ModuleType):
    for name, value in vars(module).items():
        if name.startswith("__") or isinstance(value, (ModuleType, type)) or callable(value):
            continue
        if isinstance(value, (list, dict, set, bytearray)):
            yield name
        if isinstance(value, np.ndarray) and value.flags.writeable:
            yield name


def test_backend_and_coupling_hold_no_module_level_mutable_registries():
    from ontology_rgat.current_semantics import coupling
    for module in (backend, coupling):
        assert list(_mutable_module_values(module)) == [], module.__name__


def _without_blackout(randomization):
    return replace(randomization, vision_blackout=replace(randomization.vision_blackout,
                                                          probability=0.0))


def test_the_handover_own_state_error_is_training_only_and_decays():
    randomization = _without_blackout(CONFIG.g4.training_randomization)
    spec = randomization.handover_own_state_error
    plain = make_local_backend(CONFIG)
    noisy = make_local_backend(CONFIG, randomization=randomization)
    try:
        _, info_plain = plain.reset(SEED)
        _, info_noisy = noisy.reset(SEED)
        assert "training_randomization" not in info_plain and plain.error is None
        e0, tau, t0 = noisy.error
        assert np.hypot(*e0[:2]) <= spec.horizontal_max_m_s and abs(e0[2]) <= spec.vertical_max_m_s
        assert spec.tau_min_s <= tau <= spec.tau_max_s
        assert info_noisy["training_randomization"]["tau_s"] == tau
        assert info_noisy["training_randomization"]["vision_blackout_s"] is None
        # Truth (and therefore the verdict and the reward) is untouched: same plant, same seed.
        assert info_noisy["evaluator_only"] == info_plain["evaluator_only"]
        for _ in range(5):
            a, b = plain.step(np.zeros(3)), noisy.step(np.zeros(3))
        assert a.info["evaluator_only"] == b.info["evaluator_only"]
        seen, true = noisy.observer.last_measurement, plain.observer.last_measurement
        t = float(seen.time_s) - t0
        np.testing.assert_allclose(seen.own_velocity - true.own_velocity,
                                   e0 * np.exp(-t / tau), atol=1e-9)
        np.testing.assert_allclose(seen.own_position - true.own_position,
                                   e0 * tau * (1 - np.exp(-t / tau)), atol=1e-9)
    finally:
        plain.close(); noisy.close()
    from ontology_rgat.current_semantics.backend import CurrentSpatialBackend
    with pytest.raises(ValueError, match="local plant only"):
        CurrentSpatialBackend.__init__(object.__new__(CurrentSpatialBackend), plain.flight,
                                       CONFIG, name="isaac_v12", scientific=False,
                                       randomization=randomization)


def test_the_vision_blackout_withholds_frames_only_inside_its_window():
    randomization = CONFIG.g4.training_randomization
    always = replace(randomization, vision_blackout=replace(randomization.vision_blackout,
                                                            probability=1.0))
    spec = always.vision_blackout
    plain, dark = make_local_backend(CONFIG), make_local_backend(CONFIG, randomization=always)
    try:
        _, info_plain = plain.reset(SEED)
        _, info_dark = dark.reset(SEED)
        start, duration = info_dark["training_randomization"]["vision_blackout_s"]
        assert spec.start_min_s <= start <= spec.start_max_s
        assert spec.duration_min_s <= duration <= spec.duration_max_s
        # The handover itself saw a frame: the blackout starts only after it.
        assert dark.observer.track.initialized
        assert info_dark["evaluator_only"] == info_plain["evaluator_only"]
        ages, weights, inside = [], [], []
        while True:
            a, b = plain.step(np.zeros(3)), dark.step(np.zeros(3))
            # The plant and its truth are untouched; only the observer's input changes.
            assert a.info["evaluator_only"] == b.info["evaluator_only"]
            ages.append(b.info["observation_provenance"]["vision_age_s"])
            weights.append(b.info["lost_view_weight"])
            inside.append(start + 0.2 <= b.elapsed_s < start + duration)
            if b.done or b.elapsed_s > start + duration + 0.5:
                break
        ages, inside = np.array(ages), np.array(inside)
        # Inside the window the age grows by a decision per decision; nothing is accepted.
        assert np.all(np.diff(ages[inside]) > 0.0)
        assert ages[inside].max() >= duration - 0.3
    finally:
        plain.close(); dark.close()
    # Seeded per episode: the same seed draws the same window, about half carry one.
    draws = []
    for seed in range(SEED, SEED + 40):
        backend = make_local_backend(CONFIG, randomization=randomization)
        try:
            backend.reset(seed)
            draws.append(backend.blackout)
        finally:
            backend.close()
    assert 10 <= sum(d is not None for d in draws) <= 30
