"""Guards for the minimal-observation contract (docs/MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ontology_rgat.minimal import OBSERVATION_SCHEMA_ID
from ontology_rgat.minimal.constants import DEFAULT_CONSTANTS as C
from ontology_rgat.minimal.graph_policy import MinimalGraphPolicy
from ontology_rgat.minimal.observation import (VECTOR_SIZE, LandingObservation,
                                                ObservationAssembler, OwnState,
                                                PadDetectionSample)
from ontology_rgat.minimal.ontology import (CONDITIONAL_EDGES, EDGE_INDEX, EDGES,
                                             NODES, MinimalOntology, schema_hash)
from ontology_rgat.minimal.pad_loss import (DRIFT, HIGH_ALTITUDE, LOST,
                                             TERMINAL_OCCLUSION, TIMEOUT,
                                             TRANSIENT_DROPOUT, VISIBLE, PadMemory)
from ontology_rgat.minimal.supervisor import (MODE_ABORT_HOLD, MODE_DESCENT_HOLD,
                                               MODE_NOMINAL, MODE_TERMINAL_COMMIT,
                                               MinimalSafetySupervisor)

LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
PINNED_SCHEMA_HASH = "3f0f4e9d0a872176aaafa87f0ffc0cd17fd4e4bcf326f34fc0f3fcef5382f49c"


def obs(t, own=(0, 0, 2.0), vel=(0, 0, 0), rel=(0, 0, -2.0), detected=True,
        capture=None, own_valid=True):
    capture = t - 0.075 if capture is None and detected else capture
    age = C.pad_age_cap_s if capture is None else t - capture
    return LandingObservation(t, np.array(own, float), np.array(vel, float), own_valid,
                              0.0, np.array(rel, float), detected, age, capture)


def descent(lose_below=0.25, offset=0.02, pad_velocity=(0.0, 0.0), sink=0.3,
            lose_lateral_after=None, steps=120):
    """Static or moving pad, vehicle sinking at ``sink`` while tracking it."""
    out, z, x_rel = [], 2.0, offset
    last_capture, last_rel = None, None
    for k in range(steps):
        t = k * 0.1
        visible = z > lose_below and (lose_lateral_after is None or t < lose_lateral_after)
        rel = np.array([x_rel, 0.0, -z])
        if visible:
            last_capture, last_rel = t - 0.075, rel
        held = last_rel if last_rel is not None else np.zeros(3)
        out.append(obs(t, own=(0, 0, z), vel=(0, 0, -sink if z > 0.12 else 0.0),
                       rel=held, detected=visible, capture=last_capture))
        z = max(z - sink * 0.1, 0.12)
        x_rel += pad_velocity[0] * 0.1
    return out


# ------------------------------------------------------------------ observation
def test_vector_layout_and_roundtrip():
    o = obs(1.0, own=(1, 2, 3), vel=(4, 5, 6), rel=(7, 8, -9))
    v = o.vector()
    assert v.shape == (VECTOR_SIZE,) == (13,)
    np.testing.assert_allclose(v[:9], [1, 2, 3, 4, 5, 6, 7, 8, -9])
    back = LandingObservation.from_vector(1.0, v)
    np.testing.assert_allclose(back.vector(), v, atol=1e-6)
    assert o.schema_id == OBSERVATION_SCHEMA_ID


def test_assembler_rotates_at_capture_and_holds_on_miss():
    asm = ObservationAssembler()
    yaw90 = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    asm.on_odometry(OwnState(0.0, np.zeros(3), np.zeros(3), yaw90, True))
    asm.on_odometry(OwnState(0.1, np.zeros(3), np.zeros(3), LEVEL, True))
    # Board 1 m ahead of the body (FLU +x), captured while yawed 90 deg -> ENU +y.
    asm.on_pad_detection(PadDetectionSample(0.0, True, np.array([1.0, 0.0, -2.0])))
    o = asm.assemble(0.1)
    np.testing.assert_allclose(o.pad_relative_position, [0, 1, -2], atol=1e-9)
    assert o.pad_detected and o.pad_age_s == pytest.approx(0.1)
    asm.on_pad_detection(PadDetectionSample(0.1, False, np.zeros(3)))
    o = asm.assemble(0.3)
    assert not o.pad_detected
    np.testing.assert_allclose(o.pad_relative_position, [0, 1, -2], atol=1e-9)
    assert o.pad_age_s == pytest.approx(0.3)


def test_never_detected_is_zero_at_age_cap():
    asm = ObservationAssembler()
    asm.on_odometry(OwnState(0.0, np.zeros(3), np.zeros(3), LEVEL, True))
    o = asm.assemble(0.0)
    assert not o.ever_detected and o.pad_age_s == C.pad_age_cap_s
    np.testing.assert_array_equal(o.pad_relative_position, 0.0)


# -------------------------------------------------------------------- pad loss
def classes(observations):
    memory = PadMemory()
    return [memory.update(o) for o in observations]


def test_terminal_descent_is_terminal_then_times_out():
    result = classes(descent())
    names = [a.pad_class for a in result]
    first_miss = names.index(TRANSIENT_DROPOUT)
    assert all(n == VISIBLE for n in names[:first_miss])
    terminal = [i for i, n in enumerate(names) if n == TERMINAL_OCCLUSION]
    assert terminal and terminal[0] <= first_miss + 4
    lost = [i for i, a in enumerate(result) if a.pad_class == LOST]
    assert lost and result[lost[0]].reason & TIMEOUT
    # latched: contiguous TERMINAL block, no flicker
    assert terminal == list(range(terminal[0], terminal[-1] + 1))


def test_loss_at_altitude_is_lost_not_terminal():
    result = classes(descent(lose_below=1.2))
    after = [a for a in result if a.pad_class not in (VISIBLE, TRANSIENT_DROPOUT)]
    assert after and all(a.pad_class == LOST for a in after)
    assert after[0].reason & HIGH_ALTITUDE


def test_loss_while_off_centre_is_drift():
    result = classes(descent(offset=0.4))
    after = [a for a in result if a.pad_class not in (VISIBLE, TRANSIENT_DROPOUT)]
    assert after and all(a.pad_class == LOST for a in after)
    assert after[0].reason & DRIFT


def test_moving_pad_dead_reckons_out_of_the_terminal_gate():
    # Pad drives off at 0.5 m/s: centred when lost, but the predicted offset
    # leaves the gate within the commit window.
    result = classes(descent(offset=0.0, pad_velocity=(0.5, 0.0)))
    assert any(a.pad_class == LOST and a.reason & DRIFT for a in result)


def test_pad_velocity_from_consecutive_detections():
    memory = PadMemory()
    for k in range(10):
        t = 0.1 * k
        rel = (0.3 * t, 0.0, -2.0)  # pad 0.3 m/s along x relative to a hovering UAV
        a = memory.update(obs(t, rel=rel, capture=t))
    assert a.relative_velocity_valid
    np.testing.assert_allclose(a.relative_velocity, [0.3, 0, 0], atol=1e-6)


# --------------------------------------------------------------------- ontology
def test_schema_is_pinned():
    assert schema_hash() == PINNED_SCHEMA_HASH, (
        "the ontology schema changed: bump ONTOLOGY_SCHEMA_ID and re-pin")


def test_graph_shape_bounds_and_conditional_edges():
    ontology = MinimalOntology()
    series = descent()
    graphs = [ontology.build(o) for o in series]
    for g in graphs:
        assert g.features.shape == (len(NODES), 8)
        assert np.all(np.abs(g.features) <= 1.0)
        assert g.edge_weight.shape == (len(EDGES),)
        assert np.all((g.edge_weight >= 0) & (g.edge_weight <= 1))
    observes = EDGE_INDEX[("CameraObservation", "observes", "LandingPad")]
    supports = EDGE_INDEX[("TerminalOcclusion", "supports", "DescentReadiness")]
    visible = graphs[0]
    terminal = next(g for g in graphs if g.assessment.pad_class == TERMINAL_OCCLUSION)
    assert visible.edge_weight[observes] > terminal.edge_weight[observes]
    assert terminal.edge_weight[supports] > 0.5 > visible.edge_weight[supports]
    unconditional = [i for i, e in enumerate(EDGES) if e not in CONDITIONAL_EDGES]
    assert np.all(terminal.edge_weight[unconditional] == 1.0)


def test_ontology_and_supervisor_classify_identically():
    ontology, supervisor = MinimalOntology(), MinimalSafetySupervisor()
    for o in descent(lose_below=0.25) + descent(lose_below=1.2):
        g, d = ontology.build(o), supervisor.step(o, [0, 0, -0.5])
        assert g.assessment.pad_class == d.assessment.pad_class
        assert g.assessment.reason == d.assessment.reason


# ------------------------------------------------------------------- supervisor
def test_descent_hold_is_symmetric_and_per_axis():
    s = MinimalSafetySupervisor()
    s.step(obs(0.0, rel=(0.1, 0, -2.0)), [0, 0, 0])  # start handover
    # Pad centred (view not threatened) but localization just went invalid:
    # descent is not authorized, and a climb request is held like a descent.
    o = obs(5.0, rel=(0.1, 0.0, -2.0), vel=(0, 0, 0.2), own_valid=False)
    d = s.step(o, [0.3, 0.0, 1.0])
    assert d.mode == MODE_DESCENT_HOLD
    assert d.applied[2] == pytest.approx(-0.2 / C.vertical_hold_time_constant_s)
    assert d.intervened == (False, False, True)


def test_hold_permits_a_bounded_climb_when_the_pad_is_leaving_the_frame():
    """Seed 4138: the symmetric hold blocked the climb that keeps the pad in view."""
    s = MinimalSafetySupervisor()
    s.step(obs(0.0, rel=(1.6, 0, -2.0)), [0, 0, 0])
    d = s.step(obs(5.0, rel=(1.6, 0.0, -2.0), vel=(0, 0, 0.0)), [0.0, 0.0, 2.0])
    assert d.mode == MODE_DESCENT_HOLD
    expected = C.view_recovery_climb_m_s / C.vertical_hold_time_constant_s
    assert d.applied[2] == pytest.approx(expected)       # bounded, not the 2.0 asked


def test_sink_brake_and_handover_floor():
    s = MinimalSafetySupervisor()
    d = s.step(obs(0.0, rel=(0, 0, -2.0)), [0, 0, -1.0])
    assert d.mode == MODE_NOMINAL and d.applied[2] == 0.0  # handover floor
    d = s.step(obs(3.0, rel=(0, 0, -0.5), vel=(0, 0, -1.5)), [0, 0, -1.0])
    assert d.applied[2] > 0  # sinking 1.5 m/s at 0.5 m is braked


def test_terminal_commit_and_abort():
    s = MinimalSafetySupervisor()
    modes = [s.step(o, [0.0, 0.0, -0.5]).mode for o in descent()]
    assert MODE_TERMINAL_COMMIT in modes and modes[-1] == MODE_ABORT_HOLD
    s = MinimalSafetySupervisor()
    modes = [s.step(o, [0.0, 0.0, -0.5]).mode for o in descent(lose_below=1.2)]
    assert MODE_TERMINAL_COMMIT not in modes and modes[-1] == MODE_ABORT_HOLD


def test_abort_releases_when_the_pad_returns():
    s = MinimalSafetySupervisor()
    s.step(obs(0.0), [0, 0, 0])
    d = s.step(obs(5.0, detected=False, capture=0.9, rel=(0, 0, -2.0)), [0, 0, 0])
    assert d.mode == MODE_ABORT_HOLD and d.applied[2] > 0
    d = s.step(obs(5.1), [0, 0, 0])
    assert d.mode != MODE_ABORT_HOLD


# ----------------------------------------------------------------------- policy
def test_policy_runs_and_relations_are_active(tmp_path):
    policy = MinimalGraphPolicy()
    g = MinimalOntology().build(obs(0.0))
    action = policy.act(g.features, g.edge_weight)
    assert action.shape == (3,) and np.all(np.abs(action) <= np.array(C.max_acceleration_m_s2))
    assert policy.relational_activity(g.features, g.edge_weight) > 0.01
    path = tmp_path / "policy.pt"
    policy.save(path)
    again = MinimalGraphPolicy.load(path)
    np.testing.assert_allclose(again.act(g.features, g.edge_weight), action, atol=1e-6)
    payload = torch.load(path, weights_only=False)
    payload["schema_hash"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError):
        MinimalGraphPolicy.load(path)


def test_zero_edge_weight_removes_the_message():
    policy = MinimalGraphPolicy()
    g = MinimalOntology().build(obs(0.0))
    features = torch.as_tensor(g.features).unsqueeze(0)
    weights = torch.as_tensor(g.edge_weight).clone()
    self_rel = [i for i, e in enumerate(EDGES) if e[1] != "self"]
    weights[self_rel] = 0.0
    gated = policy.encode(features, weights.unsqueeze(0))
    alone = policy.encode(features, torch.as_tensor(g.edge_weight).unsqueeze(0),
                          relation_gates=policy._self_only)
    torch.testing.assert_close(gated, alone)


# ------------------------------------------------------------------ ROS adapters
def _landing_node_module(name):
    import importlib
    import sys
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[1] / "ros2_ws/src/ontology_rgat_landing"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    return importlib.import_module(f"ontology_rgat_landing.{name}")


def test_detector_output_rotates_into_the_true_relative_position():
    """Synthetic v12 board -> PnP -> body FLU -> ENU equals pad minus body."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "isaac_sim"))
    from config_loader import load_config
    from marker_vision import MarkerBoard, MarkerPoseEstimator
    from test_marker_vision import render_board_view
    from ontology_rgat.minimal.observation import quat_wxyz_to_matrix

    detector = _landing_node_module("aruco_pad_detector")
    vision = load_config(root / "config/spatial-isaac-system-v12.yaml")["vision"]
    board = MarkerBoard.from_config(vision["board"])
    mount = vision["camera"]["mount_translation_flu_m"]
    body, yaw = np.array([0.3, -0.2, 1.5]), math.radians(30.0)
    image, camera_matrix = render_board_view(body, yaw, board=board, width=640,
                                             height=480, fov_deg=90.0, mount=mount)
    estimator = MarkerPoseEstimator(board, camera_matrix, mount)
    seen = estimator.detect(image)
    assert seen.detected
    q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    rel_enu = quat_wxyz_to_matrix(q_yaw) @ detector.board_origin_in_body(seen)
    np.testing.assert_allclose(rel_enu, -body, atol=0.03)


def test_image_decoding_handles_bgr_and_row_padding():
    detector = _landing_node_module("aruco_pad_detector")

    class Msg:  # duck-typed sensor_msgs/Image
        encoding, height, width = "bgr8", 2, 3
        step = 3 * 3 + 2  # two padding bytes per row
        data = bytes([0, 0, 255, 0, 255, 0, 255, 0, 0, 9, 9] * 2)

    image = detector.image_to_array(Msg)
    assert image.shape == (2, 3, 3)
    np.testing.assert_array_equal(image[0, 0], [255, 0, 0])   # BGR red -> RGB red
    np.testing.assert_array_equal(image[0, 2], [0, 0, 255])


def test_px4_frames_convert_to_enu_flu():
    bridge = _landing_node_module("px4_localization_bridge")
    assert bridge.ned_to_enu(1.0, 2.0, -3.0) == (2.0, 1.0, 3.0)
    from ontology_rgat.minimal.observation import quat_wxyz_to_matrix
    # PX4 identity attitude = nose north, level -> ENU/FLU yaw +90 deg.
    q = bridge.quat_ned_frd_to_enu_flu([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_wxyz_to_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(quat_wxyz_to_matrix(q) @ [0, 0, 1], [0, 0, 1], atol=1e-12)


# ------------------------------------------------- disturbance and sink limit
def test_hold_rejects_a_constant_vertical_disturbance():
    """Seed 4107: a P-only rate hold settled at -0.21 m/s under a 0.42 m/s^2 force."""
    s = MinimalSafetySupervisor()
    s.step(obs(0.0, rel=(1.6, 0, -2.0)), [0, 0, 0])
    v, force, applied = -0.21, -0.42, []
    for k in range(1, 60):
        t = 2.5 + 0.1 * k  # past the handover window
        d = s.step(obs(t, rel=(1.6, 0, -2.0), vel=(0, 0, v)), [0, 0, -1.0])
        assert d.mode == MODE_DESCENT_HOLD
        applied.append(d.applied[2])
        if len(applied) > 1:  # one decision of actuation delay
            v += 0.1 * (applied[-2] + force)
    assert abs(v) < 0.03, v
    assert s.disturbance[2] == pytest.approx(force, abs=0.05)


def test_allowed_sink_reaches_the_touchdown_target_with_lag():
    assert C.allowed_sink(C.touchdown_height_m) == pytest.approx(0.24)
    assert C.allowed_sink(0.37) < 0.55          # /9 form admitted 0.66 here
    heights = np.linspace(0.12, 3.0, 50)
    sinks = [C.allowed_sink(h) for h in heights]
    assert all(b >= a for a, b in zip(sinks, sinks[1:]))


# ----------------------------------------------------------- local adapter
def test_local_env_teacher_episode_and_no_truth_in_observation():
    from ontology_rgat.minimal.local_env import MinimalLandingEnv
    from ontology_rgat.minimal.teacher import MinimalTeacher
    env = MinimalLandingEnv()
    o, _ = env.reset(seed=4100)
    teacher, done, steps = MinimalTeacher(), False, 0
    while not done and steps < 400:
        assert isinstance(o, LandingObservation) and o.vector().shape == (13,)
        o, reward, done, info = env.step(teacher.act(o))
        steps += 1
    assert done and info["status"] != "RUNNING"
    assert np.isfinite(reward)


def test_three_arms_share_the_stream_and_reload(tmp_path):
    from ontology_rgat.minimal.arms import ARMS, build_arm, load_arm, save_arm
    from ontology_rgat.minimal.rollout import ArmController
    for name in ARMS:
        arm = build_arm(name, seed=3)
        ctl = ArmController(arm)
        action = ctl.act(obs(0.0))
        assert action.shape == (3,)
        save_arm(arm, name, tmp_path / f"{name}.pt")
        loaded_name, again, _ = load_arm(tmp_path / f"{name}.pt")
        assert loaded_name == name
        np.testing.assert_allclose(ArmController(again).act(obs(0.0)), action, atol=1e-6)


# ---------------------------------------------- measured classifier defects
def test_lost_is_absorbing_until_redetection():
    """Seed 4110: LOST flipped to TERMINAL 1.7 s later with the pad 1.0 m away."""
    memory = PadMemory()
    for k in range(6):  # tracked and centred, but at 0.6 m
        memory.update(obs(0.1 * k, rel=(0.05, 0.0, -0.6), capture=0.1 * k))
    classes = [memory.update(obs(0.5 + 0.1 * k, rel=(0.05, 0.0, -0.6), detected=False,
                                  capture=0.5, vel=(0, 0, -0.3))).pad_class
               for k in range(1, 20)]
    first_lost = classes.index(LOST)
    assert all(c == LOST for c in classes[first_lost:])
    assert memory.update(obs(3.0, rel=(0.0, 0.0, -0.3), capture=3.0)).pad_class == VISIBLE


def test_no_terminal_without_a_known_pad_velocity():
    memory = PadMemory()
    memory.update(obs(0.0, rel=(0.02, 0.0, -0.25), capture=0.0))  # one detection only
    result = [memory.update(obs(0.1 * k, rel=(0.02, 0.0, -0.25), detected=False,
                                 capture=0.0, vel=(0, 0, -0.2))) for k in range(1, 10)]
    assert all(a.pad_class != TERMINAL_OCCLUSION for a in result)
    from ontology_rgat.minimal.pad_loss import NO_PAD_VELOCITY
    assert result[-1].reason & NO_PAD_VELOCITY


def test_pad_velocity_is_a_windowed_slope_not_a_lagging_average():
    """Seed 4106: a 0.6 m/s relative velocity was missed by an EMA of differences."""
    rng = np.random.default_rng(0)
    memory = PadMemory()
    for k in range(10):
        t = 0.05 * k
        rel = np.array([0.6 * t, 0.0, -0.3]) + rng.normal(0, 0.02, 3) * [1, 1, 0]
        a = memory.update(obs(t, rel=tuple(rel), capture=t))
    assert a.relative_velocity_valid
    assert a.relative_velocity[0] == pytest.approx(0.6, abs=0.15)


def test_descent_gate_checks_relative_velocity_near_the_pad():
    """Seed 3017: one re-detection 0.08 m off-centre at 0.29 m, pad leaving at 0.92 m/s."""
    s = MinimalSafetySupervisor()
    s.step(obs(0.0, rel=(0.0, 0.0, -2.0)), [0, 0, 0])
    for k in range(6):  # pad sliding away at 0.9 m/s, vehicle low
        t = 3.0 + 0.1 * k
        d = s.step(obs(t, rel=(0.9 * (t - 3.0) - 0.4, 0.0, -0.29), capture=t), [0, 0, -1.0])
    assert d.mode == MODE_DESCENT_HOLD
    s = MinimalSafetySupervisor()
    s.step(obs(0.0, rel=(0.0, 0.0, -2.0)), [0, 0, 0])
    for k in range(6):  # same height, pad still under the vehicle
        t = 3.0 + 0.1 * k
        d = s.step(obs(t, rel=(0.03, 0.0, -0.29), capture=t), [0, 0, -1.0])
    assert d.mode == MODE_NOMINAL
