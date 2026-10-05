"""Spatial policy information boundary, geometry and actual PPO/checkpoint contracts."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from ontology_rgat.two_axis.models_v28 import ReferenceHead
from ontology_rgat.spatial.core import (
    SpatialConfig,
    Measurement,
    Estimator,
    observation,
    safety_status,
    supervised_action,
    REGISTRY_HASH,
    FIELDS,
)
from ontology_rgat.spatial.environment import SpatialLandingEnv, Evaluator, Truth
from ontology_rgat.spatial.training import (
    SpatialAgent,
    save_agent,
    load_agent,
    train_arm,
)
from ontology_rgat.two_axis.training import PPOHyperparameters, PPOTrainer
from ontology_rgat.two_axis.learning import collect_rollout
from ontology_rgat.two_axis.models import POLICY_MODES


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def measured(t=0.0, position=(0.1, -0.2, 2.0), sample=1):
    return Measurement(
        t,
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
        np.array([1.0, 0, 0, 0]),
        np.zeros(3),
        None if position is None else np.array(position),
        0.0 if position is None else 0.9,
        sample,
    )


#: The frozen v5 rung. The tests pinned to it exercise the 35-field packet, the
#: decaying-acceleration estimator and the historical module-level
#: safety_status/supervised_action helpers -- none of which the active contract
#: uses, and the last of which it refuses outright. Naming the rung is what
#: makes these contract tests instead of tests of whatever the default happens
#: to be; that ambiguity is how the default moved without anyone noticing.
LEGACY_SCHEMA = "spatial-causal-rgat/5"


def legacy_config(**overrides):
    return replace(SpatialConfig(), schema=LEGACY_SCHEMA, **overrides)


@pytest.mark.parametrize('version',['5','9'])
def test_allowlist_is_independent_of_truth_contact_reward_and_platform_pose(version):
    state = {
        "world": {"position": [1, 2, 3], "velocity": [0, 0, 0]},
        "quaternion_wxyz": [1, 0, 0, 0],
        "angular_velocity": [0, 0, 0],
        "estimator_valid": True,
        "extra": {
            "spatial_clock": {"valid": True, "sim_time_s": 0.0},
            "optical_measurement": {
                "valid": True,
                "frame": "pad_enu",
                "sample_id": 1,
                "position_m": [0.1, -0.2, 2.0],
                "confidence": 0.9,
            },
        },
    }
    cfg = replace(SpatialConfig(),schema=f'spatial-causal-rgat/{version}')
    if version=='9':
        state['extra']['optical_measurement'].update(capture_time_s=0.,
            own_position_at_capture_m=[1.,2.,3.],own_quaternion_at_capture_wxyz=[1.,0.,0.,0.])

    def encode(s):
        e = Estimator(cfg)
        e.update(Measurement.from_wire(s,capture_aligned=cfg.reference_context))
        return observation(e, safety_status(e, cfg), 0.0, cfg)

    first = encode(state)
    state.update(
        truth={"position": [999] * 3},
        position=[-999] * 3,
        velocity=[999] * 3,
        reward=1e30,
        pad={"position": [999] * 3},
        landed=True,
    )
    state["extra"]["pad_contact"] = True
    state["extra"]["truth_contact_event"] = {
        "relative_position": [999] * 3,
        "reward": 1e30,
    }
    second = encode(state)
    np.testing.assert_array_equal(first.packet.values, second.packet.values)
    np.testing.assert_array_equal(first.graph.X, second.graph.X)
    with pytest.raises(TypeError):
        Estimator(cfg).update(state)


def test_runtime_source_hash_detects_world_changes_not_results(tmp_path):
    from ontology_rgat.spatial.runtime_contract import runtime_source_hash
    world = tmp_path / "isaac_sim"
    world.mkdir()
    source = world / "world.py"
    source.write_text("version = 1\n")
    first = runtime_source_hash(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    assert runtime_source_hash(tmp_path) == first
    source.write_text("version = 2\n")
    assert runtime_source_hash(tmp_path) != first


def test_stale_isaac_source_is_rejected_before_any_reset(monkeypatch):
    from ontology_rgat.spatial.environment import IsaacBackend
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    calls = []
    class Bridge:
        def __init__(self, _): pass
        def transact(self, *_, **kwargs):
            return {"extra": {"spatial_clock": {
                "valid": True, "profile_sha256": deployment_profile()["sha256"],
                "source_sha256": "stale"}}}
        def close(self): calls.append("close")
        def reset(self, *_): calls.append("reset")
    monkeypatch.setattr("ontology_rgat.bridge.PX4Bridge", Bridge)
    with pytest.raises(ValueError, match="stale"):
        IsaacBackend(SpatialConfig())
    assert calls == ["close"]


def test_signed_packet_and_graph_represent_both_horizontal_axes():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured())
    s = safety_status(e, cfg)
    o = observation(e, s, 0, cfg)
    assert o.packet.values.shape == (35,) and o.graph.X.shape == (9, 12)
    assert o.packet.registry_sha256 == REGISTRY_HASH
    assert (
        o.packet.values[FIELDS.index("rx")] > 0
        and o.packet.values[FIELDS.index("ry")] < 0
    )
    assert o.graph.X[4, 0] > 0 and o.graph.X[4, 1] < 0


def test_v5_visual_outage_decays_acceleration_and_brakes_own_xy_velocity():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured())
    e.pad_a[:] = [1., -1., 0.]
    e.update(measured(1.5, None, sample=2))
    np.testing.assert_allclose(e.pad_a[:2], np.array([1., -1.]) / np.e)
    for i in range(2, 5):
        e.update(replace(measured(float(i), None, sample=i+1),
                         own_velocity=np.array([1., -2., 0.])))
    safety = safety_status(e, cfg)
    assert safety.abort
    action = supervised_action([1., -1., -1.], e, safety, cfg)
    assert action[0] < 0 and action[1] > 0
    # Historical v4 remains available explicitly, not silently reinterpreted.
    old = replace(cfg, schema="spatial-causal-rgat/4")
    np.testing.assert_array_equal(supervised_action([1., -1., -1.], e, safety, old)[:2],
                                  [1., -1.])


def test_v4_weight_transfer_requires_explicit_optin_and_new_nominal_completion(tmp_path):
    new = replace(legacy_config(), horizon=1.)
    old = replace(new, schema="spatial-causal-rgat/4")
    source = tmp_path / "v4.pt"
    agent = SpatialAgent("ppo_semantic_flat", old, 102)
    save_agent(source, agent, old, eligible=True, completed_episodes=100)
    kwargs = dict(mode=agent.mode, cfg=new, seed=102,
                  hyper=PPOHyperparameters(iterations=1, decisions_per_iteration=2),
                  activation_iterations=0, curriculum_enabled=False,
                  initial_checkpoint=source)
    with pytest.raises(ValueError, match="exact spatial contract"):
        train_arm(output=tmp_path / "implicit", **kwargs)
    summary = train_arm(output=tmp_path / "explicit", allow_v4_initialization=True, **kwargs)
    assert summary["selected_checkpoint"] is None
    _, meta = load_agent(tmp_path / "explicit/checkpoint_final.pt", new)
    assert not meta["eligible"]
    migration = meta["initialization_source"]["contract_migration"]
    assert not migration["eligibility_transferred"]
    assert migration["from"]["schema"] == old.schema
    assert migration["to"]["schema"] == new.schema


def test_100hz_propagation_and_no_repeated_optical_assimilation():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured())
    e.update(measured(0.1, sample=1))
    assert e.updates == 10 and e.age == pytest.approx(0.1)
    assert e.std > 0.03
    with pytest.raises(ValueError):
        e.update(measured(0.05, sample=3))
    with pytest.raises(ValueError):
        e.update(measured(10.0, sample=4))


def test_missing_pose_does_not_become_zero_position_or_immediate_abort_terminal():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured())
    for i in range(1, 35):
        e.update(measured(i * 0.1, None, sample=i))
    assert e.age > 3 and safety_status(e, cfg).abort
    assert np.linalg.norm(e.r) > 1


@pytest.mark.parametrize("axis", [0, 1])
def test_xy_safety_and_descent_gate_are_symmetric(axis):
    cfg = legacy_config()
    e = Estimator(cfg)
    r = np.array([0.0, 0.0, 0.6])
    r[axis] = 2.0
    e.update(measured(position=r))
    safety = safety_status(e, cfg)
    assert safety.inhibited
    applied = supervised_action([0, 0, -1], e, safety, cfg)
    assert applied[2] >= 0


def test_authorized_contact_is_reachable_and_unsafe_precedes_timeout():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured(position=[0, 0, 0.13]))
    s = safety_status(e, cfg)
    assert not s.inhibited
    truth = Truth(
        np.array([0, 0, 0.12]), np.array([0, 0, -0.15]), np.zeros(2), np.zeros(3), True
    )
    evaluator = Evaluator(cfg)
    evaluator.reset(truth)
    kwargs = dict(
        elapsed=cfg.horizon,
        dt=0.1,
        action=np.zeros(3),
        safety=s,
        abort_elapsed=99,
        bearings=np.zeros(2),
        visible=True,
    )
    _, status, _ = evaluator.evaluate(truth, **kwargs)
    assert status == "SUCCESS"
    _, status, _ = evaluator.evaluate(
        replace(truth, relative_velocity=np.array([0, 1, -0.15])), **kwargs
    )
    assert status == "UNSAFE_CONTACT"


def test_flat_and_graph_raw_initialization_and_spatial_vertical_gate():
    cfg = legacy_config()
    flat = SpatialAgent(POLICY_MODES[1], cfg, 10)
    graph = SpatialAgent(POLICY_MODES[2], cfg, 10)
    env = SpatialLandingEnv(cfg)
    obs, _ = env.reset(seed=11)
    p, g = graph.tensors(obs)
    assert torch.equal(flat.actor(p, g)[0], graph.actor(p, g)[0])
    assert torch.equal(flat.critic(p, g), graph.critic(p, g))
    g = g.repeat(3, 1, 1)
    g[:, 7, 0] = torch.tensor([0.0, 0.5, 1.0])
    with torch.no_grad():
        graph.actor.encoder.readout.bias.fill_(1.0)
        graph.actor.residual.weight.fill_(-1.0)
    delta = graph.actor.relational_delta(g)
    assert delta[0, 2] == 0 and torch.allclose(delta[1, 2], delta[2, 2] * 0.5)
    assert delta[0, 0] != 0 and delta[0, 1] != 0


@pytest.mark.parametrize("mode", POLICY_MODES)
def test_3d_ppo_is_finite_and_checkpoint_reload_exact(tmp_path, mode):
    cfg = replace(SpatialConfig(), horizon=0.4)
    agent = SpatialAgent(mode, cfg, 32)
    env = SpatialLandingEnv(cfg)
    batch = collect_rollout(agent, env, seed=10, decisions=8)
    metrics = PPOTrainer(agent, PPOHyperparameters()).update(
        batch, discount_time_constant_s=70.0
    )
    assert all(np.isfinite(v) for v in metrics.values())
    path = tmp_path / "model.pt"
    save_agent(path, agent, cfg, eligible=True)
    loaded, _ = load_agent(path, cfg)
    np.testing.assert_array_equal(
        agent.act(batch[0].observation, deterministic=True)[1],
        loaded.act(batch[0].observation, deterministic=True)[1],
    )
    with pytest.raises(ValueError):
        load_agent(path, replace(cfg, horizon=0.5))


def test_terminal_potential_is_zero_and_reward_order():
    cfg = legacy_config()
    e = Estimator(cfg)
    e.update(measured())
    truth = Truth(np.array([0, 0, 2.0]), np.zeros(3), np.zeros(2), np.zeros(3), False)
    evaluator = Evaluator(cfg)
    evaluator.reset(truth)
    before = evaluator.previous_goal
    _, status, parts = evaluator.evaluate(
        truth,
        elapsed=70,
        dt=0.1,
        action=np.zeros(3),
        safety=safety_status(e, cfg),
        abort_elapsed=0,
        bearings=np.zeros(2),
        visible=True,
    )
    assert status == "TASK_TIMEOUT" and parts["potential"] == pytest.approx(2 * before)
    b = Evaluator.BONUSES
    assert b["SUCCESS"] > b["TASK_TIMEOUT"] > b["SAFE_ABORT"] > b["UNSAFE_CONTACT"]


def test_spatial_curriculum_one_preserves_nominal_draws_exactly():
    a = SpatialLandingEnv()
    b = SpatialLandingEnv(difficulty=1.0)
    oa, _ = a.reset(seed=614)
    ob, _ = b.reset(seed=614)
    np.testing.assert_array_equal(oa.packet.values, ob.packet.values)
    for _ in range(5):
        oa, ra, _, _, ia = a.step([0.1, -0.1, 0.0])
        ob, rb, _, _, ib = b.step([0.1, -0.1, 0.0])
        np.testing.assert_array_equal(oa.packet.values, ob.packet.values)
        assert ra == rb and ia == ib


def test_spatial_pad_cv_ca_cv_is_position_and_velocity_continuous():
    from ontology_rgat.spatial.environment import LocalBackend

    backend = LocalBackend(SpatialConfig())
    backend.reset(16)
    for boundary in (backend.t1, backend.t1 + backend.t2):
        left = backend.pad_state(boundary - 1e-7)
        right = backend.pad_state(boundary + 1e-7)
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_camera_geometry_uses_actual_attitude_and_matches_render_geometry():
    from ontology_rgat.spatial.core import camera_bearings
    from scipy.spatial.transform import Rotation
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac_sim"))
    from keypoint_geometry import CameraModel, project_pad_points

    cfg = SpatialConfig()
    r = np.array([0.2, -0.1, 2.0])
    for angles in [(0, 0, 0), (0.2, -0.3, 0.8), (1.2, 0, 0)]:
        q = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
        bearings, depth = camera_bearings(r, q, cfg)
        _, expected_depth, optical = project_pad_points(
            [[0, 0, 0]], r, q, CameraModel(640, 480, 90, 90)
        )
        np.testing.assert_allclose(bearings, np.arctan2(optical[0, :2], optical[0, 2]))
        assert depth == pytest.approx(expected_depth[0])
    backend = SpatialLandingEnv(cfg).backend
    backend.reset(3)
    backend.position[:] = [0, 0, 2]
    backend.angles[:] = [np.pi / 2, 0, 0]
    backend.camera_time = -np.inf
    assert backend.measure().optical_position is None


def test_near_field_readiness_prefers_sub_limit_sink_to_blind_hover():
    cfg = SpatialConfig()
    evaluator = Evaluator(cfg)
    at_rest = Truth(np.array([0, 0, 0.2]), np.zeros(3), np.zeros(2), np.zeros(3), False)
    descending = replace(
        at_rest, relative_velocity=np.array([0, 0, -0.8 * cfg.touchdown_z_speed])
    )
    assert evaluator.terms(descending)[1] > evaluator.terms(at_rest)[1]


def test_v3_checkpoint_is_not_silently_relabelled_as_optical_v4(tmp_path):
    old = replace(SpatialConfig(), schema="spatial-causal-rgat/3")
    new = SpatialConfig()
    path = tmp_path / "old.pt"
    save_agent(path, SpatialAgent("ppo_vector_canonical", old, 3), old, eligible=True)
    assert old.registry_hash != new.registry_hash
    load_agent(path, old)
    with pytest.raises(ValueError, match="exact spatial contract"):
        load_agent(path, new)


def test_local_camera_runs_at_20hz_without_freshening_repeated_frames():
    env = SpatialLandingEnv(legacy_config())
    env.reset(seed=44)
    backend = env.backend
    first = backend.measure()
    backend.t = 0.01
    second = backend.measure()
    assert first.sample_id == second.sample_id
    np.testing.assert_array_equal(first.optical_position, second.optical_position)
    backend.t = 0.05
    assert backend.measure().sample_id == first.sample_id + 1


def test_terminal_coast_is_bounded_by_last_optical_alignment_and_uncertainty():
    cfg = legacy_config()
    est = Estimator(cfg)
    est.update(
        replace(
            measured(position=[0.0, 0.0, 0.3]), own_velocity=np.array([0.0, 0.0, -0.1])
        )
    )
    for i in range(1, 9):
        est.update(
            replace(
                measured(i * 0.1, None, sample=i),
                own_velocity=np.array([0.0, 0.0, -0.1]),
            )
        )
    assert est.age == pytest.approx(0.8) and not safety_status(est, cfg).inhibited
    est.std = 0.3
    assert safety_status(est, cfg).inhibited
    est.std = 0.1
    est.last_optical_r[0] = 0.4
    assert safety_status(est, cfg).inhibited


def test_level_ground_motion_prior_cannot_invent_vertical_pad_acceleration():
    cfg = SpatialConfig()
    est = Estimator(cfg)
    est.update(measured())
    for i in range(1, 10):
        est.update(measured(i * 0.05, [0, 0, 2 + 0.01 * (-1) ** i], i + 1))
    assert est.pad_v[2] == 0 and est.pad_a[2] == 0


@pytest.mark.parametrize("armed,nav", [(False, 14), (False, 18), (True, 18), (True, None)])
def test_isaac_policy_rejects_disarmed_or_non_offboard_state(armed, nav):
    from ontology_rgat.spatial.environment import IsaacBackend
    backend = object.__new__(IsaacBackend)
    backend.seed, backend.start_s = 7, 2.0
    state = {"armed": armed, "nav_state": nav}
    with pytest.raises(RuntimeError, match="requires armed PX4 in OFFBOARD"):
        backend.assert_policy_flight(state)
    backend.assert_policy_flight({"armed": True, "nav_state": 14})


@pytest.mark.parametrize("seed,sample_time,allowed", [(7, 3., True), (6, 3., False), (7, 1., False)])
def test_only_fresh_terminal_contact_can_explain_disarm(seed, sample_time, allowed):
    from ontology_rgat.spatial.environment import IsaacBackend
    backend = object.__new__(IsaacBackend)
    backend.seed, backend.start_s = 7, 2.0
    state = {"armed": False, "nav_state": 18, "extra": {
        "truth_contact_event": {"seed": seed, "sample_time_s": sample_time}}}
    if allowed:
        backend.assert_policy_flight(state, allow_contact=True)
    else:
        with pytest.raises(RuntimeError, match="requires armed PX4"):
            backend.assert_policy_flight(state, allow_contact=True)
    with pytest.raises(RuntimeError, match="requires armed PX4"):
        backend.assert_policy_flight(state)


def test_isaac_evaluator_uses_preimpact_snapshot_not_postimpact_imu():
    from ontology_rgat.spatial.environment import IsaacBackend

    backend = object.__new__(IsaacBackend)
    backend.seed = 7
    backend.start_s = 2.0
    event = {
        "seed": 7,
        "sample_time_s": 3.0,
        "relative_position": [0, 0, 0.12],
        "relative_velocity": [0, 0, -0.15],
        "roll_pitch": [0, 0],
        "angular_rate": [0, 0, 0],
    }
    backend.last_state = {
        "angular_velocity": [5, 5, 5],
        "extra": {"pad_contact": True, "truth_contact_event": event},
    }
    truth = backend.truth()
    assert truth.contact
    np.testing.assert_array_equal(truth.angular_rate, [0, 0, 0])


def test_backend_factory_drives_rollouts_validation_and_saved_metadata(tmp_path):
    calls = []

    def factory(cfg):
        calls.append(cfg)
        return SpatialLandingEnv(cfg)

    cfg = replace(SpatialConfig(), horizon=0.2)
    summary = train_arm(
        "ppo_vector_canonical",
        cfg,
        seed=61,
        hyper=PPOHyperparameters(
            iterations=2, decisions_per_iteration=4, evaluation_every=1
        ),
        output=tmp_path,
        activation_iterations=0,
        env_factory=factory,
        training_backend="test-factory",
        curriculum_enabled=False,
    )
    assert len(calls) == 6 and summary["completed_nominal_episodes"] == 4
    assert summary["training_backend"] == "test-factory"
    _, meta = load_agent(tmp_path / summary["selected_checkpoint"], cfg)
    assert meta["training_backend"] == "test-factory" and meta["eligible"]
    import json

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert len(progress["history"]) == 2
    assert progress["completed_nominal_episodes"] == 4
    assert progress["wall_seconds"] > 0


def test_spatial_curriculum_relaxes_training_only_and_preserves_reward_order():
    cfg = SpatialConfig()
    env = SpatialLandingEnv(cfg)
    env.reset(seed=71, difficulty=0)
    assert (
        env.task_cfg.touchdown_z_speed
        == cfg.curriculum.start_touchdown_vertical_speed_m_s
    )
    assert env.task_cfg.touchdown_xy_speed > cfg.touchdown_xy_speed
    assert env.evaluator.unsafe_penalty < Evaluator.BONUSES["SAFE_ABORT"]
    env.reset(seed=71, difficulty=1)
    assert env.task_cfg is cfg
    # Read the constant rather than its value: this assertion pinned a literal
    # -40 and had to be edited when SAFE_ABORT took that magnitude over.
    assert env.evaluator.unsafe_penalty == Evaluator.NOMINAL_UNSAFE_PENALTY
    assert env.evaluator.unsafe_penalty < Evaluator.BONUSES["SAFE_ABORT"]
    assert env.backend.difficulty == 1


def test_angular_curriculum_cannot_relax_nominal_or_isaac():
    cfg = SpatialConfig()
    env = SpatialLandingEnv(cfg)
    env.reset(seed=71, difficulty=0, angular_curriculum_scales=(2, 4))
    assert env.task_cfg.touchdown_tilt == 2 * cfg.touchdown_tilt
    assert env.task_cfg.touchdown_rate == 4 * cfg.touchdown_rate
    a, _ = env.reset(seed=71, difficulty=1, angular_curriculum_scales=(2, 4))
    assert env.task_cfg is cfg
    other = SpatialLandingEnv(cfg)
    b, _ = other.reset(seed=71)
    np.testing.assert_array_equal(a.packet.values, b.packet.values)
    for _ in range(10):
        a, ra, _, _, ia = env.step([0.1, -0.1, 0])
        b, rb, _, _, ib = other.step([0.1, -0.1, 0])
        np.testing.assert_array_equal(a.packet.values, b.packet.values)
        assert ra == rb and ia == ib
    actual = SpatialLandingEnv(cfg, backend=object())
    with pytest.raises(ValueError, match="local training"):
        actual.reset(seed=71, angular_curriculum_scales=(2, 4))


def test_spatial_replay_success_does_not_promote_active_difficulty():
    from ontology_rgat.two_axis.curriculum import CurriculumScheduler, EpisodeRole
    from ontology_rgat.spatial.training import CurriculumTrainingEnv

    cfg = replace(SpatialConfig(), horizon=0.1)
    scheduler = CurriculumScheduler(cfg.curriculum, difficulty=1)
    scheduler.episodes = 5  # The sixth completed episode is easy replay.
    env = CurriculumTrainingEnv(cfg, scheduler, SpatialLandingEnv)
    env.reset(seed=72)
    assert env.role is EpisodeRole.EASY_REPLAY and env.env.difficulty == 0
    env.step(np.zeros(3))
    assert env.episode_history[-1]["difficulty"] == 0
    assert env.episode_history[-1]["role"] == "easy_replay"
    assert scheduler.state()["window_filled"] == 0



def test_attempting_outranks_waiting_on_the_measured_reward():
    """Rank the arms by measuring returns, never by deriving a break-even.

    The derivation this replaces compared discounted TERMINAL values only and
    concluded landing had to already succeed 53.9 % of the time to beat running
    out the clock. That omitted the dense readiness/potential terms, which pay
    an approaching arm a large positive running return, and it drove the
    spatial table away from the 2D reference for nothing. Measured here on the
    objective the trainer actually optimizes: an attempting controller beats
    both passive arms by a wide margin under the shared reference table.

    Diagnostic controller only -- never a teacher, a baseline or a PPO target.
    """
    import math

    from ontology_rgat.spatial.environment import Evaluator

    cfg = SpatialConfig()
    table = Evaluator.BONUSES
    assert (
        table["SUCCESS"] > table["TASK_TIMEOUT"] > table["SAFE_ABORT"]
        > table["UNSAFE_CONTACT"]
    )
    # An unsafe outcome is identified by status, never by matching a magnitude:
    # a table revision once gave SAFE_ABORT the unsafe outcomes' own value.
    assert "SAFE_ABORT" not in Evaluator.UNSAFE_STATUSES

    gains = dict(kp=2.0, kd=3.0, kz=2.0, vdes=0.2, align=0.3, vtol=0.15)
    amax = np.asarray(cfg.max_acceleration)

    def rollout(env, seed, arm):
        env.reset(seed=seed)
        reference = env.backend.velocity.copy()
        total, start = 0.0, env.time
        for _ in range(int(cfg.horizon / cfg.dt) + 2):
            est = env.estimator
            if cfg.direct_acceleration:
                # The direct contract carries no integrated velocity target,
                # so the reference is the measured velocity each step.
                reference = env.backend.velocity.copy()
            r, rv = est.r.copy(), est.rv.copy()
            desired = np.asarray(est.pad_v, dtype=float).copy()
            desired[:2] += -gains["kp"] * r[:2] - gains["kd"] * rv[:2]
            if arm == "attempt":
                aligned = (np.linalg.norm(r[:2]) < gains["align"]
                           and np.linalg.norm(rv[:2]) < gains["vtol"])
                desired[2] = -gains["vdes"] if aligned else gains["kz"] * (1.0 - r[2])
            else:
                desired[2] = gains["kz"] * (2.2 - r[2])
            action = np.clip((desired - reference) / (0.6 * amax), -1, 1)
            if arm == "zero":
                action = np.zeros(3)
            _, reward, done, _, info = env.step(action)
            reference += np.asarray(info["applied_acceleration_m_s2"]) * cfg.dt
            total += float(math.exp(-(env.time - start) / cfg.discount_tau) * reward)
            if done:
                return total, info["status"]
        return total, "NO_TERMINAL"

    seeds = range(3001, 3017)
    returns, landings = {}, {}
    for arm in ("attempt", "hold", "zero"):
        env = SpatialLandingEnv(cfg)
        rows = [rollout(env, seed, arm) for seed in seeds]
        env.close()
        returns[arm] = float(np.mean([value for value, _ in rows]))
        landings[arm] = sum(status == "SUCCESS" for _, status in rows) / len(rows)

    # The task admits landing at all -- if this fails, the ceiling moved and no
    # reward conclusion below it is meaningful. The rate is contract-dependent
    # (the same gains reach 86.7 % on the frozen v5 plant and 33.3 % on the
    # active one), so this is a floor on feasibility, not a performance claim.
    assert landings["attempt"] >= 0.25, landings
    assert landings["hold"] == 0.0 and landings["zero"] == 0.0
    # The claim is the RANKING, not a margin: waiting out the clock must never
    # be worth more than attempting. Margins are contract-dependent (measured
    # +20.3 vs -24.5 on the frozen v5 plant, +0.36 vs -4.11 on the active one),
    # so pinning one would only invite recalibration whenever the plant moves.
    assert returns["attempt"] > returns["hold"] > returns["zero"], returns


def test_relation_activation_must_outlast_the_actor_value_warmup(tmp_path):
    """The silent no-op that made the proposed arm identical to the baseline.

    The activation phase constructs a fresh PPOTrainer, so its update counter
    restarts at zero while the actor stays gated on
    `updates > value_warmup_iterations`. At the shipped defaults
    (activation_iterations 2, value warmup 2) the actor is never updated. The
    critic readout trains normally, the actor readout keeps its exact 0.0
    initialization, `relational_delta` therefore returns exact zeros, and every
    guard scale is rejected on residual norm -- leaving ppo_ontology_rgat
    numerically equal to ppo_semantic_flat with nothing in the artifacts naming
    the cause. Measured in results/spatial_fixed_nominal_20261005:
    actor_readout_norm 0.0 against critic_readout_norm 0.1396.
    """
    from ontology_rgat.spatial.training import train_arm
    from ontology_rgat.two_axis.training import PPOHyperparameters

    cfg = SpatialConfig()
    blocked = PPOHyperparameters(iterations=4, value_warmup_iterations=2)
    with pytest.raises(ValueError, match="never update the actor"):
        train_arm("ppo_ontology_rgat", cfg, seed=5, hyper=blocked,
                  output=tmp_path / "blocked", activation_iterations=2)
    # Equal is still a no-op; the actor needs at least one ungated update.
    with pytest.raises(ValueError, match="never update the actor"):
        train_arm("ppo_ontology_rgat", cfg, seed=5, hyper=blocked,
                  output=tmp_path / "equal", activation_iterations=1)
    # The baseline arms never run an activation phase, so they are unaffected.
    for arm in ("ppo_vector_canonical", "ppo_semantic_flat"):
        head = ReferenceHead(arm, "actor", cfg.ontology, 5, packet_dim=len(cfg.packet_fields),
                             action_dim=3, descent_axis=2)
        assert not hasattr(head, "encoder")


def test_spatial_selection_matches_reference_safety_weighting():
    from ontology_rgat.two_axis.training import selection_score

    # One success cannot outweigh one unsafe contact by treating unsafe as
    # a small (-200) inconvenience. Reference unsafe weight is -2500.
    assert selection_score(["SUCCESS", "UNSAFE_CONTACT"], [25, -40]) < selection_score(
        ["TASK_TIMEOUT", "TASK_TIMEOUT"], [-12, -12]
    )


def test_spatial_initialization_uses_reference_output_gain_and_zero_biases():
    agent = SpatialAgent("ppo_semantic_flat", SpatialConfig(), 73)
    for head in (agent.actor, agent.critic):
        layers = [layer for layer in head.raw if isinstance(layer, torch.nn.Linear)]
        assert all(torch.count_nonzero(layer.bias) == 0 for layer in layers)
        # The last layer should not inherit a random +/-0.14 acceleration
        # bias from torch.nn.Linear's default initialization.
        assert layers[-1].weight.std() < 0.025
    assert agent.initialization == "reference-v28-gaussian"


def test_finetune_records_ppo_provenance_without_overwriting_source(tmp_path):
    import hashlib

    cfg = replace(SpatialConfig(), horizon=0.1)
    original = tmp_path / "original.pt"
    agent = SpatialAgent("ppo_vector_canonical", cfg, 74)
    save_agent(original, agent, cfg, eligible=True)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    summary = train_arm(
        agent.mode,
        cfg,
        seed=74,
        hyper=PPOHyperparameters(iterations=1, decisions_per_iteration=2),
        output=tmp_path / "fine",
        activation_iterations=0,
        curriculum_enabled=False,
        initial_checkpoint=original,
    )
    assert summary["initialization_source"]["sha256"] == digest
    assert summary["initialization_source"]["optimizer_reset"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    _, meta = load_agent(tmp_path / "fine" / summary["selected_checkpoint"], cfg)
    assert meta["initialization_source"]["sha256"] == digest
    with pytest.raises(ValueError, match="same-mode"):
        train_arm(
            "ppo_semantic_flat",
            cfg,
            seed=74,
            hyper=PPOHyperparameters(iterations=1, decisions_per_iteration=2),
            output=tmp_path / "invalid",
            initial_checkpoint=original,
        )


def test_value_warmup_changes_only_critic_then_enables_policy_updates():
    cfg = replace(SpatialConfig(),horizon=.2)
    agent=SpatialAgent("ppo_vector_canonical",cfg,75)
    actor=copy.deepcopy(agent.actor.state_dict())
    critic=copy.deepcopy(agent.critic.state_dict())
    trainer=PPOTrainer(agent,PPOHyperparameters(
        iterations=3,decisions_per_iteration=8,value_warmup_iterations=2))
    for iteration in range(3):
        data=collect_rollout(agent,SpatialLandingEnv(cfg),seed=76+iteration,decisions=8)
        result=trainer.update(data,discount_time_constant_s=70)
        assert bool(result['policy_updated']) == (iteration==2)
        if iteration<2:
            assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in actor.items())
    assert any(not torch.equal(v,agent.actor.state_dict()[k]) for k,v in actor.items())
    assert any(not torch.equal(v,agent.critic.state_dict()[k]) for k,v in critic.items())


def test_truncated_nominal_rollout_cannot_select_a_checkpoint(tmp_path):
    summary = train_arm(
        "ppo_vector_canonical",
        SpatialConfig(),
        seed=62,
        hyper=PPOHyperparameters(iterations=1, decisions_per_iteration=2),
        output=tmp_path,
        activation_iterations=0,
    )
    assert summary["selected_checkpoint"] is None
    assert summary["completed_nominal_episodes"] == 0
    _, meta = load_agent(tmp_path / "checkpoint_final.pt", SpatialConfig())
    assert not meta["eligible"]


def test_deployment_contract_rejects_scientific_profile_mismatch_before_socket():
    from ontology_rgat.spatial.environment import IsaacBackend

    with pytest.raises(ValueError, match="deployment contract"):
        IsaacBackend(replace(SpatialConfig(), isaac_profile_sha256="different"))


def test_runtime_hash_ignores_view_only_but_not_camera_or_illumination():
    from ontology_rgat.spatial.runtime_contract import runtime_profile_hash

    config = {
        "isaac": {
            "viewport_follow": {"offset_m": [0, 1, 2]},
            "ambient_light_intensity": 500,
        },
        "vision": {"camera": {"rate_hz": 20}},
    }
    first = runtime_profile_hash(config)
    config["isaac"]["viewport_follow"]["offset_m"] = [0, 2, 3]
    assert runtime_profile_hash(config) == first
    config["isaac"]["ambient_light_intensity"] = 600
    assert runtime_profile_hash(config) != first


def test_zero_context_fast_path_preserves_activation_gradients(monkeypatch):
    cfg=SpatialConfig()
    agent=SpatialAgent("ppo_ontology_rgat",cfg,77)
    obs,_=SpatialLandingEnv(cfg).reset(seed=78)
    _,g=agent.tensors(obs)
    agent.actor.set_adaptation(cfg.ontology,False)
    original=agent.actor.encoder.forward
    def forbidden(_):
        raise AssertionError("frozen zero context should not encode")
    monkeypatch.setattr(agent.actor.encoder,"forward",forbidden)
    with torch.no_grad():
        np.testing.assert_array_equal(agent.actor.relational_delta(g).numpy(),np.zeros((1,3)))
    monkeypatch.setattr(agent.actor.encoder,"forward",original)
    agent.actor.set_adaptation(cfg.ontology,True)
    delta=agent.actor.relational_delta(g)
    delta.sum().backward()
    assert torch.count_nonzero(agent.actor.encoder.readout.weight.grad)>0
