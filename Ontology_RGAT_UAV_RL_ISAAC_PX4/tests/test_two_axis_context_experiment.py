"""Regression contract for the primary two-axis context-RGAT experiment."""
from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from ontology_rgat.two_axis.config import load_config
from ontology_rgat.two_axis.contracts import (experiment_signature,
                                              load_v2_registry,
                                              make_causal_packet)
from ontology_rgat.two_axis.dynamics import (PlanarState,
                                             acceleration_to_thrust_pitch,
                                             reachable_pitch_envelope,
                                             realized_acceleration, step_planar)
from ontology_rgat.two_axis.environment import TwoAxisLandingEnv
from ontology_rgat.two_axis.estimation import (CausalPadEstimator,
                                               TrackEstimate,
                                               average_acceleration)
from ontology_rgat.two_axis.learning import (collect_rollout,
                                             ppo_minibatch_update)
from ontology_rgat.two_axis.models import (POLICY_MODES, TwoAxisPPOAgent,
                                           capacity_matched_mlp_hidden)
from ontology_rgat.two_axis.ontology import (FEATURE_CHANNELS, GRAPH_EDGES,
                                             GRAPH_SCHEMA_HASH, NODE_NAMES,
                                             QUERY_NODES, build_context_graph,
                                             query_reachable,
                                             semantic_flat_features)
from ontology_rgat.two_axis.reward import (compute_reward,
                                           conservative_return_bounds)
from ontology_rgat.two_axis.reward_audit import run_reward_audit
from ontology_rgat.two_axis.safety import (ContactState, SafetySupervisor,
                                           TerminalReason, classify_contact,
                                           interpolate_contact)
from ontology_rgat.two_axis.scenario import (sample_scenario,
                                             scenario_from_distances,
                                             split_seed)
from ontology_rgat.two_axis.sensing import (DropoutSchedule, PadMeasurement,
                                            project_pad)


def _measurement(t, *, detected=True, ex=1.0):
    return PadMeasurement(t, detected, 0.0, detected, ex if detected else 0.0,
                          detected, 1.0 if detected else 0.0, detected, None)


def test_primary_configuration_is_strictly_two_axis_and_legacy_off():
    cfg = load_config()
    assert cfg.pipelines == POLICY_MODES
    assert cfg.dynamics.ax_max_m_s2 == 1.2
    assert cfg.dynamics.az_max_m_s2 == 2.0
    text = (cfg.__class__.__module__)  # ensures executable typed config was loaded
    assert "two_axis" in text


def test_three_segment_trajectory_is_position_velocity_continuous():
    cfg = load_config()
    scenario = sample_scenario(17, cfg.scenario, cfg.timing)
    eps = 1e-8
    for boundary in (scenario.T1_s, scenario.T1_s + scenario.T2_s):
        left = scenario.state_at(boundary - eps)
        exact = scenario.state_at(boundary)
        right = scenario.state_at(boundary + eps)
        assert left[0] == pytest.approx(exact[0], abs=1e-6)
        assert right[0] == pytest.approx(exact[0], abs=1e-6)
        assert left[1] == pytest.approx(exact[1], abs=1e-6)
        assert right[1] == pytest.approx(exact[1], abs=1e-6)
    x_end = scenario.state_at(scenario.duration_s)[0]
    assert x_end - scenario.x0_m == pytest.approx(sum(scenario.lengths_m))
    assert scenario.v3_m_s == pytest.approx(
        scenario.v1_m_s + scenario.a2_m_s2 * scenario.T2_s)


def test_randomized_sampler_limits_and_seed_streams_are_reproducible():
    cfg = load_config()
    first = sample_scenario(81, cfg.scenario, cfg.timing)
    second = sample_scenario(81, cfg.scenario, cfg.timing)
    assert first == second
    assert first.duration_s <= cfg.timing.mission_duration_limit_s
    assert (first.v3_m_s + cfg.scenario.final_speed_margin_m_s
            <= cfg.scenario.max_final_pad_speed_m_s)
    streams = split_seed(81)
    assert len({streams.scenario_seed, streams.sensor_seed,
                streams.policy_seed}) == 3


def test_distance_mode_uses_stable_kinematics_including_zero_acceleration():
    scenario = scenario_from_distances(
        seed=2, v1_m_s=2.0, a2_m_s2=0.0,
        L1_m=4.0, L2_m=6.0, L3_m=8.0, initial_height_m=5.0)
    assert scenario.T1_s == pytest.approx(2.0)
    assert scenario.T2_s == pytest.approx(3.0)
    assert scenario.T3_s == pytest.approx(4.0)
    assert scenario.v3_m_s == pytest.approx(2.0)


def test_hover_pitch_sign_lag_and_reachable_envelope():
    cfg = load_config()
    dyn = cfg.dynamics
    hover = PlanarState(0, 5, 0, 0, 0, 0,
                        dyn.mass_kg * dyn.gravity_m_s2)
    assert realized_acceleration(hover, dyn) == pytest.approx([0.0, 0.0])
    forward = acceleration_to_thrust_pitch(np.array([1.0, 0.0]), dyn)
    assert forward.theta_rad > 0.0
    stepped = step_planar(hover, np.array([1.0, 0.0]), .01, dyn)
    assert 0.0 < stepped.theta_rad < forward.theta_rad
    assert stepped.x_m > 0.0 and stepped.y_m == stepped.vy_m_s == 0.0
    lo, hi = reachable_pitch_envelope(dyn)
    assert lo < 0 < hi
    assert max(abs(lo), abs(hi)) < dyn.pitch_limit_rad


def test_camera_geometry_matches_nadir_and_rotates_with_actual_pitch():
    camera = load_config().camera
    h = 5.0
    edge = h * math.tan(camera.fov_rad / 2.0)
    assert project_pad(edge, h, 0.0, camera).geometric_visible
    assert not project_pad(edge * 1.01, h, 0.0, camera).geometric_visible
    nadir = project_pad(2.0, h, 0.0, camera)
    pitched = project_pad(2.0, h, math.radians(5), camera)
    assert pitched.bearing_rad > nadir.bearing_rad
    assert not project_pad(0.0, -0.1, 0.0, camera).geometric_visible


def test_estimator_uses_actual_measurement_gaps_and_idempotent_timestamps():
    assert average_acceleration(2.0, 4.0, 0.0, 2.0) == pytest.approx(1.0)
    estimator = CausalPadEstimator(load_config().estimator)
    first = estimator.update(_measurement(0.0), own_x_m=0.0)
    duplicate = estimator.update(_measurement(0.0, ex=100.0), own_x_m=0.0)
    assert duplicate == first
    missing = estimator.update(_measurement(2.0, detected=False), own_x_m=0.0)
    assert missing.position_std_m > first.position_std_m
    assert missing.time_since_detection_s == pytest.approx(2.0)


def test_occluded_packet_has_no_current_pad_truth_input():
    cfg = load_config()
    registry = load_v2_registry()
    state = PlanarState(1, 5, .2, 0, 0, 0, 14.715, 2.0)
    track = TrackEstimate(2, True, 3, .5, .1, .2, .3, .4, 1.0, False)
    missing = _measurement(2.0, detected=False)
    packet = make_causal_packet(
        state=state, track=track, measurement=missing,
        previous_action=np.zeros(2), landing_inhibited=True,
        abort_requested=False, mission_deadline_s=30.0,
        fov_rad=cfg.camera.fov_rad, registry=registry)
    assert packet.values.shape == (registry.dimension,)
    assert all("current_hidden_pad_state" not in str(field["source"])
               for field in registry.fields)
    with pytest.raises(TypeError):
        make_causal_packet(  # no privileged keyword is accepted
            state=state, track=track, measurement=missing,
            previous_action=np.zeros(2), landing_inhibited=True,
            abort_requested=False, mission_deadline_s=30.0,
            fov_rad=cfg.camera.fov_rad, true_pad_x=999.0)


def test_long_loss_grows_uncertainty_inhibits_landing_and_latches_abort():
    cfg = load_config()
    estimator = CausalPadEstimator(cfg.estimator)
    estimator.update(_measurement(0.0), own_x_m=0.0)
    track = estimator.update(_measurement(3.1, detected=False), own_x_m=0.0)
    state = PlanarState(0, 3, 0, -.5, 0, 0,
                        cfg.dynamics.mass_kg * cfg.dynamics.gravity_m_s2, 3.1)
    supervisor = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety)
    decision = supervisor.apply(np.array([0.0, -1.0]), state, track)
    assert decision.landing_inhibited
    assert decision.abort_requested
    assert decision.applied_m_s2[1] > 0.0


def test_contact_classification_uses_physics_and_authorization_only():
    safety = load_config().safety
    safe = ContactState(1.0, .1, .1, -.1, 0.0, 0.0)
    assert classify_contact(safe, authorized=True, config=safety) is TerminalReason.SUCCESS
    assert (classify_contact(safe, authorized=False, config=safety)
            is TerminalReason.UNAUTHORIZED_CONTACT)
    fast = ContactState(1.0, .1, 1.0, -1.0, 0.0, 0.0)
    assert (classify_contact(fast, authorized=True, config=safety)
            is TerminalReason.UNSAFE_CONTACT)
    before = PlanarState(0, .01, 0, -2, 0, 0, 10, 0)
    after = PlanarState(0, -.01, 0, -2, 0, 0, 10, .01)
    impact = interpolate_contact(before, after, (0, 1), (.01, 1))
    assert impact is not None and impact.vertical_speed_m_s == pytest.approx(-2.0)


def test_reward_is_bounded_time_scaled_and_terminal_separation_holds():
    cfg = load_config()
    result = compute_reward(
        ex_true_m=3.0, h_true_m=4.0, measured_bearing_rad=0.0,
        bearing_valid=True, normalized_policy_action=np.array([1.0, -1.0]),
        fov_rad=cfg.camera.fov_rad, dt_s=.1,
        terminal_reason=TerminalReason.SUCCESS, config=cfg.reward)
    assert result.terminal == 10.0
    assert -0.01 < result.running <= 0.0
    bounds = conservative_return_bounds(
        cfg.reward, cfg.timing.discount_time_constant_s,
        cfg.timing.mission_duration_limit_s)
    assert bounds["SUCCESS"][0] > bounds["SAFE_ABORT"][1]
    assert bounds["SAFE_ABORT"][0] > bounds["TASK_TIMEOUT"][1]
    assert bounds["TASK_TIMEOUT"][0] > bounds["UNSAFE_CONTACT"][1]
    audit = run_reward_audit(cfg)["trajectories"]
    assert all(item["terminal_payment_count"] == 1 for item in audit.values())
    assert (audit["safe efficient landing"]["discounted_return"]
            > audit["safe delayed landing"]["discounted_return"]
            > audit["stable near-pad hovering until deadline"]["discounted_return"])
    assert (audit["recoverable short loss followed by landing"]["discounted_return"]
            > audit["unnecessary induced loss followed by abort"]["discounted_return"])


def test_graph_and_semantic_flat_share_features_registry_and_query_paths():
    env = TwoAxisLandingEnv(perturbations=False)
    observation, _ = env.reset(seed=9)
    graph = observation.graph
    assert graph.packet_registry_hash == observation.packet.registry_sha256
    assert semantic_flat_features(graph).shape == (graph.X.size,)
    assert len(GRAPH_EDGES) == 17 + 2 * 9 + 11
    for source in NODE_NAMES[:-2]:
        for query in QUERY_NODES:
            assert query_reachable(source, query)
    # Query nodes carry bias and identity only before message passing.
    base_width = len(FEATURE_CHANNELS)
    for query in QUERY_NODES:
        row = graph.X[NODE_NAMES.index(query)]
        assert np.count_nonzero(row[:base_width]) == 1
        assert row[FEATURE_CHANNELS.index("bias")] == 1.0


@pytest.mark.parametrize("mode", POLICY_MODES)
def test_all_policy_arms_run_finite_rollout_and_one_ppo_minibatch(mode):
    cfg = load_config()
    agent = TwoAxisPPOAgent(mode, seed=4)
    rollout = collect_rollout(
        agent, TwoAxisLandingEnv(cfg, perturbations=False), seed=4, decisions=4)
    assert all(item.raw_command.shape == (2,) for item in rollout)
    assert all(np.isfinite(item.observation.graph.X).all() for item in rollout)
    limits = np.array([cfg.dynamics.ax_max_m_s2, cfg.dynamics.az_max_m_s2])
    assert all(np.all(np.abs(item.requested_acceleration_m_s2) <= limits + 1e-12)
               and np.all(np.abs(item.applied_acceleration_m_s2) <= limits + 1e-12)
               for item in rollout)
    metrics = ppo_minibatch_update(
        agent, rollout,
        discount_time_constant_s=cfg.timing.discount_time_constant_s)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["ratio_min"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["ratio_max"] == pytest.approx(1.0, abs=1e-5)


def test_rgat_actor_gradient_matches_finite_difference_and_capacity_controls_exist():
    env = TwoAxisLandingEnv(perturbations=False)
    observation, _ = env.reset(seed=12)
    agent = TwoAxisPPOAgent("ppo_ontology_rgat", seed=12)
    packets, graphs = agent.tensors(observation)
    parameter = next(param for name, param in agent.actor.named_parameters()
                     if "layer1" in name and param.numel() > 0)
    index = (0,) * parameter.ndim
    agent.actor.zero_grad(set_to_none=True)
    objective = agent.actor(packets, graphs)[0].sum()
    objective.backward()
    analytic = float(parameter.grad[index])
    original = float(parameter.data[index])
    epsilon = 1e-3
    with torch.no_grad():
        parameter.data[index] = original + epsilon
        plus = float(agent.actor(packets, graphs)[0].sum())
        parameter.data[index] = original - epsilon
        minus = float(agent.actor(packets, graphs)[0].sum())
        parameter.data[index] = original
    numeric = (plus - minus) / (2.0 * epsilon)
    assert analytic == pytest.approx(numeric, abs=2e-3, rel=2e-2)
    target = agent.parameter_count()["total"]
    for mode in ("ppo_vector_canonical", "ppo_semantic_flat"):
        hidden, count = capacity_matched_mlp_hidden(mode, target)
        assert hidden > 0
        assert abs(count - target) / target < 0.03


def test_old_three_channel_checkpoint_signature_is_rejected():
    cfg = load_config()
    signature = experiment_signature(cfg, graph_schema_hash=GRAPH_SCHEMA_HASH)
    legacy = asdict(signature)
    legacy["action_contract_hash"] = "legacy-three-channel-action"
    with pytest.raises(ValueError, match="checkpoint signature mismatch"):
        signature.assert_compatible(legacy)
