"""Regression contract for the primary two-axis context-RGAT experiment."""
from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from dataclasses import replace
from pathlib import Path

from ontology_rgat.two_axis.comparator import PNGains, PNLandingController
from ontology_rgat.two_axis.config import (_validate_actuation_authority,
                                           _validate_terminal_ordering,
                                           _validate_touchdown_reachability,
                                           load_config)
from ontology_rgat.two_axis.curriculum import (CurriculumScheduler,
                                               EpisodeRole,
                                               checkpoint_is_eligible,
                                               landing_rates,
                                               replay_difficulty,
                                               stage_configs)
from ontology_rgat.two_axis.training import (EpisodeDrivenCollector,
                                             PPOHyperparameters, PPOTrainer,
                                             TEST_SEEDS, TUNING_SEEDS,
                                             VALIDATION_SEEDS, evaluate_policy)
from ontology_rgat.two_axis.diagnostics import (detect_altitude_plateau,
                                                graph_utilisation,
                                                record_episode, summarize)
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
                                             QUERY_NODES, SEMANTIC_NODES,
                                             build_context_graph,
                                             query_reachable,
                                             semantic_flat_features)
from ontology_rgat.two_axis.reward import (compute_reward,
                                           conservative_return_bounds,
                                           landing_readiness,
                                           terminal_ordering_report)
from ontology_rgat.two_axis.reward_audit import run_reward_audit
from ontology_rgat.two_axis.safety import (ContactState, SafetySupervisor,
                                           TerminalReason, classify_contact,
                                           hard_envelope_violation,
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
    assert cfg.dynamics.ax_max_m_s2 == 2.5
    assert cfg.dynamics.az_max_m_s2 == 2.0
    text = (cfg.__class__.__module__)  # ensures executable typed config was loaded
    assert "two_axis" in text


def test_longitudinal_authority_exceeds_every_pad_acceleration_drawn():
    """A pad the vehicle cannot out-accelerate is untrackable by any arm."""
    cfg = load_config()
    assert cfg.dynamics.ax_max_m_s2 > cfg.scenario.a2_range_m_s2[1]
    worst = max(sample_scenario(seed, cfg.scenario, cfg.timing).a2_m_s2
                for seed in range(2000, 2040))
    assert worst < cfg.dynamics.ax_max_m_s2
    with pytest.raises(ValueError, match="physically untrackable"):
        _validate_actuation_authority(replace(
            cfg, dynamics=replace(cfg.dynamics, ax_max_m_s2=1.2)))


def test_terminal_table_cannot_make_aborting_a_shortcut():
    """SUCCESS > TASK_TIMEOUT > SAFE_ABORT > unsafe, enforced at load time."""
    cfg = load_config()
    table = dict(cfg.reward.terminal_bonus)
    assert table["SUCCESS"] > table["TASK_TIMEOUT"] > table["SAFE_ABORT"]
    assert table["SAFE_ABORT"] > max(table[name]
                                     for name in cfg.reward.unsafe_reasons)
    shortcut = replace(cfg, reward=replace(
        cfg.reward, terminal_bonus=tuple(
            (name, -3.0 if name == "SAFE_ABORT" else value)
            for name, value in cfg.reward.terminal_bonus)))
    with pytest.raises(ValueError, match="terminal reward ordering violated"):
        _validate_terminal_ordering(shortcut)


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
    assert result.terminal == 25.0
    assert -0.01 < result.running <= 0.0
    bounds = conservative_return_bounds(
        cfg.reward, cfg.timing.discount_time_constant_s,
        cfg.timing.mission_duration_limit_s)
    assert bounds["SUCCESS"][0] > bounds["TASK_TIMEOUT"][1]
    # Keeping the pad in view until the deadline must outrank inducing an
    # abort, by more than the running cost of flying to the deadline.
    report = terminal_ordering_report(
        cfg.reward, cfg.timing.discount_time_constant_s,
        cfg.timing.mission_duration_limit_s)
    assert report["TASK_TIMEOUT_over_SAFE_ABORT"] > 0.0
    assert report["SAFE_ABORT_over_UNSAFE"] > 0.0
    assert report["margin"] > 0.0
    assert report["late_abort_still_worse"] > 0.0
    audit = run_reward_audit(cfg)["trajectories"]
    assert all(item["terminal_payment_count"] == 1 for item in audit.values())
    assert (audit["safe efficient landing"]["discounted_return"]
            > audit["safe delayed landing"]["discounted_return"]
            > audit["stable near-pad hovering until deadline"]["discounted_return"])
    assert (audit["stable near-pad hovering until deadline"]["discounted_return"]
            > audit["unnecessary induced loss followed by abort"]["discounted_return"])
    assert (audit["recoverable short loss followed by landing"]["discounted_return"]
            > audit["unnecessary induced loss followed by abort"]["discounted_return"])


def test_goal_cost_keeps_a_horizontal_gradient_at_altitude():
    """One shared saturation hid longitudinal error behind the altitude term."""
    cfg = load_config()

    def goal(ex, h):
        return compute_reward(
            ex_true_m=ex, h_true_m=h, measured_bearing_rad=0.0,
            bearing_valid=True, normalized_policy_action=np.zeros(2),
            fov_rad=cfg.camera.fov_rad, dt_s=.1, terminal_reason=None,
            config=cfg.reward).goal_cost

    high = goal(2.0, 8.0) - goal(1.0, 8.0)
    low = goal(2.0, 1.0) - goal(1.0, 1.0)
    assert high > 0.0 and low > 0.0
    # The same metre of drift must still be felt from 8 m up.
    assert high > 0.5 * low
    assert goal(4.0, 1.0) > goal(1.0, 4.0)  # horizontal error is prioritised


def test_readiness_is_paid_as_a_difference_so_hovering_earns_nothing():
    cfg = load_config()
    ready = landing_readiness(ex_true_m=0.02, relative_speed_m_s=0.02,
                              h_true_m=0.1, pitch_rad=0.0, safety=cfg.safety)
    assert 0.9 < ready <= 1.0
    far = landing_readiness(ex_true_m=2.0, relative_speed_m_s=0.02,
                            h_true_m=0.1, pitch_rad=0.0, safety=cfg.safety)
    assert far < 0.1 * ready  # misalignment alone collapses readiness
    # The signal must carry a gradient over the whole approach, not only the
    # last metre: a clipped version was exactly zero above 1 m, which left
    # the approach unshaped and PPO hovering to the deadline.
    profile = [landing_readiness(ex_true_m=0.0, relative_speed_m_s=0.0,
                                 h_true_m=h, pitch_rad=0.0, safety=cfg.safety)
               for h in (8.0, 6.0, 4.0, 2.0, 1.0, 0.4, 0.1)]
    assert all(b > a for a, b in zip(profile, profile[1:]))
    assert profile[0] > 0.0

    def running(current, previous):
        return compute_reward(
            ex_true_m=0.02, h_true_m=0.1, measured_bearing_rad=0.0,
            bearing_valid=True, normalized_policy_action=np.zeros(2),
            fov_rad=cfg.camera.fov_rad, dt_s=.1, terminal_reason=None,
            config=cfg.reward, readiness=current,
            previous_readiness=previous).readiness_delta_reward

    assert running(ready, ready) == 0.0           # holding pays nothing
    assert running(ready, 0.0) > 0.0              # becoming ready pays
    assert running(0.0, ready) == -running(ready, 0.0)  # leaving gives it back


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


def _hover(height, *, vz=0.0, ex_est=0.0, pad_v=0.0, vx=0.0, time_s=5.0):
    cfg = load_config()
    state = PlanarState(0.0, height, vx, vz, 0.0, 0.0,
                        cfg.dynamics.mass_kg * cfg.dynamics.gravity_m_s2, time_s)
    track = TrackEstimate(time_s, True, ex_est, pad_v, 0.0, 0.02, 0.05, 0.1,
                          0.0, True)
    return cfg, state, track


def test_an_authorized_touchdown_is_reachable_at_all():
    """The abort-hold margin alone made SUCCESS structurally impossible.

    ``stopping_margin`` equals the height when the vehicle is not descending,
    so every state below ``minimum_abort_hold_height_m`` was inhibited and
    every contact -- however gentle and well aligned -- scored
    UNAUTHORIZED_CONTACT. The terminal-descent corridor replaces that margin
    with a stricter touchdown test instead of removing it.
    """
    cfg, state, track = _hover(0.2, vz=-0.1)
    supervisor = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety,
                                  cfg.camera)
    decision = supervisor.apply(np.array([0.0, -0.3]), state, track)
    assert decision.terminal_descent
    assert not decision.landing_inhibited
    assert "terminal_descent_corridor" in decision.reasons
    # The corridor is not a blanket exemption: a misaligned vehicle at the same
    # height is still inhibited and still has its descent braked.
    _, drifted, far = _hover(0.2, vz=-0.1, ex_est=0.9)
    blocked = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety,
                               cfg.camera).apply(
        np.array([0.0, -0.3]), drifted, far)
    assert not blocked.terminal_descent
    assert blocked.landing_inhibited
    assert blocked.applied_m_s2[1] > -0.3


def test_touchdown_gate_tightens_with_the_observable_camera_footprint():
    """A fixed 0.35 m gate is wider than the frame in the last few centimetres."""
    cfg = load_config()
    supervisor = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety,
                                  cfg.camera)
    wide = supervisor.touchdown_gate_width_m(3.0)
    mid = supervisor.touchdown_gate_width_m(0.5)
    tight = supervisor.touchdown_gate_width_m(0.1)
    assert wide == cfg.safety.touchdown_horizontal_error_m
    assert tight < mid <= wide
    assert tight == pytest.approx(0.1 * math.tan(cfg.camera.fov_rad / 2.0))


def test_flare_commit_is_bounded_and_cannot_outlive_the_abort_threshold():
    cfg = load_config()
    supervisor = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety,
                                  cfg.camera)
    _, state, track = _hover(0.2, vz=-0.2, time_s=5.0)
    assert supervisor.apply(np.zeros(2), state, track).terminal_descent
    # The gate is unsatisfiable this low, but the commit carries the descent.
    _, lower, stale = _hover(0.05, vz=-0.2, ex_est=0.04, time_s=5.3)
    stale = replace(stale, time_since_detection_s=0.3)
    assert supervisor.apply(np.zeros(2), lower, stale).terminal_descent
    # ... only for a bounded window, which is shorter than the abort threshold.
    _, late, old = _hover(0.05, vz=-0.2, ex_est=0.04,
                          time_s=5.0 + cfg.safety.terminal_descent_commit_s + 0.2)
    assert not supervisor.apply(np.zeros(2), late, old).terminal_descent
    assert (cfg.safety.terminal_descent_commit_s
            < cfg.estimator.prolonged_loss_s)


def test_prolonged_loss_starts_a_bounded_recovery_rather_than_a_terminal_abort():
    cfg = load_config()
    supervisor = SafetySupervisor(cfg.dynamics, cfg.estimator, cfg.safety,
                                  cfg.camera)
    estimator = CausalPadEstimator(cfg.estimator)
    estimator.update(_measurement(0.0), own_x_m=0.0)
    lost = estimator.update(_measurement(3.1, detected=False), own_x_m=0.0)
    state = PlanarState(0, 3, 0, -.5, 0, 0,
                        cfg.dynamics.mass_kg * cfg.dynamics.gravity_m_s2, 3.1)
    latched = supervisor.apply(np.array([0.0, -1.0]), state, lost)
    assert latched.abort_requested and latched.applied_m_s2[1] > 0.0
    assert not latched.abort_expired  # the recovery window is still open
    # Reacquiring the pad inside the window clears the abort and hands control
    # back to the learned policy. One detection after a long coast is not
    # enough: the velocity covariance has to re-converge first, so the
    # supervisor keeps the recovery manoeuvre until the track is trustworthy.
    recovered = None
    for step in range(1, 12):
        moment = 3.1 + 0.1 * step
        regained = estimator.update(_measurement(moment), own_x_m=0.0)
        recovered = supervisor.apply(np.array([0.0, -1.0]),
                                     replace(state, time_s=moment), regained)
        if recovered.abort_recovered:
            break
    assert recovered is not None and recovered.abort_recovered
    assert not recovered.abort_requested
    assert supervisor.abort_recovery_count == 1
    # A window that expires without reacquisition does end the episode.
    supervisor.reset()
    supervisor.apply(np.zeros(2), state, lost)
    expired = estimator.update(_measurement(12.0, detected=False), own_x_m=0.0)
    beyond = supervisor.apply(
        np.zeros(2),
        replace(state, time_s=3.1 + cfg.safety.maximum_backup_duration_s + 0.5),
        expired)
    assert beyond.abort_expired and beyond.abort_requested


def test_contact_uses_the_landing_gear_plane_with_a_ground_fallback():
    cfg = load_config()
    gear = cfg.safety.touchdown_height_m
    assert 0.0 < gear < cfg.safety.minimum_abort_hold_height_m
    before = PlanarState(0, gear + .02, 0, -.2, 0, 0, 10, 0)
    after = PlanarState(0, gear - .02, 0, -.2, 0, 0, 10, .01)
    on_gear = interpolate_contact(before, after, (0, 1), (.002, 1),
                                  contact_height_m=gear)
    assert on_gear is not None
    assert on_gear.time_s == pytest.approx(0.005)
    # Already below the gear plane: the 0 m crossing is the numerical fallback.
    low_before = PlanarState(0, .01, 0, -2, 0, 0, 10, 0)
    low_after = PlanarState(0, -.01, 0, -2, 0, 0, 10, .01)
    assert interpolate_contact(low_before, low_after, (0, 1), (.02, 1),
                               contact_height_m=gear) is not None
    assert interpolate_contact(before, PlanarState(0, gear + .01, 0, -.2, 0, 0,
                                                   10, .01),
                               (0, 1), (.002, 1), contact_height_m=gear) is None


def test_hard_envelope_is_measured_against_the_pad_not_the_world_origin():
    """The pad finishes hundreds of metres downrange on a long mission."""
    cfg = load_config()
    far = PlanarState(300.0, 1.0, 2.0, -.2, 0, 0, 14.7, 40.0)
    assert not hard_envelope_violation(far, cfg.safety, pad_x_m=300.2)
    assert hard_envelope_violation(far, cfg.safety, pad_x_m=0.0)
    scenario = sample_scenario(2000, cfg.scenario, cfg.timing)
    travelled = scenario.state_at(scenario.duration_s)[0]
    assert travelled > cfg.safety.max_horizontal_range_m



def curriculum_enabled_config():
    """The shipped experiment runs with ``curriculum.enabled: false``.

    The historical disabled unsafe ramp is not safe to re-enable verbatim;
    use an ordered (-45) ramp for machinery tests. Tests that
    exercise promotion, replay roles or the tolerance ramp build their own enabled
    config; tests of what the experiment actually runs use ``load_config()``.
    """
    cfg = load_config()
    return replace(cfg, curriculum=replace(cfg.curriculum, enabled=True,
                                          start_unsafe_contact_penalty=-45.0))


def test_curriculum_difficulty_one_reproduces_the_nominal_contract_exactly():
    """Validation and test must never run an easier task than the contract."""
    cfg = curriculum_enabled_config()
    nominal = stage_configs(cfg, 1.0)
    assert nominal.scenario == cfg.scenario
    assert nominal.safety == cfg.safety
    assert nominal.reward == cfg.reward
    assert checkpoint_is_eligible(nominal.difficulty)
    assert not checkpoint_is_eligible(0.9)
    easy = stage_configs(cfg, 0.0)
    # Easiness is a slower pad, never a lower approach: the camera footprint is
    # 0.7*h, so a lower start shrinks the frame and makes tracking harder.
    assert easy.scenario.initial_height_range_m == cfg.scenario.initial_height_range_m
    assert easy.scenario.v1_range_m_s[1] < cfg.scenario.v1_range_m_s[1]
    assert easy.scenario.a2_range_m_s2[1] < cfg.scenario.a2_range_m_s2[1]
    # Even the easiest episode meets the pad acceleration event.
    assert easy.scenario.T1_range_s[1] <= 0.3
    assert (easy.safety.touchdown_vertical_speed_m_s
            > cfg.safety.touchdown_vertical_speed_m_s)
    assert (dict(easy.reward.terminal_bonus)["UNSAFE_CONTACT"]
            == cfg.curriculum.start_unsafe_contact_penalty)
    # The ramp is monotone, so a shortcut never becomes cheaper with progress.
    penalties = [dict(stage_configs(cfg, d).reward.terminal_bonus)["UNSAFE_CONTACT"]
                 for d in np.linspace(0.0, 1.0, 11)]
    assert all(b <= a + 1e-12 for a, b in zip(penalties, penalties[1:]))
    assert penalties[-1] == dict(cfg.reward.terminal_bonus)["UNSAFE_CONTACT"]



def test_shipped_experiment_trains_and_scores_on_the_same_objective():
    """The gate-release contract, pinned.

    With the curriculum on, promotion needed a 0.6 landing rate over 20 active
    episodes. The arms reached 0.34, so difficulty stayed near 0 for all 800
    iterations while UNSAFE was ramped to -32 against a -30 TASK_TIMEOUT, and
    scoring then happened at difficulty 1.0 where UNSAFE is -50: the arms were
    trained under one objective and scored under another. A second gate,
    ``checkpoint_is_eligible``, requires difficulty >= 1.0, so nothing was ever
    written to disk either -- a 33 % nominal policy measured at it=600 was lost.
    Disabling the schedule closes both gaps at once.
    """
    cfg = load_config()
    assert not cfg.curriculum.enabled
    scheduler = CurriculumScheduler(cfg.curriculum, difficulty=0.0)
    # Training difficulty is pinned at nominal, which is what evaluation uses.
    assert scheduler.difficulty == 1.0 and scheduler.at_nominal
    assert checkpoint_is_eligible(scheduler.difficulty)
    # No replay interleave: every episode is the scored task.
    assert all(scheduler.role_for(i) is EpisodeRole.ACTIVE for i in range(24))
    # Promotion is inert and can never move difficulty off nominal.
    for _ in range(4 * cfg.curriculum.promotion_window_episodes):
        assert not scheduler.record(role=EpisodeRole.ACTIVE, landed=True)
    assert scheduler.difficulty == 1.0 and scheduler.promotions == 0
    # Every requested difficulty collapses to the nominal contract, so no
    # training episode can be drawn from a softer distribution or a softer
    # terminal table than the one the arms are scored against.
    for d in (0.0, 0.25, 0.5, 1.0):
        stage = stage_configs(cfg, d)
        assert stage.difficulty == 1.0
        assert stage.scenario == cfg.scenario
        assert stage.safety == cfg.safety
        assert stage.reward == cfg.reward
        assert replay_difficulty(scheduler.difficulty, EpisodeRole.ACTIVE) == 1.0


def test_promotion_ignores_replays_and_reports_the_rates_separately():
    cfg = curriculum_enabled_config()
    scheduler = CurriculumScheduler(cfg.curriculum, difficulty=0.5)
    roles = [scheduler.role_for(i) for i in range(12)]
    assert EpisodeRole.EASY_REPLAY in roles and EpisodeRole.BRIDGE_REPLAY in roles
    assert roles.count(EpisodeRole.ACTIVE) >= 8
    # A long run of successful easy replays must not promote the curriculum.
    for _ in range(50):
        scheduler.record(role=EpisodeRole.EASY_REPLAY, landed=True)
    assert scheduler.difficulty == 0.5 and scheduler.promotions == 0
    for _ in range(cfg.curriculum.promotion_window_episodes):
        scheduler.record(role=EpisodeRole.ACTIVE, landed=True)
    assert scheduler.difficulty > 0.5 and scheduler.promotions == 1
    rates = landing_rates([
        (EpisodeRole.EASY_REPLAY, 0.0, "SUCCESS"),
        (EpisodeRole.EASY_REPLAY, 0.0, "SUCCESS"),
        (EpisodeRole.ACTIVE, 1.0, "TASK_TIMEOUT"),
        (EpisodeRole.ACTIVE, 1.0, "SAFE_ABORT")])
    # The overall rate says 50%; nominal-difficulty landing says 0%.
    assert rates["overall"] == pytest.approx(0.5)
    assert rates["active_curriculum"] == pytest.approx(0.0)
    assert rates["nominal"] == pytest.approx(0.0)
    assert rates["safe_abort"] == pytest.approx(0.25)


def test_every_arm_receives_the_identical_episode_from_one_seed():
    """The only intended difference is the state representation."""
    cfg = load_config()
    traces = {}
    for mode in POLICY_MODES:
        env = TwoAxisLandingEnv(cfg, perturbations=True)
        observation, info = env.reset(seed=31, difficulty=0.4)
        rows = []
        for step in range(12):
            action = np.array([0.2 * math.sin(step), -0.3])
            observation, reward, terminated, _, step_info = env.step(action)
            rows.append((reward, tuple(step_info["applied_acceleration_m_s2"]),
                         step_info["landing_inhibited"], step_info["status"]))
            if terminated:
                break
        traces[mode] = (info["scenario"], info["dropout_schedule"],
                        info["observation_registry_hash"],
                        info["curriculum_difficulty"], rows)
    reference = traces[POLICY_MODES[0]]
    for mode in POLICY_MODES[1:]:
        assert traces[mode] == reference, f"{mode} saw a different episode"


def test_pn_comparator_is_feasible_and_is_never_a_teacher():
    """The non-learned arm verifies the envelope; it supervises nothing."""
    cfg = load_config()
    env = TwoAxisLandingEnv(cfg, perturbations=True)
    controller = PNLandingController(cfg)
    records = []
    for seed in range(2000, 2008):
        controller.reset()
        records.append(record_episode(
            env, lambda e: controller.act(e.state, e.track), seed=seed,
            difficulty=1.0))
    report = summarize(records)
    # The task must be physically solvable at nominal difficulty by a
    # non-learned controller, or a learned arm's failure says nothing.
    assert report["landing_rate"] > 0.0
    assert all(record.difficulty == 1.0 for record in records)
    limits = np.array([cfg.dynamics.ax_max_m_s2, cfg.dynamics.az_max_m_s2])
    for record in records:
        requested = np.column_stack([record.requested_ax, record.requested_az])
        assert np.all(np.abs(requested) <= limits + 1e-9)
    source = (Path(__file__).resolve().parents[1]
              / "python/ontology_rgat/two_axis/comparator.py").read_text("utf-8")
    assert "behaviour cloning" in source or "behavior cloning" in source
    learning = (Path(__file__).resolve().parents[1]
                / "python/ontology_rgat/two_axis/learning.py").read_text("utf-8")
    for forbidden in ("PNLandingController", "comparator", "behavior_cloning"):
        assert forbidden not in learning


def test_plateau_detector_attributes_the_chain_upstream():
    cfg = load_config()
    env = TwoAxisLandingEnv(cfg, perturbations=True)
    # A policy that only descends drifts off a moving pad and is braked.
    descend_only = lambda _env: np.array([0.0, -1.0])
    record = record_episode(env, descend_only, seed=2003, difficulty=1.0)
    report = detect_altitude_plateau(record)
    assert set(report["chain"]) == {
        "horizontal_error_growing", "fov_margin_shrinking",
        "confidence_falling", "landing_inhibited_active", "descent_overridden"}
    if report["plateau"]:
        assert report["upstream_cause"] in {
            "horizontal_tracking", "visibility_geometry",
            "track_trust_or_stopping_margin", "unattributed"}
    assert summarize([record])["episodes"] == 1


def test_structural_graph_audit_finds_no_dead_semantic_node():
    """Task-5 audit: every declared node must carry information in rollouts."""
    cfg = load_config()
    env = TwoAxisLandingEnv(cfg, perturbations=True)
    controller = PNLandingController(cfg)
    features = []
    for seed in range(1000, 1004):
        controller.reset()
        observation, _ = env.reset(seed=seed, difficulty=1.0)
        while True:
            features.append(observation.graph.X.copy())
            observation, _r, terminated, _t, _i = env.step(
                controller.act(env.state, env.track))
            if terminated:
                break
    audit = graph_utilisation(features)
    assert audit["samples"] > 100
    assert audit["dead_nodes"] == []
    for name in SEMANTIC_NODES:
        assert audit["nodes"][name]["active_channels"] >= 4
        assert audit["nodes"][name]["std"] > 0.0
    # The two readout queries are identical by design: they differ only in the
    # identity one-hot and in which role's encoder reads them.
    assert audit["duplicate_node_rows"] == ["PolicyNode==ValueNode"]
    for query in QUERY_NODES:
        assert audit["nodes"][query]["active_channels"] == 1


def test_curriculum_start_keeps_the_pad_inside_the_frame_at_every_difficulty():
    """An easy episode that begins blind is harder than the nominal task.

    A fixed 2 m start offset is 59 degrees from nadir at a 1.2 m curriculum
    start, against a 35 degree half-FOV, so the vehicle began with no pad in
    frame, the tracker never initialised and the supervisor latched an abort.
    The nominal 4-8 m case is unchanged, which this also pins.
    """
    cfg = load_config()
    env = TwoAxisLandingEnv(cfg, perturbations=True)
    for difficulty in (0.0, 0.25, 0.5, 1.0):
        for seed in range(5000, 5012):
            env.reset(seed=seed, difficulty=difficulty)
            assert env.measurement is not None and env.scenario is not None
            assert env.measurement.geometric_visible, (
                f"pad outside the frame at t=0 for difficulty {difficulty}")
    # The nominal start offset is still exactly 2 m: 0.5 * h >= 2 for h >= 4.
    for seed in range(5000, 5012):
        env.reset(seed=seed, difficulty=1.0)
        pad_x = env.scenario.state_at(0.0)[0]
        assert pad_x - env.state.x_m == pytest.approx(2.0)


def test_ppo_trainer_keeps_optimiser_state_and_takes_many_clipped_steps():
    """The bounded helper rebuilt Adam each call and took one full-batch step.

    Rebuilding the optimiser discards the moment estimates every iteration, so
    the effective update is scaled SGD; a single full-batch step leaves the
    sampling ratio at exactly 1 on the only step taken, so the PPO clip never
    engages. Both are fixed in the trainer used for real runs.
    """
    cfg = load_config()
    agent = TwoAxisPPOAgent("ppo_vector_canonical", seed=5)
    hyper = PPOHyperparameters(epochs=2, minibatch_size=64)
    trainer = PPOTrainer(agent, hyper)
    env = TwoAxisLandingEnv(cfg, perturbations=False)
    scheduler = CurriculumScheduler(cfg.curriculum, difficulty=0.0)
    collector = EpisodeDrivenCollector(env, scheduler, base_seed=77_000)
    rollout = collector.collect(agent, 256)
    actor_optimizer = trainer.actor_optimizer
    metrics = trainer.update(
        rollout, discount_time_constant_s=cfg.timing.discount_time_constant_s)
    assert metrics["minibatches"] >= 4          # many steps, not one
    assert all(np.isfinite(value) for value in metrics.values())
    state = trainer.actor_optimizer.state_dict()["state"]
    assert state, "optimiser moments must exist after the first update"
    trainer.update(
        rollout, discount_time_constant_s=cfg.timing.discount_time_constant_s)
    # The same optimiser object carries its moments into the next iteration.
    assert trainer.actor_optimizer is actor_optimizer
    assert trainer.actor_optimizer.state_dict()["state"]


def test_collector_opens_each_episode_at_the_scheduled_difficulty():
    cfg = curriculum_enabled_config()
    agent = TwoAxisPPOAgent("ppo_vector_canonical", seed=6)
    env = TwoAxisLandingEnv(cfg, perturbations=False)
    scheduler = CurriculumScheduler(cfg.curriculum, difficulty=0.5)
    collector = EpisodeDrivenCollector(env, scheduler, base_seed=78_000)
    collector.collect(agent, 1200)
    assert collector.episode_index >= 2
    roles = {role for role, _d, _r in collector.completed}
    assert roles  # at least one episode finished and was reported back
    for role, difficulty, reason in collector.completed:
        assert difficulty == pytest.approx(replay_difficulty(0.5, role))
        assert reason in {item.value for item in TerminalReason}


def test_evaluation_always_runs_the_nominal_distribution():
    cfg = load_config()
    agent = TwoAxisPPOAgent("ppo_semantic_flat", seed=7)
    report = evaluate_policy(agent, cfg, seeds=[2000, 2001, 2002],
                             difficulty=1.0)
    assert report["episodes"] == 3.0
    for key in ("landing_rate", "safe_abort_rate", "task_timeout_rate",
                "unsafe_rate", "fov_capture_rate"):
        assert 0.0 <= report[key] <= 1.0
    # Held-out blocks are disjoint, so nothing tuned on is reported on.
    assert not set(TUNING_SEEDS) & set(VALIDATION_SEEDS)
    assert not set(TUNING_SEEDS) & set(TEST_SEEDS)
    assert not set(VALIDATION_SEEDS) & set(TEST_SEEDS)


def test_discounted_break_even_is_conditional_not_a_landing_guarantee():
    """Ordered terminal payments do not make a zero-success attempt profitable."""
    cfg = curriculum_enabled_config()
    tau = cfg.timing.discount_time_constant_s
    deadline = cfg.timing.mission_duration_limit_s
    table = dict(cfg.reward.terminal_bonus)
    early = math.exp(-8.0 / tau)
    late = math.exp(-deadline / tau)
    # The horizon must outlast the mission, or WHEN an outcome lands dominates
    # WHAT it was.
    assert tau >= 3.0 * deadline

    def break_even(unsafe_value):
        waiting = table["TASK_TIMEOUT"] * late
        failed = unsafe_value * early
        return (waiting - failed) / (table["SUCCESS"] * early - failed)

    rungs = [stage_configs(cfg, d) for d in (0.0, 0.5, 1.0)]
    unsafe = [dict(r.reward.terminal_bonus)["UNSAFE_CONTACT"] for r in rungs]
    # Ordering holds at every rung, not just at nominal.
    for value in unsafe:
        assert value < table["SAFE_ABORT"] < table["TASK_TIMEOUT"]
    points = [break_even(value) for value in unsafe]
    assert all(b < a for a, b in zip(unsafe, unsafe[1:]))      # ramp tightens
    assert all(a < b for a, b in zip(points, points[1:]))      # cost rises with it
    # The easiest rung must be cheap enough for a policy starting at zero to
    # find, and the hardest must still pay off for the policy the curriculum
    # only promotes once it lands at ``promotion_landing_rate``.
    assert 0.0 < points[0] < points[-1]
    assert points[-1] < cfg.curriculum.promotion_landing_rate
