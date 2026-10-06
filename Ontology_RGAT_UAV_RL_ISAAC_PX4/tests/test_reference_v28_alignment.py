"""Source-equation and operational regression checks, not performance claims."""
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import math
import json
import subprocess

import numpy as np
import pytest
import torch

from ontology_rgat.two_axis.config import load_config, REFERENCE_CONFIG_PATH, _validate
from ontology_rgat.two_axis.contracts import experiment_signature
from ontology_rgat.two_axis.curriculum import CurriculumScheduler, stage_configs
from ontology_rgat.two_axis.environment import TwoAxisLandingEnv
from ontology_rgat.two_axis.models import TwoAxisPPOAgent, POLICY_MODES
from ontology_rgat.two_axis.models_v28 import relational_contribution
from ontology_rgat.two_axis.ontology_v28 import GRAPH_EDGES, GRAPH_SCHEMA_HASH, READOUT_GROUPS
from ontology_rgat.two_axis.reward import compute_reward, landing_readiness
from ontology_rgat.two_axis.safety import TerminalReason
from ontology_rgat.two_axis.sensing import observe_pad

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference():
    return load_config(REFERENCE_CONFIG_PATH)


def test_reference_schema_and_parameter_counts_match_upstream():
    cfg = reference()
    env = TwoAxisLandingEnv(cfg)
    observation, _ = env.reset(seed=42)
    assert observation.graph.X.shape == (9, 12)
    assert len(GRAPH_EDGES) == 26
    assert len(READOUT_GROUPS) == 4
    assert observation.graph.schema_hash == GRAPH_SCHEMA_HASH
    for mode, count in zip(POLICY_MODES, (7445, 15317, 17001)):
        assert TwoAxisPPOAgent(mode, graph_config=cfg.ontology).parameter_count()["total"] == count


def test_flat_and_graph_start_identical_and_attention_can_learn():
    cfg = reference()
    env = TwoAxisLandingEnv(cfg)
    obs, _ = env.reset(seed=42)
    flat = TwoAxisPPOAgent("ppo_semantic_flat", seed=10, graph_config=cfg.ontology)
    graph = TwoAxisPPOAgent("ppo_ontology_rgat", seed=10, graph_config=cfg.ontology)
    packets, graphs = graph.tensors(obs)
    assert torch.equal(flat.actor(packets,graphs)[0], graph.actor(packets,graphs)[0])
    assert torch.equal(flat.critic(packets,graphs), graph.critic(packets,graphs))
    assert not relational_contribution(graph,packets,graphs)["relation_active_on_probe"]
    graph.actor.set_adaptation(cfg.ontology, True)
    optimizer = torch.optim.Adam(graph.actor.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        graph.actor(packets,graphs)[0][:,0].sum().backward()
        optimizer.step()
    assert graph.actor.encoder.attention.grad.abs().sum() > 0
    assert graph.actor.encoder.kernel.grad.abs().sum() > 0
    assert graph.actor.log_std.requires_grad is False
    assert relational_contribution(graph,packets,graphs)["relation_active_on_probe"]
    assert all(p.grad is None for p in graph.actor.raw.parameters())


def test_vertical_residual_gate_keeps_raw_policy_and_climb_intact():
    cfg = reference()
    agent = TwoAxisPPOAgent("ppo_ontology_rgat", graph_config=cfg.ontology)
    graphs = torch.zeros(3,9,12)
    graphs[:,7,0] = torch.tensor([0., .5, 1.])
    with torch.no_grad():
        agent.actor.encoder.readout.bias.fill_(1)
        agent.actor.residual.weight.fill_(-1)
    down = agent.actor.relational_delta(graphs)[:,1]
    assert down[0] == 0 and torch.allclose(down[1], .5*down[2])
    with torch.no_grad():
        agent.actor.residual.weight.fill_(1)
    up = agent.actor.relational_delta(graphs)[:,1]
    assert torch.all(up > 0) and torch.equal(up[0],up[2])


def test_same_timestamp_noise_is_paired_despite_visibility_history():
    cfg = reference().camera
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    common = dict(h_m=5., theta_rad=0., config=cfg)
    observe_pad(timestamp_s=0.,ex_m=100.,rng=a,**common)
    observe_pad(timestamp_s=0.,ex_m=0.,rng=b,**common)
    ma = observe_pad(timestamp_s=.1,ex_m=0.,rng=a,**common)
    mb = observe_pad(timestamp_s=.1,ex_m=0.,rng=b,**common)
    assert ma == mb


def test_sensor_updates_at_physics_rate(monkeypatch):
    env = TwoAxisLandingEnv(reference(), perturbations=False)
    obs,_ = env.reset(seed=42)
    calls = []
    original = env.estimator.update
    def update(measurement, **kwargs):
        calls.append(measurement.timestamp_s)
        return original(measurement, **kwargs)
    monkeypatch.setattr(env.estimator,"update",update)
    env.step(np.zeros(2))
    assert len(calls) == 10
    np.testing.assert_allclose(calls, np.arange(1,11)*.01)


def test_interpolated_contact_is_the_returned_state_and_elapsed_time():
    env = TwoAxisLandingEnv(reference(), perturbations=False)
    env.reset(seed=42)
    env.state = replace(env.state,z_m=.041,vz_m_s=-.3)
    _,_,done,_,info = env.step(np.zeros(2))
    assert done
    assert env.state.time_s == pytest.approx(info["contact"]["time_s"])
    assert info["dt_s"] == pytest.approx(env.state.time_s)
    assert env.state.z_m == pytest.approx(env.stage.safety.touchdown_height_m)


def test_curriculum_floor_reaches_nominal_with_zero_landings():
    cfg = reference()
    scheduler = CurriculumScheduler(cfg.curriculum)
    scheduler.set_budget_progress(.8)
    assert scheduler.at_nominal
    stage = stage_configs(cfg,1.)
    assert stage.scenario is cfg.scenario and stage.reward is cfg.reward and stage.safety is cfg.safety
    assert stage_configs(cfg,0.).scenario.initial_height_range_m == (.05,1.)
    env = TwoAxisLandingEnv(cfg,perturbations=False)
    for seed in range(10):
        env.reset(seed=seed,difficulty=0.)
        assert env.measurement.geometric_visible


def test_potential_shaping_is_discount_consistent_and_zero_at_terminal():
    cfg = reference()
    common = dict(ex_true_m=1.,h_true_m=2.,measured_bearing_rad=0.,bearing_valid=True,
        normalized_policy_action=np.zeros(2),fov_rad=cfg.camera.fov_rad,dt_s=.07,
        config=cfg.reward,previous_goal_cost=.4,discount_time_constant_s=70.)
    running = compute_reward(terminal_reason=None,**common)
    expected = math.exp(-.07/70)*(-2*running.goal_cost) + .8
    assert running.potential_shaping == pytest.approx(expected)
    terminal = compute_reward(terminal_reason=TerminalReason.SUCCESS,**common)
    assert terminal.potential_shaping == pytest.approx(.8)


def test_readiness_rewards_slow_closing_and_penalizes_pitch_rate():
    cfg = reference()
    common = dict(ex_true_m=.2,h_true_m=.1,pitch_rad=0.,vertical_speed_m_s=-.05,
                  safety=cfg.safety,reward_config=cfg.reward)
    closing = landing_readiness(relative_speed_m_s=-.12,**common)
    retreating = landing_readiness(relative_speed_m_s=.12,**common)
    assert closing > retreating
    assert landing_readiness(relative_speed_m_s=-.12,pitch_rate_rad_s=1.,**common) < closing


def test_versioned_signature_rejects_legacy_graph():
    reference_signature = experiment_signature(reference(), graph_schema_hash=GRAPH_SCHEMA_HASH)
    from ontology_rgat.two_axis.ontology import GRAPH_SCHEMA_HASH as legacy
    legacy_signature = experiment_signature(load_config(), graph_schema_hash=legacy)
    with pytest.raises(ValueError):
        reference_signature.assert_compatible(asdict(legacy_signature))


def test_peer_restarts_cannot_evade_retry_budget(monkeypatch):
    from ontology_rgat.ppo import recurrent_train as rt
    from ontology_rgat.bridge import GatewayTimeout
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise GatewayTimeout("gateway timeout")
    monkeypatch.setattr(rt,"collect_episode",failed)
    env = SimpleNamespace(cfg=SimpleNamespace(external={"episode_recoveries":2}),
                          recover_infrastructure=lambda:False)
    with pytest.raises(GatewayTimeout):
        rt.collect_episode_resilient(env,object(),"test",42)
    assert len(calls) == 3


def test_explicit_legacy_adapter_preserves_requested_units():
    from ontology_rgat.two_axis.adapter import px4_gateway_command
    cfg = reference()
    controller = SimpleNamespace(max_acceleration=np.array([3.,3.]),action_scale=1.,
        max_longitudinal_tilt=math.radians(20),tilt_channel_enabled=True)
    command = px4_gateway_command(np.array([.5,-.5]),cfg.dynamics,controller=controller)
    assert command[0]*3 == pytest.approx(1.25)
    assert command[1]*3 == pytest.approx(-1.)
    from ontology_rgat.controllers.planar_controller import STANDARD_GRAVITY
    assert STANDARD_GRAVITY*math.tan(command[2]*controller.max_longitudinal_tilt) == pytest.approx(1.25)
    controller.max_acceleration = np.array([.1,.1])
    with pytest.raises(ValueError,match="no silent clipping"):
        px4_gateway_command(np.ones(2),cfg.dynamics,controller=controller)


def test_run_sh_never_reaches_a_flight_stack_without_being_asked():
    """Bare `run.sh` now TRAINS, but it must still never acquire a stack.

    The previous guard asserted bare `run.sh` was help and ran it with a 5 s
    timeout. Once the bare route started the pipeline that assertion made the
    test itself launch a behaviour-cloning job, which outlived the SIGKILL and
    ran to completion in the background. So this never invokes the working
    route: it reads the plan with --dry-run, and checks the routing table for
    the flight entries rather than executing them.
    """
    script = ROOT.parent / "run.sh"
    help_text = subprocess.run(["bash", str(script), "--help"],
                               capture_output=True, text=True, timeout=30)
    assert help_text.returncode == 0 and "reference-smoke" in help_text.stdout
    assert "Stopping" not in help_text.stdout
    # The bare route is named, and says plainly that it is long-running.
    assert "TRAINS" in help_text.stdout

    plan = subprocess.run(["bash", str(script), "all", "--dry-run"],
                          capture_output=True, text=True, timeout=120)
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert "Isaac/PX4 : off" in plan.stdout
    # A dry run plans only; it must not have taken the lock or made a root.
    assert "clone" not in plan.stdout.lower().replace("isaac/px4", "")

    # Isaac stays behind its own explicit routes, and takeover stays opt-in.
    body = script.read_text()
    isaac = body[body.index("isaac-legacy)"):]
    assert "--takeover" not in body.split("case")[0]
    assert "run_isaac_legacy_entry.sh" in isaac
    for route in ("reference-smoke)", "status)", "--help|-h)"):
        assert route in body


def test_fairness_is_derived_not_assumed():
    from run_two_axis_pipeline import fairness_report
    rows = [dict(arm=arm,seed=1,signature={},hyperparameters={},checkpoint_eligible=True,
                 validation_seeds=[2000],test_seeds=[3000],comparison_factors=["state_representation"])
            for arm in POLICY_MODES]
    assert fairness_report(rows)["passes"]
    rows[1]["test_seeds"] = [3001]
    assert not fairness_report(rows)["passes"]
    assert not fairness_report(rows[:1])["complete_arm_seed_matrix"]


def test_masked_pretraining_uses_train_packets_and_keeps_zero_context():
    from ontology_rgat.two_axis.pretraining import pretrain_causal_encoder
    cfg = reference()
    cfg = replace(cfg,ontology=replace(cfg.ontology,pretrain_episodes=1,pretrain_decisions=4,pretrain_epochs=1))
    agent = TwoAxisPPOAgent("ppo_ontology_rgat",graph_config=cfg.ontology)
    info = pretrain_causal_encoder(agent,cfg,seed=1)
    assert info["environment_steps"] == 4
    assert info["uses_reward_or_truth_or_future"] is False
    assert info["seeds"][0] >= 5_000_000
    assert torch.count_nonzero(agent.actor.encoder.readout.weight) == 0
    assert torch.equal(agent.actor.encoder.kernel, agent.critic.encoder.kernel)


def test_invalid_timing_and_nominal_evaluation_fail_closed():
    from ontology_rgat.two_axis.training import evaluate_policy
    cfg = reference()
    with pytest.raises(ValueError):
        _validate(replace(cfg,timing=replace(cfg.timing,physics_dt_s=0)))
    with pytest.raises(ValueError):
        evaluate_policy(None,cfg,seeds=[2000],difficulty=.5)


def test_matlab_generated_packet_graph_and_reward_golden():
    from ontology_rgat.two_axis.contracts import load_reference_registry, normalize_reference_fields
    from ontology_rgat.contracts.observation import CausalObservationPacket
    from ontology_rgat.two_axis.ontology_v28 import build_context_graph
    from ontology_rgat.two_axis.reward import goal_cost
    cfg = reference()
    golden = json.loads((ROOT/"tests/fixtures/reference_v28_matlab.json").read_text())
    registry = load_reference_registry()
    normalized = normalize_reference_fields(golden["packet_physical"], half_fov=cfg.camera.fov_rad/2)
    packet = CausalObservationPacket.from_fields(normalized,timestamp_s=0.,registry=registry)
    np.testing.assert_allclose(packet.values,golden["normalized_packet"],atol=5e-8)
    graph = build_context_graph(packet,registry,cfg)
    np.testing.assert_allclose(graph.X,golden["graph_X"],atol=1e-7)
    def readiness(ex,h,rv,theta,rate):
        return landing_readiness(ex_true_m=ex,h_true_m=h,relative_speed_m_s=rv,
            vertical_speed_m_s=-.2,pitch_rad=theta,pitch_rate_rad_s=rate,
            safety=cfg.safety,reward_config=cfg.reward)
    result = compute_reward(ex_true_m=.2,h_true_m=4.,measured_bearing_rad=.1,
        bearing_valid=True,normalized_policy_action=np.array([.2,-.1]),fov_rad=cfg.camera.fov_rad,
        dt_s=.1,terminal_reason=None,config=cfg.reward,readiness=readiness(.2,4,-.12,.03,.05),
        previous_readiness=readiness(.25,4.02,-.1,.02,.04),previous_goal_cost=goal_cost(.25,4.02,cfg.reward))
    assert result.total == pytest.approx(golden["reward"],abs=1e-12)
    assert result.readiness == pytest.approx(golden["reward_components"]["landingReadiness"],abs=1e-12)


def test_reference_perturbations_include_clean_and_bounded_pitch_events():
    from ontology_rgat.two_axis.sensing import DropoutSchedule
    cases = [DropoutSchedule.reference_mixture(np.random.default_rng(s),50) for s in range(400)]
    clean = sum(not item.intervals_s for item in cases)
    assert 150 < clean < 250
    assert any(item.pitch_event is not None for item in cases)
    assert all(abs(item.pitch_event[2]) <= math.radians(2) for item in cases if item.pitch_event)


def test_reference_tracker_uses_causal_own_velocity_prior_and_is_idempotent():
    from ontology_rgat.two_axis.estimation import CausalPadEstimator
    from ontology_rgat.two_axis.sensing import PadMeasurement
    estimator = CausalPadEstimator(reference().estimator)
    measurement = PadMeasurement(0.,True,0.,True,.2,True,.9,True)
    first = estimator.update(measurement,own_x_m=1.,own_vx_m_s=2.)
    assert first.pad_vx_m_s == 2.
    assert estimator.update(measurement,own_x_m=99.,own_vx_m_s=99.) is first
    with pytest.raises(ValueError):
        estimator.update(replace(measurement,timestamp_s=-1),own_x_m=1.)


def test_reference_checkpoint_score_penalizes_unsafe_contact_strongly():
    from ontology_rgat.two_axis.training import selection_score
    unsafe_candidate = selection_score(["SUCCESS","UNSAFE_CONTACT"],[25,-40])
    hovering_candidate = selection_score(["TASK_TIMEOUT","TASK_TIMEOUT"],[-12,-12])
    assert hovering_candidate > unsafe_candidate


def test_reenabling_historical_ramp_requires_fixing_abort_order():
    cfg = load_config()
    with pytest.raises(ValueError,match="SAFE_ABORT"):
        _validate(replace(cfg,curriculum=replace(cfg.curriculum,enabled=True)))
