"""The learner's own contracts: configuration, semantics, observation, rewards.

These are the invariants that used to be enforced by MATLAB assertions spread
through ``defaultExternalConfig``, ``makeObservation`` and
``buildOntologyGraph``. Each is a thing that has actually gone wrong: a schema
that drifted from the model, an observation that changed width, a shaping term
that stopped being policy-invariant.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ontology_rgat.config import default_config
from ontology_rgat.env import terminal_status, state_to_model, truth_state
from ontology_rgat.expert import expert_action
from ontology_rgat.mathx import euler_to_quat, quat_to_euler_zyx, wrap_pi
from ontology_rgat.rewards import manual_dense, proposed_pbrs, sparse_task
from ontology_rgat.semantic import (GOAL_NODE, SemanticState, build_ontology_graph,
                                    compute_features, make_observation)


@pytest.fixture
def cfg():
    return default_config("quick")


# ------------------------------------------------------------------- config
def test_pbrs_gamma_matches_ppo_gamma(cfg):
    """Potential-based shaping is only policy-invariant when the two agree."""
    assert cfg.reward.pbrs.gamma == cfg.ppo.gamma


def test_ontology_has_the_three_external_nodes_before_the_goal(cfg):
    names = cfg.ontology.node_names
    assert names[10] == "PadMotion"
    assert names[11] == "BatteryReserve"
    assert names[12] == "GnssIntegrity"
    assert names[13] == "SafeLanding"
    assert cfg.ontology.n_nodes == 14
    assert cfg.ontology.in_dim == 4 + 14


def test_relative_speed_criterion_matches_system_yaml(cfg):
    """cfg.criteria and config/system.yaml both state the success criteria."""
    import yaml
    with open(cfg.paths.system_yaml, encoding="utf-8") as handle:
        system = yaml.safe_load(handle)
    landing = system["landing"]
    assert cfg.criteria.rel_speed_xy == pytest.approx(landing["success_rel_speed_xy_m_s"])
    assert cfg.criteria.xy == pytest.approx(landing["success_xy_m"])
    assert cfg.criteria.vz == pytest.approx(landing["success_vz_m_s"])
    assert math.degrees(cfg.criteria.tilt) == pytest.approx(landing["success_tilt_deg"])


def test_collective_mapping_matches_system_yaml(cfg):
    """A mis-scaled action is an invalid experiment, not a degraded one."""
    import yaml
    with open(cfg.paths.system_yaml, encoding="utf-8") as handle:
        px4 = yaml.safe_load(handle)["px4"]
    assert cfg.rl.collective_span == pytest.approx(px4["collective_span"])
    assert math.degrees(cfg.rl.max_roll_pitch) == pytest.approx(px4["max_roll_pitch_deg"])
    assert math.degrees(cfg.rl.max_yaw_rate) == pytest.approx(px4["max_yaw_rate_deg_s"])


def test_derive_does_not_mutate_the_original(cfg):
    """The sweeps rely on this; MATLAB structs gave it for free, dicts do not."""
    scaled = cfg.derive(**{"external.pad_scale": 2.5})
    assert scaled.external.pad_scale == 2.5
    assert cfg.external.pad_scale == 1.0


# ------------------------------------------------------------------- mathx
def test_quaternion_euler_round_trip():
    for rpy in ([0.1, -0.2, 0.3], [0.0, 0.0, 3.0], [-0.4, 0.15, -2.8]):
        back = quat_to_euler_zyx(euler_to_quat(rpy))
        np.testing.assert_allclose(back, rpy, atol=1e-12)


def test_wrap_pi_is_idempotent_on_its_own_range():
    angles = np.linspace(-20.0, 20.0, 101)
    once = wrap_pi(angles)
    np.testing.assert_allclose(wrap_pi(once), once, atol=1e-12)
    assert np.all(once > -math.pi - 1e-12) and np.all(once <= math.pi + 1e-12)


# -------------------------------------------------------------- observation
def _meas(**overrides):
    base = {"pos": np.array([0.4, -0.3, 2.1]), "vel": np.array([0.2, -0.1, -0.5]),
            "rpy": np.array([0.05, -0.03, 0.4]), "omega": np.array([0.1, 0.0, -0.2]),
            "marker_quality": 0.8, "pad_velocity": np.array([0.7, -0.3, 0.0]),
            "pad_speed": 0.76, "battery": {"enabled": True}}
    base.update(overrides)
    return base


def _diag(**overrides):
    base = {"mean_wind_i": np.array([1.3, 0.4, 0.0]), "aero_force_mag": 0.6,
            "pad_velocity_i": np.array([0.7, -0.3, 0.0]),
            "battery": {"enabled": True, "hover_seconds_remaining": 20.0,
                        "landing_reserve_s": 2.5, "reserve": 0.6}}
    base.update(overrides)
    return base


def test_observation_matches_the_declared_width_and_is_bounded(cfg):
    sem = compute_features(_diag(), _meas(), None, cfg)
    obs = make_observation(_meas(), sem, cfg)
    assert obs.shape == (cfg.rl.obs_dim,)
    assert np.all(np.abs(obs) <= 3.0)
    assert np.isfinite(obs).all()


def test_observation_carries_the_deck_velocity_feed_forward(cfg):
    """The policy must be able to lead the deck, not only react to error."""
    still = _meas(pad_velocity=np.zeros(3))
    moving = _meas(pad_velocity=np.array([1.2, 0.0, 0.0]))
    sem_still = compute_features(_diag(pad_velocity_i=np.zeros(3)), still, None, cfg)
    sem_moving = compute_features(_diag(), moving, None, cfg)
    a = make_observation(still, sem_still, cfg)
    b = make_observation(moving, sem_moving, cfg)
    assert not np.allclose(a[14:16], b[14:16])


def test_static_matched_deck_gives_zero_pad_motion(cfg):
    """A stationary deck with a matched vehicle is the fixed-pad control case."""
    meas = _meas(vel=np.array([0.0, 0.0, -0.5]), pad_velocity=np.zeros(3))
    sem = compute_features(_diag(pad_velocity_i=np.zeros(3)), meas, None, cfg)
    assert sem.pad_motion == pytest.approx(0.0)


def test_a_link_without_gnss_reads_as_open_sky_not_as_an_outage(cfg):
    """The protocol fake models no canyon; that must not read as no fix."""
    sem = compute_features(_diag(), _meas(), None, cfg)
    assert sem.gnss_integrity == 1.0
    assert sem.gnss_nlos_fraction == 0.0


def test_integrity_is_bounded_by_the_worse_of_the_two_receivers(cfg):
    """The fallback pose is a difference of two fixes, so the weaker one rules."""
    good = {"enabled": True, "quality": 0.9, "deck_quality": 0.2}
    sem = compute_features(_diag(gnss=good), _meas(), None, cfg)
    assert sem.gnss_integrity == pytest.approx(0.2)


def test_a_lost_fix_is_zero_integrity_whatever_the_quality_says(cfg):
    lost = {"enabled": True, "valid": False, "quality": 0.8, "deck_quality": 0.8}
    sem = compute_features(_diag(gnss=lost), _meas(), None, cfg)
    assert sem.gnss_integrity == 0.0


def test_markers_and_the_fix_substitute_for_each_other(cfg):
    """Either source alone can carry the landing; that is what nav confidence is."""
    canyon = {"enabled": True, "quality": 0.05, "deck_quality": 0.05}
    open_sky = {"enabled": True, "quality": 0.95, "deck_quality": 0.95}
    seen = _meas(marker_quality=0.9)
    lost = _meas(marker_quality=0.0)

    assert compute_features(_diag(gnss=canyon), seen, None, cfg).nav_confidence > 0.85
    assert compute_features(_diag(gnss=open_sky), lost, None, cfg).nav_confidence > 0.85
    both_bad = compute_features(_diag(gnss=canyon), lost, None, cfg)
    assert both_bad.nav_confidence < 0.2
    # And flying blind has to cost less touchdown safety than seeing the pad.
    assert both_bad.touchdown_safety < compute_features(
        _diag(gnss=canyon), seen, None, cfg).touchdown_safety


def test_the_observation_carries_the_receivers_own_account_of_itself(cfg):
    """A policy that cannot see the fix quality cannot tell a good pose from a
    bad one: in this environment the two look identical."""
    canyon = {"enabled": True, "quality": 0.1, "deck_quality": 0.1,
              "sigma_xy_m": 18.0, "nlos_detected_fraction": 0.5}
    clear = {"enabled": True, "quality": 0.98, "deck_quality": 0.98,
             "sigma_xy_m": 1.2, "nlos_detected_fraction": 0.0}
    a = make_observation(_meas(), compute_features(_diag(gnss=canyon), _meas(), None, cfg), cfg)
    b = make_observation(_meas(), compute_features(_diag(gnss=clear), _meas(), None, cfg), cfg)
    assert not np.allclose(a[-3:], b[-3:])
    assert np.all(np.abs(a) <= 3.0)


def test_a_link_without_energy_leaves_the_channels_neutral(cfg):
    """The MAVLink fallback reports no battery; that must not read as empty."""
    diag = _diag(battery={"enabled": False})
    sem = compute_features(diag, _meas(battery={"enabled": False}), None, cfg)
    assert sem.battery_reserve == 1.0
    assert sem.energy_margin == 1.0


# -------------------------------------------------------------- graph shape
def test_graph_features_and_topology(cfg):
    sem = compute_features(_diag(), _meas(), None, cfg)
    graph = build_ontology_graph(sem, cfg)
    assert graph.X.shape == (cfg.ontology.in_dim, cfg.ontology.n_nodes)
    assert graph.goal_node == GOAL_NODE
    # No future-label leakage: the goal node carries no value of its own.
    assert graph.X[0, GOAL_NODE] == 0.0
    # Every node has a self-loop, and the goal is reachable from every node.
    self_loops = {int(s) for s, d, r in zip(graph.src, graph.dst, graph.rel) if s == d}
    assert self_loops == set(range(cfg.ontology.n_nodes))
    assert graph.src.max() < cfg.ontology.n_nodes
    assert graph.rel.max() == cfg.ontology.n_relations - 1


def test_a_shorter_schema_is_refused(cfg):
    """A potential trained before GnssIntegrity existed must not load."""
    cfg.ontology.n_nodes = 13
    with pytest.raises(ValueError, match="14 nodes"):
        build_ontology_graph(SemanticState(), cfg)


# ------------------------------------------------------------------ rewards
def test_sparse_reward_scores_depletion_as_a_failure(cfg):
    """Running out of energy in the air is a mission failure, not a neutral end.

    It has its own configurable term rather than being folded into the timeout
    -- the shipped values happen to coincide, but they are separately tunable
    because they are different events.
    """
    assert "battery_depleted" in cfg.reward.sparse
    depleted = sparse_task("battery_depleted", cfg)
    assert depleted <= sparse_task("timeout", cfg)
    assert depleted < sparse_task("running", cfg)
    assert sparse_task("success", cfg) > sparse_task("unsafe_touchdown", cfg)


def test_crash_penalty_is_graded_not_a_cliff(cfg):
    """A near miss must cost less than a hard crash, or there is no gradient."""
    class Fake:
        def __init__(self, sem):
            self.sem = sem
            self.meas = {"vel": np.zeros(3)}

    sem = compute_features(_diag(), _meas(), None, cfg)
    cur = nxt = Fake(sem)
    near = manual_dense(cur, nxt, np.zeros(4), "unsafe_touchdown", 1.05, cfg)
    hard = manual_dense(cur, nxt, np.zeros(4), "unsafe_touchdown", 5.0, cfg)
    assert near > hard


def test_pbrs_shaping_vanishes_on_a_constant_potential(cfg):
    """With gamma*Phi' == Phi the shaping term is zero, which is the invariance."""
    class Constant:
        def predict(self, graph):
            return 0.0

    sem = compute_features(_diag(), _meas(), None, cfg)
    graph = build_ontology_graph(sem, cfg)

    class Fake:
        def __init__(self):
            self.graph = graph

    r, parts = proposed_pbrs(Fake(), Fake(), "running", Constant(), cfg)
    assert parts["shape"] == pytest.approx(0.0)
    assert r == pytest.approx(sparse_task("running", cfg))


# -------------------------------------------------------------- termination
def _state(pos, vel=(0, 0, 0), rpy=(0, 0, 0), omega=(0, 0, 0)):
    return np.concatenate([np.asarray(pos, float), np.asarray(vel, float),
                           euler_to_quat(rpy), np.asarray(omega, float)])


def test_touchdown_on_the_ground_before_flight_is_not_a_landing(cfg):
    """Control is handed over with the vehicle possibly still on the deck."""
    x = _state([0.0, 0.0, 0.0])
    done, status, _ = terminal_status(x, cfg, has_been_airborne=False)
    assert not done and status == "running"


def test_unmatched_horizontal_speed_fails_the_landing(cfg):
    """Touching a moving deck without sharing its velocity tips the airframe."""
    slow = _state([0.05, 0.0, 0.0], vel=[0.1, 0.0, -0.2])
    fast = _state([0.05, 0.0, 0.0], vel=[1.2, 0.0, -0.2])
    assert terminal_status(slow, cfg)[1] == "success"
    assert terminal_status(fast, cfg)[1] == "unsafe_touchdown"


def test_losing_the_deck_is_a_mission_failure_not_a_crash(cfg):
    x = _state([cfg.sim.world_xy_limit + 1.0, 0.0, 3.0])
    done, status, _ = terminal_status(x, cfg)
    assert done and status == "flight_failure"


def test_depletion_in_the_air_is_its_own_failure(cfg):
    x = _state([0.1, 0.0, 2.0])
    done, status, _ = terminal_status(x, cfg, battery_depleted=True)
    assert done and status == "battery_depleted"


# ------------------------------------------------------- state conversion
def _wire_state(**overrides):
    base = {
        "position": [0.4, -0.3, 2.1], "velocity": [0.2, -0.1, -0.5],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
        "acceleration": [0.0, 0.0, -9.8], "wind": [1.3, 0.4, 0.0],
        "aero_force": [0.1, 0.0, 0.0], "marker_quality": 0.75,
        "pad": {"valid": True, "source": "lissajous", "position": [3.0, 1.0, 0.32],
                "velocity": [0.7, -0.3, 0.0], "yaw": 0.4, "yaw_rate": 0.1},
        "battery": {"enabled": True, "hover_seconds_remaining": 20.0,
                    "landing_reserve_s": 2.5, "reserve": 0.6, "power_w": 190.0},
        "landed": False, "armed": True,
    }
    base.update(overrides)
    return base


def test_state_to_model_keeps_position_pad_relative(cfg):
    x, diag = state_to_model(_wire_state(), cfg)
    np.testing.assert_allclose(x[0:3], [0.4, -0.3, 2.1])
    np.testing.assert_allclose(diag["pad_position_i"], [3.0, 1.0, 0.32])
    assert diag["pad_speed"] == pytest.approx(math.hypot(0.7, 0.3))


def test_missing_battery_block_defaults_to_disabled(cfg):
    _, diag = state_to_model(_wire_state(battery={}), cfg)
    assert diag["battery"]["enabled"] is False
    assert diag["battery"]["reserve"] == 1.0


def test_sensor_and_ground_truth_contracts_are_separate(cfg):
    _, diag = state_to_model(_wire_state(), cfg)
    assert diag["data_policy"] == "sensor_for_control_gt_for_validation_only"
    assert np.array_equal(diag["sensor"]["position"], [0.4, -0.3, 2.1])
    assert np.array_equal(diag["ground_truth"]["world_position"], [0.4, -0.3, 2.1])
    assert "world_position" not in diag["sensor"]


def test_the_episode_is_scored_on_truth_and_flown_on_the_measurement(cfg):
    """Under a canyon fix the two differ by metres, and only one of them is
    the landing. Grading on the sensor would score the receiver's mistake."""
    wire = _wire_state(position=[3.0, 0.0, 0.02],
                       truth={"valid": True, "position": [0.05, 0.0, 0.02],
                              "velocity": [0.1, 0.0, -0.2]})
    x, diag = state_to_model(wire, cfg)
    np.testing.assert_allclose(x[0:3], [3.0, 0.0, 0.02])       # what it flies on
    scored = truth_state(x, diag)
    np.testing.assert_allclose(scored[0:3], [0.05, 0.0, 0.02])  # what it is graded on
    assert terminal_status(scored, cfg)[1] == "success"
    assert terminal_status(x, cfg)[1] == "unsafe_touchdown"


def test_without_a_truth_block_the_sensor_is_all_there_is(cfg):
    """The fixed-pad experiment and the protocol fake both work this way."""
    x, diag = state_to_model(_wire_state(), cfg)
    assert not diag["ground_truth"]["valid"]
    np.testing.assert_allclose(truth_state(x, diag), x)


# ------------------------------------------------------------------- expert
def test_expert_action_is_inside_the_action_box(cfg):
    for z in (0.1, 1.0, 4.5):
        x = _state([0.6, -0.4, z], vel=[0.3, 0.2, -0.4], rpy=[0.05, -0.02, 0.6])
        a = expert_action(x, cfg, None)
        assert a.shape == (4,)
        assert np.all(np.abs(a) <= 1.0)


def test_expert_descends_faster_when_the_energy_margin_is_thin(cfg):
    """Without this, every low-reserve episode is a negative training example."""
    class Ctx:
        def __init__(self, margin):
            self.meas = {"pad_velocity": np.zeros(3)}
            self.sem = SemanticState(energy_margin=margin)

    # Low, because above roughly 1.9 m the nominal profile already asks for the
    # fastest descent cfg.criteria.vz permits and urgency has nothing to add.
    x = _state([0.0, 0.0, 0.5], vel=[0.0, 0.0, 0.0])
    comfortable = expert_action(x, cfg, Ctx(1.0))
    urgent = expert_action(x, cfg, Ctx(-0.5))
    # A lower collective command is a faster descent.
    assert urgent[0] < comfortable[0]
    # And never faster than the criteria would accept, at any altitude.
    for z in (0.2, 1.0, 3.0, 4.8):
        a = expert_action(_state([0.0, 0.0, z]), cfg, Ctx(-3.0))
        assert a[0] >= -1.0


def test_expert_fades_the_deck_velocity_lead_at_touchdown(cfg):
    """A chase feed-forward must not become a permanent landing offset."""
    class Ctx:
        meas = {"pad_velocity": np.array([3.0, 0.0, 0.0])}
        sem = SemanticState(energy_margin=1.0)

    stopped = expert_action(_state([0.0, 0.0, 0.0]), cfg, None)
    touchdown = expert_action(_state([0.0, 0.0, 0.0]), cfg, Ctx())
    approach = expert_action(_state([0.0, 0.0, 4.0]), cfg, Ctx())
    assert touchdown[2] == pytest.approx(stopped[2])
    assert approach[2] > touchdown[2]
