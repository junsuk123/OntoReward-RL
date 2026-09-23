"""The reduced control envelope every arm of the comparison flies.

What is fixed here is not tuning -- it is the set of statements the comparison
rests on. If any of them stops being true, an arm has authority the others do
not, or a degree of freedom the experiment declares constrained is not.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config                     # noqa: E402
import pad_motion                                          # noqa: E402

from ontology_rgat.benchmarks.experiment import load_experiment   # noqa: E402
from ontology_rgat.controllers import (                    # noqa: E402
    PLANAR_ACTION_DIM, PlanarLongitudinalController, PNGuidanceConfig,
    PNGuidanceController, PNGuidanceState)
from ontology_rgat.initialization import nadir_image_setpoint     # noqa: E402
from ontology_rgat.perception.semantic_observation import (       # noqa: E402
    SemanticObservation)

from conftest import default_experiment_config             # noqa: E402


SETPOINT = nadir_image_setpoint()


def _observation(centroid, *, scale=0.06, visible=1.0, loss=0.0, vz_safety=0.6):
    return SemanticObservation(
        keypoint_confidence=0.9, visible_keypoint_fraction=float(visible),
        image_alignment=0.5, apparent_target_scale=min(1.0, float(scale) / 0.75),
        image_plane_motion_safety=0.9, scale_rate_safety=0.9,
        visibility_memory=1.0, reacquisition_trend=0.5,
        vertical_motion_safety=float(vz_safety), attitude_stability=0.9,
        battery_risk=0.1, visual_loss_risk=float(loss),
        centroid_xy=tuple(float(v) for v in centroid), raw_scale=float(scale),
        visual_loss_duration_s=0.0 if loss <= 0.0 else 0.5)


# ----------------------------------------------------------- the controller

def test_the_two_constrained_degrees_of_freedom_are_never_commanded():
    """Lateral velocity and yaw rate are constants of the experiment.

    Not "small", not "regulated to zero by a gain" -- exactly zero, for every
    action, at every point in the curriculum. A policy cannot buy lateral
    authority by saturating a channel, because there is no channel.
    """
    controller = PlanarLongitudinalController(dt=0.1)
    rng = np.random.default_rng(11)
    for curriculum in (0.0, 0.37, 1.0):
        controller.set_curriculum(curriculum)
        controller.reset()
        for _ in range(200):
            command = controller.command(rng.uniform(-1.0, 1.0, PLANAR_ACTION_DIM))
            assert command.velocity_body_heading_m_s[1] == 0.0
            assert command.yaw_rate_rad_s == 0.0
            assert command.as_array()[1] == 0.0
            assert command.as_array()[3] == 0.0


def test_the_action_is_acceleration_and_is_integrated_not_applied():
    """A step input must not become a step in velocity.

    The reduced study's dynamics integrate an acceleration command, and the
    live envelope has to do the same or the two are not the same problem: an
    arm that could command a velocity directly would skip the transient the
    task is largely about.
    """
    controller = PlanarLongitudinalController(
        max_velocity=(1.6, 0.9), max_acceleration=(1.2, 0.8), dt=0.1,
        curriculum_min_action_scale=1.0)
    first = controller.command([1.0, 1.0, 0.0])
    assert first.velocity_body_heading_m_s[0] == pytest.approx(0.12)
    assert first.velocity_body_heading_m_s[2] == pytest.approx(0.08)
    second = controller.command([1.0, 1.0, 0.0])
    assert second.velocity_body_heading_m_s[0] == pytest.approx(0.24)
    # And it saturates at the velocity limit rather than integrating forever.
    for _ in range(400):
        command = controller.command([1.0, 1.0, 0.0])
    np.testing.assert_allclose(
        command.velocity_body_heading_m_s[[0, 2]], [1.6, 0.9])


def test_the_tilt_channel_is_rate_limited_and_bounded():
    controller = PlanarLongitudinalController(
        dt=0.1, max_longitudinal_tilt_deg=12.0, max_tilt_rate_deg_s=60.0,
        curriculum_min_action_scale=1.0)
    first = controller.command([0.0, 0.0, 1.0])
    # One step of 60 deg/s at dt = 0.1 s is 6 deg, not the 12 deg limit.
    assert math.degrees(first.longitudinal_tilt_rad) == pytest.approx(6.0)
    for _ in range(20):
        command = controller.command([0.0, 0.0, 1.0])
    assert math.degrees(command.longitudinal_tilt_rad) == pytest.approx(12.0)
    # The tilt it reports and the acceleration feed-forward it implies agree.
    assert command.tilt_feedforward_m_s2 == pytest.approx(
        9.80665 * math.tan(command.longitudinal_tilt_rad))


def test_the_tilt_channel_can_be_switched_off_without_changing_the_action_space():
    """The two-channel ablation keeps its checkpoints loadable."""
    controller = PlanarLongitudinalController(
        dt=0.1, tilt_channel_enabled=False, curriculum_min_action_scale=1.0)
    for _ in range(50):
        command = controller.command([0.5, -0.5, 1.0])
    assert command.longitudinal_tilt_rad == 0.0
    # The action is still three-dimensional, so a policy trained with the
    # channel on can be evaluated with it off and vice versa.
    with pytest.raises(ValueError):
        controller.command([0.5, -0.5])


def test_the_curriculum_scales_every_channel_and_nothing_else():
    controller = PlanarLongitudinalController(dt=0.1,
                                              curriculum_min_action_scale=0.35)
    controller.set_curriculum(0.0)
    for _ in range(400):
        low = controller.command([1.0, 1.0, 1.0])
    controller.reset()
    controller.set_curriculum(1.0)
    for _ in range(400):
        high = controller.command([1.0, 1.0, 1.0])
    ratio = 0.35
    np.testing.assert_allclose(low.velocity_body_heading_m_s[[0, 2]],
                               high.velocity_body_heading_m_s[[0, 2]] * ratio)
    assert low.longitudinal_tilt_rad == pytest.approx(
        high.longitudinal_tilt_rad * ratio)
    assert low.yaw_rate_rad_s == high.yaw_rate_rad_s == 0.0


def test_a_velocity_law_reaches_its_target_through_the_acceleration_channel():
    """The one conversion every analytic arm goes through.

    PN guidance, the privileged PD teacher and the retired image servo are all
    velocity laws. They must reach PX4 through the same integrator and the
    same limits as the policy, or the control condition is not comparable.
    """
    controller = PlanarLongitudinalController(dt=0.1,
                                              curriculum_min_action_scale=1.0)
    for _ in range(40):
        command = controller.command(controller.action_for_velocity(0.5, -0.2))
    np.testing.assert_allclose(command.velocity_body_heading_m_s,
                               [0.5, 0.0, -0.2], atol=1e-9)
    # Holding a velocity needs no acceleration, so the implied tilt returns to
    # zero rather than standing at whatever got it there.
    assert command.longitudinal_tilt_rad == pytest.approx(0.0, abs=1e-9)
    # An unreachable target is approached at the envelope's maximum rate.
    controller.reset()
    action = controller.action_for_velocity(99.0, 99.0)
    np.testing.assert_allclose(action[:2], [1.0, 1.0])


# --------------------------------------------------------------- PN guidance

def _pn(controller=None):
    controller = controller or PlanarLongitudinalController(
        dt=0.1, curriculum_min_action_scale=1.0)
    return PNGuidanceController(
        PNGuidanceConfig(noise_std=0.0),
        nadir_column=float(SETPOINT[0]), tan_half_horizontal=1.0,
        max_longitudinal_acceleration_m_s2=float(controller.max_acceleration[0]),
        max_vertical_acceleration_m_s2=float(controller.max_acceleration[1]),
        max_longitudinal_tilt_rad=float(controller.max_longitudinal_tilt),
        dt=0.1)


def test_pn_guidance_reads_nothing_but_the_encoder_and_its_own_velocity():
    """The information boundary, checked on the signature rather than asserted."""
    import inspect

    parameters = tuple(inspect.signature(PNGuidanceController.action).parameters)
    assert parameters == ("self", "semantic", "body_velocity", "state", "rng")
    forbidden = ("truth", "relative", "platform", "pad", "deck", "critic",
                 "privileged", "geometric")
    source = inspect.getsource(PNGuidanceController)
    for token in forbidden:
        assert f"semantic.{token}" not in source
        assert f'"{token}' not in source.split('"""')[-1]


def test_pn_guidance_turns_a_growing_bearing_into_a_correction():
    """The law's defining behaviour: a line-of-sight rate produces a command.

    A pure pursuit law commands on the bearing itself and lags a moving
    target; PN commands on its RATE, which is what lets it lead one. Feeding a
    bearing that grows step by step must therefore produce a command that
    grows with it, and a constant bearing must not.
    """
    guidance = _pn()
    state = PNGuidanceState()
    body = np.array([0.0, 0.0, 0.0])
    steady = None
    for offset in (0.00, 0.00, 0.00, 0.00, 0.00):
        steady = guidance.action(_observation(SETPOINT + [offset, 0.0]),
                                 body, state)
    assert abs(state.bearing_rate) < 1e-6

    state = PNGuidanceState()
    moving = None
    for offset in (0.00, 0.05, 0.10, 0.15, 0.20):
        moving = guidance.action(_observation(SETPOINT + [offset, 0.0]),
                                 body, state)
    assert state.bearing_rate > 0.0
    assert abs(moving[0]) > abs(steady[0])


def test_pn_guidance_climbs_while_the_pad_is_lost_and_not_after_the_commit():
    guidance = _pn()
    state = PNGuidanceState()
    body = np.array([0.0, 0.0, 0.0])
    blind = guidance.action(_observation(SETPOINT, visible=0.0, loss=0.6),
                            body, state)
    assert state.diagnostics["mode"] == "search"
    assert blind[1] > 0.0, "a blind vehicle climbs to widen its footprint"

    # Past the flare the marker legitimately leaves a downward camera, so
    # losing it must not restart the climb.
    state = PNGuidanceState()
    guidance.action(_observation(SETPOINT, scale=0.5), body, state)
    assert state.committed
    committed = guidance.action(
        _observation(SETPOINT, scale=0.5, visible=0.0, loss=0.6), body, state)
    assert state.diagnostics["mode"] == "flare"
    assert committed[1] < 0.0


def test_pn_guidance_emits_the_shared_three_channel_action():
    guidance = _pn()
    state = PNGuidanceState()
    action = guidance.action(_observation(SETPOINT + [0.2, 0.0]),
                             np.array([0.1, 0.0, -0.2]), state)
    assert action.shape == (PLANAR_ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)
    # The tilt channel carries the tilt its own longitudinal command implies,
    # so the control condition exercises all three channels rather than
    # leaving one silent.
    assert np.sign(action[2]) == np.sign(action[0]) or action[0] == 0.0


def test_pn_guidance_needs_its_own_body_velocity_and_nothing_wider():
    guidance = _pn()
    with pytest.raises(ValueError):
        guidance.action(_observation(SETPOINT), np.zeros(6), PNGuidanceState())


# ------------------------------------------------------------- the protocol

def test_the_planar_fields_are_optional_and_bounded():
    """An older gateway and a recorded run both still speak the protocol."""
    from ontology_rgat_px4.protocol import (PLANAR_TILT_LIMIT_RAD, ProtocolError,
                                            validate_velocity_extras)

    assert validate_velocity_extras({"command": [0.0] * 4}) == (0.0, None)
    tilt, yaw = validate_velocity_extras(
        {"tilt_rad": 0.2, "yaw_rad": -1.1})
    assert tilt == pytest.approx(0.2) and yaw == pytest.approx(-1.1)
    for bad in ({"tilt_rad": PLANAR_TILT_LIMIT_RAD + 0.01},
                {"tilt_rad": float("nan")},
                {"yaw_rad": float("inf")},
                {"tilt_rad": "sideways"}):
        with pytest.raises(ProtocolError):
            validate_velocity_extras(bad)


def test_the_gateway_turns_a_tilt_into_an_acceleration_and_holds_the_heading():
    """The two mechanisms the envelope needs from the gateway.

    A tilt cannot be requested through a velocity setpoint directly; what PX4
    accepts is an acceleration feed-forward, and its position controller turns
    a horizontal acceleration into exactly the attitude that produces it. The
    heading is held absolutely rather than at a zero rate, because a zero rate
    lets yaw free-run and drift off the deck's track.
    """
    source = (ROOT / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4"
              / "ros2_gateway.py").read_text(encoding="utf-8")
    assert "GRAVITY_M_S2 * math.tan(tilt)" in source
    assert "sp.acceleration = [float(value)" in source
    assert "sp.yaw = float(yaw_enu_to_ned(self.velocity_yaw_hold_rad))" in source
    assert "sp.yawspeed = 0.0" in source
    # Both are clamped by the protocol bound as well as by the learner's own
    # envelope, so a malformed command cannot ask the airframe to invert.
    assert "PLANAR_TILT_LIMIT_RAD" in source


# ----------------------------------------------------------------- the decks

def test_the_three_decks_are_straight_constant_heading_speed_profiles():
    time_axis = np.arange(0.0, 30.0, 0.05)
    peaks = []
    for name in pad_motion.SEGMENTED_CRUISE_SCENARIOS:
        speeds = pad_motion.segmented_cruise_speeds(time_axis, name)
        targets = [value * pad_motion.SEGMENTED_CRUISE_SPEED_SCALE
                   for value in pad_motion.SEGMENTED_CRUISE_SPEEDS_M_S[name]]
        # Settled inside each segment, at the scenario's own speeds.
        for when, target in zip((2.5, 14.0, 29.0), targets):
            assert speeds[np.argmin(abs(time_axis - when))] == pytest.approx(
                target, abs=1e-6)
        # A real vehicle changes speed at a bounded rate.
        rate = np.max(np.abs(np.diff(speeds)) / np.diff(time_axis))
        assert rate <= pad_motion.SEGMENTED_CRUISE_ACCELERATION_M_S2 * (
            pad_motion.SEGMENTED_CRUISE_SPEED_SCALE + 1e-9)
        peaks.append(float(speeds.max()))
    # The fastest segment of the fastest deck lands on the carrier's ceiling.
    assert max(peaks) == pytest.approx(1.0, abs=1e-6)
    # And the three decks are ordered, so "slow/medium/fast" mean something.
    assert peaks == sorted(peaks)


def test_the_first_speed_change_arrives_before_any_landing_can():
    """A controller must not be able to land before the disturbance.

    The reduced study places the first switch at 3 s for exactly this reason:
    a deck whose speed changes after the episode is already over is not a
    disturbance, it is a constant-velocity deck with a longer name.
    """
    assert pad_motion.SEGMENTED_CRUISE_SWITCH_TIMES_S[0] <= 3.0
    assert (pad_motion.SEGMENTED_CRUISE_SWITCH_TIMES_S[1]
            > pad_motion.SEGMENTED_CRUISE_SWITCH_TIMES_S[0])


def test_the_decks_are_registered_in_every_vocabulary_that_gates_a_run():
    from ontology_rgat_px4.protocol import BENCHMARK_SCENARIOS as GATEWAY

    for name in pad_motion.SEGMENTED_CRUISE_SCENARIOS:
        assert name in pad_motion.BENCHMARK_SCENARIOS, name
        assert name in GATEWAY, name
    # They are open straights, so they must NOT be treated as closed tracks --
    # a closed track carries its phase across a reset instead of redrawing a
    # heading, which would bend the straight the envelope depends on.
    for name in pad_motion.SEGMENTED_CRUISE_SCENARIOS:
        assert name not in pad_motion.CLOSED_TRACK_SCENARIOS


def _planar_system():
    return load_config(ROOT / "config/shin2026-planar-system.yaml")


def _lane_trajectory():
    return pad_motion.PadTrajectory(
        pad_motion.PadMotionConfig.from_mapping(_planar_system()))


def test_every_pair_drives_one_fixed_lane_instead_of_a_drawn_heading():
    """The row sets off in the same direction, and keeps going straight.

    The open straight decks drew a fresh heading every episode, which is what
    made the episode sequence a 2-D random walk. Here the heading is a property
    of the profile, so every pair drives the same lane and the only freedom
    left is which way along it.
    """
    deck = _lane_trajectory()
    lane = float(deck.cfg.segmented_cruise_heading_rad)
    headings = set()
    clock = 0.0
    for episode in range(60):
        info = deck.reset(seed=4000 + episode, sim_time=clock, speed_scale=1.0,
                          scenario="segmented_cruise_fast")
        headings.add(round(float(info["heading_rad"]) % (2.0 * math.pi), 9))
        clock += 30.0
    assert headings <= {round(lane % (2.0 * math.pi), 9),
                        round((lane + math.pi) % (2.0 * math.pi), 9)}, headings
    # Both directions are actually used, or the "shuttle" is not one.
    assert len(headings) == 2


def test_a_lane_deck_never_leaves_its_lane():
    """Zero lateral motion is what makes the envelope's two constraints correct.

    A deck that drifted across its lane would make "hold zero lateral velocity
    and one yaw" wrong rather than merely restrictive, and the vehicle would
    walk off the track it cannot steer back onto.
    """
    deck = _lane_trajectory()
    clock = 0.0
    excursions = []
    for episode in range(40):
        deck.reset(seed=5000 + episode, sim_time=clock, speed_scale=1.0,
                   scenario="segmented_cruise_medium")
        for offset in np.linspace(0.0, 30.0, 16):
            position, velocity = deck.pose(clock + float(offset))
            excursions.append(abs(float(position[1])))
            # And the velocity is along the lane, both ways.
            assert abs(float(velocity[1])) < 1e-9
        clock += 30.0
    assert max(excursions) < 1e-6


def test_the_reversal_happens_between_episodes_and_never_inside_one():
    """Every episode is a straight line at a constant heading.

    A reversal inside an episode would be a manoeuvre the reduced envelope was
    built to exclude, and the vehicle -- which cannot yaw -- would be asked to
    follow a deck that turned round behind it.
    """
    deck = _lane_trajectory()
    clock = 0.0
    for episode in range(40):
        deck.reset(seed=6000 + episode, sim_time=clock, speed_scale=1.0,
                   scenario="segmented_cruise_fast")
        samples = [deck.pose(clock + float(offset))[1]
                   for offset in np.linspace(0.5, 30.0, 24)]
        signs = {float(np.sign(v[0])) for v in samples if abs(v[0]) > 1e-6}
        assert len(signs) <= 1, f"episode {episode} reversed mid-flight"
        clock += 30.0


def test_the_shuttle_bounds_the_sequence_without_the_arena_clamp_firing():
    """The clamp pins the deck with zero velocity, so it must stay a backstop.

    A pinned deck is a stationary target wearing a moving target's name. The
    bound has to come from the shuttle, and the arena has to sit clear of what
    the shuttle actually reaches.
    """
    system = _planar_system()
    cfg = pad_motion.PadMotionConfig.from_mapping(system)
    deck = pad_motion.PadTrajectory(cfg)
    clock = 0.0
    reach = []
    for episode in range(120):
        deck.reset(seed=7000 + episode, sim_time=clock, speed_scale=1.0,
                   scenario="segmented_cruise_fast")
        for offset in np.linspace(0.0, 30.0, 8):
            position, velocity = deck.pose(clock + float(offset))
            reach.append(abs(float(position[0])))
            # A clamped deck reports exactly zero velocity; the deck must be
            # moving for the whole sequence.
            assert np.linalg.norm(velocity) > 0.0
        clock += 30.0
    # Bounded by the leg plus one episode, and clear of the clamp.
    assert max(reach) <= cfg.segmented_cruise_leg_m + 40.0
    assert max(reach) < 0.75 * cfg.arena_radius_m


def test_the_row_is_across_the_lane_and_stays_separated():
    """Pairs spawn side by side and drive in parallel, so the gap is constant.

    The grid the retired profile used exists because its decks wander in
    arbitrary directions. These do not move laterally at all, so a single
    lateral spacing IS the permanent separation -- and it has to beat what a
    camera can see sideways at the highest entry altitude.
    """
    system = _planar_system()
    parallel = system["parallel"]
    offsets = np.asarray(parallel["pair_offsets_enu_m"], dtype=float)
    # One row entry per pair the runner will ever spawn; the world refuses to
    # start otherwise.
    from run_three_pipeline import MAX_PARALLEL_PAIRS

    assert offsets.shape[0] >= MAX_PARALLEL_PAIRS
    assert len(parallel["route_phase_fractions"]) >= MAX_PARALLEL_PAIRS
    lane = float(pad_motion.PadMotionConfig.from_mapping(
        system).segmented_cruise_heading_rad)
    along = np.array([math.cos(lane), math.sin(lane)])
    across = np.array([-math.sin(lane), math.cos(lane)])
    # Every pair starts on the same line across the lane: no along-lane stagger.
    assert np.allclose(offsets[:, :2] @ along, 0.0)
    assert np.allclose(offsets[:, 2], 0.0)
    lateral = offsets[:, :2] @ across
    gaps = np.diff(np.sort(lateral))
    assert gaps.min() > 0.0, "two pairs share a lane"
    # The camera's sideways reach is about the altitude itself for this
    # 90-degree lens, so at the highest entry altitude the separation has to
    # beat that by a wide factor. Ten times, measured against the profile's own
    # entry draw rather than against a number written here.
    entry_altitudes = ((system.get("benchmark") or {}).get(
        "initial_conditions") or {}).get("relative_altitude_m", (2.0, 8.0))
    assert gaps.min() >= 10.0 * float(max(entry_altitudes))
    # Laid out alternating outward from the origin, so the row is symmetric as
    # a whole and a four-pair run -- the usual case -- stays within two lanes
    # of the origin rather than trailing off to one side of it.
    assert abs(float(lateral.mean())) <= gaps.min()
    assert float(np.max(np.abs(lateral[:4]))) <= 2.0 * gaps.min()


def test_the_world_bound_clears_everything_the_row_can_reach():
    """The gateway pins the drone's own position target on this circle.

    It is measured about the WORLD origin, so it has to clear the lane the deck
    can drive plus the width of the row. Pinned there, the vehicle stops
    following a deck that is still driving, which reads as a policy failure
    rather than as the configuration error it is.
    """
    system = _planar_system()
    cfg = pad_motion.PadMotionConfig.from_mapping(system)
    offsets = np.asarray(system["parallel"]["pair_offsets_enu_m"], dtype=float)
    row = float(np.max(np.linalg.norm(offsets[:, :2], axis=1)))
    landing = system["landing"]
    assert float(landing["world_radius_m"]) > cfg.arena_radius_m + row
    # The pad-relative stray bound is a different quantity and is untouched.
    assert float(landing["world_xy_limit_m"]) == 200.0


def test_the_headline_experiment_flies_and_scores_the_three_decks():
    config = load_experiment(default_experiment_config())
    scenarios = set(pad_motion.SEGMENTED_CRUISE_SCENARIOS)
    assert set(config["training"]["scenarios"]) == scenarios
    assert set(config["behavior_cloning"]["scenarios"]) == scenarios
    flown = {name for name, count in config["evaluation"].items() if int(count)}
    assert flown == scenarios
    # Equal budgets, so no deck dominates the aggregate.
    counts = {int(config["evaluation"][name]) for name in scenarios}
    assert len(counts) == 1


# ------------------------------------------------------------- the entry pose

def test_the_planar_profile_seeds_the_entry_on_the_deck_track():
    system = load_config(ROOT / "config/shin2026-planar-system.yaml")
    benchmark = system.get("benchmark") or {}
    assert benchmark.get("planar_entry") is True
    initial = benchmark.get("initial_conditions") or {}
    # The along-track stand-off is the remaining freedom; the lateral and
    # yaw draws are still declared because they are still MADE and discarded,
    # which is what keeps a paired seed sweep paired across entry modes.
    assert "relative_longitudinal_m" in initial
    assert "relative_lateral_y_m" in initial
    assert "platform_yaw_misalignment_deg" in initial


def test_the_planar_entry_puts_the_vehicle_on_the_line_and_aligned_with_it():
    """Read off the world's own reset code rather than re-implemented here."""
    source = (ROOT / "isaac_sim/landing_world.py").read_text(encoding="utf-8")
    assert "planar_entry" in source
    # The lateral component is set to zero in the deck's heading frame and the
    # yaw misalignment is not drawn into the pose.
    assert "body = np.array([float(body[0]) + float(along), 0.0," in source
    assert "rpy_deg = np.zeros(3)" in source
    # Every draw the full Table-I box makes is still made, so the seed is
    # consumed identically and a paired sweep stays paired.
    assert "_discarded_lateral = rng.uniform(*y_range)" in source
    assert "_discarded_yaw = rng.uniform(*yaw_range)" in source
