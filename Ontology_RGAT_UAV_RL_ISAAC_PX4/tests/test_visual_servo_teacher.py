"""The estimator-free image servo that warm starts both arms.

The controller's signs are the risk here: the landing camera is mounted
rotated and pitched, so "which way is the pad" in the image is not something
to assert from intuition. Every directional test below takes its ground truth
from ``pad_image_position`` -- the same projection ``pad_view_margin`` uses to
decide whether an episode may start -- so a sign error in the servo shows up
here rather than after six hours of flying.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.initialization import (            # noqa: E402
    camera_centered_hover_offset, nadir_image_setpoint, pad_image_position)
from ontology_rgat.perception.semantic_observation import (  # noqa: E402
    SemanticObservation)
from run_three_pipeline import _visual_servo_teacher_action  # noqa: E402


LEVEL = (1.0, 0.0, 0.0, 0.0)
LIMIT = np.array([2.0, 2.0, 1.0])
SETPOINT = nadir_image_setpoint()


def observation(centroid, *, scale=0.25, visible=1.0, loss=0.0):
    """A semantic sample carrying only what the servo is allowed to read."""
    return SemanticObservation(
        keypoint_confidence=0.9, visible_keypoint_fraction=float(visible),
        image_alignment=0.5, apparent_target_scale=min(1.0, float(scale)),
        image_plane_motion_safety=0.9, scale_rate_safety=0.9,
        visibility_memory=1.0, reacquisition_trend=0.5,
        vertical_motion_safety=0.9, attitude_stability=0.9, battery_risk=0.1,
        visual_loss_risk=float(loss),
        centroid_xy=tuple(float(v) for v in centroid), raw_scale=float(scale))


def _when_the_servo_is_the_configured_teacher(test):
    """Skip a servo-tuning invariant while a different teacher is selected.

    The gains still have to be self-consistent when the servo is switched back
    on, so these stay executable rather than being deleted with it.
    """
    import functools

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        from config_loader import load_config
        from run_three_pipeline import (
            VISUAL_SERVO_TEACHER, behavior_cloning_settings)

        from conftest import default_experiment_config

        # The experiment anyone actually runs, not a file named here: the
        # headline config moved on 2026-09-22 and a gate reading the old one
        # decides these tests against a configuration nobody flies.
        settings = behavior_cloning_settings(
            load_config(str(default_experiment_config())))
        if settings.get("teacher") != VISUAL_SERVO_TEACHER:
            pytest.skip(f"teacher is {settings.get('teacher')}")
        return test(*args, **kwargs)
    return wrapper


def command(centroid, previous=None, *, integral=(0.0, 0.0),
            committed_already=False, **kwargs):
    action, carried, *_ = _visual_servo_teacher_action(
        observation(centroid, **kwargs), previous, LIMIT,
        setpoint=SETPOINT, dt=0.1, integral=np.asarray(integral, dtype=float),
        committed_already=committed_already, noise_std=0.0)
    return action, carried


@pytest.mark.parametrize("offset_enu", [
    [-2.0, 0.0, 0.0],     # behind the deck: the entry pose's own direction
    [2.0, 0.0, 0.0],      # overshot past it
    [0.0, 1.5, 0.0],      # deck to the vehicle's right
    [0.0, -1.5, 0.0],     # deck to the vehicle's left
    [-1.5, 1.0, 0.0],
    [1.2, -0.8, 0.0],
])
def test_the_servo_flies_toward_the_pad_the_projection_actually_shows(offset_enu):
    """The commanded direction must agree with the geometry, not with a guess.

    ``position`` is the UAV in the pad frame, so the pad lies at ``-position``
    horizontally. At identity attitude the body axes are the pad frame's, so
    the two are directly comparable.
    """
    position = camera_centered_hover_offset(5.0) + np.asarray(offset_enu, float)
    centroid = pad_image_position(position, LEVEL)
    assert centroid is not None and np.all(np.abs(centroid) <= 1.0), (
        "the test pose must keep the pad in frame")

    action, _ = command(centroid)
    commanded = action[:2] * LIMIT[:2]
    toward_pad = -np.asarray(position[:2], dtype=float)

    assert np.linalg.norm(commanded) > 1e-3, "a displaced vehicle must be commanded"
    cosine = float(np.dot(commanded, toward_pad)
                   / (np.linalg.norm(commanded) * np.linalg.norm(toward_pad)))
    # Carrying the error in tangent-of-angle units rather than frame fractions
    # is what buys the last of this: the raw frame is not square, so the same
    # metre reads 1.6x larger down the column axis than the row axis, and the
    # commanded bearing drifts off the true one. Measured over these poses the
    # worst case improves from 0.989 to 0.998.
    assert cosine > 0.99, f"commanded {commanded} does not point at the pad {toward_pad}"


def test_the_nadir_projection_is_the_servo_fixed_point_at_every_altitude():
    """Directly above the pad is where the horizontal command must vanish.

    A servo written against the image *centre* holds station behind the deck
    instead, and never lands. The distinction only exists because the camera
    is pitched, so it is worth pinning at both ends of the altitude range.
    """
    for altitude in (8.0, 5.0, 2.0, 0.7):
        centroid = pad_image_position([0.0, 0.0, altitude], LEVEL)
        action, _ = command(centroid)
        assert np.linalg.norm(action[:2]) < 1e-9, (
            f"a vehicle directly above at {altitude} m is already there")


def test_centring_the_pad_in_the_image_is_not_the_landing_point():
    """The naive servo's fixed point must still command a correction here."""
    centroid = pad_image_position(camera_centered_hover_offset(5.0), LEVEL)
    assert np.allclose(centroid, [0.0, 0.0], atol=1e-9)

    action, _ = command(centroid)
    # Camera-centred hover sits behind the deck, so the correction is forward.
    assert action[0] > 0.0
    assert abs(action[1]) < 1e-9


def test_a_driving_deck_does_not_escape_a_type_one_servo():
    """The failure the privileged teacher was written to avoid.

    A proportional-only image servo has to hold a standing error to generate
    any chase velocity, so a deck that keeps driving keeps pulling away. The
    integral term is what supplies that velocity at zero position error, so
    holding the vehicle at a fixed image error must build a lasting command
    rather than a constant one.
    """
    centroid = pad_image_position(
        camera_centered_hover_offset(5.0) + np.array([-1.0, 0.0, 0.0]), LEVEL)
    integral = np.zeros(2)
    first = None
    for _ in range(25):
        action, integral = command(centroid, integral=integral)
        if first is None:
            first = float(action[0])
    assert float(action[0]) > first > 0.0, "the chase velocity must keep building"

    # And once the error is removed the integrator still carries the deck's
    # velocity, instead of commanding zero and letting it drive away.
    settled, _ = command(SETPOINT, integral=integral)
    assert settled[0] > 0.0


def test_a_blind_vehicle_climbs_and_does_not_wind_the_integrator_up():
    """A centroid nobody can see is not evidence about where the deck is."""
    integral = np.zeros(2)
    for _ in range(20):
        action, integral = command(
            [0.9, 0.9], integral=integral, visible=0.0, loss=1.0, scale=0.15)
    assert np.allclose(integral, 0.0), "a blind servo must not integrate"
    assert action[2] > 0.0, "losing the deck while still high must climb"


def test_a_committed_flare_does_not_climb_when_the_marker_fills_the_frame():
    """The marker legitimately leaves a downward camera at touchdown.

    Treating that expected disappearance as a recovery event traps the vehicle
    centimetres above the deck, which is the trap the privileged teacher
    documents as well.
    """
    action, _ = command(SETPOINT, visible=0.0, loss=1.0, scale=0.95)
    assert action[2] <= 0.0


def test_the_cone_widens_with_range_so_it_never_beats_the_landing_criterion():
    """A cone fixed in angle keeps shrinking in metres all the way down.

    At 0.5 m of range 0.30 of it is 0.15 m, against a 0.35 m position gate --
    the servo would refuse a landing the criteria would have accepted.
    Widening with apparent scale, which goes as 1 / range, holds the metres it
    implies roughly constant over the last stretch.
    """
    offset = SETPOINT + np.array([0.0, 0.30 / 0.625])   # bearing 0.30

    # Both readings sit in the same rate band, so the only thing that differs
    # is how much of the cone the bearing uses up.
    far, _ = command(offset, scale=0.13)
    near, _ = command(offset, scale=0.18)
    assert -near[2] > -far[2], "the same bearing must be more acceptable closer in"


def test_descent_backs_off_beside_the_deck_and_stops_at_the_limits():
    aligned, _ = command(SETPOINT, scale=0.15)
    misaligned, _ = command(SETPOINT + np.array([0.5, 0.0]), scale=0.15)
    assert -aligned[2] > -misaligned[2] * 3.0, (
        "a centred vehicle must come down far faster than one beside the deck")

    extreme, _ = command([1.0, 1.0], scale=0.01)
    assert np.all(np.abs(extreme) <= 0.90 + 1e-9)


def test_the_servo_reads_nothing_privileged():
    """Structural, not stylistic: the label must be reproducible in deployment.

    ``SemanticObservation`` is the whole input. If the controller ever reaches
    for relative state it cannot come from here, and this call fails.
    """
    action, _ = command(pad_image_position([0.0, 1.0, 4.0], LEVEL))
    assert np.all(np.isfinite(action))

    with pytest.raises(AttributeError):
        observation([0.0, 0.0]).true_relative_state


def test_the_primary_experiment_actually_warms_both_arms_up():
    """Enabling this is the point; the block is easy to leave unread.

    ``behavior_cloning`` lived under ``seminar_fast`` while only the deadline
    profile used it, so a block written at the top level of the primary
    experiment silently did nothing. Both spellings must resolve, and the
    primary config must be the one that is on.
    """
    from config_loader import load_config
    from run_three_pipeline import (
        VISUAL_SERVO_TEACHER, behavior_cloning_settings)

    config = load_config(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    settings = behavior_cloning_settings(config)

    from run_three_pipeline import PRIVILEGED_VELOCITY_TEACHER

    assert settings.get("enabled") is True
    # Either teacher is a valid choice -- the servo is the better kind and the
    # PD is the one that lands today -- but it has to be one the runner knows,
    # or _prepare_fast_demonstrations refuses the run.
    assert settings["teacher"] in (
        VISUAL_SERVO_TEACHER, PRIVILEGED_VELOCITY_TEACHER)
    # Both arms are cloned from one shared teacher, so the warm start cannot
    # favour either side of the comparison.
    assert settings["ppo_anchor"]["enabled"] is True

    # The legacy location still resolves for the profiles that use it.
    legacy = behavior_cloning_settings(
        {"seminar_fast": {"behavior_cloning": {"enabled": True, "epochs": 3}}})
    assert legacy["epochs"] == 3
    assert behavior_cloning_settings({}) == {}


def test_the_demonstration_encoder_source_stays_estimator_free():
    """A teacher encoder with state estimation would store privileged inputs."""
    from config_loader import load_config
    from ontology_rgat.pipelines import get_pipeline
    from run_three_pipeline import behavior_cloning_settings

    config = load_config(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    source = behavior_cloning_settings(config)["source_pipeline"]
    assert not get_pipeline(source).state_estimation_enabled


def test_tangent_angle_units_aim_better_than_raw_frame_fractions():
    """Why the error is scaled by each axis's own half-angle tangent.

    The frame is not square: the column axis spans tan(45 deg) and the row
    axis tan(32 deg), so a raw frame fraction means 1.6x more offset down one
    axis than the other and the commanded bearing leans away from the true
    one. Scaling each axis back to ``offset / range`` removes that lean.

    Magnitude is deliberately not asserted: a pitched camera really does see
    a different angle per metre fore/aft than laterally, because fore/aft
    motion changes the range as well. Only the bearing has to be right.
    """
    improved = []
    for offset in ([-2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [-1.5, 1.0, 0.0],
                   [1.2, -0.8, 0.0]):
        position = camera_centered_hover_offset(5.0) + np.asarray(offset, float)
        centroid = pad_image_position(position, LEVEL)
        toward_pad = -np.asarray(position[:2], dtype=float)

        def bearing(tan_half):
            action, *_ = _visual_servo_teacher_action(
                observation(centroid), None, LIMIT, setpoint=SETPOINT, dt=0.1,
                integral=np.zeros(2), tan_half=tan_half, noise_std=0.0)
            commanded = action[:2] * LIMIT[:2]
            return float(np.dot(commanded, toward_pad)
                         / (np.linalg.norm(commanded) * np.linalg.norm(toward_pad)))

        angle_units = bearing((1.0, 0.625))
        frame_fractions = bearing((1.0, 1.0))
        assert angle_units >= frame_fractions - 1e-9
        improved.append(angle_units > frame_fractions + 1e-6)

    assert any(improved), "the correction must actually change the aim"


@pytest.mark.parametrize("lateral_m, altitude_m, apparent, floor, ceiling", [
    (2.00, 5.0, 0.05, 0.00, 0.10),   # too far out to commit to coming down
    (0.76, 5.0, 0.05, 0.15, 0.35),   # the error the first tuned flight held
    (0.30, 5.0, 0.05, 0.40, 0.51),   # centred: the full approach rate
    (0.30, 2.0, 0.12, 0.10, 0.25),   # mid band, still working the error off
    (0.10, 1.0, 0.25, 0.20, 0.30),   # committed: the flare rate, unconditional
])
def test_the_descent_rate_follows_how_much_cone_is_left(
        lateral_m, altitude_m, apparent, floor, ceiling):
    """Descending while only roughly aligned destabilises the servo.

    The bearing to a pad that is not directly below grows as the range
    shrinks, so a flat descent rate drives the deck out of frame and hands the
    vehicle to the climb branch. Flown at a flat 0.50 m/s that is exactly what
    happened: seed 90000 ended 8.6 m out and climbing, against 0.32 m and
    still descending from the same entry at 0.35 m/s.

    So the rate is proportional to the cone left, not switched on at its edge.
    """
    centroid = pad_image_position([0.0, lateral_m, altitude_m], LEVEL)
    action, _ = command(centroid, scale=apparent)
    rate = -action[2] * LIMIT[2]
    assert floor <= rate <= ceiling, f"{rate:.3f} m/s outside [{floor}, {ceiling}]"


def test_a_committed_flare_keeps_coming_down_whatever_the_bearing_says():
    """The last half metre is not servoable, and must not be waited out.

    The camera is mounted 0.16 m below the airframe, so below about half a
    metre the bearing to anything not exactly underneath blows up and the
    marker leaves the frame on its own. Flown open loop against the pre-commit
    rule the descent asymptoted to a halt at 0.37 m with the pad still in
    view. Past the commit the flare rate has to run regardless.
    """
    badly_off = SETPOINT + np.array([0.0, 0.95])   # near the frame edge

    hesitating, _ = command(badly_off, scale=0.15)
    committed, _ = command(badly_off, scale=0.95)
    assert -committed[2] * LIMIT[2] > -hesitating[2] * LIMIT[2]
    assert -committed[2] * LIMIT[2] == pytest.approx(0.25, abs=1e-6)


def test_the_descent_schedule_fits_inside_the_episode_horizon():
    """Arriving correctly after the horizon has expired is not a landing.

    Table I starts the vehicle between 2 m and 8 m above the deck and an
    episode is 30 simulated seconds. The schedule has to bring the worst case
    down with enough margin left for the alignment pass, and every rate has to
    stay inside the 0.55 m/s the touchdown criteria admit.
    """
    from ontology_rgat.config import default_config

    limit = float(default_config().criteria["vz"])
    rates = {}
    for label, apparent in (("approach", 0.05), ("descent", 0.30),
                            ("flare", 0.95)):
        action, _ = command(SETPOINT, scale=apparent)
        rates[label] = -action[2] * LIMIT[2]
        assert 0.0 < rates[label] < limit, f"{label} rate is outside the criteria"

    # 8 m is the worst Table-I draw. Apparent scale goes as 1 / range, and the
    # measured stack reads 0.05 at 5 m, so the schedule changes at about 1 m
    # and again at about 0.63 m.
    seconds = ((8.0 - 1.0) / rates["approach"]
               + (1.0 - 0.63) / rates["descent"]
               + 0.63 / rates["flare"])
    assert seconds < 20.0, (
        f"the descent alone needs {seconds:.0f} s of a 30 s episode")


@_when_the_servo_is_the_configured_teacher
def test_the_servo_outruns_the_deck_it_has_to_hold_station_on():
    """Saturation, which no amount of gain tuning can work around.

    The horizontal clamp has to cover the deck's own top speed *and* leave
    authority to close a position error. Set equal to it, the whole budget
    goes on matching the deck and the lateral error stops converging: four
    consecutive tunings measured 0.32-0.37 m of standing error against a
    0.35 m position criterion, from completely different vertical schedules.
    """
    from config_loader import load_config
    from run_three_pipeline import behavior_cloning_settings

    from ontology_rgat.initialization import curriculum_motion_scale

    system = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    settings = behavior_cloning_settings(
        load_config(ROOT / "config/experiments/two_pipeline_comparison.yaml"))

    # The deck the demonstration pass actually flies against, not the one the
    # profile tops out at: the teacher stage sets its own curriculum.
    deck_top_speed = (float(system["pad"]["speed_range_m_s"][1])
                      * curriculum_motion_scale(
                          float(settings["curriculum"]), 0.35))
    clamp = float(settings["horizontal_speed_limit_m_s"])

    spare = clamp - deck_top_speed
    assert spare >= 0.20, (
        f"{clamp} m/s leaves only {spare:.2f} m/s to close a position error "
        f"against a {deck_top_speed:.2f} m/s deck, so it will not converge")
    assert float(settings["integral_limit"]) >= deck_top_speed, (
        "the integrator must be able to hold the whole chase velocity")

    # And still inside the action envelope the episode is flown with.
    assert clamp <= 2.0


def test_the_commit_latches_because_apparent_scale_dies_with_the_marker():
    """Apparent scale is measured from the keypoints, so it cannot survive them.

    Tested fresh every step, the flare un-commits at exactly the moment it is
    needed: losing the deck drops apparent scale to zero, the commit test
    fails, and the climb branch fires. The sampled trajectory did precisely
    that -- down to 0.385 m, deck out of frame, back up to 0.498 m and away.
    """
    blind = dict(visible=0.0, loss=1.0, scale=0.0)

    unlatched, _ = command(SETPOINT, **blind)
    assert unlatched[2] > 0.0, "a blind vehicle that never committed climbs"

    latched, _ = command(SETPOINT, committed_already=True, **blind)
    assert latched[2] < 0.0, "a committed flare keeps coming down blind"


@_when_the_servo_is_the_configured_teacher
def test_the_integrator_can_actually_hold_the_deck_velocity():
    """gain x limit is the whole chase velocity the integrator can command.

    Set below the deck's own speed it can never cancel the motion, and the
    proportional term has to hold a standing error to make up the difference
    -- which is the type-0 behaviour the integral exists to remove.
    """
    from config_loader import load_config
    from ontology_rgat.initialization import curriculum_motion_scale
    from run_three_pipeline import behavior_cloning_settings

    system = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    settings = behavior_cloning_settings(
        load_config(ROOT / "config/experiments/two_pipeline_comparison.yaml"))

    deck = (float(system["pad"]["speed_range_m_s"][1])
            * curriculum_motion_scale(float(settings["curriculum"]), 0.35))
    authority = (float(settings["integral_gain"])
                 * float(settings["integral_limit"]))
    assert authority > deck, (
        f"the integrator tops out at {authority:.2f} m/s against a "
        f"{deck:.2f} m/s deck")


def _servo(centroid, previous=None, *, integral=(0.0, 0.0), image_rate=None,
           scale=0.25, visible=1.0, loss=0.0, **gains):
    """The raw call, for the tests that need the carried state back."""
    return _visual_servo_teacher_action(
        observation(centroid, scale=scale, visible=visible, loss=loss),
        previous, LIMIT, setpoint=SETPOINT, dt=0.1,
        integral=np.asarray(integral, dtype=float), image_rate=image_rate,
        noise_std=0.0, **gains)


def test_the_image_rate_is_filtered_rather_than_differenced():
    """A one-frame jump must not reach the damping term at full size.

    Most of the image motion this loop sees is its own airframe: a velocity
    command is delivered by tilting, and the camera is bolted on. Raw, the
    damping term commands against that tilt and tilts further -- the positive
    feedback the servo's 0-of-16 record came from.
    """
    settled = observation((SETPOINT[0], 0.0))
    jumped = (SETPOINT[0] + 0.20, 0.0)
    _action, _carried, _committed, raw = _servo(
        jumped, settled, rate_filter_s=0.0)
    _action, _carried, _committed, filtered = _servo(
        jumped, settled, image_rate=np.zeros(2), rate_filter_s=0.30)

    assert abs(raw[0]) > 1.0, "a 0.20 jump in 0.1 s is a large raw rate"
    # dt / (tau + dt) of it on the first step, and no more.
    assert filtered[0] == pytest.approx(raw[0] * 0.1 / 0.4, rel=1e-6)

    # It converges on a rate that persists, so real deck motion is still damped.
    carried_rate = filtered
    for _ in range(40):
        _a, _c, _m, carried_rate = _servo(
            jumped, settled, image_rate=carried_rate, rate_filter_s=0.30)
    assert carried_rate[0] == pytest.approx(raw[0], rel=0.05)


def test_the_filter_moves_the_commanded_velocity():
    """The filtered rate is what the damping term actually uses."""
    settled = observation((SETPOINT[0], 0.0))
    jumped = (SETPOINT[0] + 0.20, 0.0)
    raw_action, *_ = _servo(jumped, settled, rate_filter_s=0.0)
    filtered_action, *_ = _servo(jumped, settled, image_rate=np.zeros(2),
                                 rate_filter_s=0.30)
    assert not np.allclose(raw_action[:2], filtered_action[:2])
    # Damping opposes the jump, so filtering it leaves more of the correction.
    assert filtered_action[0] > raw_action[0]


def test_a_saturated_command_stops_the_integrator_growing():
    """Conditional integration, which is what keeps it off its clamp."""
    far = (SETPOINT[0] + 0.85, 0.35)
    wound = (2.5, 1.0)
    _action, held, *_ = _servo(far, integral=wound, anti_windup=True)
    _action, grown, *_ = _servo(far, integral=wound, anti_windup=False)

    assert np.allclose(held, wound), "a saturated command must hold the integral"
    assert np.linalg.norm(grown) > np.linalg.norm(wound)


def test_the_integrator_still_builds_when_the_command_is_unsaturated():
    """Anti-windup must not cost the type-1 behaviour the loop is built on."""
    gentle = (SETPOINT[0] + 0.05, 0.0)
    _action, carried, *_ = _servo(gentle, integral=(0.0, 0.0), anti_windup=True)
    assert np.linalg.norm(carried) > 0.0

    # And an error that unwinds a saturated command is still integrated: the
    # clamp only holds what would drive the command further in.
    opposed = (SETPOINT[0] - 0.40, -0.35)
    _action, unwound, *_ = _servo(opposed, integral=(2.5, 1.0), anti_windup=True)
    assert np.linalg.norm(unwound) < np.linalg.norm((2.5, 1.0))


def test_a_blind_frame_neither_integrates_nor_filters_stale_motion():
    """Nothing about the new state may resurrect the blind-frame rule."""
    lost = (SETPOINT[0] + 0.3, 0.2)
    _action, carried, *_ = _servo(lost, integral=(0.4, -0.2), visible=0.2)
    assert np.allclose(carried, (0.4, -0.2))
