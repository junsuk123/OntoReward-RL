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


def command(centroid, previous=None, *, integral=(0.0, 0.0), **kwargs):
    action, carried = _visual_servo_teacher_action(
        observation(centroid, **kwargs), previous, LIMIT,
        setpoint=SETPOINT, dt=0.1, integral=np.asarray(integral, dtype=float),
        noise_std=0.0)
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


def test_descent_waits_for_alignment_and_stops_at_the_limits():
    aligned, _ = command(SETPOINT, scale=0.15)
    assert aligned[2] < 0.0, "a centred distant target may descend"

    misaligned, _ = command(SETPOINT + np.array([0.5, 0.0]), scale=0.15)
    assert misaligned[2] == 0.0, "do not descend beside the deck"

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

    assert settings.get("enabled") is True
    assert settings["teacher"] == VISUAL_SERVO_TEACHER
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
            action, _ = _visual_servo_teacher_action(
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


@pytest.mark.parametrize("lateral_m, altitude_m, apparent, expected", [
    (1.65, 5.0, 0.05, 0.0),      # the tracking error the first flights held
    (0.76, 5.0, 0.05, -0.35),    # ... and the one they held on the good seeds
    (0.40, 2.0, 0.20, -0.35),
    (0.20, 1.0, 0.60, -0.14),
    (0.08, 0.5, 0.95, -0.04),
])
def test_the_descent_funnel_tightens_on_the_way_down(
        lateral_m, altitude_m, apparent, expected):
    """The regression that cost the first three teacher flights.

    Every one of them timed out at the 300-step horizon with the pad in view
    the whole way -- 0.0% and 1.7% geometric FOV loss -- and a vertical speed
    of 0.01 m/s. The servo was tracking correctly and simply never opened its
    own descent gate, because the alignment cone was narrower than the bearing
    the horizontal loop actually holds at 5 m.

    The cone is fixed in angle, so the distance it admits shrinks with range
    on its own: about 1.3 m at 5 m, 0.2 m at 1 m, without measuring altitude.
    """
    centroid = pad_image_position([0.0, lateral_m, altitude_m], LEVEL)
    action, _ = command(centroid, scale=apparent)
    assert action[2] * LIMIT[2] == pytest.approx(expected, abs=1e-6)
