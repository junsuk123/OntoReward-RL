"""The low-reserve episodes the warm-start teacher currently cannot land.

The pack is drawn per episode and the draw is wide. Measured over 11 teacher
flights on 2026-09-19: five began with a reduced pack and the teacher landed
none of them, three of those while already inside the 0.35 m position
criterion. It was aligned and simply ran out of charge on the way down, so the
demonstrations contain no successful low-reserve landing at all -- which is
the failure ``expert.expert_action`` documents and guards against.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.perception.semantic_observation import (  # noqa: E402
    SemanticObservation)
import run_three_pipeline                                     # noqa: E402
from run_three_pipeline import (                              # noqa: E402
    _privileged_velocity_teacher_action)


LIMIT = np.array([2.0, 2.0, 1.0])


def semantic(battery_risk):
    """A sample that is visually healthy, so only the energy term moves."""
    return SemanticObservation(
        keypoint_confidence=0.9, visible_keypoint_fraction=1.0,
        image_alignment=0.9, apparent_target_scale=0.4,
        image_plane_motion_safety=0.9, scale_rate_safety=0.9,
        visibility_memory=1.0, reacquisition_trend=0.5,
        vertical_motion_safety=0.9, attitude_stability=0.9,
        battery_risk=float(battery_risk), visual_loss_risk=0.0,
        centroid_xy=(0.0, 0.0), raw_scale=0.3)


def descent(altitude, *, battery_risk, energy_urgency):
    """Commanded descent rate, centred over the deck and holding station."""
    action = _privileged_velocity_teacher_action(
        np.array([0.0, 0.0, -float(altitude), 0.0, 0.0, 0.0]),
        np.zeros(3), semantic(battery_risk), LIMIT,
        energy_urgency=energy_urgency, noise_std=0.0)
    return -float(action[2]) * LIMIT[2]


@pytest.mark.parametrize("altitude", [3.0, 0.9, 0.3])
def test_the_teacher_is_unchanged_while_the_feature_is_off(altitude):
    """It ships off, so enabling it is a visible config edit and not a default."""
    assert descent(altitude, battery_risk=1.0, energy_urgency=0.0) == pytest.approx(
        descent(altitude, battery_risk=0.0, energy_urgency=0.0))


@pytest.mark.parametrize("altitude", [3.0, 0.9, 0.3])
def test_a_thin_pack_buys_descent_rate(altitude):
    nominal = descent(altitude, battery_risk=0.0, energy_urgency=1.0)
    urgent = descent(altitude, battery_risk=1.0, energy_urgency=1.0)
    assert urgent > nominal, "a nearly empty pack has to come down faster"


def test_urgency_never_asks_for_a_touchdown_the_criteria_would_reject():
    """A landing that arrives too fast to pass the gate is not a demonstration."""
    from ontology_rgat.config import default_config

    limit = float(default_config().criteria["vz"])
    for altitude in (4.0, 2.0, 1.0, 0.5, 0.2):
        for risk in (0.0, 0.5, 1.0):
            rate = descent(altitude, battery_risk=risk, energy_urgency=4.0)
            assert rate < limit, f"{rate:.3f} m/s at {altitude} m exceeds {limit}"


def test_urgency_does_not_turn_a_hold_or_a_climb_into_a_dive():
    """Only an existing descent is compressed; the safety branches are intact."""
    # Laterally displaced: the teacher holds level until it is over the deck.
    holding = _privileged_velocity_teacher_action(
        np.array([1.2, 0.0, -2.0, 0.0, 0.0, 0.0]), np.zeros(3),
        semantic(1.0), LIMIT, energy_urgency=4.0, noise_std=0.0)
    assert holding[2] == pytest.approx(0.0)

    # Visually lost and high: it climbs to reacquire, however thin the pack.
    lost = SemanticObservation(
        keypoint_confidence=0.1, visible_keypoint_fraction=0.0,
        image_alignment=0.0, apparent_target_scale=0.05,
        image_plane_motion_safety=0.5, scale_rate_safety=0.5,
        visibility_memory=0.0, reacquisition_trend=0.0,
        vertical_motion_safety=0.5, attitude_stability=0.9,
        battery_risk=1.0, visual_loss_risk=1.0,
        centroid_xy=(0.0, 0.0), raw_scale=0.0)
    climbing = _privileged_velocity_teacher_action(
        np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]), np.zeros(3), lost, LIMIT,
        energy_urgency=4.0, noise_std=0.0)
    assert climbing[2] > 0.0


def test_urgency_only_compresses_a_descent_that_is_already_aligned():
    """Coming down faster while off-centre is how the deck leaves the frame.

    The bearing to a pad that is not directly below grows as the range
    shrinks, so an accelerated descent from a marginal position drives the
    marker out of view and hands the vehicle to the climb branch. Measured:
    geometric FOV loss ran 0.284 over the first four flights with urgency
    ungated against 0.124 over the 24 before it, and a seed that had landed
    twice at 0.12 m timed out at 0.74 m instead.
    """
    def rate(lateral):
        action = _privileged_velocity_teacher_action(
            np.array([float(lateral), 0.0, -2.0, 0.0, 0.0, 0.0]), np.zeros(3),
            semantic(0.6), LIMIT, energy_urgency=1.0, noise_std=0.0)
        return -float(action[2]) * LIMIT[2]

    nominal = 0.35
    assert rate(0.05) > nominal, "a centred vehicle low on charge hurries"
    assert rate(0.40) == pytest.approx(nominal), (
        "a marginal one keeps the descent it can actually track")


def test_the_gate_still_clears_the_episodes_the_feature_exists_for():
    """The three near-misses it was added to recover, by their own numbers."""
    for lateral in (0.084, 0.192, 0.222):
        action = _privileged_velocity_teacher_action(
            np.array([lateral, 0.0, -2.0, 0.0, 0.0, 0.0]), np.zeros(3),
            semantic(0.5), LIMIT, energy_urgency=1.0, noise_std=0.0)
        assert -float(action[2]) * LIMIT[2] > 0.35, (
            f"{lateral} m was inside the position criterion and must hurry")


def test_the_reacquisition_climb_stops_at_its_ceiling():
    """A lost deck must not turn the climb into a runaway.

    Chasing a deck that drives loses the pad from a 60-degree down camera
    repeatedly. Unbounded, the 0.22 m/s climb spent whole flights going up:
    2026-09-21 two of eight traced route-deck flights ended at 6.5 m and
    12.0 m on 35% and 69% climb steps, both with an empty pack.
    """
    lost = SemanticObservation(
        keypoint_confidence=0.1, visible_keypoint_fraction=0.0,
        image_alignment=0.0, apparent_target_scale=0.05,
        image_plane_motion_safety=0.5, scale_rate_safety=0.5,
        visibility_memory=0.0, reacquisition_trend=0.0,
        vertical_motion_safety=0.5, attitude_stability=0.9,
        battery_risk=0.3, visual_loss_risk=1.0,
        centroid_xy=(0.0, 0.0), raw_scale=0.0)

    def climb_at(altitude, ceiling):
        action = _privileged_velocity_teacher_action(
            np.array([1.2, 0.0, -float(altitude), 0.0, 0.0, 0.0]),
            np.zeros(3), lost, LIMIT, visual_loss_climb_source="geometric",
            geometric_in_fov=False, climb_ceiling_m=ceiling, noise_std=0.0)
        return float(action[2]) * float(LIMIT[2])

    # Below the ceiling it still climbs, above it holds altitude and keeps
    # chasing; a ceiling of zero is the unbounded behaviour this started as.
    assert climb_at(4.0, 8.0) == pytest.approx(0.22, abs=1e-6)
    assert climb_at(7.9, 8.0) == pytest.approx(0.22, abs=1e-6)
    assert climb_at(8.0, 8.0) == pytest.approx(0.0, abs=1e-6)
    assert climb_at(12.0, 8.0) == pytest.approx(0.0, abs=1e-6)
    assert climb_at(12.0, 0.0) == pytest.approx(0.22, abs=1e-6)


def _teacher_state(altitude, lateral, *, relative_speed=0.0):
    """Relative truth for a vehicle ``lateral`` m off and ``altitude`` m up."""
    return np.array([lateral, 0.0, -float(altitude), float(relative_speed), 0.0, 0.0])


def test_the_hold_threshold_is_the_landing_gate_not_a_tuning_constant():
    """The teacher must hold against the criterion the run actually scores.

    Descending while the deck moves faster than criteria.rel_speed_xy under
    the vehicle buys a contact the landing gate marks unsafe -- which is how
    straight_8mps scored unsafe_pad_contact on 2026-09-22, three millimetres
    outside the position gate with every other gate passed. The value is
    therefore read from the criteria rather than written in the teacher.
    """
    import inspect

    from ontology_rgat.config import default_config

    signature = inspect.signature(_privileged_velocity_teacher_action)
    default = signature.parameters["contact_relative_speed_m_s"].default
    assert default == default_config().criteria.rel_speed_xy

    # The binder passes the live criterion, so a profile that changes the gate
    # moves the teacher with it.
    source = inspect.getsource(run_three_pipeline._privileged_velocity_teacher)
    source += inspect.getsource(
        run_three_pipeline._PrivilegedVelocityTeacher.__call__)
    assert "environment.cfg.criteria.rel_speed_xy" in source

    # Above the criterion the descent is held; below it the ladder runs.
    quiet = _privileged_velocity_teacher_action(
        _teacher_state(0.50, 0.20, relative_speed=0.20), np.zeros(3),
        semantic(0.2), LIMIT, contact_relative_speed_m_s=0.45, noise_std=0.0)
    busy = _privileged_velocity_teacher_action(
        _teacher_state(0.50, 0.20, relative_speed=0.60), np.zeros(3),
        semantic(0.2), LIMIT, contact_relative_speed_m_s=0.45, noise_std=0.0)
    assert quiet[2] < 0.0, "a quiet deck under an aligned vehicle descends"
    assert busy[2] == pytest.approx(0.0), "a deck above the gate holds"


def test_a_reacquired_deck_gives_back_the_altitude_the_climb_bought():
    """The climb must not ratchet: seeing the deck again undoes it.

    While the vehicle is outside pd_approach_lateral_m only two branches can
    fire -- climb when the pad is out of frame, hold when it is in -- so every
    loss adds altitude and nothing removes it. Measured 2026-09-22: a circle
    flight ratcheted 3.20 -> 5.09 m over eleven cycles and parked at the climb
    ceiling while its pack drained. On the escape-burst deck the pad leaving
    the frame is the whole scenario, so this is the ordinary case.
    """
    far = _teacher_state(4.0, 2.0, relative_speed=0.60)
    recovery = dict(recovery_descent_m_s=0.20, recovery_altitude_m=2.0,
                    approach_lateral_m=1.0, climb_altitude_m=3.0,
                    climb_lateral_m=1.0, visual_loss_climb_source="geometric")

    def vz(in_fov, state=far, **overrides):
        action = _privileged_velocity_teacher_action(
            state, np.zeros(3), semantic(0.2), LIMIT, geometric_in_fov=in_fov,
            noise_std=0.0, **dict(recovery, **overrides))
        return float(action[2]) * float(LIMIT[2])

    # Out of frame and high: still climbs to reacquire.
    assert vz(False) == pytest.approx(0.22, abs=1e-6)
    # Back in frame, too far to run the approach descent: comes back down
    # instead of holding the altitude the climb bought.
    assert vz(True) == pytest.approx(-0.20, abs=1e-6)
    # Not below the recovery band, and never when it is off.
    assert vz(True, state=_teacher_state(1.5, 2.0, relative_speed=0.60)) == \
        pytest.approx(0.0, abs=1e-6)
    assert vz(True, recovery_descent_m_s=0.0) == pytest.approx(0.0, abs=1e-6)
