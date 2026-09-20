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
