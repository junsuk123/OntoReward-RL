# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""The envelope refuses a descent while the pad is outside the approach cone.

Measured 2026-09-25 over 1302 training episodes of the two learned arms:
losing the pad from frame is a one-way door. Reacquisition events were ZERO in
every episode, the climb-while-blind fraction was zero and the
descend-while-blind fraction one, and of the 609 episodes whose mean
visible-keypoint fraction fell below 0.3 not one landed, against 4.5 % and
6.1 % for the episodes that kept it. The estimator followed rather than led --
0.6-0.7 m of position RMSE while the pad was in view, 2.9-5.6 m once it was
not -- so the collapse was geometric, not a PPO instability: entropy, KL and
the learning rate never moved.

This is an envelope constraint and belongs with the velocity, acceleration and
tilt limits: every arm flies under it, the reward is untouched, and the action
dimension is unchanged so stored checkpoints stay loadable.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from ontology_rgat.controllers import PlanarLongitudinalController   # noqa: E402

REFERENCE_SCALE = 0.65


def _controller(tolerance=0.80, **kw):
    return PlanarLongitudinalController(
        dt=0.1, curriculum_min_action_scale=1.0,
        approach_cone_tolerance=tolerance,
        approach_cone_flare_range_m=1.20,
        approach_cone_reference_scale=REFERENCE_SCALE, **kw)


def _scale_at(range_m):
    return REFERENCE_SCALE / float(range_m)


def _sink(controller, steps=12):
    """Command a hard descent and return the vertical setpoint it produced."""
    for _ in range(steps):
        command = controller.command(np.array([0.0, -1.0, 0.0]))
    return float(command.velocity_body_heading_m_s[2])


def test_a_descent_outside_the_cone_is_refused():
    controller = _controller()
    controller.observe_pad(bearing_tangent=1.4, raw_scale=_scale_at(6.0),
                           trustworthy=True)
    assert not controller.descent_permitted()
    assert _sink(controller) == pytest.approx(0.0), (
        "the vertical setpoint must not go negative outside the cone")


def test_a_descent_inside_the_cone_is_untouched():
    controller = _controller()
    controller.observe_pad(bearing_tangent=0.2, raw_scale=_scale_at(6.0),
                           trustworthy=True)
    assert controller.descent_permitted()
    assert _sink(controller) < -0.5


def test_a_blind_vehicle_may_not_descend_but_may_still_climb():
    """Zero reacquisitions in 1302 episodes: sinking blind never recovers."""
    controller = _controller()
    controller.observe_pad(bearing_tangent=0.0, raw_scale=_scale_at(6.0),
                           trustworthy=False)
    assert not controller.descent_permitted()
    assert _sink(controller) == pytest.approx(0.0)
    # The constraint is a floor, not a commanded climb: the policy keeps the
    # channel and can still go up if it has learned to.
    controller.reset()
    controller.observe_pad(0.0, _scale_at(6.0), trustworthy=False)
    for _ in range(12):
        command = controller.command(np.array([0.0, 1.0, 0.0]))
    assert float(command.velocity_body_heading_m_s[2]) > 0.5


def test_the_flare_is_exempt_or_nothing_could_land():
    """Inside the flare the pad fills a downward camera and leaves it."""
    controller = _controller()
    controller.observe_pad(bearing_tangent=1.4, raw_scale=_scale_at(0.8),
                           trustworthy=False)
    assert controller.descent_permitted()
    assert _sink(controller) < -0.5


def test_the_cone_is_off_unless_configured_and_never_touches_the_action_space():
    """A zero tolerance restores the envelope every retired run flew."""
    controller = _controller(tolerance=0.0)
    controller.observe_pad(bearing_tangent=9.0, raw_scale=_scale_at(50.0),
                           trustworthy=False)
    assert controller.descent_permitted()
    assert _sink(controller) < -0.5
    # Shape and limits are unchanged either way, so checkpoints stay loadable.
    for tolerance in (0.0, 0.80):
        envelope = _controller(tolerance=tolerance).envelope()
        assert envelope["action_dimension"] == 3
        assert envelope["approach_cone_tolerance"] == pytest.approx(tolerance)
