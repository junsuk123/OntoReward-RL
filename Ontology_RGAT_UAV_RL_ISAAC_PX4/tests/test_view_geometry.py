import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from view_geometry import (  # noqa: E402
    angular_separation_deg, paired_view_pose, street_offset_enu)


def test_metasejong_start_view_matches_the_clear_authored_direction():
    # The first route segment points south. Behind it is north, while positive
    # across is east: this keeps the camera over the open approach rather than
    # fifteen metres west inside the trees seen in the failed live viewport.
    offset = street_offset_enu([-9.0, 3.5, 9.0], -0.5 * math.pi)

    assert offset == pytest.approx([3.5, 9.0, 9.0])


def test_pair_camera_keeps_drone_and_ugv_inside_a_40_degree_view():
    offset = street_offset_enu([-9.0, 3.5, 9.0], -0.5 * math.pi)
    deck = np.array([-65.0, 131.0, 16.896])

    # Exercise the normal 2.5 m entry hover and the 30 m terminal-separation
    # boundary. Zoom must keep both subjects framed in either condition.
    for separation in (2.5, 7.0, 15.0, 30.0):
        uav = deck + np.array([0.0, 0.0, separation])
        eye, target, zoom = paired_view_pose(uav, deck, offset, pair_span_m=7.0)
        assert target == pytest.approx(0.5 * (uav + deck))
        assert zoom == pytest.approx(max(1.0, separation / 7.0))
        assert angular_separation_deg(eye, uav, deck) < 40.0


def test_pair_camera_zoom_is_rotation_invariant_around_the_route():
    deck = np.array([-100.0, 100.0, 15.0])
    uav = deck + np.array([12.0, -5.0, 4.0])
    angles = []
    for heading in np.linspace(-math.pi, math.pi, 17):
        offset = street_offset_enu([-9.0, 3.5, 9.0], heading)
        eye, _, _ = paired_view_pose(uav, deck, offset, pair_span_m=7.0)
        angles.append(angular_separation_deg(eye, uav, deck))

    assert max(angles) < 40.0


def test_operator_camera_offset_is_a_true_vehicle_side_view():
    # No along-road component: the eye stays beside the UGV rather than far
    # behind the whole fleet. Rotating the route rotates the side view with it.
    for heading in np.linspace(-math.pi, math.pi, 9):
        offset = street_offset_enu([0.0, -12.0, 4.5], heading)
        along = np.array([math.cos(heading), math.sin(heading), 0.0])
        across = np.array([-along[1], along[0], 0.0])
        assert np.dot(offset, along) == pytest.approx(0.0, abs=1e-10)
        assert np.dot(offset, across) == pytest.approx(-12.0)
        assert offset[2] == pytest.approx(4.5)
