import numpy as np
import pytest

from ontology_rgat_px4.frames import (
    enu_to_ned,
    geodetic_to_enu,
    euler_zyx_to_quat_wxyz,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    quat_enu_flu_to_ned_frd,
    quat_ned_frd_to_enu_flu,
    quat_wxyz_to_matrix,
    yaw_enu_to_ned,
    yaw_from_quat_wxyz,
)


def test_vector_frame_roundtrips():
    vector = np.array([1.3, -2.2, 4.1])
    np.testing.assert_allclose(ned_to_enu(enu_to_ned(vector)), vector)
    np.testing.assert_allclose(frd_to_flu(flu_to_frd(vector)), vector)


def test_quaternion_frame_roundtrip_by_rotation_matrix():
    for rpy in ((0.0, 0.0, 0.0), (0.2, -0.3, 1.1), (-1.0, 0.4, -2.2)):
        q_enu = euler_zyx_to_quat_wxyz(*rpy)
        q_back = quat_ned_frd_to_enu_flu(quat_enu_flu_to_ned_frd(q_enu))
        np.testing.assert_allclose(quat_wxyz_to_matrix(q_back), quat_wxyz_to_matrix(q_enu), atol=1e-12)


def test_zero_enu_attitude_has_expected_ned_heading():
    q_ned = quat_enu_flu_to_ned_frd((1.0, 0.0, 0.0, 0.0))
    expected = euler_zyx_to_quat_wxyz(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(quat_wxyz_to_matrix(q_ned), quat_wxyz_to_matrix(expected), atol=1e-12)



def test_scalar_yaw_conversion_matches_the_quaternion_path():
    for yaw_enu in (0.0, 0.3, -1.2, 2.9, -3.0):
        q_ned = quat_enu_flu_to_ned_frd(euler_zyx_to_quat_wxyz(0.0, 0.0, yaw_enu))
        assert np.isclose(yaw_enu_to_ned(yaw_enu), yaw_from_quat_wxyz(q_ned))


def test_the_gateway_and_the_city_project_identically():
    """Two packages, one tangent plane.

    ``isaac_sim/osm_city.py`` lays the city out by projecting OpenStreetMap
    lat/lon into world ENU; the gateway reads PX4's own origin back with
    ``frames.geodetic_to_enu``. They are deployed separately -- the gateway
    runs from an ASCII mirror and cannot import the simulator -- so the formula
    is written twice on purpose. If the two ever disagree, the deck is
    reconciled into the wrong place and the drone flies to the wrong street.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac_sim"))
    from osm_city import project_to_local

    lat0, lon0 = 37.5636, 126.9850
    for lat, lon in ((37.5650, 126.9861), (37.5601, 126.9840), (lat0, lon0)):
        theirs = project_to_local([lat], [lon], lat0, lon0)[0]
        ours = geodetic_to_enu(lat, lon, lat0, lon0)
        assert ours[:2] == pytest.approx(theirs, abs=1e-6)


def test_a_local_frame_offset_is_what_sent_the_drone_sideways():
    """The regression this exists to stop.

    PX4 pins its local frame at the spawn point; the deck is broadcast in the
    world frame the city is laid out in. With the shipped route the two are
    tens of metres apart, so publishing a world-frame target as a local
    setpoint aims the vehicle a block away.
    """
    lat0, lon0 = 37.5636, 126.9850
    # PX4's EKF origin, as it reported it in the flight log that exposed this.
    offset = geodetic_to_enu(37.56331, 126.98462, lat0, lon0)

    assert np.linalg.norm(offset[:2]) > 30.0
    # Round trip: a world target brought into the local frame and back is the
    # target again, which is the property both call sites rely on.
    target_world = np.array([12.0, -8.0, 6.0])
    local = target_world - offset
    assert local + offset == pytest.approx(target_world)


def test_the_frame_offset_carries_altitude():
    """Up is the axis a plan view never shows, and the one that broke this.

    PX4 pins its local frame at the spawn point, which here is the roof of a
    lorry three metres off the road. With the vertical left at zero every
    world-frame altitude came out that much low, the touchdown test fired while
    the vehicle was still in the air, and the resulting nonsense looked like a
    perception error rather than a frame error.
    """
    lat0, lon0, alt0 = 37.5636, 126.9850, 25.0
    # Same lat/lon, three metres up: a pure vertical offset and nothing else.
    offset = geodetic_to_enu(lat0, lon0, lat0, lon0, alt0 + 3.29, alt0)

    assert offset[0] == pytest.approx(0.0, abs=1e-9)
    assert offset[1] == pytest.approx(0.0, abs=1e-9)
    assert offset[2] == pytest.approx(3.29)


def test_altitude_defaults_leave_the_vertical_alone():
    """A caller that has no altitude must not silently invent one."""
    offset = geodetic_to_enu(37.5650, 126.9861, 37.5636, 126.9850)
    assert offset[2] == pytest.approx(0.0)
