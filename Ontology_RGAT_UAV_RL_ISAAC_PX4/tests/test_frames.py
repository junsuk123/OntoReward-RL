import numpy as np

from ontology_rgat_px4.frames import (
    enu_to_ned,
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
