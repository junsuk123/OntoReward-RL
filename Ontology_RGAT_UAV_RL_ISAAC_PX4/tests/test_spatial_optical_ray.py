import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ontology_rgat_px4.optical_geometry import imu_rotated_relative_position
from ontology_rgat_px4.ros2_gateway import optical_board_pose_to_enu


@pytest.mark.parametrize('own_euler', [[0,0,0],[.1,-.1,.8],[-.2,.3,-2.]])
@pytest.mark.parametrize('optical_euler', [[0,0,0],[.15,-.12,.5],[-.1,.2,1.]])
def test_relative_ray_conversion_cancels_pnp_attitude_position_coupling(own_euler,optical_euler):
    own, optical = Rotation.from_euler('xyz',own_euler),Rotation.from_euler('xyz',optical_euler)
    expected = np.array([.2,-.15,3.])
    measured_body_ray = own.inv().apply(expected)
    measured_board_position = optical.apply(measured_body_ray)
    got = imu_rotated_relative_position(measured_board_position,
        optical.as_quat()[[3,0,1,2]],own.as_quat()[[3,0,1,2]])
    np.testing.assert_allclose(got,expected,atol=1e-14)


def test_relative_ray_needs_no_board_pose_or_simulator_state():
    q = np.array([1.,0.,0.,0.])
    np.testing.assert_allclose(imu_rotated_relative_position([.2,.1,2.],q,-q),[.2,.1,2.])
    with pytest.raises(ValueError):
        imu_rotated_relative_position([np.nan,0,2.],q,q)


def test_gateway_position_correction_preserves_independent_attitude_gate():
    optical = Rotation.from_euler('xyz', [np.deg2rad(8), 0., .4])
    own = Rotation.identity()
    expected = np.array([.2, -.1, 3.])
    position, independent_quaternion = optical_board_pose_to_enu(
        optical.apply(expected), optical.as_quat()[[3,0,1,2]], [1.,0.,0.,0.])
    np.testing.assert_allclose(position, expected, atol=1e-14)
    independent = Rotation.from_quat(independent_quaternion[[1,2,3,0]])
    assert np.isclose((own.inv()*independent).magnitude(), np.deg2rad(8))
