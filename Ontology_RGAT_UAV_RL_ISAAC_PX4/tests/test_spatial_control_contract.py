"""Offline spatial control/wire checks, not an Isaac flight or safety proof."""
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ontology_rgat.bridge import BridgeError, PX4Bridge
from ontology_rgat.controllers.spatial_controller import (
    SPATIAL_ACTION_CONTRACT, SpatialAccelerationController)
from ontology_rgat_px4.frames import enu_to_ned
from ontology_rgat_px4.protocol import (
    ProtocolError, decode, encode, validate_spatial_velocity_action)
from ontology_rgat_px4.ros2_gateway import _Px4GatewayNode, velocity_command_enu


def controller(**kwargs):
    control = SpatialAccelerationController(**kwargs)
    control.reset(own_velocity_enu_m_s=np.zeros(3), yaw_enu_rad=0.)
    return control


def test_three_inputs_are_actual_xyz_not_planar_tilt():
    command = controller().command([0.,1.,0.])
    np.testing.assert_allclose(command.acceleration_enu_m_s2, [0.,2.,0.])
    np.testing.assert_allclose(command.velocity_enu_m_s, [0.,.2,0.])
    assert command.derived_roll_pitch_rad[0] < 0
    vertical = controller().command([0.,0.,1.])
    assert vertical.velocity_enu_m_s[2] == pytest.approx(.1)
    assert vertical.derived_roll_pitch_rad == (0.,0.)
    assert "tilt_rad" not in vertical.wire_payload()


def test_reset_uses_measured_own_velocity_and_requires_explicit_handover():
    control = SpatialAccelerationController()
    with pytest.raises(RuntimeError, match="reset"):
        control.command([0.,0.,0.])
    control.reset(own_velocity_enu_m_s=[1.,2.,-.1], yaw_enu_rad=math.pi/2)
    np.testing.assert_allclose(control.command([0.,0.,0.]).velocity_enu_m_s, [1.,2.,-.1])
    with pytest.raises(ValueError, match="handover"):
        control.reset(own_velocity_enu_m_s=[100.,0.,0.], yaw_enu_rad=0.)


def test_joint_cone_thrust_and_velocity_limits_under_random_commands():
    control = controller(max_velocity=(1.,1.,.5), max_acceleration=(8.,8.,4.),
                         max_thrust_weight_ratio=1.1)
    rng = np.random.default_rng(83)
    previous = np.zeros(3)
    for _ in range(400):
        command = control.command(rng.uniform(-2.,2.,3))
        v, a = command.velocity_enu_m_s, command.acceleration_enu_m_s2
        np.testing.assert_allclose(v-previous, a*control.dt, atol=1e-12)
        assert np.all(np.abs(v) <= control.max_velocity+1e-12)
        assert np.linalg.norm(a[:2]) <= (9.80665+a[2])*math.tan(control.max_tilt)+1e-12
        assert command.thrust_weight_ratio <= 1.1+1e-12
        validate_spatial_velocity_action(command.wire_payload())
        previous = v.copy()


def test_world_axes_do_not_rotate_with_vehicle_heading():
    np.testing.assert_allclose(velocity_command_enu([1.,2.,3.,0.], math.pi/2, frame="enu"),
                               [1.,2.,3.])
    np.testing.assert_allclose(velocity_command_enu([1.,2.,3.,0.], math.pi/2), [-2.,1.,3.])
    np.testing.assert_allclose(enu_to_ned([1.,2.,3.]), [2.,1.,-3.])


def test_wire_type_contract_and_frame_are_explicit():
    command = controller().command([.1,.2,.3])
    msg = {"v":1, "seq":5, "type":"spatial_velocity_action", **command.wire_payload()}
    restored = decode(encode(msg))
    velocity, acceleration, yaw = validate_spatial_velocity_action(restored)
    np.testing.assert_allclose(acceleration, [.2,.4,.3])
    assert restored["action_contract"] == SPATIAL_ACTION_CONTRACT
    for change in ({"frame":"body_heading"}, {"action_contract":"planar"},
                   {"tilt_rad":0.}, {"command":[0.,0.,0.,0.]},
                   {"acceleration_enu_m_s2":[8.,8.,-4.]},
                   {"yaw_rad":float("nan")}, {"velocity_enu_m_s":[11.,0.,0.]}):
        with pytest.raises(ProtocolError):
            validate_spatial_velocity_action({**msg, **change})


def test_bridge_sends_distinct_message_without_a_real_socket():
    bridge = object.__new__(PX4Bridge)
    messages = []
    bridge.transact = lambda kind, payload, expected: messages.append((kind,payload)) or {"test":True}
    bridge.validate_state_with_estimator_grace = lambda state: state
    bridge.pace_to_control_period = lambda state: state
    command = controller().command([0.,1.,0.])
    assert bridge.step_spatial_velocity(command) == {"test":True}
    assert messages[0][0] == "spatial_velocity_action"
    validate_spatial_velocity_action(messages[0][1])
    with pytest.raises(BridgeError, match="typed SpatialCommand"):
        bridge.step_spatial_velocity([0.,1.,0.])
    with pytest.raises(BridgeError):
        bridge.step_spatial_velocity(replace(command, acceleration_enu_m_s2=np.array([8.,8.,0.])))
    assert len(messages) == 1


def node_without_ros(target="sitl"):
    # Instantiate the actual nested NodeImpl without __init__ (which owns ROS
    # sockets/timers). Execute its real receive/publish methods with fake IO.
    class SkipInit(type):
        def __call__(cls, *args, **kwargs):
            return object.__new__(cls)
    class Node(metaclass=SkipInit):
        pass
    cfg = SimpleNamespace(target=target, control_hz=50., max_altitude_m=25., world_radius_m=140.)
    node = _Px4GatewayNode(cfg, None, [Node]+[SimpleNamespace]*21)
    node.sample = SimpleNamespace(px4_time_us=1_000_000, extra={},
        quaternion_enu_flu_wxyz=[math.sqrt(.5),0.,0.,math.sqrt(.5)])
    node.spatial_sim_time_s = None
    node.last_command_seq = -1
    node.last_action_ns = 0
    node.command_interface = "attitude"
    node.velocity_position_target_enu = None
    node.px4_world_position = np.array([0.,0.,5.])
    node.deck_position_enu = np.zeros(3)
    node.world_from_px4 = np.zeros(3)
    node.udp = SimpleNamespace(peer=("127.0.0.1",12345))
    node.battery_armed = True
    node._publish_flight_state = lambda: None
    node._publish_offboard_mode = lambda **_: None
    node._timestamp_us = lambda: 1_000_000
    published = []
    node.trajectory_pub = SimpleNamespace(publish=published.append)
    return node, published


def test_actual_gateway_publish_preserves_xyz_feedforward_and_resets_on_legacy():
    node, published = node_without_ros()
    command = controller().command([.5,1.,-.5])
    node._on_udp({"type":"spatial_velocity_action", "seq":1, **command.wire_payload()})
    node._publish_velocity_setpoint()
    np.testing.assert_allclose(published[-1].velocity, enu_to_ned(command.velocity_enu_m_s))
    np.testing.assert_allclose(published[-1].acceleration, enu_to_ned(command.acceleration_enu_m_s2))
    assert node.sample.extra["velocity_command_frame"] == "enu"
    node._on_udp({"type":"velocity_action", "seq":2, "command":[1.,0.,0.,0.]})
    node._publish_velocity_setpoint()
    assert node.velocity_acceleration_enu is None
    assert all(math.isnan(v) for v in published[-1].acceleration)
    np.testing.assert_allclose(published[-1].velocity, [1.,0.,0.], atol=1e-12)


def test_hardware_gateway_refuses_experimental_spatial_route():
    node, published = node_without_ros("hardware")
    with pytest.raises(ProtocolError, match="SITL-only"):
        node._on_udp({"type":"spatial_velocity_action", "seq":1,
                      **controller().command([0.,0.,0.]).wire_payload()})
    assert not published and node.last_command_seq == -1


def test_spatial_position_integration_uses_physics_clock_not_px4_rebase():
    node, published = node_without_ros()
    node.spatial_sim_time_s = 10.
    command = controller().command([1.,0.,0.])
    node._on_udp({'type':'spatial_velocity_action','seq':1,**command.wire_payload()})
    node.spatial_sim_time_s = 10.02
    node.sample.px4_time_us = 100_000_000
    node._publish_velocity_setpoint()
    assert node._task_time_us() == 10_020_000
    np.testing.assert_allclose(node.velocity_position_target_enu, [.004,0.,5.])


def test_optical_enu_rotation_needs_only_camera_and_own_imu():
    from ontology_rgat_px4.frames import euler_zyx_to_quat_wxyz
    from ontology_rgat_px4.ros2_gateway import optical_board_pose_to_enu
    optical_q=euler_zyx_to_quat_wxyz(.02,-.03,.4)
    own_q=euler_zyx_to_quat_wxyz(.02,-.03,.4+math.pi/2)
    position, attitude=optical_board_pose_to_enu([1.,0.,2.], optical_q, own_q)
    np.testing.assert_allclose(position,[0.,1.,2.],atol=1e-12)
    np.testing.assert_allclose(attitude,own_q,atol=1e-12)


def test_contact_snapshot_rejects_old_reset_and_nonfinite_event():
    import json
    node,_=node_without_ros();node.spatial_episode_seed=7;node.spatial_reset_seq=4
    node.spatial_episode_id='new-4'
    event={'source':'physics-first-contact','seed':7,'reset_seq':3,'episode_id':'new-4',
           'sample_time_s':1.,'contact_time_s':1.004,'relative_position':[0,0,.12],
           'relative_velocity':[0,0,-.15],'roll_pitch':[0,0],'angular_rate':[0,0,0]}
    node._on_spatial_contact_event(SimpleNamespace(data=json.dumps(event)))
    assert 'truth_contact_event' not in node.sample.extra
    event['reset_seq']=4
    node._on_spatial_contact_event(SimpleNamespace(data=json.dumps(event)))
    assert node.sample.extra['truth_contact_event']==event
    event['angular_rate']=[float('nan'),0,0]
    node._on_spatial_contact_event(SimpleNamespace(data=json.dumps(event)))
    assert node.sample.extra['truth_contact_event']['angular_rate']==[0,0,0]
    node.sample.extra.clear()
    event.update(angular_rate=[0,0,0],episode_id='previous-client-4')
    node._on_spatial_contact_event(SimpleNamespace(data=json.dumps(event)))
    assert 'truth_contact_event' not in node.sample.extra


def test_spatial_profile_is_not_a_live_training_configuration():
    from ontology_rgat.benchmarks.experiment import load_experiment
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((root/"config/control/spatial_acceleration_v1.yaml").read_text())
    assert profile["execution_status"] == "control_boundary_only"
    SpatialAccelerationController.from_mapping(profile["control"])
    with pytest.raises(ValueError, match="no compatible live training backend"):
        load_experiment(root/"config/control/spatial_acceleration_v1.yaml")
