"""Direct spatial actuator contract; numerical checks are not flight evidence."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ontology_rgat.controllers.spatial_controller import SpatialAccelerationController
from ontology_rgat.spatial.core import SpatialConfig, Estimator, observation, safety_status
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.dynamics import GRAVITY, advance_attitude_thrust
from ontology_rgat.two_axis.dynamics import PlanarState, step_planar
from ontology_rgat.two_axis.config import DynamicsConfig
from ontology_rgat_px4.protocol import validate_spatial_acceleration_action, ProtocolError
from ontology_rgat_px4.ros2_gateway import px4_direct_acceleration_input
from ontology_rgat.bridge import PX4Bridge, BridgeError


def direct():
    c = SpatialAccelerationController(max_acceleration=(2.5, 2.5, 2.), acceleration_only=True)
    c.reset(own_velocity_enu_m_s=[0., 0., 0.], yaw_enu_rad=0.)
    return c


def test_direct_command_has_no_hidden_integrated_velocity_reference():
    c = direct()
    a = c.command([.4, .2, -.1], own_velocity_enu_m_s=[.1, .2, -.1])
    for _ in range(10):
        b = c.command([.4, .2, -.1], own_velocity_enu_m_s=[.1, .2, -.1])
        np.testing.assert_array_equal(a.acceleration_enu_m_s2, b.acceleration_enu_m_s2)
        np.testing.assert_array_equal(a.velocity_enu_m_s, b.velocity_enu_m_s)
    assert 'velocity_enu_m_s' not in a.wire_payload()
    validate_spatial_acceleration_action(a.wire_payload())
    with pytest.raises(ValueError, match='current measured'):
        c.command([0., 0., 0.])


def test_direct_overspeed_braking_is_physically_bounded():
    c = direct()
    command = c.command([1., -1., 1.], own_velocity_enu_m_s=[20., -20., 10.])
    np.testing.assert_allclose(command.acceleration_enu_m_s2, [-2.5, 2.5, -2.])
    validate_spatial_acceleration_action(command.wire_payload())


@pytest.mark.parametrize('az', [-2., 0., 2.])
def test_px4_mapping_realizes_net_acceleration_not_vertical_cross_coupling(az):
    requested = np.array([1., -.7, az])
    internal = px4_direct_acceleration_input(requested)
    body_z = np.r_[internal[:2], GRAVITY]
    body_z /= np.linalg.norm(body_z)
    thrust = (GRAVITY+internal[2])/body_z[2]
    np.testing.assert_allclose(thrust*body_z-[0, 0, GRAVITY], requested, atol=1e-12)


def test_direct_protocol_rejects_mixed_control_contracts():
    message = direct().command([0., 0., 0.], own_velocity_enu_m_s=[0., 0., 0.]).wire_payload()
    for change in ({'velocity_enu_m_s':[0,0,0]}, {'position_enu_m':[0,0,1]},
                   {'action_contract':'spatial-enu-net-acceleration-v1'}, {'frame':'ned'},
                   {'tilt_rad':0.}, {'acceleration_enu_m_s2':[math.nan,0,0]}):
        with pytest.raises(ProtocolError):
            validate_spatial_acceleration_action(dict(message, **change))


def test_bridge_cannot_silently_switch_between_direct_and_velocity_routes():
    bridge = object.__new__(PX4Bridge)
    sent = []
    bridge.transact = lambda kind, payload, expected: sent.append((kind,payload)) or {}
    bridge.validate_state_with_estimator_grace = lambda s:s
    bridge.pace_to_control_period = lambda s:s
    command = direct().command([0,0,0], own_velocity_enu_m_s=[0,0,0])
    bridge.step_spatial_acceleration(command)
    assert sent[0][0] == 'spatial_acceleration_action'
    with pytest.raises(BridgeError, match='actuation mode'):
        bridge.step_spatial_velocity(command)
    assert len(sent) == 1


def test_actual_gateway_direct_publish_has_no_position_or_velocity_pid_target():
    from test_spatial_control_contract import node_without_ros
    from ontology_rgat_px4.frames import enu_to_ned
    node, published = node_without_ros()
    modes = []
    node._publish_offboard_mode = lambda **flags: modes.append(flags)
    command = direct().command([.4,.2,-.5], own_velocity_enu_m_s=[0,0,0])
    node._on_udp(dict(type='spatial_acceleration_action', seq=1, **command.wire_payload()))
    node._publish_spatial_acceleration_setpoint()
    assert node.command_interface == 'spatial_acceleration'
    assert modes[-1] == {'acceleration':True}
    assert all(math.isnan(v) for v in published[-1].position+published[-1].velocity)
    assert node.velocity_position_target_enu is None
    np.testing.assert_allclose(published[-1].acceleration,
        enu_to_ned(px4_direct_acceleration_input(command.acceleration_enu_m_s2)))
    hardware, _ = node_without_ros('hardware')
    with pytest.raises(ProtocolError, match='SITL-only'):
        hardware._on_udp(dict(type='spatial_acceleration_action', seq=1, **command.wire_payload()))


def test_direct_spatial_plant_matches_reference_on_planar_invariant_subspace():
    cfg = DynamicsConfig(gravity_m_s2=GRAVITY, ax_max_m_s2=2.5, az_max_m_s2=2.,
        pitch_limit_rad=math.radians(25), max_thrust_weight_ratio=1.5,
        clip_actual_pitch=False, thrust_integrator='euler')
    state = PlanarState(0., 2., 0., 0., 0., 0., cfg.mass_kg*GRAVITY)
    angles, rates, thrust = np.zeros(3), np.zeros(3), GRAVITY
    position, velocity = np.array([0.,0.,2.]), np.zeros(3)
    c = direct()
    for _ in range(100):
        command = c.command([.3,0.,-.2], own_velocity_enu_m_s=velocity)
        angles, rates, thrust, acceleration, body_rates = advance_attitude_thrust(
            angles, rates, thrust, command, .01)
        position += velocity*.01 + .5*acceleration*.01**2
        velocity += acceleration*.01
        state = step_planar(state, command.acceleration_enu_m_s2[[0,2]], .01, cfg)
        np.testing.assert_allclose([position[0],position[2],velocity[0],velocity[2],
            angles[1],body_rates[1],thrust], [state.x_m,state.z_m,state.vx_m_s,
            state.vz_m_s,state.theta_rad,state.pitch_rate_rad_s,state.thrust_n/cfg.mass_kg], atol=1e-12)


def test_v6_packet_prediction_is_causal_and_versioned():
    cfg = replace(SpatialConfig(), schema='spatial-causal-rgat/6')
    env = SpatialLandingEnv(cfg)
    first, _ = env.reset(seed=16)
    assert first.packet.values.shape == (39,)
    assert first.graph.X.shape == (9,12)
    assert first.packet.registry_sha256 != SpatialConfig().registry_hash
    before = observation(env.estimator, env.safety, 0., cfg)
    env.estimator.pad_a[:] = [1.,-.5,0.]
    after = observation(env.estimator, env.safety, 0., cfg)
    assert not np.array_equal(before.packet.values[-4:], after.packet.values[-4:])
    for _ in range(10):
        _, _, done, _, info = env.step([0.,0.,0.])
        assert np.isfinite(info['truth_relative_position']).all()
        if done:
            break
@pytest.mark.parametrize('version,expected', [('5', (.9,.6,.5)),('8',(.2,.15,1.))])
def test_direct_entry_converges_before_same_seeded_disturbance(monkeypatch,version,expected):
    from ontology_rgat.spatial.environment import IsaacBackend
    from ontology_rgat.spatial.core import SpatialConfig
    from ontology_rgat.spatial.runtime_contract import deployment_profile,runtime_source_hash
    from dataclasses import replace
    cfg = replace(SpatialConfig(),schema=f'spatial-causal-rgat/{version}')
    captured=[]
    class Bridge:
        def __init__(self,c): captured.append(c.external)
        def get_state(self): pytest.fail('identity query must not validate a previous flight')
        def transact(self, *_, **kwargs):
            return {'estimator_valid':False,'armed':False,'extra':{'spatial_clock':{'valid':True,
                'profile_sha256':deployment_profile(cfg.schema)['sha256'],
                'source_sha256':runtime_source_hash()}}}
        def close(self): pass
    monkeypatch.setattr('ontology_rgat.bridge.PX4Bridge',Bridge)
    backend=IsaacBackend(cfg)
    actual=captured[0]
    assert (actual.entry_tolerance,actual.entry_speed_tolerance,actual.entry_settle)==expected
    assert backend.entry_contract['seeded_disturbance_after_entry']
