from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ontology_rgat.spatial.core import SpatialConfig, Estimator, Measurement, safety_status
from ontology_rgat.spatial.safety import ReferenceSpatialSupervisor
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.dynamics import advance_attitude_thrust
from ontology_rgat.controllers.spatial_controller import SpatialAccelerationController
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.training import SpatialAgent, save_agent, load_agent, train_arm
from ontology_rgat.two_axis.models import POLICY_MODES
from ontology_rgat.two_axis.training import PPOHyperparameters


def config():
    return replace(SpatialConfig(),schema='spatial-causal-rgat/10')


def estimate(height=2.,age=0.,std=.02,velocity_std=.1,velocity=(0.,0.,0.)):
    own=Measurement(0.,np.array([0.,0.,height]),np.asarray(velocity),
                    np.array([1.,0.,0.,0.]),np.zeros(3),np.array([0.,0.,height]),1.,0)
    return SimpleNamespace(own=own,r=np.array([0.,0.,height]),rv=np.asarray(velocity),
                           initialized=True,std=std,velocity_std=velocity_std,
                           age=age,last_t=0.)


def test_v10_has_new_signature_without_changing_v9_or_disturbances():
    cfg=config();old=replace(cfg,schema='spatial-causal-rgat/9')
    assert cfg.signature!=old.signature and cfg.registry_hash!=old.registry_hash
    assert cfg.matched_disturbances and cfg.reference_context and cfg.reference_tracking
    assert cfg.packet_fields==old.packet_fields
    assert deployment_profile(cfg.schema)['sha256']==deployment_profile(old.schema)['sha256']
    with pytest.raises(ValueError,match='episode-owned'):
        safety_status(estimate(),cfg)


def test_uncertain_reacquisition_does_not_clear_latched_abort():
    cfg=config();s=ReferenceSpatialSupervisor();e=estimate(age=3.)
    assert s.status(e,cfg).abort
    e.age=.1;e.std=1.
    assert s.status(e,cfg).abort
    e.std=.02;e.velocity_std=1.1
    assert s.status(e,cfg).abort
    e.velocity_std=.1
    recovered=s.status(e,cfg)
    assert not recovered.abort and 'track_reacquired' in recovered.reasons


def test_delay_stopping_margin_blocks_descent_before_low_altitude():
    cfg=config();s=ReferenceSpatialSupervisor();e=estimate(height=1.2,velocity=(0,0,-1.))
    status=s.status(e,cfg)
    assert s.stopping_margin==pytest.approx(1.2-.25-1/(2*(2-np.sqrt(3)*.5)))
    assert status.inhibited and 'vertical_stopping_margin' in status.reasons
    assert s.action([0,0,-1],e,status,cfg)[2]==1.


def test_terminal_corridor_is_reachable_but_commit_bounded_and_lateral_checked():
    cfg=config();s=ReferenceSpatialSupervisor();e=estimate(height=.3,velocity=(0,0,-.2))
    status=s.status(e,cfg)
    assert s.terminal_descent and not status.inhibited
    assert s.action([0,0,-.01],e,status,cfg)[2]<0
    e.age=1.;e.last_t=1.;e.r[2]=.13
    assert not s.status(e,cfg).inhibited
    e.r[0]=.48
    assert s.status(e,cfg).inhibited
    e.r[0]=0.;e.last_t=1.6;e.age=1.6
    assert s.status(e,cfg).inhibited


def test_abort_overrides_upward_actor_and_brakes_own_not_relative_velocity():
    cfg=config();s=ReferenceSpatialSupervisor();e=estimate(age=4.,velocity=(.3,-.2,.5))
    e.rv=np.array([-4.,4.,-2.])
    status=s.status(e,cfg)
    action=s.action([1,1,1],e,status,cfg)
    assert action[0]<0 and action[1]>0 and action[2]<0
    np.testing.assert_array_equal(action,s.action([-1,-1,-1],e,status,cfg))


def test_v10_view_reward_uses_captured_measurement_and_attitude_limit_is_conservative():
    from ontology_rgat.spatial.core import camera_bearings
    cfg=config();env=SpatialLandingEnv(cfg);env.reset(seed=126)
    _,_,_,_,info=env.step([0.,0.,0.])
    m=env.estimator.own
    bearing,_=camera_bearings(m.optical_position,m.optical_quaternion,cfg)
    expected=min(1.,np.mean((bearing/(np.asarray(cfg.fov)/2))**2))
    assert info['reward_components']['view']==pytest.approx(expected)
    assert env.controller.max_tilt==pytest.approx(np.deg2rad(20.))
    env.close()


@pytest.mark.parametrize('force',[(.75,.75,.75),(-.75,-.75,-.75),(.75,-.75,.75)])
def test_owned_abort_hold_rejects_persistent_force_in_reduced_plant(force):
    cfg=config();s=ReferenceSpatialSupervisor();e=estimate(age=4.)
    controller=SpatialAccelerationController(max_velocity=cfg.max_velocity,
        max_acceleration=cfg.max_acceleration,dt=cfg.dt,acceleration_only=True)
    controller.reset(own_velocity_enu_m_s=e.own.own_velocity,yaw_enu_rad=0.)
    angles=np.zeros(3);rates=np.zeros(3);thrust=9.80665
    p=e.own.own_position.copy();v=e.own.own_velocity.copy()
    for i in range(80):
        status=s.status(e,cfg)
        action=s.action([1.,1.,1.],e,status,cfg)
        command=controller.command(action,own_velocity_enu_m_s=v)
        for _ in range(10):
            angles,rates,thrust,a,gyro=advance_attitude_thrust(angles,rates,thrust,
                command,.01,force_body=force,torque_body=[.004,-.004,.004])
            p+=v*.01+.5*a*.01**2;v+=a*.01
        q=Rotation.from_euler('xyz',angles).as_quat()[[3,0,1,2]]
        e.own=replace(e.own,time_s=(i+1)*.1,own_position=p.copy(),
                      own_velocity=v.copy(),quaternion=q,angular_rate=gyro)
        e.r=p.copy();e.rv=v.copy();e.last_t=e.own.time_s;e.age=4.+e.last_t
    assert np.linalg.norm(v[:2])<.2 and abs(v[2])<.1
    assert p[2]>.5


@pytest.mark.parametrize('mode',POLICY_MODES)
def test_v10_ppo_save_reload_and_full_episode(mode,tmp_path):
    import torch
    torch.set_num_threads(1)
    cfg=replace(config(),horizon=.4)
    summary=train_arm(mode,cfg,seed=22,hyper=PPOHyperparameters(iterations=1,
        decisions_per_iteration=4,epochs=1,value_warmup_iterations=0),
        output=tmp_path,activation_iterations=0,curriculum_enabled=False)
    assert summary['selected_checkpoint'] and summary['completed_nominal_episodes']
    agent,_=load_agent(tmp_path/summary['selected_checkpoint'],cfg)
    with pytest.raises(ValueError,match='exact spatial contract'):
        load_agent(tmp_path/summary['selected_checkpoint'],replace(cfg,schema='spatial-causal-rgat/9'))
    env=SpatialLandingEnv(cfg);obs,_=env.reset(seed=930)
    _,action,_,_=agent.act(obs,deterministic=True)
    assert np.isfinite(env.step(action)[1]);env.close()
