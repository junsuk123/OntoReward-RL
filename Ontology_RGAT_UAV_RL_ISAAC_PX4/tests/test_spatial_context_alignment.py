from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from ontology_rgat.spatial.core import SpatialConfig, Estimator, Measurement, observation, safety_status
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.training import SpatialAgent, save_agent, load_agent
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.two_axis.models import POLICY_MODES
from ontology_rgat.two_axis.pretraining import pretrain_causal_encoder
from ontology_rgat.two_axis.learning import collect_rollout
from ontology_rgat.two_axis.training import PPOTrainer, PPOHyperparameters
from ontology_rgat_px4.optical_timing import OwnStateHistory


def cfg9():
    return replace(SpatialConfig(),schema='spatial-causal-rgat/9')


def measurement(t=0., capture=None, relative=(.3,-.2,2.), own=(0.,0.,2.), sample=1):
    capture = t if capture is None else capture
    return Measurement(t,np.array(own),np.zeros(3),np.array([1.,0.,0.,0.]),np.zeros(3),
                       np.array(relative),.9,sample,capture,np.array(own),np.array([1.,0.,0.,0.]))


def test_v9_preserves_disturbances_and_exact_safety_but_separates_signature():
    cfg = cfg9()
    old = replace(cfg,schema='spatial-causal-rgat/8')
    assert cfg.matched_disturbances and cfg.direct_acceleration and cfg.reference_tracking
    assert cfg.signature != old.signature
    assert deployment_profile(cfg.schema)['sha256'] != deployment_profile(old.schema)['sha256']
    a,b = SpatialLandingEnv(cfg),SpatialLandingEnv(old)
    a.reset(seed=42); b.reset(seed=42)
    for field in ('external_force_n','external_torque_nm','initial_velocity_m_s','initial_angular_rate_rad_s'):
        np.testing.assert_array_equal(getattr(a.backend.domain_sample,field),getattr(b.backend.domain_sample,field))
    assert a.safety == b.safety
    a.close(); b.close()


def test_reference_channel_meanings_and_signed_axis_information():
    cfg=cfg9(); e=Estimator(cfg); e.update(measurement())
    e.std,e.velocity_std,e.acceleration_std = .2,.3,.4
    e.rv[:] = [.4,-.5,-.1]
    e.pad_v[:] = [.2,-.1,0.]; e.pad_a[:] = [.15,-.05,0.]
    obs=observation(e,safety_status(e,cfg),0.,cfg)
    X=obs.graph.X
    assert X.shape == (2,9,12) and obs.packet.values.shape == (47,)
    np.testing.assert_allclose(X[:,4,1],[-.3/3,.2/3],atol=1e-7)
    np.testing.assert_allclose(X[:,4,3],[-.4/3,.5/3],atol=1e-7)
    np.testing.assert_allclose(X[:,5,1],np.tanh([-.3/3-.5*.4/3,.2/3+.5*.5/3]),atol=1e-7)
    np.testing.assert_allclose(X[:,1,5],.9)
    np.testing.assert_allclose(X[:,1,6],.4/3,atol=1e-7)
    expected=.9*(1-math.hypot(.3,.2)/3)*(1-math.hypot(.4,.5)/3)
    np.testing.assert_allclose(X[:,7,0],expected,atol=1e-7)
    np.testing.assert_allclose(X[:,:,10],1.)
    np.testing.assert_allclose(X[0,:,11],np.arange(1,10)/9)


def test_planar_slice_matches_reference_context_equations_in_every_channel():
    from ontology_rgat.contracts.observation import CausalObservationPacket
    from ontology_rgat.two_axis.contracts import load_reference_registry, normalize_reference_fields
    from ontology_rgat.two_axis.config import load_config, REFERENCE_CONFIG_PATH
    from ontology_rgat.two_axis.ontology_v28 import build_context_graph
    from ontology_rgat.spatial.core import camera_bearings
    from scipy.spatial.transform import Rotation
    cfg=cfg9(); e=Estimator(cfg)
    q=Rotation.from_euler('y',.03).as_quat()[[3,0,1,2]]
    m=replace(measurement(relative=(.3,0.,2.)),quaternion=q,optical_quaternion=q,
              own_velocity=np.array([.4,0.,-.1]),angular_rate=np.array([0.,.02,0.]))
    e.update(m); e.pad_v[:]=[.2,0.,0.]; e.pad_a[:]=[.1,0.,0.]; e.rv[:]=[.2,0.,-.1]
    e.std,e.velocity_std,e.acceleration_std=.2,.3,.4
    safety=safety_status(e,cfg); obs=observation(e,safety,0.,cfg)
    p=dict(zip(cfg.packet_fields,obs.packet.values))
    decode=lambda x,s: float(x)*s/(1-abs(float(x)))
    ref=load_config(REFERENCE_CONFIG_PATH)
    ref=replace(ref,camera=replace(ref.camera,fov_rad=cfg.fov[0]))
    raw={name:0. for name in load_reference_registry().field_names}
    raw.update(h=2.,vx=.4,vz=-.1,sinTheta=math.sin(.03),cosTheta=math.cos(.03),pitchRate=.02,
        detected=1.,detectionConfidence=.9,bearingValid=1.,trackInitialized=1.,
        exEstimate=-.3,relativeVxEstimate=-.2,padVxEstimate=.2,padAxEstimate=.1,
        positionStd=.2,velocityStd=.3,accelerationStd=.4,timeSinceLastDetection=0.,
        measuredBearing=camera_bearings(m.optical_position,q,cfg)[0][0],
        predictedBearing=decode(p['pred_bx'],cfg.fov[0]/2),
        predictedFovMargin=decode(p['pred_mx'],cfg.fov[0]/2),
        remainingMissionTime=70.,landingInhibited=float(safety.inhibited),abortRequested=float(safety.abort))
    fields=normalize_reference_fields(raw,half_fov=cfg.fov[0]/2)
    packet=CausalObservationPacket.from_fields(fields,timestamp_s=0.,registry=load_reference_registry())
    expected=build_context_graph(packet,load_reference_registry(),ref).X
    np.testing.assert_allclose(obs.graph.X[0],expected,atol=2e-7)


def test_loss_increases_recovery_uncertainty_and_inhibits_both_planes():
    cfg=cfg9(); e=Estimator(cfg); e.update(measurement())
    for i in range(1,42):
        e.update(replace(measurement(i*.1,sample=i+1),optical_position=None,confidence=0.))
    X=observation(e,safety_status(e,cfg),4.1,cfg).graph.X
    assert np.all(X[:,6,0] == 1) and np.all(X[:,7,0] == 0)
    assert np.all(X[:,8,0] == 1) and np.all(X[:,4,6] > 0)


def test_missing_y_alignment_blocks_joint_descent_evidence():
    cfg=cfg9(); e=Estimator(cfg); e.update(measurement(relative=(0.,4.,2.)))
    X=observation(e,safety_status(e,cfg),0.,cfg).graph.X
    assert np.all(X[:,7,0] == 0)


@pytest.mark.parametrize('mode',POLICY_MODES)
def test_all_arms_collect_update_and_reload_paired_graphs(mode,tmp_path):
    torch.set_num_threads(1)
    cfg=replace(cfg9(),horizon=.4)
    agent=SpatialAgent(mode,cfg,17)
    if mode == 'ppo_ontology_rgat':
        pre=pretrain_causal_encoder(agent,cfg,seed=17,env_factory=SpatialLandingEnv,action_dimension=3)
        assert pre['environment_steps'] > 0 and not pre['uses_actions_as_targets']
        agent.actor.set_adaptation(cfg.ontology,True)
        agent.critic.set_adaptation(cfg.ontology,True)
    env=SpatialLandingEnv(cfg)
    rollout=collect_rollout(agent,env,seed=100000,episodes=2,max_episode_decisions=20)
    report=PPOTrainer(agent,PPOHyperparameters(iterations=1)).update(rollout,discount_time_constant_s=70.)
    assert math.isfinite(report['actor_loss'])
    path=tmp_path/'model.pt'; save_agent(path,agent,cfg,eligible=True)
    loaded,_=load_agent(path,cfg)
    for k,v in agent.state_dict().items():
        assert torch.equal(v,loaded.state_dict()[k])
    with pytest.raises(ValueError,match='exact spatial contract'):
        load_agent(path,replace(cfg,schema='spatial-causal-rgat/8'))
    env.close()


def test_flat_graph_raw_initialization_and_two_plane_residual_gate():
    torch.set_num_threads(1)
    cfg=cfg9(); env=SpatialLandingEnv(cfg); obs,_=env.reset(seed=42)
    flat,graph=(SpatialAgent(mode,cfg,17) for mode in POLICY_MODES[1:])
    p,g=graph.tensors(obs)
    assert torch.equal(flat.actor(p,g)[0],graph.actor(p,g)[0])
    assert torch.equal(flat.critic(p,g),graph.critic(p,g))
    with torch.no_grad():
        graph.actor.encoder.readout.bias.fill_(1.)
        graph.actor.residual.weight.fill_(-1.)
    g[:,:,7,0]=1.; full=graph.actor.relational_delta(g)
    g[:,1,7,0]=.25; gated=graph.actor.relational_delta(g)
    assert torch.allclose(gated[:,:2],full[:,:2])
    assert torch.allclose(gated[:,2],.25*full[:,2])
    env.close()


def test_capture_age_and_repeated_sample_cannot_refresh_track():
    e=Estimator(cfg9())
    m=measurement(.1,capture=0.)
    e.update(m)
    assert e.age == pytest.approx(.1)
    e.update(replace(m,time_s=.2,own_position=np.array([.5,0.,2.])))
    assert e.age == pytest.approx(.2)
    np.testing.assert_allclose(e.reference_track.position,[-.3,.2,0.])
    old=replace(m,time_s=.3,sample_id=2,optical_time_s=-.1)
    e.update(old)
    assert e.age == pytest.approx(.3)


def test_capture_aligned_abg_does_not_interpret_own_motion_as_pad_motion():
    e=Estimator(cfg9()); e.update(measurement(relative=(0.,0.,2.)))
    for i in range(1,41):
        t=i*.05; capture=t-.05
        own_now=np.array([t,0.,2.]); own_then=np.array([capture,0.,2.])
        m=Measurement(t,own_now,np.array([1.,0.,0.]),np.array([1.,0.,0.,0.]),np.zeros(3),
                      own_then.copy(),1.,i+1,capture,own_then.copy())
        e.update(m)
    np.testing.assert_allclose(e.pad_v,np.zeros(3),atol=1e-10)
    np.testing.assert_allclose(e.r,[2.,0.,2.],atol=1e-10)


def test_own_history_interpolation_is_bounded_and_resets_on_clock_rewind():
    h=OwnStateHistory()
    h.append(1.,[0,0,2],[1,0,0],[1,0,0,0])
    h.append(1.1,[.1,0,2],[1,0,0],[-1,0,0,0])
    p,v,q=h.at(1.05)
    np.testing.assert_allclose(p,[.05,0,2]); np.testing.assert_allclose(q,[1,0,0,0])
    assert h.at(.9) is None and h.at(1.3) is None
    h.append(.2,[1,0,2],[0,0,0],[1,0,0,0])
    assert len(h.samples)==1 and h.at(1.05) is None


def test_local_frames_are_delayed_once_and_keep_capture_own_state():
    from ontology_rgat.spatial.environment import LocalBackend
    backend=LocalBackend(cfg9()); first,_=backend.reset(42)
    backend.t=.05; backend.position[0]+=.1
    held=backend.measure()
    assert held.sample_id==first.sample_id and held.optical_time_s==0.
    backend.t=.13; backend.position[0]+=.2
    delayed=backend.measure()
    assert delayed.sample_id==2 and delayed.optical_time_s==.05
    assert delayed.time_s-delayed.optical_time_s==pytest.approx(.08)
    assert delayed.optical_own_position[0]==pytest.approx(first.own_position[0]+.1)
    assert delayed.own_position[0]==pytest.approx(first.own_position[0]+.3)


def test_capture_timestamp_never_pairs_with_newer_annotator_pixels():
    from isaac_sim.sensor_profiles import camera_frame_snapshot
    class Camera:
        def get_rgba(self): return np.full((2,2,4),9.)
        def get_current_frame(self): return {'rgb':np.ones((2,2,4)), 'rendering_time':.2}
    image,stamp=camera_frame_snapshot(Camera(),capture_aligned=True)
    assert stamp==.2 and np.all(image==1.)
    image,stamp=camera_frame_snapshot(Camera(),capture_aligned=False)
    assert stamp is None and np.all(image==9.)


def test_v9_fails_closed_on_missing_capture_contract_and_defers_future_frame():
    state=dict(world=dict(position=[0,0,2],velocity=[0,0,0]),
               quaternion_wxyz=[1,0,0,0],angular_velocity=[0,0,0],estimator_valid=True,
               extra=dict(spatial_clock=dict(valid=True,sim_time_s=1.),
                          optical_measurement=dict(valid=True,frame='pad_enu',position_m=[0,0,2],confidence=1.,sample_id=1)))
    with pytest.raises(ValueError,match='capture-aligned'):
        Measurement.from_wire(state,capture_aligned=True)
    assert Measurement.from_wire(state).optical_position is not None
    state['extra']['optical_measurement'].update(capture_time_s=1.01,
        own_position_at_capture_m=[0,0,2],own_quaternion_at_capture_wxyz=[1,0,0,0])
    future=Measurement.from_wire(state,capture_aligned=True)
    assert future.optical_position is None and future.confidence==0.
    state['extra']['spatial_clock']['sim_time_s']=1.02
    current=Measurement.from_wire(state,capture_aligned=True)
    assert current.optical_time_s==1.01 and current.optical_position is not None
