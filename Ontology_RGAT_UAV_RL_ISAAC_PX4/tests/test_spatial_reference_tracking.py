from dataclasses import replace
import numpy as np
import pytest

from ontology_rgat.spatial.core import SpatialConfig, Estimator, Measurement, observation
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.two_axis.estimation import CausalPadEstimator
from ontology_rgat.two_axis.config import EstimatorConfig
from ontology_rgat.two_axis.sensing import PadMeasurement


@pytest.mark.parametrize('dt', [.01, .05])
def test_reference_abg_planar_numerical_equivalence(dt):
    spatial = Estimator(replace(SpatialConfig(), schema='spatial-causal-rgat/7'))
    reference = CausalPadEstimator(EstimatorConfig(model='alpha_beta_gamma_v28',
        process_acceleration_std_m_s2=1.5, measurement_position_std_m=.02))
    for i in range(200):
        t = i*dt
        own = np.array([.2*t, 0., 2.])
        relative = .3 + .1*t + .02*t*t
        detected = not 60 <= i < 100
        pm = PadMeasurement(t, detected, 0., detected, relative, detected, 1., detected)
        expected = reference.update(pm, own_x_m=own[0], own_vx_m_s=.2)
        m = Measurement(t, own, np.array([.2,0.,0.]), np.array([1.,0.,0.,0.]),
            np.zeros(3), np.array([-relative, 0., 2.]) if detected else None, 1., i)
        spatial.update(m)
        np.testing.assert_allclose([spatial.reference_track.position[0], spatial.pad_v[0],
            spatial.pad_a[0], spatial.std, spatial.velocity_std, spatial.acceleration_std],
            [expected.pad_x_m, expected.pad_vx_m_s, expected.pad_ax_m_s2,
             expected.position_std_m, expected.velocity_std_m_s, expected.acceleration_std_m_s2],
            atol=1e-12)


def test_repeated_camera_sample_is_not_a_new_reference_correction():
    est = Estimator(replace(SpatialConfig(), schema='spatial-causal-rgat/7'))
    m = Measurement(0., np.array([0.,0.,2.]), np.zeros(3), np.array([1.,0.,0.,0.]),
                    np.zeros(3), np.array([0.,0.,2.]), 1., 1)
    est.update(m)
    est.update(replace(m, time_s=.01, optical_position=np.array([.2,0.,2.])))
    np.testing.assert_array_equal(est.reference_track.position, [0.,0.,0.])
    assert est.last_detection == 0.


def test_new_optical_frame_at_same_clock_cannot_explode_abg_velocity():
    est = Estimator(replace(SpatialConfig(), schema='spatial-causal-rgat/8'))
    m = Measurement(0.,np.array([0.,0.,2.]),np.zeros(3),np.array([1.,0.,0.,0.]),
                    np.zeros(3),np.array([0.,0.,2.]),1.,1)
    est.update(m)
    newer = replace(m,optical_position=np.array([.02,0.,2.]),sample_id=2)
    est.update(newer)
    np.testing.assert_array_equal(est.pad_v,np.zeros(3))
    assert est.last_sample == 1
    est.update(replace(newer,time_s=.05))
    assert est.last_sample == 2
    np.testing.assert_allclose(est.pad_v,[-.008,0.,0.])


def test_reference_packet_carries_uncertainty_and_previous_action_without_graph_leak():
    cfg = replace(SpatialConfig(), schema='spatial-causal-rgat/7')
    env = SpatialLandingEnv(cfg)
    obs, _ = env.reset(seed=42)
    assert obs.packet.values.shape == (43,)
    assert obs.graph.X.shape == (9,12)
    obs, _, _, _, _ = env.step([.1,.2,-.3])
    np.testing.assert_allclose(obs.packet.values[-3:], [.1,.2,-.3])
    before = obs.graph.X.copy()
    env.estimator.previous_action[:] = [-.9,.8,.7]
    after = observation(env.estimator, env.safety, env.time-env.start, cfg)
    np.testing.assert_array_equal(before, after.graph.X)
    assert cfg.registry_hash != replace(cfg, schema='spatial-causal-rgat/6').registry_hash
    assert deployment_profile(cfg.schema)['sha256'] == deployment_profile('spatial-causal-rgat/6')['sha256']
    env.close()
