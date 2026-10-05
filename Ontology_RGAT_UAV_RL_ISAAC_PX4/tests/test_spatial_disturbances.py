from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation

from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import LocalBackend, SpatialLandingEnv
from ontology_rgat.spatial.dynamics import advance_attitude_thrust, IRIS_MASS_KG, GRAVITY
from ontology_rgat.benchmarks.randomization import sample_domain_randomization
from ontology_rgat.controllers.spatial_controller import SpatialAccelerationController


def test_spatial_seeded_disturbance_draw_is_exactly_isaacs_sampler():
    cfg = replace(SpatialConfig(), schema='spatial-causal-rgat/8')
    for seed in [42, 823, 14002]:
        backend = LocalBackend(cfg)
        backend.reset(seed)
        assert backend.domain_sample.to_dict() == sample_domain_randomization(seed).to_dict()
        assert backend.domain_initial_pending
        assert cfg.signature != replace(cfg, schema='spatial-causal-rgat/7').signature


def test_body_force_uses_authored_mass_and_rotates_to_enu():
    angles = np.array([.1, -.12, .8])
    c = SpatialAccelerationController(acceleration_only=True)
    c.reset(own_velocity_enu_m_s=[0,0,0], yaw_enu_rad=.8)
    command = c.command([0,0,0], own_velocity_enu_m_s=[0,0,0])
    baseline = advance_attitude_thrust(angles, np.zeros(3), GRAVITY, command, .01)
    force = np.array([.2,-.3,.1])
    perturbed = advance_attitude_thrust(angles, np.zeros(3), GRAVITY, command, .01,
                                      force_body=force)
    np.testing.assert_allclose(perturbed[3]-baseline[3],
        Rotation.from_euler('xyz', perturbed[0]).apply(force)/IRIS_MASS_KG, atol=1e-14)


def test_nominal_disturbance_is_not_suppressed_by_curriculum_or_arm():
    cfg = replace(SpatialConfig(), schema='spatial-causal-rgat/8')
    first, second = SpatialLandingEnv(cfg), SpatialLandingEnv(cfg)
    a, _ = first.reset(seed=14002)
    b, _ = second.reset(seed=14002, difficulty=1.)
    np.testing.assert_array_equal(a.packet.values, b.packet.values)
    for _ in range(3):
        a, ra, da, _, ia = first.step([.05,-.03,.1])
        b, rb, db, _, ib = second.step([.05,-.03,.1])
        np.testing.assert_array_equal(a.packet.values, b.packet.values)
        assert (ra,da,ia) == (rb,db,ib)
        assert not first.backend.domain_initial_pending
    # Neither exact force nor the future initial perturbation is a feature.
    assert a.packet.values.shape == (43,)
    assert a.graph.X.shape == (9,12)


def test_zero_difficulty_removes_disturbance_but_not_changes_nominal_contract():
    base = replace(SpatialConfig(), schema='spatial-causal-rgat/7')
    easy_old = SpatialLandingEnv(base, difficulty=0.)
    easy_new = SpatialLandingEnv(replace(base,schema='spatial-causal-rgat/8'), difficulty=0.)
    easy_old.reset(seed=123)
    easy_new.reset(seed=123)
    for _ in range(3):
        a = easy_old.step([.1,-.1,.2])
        b = easy_new.step([.1,-.1,.2])
        np.testing.assert_allclose(a[0].packet.values, b[0].packet.values, atol=1e-14)
        assert a[1] == b[1]
