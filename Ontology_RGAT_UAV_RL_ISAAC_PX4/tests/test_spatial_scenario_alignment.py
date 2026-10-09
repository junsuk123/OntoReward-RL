from dataclasses import replace
import numpy as np
import pytest

from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import LocalBackend
from ontology_rgat.spatial.scenarios import sample_spatial_scenario,SCENARIO
from ontology_rgat.spatial.runtime_contract import deployment_profile
from isaac_sim.pad_motion import PadMotionConfig,PadTrajectory,BENCHMARK_SCENARIOS
from ontology_rgat_px4.protocol import BENCHMARK_SCENARIOS as WIRE_SCENARIOS


@pytest.mark.parametrize('seed',[42,2000,2001,9000,9001,14000,100008])
def test_local_and_actual_scenario_share_exact_draw_and_continuous_analytic_motion(seed):
    cfg=replace(SpatialConfig(),schema='spatial-causal-rgat/9')
    backend=LocalBackend(cfg);backend.reset(seed)
    shared=sample_spatial_scenario(seed)
    for name in ('t1','t2','v0','a2'):
        assert getattr(backend,name)==getattr(shared,name)
    profile=deployment_profile(cfg.schema)['resolved_scientific_configuration']
    trajectory=PadTrajectory(PadMotionConfig.from_mapping(profile))
    manifest=trajectory.reset(seed,100.,1.,SCENARIO)
    origin=np.array(manifest['position_enu_m'])
    for t in [0.,.1,shared.t1,shared.t1+shared.t2,12.,70.]:
        p,v=backend.pad_state(t)
        p2,v2=trajectory.pose(100.+t)
        np.testing.assert_allclose(p2-origin,p,atol=1e-12)
        np.testing.assert_allclose(v2,v,atol=1e-12)
    for boundary in [shared.t1,shared.t1+shared.t2]:
        pa,va=shared.state(boundary-1e-8);pb,vb=shared.state(boundary+1e-8)
        assert np.linalg.norm(pa-pb)<4e-8 and np.linalg.norm(va-vb)<2e-8
    assert 0 < shared.a2 < cfg.max_acceleration[0]
    assert shared.v0+shared.a2*shared.t2 <= 1.6
    assert SCENARIO in WIRE_SCENARIOS and SCENARIO in BENCHMARK_SCENARIOS


def test_new_episode_preserves_deck_position_but_restarts_its_clock():
    profile=deployment_profile('spatial-causal-rgat/9')['resolved_scientific_configuration']
    cfg=PadMotionConfig.from_mapping(profile)
    assert cfg.route_start=='continue'
    trajectory=PadTrajectory(cfg);trajectory.reset(42,0.,1.,SCENARIO)
    prior,_=trajectory.pose(15.)
    trajectory.reset(43,15.,1.,SCENARIO)
    now,_=trajectory.pose(15.)
    np.testing.assert_allclose(now,prior)


def test_hard_reset_restarts_pad_instead_of_carrying_prior_episode_position():
    profile=deployment_profile('spatial-causal-rgat/9')['resolved_scientific_configuration']
    cfg=PadMotionConfig.from_mapping(profile)
    used=PadTrajectory(cfg);used.reset(42,0.,1.,SCENARIO)
    prior,_=used.pose(15.)
    hard=used.reset(43,15.,1.,SCENARIO,hard_reset=True)
    fresh=PadTrajectory(cfg)
    expected=fresh.reset(43,15.,1.,SCENARIO,hard_reset=True)
    assert np.linalg.norm(np.asarray(hard['position_enu_m'])-prior) > 0.1
    np.testing.assert_allclose(hard['position_enu_m'],expected['position_enu_m'])
    np.testing.assert_allclose(used.pose(20.)[0],fresh.pose(20.)[0])
