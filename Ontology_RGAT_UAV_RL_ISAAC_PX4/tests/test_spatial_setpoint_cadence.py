import ast
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat_px4.ros2_gateway import setpoint_publication_due


@pytest.mark.parametrize('real_time_factor',[.05,.1,.2,.5,1.])
def test_spatial_heartbeat_never_floods_slow_lockstep(real_time_factor):
    previous=None
    published=[]
    for wall in np.arange(0.,2./real_time_factor,.02):
        sim=wall*real_time_factor
        if setpoint_publication_due('sitl',sim,previous,50.):
            published.append(sim);previous=sim
    assert len(published)==100
    np.testing.assert_allclose(np.diff(published),.02,atol=1e-12)


def test_paused_clock_does_not_replay_or_manufacture_heartbeats():
    assert not setpoint_publication_due('sitl',10.,10.,50.)
    assert setpoint_publication_due('sitl',11.,10.,50.)
    assert not setpoint_publication_due('sitl',11.,11.,50.)
    assert setpoint_publication_due('sitl',0.,11.,50.)  # Explicit clock reset.


def test_hardware_and_legacy_keep_wall_timer():
    assert setpoint_publication_due('hardware',1.,1.,50.)
    assert setpoint_publication_due('sitl',None,None,50.)


def test_control_tick_paces_after_deadman_before_publish_and_prestream():
    source=(Path(__file__).resolve().parents[1]/
        'ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4/ros2_gateway.py').read_text()
    method=next(n for n in ast.walk(ast.parse(source))
                if isinstance(n,ast.FunctionDef) and n.name=='_control_tick')
    text=ast.get_source_segment(source,method)
    assert text.index('action_age_seconds')<text.index('setpoint_publication_due')
    assert text.index('setpoint_publication_due')<text.index('self._publish_spatial_acceleration_setpoint()')
    assert text.index('setpoint_publication_due')<text.index('self.prestream += 1')
