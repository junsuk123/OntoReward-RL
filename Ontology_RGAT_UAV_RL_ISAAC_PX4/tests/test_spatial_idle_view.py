import importlib.util
from pathlib import Path

import pytest


def viewer():
    spec=importlib.util.spec_from_file_location('spatial_idle_view',
        Path(__file__).resolve().parents[1]/'tools/show_spatial_view.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_idle_view_snapshot_only_requests_state_and_is_not_landing_evidence():
    class Bridge:
        def transact(self,command,payload,response):
            assert (command,payload,response)==('state',{},('state',))
            return {'armed':False,'landed':True}
    report=viewer().disarmed_view_snapshot(Bridge())
    assert not report['task_landing_evidence'] and not report['reset_or_arm_sent']
    assert report['state']['landed'] is True


@pytest.mark.parametrize('armed',[True,None])
def test_idle_view_rejects_armed_or_unknown_state(armed):
    class Bridge:
        def transact(self,*_): return {'armed':armed}
    with pytest.raises(RuntimeError,match='disarmed'):
        viewer().disarmed_view_snapshot(Bridge())
