import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ontology_rgat.spatial.auditing import terminal_hold_audit


def terminal():
    return dict(own_velocity_enu_m_s=[.05,-.05,.01],
                own_quaternion_wxyz=[1.,0.,0.,0.],
                truth_relative_position=[20.,0.,1.],truth_contact=False)


def test_terminal_audit_uses_own_velocity_not_moving_pad_relative_speed():
    info=terminal()
    info['truth_relative_velocity']=[-1.,0.,.01]
    before=copy.deepcopy(info)
    result=terminal_hold_audit(info)
    assert result['hold_verified'] and not result['sustained_stability_claim']
    assert info==before


@pytest.mark.parametrize('change',[
    {'own_velocity_enu_m_s':[.2,.2,0.]},
    {'own_velocity_enu_m_s':[0.,0.,-.11]},
    {'own_velocity_enu_m_s':[0.,0.,float('nan')]},
    {'truth_relative_position':[0.,0.,.49]},
    {'truth_contact':True},
    {'truth_contact':'false'},
    {'own_quaternion_wxyz':[0.,0.,0.,0.]},
])
def test_unsafe_or_missing_snapshot_cannot_verify_hold(change):
    info=terminal();info.update(change)
    assert not terminal_hold_audit(info)['hold_verified']
    assert not terminal_hold_audit({})['hold_verified']


def test_terminal_tilt_is_world_thrust_axis_norm_not_one_body_axis():
    info=terminal()
    q=Rotation.from_euler('xyz',np.deg2rad([4.,4.,90.])).as_quat()
    info['own_quaternion_wxyz']=q[[3,0,1,2]].tolist()
    result=terminal_hold_audit(info)
    assert result['evaluated'] and not result['hold_verified']
    assert not result['checks']['tilt']
