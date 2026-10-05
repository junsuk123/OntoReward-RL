import copy
import pytest
from ontology_rgat.spatial.reporting import wilson_interval,trace_metrics


def test_zero_successes_still_has_nonzero_uncertainty():
    lo,hi=wilson_interval(0,2)
    assert lo==pytest.approx(0.) and .65<hi<.66
    assert wilson_interval(2,2)[0]>.34
    with pytest.raises(ValueError): wilson_interval(0,0)


def test_trace_metrics_are_time_weighted_and_keep_strict_contact_outcome():
    rows=[]
    for i,(dt,age) in enumerate([(.1,0.),(.2,.6),(.1,0.)]):
        rows.append(dict(action=[0.,0.,1. if i==1 else 0.],info=dict(
            dt_s=dt,elapsed_s=[.1,.3,.4][i],optical_age_s=age,
            optical_detected=age==0,safety_intervened=i==1,
            applied_acceleration_m_s2=[2.,0.,0.],truth_roll_pitch=[.01,.02],
            truth_angular_rate=[0.,0.,0.],estimated_relative_position=[.1,0.,.2],
            truth_relative_position=[0.,0.,.2],truth_relative_velocity=[0.,0.,-.2],
            truth_contact=i==2,status='UNAUTHORIZED_CONTACT' if i==2 else 'RUNNING')))
    got=trace_metrics(rows)
    assert got['complete'] and got['time_to_land_s'] is None
    assert got['touchdown_relative_velocity_m_s']==[0.,0.,-.2]
    assert got['requested_action_saturation_fraction']==pytest.approx(.5)
    assert got['acceleration_rms_m_s2']==[2.,0.,0.]
    assert got['track_loss_events']==got['track_reacquisitions']==1
    assert got['track_reacquisition_times_s']==pytest.approx([.3])
    wrong=copy.deepcopy(rows);wrong[1]['info']['elapsed_s']=.05
    with pytest.raises(ValueError,match='clock'):trace_metrics(wrong)
