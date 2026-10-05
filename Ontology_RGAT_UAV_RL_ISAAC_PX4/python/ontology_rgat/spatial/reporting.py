"""Trace-based descriptive metrics; no test-driven checkpoint selection."""
import math
import numpy as np


def wilson_interval(successes, count, z=1.959963984540054):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError('positive trial count and admissible successes required')
    rate=successes/count
    centre=(rate+z*z/(2*count))/(1+z*z/count)
    half=z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.,centre-half),min(1.,centre+half)]


def trace_metrics(rows):
    """Time-weighted tracking/control measures, plus strict terminal geometry.

    An optical solve being available is not ground-truth geometric visibility.
    Loss/reacquisition below refers to the accepted causal track (age > .5 s).
    """
    if not rows:
        raise ValueError('empty episode trace')
    info=[r['info'] for r in rows]
    dt=np.array([r['dt_s'] for r in info],float)
    if not np.isfinite(dt).all() or np.any(dt<=0):
        raise ValueError('invalid trace decision intervals')
    times=np.array([r['elapsed_s'] for r in info],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('episode trace must have a strictly increasing clock')
    def rms(values):
        array=np.asarray(values,float)
        if not np.isfinite(array).all():
            raise ValueError('nonfinite trace values')
        weights=dt.reshape((-1,)+(1,)*(array.ndim-1))
        return np.sqrt(np.sum(weights*array**2,axis=0)/dt.sum()).tolist()
    action=np.asarray([r['action'] for r in rows])
    age=np.array([r['optical_age_s'] for r in info])
    lost=age>.5
    loss_count=int(lost[0])+int(np.sum((~lost[:-1])&lost[1:]))
    reacquisition_times=[]
    start=times[0]-dt[0] if lost[0] else None
    for t, duration, missing in zip(times,dt,lost):
        if missing and start is None:
            start=t-duration
        elif not missing and start is not None:
            reacquisition_times.append(float(t-start))
            start=None
    last=info[-1]
    terminal=last['status']!='RUNNING'
    if any(r['status']!='RUNNING' for r in info[:-1]):
        raise ValueError('trace continued after a task terminal')
    return dict(
        complete=terminal, status=last['status'], steps=len(rows), elapsed_s=float(times[-1]),
        optical_measurement_available_fraction=float(np.dot(dt,[r['optical_detected'] for r in info])/dt.sum()),
        accepted_track_available_fraction=float(np.dot(dt,~lost)/dt.sum()),
        maximum_accepted_track_age_s=float(age.max()), track_loss_events=loss_count,
        track_reacquisitions=len(reacquisition_times),
        track_reacquisition_times_s=reacquisition_times,
        acceleration_rms_m_s2=rms([r['applied_acceleration_m_s2'] for r in info]),
        requested_action_saturation_fraction=float(np.dot(dt,np.any(np.abs(action)>=.999,axis=1))/dt.sum()),
        safety_intervention_fraction=float(np.dot(dt,[r['safety_intervened'] for r in info])/dt.sum()),
        roll_pitch_rms_rad=rms([r['truth_roll_pitch'] for r in info]),
        roll_pitch_peak_rad=np.max(np.abs([r['truth_roll_pitch'] for r in info]),axis=0).tolist(),
        angular_rate_rms_rad_s=rms([r['truth_angular_rate'] for r in info]),
        relative_position_estimation_rmse_m=rms([
            np.array(r['estimated_relative_position'])-r['truth_relative_position'] for r in info]),
        touchdown_relative_position_m=last['truth_relative_position'] if last['truth_contact'] else None,
        touchdown_relative_velocity_m_s=last['truth_relative_velocity'] if last['truth_contact'] else None,
        time_to_land_s=float(times[-1]) if last['status']=='SUCCESS' else None)
