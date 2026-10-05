"""Two spatial projections of the reference nine-node, twelve-channel graph.

Each ENU horizontal axis retains its signed position/velocity/correction and
the upstream channel meanings. Both planes share conservative 3D descent
evidence; a well-aligned x plane cannot authorize descent with unsafe y.
Input is exclusively the common normalized causal packet, never an estimator
object, future samples, controller commands, labels or simulator truth.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def reference_context_graphs(values, cfg):
    from .core import FIELDS, SCALES, DIRECT
    p = dict(zip(cfg.packet_fields, np.asarray(values, dtype=float)))
    direct = set(DIRECT.tolist())
    for i, (name, scale) in enumerate(zip(FIELDS, SCALES)):
        if i not in direct:
            p[name] = scale*p[name]/max(1-abs(p[name]), 1e-7)
    half = np.asarray(cfg.fov)/2
    for name, scale in zip(('pred_bx','pred_by','pred_mx','pred_my',
                            'meas_bx','meas_by'), np.r_[half, half, half]):
        p[name] = scale*p[name]/max(1-abs(p[name]), 1e-7)
    p['accstd'] /= max(1-p['accstd'], 1e-7)
    age = min(p['age']/cfg.loss_timeout, 1.)
    pos_u, vel_u, acc_u = (min(p[n]/s,1.) for n,s in
                           [('posstd',5.),('velstd',5.),('accstd',3.)])
    q=np.array([p[n] for n in ('qx','qy','qz','qw')])
    rotation=Rotation.from_quat(q)
    thrust_axis=rotation.apply([0.,0.,1.])
    axis_derivative=np.cross(rotation.apply([p['wx'],p['wy'],p['wz']]),thrust_axis)
    tilts=np.arctan2(thrust_axis[:2],thrust_axis[2])
    tilt_rates=(thrust_axis[2]*axis_derivative[:2]-thrust_axis[:2]*axis_derivative[2]) / np.maximum(
        thrust_axis[2]**2+thrust_axis[:2]**2,1e-9)
    attitude_limit, rate_limit = math.radians(20), math.radians(90)
    attitude_risk = min(float(np.linalg.norm(tilts))/attitude_limit,1.)
    rate_risk = min(float(np.linalg.norm(tilt_rates))/rate_limit,1.)
    position_risk = min(math.hypot(p['rx'],p['ry'])/3.,1.)
    speed_risk = min(math.hypot(p['rvx'],p['rvy'])/3.,1.)
    confidence = p['confidence']*p['initialized']
    eligibility = (confidence*(1-position_risk)*(1-speed_risk)*
                   (1-attitude_risk)*(1-rate_risk)*(1-p['inhibited']))
    inhibit = max(p['inhibited'],p['abort'],age,pos_u,vel_u)
    planes = []
    for axis, suffix in enumerate(('x','y')):
        ex, rv = -p['r'+suffix], -p['rv'+suffix]  # upstream pad-minus-UAV
        pv, pa = p['pv'+suffix], p['pa'+suffix]
        vx, vz, h = p['v'+suffix], p['vz'], p['rz']
        theta, omega = tilts[axis],tilt_rates[axis]
        # Optical image-down is opposite ENU +y at held yaw zero.
        sign = 1. if axis == 0 else -1.
        measured = sign*p['meas_b'+suffix]
        predicted = sign*p['pred_b'+suffix]
        margin = p['pred_m'+suffix]
        urgency = max(0., -margin/half[axis])
        recovery = min(1.,max(1-p['detected'],age,urgency))
        correction = math.tanh(ex/3 + .5*rv/3)
        local_pos, local_speed = min(abs(ex)/3,1.), min(abs(rv)/3,1.)
        local_attitude, local_rate = min(abs(theta)/attitude_limit,1.),min(abs(omega)/rate_limit,1.)
        rows = [
            [p['detected'],measured,predicted,margin,p['optical_valid'],p['confidence'],age,predicted,urgency],
            [min(abs(pv)/10,1),pv/10,min(abs(pa)/2,1),pa/2,p['initialized'],confidence,max(vel_u,acc_u),pa/2,acc_u],
            [min(h/8,1),vz/1.5,min(abs(vx)/10,1),vx/10,1,1,0,vz/1.5,0],
            [local_attitude,theta/attitude_limit,local_rate,omega/rate_limit,1,1,0,omega/rate_limit,local_attitude],
            [local_pos,ex/3,local_speed,rv/3,p['initialized'],confidence,max(pos_u,vel_u),rv/3,urgency],
            [abs(correction),correction,local_speed,-rv/3,p['initialized'],confidence,max(pos_u,vel_u),pa/2,local_pos],
            [recovery,-np.sign(predicted)*recovery,urgency,margin/half[axis],p['initialized'],confidence,max(age,pos_u),predicted/half[axis],recovery],
            [eligibility,eligibility,1-speed_risk,-speed_risk,p['initialized'],confidence,max(pos_u,vel_u,age),vz/1.5,1-eligibility],
            [inhibit,p['abort'],age,p['inhibited'],1,1,max(pos_u,vel_u,age),p['abort'],inhibit],
        ]
        planes.append([row+[p['remaining'],1.,(i+1)/9] for i,row in enumerate(rows)])
    return np.clip(np.asarray(planes,dtype=np.float32),-1.,1.)
