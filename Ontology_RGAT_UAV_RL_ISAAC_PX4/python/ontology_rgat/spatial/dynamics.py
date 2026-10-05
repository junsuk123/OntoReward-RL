"""Version-six direct-acceleration attitude/thrust plant.

The planar invariant subspace follows the reference's second-order attitude
and first-order thrust dynamics. Translation comes from realized thrust and
attitude, not a separate acceleration filter or hidden position/velocity PID.
This is a reduced-order plant, not a claim of exact PX4/PhysX equivalence.
"""
import math
import numpy as np

GRAVITY = 9.80665
ATTITUDE_OMEGA = 10.0
ATTITUDE_DAMPING = 1.0
THRUST_TAU = 0.05
RATE_LIMIT = math.pi / 2
# Authored properties of Pegasus assets/Robots/Iris/iris.usd /vehicle/body.
IRIS_MASS_KG = 1.5
IRIS_INERTIA_KG_M2 = np.array([.029125, .029125, .055225])


def advance_attitude_thrust(angles, euler_rates, thrust_acceleration, command, dt,
                           *, force_body=None, torque_body=None, attitude_gain_scale=1.):
    target = np.r_[command.derived_roll_pitch_rad, command.yaw_enu_rad]
    error = (target-angles+math.pi) % (2*math.pi)-math.pi
    angular_acceleration = ATTITUDE_OMEGA**2*np.asarray(attitude_gain_scale)*error
    angular_acceleration -= 2*ATTITUDE_DAMPING*ATTITUDE_OMEGA*euler_rates
    if torque_body is not None:
        # Body disturbance in Euler coordinates. This is a reduced-order
        # attitude servo, not exact PX4 rigid-body/control-gain equivalence.
        p, q, r = np.asarray(torque_body)/IRIS_INERTIA_KG_M2
        sr0, cr0 = math.sin(angles[0]), math.cos(angles[0])
        cp0 = max(math.cos(angles[1]), .05)
        angular_acceleration += [p+(sr0*q+cr0*r)*math.sin(angles[1])/cp0,
                                 cr0*q-sr0*r, (sr0*q+cr0*r)/cp0]
    rates = np.clip(euler_rates + dt*angular_acceleration, -RATE_LIMIT, RATE_LIMIT)
    angles = angles + dt*rates
    thrust = thrust_acceleration + dt/THRUST_TAU*(
        GRAVITY*command.thrust_weight_ratio-thrust_acceleration)
    roll, pitch, yaw = angles
    sr, cr, sp, cp, sy, cy = (math.sin(roll), math.cos(roll), math.sin(pitch),
                             math.cos(pitch), math.sin(yaw), math.cos(yaw))
    body_z = np.array([cy*sp*cr+sy*sr, sy*sp*cr-cy*sr, cp*cr])
    acceleration = thrust*body_z - np.array([0., 0., GRAVITY])
    if force_body is not None:
        rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, body_z[0]],
                             [sy*cp, sy*sp*sr+cy*cr, body_z[1]],
                             [-sp, cp*sr, body_z[2]]])
        acceleration += rotation @ np.asarray(force_body)/IRIS_MASS_KG
    # Euler coordinate derivatives are not gyro body rates off the planar slice.
    body_rates = np.array([rates[0]-rates[2]*sp,
                          rates[1]*cr+rates[2]*sr*cp,
                          -rates[1]*sr+rates[2]*cr*cp])
    return angles, rates, thrust, acceleration, body_rates
