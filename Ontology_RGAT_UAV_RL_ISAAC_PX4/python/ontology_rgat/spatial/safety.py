"""Version-ten causal reference trust, braking and terminal corridor.

The nominal touchdown limits are unchanged. A latched abort owns ALL axes.
Its bounded own-EKF position hold is an explicit extension for the persistent
forces requested for Isaac/local training; pure velocity PD has steady drift
under those forces. No pad truth, diagnostic controller or policy target is
used. This supervisor is shared unchanged by all three representations.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .core import Safety
from .dynamics import THRUST_TAU, ATTITUDE_OMEGA, IRIS_MASS_KG


class ReferenceSpatialSupervisor:
    def __init__(self):
        self.abort_latched = False
        self.hold_position = None
        self.gate_satisfied_at_s = None
        self.terminal_descent = False
        self.stopping_margin = 0.

    def status(self, est, cfg):
        own = est.own
        thrust_axis = Rotation.from_quat(own.quaternion[[1,2,3,0]]).apply([0,0,1])
        tilt = math.acos(float(np.clip(thrust_axis[2],-1.,1.)))
        settled = (tilt <= cfg.touchdown_tilt and
                   np.linalg.norm(own.angular_rate[:2]) <= cfg.touchdown_rate)
        trustworthy = (est.initialized and est.std <= .75
                       and est.velocity_std <= 1. and est.age <= .5)
        reasons = [] if trustworthy else ['track_not_trustworthy']
        envelope = np.linalg.norm(est.r[:2]) > 60 or est.r[2] > 20
        if est.age >= cfg.loss_timeout or envelope:
            if not self.abort_latched:
                self.hold_position = own.own_position.copy()
                self.hold_position[2] += max(1.-est.r[2],0.)
            self.abort_latched = True
            reasons.append('prolonged_visual_loss' if not envelope else 'causal_envelope')
        elif self.abort_latched and trustworthy:
            self.abort_latched = False
            self.hold_position = None
            self.gate_satisfied_at_s = None
            reasons.append('track_reacquired')
        # Gate, corridor and stopping margin are heights above the stock-gear
        # touchdown, so the /4 legs shift the body figure, not the law. The
        # camera footprint is a body quantity: the mount is 0.16 m below the
        # body whatever the gear.
        height = float(est.r[2]) - cfg.landing_gear_extension_m
        footprint = max(float(est.r[2])-.16,0.)*min(np.tan(np.asarray(cfg.fov)/2))
        gate_width = min(cfg.pad_half_width,footprint)
        gate = (trustworthy and 0 < height <= 1. and settled
                and np.linalg.norm(est.r[:2]) <= gate_width
                and np.linalg.norm(est.rv[:2]) <= 1.5*cfg.touchdown_xy_speed)
        if height > 1. or height <= 0 or self.abort_latched:
            self.gate_satisfied_at_s = None
        elif gate:
            self.gate_satisfied_at_s = est.last_t
        committed = (self.gate_satisfied_at_s is not None and settled
                     and est.last_t-self.gate_satisfied_at_s <= 1.5
                     and np.linalg.norm(est.r[:2])+2*est.std <= cfg.pad_half_width
                     and np.linalg.norm(est.rv[:2]) <= 1.5*cfg.touchdown_xy_speed)
        self.terminal_descent = bool(not self.abort_latched and (gate or committed))
        downward = max(0.,-float(own.own_velocity[2]))
        delay = THRUST_TAU+2/cfg.attitude_omega+cfg.actuation_delay_s
        # Available NET acceleration is constrained by the shared adapter,
        # not the larger unconstrained rotor thrust capacity.
        force_acceleration_bound = math.sqrt(3)*.75/IRIS_MASS_KG
        braking = max(.1,cfg.max_acceleration[2]-force_acceleration_bound)
        stopping_distance = downward*delay+downward**2/(2*braking)
        self.stopping_margin = height-stopping_distance
        inhibited = not trustworthy
        if self.terminal_descent:
            inhibited = False
            reasons.append('terminal_descent_corridor')
        elif self.stopping_margin < 1.:
            inhibited = True
            reasons.append('vertical_stopping_margin')
        if self.abort_latched:
            inhibited = True
            reasons.append('abort_own_position_hold')
        return Safety(bool(inhibited),bool(self.abort_latched),
                      float(not inhibited),tuple(reasons))

    def action(self, action, est, safety, cfg):
        requested = np.asarray(action,dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError('finite spatial action required')
        limits = np.asarray(cfg.max_acceleration)
        applied = np.clip(requested,-1.,1.)*limits
        downward = max(0.,-float(est.own.own_velocity[2]))
        delay = THRUST_TAU+2/cfg.attitude_omega+cfg.actuation_delay_s
        if safety.inhibited:
            if cfg.vertical_inhibit_holds:
                # Hold the vertical RATE. Rewriting only the negative command
                # (the /1 law below) brakes a descent with up to full upward
                # authority and lets every positive command through, and this
                # plant has no restoring force on altitude -- so zero-mean
                # exploration integrates upward. See
                # SpatialConfig.vertical_inhibit_holds for the measurement.
                # Descent stays forbidden either way; what is removed is the
                # free climb, not the inhibit.
                applied[2] = float(np.clip(
                    -float(est.own.own_velocity[2])/delay,-limits[2],limits[2]))
            elif applied[2] < 0:
                applied[2] = min(limits[2],downward/delay)
        if self.terminal_descent:
            # The reference approach margin (1.5x) on every rung through /8;
            # mechanical SUCCESS still uses the unchanged stricter nominal
            # limit, never this approach margin. SpatialConfig
            # .terminal_descent_speed_factor is where a rung narrows it.
            speed_limit = cfg.terminal_descent_speed_factor*cfg.touchdown_z_speed
            if downward > speed_limit:
                applied[2] = max(applied[2],(downward-speed_limit)/delay)
        if safety.abort:
            if self.hold_position is None:
                raise RuntimeError('abort requires a latched causal own-position anchor')
            applied = (1.5*(self.hold_position-est.own.own_position)
                       -2.4*est.own.own_velocity)
            if est.own.own_velocity[2] < -.1:
                applied[2] = limits[2]
        return np.clip(applied/limits,-1.,1.)
