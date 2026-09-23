"""The reduced planar control envelope shared by every arm of the comparison.

Why this exists
---------------
The four-degree-of-freedom command ``[v_x, v_y, v_z, omega_z]`` let the policy
solve three problems at once -- track the deck laterally, hold a heading, and
land -- and the ontology's contribution was invisible underneath that. The
reduced 2-D study (``ugv_landing_2d_workspace``) removed the two degrees of
freedom that carry no information about the question being asked, and this
module is that envelope on the live stack.

What is constrained, and by what
--------------------------------
================  ======================================================
lateral position  Commanded body-heading lateral velocity is *identically*
                  zero. The three named decks drive a straight line at a
                  constant heading and the entry pose is seeded on that
                  line, so zero lateral velocity keeps the vehicle centred
                  on the deck's track by construction. No run-time deck
                  truth is consulted -- see ``docs/PLANAR_ENVELOPE.md``.
heading (yaw)     Yaw rate is identically zero and the gateway holds the
                  absolute entry yaw, which the reset aims along the deck's
                  constant heading.
roll              Never commanded. With zero lateral velocity demand PX4's
                  velocity loop only rolls to reject disturbance.
================  ======================================================

What is left is a sagittal-plane problem: move fore/aft, climb or descend, and
choose how far to tilt longitudinally while doing it.

The action
----------
``a = [a_fwd, a_z, tilt_long]``, each normalized to ``[-1, 1]``.

``a_fwd`` and ``a_z`` are *acceleration* commands, integrated here into the
velocity setpoint PX4 flies, exactly as the reduced study's ``axMax``/``azMax``
accelerations are integrated by its own dynamics. ``tilt_long`` is the
longitudinal tilt the vehicle is asked to hold; it reaches PX4 as a bounded
longitudinal acceleration feed-forward (``a = g tan(theta)``) on the same
setpoint, so it points the fixed camera and costs a transient force. The three
channels are therefore not independent in steady state -- a quadrotor is
underactuated in its sagittal plane and pretending otherwise would be fiction.
That coupling *is* the trade-off the FOV-retention question is about: tilt to
keep the pad in frame and you perturb the approach.

Every arm -- PN guidance, the plain PPO baseline and the ontology-graph
proposal -- emits this same three-vector through this same object, so a
difference between arms is never a difference in control authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


PLANAR_ACTION_DIM = 3
PLANAR_ACTION_NAMES = (
    "longitudinal_acceleration",
    "vertical_acceleration",
    "longitudinal_tilt",
)
# Physical units each normalized channel is reported in, for the dashboard and
# the RViz HUD.
PLANAR_ACTION_UNITS = ("m/s^2", "m/s^2", "deg")

STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True)
class PlanarCommand:
    """One low-level command in the reduced envelope.

    ``velocity_body_heading_m_s`` keeps three components so that every existing
    consumer of a velocity command keeps working; its lateral component is
    always exactly zero.
    """

    velocity_body_heading_m_s: np.ndarray
    yaw_rate_rad_s: float
    longitudinal_tilt_rad: float
    longitudinal_acceleration_m_s2: float
    vertical_acceleration_m_s2: float
    # The clipped action this command came from. The reward's attitude term is
    # defined on the normalized channel rather than on radians, so that its
    # weight means the same thing it meant for Table III's normalized yaw rate.
    normalized_action: np.ndarray

    def as_array(self) -> np.ndarray:
        """``[vx, vy, vz, yaw_rate]`` -- the existing velocity wire format."""
        return np.r_[self.velocity_body_heading_m_s, self.yaw_rate_rad_s]

    def as_planar_array(self) -> np.ndarray:
        """``[vx, vy, vz, yaw_rate, tilt]`` -- what the logs and plots carry."""
        return np.r_[self.velocity_body_heading_m_s, self.yaw_rate_rad_s,
                     self.longitudinal_tilt_rad]

    @property
    def tilt_feedforward_m_s2(self) -> float:
        """Longitudinal acceleration the commanded tilt corresponds to."""
        return STANDARD_GRAVITY * math.tan(self.longitudinal_tilt_rad)


class PlanarLongitudinalController:
    """Saturating, rate-limited adapter from the 3-D action to PX4 setpoints.

    Deterministic and shared: the reward mode cannot reach its gains or limits,
    so no arm can buy authority the others do not have.
    """

    def __init__(self, max_velocity=(1.6, 0.9),
                 max_acceleration=(1.2, 0.8),
                 max_longitudinal_tilt_deg=12.0,
                 max_tilt_rate_deg_s=60.0,
                 dt=0.1, curriculum_min_action_scale=0.35,
                 tilt_channel_enabled=True):
        self.max_velocity = np.asarray(max_velocity, dtype=float).reshape(-1)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float).reshape(-1)
        if self.max_velocity.shape != (2,) or self.max_acceleration.shape != (2,):
            raise ValueError(
                "the planar envelope takes two velocity and two acceleration "
                "limits: (longitudinal, vertical)")
        self.max_longitudinal_tilt = math.radians(float(max_longitudinal_tilt_deg))
        self.max_tilt_rate = math.radians(float(max_tilt_rate_deg_s))
        self.dt = float(dt)
        self.curriculum_min_action_scale = float(curriculum_min_action_scale)
        self.tilt_channel_enabled = bool(tilt_channel_enabled)
        if self.dt <= 0 or np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("controller period and limits must be positive")
        if not 0.0 < self.max_longitudinal_tilt < 0.5 * math.pi:
            raise ValueError("the longitudinal tilt limit must be a positive angle below 90 deg")
        if self.max_tilt_rate <= 0.0:
            raise ValueError("the tilt rate limit must be positive")
        if not 0.0 < self.curriculum_min_action_scale <= 1.0:
            raise ValueError("minimum action-envelope scale must be in (0, 1]")
        self.set_curriculum(1.0)
        self.reset()

    # ------------------------------------------------------------- building
    @classmethod
    def from_mapping(cls, control=None, *, dt=None):
        """Build the shared controller from the experiment control section.

        ``max_velocity_m_s``/``max_acceleration_m_s2`` are accepted in both the
        planar two-value form and the retired three-value ``(x, y, z)`` form,
        in which case the lateral entry is dropped -- there is no lateral
        channel left for it to bound.
        """
        control = control or {}
        period = float(control.get("dt_seconds", 0.1) if dt is None else dt)

        def planar(name, default):
            value = np.asarray(control.get(name, default), dtype=float).reshape(-1)
            if value.size == 3:
                return (float(value[0]), float(value[2]))
            if value.size != 2:
                raise ValueError(f"control.{name} must have two or three values")
            return (float(value[0]), float(value[1]))

        return cls(
            max_velocity=planar("max_velocity_m_s", (1.6, 0.9)),
            max_acceleration=planar("max_acceleration_m_s2", (1.2, 0.8)),
            max_longitudinal_tilt_deg=float(
                control.get("max_longitudinal_tilt_deg", 12.0)),
            max_tilt_rate_deg_s=float(control.get("max_tilt_rate_deg_s", 60.0)),
            dt=period,
            curriculum_min_action_scale=float(
                control.get("curriculum_min_action_scale", 0.35)),
            tilt_channel_enabled=bool(
                control.get("tilt_channel_enabled", True)),
        )

    # ------------------------------------------------------------- envelope
    def set_curriculum(self, curriculum: float) -> None:
        """Scale the safe envelope from hover/slow-follow to its final limits."""
        value = float(curriculum)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("curriculum must be finite and in [0, 1]")
        self.curriculum = value
        self.action_scale = (self.curriculum_min_action_scale
                             + (1.0 - self.curriculum_min_action_scale) * value)

    def reset(self) -> None:
        self._velocity = np.zeros(2, dtype=float)   # longitudinal, vertical
        self._tilt = 0.0

    # -------------------------------------------------------------- command
    def command(self, normalized_action) -> PlanarCommand:
        action = np.asarray(normalized_action, dtype=float).reshape(-1)
        if action.shape != (PLANAR_ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                "the planar action must contain three finite values "
                "[a_fwd, a_z, tilt]")
        action = np.clip(action, -1.0, 1.0)
        scale = self.action_scale

        acceleration = action[:2] * self.max_acceleration * scale
        velocity_limit = self.max_velocity * scale
        self._velocity = np.clip(self._velocity + acceleration * self.dt,
                                 -velocity_limit, velocity_limit)

        tilt_limit = self.max_longitudinal_tilt * scale
        if self.tilt_channel_enabled:
            target = float(action[2]) * tilt_limit
        else:
            target = 0.0
        delta = float(np.clip(target - self._tilt,
                              -self.max_tilt_rate * self.dt,
                              self.max_tilt_rate * self.dt))
        self._tilt = float(np.clip(self._tilt + delta, -tilt_limit, tilt_limit))

        return PlanarCommand(
            # The lateral component is a constant of the experiment, not a
            # value the policy or a gain can move.
            velocity_body_heading_m_s=np.array(
                [self._velocity[0], 0.0, self._velocity[1]], dtype=float),
            yaw_rate_rad_s=0.0,
            longitudinal_tilt_rad=self._tilt,
            longitudinal_acceleration_m_s2=float(acceleration[0]),
            vertical_acceleration_m_s2=float(acceleration[1]),
            normalized_action=action.copy(),
        )

    # ---------------------------------------------------- velocity targets
    @property
    def velocity(self) -> np.ndarray:
        """The integrated ``(longitudinal, vertical)`` setpoint, in m/s."""
        return self._velocity.copy()

    @property
    def velocity_limit_flu(self) -> np.ndarray:
        """``(forward, lateral, vertical)`` ceiling, for three-axis callers.

        The lateral entry repeats the forward one so that code written against
        the retired three-axis envelope keeps its shape and never divides by
        zero. Nothing can act on it: :meth:`action_for_velocity` drops the
        lateral target, and the command this object emits has no lateral
        component to give it.
        """
        forward, vertical = self.max_velocity * self.action_scale
        return np.array([forward, forward, vertical], dtype=float)

    def action_for_velocity(self, forward_m_s: float, vertical_m_s: float, *,
                            tilt_rad: float | None = None) -> np.ndarray:
        """The normalized action that asks for a given velocity next step.

        Every analytic controller in this project -- the PN guidance arm, the
        privileged PD teacher, the retired image servo -- is written as a
        *velocity* law, while the shared envelope commands acceleration. This
        is the one place that conversion happens, so all of them reach PX4
        through exactly the same integrator and the same limits as the policy.

        The conversion is deadbeat: ``a = (v_target - v_now) / dt``, clipped by
        the acceleration limit, so a reachable target is reached in one step
        and an unreachable one is approached at the envelope's maximum rate.

        ``tilt_rad`` defaults to the tilt the requested longitudinal
        acceleration actually implies, so an analytic arm exercises the third
        channel consistently rather than leaving it at zero.
        """
        target = np.array([float(forward_m_s), float(vertical_m_s)], dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("a velocity target must be finite")
        limit = self.max_velocity * self.action_scale
        target = np.clip(target, -limit, limit)
        acceleration_limit = self.max_acceleration * self.action_scale
        required = (target - self._velocity) / self.dt
        normalized = np.clip(required / acceleration_limit, -1.0, 1.0)
        if tilt_rad is None:
            longitudinal = float(normalized[0] * acceleration_limit[0])
            tilt = math.atan2(longitudinal, STANDARD_GRAVITY)
        else:
            tilt = float(tilt_rad)
        tilt_limit = self.max_longitudinal_tilt * self.action_scale
        return np.array([normalized[0], normalized[1],
                         float(np.clip(tilt / tilt_limit, -1.0, 1.0))],
                        dtype=float)

    # ------------------------------------------------------------ reporting
    def envelope(self) -> dict:
        """What the dashboard and the manifest print for this envelope."""
        scale = self.action_scale
        return {
            "interface": "planar_longitudinal",
            "action_dimension": PLANAR_ACTION_DIM,
            "action_names": list(PLANAR_ACTION_NAMES),
            "action_units": list(PLANAR_ACTION_UNITS),
            "max_longitudinal_acceleration_m_s2": float(
                self.max_acceleration[0] * scale),
            "max_vertical_acceleration_m_s2": float(
                self.max_acceleration[1] * scale),
            "max_longitudinal_velocity_m_s": float(self.max_velocity[0] * scale),
            "max_vertical_velocity_m_s": float(self.max_velocity[1] * scale),
            "max_longitudinal_tilt_deg": float(
                math.degrees(self.max_longitudinal_tilt * scale)),
            "tilt_channel_enabled": bool(self.tilt_channel_enabled),
            "lateral_velocity_m_s": 0.0,
            "yaw_rate_deg_s": 0.0,
            "action_scale": float(scale),
        }
