"""Onboard energy budget for the landing episode.

Why a model instead of PX4's battery
------------------------------------
PX4 SITL's simulated ``battery_status`` is the documented cause of the stale
arming failure after long lockstep sessions, it is not published over the
uXRCE-DDS bridge by default, and it is not something an experiment can seed. So
SITL's authoritative energy state is integrated here from PX4's own normalised
thrust setpoint, which the gateway already receives. On hardware the real
telemetry is preferred and the model only fills gaps.

Why the pack starts nearly empty
--------------------------------
A 3S 3500 mAh pack holds roughly 140 kJ and a hover costs about 185 W, so a
14 s episode consumes well under 2% of it. A full pack therefore makes the
reserve a constant that a policy is right to ignore. Each episode instead begins
at a seeded, nearly depleted state of charge, expressed as the hover seconds it
buys: the study is about committing to a landing on the last few percent of a
pack, which is when energy actually competes with precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatteryConfig:
    enabled: bool
    capacity_j: float
    nominal_voltage_v: float
    vehicle_mass_kg: float
    rotor_count: int
    rotor_radius_m: float
    air_density_kg_m3: float
    hover_efficiency: float
    avionics_power_w: float
    hover_seconds_min: float
    hover_seconds_max: float
    landing_reserve_s: float
    depleted_fraction: float
    prefer_px4_telemetry_on_hardware: bool
    horizon_s: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BatteryConfig":
        battery = data.get("battery", {}) or {}
        landing = data.get("landing", {}) or {}
        low, high = (float(v) for v in battery.get("episode_hover_seconds_range", (6.0, 45.0)))
        if not 0.0 < low <= high:
            raise ValueError("battery.episode_hover_seconds_range must be positive and ordered")
        capacity_j = (float(battery.get("capacity_mah", 3500.0)) / 1000.0
                      * float(battery.get("nominal_voltage_v", 11.1)) * 3600.0)
        efficiency = float(battery.get("hover_efficiency", 0.45))
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("battery.hover_efficiency must be in (0, 1]")
        return cls(
            enabled=bool(battery.get("enabled", True)),
            capacity_j=capacity_j,
            nominal_voltage_v=float(battery.get("nominal_voltage_v", 11.1)),
            vehicle_mass_kg=float(battery.get("vehicle_mass_kg", 1.5)),
            rotor_count=int(battery.get("rotor_count", 4)),
            rotor_radius_m=float(battery.get("rotor_radius_m", 0.13)),
            air_density_kg_m3=float(battery.get("air_density_kg_m3", 1.225)),
            hover_efficiency=efficiency,
            avionics_power_w=float(battery.get("avionics_power_w", 12.0)),
            hover_seconds_min=low,
            hover_seconds_max=high,
            landing_reserve_s=float(battery.get("landing_reserve_s", 2.5)),
            depleted_fraction=float(battery.get("depleted_fraction", 0.0)),
            prefer_px4_telemetry_on_hardware=bool(
                battery.get("prefer_px4_telemetry_on_hardware", True)),
            horizon_s=float(landing.get("max_time_s", 14.0)),
        )


class BatteryModel:
    """Integrates electrical energy from the commanded collective.

    Momentum theory gives the induced power of a rotor disk producing thrust
    ``T``: ``P = T**1.5 / sqrt(2 rho A)``. Dividing by a combined rotor/motor
    efficiency and adding a fixed avionics draw is enough to price a landing,
    which is all this has to do -- it never feeds the vehicle dynamics, Isaac
    does that.
    """

    def __init__(self, cfg: BatteryConfig):
        self.cfg = cfg
        disk_area = cfg.rotor_count * math.pi * cfg.rotor_radius_m ** 2
        if disk_area <= 0.0:
            raise ValueError("battery rotor geometry must give a positive disk area")
        self._induced = 1.0 / math.sqrt(2.0 * cfg.air_density_kg_m3 * disk_area)
        self.hover_thrust_n = cfg.vehicle_mass_kg * 9.80665
        self.hover_power_w = self.electrical_power(self.hover_thrust_n)
        self.initial_j = cfg.hover_seconds_max * self.hover_power_w
        self.remaining_j = self.initial_j
        self.power_w = 0.0
        self.source = "model"
        self.px4_fraction: float | None = None
        self.px4_voltage_v: float | None = None

    def electrical_power(self, thrust_n: float) -> float:
        thrust_n = max(float(thrust_n), 0.0)
        mechanical = (thrust_n ** 1.5) * self._induced
        return mechanical / self.cfg.hover_efficiency + self.cfg.avionics_power_w

    def reset(self, hover_seconds: float | None = None) -> None:
        """Start an episode with a seeded reserve, in hover seconds."""
        if hover_seconds is None:
            hover_seconds = self.cfg.hover_seconds_max
        hover_seconds = float(hover_seconds)
        if not math.isfinite(hover_seconds) or hover_seconds <= 0.0:
            raise ValueError("battery reserve must be a positive number of hover seconds")
        hover_seconds = min(hover_seconds, self.cfg.capacity_j / self.hover_power_w)
        self.initial_j = hover_seconds * self.hover_power_w
        self.remaining_j = self.initial_j
        self.power_w = 0.0

    def integrate(self, thrust_norm: float, hover_thrust_norm: float, dt: float) -> None:
        """Charge one control period against the commanded normalised thrust."""
        if not self.cfg.enabled or dt <= 0.0 or not math.isfinite(dt):
            return
        if hover_thrust_norm <= 0.0:
            return
        thrust_n = max(float(thrust_norm), 0.0) / hover_thrust_norm * self.hover_thrust_n
        self.power_w = self.electrical_power(thrust_n)
        self.remaining_j = max(0.0, self.remaining_j - self.power_w * dt)

    def adopt_px4(self, remaining_fraction: float, voltage_v: float) -> None:
        """Take PX4's own state of charge as authoritative (hardware)."""
        if not math.isfinite(remaining_fraction):
            return
        self.px4_fraction = min(max(float(remaining_fraction), 0.0), 1.0)
        self.px4_voltage_v = float(voltage_v)
        self.remaining_j = self.px4_fraction * self.cfg.capacity_j
        self.initial_j = max(self.initial_j, self.remaining_j)
        self.source = "px4"

    @property
    def hover_seconds_remaining(self) -> float:
        return self.remaining_j / self.hover_power_w

    @property
    def depleted(self) -> bool:
        if not self.cfg.enabled:
            return False
        return self.remaining_j <= self.cfg.depleted_fraction * self.initial_j

    def sample(self) -> dict[str, Any]:
        """The battery fields carried in every state reply."""
        hover_s = self.hover_seconds_remaining
        horizon = max(self.cfg.horizon_s, 1e-6)
        return {
            "enabled": bool(self.cfg.enabled),
            "source": self.source,
            "remaining_j": float(self.remaining_j),
            "initial_j": float(self.initial_j),
            "capacity_j": float(self.cfg.capacity_j),
            "energy_used_j": float(max(0.0, self.initial_j - self.remaining_j)),
            "power_w": float(self.power_w),
            "hover_power_w": float(self.hover_power_w),
            "hover_seconds_remaining": float(hover_s),
            # Normalised against the episode horizon, which is what the ontology
            # consumes: 1 means the reserve outlasts any landing attempt.
            "reserve": float(min(max(hover_s / horizon, 0.0), 1.0)),
            "landing_reserve_s": float(self.cfg.landing_reserve_s),
            "state_of_charge": float(min(max(self.remaining_j / self.cfg.capacity_j, 0.0), 1.0)),
            "voltage_v": (float(self.px4_voltage_v) if self.px4_voltage_v is not None
                          else float(self.cfg.nominal_voltage_v)),
            "depleted": bool(self.depleted),
        }
