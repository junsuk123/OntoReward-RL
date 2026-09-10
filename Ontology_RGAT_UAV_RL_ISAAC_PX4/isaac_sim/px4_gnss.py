"""Pegasus sensor adapter for the urban GNSS model.

This module is imported only inside Isaac Sim.  The numerical conversion lives
in :mod:`gnss`, so normal unit tests do not need an Isaac/Pegasus installation.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from pegasus.simulator.logic.sensors import Sensor

from gnss import GnssConfig, UrbanGnss, hil_gps_measurement


class UrbanGnssSensor(Sensor):
    """Feed the city receiver into PX4's real HIL_GPS input."""

    def __init__(self, model: UrbanGnss, deck, cfg: GnssConfig):
        super().__init__(sensor_type="GPS", update_rate=cfg.update_rate_hz)
        self.model = model
        self.deck = deck
        self.cfg = cfg
        if not 0.0 <= cfg.dr_enter_quality <= cfg.dr_exit_quality <= 1.0:
            raise ValueError("GNSS DR quality thresholds must satisfy 0 <= enter <= exit <= 1")
        if cfg.recovery_epochs < 1 or cfg.bootstrap_s < 0.0:
            raise ValueError("GNSS recovery_epochs must be positive and bootstrap_s nonnegative")
        self.aiding_enabled = True
        self.mode = "bootstrap"
        self._elapsed_s = 0.0
        self._good_epochs = 0
        self._imu_dominant = False

    @Sensor.update_at_rate
    def update(self, state, dt: float):
        uav_fix, _ = self.model.update(state.position, self.deck.position, dt)
        self._elapsed_s += max(float(dt), 0.0)
        hil_fix = uav_fix
        if self._elapsed_s < self.cfg.bootstrap_s:
            # PX4 needs one stable global origin before it can expose a valid
            # local position. This happens before policy handover and does not
            # hide any degradation during an episode.
            hil_fix = replace(
                uav_fix, valid=True, fix_type=3,
                satellites_tracked=max(uav_fix.satellites_tracked, 10),
                hdop=1.0, vdop=1.6, sigma_xy_m=self.cfg.eph_floor_m,
                error_enu_m=np.zeros(3), velocity_error_enu_m_s=np.zeros(3))
            self.aiding_enabled = True
            self.mode = "bootstrap"
        else:
            if not uav_fix.valid:
                # A real loss of fix is the only case in which GPS is removed
                # completely. PX4 then propagates its EKF state on IMU/baro.
                self.aiding_enabled = False
                self._imu_dominant = True
                self._good_epochs = 0
                self.mode = "inertial_dead_reckoning"
                hil_fix = replace(uav_fix, valid=False, fix_type=0)
            else:
                # A noisy but valid receiver solution is safer than an
                # artificial long outage. Its truthful EPH/EPV makes PX4 weight
                # it weakly, so this remains IMU-dominant while bounding drift.
                self.aiding_enabled = True
                if self._imu_dominant:
                    if uav_fix.quality >= self.cfg.dr_exit_quality:
                        self._good_epochs += 1
                        if self._good_epochs >= self.cfg.recovery_epochs:
                            self._imu_dominant = False
                            self._good_epochs = 0
                    else:
                        self._good_epochs = 0
                elif uav_fix.quality <= self.cfg.dr_enter_quality:
                    self._imu_dominant = True
                    self._good_epochs = 0
                self.mode = "imu_dominant" if self._imu_dominant else "gnss_aided"
        return hil_gps_measurement(
            hil_fix,
            state.position,
            state.linear_velocity,
            self._origin_lat,
            self._origin_lon,
            self._origin_alt,
            self.cfg,
        )
