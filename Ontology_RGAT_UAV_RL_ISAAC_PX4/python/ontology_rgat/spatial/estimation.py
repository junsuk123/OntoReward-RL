"""Spatial extension of upstream v2.8's causal absolute-pad ABG tracker.

The same gains are applied only to fresh optical solves, not repeated camera
frames. The reference's 100 Hz camera and the real 20 Hz camera are explicitly
different sensor contracts; observations are never synthesized to bridge them.
"""
import math
import numpy as np


class ReferenceSpatialTrack:
    def __init__(self, *, capture_aligned=False):
        self.capture_aligned = capture_aligned
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.std = np.tile(np.array([2., 3., 2.])[:, None], (1, 3))
        self.initialized = False
        self.last_t = None
        self.last_detection = -math.inf
        self.last_sample = -1

    def update(self, m, *, level_ground):
        if self.capture_aligned:
            return self._update_capture_aligned(m, level_ground=level_ground)
        if self.last_t is not None and m.time_s < self.last_t - 1e-8:
            raise ValueError("simulation clock reversed; reset the episode")
        dt = 0. if self.last_t is None else m.time_s - self.last_t
        if self.last_t is not None and abs(dt) <= 1e-12:
            # DDS can deliver a newer optical frame before the next clock
            # callback. Upstream also coalesces equal-time measurements. Do
            # not divide its innovation by ~0 or consume the sample ID: the
            # next increasing clock sample can still assimilate that frame.
            return False
        if dt > 2.:
            raise ValueError("sensor gap exceeds 2 simulated seconds")
        if self.initialized:
            self.position += self.velocity*dt + .5*self.acceleration*dt*dt
            self.velocity += self.acceleration*dt
            self.acceleration *= math.exp(-dt/1.5)
            process = np.array([.5*1.5*dt*dt, 1.5*dt, 1.5*math.sqrt(max(dt, 0.))])
            self.std = np.hypot(self.std, process[:, None])
        accepted = False
        if m.optical_position is not None and m.sample_id != self.last_sample:
            # Optical position is UAV minus pad; own EKF pose is causal.
            measured = m.own_position - m.optical_position
            if not self.initialized:
                self.position[:] = measured
                self.velocity[:] = m.own_velocity
                self.acceleration[:] = 0.
                self.initialized = accepted = True
            else:
                innovation = measured-self.position
                if np.all(np.abs(innovation) <= np.maximum(4*np.maximum(self.std[0], .02), .25)):
                    gap = max(m.time_s-self.last_detection, 1e-12)
                    self.position += .20*innovation
                    self.velocity += .02/gap*innovation
                    self.acceleration = np.clip(self.acceleration + .0001/gap**2*innovation, -3., 3.)
                    floor = np.array([.02, .02/gap*.02, 1.5*.25])
                    self.std = np.maximum(floor[:, None], self.std*np.array([.5, .8, .85])[:, None])
                    accepted = True
            self.last_sample = m.sample_id
        if accepted:
            self.last_detection = m.time_s
        if level_ground:
            self.velocity[2] = self.acceleration[2] = 0.
        self.last_t = m.time_s
        return accepted

    def _update_capture_aligned(self, m, *, level_ground):
        """ABG innovation at camera capture time, propagated to decision time.

        The own pose paired with the image comes solely from timestamped EKF
        history. Repeated, out-of-order, future and >0.5s-old images do not
        refresh track age. This is not a second optical correction at 100 Hz.
        """
        now = m.time_s
        if self.last_t is not None and now < self.last_t-1e-8:
            raise ValueError('simulation clock reversed; reset the episode')
        dt = 0. if self.last_t is None else now-self.last_t
        if dt > 2.:
            raise ValueError('sensor gap exceeds 2 simulated seconds')
        if self.last_t is not None and dt <= 1e-12:
            return False
        if self.initialized:
            self.position += self.velocity*dt + .5*self.acceleration*dt*dt
            self.velocity += self.acceleration*dt
            self.acceleration *= math.exp(-dt/1.5)
            process = np.array([.5*1.5*dt*dt,1.5*dt,1.5*math.sqrt(dt)])
            self.std = np.hypot(self.std,process[:,None])
        accepted = False
        capture = m.optical_time_s
        fresh = (m.optical_position is not None and m.sample_id != self.last_sample
                 and capture is not None and m.optical_own_position is not None
                 and capture > self.last_detection and 0 <= now-capture <= .5)
        if fresh:
            lag = now-capture
            measured = m.optical_own_position-m.optical_position
            if not self.initialized:
                self.velocity[:] = m.own_velocity
                self.acceleration[:] = 0.
                self.position[:] = measured+self.velocity*lag
                self.initialized = accepted = True
            else:
                predicted_at_capture = self.position-self.velocity*lag+.5*self.acceleration*lag**2
                innovation = measured-predicted_at_capture
                if np.all(np.abs(innovation) <= np.maximum(4*np.maximum(self.std[0],.02),.25)):
                    gap = max(capture-self.last_detection,1e-3)
                    dv, da = .02/gap*innovation, .0001/gap**2*innovation
                    self.position += .20*innovation + dv*lag + .5*da*lag**2
                    self.velocity += dv+da*lag
                    self.acceleration = np.clip(self.acceleration+da,-3.,3.)
                    floor = np.array([.02,.02/gap*.02,1.5*.25])
                    self.std = np.maximum(floor[:,None],self.std*np.array([.5,.8,.85])[:,None])
                    accepted = True
            self.last_sample = m.sample_id
        if accepted:
            self.last_detection = capture
        if level_ground:
            self.velocity[2] = self.acceleration[2] = 0.
        self.last_t = now
        return accepted
