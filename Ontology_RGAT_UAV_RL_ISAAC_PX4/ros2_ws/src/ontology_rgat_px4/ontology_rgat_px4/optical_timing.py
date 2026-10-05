"""Bounded, causal interpolation of already received own EKF samples.

The gateway associates receipt with its latest simulation-clock callback.
That removes full camera-frame age, but is not hardware timestamp calibration.
No deck telemetry or simulator pose may be stored in this history.
"""
from collections import deque
import math
import numpy as np


class OwnStateHistory:
    def __init__(self, max_samples=512):
        self.samples = deque(maxlen=max_samples)

    def append(self, t, position, velocity, quaternion):
        if t is None or not math.isfinite(t):
            return
        p,v,q = (np.asarray(x,dtype=float).copy() for x in (position,velocity,quaternion))
        if p.shape != (3,) or v.shape != (3,) or q.shape != (4,) or not all(
            np.isfinite(x).all() for x in (p,v,q)) or np.linalg.norm(q) < 1e-9:
            return
        if self.samples and t < self.samples[-1][0]:
            self.samples.clear()
        if self.samples and t == self.samples[-1][0]:
            self.samples.pop()
        self.samples.append((float(t),p,v,q/np.linalg.norm(q)))

    def at(self, t, *, max_extrapolation=.05):
        if not self.samples or not math.isfinite(t) or t < self.samples[0][0]:
            return None
        a = self.samples[-1]
        if t >= a[0]:
            gap = t-a[0]
            return None if gap > max_extrapolation else (a[1]+gap*a[2],a[2].copy(),a[3].copy())
        for a,b in zip(self.samples,list(self.samples)[1:]):
            if a[0] <= t <= b[0]:
                if b[0]-a[0] > .2:
                    return None
                f = (t-a[0])/(b[0]-a[0])
                qb = b[3] if np.dot(a[3],b[3]) >= 0 else -b[3]
                q = (1-f)*a[3]+f*qb
                return ((1-f)*a[1]+f*b[1],(1-f)*a[2]+f*b[2],q/np.linalg.norm(q))
        return None
