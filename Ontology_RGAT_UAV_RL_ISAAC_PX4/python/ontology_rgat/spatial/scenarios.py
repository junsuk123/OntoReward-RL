"""Evaluator/plant-only seeded spatial CV--CA--CV scenario, shared with Isaac.

No phase, switch time or true target velocity belongs in the causal packet.
The draw order preserves the existing local spatial scenario exactly.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

SCENARIO = 'spatial_reference_cv_ca_cv'
PLANAR_SCENARIO = 'matlab_planar_cv_ca_cv'


@dataclass(frozen=True)
class SpatialScenario:
    t1: float
    t2: float
    v0: float
    a2: float
    heading: float

    def state(self,t):
        t=max(0.,float(t))
        middle=min(max(t-self.t1,0.),self.t2)
        coast=max(0.,t-self.t1-self.t2)
        distance=self.v0*t+.5*self.a2*middle**2+self.a2*self.t2*coast
        direction=np.array([math.cos(self.heading),math.sin(self.heading),0.])
        return direction*distance,direction*(self.v0+self.a2*middle)

    def manifest(self, schema=SCENARIO):
        return dict(schema=schema,units='seconds, metres, radians',**asdict(self),
                    v3=self.v0+self.a2*self.t2,mission_horizon_s=70.)


def sample_spatial_scenario(seed):
    rng=np.random.default_rng(seed)
    # Three initial-position draws precede the motion draw in LocalBackend.
    rng.random(3)
    return SpatialScenario(float(rng.uniform(.5,2.)),float(rng.uniform(1.,2.)),
                           float(rng.uniform(.3,.8)),float(rng.uniform(.1,.4)),
                           float(rng.uniform(-.2,.2)))


def sample_planar_scenario(seed):
    """The paired spatial draw projected onto the MATLAB x-z plane.

    Calling the spatial sampler first preserves its complete RNG draw order;
    only the evaluator/plant heading is constrained to zero afterwards.
    """
    sample = sample_spatial_scenario(seed)
    return SpatialScenario(sample.t1, sample.t2, sample.v0, sample.a2, 0.0)
