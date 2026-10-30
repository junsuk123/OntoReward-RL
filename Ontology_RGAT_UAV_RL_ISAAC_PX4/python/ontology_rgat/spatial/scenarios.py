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


# spatial_current_semantics_v1 task v5 on Isaac (guide 11.6, user 2026-10-30).
# The episode's composition draw is made ONCE, by the learner
# (current_semantics/scenario.py, from the episode seed), and travels to Isaac
# as the scenario name; the gateway forwards it, pad_motion.py drives the deck
# along it and landing_world.py puts the entry at its offset.  Isaac draws
# nothing of its own for these fields, so the local plant and Isaac fly the
# same geometry from the same seed.  repr() round-trips a float exactly.
CURRENT_OMNI_SCENARIO = 'spatial_current_omni/1'
CURRENT_OMNI_RADIUS_MAX_M = 2.0
CURRENT_OMNI_A2_MAX_M_S2 = 2.0   # below the 2.5 m/s^2 lateral authority


def encode_current_omni(heading, bearing, radius, a2=None):
    """heading: pad travel (ENU rad); bearing: pad seen from the drone at reset, so
    own-minus-pad = -radius (cos, sin); a2: CA-phase override, None = the seed's."""
    name = ':'.join([CURRENT_OMNI_SCENARIO, repr(float(heading)), repr(float(bearing)),
                     repr(float(radius)), 'nominal' if a2 is None else repr(float(a2))])
    parse_current_omni(name)
    return name


def parse_current_omni(name):
    """The fields of a current-omni scenario name; None for any other scenario.
    A malformed or out-of-range current-omni name raises ValueError."""
    name = str(name)
    if not name.startswith(CURRENT_OMNI_SCENARIO + ':'):
        return None
    fields = name.split(':')[1:]
    if len(fields) != 4:
        raise ValueError(f'{CURRENT_OMNI_SCENARIO} needs heading:bearing:radius:a2')
    try:
        heading, bearing, radius = (float(v) for v in fields[:3])
        a2 = None if fields[3] == 'nominal' else float(fields[3])
    except ValueError as exc:
        raise ValueError(f'malformed {CURRENT_OMNI_SCENARIO} field: {exc}') from None
    values = (heading, bearing, radius) + (() if a2 is None else (a2,))
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'{CURRENT_OMNI_SCENARIO} fields must be finite')
    if not (-math.pi <= heading <= math.pi and -math.pi <= bearing <= math.pi):
        raise ValueError(f'{CURRENT_OMNI_SCENARIO} angles must lie in [-pi, pi]')
    if not 0.0 < radius <= CURRENT_OMNI_RADIUS_MAX_M:
        raise ValueError(f'{CURRENT_OMNI_SCENARIO} radius must lie in (0, '
                         f'{CURRENT_OMNI_RADIUS_MAX_M}] m')
    if a2 is not None and not 0.0 < a2 <= CURRENT_OMNI_A2_MAX_M_S2:
        raise ValueError(f'{CURRENT_OMNI_SCENARIO} a2 must lie in (0, '
                         f'{CURRENT_OMNI_A2_MAX_M_S2}] m/s^2')
    return dict(heading_rad=heading, bearing_rad=bearing, radius_m=radius, a2_m_s2=a2)


def current_omni_motion(seed, parsed):
    """The seed's CV-CA-CV draw with the scenario's heading (and a2 override): exactly
    what ScenarioLocalBackend leaves in the local plant for the same seed."""
    sample = sample_spatial_scenario(seed)
    a2 = sample.a2 if parsed['a2_m_s2'] is None else parsed['a2_m_s2']
    return SpatialScenario(sample.t1, sample.t2, sample.v0, a2, parsed['heading_rad'])
