"""spatial-current-observation/3: the pad track re-acquires after repeated rejections.

The reference ABG track initialises the pad velocity to the OWN velocity and
corrects it by 0.02/gap per frame; at Isaac's ~10 Hz, or after a handover kick,
its prediction drifts past the innovation gate and it never accepts a frame
again (isaac-eval/2: 8 of 14 missed approaches, G0_G4_STATUS_KO.md 10.8).
"""
from __future__ import annotations

import numpy as np
import pytest

from ontology_rgat.current_semantics import OBSERVATION_VERSION
from ontology_rgat.current_semantics import backend as backend_module
from ontology_rgat.current_semantics.backend import CurrentObserver
from ontology_rgat.current_semantics.contracts import (OBSERVATION_DIMENSION,
                                                       TRACK_REACQUIRE_AFTER_REJECTIONS,
                                                       observation_manifest)
from ontology_rgat.spatial.core import Measurement
from ontology_rgat.spatial.estimation import ReferenceSpatialTrack

PAD_SPEED = 0.8   # top of the nominal v0 range, m/s


def stream(*, kick: float, rate_hz: float, seconds: float = 4.0, seed: int = 0):
    """Hover at 2 m over a pad driving +x; 100 Hz samples, optical frames at ``rate_hz``.
    ``kick`` is the own x velocity reported for the first 0.3 s (the handover transient)."""
    rng = np.random.default_rng(seed)
    out, sample = [], 0
    period = 1.0 / rate_hz
    for k in range(int(round(seconds / 0.01)) + 1):
        t = k * 0.01
        own, pad = np.array([0.0, 0.0, 2.0]), np.array([PAD_SPEED * t, 0.0, 0.0])
        frame = None
        if abs(t / period - round(t / period)) < 1e-6:
            frame, sample = own - pad + rng.normal(0.0, 0.02, 3), sample + 1
        seen = frame is not None
        out.append(Measurement(
            time_s=t, own_position=own, own_velocity=np.array([kick if t < 0.3 else 0.0, 0, 0]),
            quaternion=np.array([1.0, 0, 0, 0]), angular_rate=np.zeros(3),
            optical_position=frame, confidence=0.5 if seen else 0.0, sample_id=sample,
            optical_time_s=t if seen else None, optical_own_position=own if seen else None,
            optical_quaternion=np.array([1.0, 0, 0, 0]) if seen else None))
    return out


def fly(measurements, observer):
    observer.reset("synthetic")
    accepted, vectors = [], []
    for m in measurements:
        if observer.ingest(m):
            accepted.append(m.time_s)
        if round(m.time_s * 100) % 10 == 0:
            vectors.append(observer.vector()[0])
    return accepted, vectors


def test_the_reference_track_locks_out_where_isaac_does():
    for kick, rate in ((-0.8, 20.0), (0.0, 10.0), (-0.8, 10.0)):
        track = ReferenceSpatialTrack(capture_aligned=True)
        last = max(m.time_s for m in stream(kick=kick, rate_hz=rate)
                   if track.update(m, level_ground=True))
        assert last < 1.0, (kick, rate)       # nothing accepted after the first second
        assert abs(track.position[0] - PAD_SPEED * 4.0) > 1.0


@pytest.mark.parametrize("kick, rate", [(-0.8, 20.0), (0.0, 10.0), (-0.8, 10.0)])
def test_reacquisition_recovers_position_and_velocity(kick, rate):
    observer = CurrentObserver()
    accepted, _ = fly(stream(kick=kick, rate_hz=rate), observer)
    assert observer.reacquisitions >= 1 and max(accepted) == pytest.approx(4.0)
    assert observer.track.position[0] == pytest.approx(PAD_SPEED * 4.0, abs=0.05)
    assert observer.track.velocity[0] == pytest.approx(PAD_SPEED, abs=0.05)
    assert observer.track.velocity[2] == 0.0
    _, provenance = observer.vector()
    assert provenance["track_reacquisitions"] == observer.reacquisitions


def test_nothing_changes_while_the_track_accepts(monkeypatch):
    measurements = stream(kick=0.0, rate_hz=20.0)
    with_rule, without = CurrentObserver(), CurrentObserver()
    a, va = fly(measurements, with_rule)
    monkeypatch.setattr(backend_module, "TRACK_REACQUIRE_AFTER_REJECTIONS", 10 ** 9)
    b, vb = fly(measurements, without)
    assert with_rule.reacquisitions == 0 and a == b
    assert all(np.array_equal(x, y) and x.shape == (OBSERVATION_DIMENSION,)
               for x, y in zip(va, vb))


def test_rule_is_registered_in_the_observation_contract():
    assert OBSERVATION_VERSION == "spatial-current-observation/3"
    track = observation_manifest()["pad_track"]
    assert track["reacquire_after_rejections"] == TRACK_REACQUIRE_AFTER_REJECTIONS == 3
    assert "least-squares" in track["reacquire_rule"]
