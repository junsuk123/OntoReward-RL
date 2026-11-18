"""WMON v0.1 (run_WMON_demo.m) port and its opt-in pad track.

* ``spatial.wmon`` must reproduce MATLAB R2024b's ``inferEpisode`` and
  ``batchLoss`` (fixture: tests/fixtures/wmon_matlab_reference.npz, written by
  tests/fixtures/make_wmon_fixture.m from the verbatim local functions).
* ``spatial.wmon_track.WMONSpatialTrack`` is a drop-in for the capture-aligned
  ABG track behind ``CurrentObserver(track_factory=...)``; the registered
  observation contract and the default observer stay the ABG ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.current_semantics.backend import (CurrentObserver, CurrentSpatialBackend,
                                                     make_local_backend)
from ontology_rgat.current_semantics.contracts import (OBSERVATION_DIMENSION, load_config,
                                                       observation_manifest)
from ontology_rgat.spatial import wmon
from ontology_rgat.spatial.core import Measurement
from ontology_rgat.spatial.estimation import ReferenceSpatialTrack
from ontology_rgat.spatial.wmon_track import (ESTIMATOR_ID, WMONSpatialTrack,
                                              WMONTrackSettings, wmon_track_factory)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "wmon_matlab_reference.npz"
PAD_SPEED = 0.8


# ------------------------------------------------------------- MATLAB parity

@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE)


def test_trained_coefficients_are_the_matlab_ones(reference):
    np.testing.assert_array_equal(reference["theta"], wmon.MATLAB_TRAINED_THETA)
    np.testing.assert_allclose(reference["theta_init"], wmon.MATLAB_INITIAL_THETA, atol=1e-15)
    params = wmon.WMONParams.from_theta(reference["theta"])
    np.testing.assert_allclose(params.to_theta(), reference["theta"], atol=1e-14)
    assert params == wmon.MATLAB_TRAINED


@pytest.mark.parametrize("variant, theta_key, bonds", [("full_state", "theta", True),
                                                      ("nobond_state", "theta", False),
                                                      ("init_state", "theta_init", True)])
def test_infer_episode_reproduces_matlab(reference, variant, theta_key, bonds):
    params = wmon.WMONParams.from_theta(reference[theta_key])
    for seed in reference["seeds"]:
        obs = reference[f"ep{seed}_obs"]
        valid = reference[f"ep{seed}_posValid"].ravel().astype(bool)
        assert not valid.all(), "every fixture episode carries a GNSS dropout"
        estimate = wmon.infer_episode(params, obs, valid, 0.1, use_bonds=bonds)
        np.testing.assert_allclose(estimate.state, reference[f"ep{seed}_{variant}"],
                                   rtol=0, atol=1e-10, err_msg=f"seed {seed}")
        if variant == "full_state":
            np.testing.assert_allclose(estimate.energy, reference[f"ep{seed}_full_energy"],
                                       rtol=0, atol=1e-10)
            np.testing.assert_allclose(estimate.confidence,
                                       reference[f"ep{seed}_full_confidence"], rtol=0,
                                       atol=1e-12)
            np.testing.assert_array_equal(wmon.hold_missing_position(obs, valid),
                                          reference[f"ep{seed}_raw_state"])


def test_batch_loss_reproduces_matlab(reference):
    for key, loss_key in (("theta", "loss_theta"), ("theta_init", "loss_init")):
        params = wmon.WMONParams.from_theta(reference[key])
        losses = [wmon.episode_loss(params, reference[f"ep{s}_obs"],
                                    reference[f"ep{s}_posValid"].ravel().astype(bool),
                                    reference[f"ep{s}_xTrue"], 0.1)
                  for s in reference["seeds"]]
        assert np.mean(losses) == pytest.approx(float(reference[loss_key]), rel=1e-12)


def test_bonds_are_what_bridge_the_dropout(reference):
    """The prototype's claim, on its own fixture: without bonds the position atom
    collapses during the dropout; with them it stays near truth."""
    for seed in reference["seeds"]:
        valid = reference[f"ep{seed}_posValid"].ravel().astype(bool)
        truth = reference[f"ep{seed}_xTrue"]
        dark = ~valid
        full = np.abs(reference[f"ep{seed}_full_state"][:2, dark] - truth[:2, dark]).max()
        none = np.abs(reference[f"ep{seed}_nobond_state"][:2, dark] - truth[:2, dark]).max()
        assert full < 0.5 < none


def test_the_core_is_dimension_generic(reference):
    """A 3-axis episode is three independent 1-axis ones (shared particles per axis)."""
    obs = reference["ep504_obs"]
    valid = reference["ep504_posValid"].ravel().astype(bool)
    two = wmon.infer_episode(wmon.MATLAB_TRAINED, obs, valid, 0.1).state
    for axis in range(2):
        one = wmon.infer_episode(wmon.MATLAB_TRAINED, obs[axis::2], valid, 0.1).state
        np.testing.assert_allclose(one, two[axis::2], atol=1e-12)


def test_spsa_adam_descends_a_quadratic():
    target = np.array([1.0, 2.0, 0.5, 3.0, 2.0, -1.0, 0.2])

    def batch_loss(theta, ids):
        return float(np.sum((theta - target) ** 2))

    result = wmon.spsa_adam(batch_loss, 4, np.zeros(7), epochs=400, learning_rate=0.05,
                            rng=np.random.default_rng(0))
    assert batch_loss(result.theta, []) < 0.05 * batch_loss(np.zeros(7), [])


@pytest.mark.parametrize("window, modes", [(1, 0), (12, 6), (12, -1), (4, 2)])
def test_fourier_mask_needs_a_high_band(window, modes):
    with pytest.raises(ValueError):
        wmon.fourier_mask(window, modes)


# ----------------------------------------------------------------- pad track

def stream(*, kick=0.0, rate_hz=10.0, seconds=4.0, seed=0, dark=None):
    """Hover at 2 m over a pad driving +x (test_cs3d_reacquire.py's stream);
    ``dark`` = (start, end) withholds every optical frame in that window."""
    rng = np.random.default_rng(seed)
    out, sample = [], 0
    period = 1.0 / rate_hz
    for k in range(int(round(seconds / 0.01)) + 1):
        t = k * 0.01
        own, pad = np.array([0.0, 0.0, 2.0]), np.array([PAD_SPEED * t, 0.0, 0.0])
        frame = None
        if abs(t / period - round(t / period)) < 1e-6 and not (dark and dark[0] <= t < dark[1]):
            frame, sample = own - pad + rng.normal(0.0, 0.02, 3), sample + 1
        seen = frame is not None
        out.append(Measurement(
            time_s=t, own_position=own, own_velocity=np.array([kick if t < 0.3 else 0.0, 0, 0]),
            quaternion=np.array([1.0, 0, 0, 0]), angular_rate=np.zeros(3),
            optical_position=frame, confidence=0.5 if seen else 0.0, sample_id=sample,
            optical_time_s=t if seen else None, optical_own_position=own if seen else None,
            optical_quaternion=np.array([1.0, 0, 0, 0]) if seen else None))
    return out


@pytest.mark.parametrize("kick, rate", [(-0.8, 20.0), (0.0, 10.0), (-0.8, 10.0)])
def test_wmon_track_never_locks_out(kick, rate):
    """Where the ABG track stops accepting after the first second (lockout), WMON
    assimilates every fresh frame and converges to the moving pad."""
    track = WMONSpatialTrack()
    measurements = stream(kick=kick, rate_hz=rate)
    accepted = [m.time_s for m in measurements if track.update(m, level_ground=True)]
    frames = len({m.sample_id for m in measurements if m.optical_position is not None})
    assert len(accepted) == frames and max(accepted) == pytest.approx(4.0)
    assert track.position[0] == pytest.approx(PAD_SPEED * 4.0, abs=0.06)
    assert track.velocity[0] == pytest.approx(PAD_SPEED, abs=0.12)
    assert track.velocity[2] == 0.0 and track.acceleration[2] == 0.0


def test_wmon_track_bridges_a_blackout():
    track = WMONSpatialTrack()
    errors = []
    for m in stream(rate_hz=10.0, seconds=5.0, dark=(2.0, 3.0)):
        track.update(m, level_ground=True)
        if 2.0 <= m.time_s < 3.0:
            errors.append(abs(track.position[0] - PAD_SPEED * m.time_s))
    assert max(errors) < 0.15                       # bonds carry the pad through
    assert track.dissolutions == 1                  # the stale wave window restarted
    assert track.position[0] == pytest.approx(PAD_SPEED * 5.0, abs=0.06)


def test_wmon_track_keeps_the_reference_freshness_rule():
    measurements = stream(rate_hz=10.0, seconds=1.0)
    wmon_track, abg = WMONSpatialTrack(), ReferenceSpatialTrack(capture_aligned=True)
    for m in measurements:
        a, b = wmon_track.update(m, level_ground=True), abg.update(m, level_ground=True)
        if not abg.initialized or b:
            assert a == b
        assert wmon_track.last_sample == abg.last_sample
    last = measurements[-1]
    late = Measurement(time_s=last.time_s + 0.6, own_position=last.own_position,
                       own_velocity=last.own_velocity, quaternion=last.quaternion,
                       angular_rate=last.angular_rate,
                       optical_position=np.zeros(3), confidence=0.5,
                       sample_id=last.sample_id + 1, optical_time_s=last.time_s + 0.05,
                       optical_own_position=last.own_position,
                       optical_quaternion=last.quaternion)
    assert wmon_track.update(late, level_ground=True) is False     # > 0.5 s old frame
    with pytest.raises(ValueError, match="reversed"):
        wmon_track.update(measurements[0], level_ground=True)


def test_settings_round_trip(tmp_path):
    settings = WMONTrackSettings(params=wmon.WMONParams.from_theta(np.zeros(7)), window=10,
                                 n_modes=2, velocity_span=3)
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"settings": settings.to_dict()}))
    assert WMONTrackSettings.load(path) == settings


# --------------------------------------------------------- observer / backend

def test_the_registered_observer_and_contract_are_untouched():
    observer = CurrentObserver()
    assert type(observer.track) is ReferenceSpatialTrack
    assert observation_manifest()["pad_track"]["estimator"] == \
        "spatial.estimation.ReferenceSpatialTrack(capture_aligned=True)"
    for m in stream(seconds=1.0):
        observer.ingest(m)
    _, provenance = observer.vector()
    assert "pad_estimator" not in provenance


def test_observer_with_the_wmon_track():
    observer = CurrentObserver(track_factory=wmon_track_factory())
    observer.reset("synthetic")
    assert isinstance(observer.track, WMONSpatialTrack)
    for m in stream(kick=-0.8, rate_hz=10.0):
        observer.ingest(m)
    vector, provenance = observer.vector()
    assert vector.shape == (OBSERVATION_DIMENSION,) and np.isfinite(vector).all()
    assert provenance["pad_estimator"] == ESTIMATOR_ID
    assert provenance["pad_estimator_diagnostics"]["frames_assimilated"] == 41
    assert observer.reacquisitions == 0


def test_scientific_use_of_a_replaced_track_is_refused():
    from ontology_rgat.current_semantics.backend import local_plant_config
    from ontology_rgat.spatial.environment import LocalBackend
    config = load_config()
    flight = LocalBackend(local_plant_config(config))
    try:
        with pytest.raises(ValueError, match="track_factory"):
            CurrentSpatialBackend(flight, config, name="spatial_local_v12", scientific=True,
                                  track_factory=wmon_track_factory())
    finally:
        flight.close()


def test_a_local_flight_runs_on_the_wmon_track():
    config = load_config()
    backend = make_local_backend(config, scenario="in_distribution",
                                 track_factory=wmon_track_factory())
    try:
        vector, info = backend.reset(910_000)
        assert info["observation_provenance"]["pad_estimator"] == ESTIMATOR_ID
        for _ in range(20):
            step = backend.step(np.zeros(3))
            assert np.isfinite(step.observation).all()
            if step.done:
                break
    finally:
        backend.close()
