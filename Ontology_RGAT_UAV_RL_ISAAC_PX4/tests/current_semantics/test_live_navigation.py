"""Opt-in live own navigation (observation profile /4) and its estimators.

* spatial.kinematic_kf: constant-acceleration Kalman on the WMON sensor set.
* spatial.wmon.WMONStream: the MATLAB WMON one sample at a time.
* current_semantics.navigation.LiveNavigation: GNSS outages + IMU in front of
  the observer; drone_navigationValid / drone_navigationAge become live.
The registered /3 observation, its manifest and the default observer stay as they were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.current_semantics.backend import (CurrentObserver, CurrentSpatialBackend,
                                                     local_plant_config, make_local_backend)
from ontology_rgat.current_semantics.contracts import (OBSERVATION_NAMES, OBSERVATION_VERSION,
                                                       contract_manifest, load_config,
                                                       observation_manifest)
from ontology_rgat.current_semantics.navigation import (LIVE_NAVIGATION_OBSERVATION_VERSION,
                                                        LiveNavigation, NavigationDegradation,
                                                        NavigationEstimator)
from ontology_rgat.spatial import wmon
from ontology_rgat.spatial.core import Measurement
from ontology_rgat.spatial.kinematic_kf import KalmanNoise, KinematicKalman, kalman_episode

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "wmon_matlab_reference.npz"
VALID = OBSERVATION_NAMES.index("drone_navigationValid")
AGE = OBSERVATION_NAMES.index("drone_navigationAge")


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE)


# ------------------------------------------------------------- estimators

def test_wmon_stream_is_infer_episode(reference):
    for seed in reference["seeds"]:
        obs = reference[f"ep{seed}_obs"]
        valid = reference[f"ep{seed}_posValid"].ravel().astype(bool)
        stream = wmon.WMONStream(wmon.MATLAB_TRAINED, 2)
        out = np.stack([stream.step(obs[:2, k], obs[2:4, k], obs[4:, k], valid[k], 0.1)
                        for k in range(obs.shape[1])], axis=1)
        np.testing.assert_allclose(out, reference[f"ep{seed}_full_state"], rtol=0, atol=1e-10)


def test_kalman_bridges_the_fixture_dropouts(reference):
    noise = KalmanNoise(0.52, 0.32, 0.38, 0.16)
    for seed in reference["seeds"]:
        obs = reference[f"ep{seed}_obs"]
        valid = reference[f"ep{seed}_posValid"].ravel().astype(bool)
        truth = reference[f"ep{seed}_xTrue"]
        state = kalman_episode(noise, obs, valid, 0.1)
        assert np.isfinite(state).all()
        kf = np.sqrt(np.mean((state[:2, 11:] - truth[:2, 11:]) ** 2))
        raw = np.sqrt(np.mean((wmon.hold_missing_position(obs, valid)[:2, 11:]
                               - truth[:2, 11:]) ** 2))
        assert kf < 0.25 * raw


def test_kalman_is_dimension_generic_and_needs_a_first_fix(reference):
    obs = reference["ep504_obs"]
    valid = reference["ep504_posValid"].ravel().astype(bool)
    noise = KalmanNoise(0.3, 0.2, 0.2, 1.0)
    two = kalman_episode(noise, obs, valid, 0.1)
    one = kalman_episode(noise, obs[0::2], valid, 0.1)
    np.testing.assert_allclose(one, two[0::2], atol=1e-12)
    with pytest.raises(ValueError):
        KinematicKalman(noise, 1).step([0.0], [0.0], [0.0], False, 0.1)


# --------------------------------------------------------------- manifest

def test_registered_manifest_is_untouched_and_profile_4_is_hashed_apart():
    nav = LiveNavigation()
    assert observation_manifest()["version"] == OBSERVATION_VERSION
    assert "navigation" not in observation_manifest()
    live = observation_manifest(nav.manifest_part())
    assert live["version"] == LIVE_NAVIGATION_OBSERVATION_VERSION
    config = load_config()
    registered = contract_manifest(config)
    assert registered == contract_manifest(config, navigation=None)
    profile4 = contract_manifest(config, navigation=nav.manifest_part())
    assert profile4["observation_hash"] != registered["observation_hash"]
    assert profile4["algorithm_hash"] != registered["algorithm_hash"]
    other = contract_manifest(config, navigation=LiveNavigation(
        estimator=NavigationEstimator("hold")).manifest_part())
    assert other["observation_hash"] != profile4["observation_hash"]


# ------------------------------------------------------------- navigation

def own_stream(seconds=8.0, speed=0.5):
    """Own vehicle moving +x at ``speed``; 100 Hz samples, no optical frames."""
    out = []
    for k in range(int(round(seconds / 0.01)) + 1):
        t = k * 0.01
        out.append(Measurement(time_s=t, own_position=np.array([speed * t, 0.0, 2.0]),
                               own_velocity=np.array([speed, 0.0, 0.0]),
                               quaternion=np.array([1.0, 0, 0, 0]), angular_rate=np.zeros(3),
                               optical_position=None, confidence=0.0, sample_id=0))
    return out


@pytest.mark.parametrize("kind", ["hold", "kalman", "wmon"])
def test_outage_drops_the_fixes_and_ages_the_channels(kind):
    spec = NavigationDegradation(outage_probability=1.0, outage_start_min_s=2.0,
                                 outage_start_max_s=2.0, outage_duration_min_s=3.0,
                                 outage_duration_max_s=3.0)
    est = NavigationEstimator(kind, kalman=KalmanNoise(0.4, 0.05, 0.2, 10.0)
                              if kind == "kalman" else None)
    nav = LiveNavigation(spec, est)
    nav.reset(7)
    errors_in, statuses = [], []
    for m in own_stream():
        out = nav(m)
        statuses.append((m.time_s, *nav.status(m.time_s)))
        if 2.5 <= m.time_s < 5.0:
            errors_in.append(abs(out.own_position[0] - m.own_position[0]))
    assert nav.outage == (2.0, 3.0)
    by_time = {round(t, 2): (v, a) for t, v, a in statuses}
    assert by_time[1.0][0] and by_time[1.0][1] < 0.1
    assert not by_time[4.0][0] and by_time[4.0][1] == pytest.approx(2.0, abs=0.11)
    assert by_time[7.0][0]
    if kind == "hold":
        assert max(errors_in) > 1.0                 # the last fix goes stale
    else:
        assert max(errors_in) < 0.5                 # velocity + IMU carry the estimate


def test_same_seed_same_degradation_for_every_estimator():
    a = LiveNavigation(estimator=NavigationEstimator("hold"))
    b = LiveNavigation(estimator=NavigationEstimator("wmon"))
    a.reset(123), b.reset(123)
    assert a.outage == b.outage
    for m in own_stream(2.0):
        a(m), b(m)
    assert a.fixes == b.fixes


def test_default_observer_keeps_constant_navigation_channels():
    observer = CurrentObserver()
    for m in own_stream(0.5):
        observer.ingest(m)
    vector, provenance = observer.vector()
    assert vector[VALID] == 1.0 and vector[AGE] == 0.0
    assert provenance["navigation"].startswith("valid 1, age 0 by construction")


def test_scientific_use_of_live_navigation_is_refused():
    from ontology_rgat.spatial.environment import LocalBackend
    config = load_config()
    flight = LocalBackend(local_plant_config(config))
    try:
        with pytest.raises(ValueError, match="spatial-current-observation/4"):
            CurrentSpatialBackend(flight, config, name="spatial_local_v12", scientific=True,
                                  navigation=LiveNavigation())
    finally:
        flight.close()


def test_local_flight_with_live_navigation():
    config = load_config()
    nav = LiveNavigation(NavigationDegradation(outage_start_min_s=1.0, outage_start_max_s=1.0,
                                               outage_duration_min_s=2.0,
                                               outage_duration_max_s=2.0))
    backend = make_local_backend(config, scenario="in_distribution", navigation=nav)
    try:
        vector, info = backend.reset(910_000)
        valid = [vector[VALID]]
        for _ in range(40):
            step = backend.step(np.zeros(3))
            assert np.isfinite(step.observation).all()
            valid.append(step.observation[VALID])
            info = step.info
            if step.done:
                break
        assert 0.0 in valid and 1.0 in valid
        provenance = info["observation_provenance"]["navigation"]
        assert provenance["version"] == LIVE_NAVIGATION_OBSERVATION_VERSION
        assert provenance["estimator"] == "wmon"
    finally:
        backend.close()


# ------------------------------------------------- /4 through the config (training path)

NAV4 = {kind: ROOT / "config" / "current_semantics" / f"spatial_view_contact_v1_nav4_{kind}.yaml"
        for kind in ("wmon", "kalman")}


@pytest.mark.parametrize("kind", ["wmon", "kalman"])
def test_nav4_config_is_v1_plus_its_profile(kind):
    from dataclasses import replace
    v1, v4 = load_config(), load_config(NAV4[kind])
    assert v1.navigation is None and v4.navigation.estimator.kind == kind
    assert replace(v4, navigation=None) == v1          # nothing else differs
    m1, m4 = contract_manifest(v1), contract_manifest(v4)
    assert m4["versions"]["observation"] == LIVE_NAVIGATION_OBSERVATION_VERSION
    for key in ("observation_hash", "algorithm_hash"):
        assert m4[key] != m1[key]
    for key in ("task_hash", "g3_hash", "g4_hash", "calibration_hash", "action_hash"):
        assert m4[key] == m1[key]
    with pytest.raises(ValueError, match="differs"):
        contract_manifest(v4, navigation=LiveNavigation(
            estimator=NavigationEstimator("hold")).manifest_part())


def test_nav4_runs_in_its_own_root_and_never_in_a_registered_one():
    from ontology_rgat.current_semantics.experiment import TASK_RUN_ROOTS, run_root_for
    v1 = load_config()
    wmon_cfg, kalman_cfg = load_config(NAV4["wmon"]), load_config(NAV4["kalman"])
    roots = {run_root_for(c, None) for c in (v1, wmon_cfg, kalman_cfg)}
    assert len(roots) == 3
    assert run_root_for(v1, None) == TASK_RUN_ROOTS["spatial-current-task/5"]
    for registered in TASK_RUN_ROOTS.values():
        with pytest.raises(ValueError, match="observation /3"):
            run_root_for(wmon_cfg, registered)
    with pytest.raises(ValueError, match="another navigation profile"):
        run_root_for(wmon_cfg, run_root_for(kalman_cfg, None))


def test_backends_from_a_nav4_config_fly_its_profile():
    config = load_config(NAV4["kalman"])
    backend = make_local_backend(config, scenario="in_distribution")
    try:
        assert backend.navigation is not None and backend.navigation.profile == config.navigation
        vector, info = backend.reset(910_000)
        assert info["observation_provenance"]["navigation"]["estimator"] == "kalman"
    finally:
        backend.close()
    with pytest.raises(ValueError, match="differs from the config"):
        make_local_backend(config, navigation=LiveNavigation(
            estimator=NavigationEstimator("hold")))
    plain = make_local_backend(load_config())
    try:
        assert plain.navigation is None
    finally:
        plain.close()
