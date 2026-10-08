"""Stress scenarios are evaluation-only and change only their named factor."""
from __future__ import annotations

import numpy as np

from ontology_rgat.minimal.stress import SCENARIOS, StressBackend, StressScenario
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import LocalBackend


def _run(backend, seed, steps=30):
    from ontology_rgat.minimal.local_env import MinimalLandingEnv
    from ontology_rgat.minimal.teacher import MinimalTeacher
    env = MinimalLandingEnv(SpatialConfig(), backend=backend)
    obs, _ = env.reset(seed=seed)
    teacher, out = MinimalTeacher(), []
    for _ in range(steps):
        obs, _, done, info = env.step(teacher.act(obs))
        out.append(info["truth_pad_minus_body"])
        if done:
            break
    return np.asarray(out)


def test_identity_scenario_reproduces_the_nominal_episode():
    cfg = SpatialConfig()
    nominal = _run(LocalBackend(cfg), 4100)
    identity = _run(StressBackend(cfg, StressScenario("identity")), 4100)
    np.testing.assert_allclose(identity, nominal)


def test_fast_pad_doubles_only_the_pad_motion():
    cfg = SpatialConfig()
    nominal, fast = LocalBackend(cfg), StressBackend(cfg, SCENARIOS["fast_pad"])
    nominal.reset(4100), fast.reset(4100)
    assert fast.v0 == 2 * nominal.v0 and fast.a2 == 2 * nominal.a2
    np.testing.assert_allclose(fast.position, nominal.position)
    np.testing.assert_allclose(fast.domain_sample.external_force_n,
                               nominal.domain_sample.external_force_n)


def test_strong_wind_doubles_only_the_force():
    cfg = SpatialConfig()
    nominal, windy = LocalBackend(cfg), StressBackend(cfg, SCENARIOS["strong_wind"])
    nominal.reset(4101), windy.reset(4101)
    np.testing.assert_allclose(windy.domain_sample.external_force_n,
                               2 * np.asarray(nominal.domain_sample.external_force_n))
    assert windy.v0 == nominal.v0


def test_terminal_commit_holds_a_crosswind():
    """Combined stress, seeds 4102/4106/4126: with the policy's lateral capped
    at 0.5 m/s^2 the blind stage drifted to 0.30-0.44 m/s against a 0.7-0.9
    m/s^2 crosswind and touched down UNSAFE. The supervisor now flies it."""
    from ontology_rgat.minimal.rollout import UNSAFE, make_env
    from ontology_rgat.minimal.teacher import MinimalTeacher
    for seed in (4102, 4106, 4126):
        env = make_env("combined")
        obs, _ = env.reset(seed=seed)
        teacher, done, info = MinimalTeacher(), False, {}
        while not done:
            obs, _, done, info = env.step(teacher.act(obs))
        assert info["status"] not in UNSAFE, (seed, info["status"])
