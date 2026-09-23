"""A stalled renderer is infrastructure, and the episode retry must own it.

On 2026-09-23 a two-arm run died at episode 128 of 1008 with

    TimeoutError: no Shin benchmark camera frame received

three stack restarts into a recovery whose retry budget was untouched. The ROS
frame buffer raises a bare ``TimeoutError``; ``collect_episode_resilient``
catches ``BridgeError``, so it went straight through the retry, out of the
worker thread, and ended the run. Every other way a rebuilt stack interrupts a
step -- gateway timeout, entry reset, recoverable failsafe -- was already
handled; this one was not, purely because of its type.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ontology_rgat.benchmarks.px4_adapter import ShinPX4Adapter
from ontology_rgat.bridge import (BridgeError, GatewayTimeout,
                                  SimulatorFrameTimeout)
from ontology_rgat.ppo import recurrent_train


def _state():
    return {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "world": {"velocity": [0.0, 0.0, 0.0]}}


class _Bridge:
    def reset(self, seed, scenario="training_random_walk", **_kw):
        return _state()

    def step_velocity(self, _command):
        return _state()


class _Controller:
    def reset(self):
        pass

    def command(self, _action):
        return SimpleNamespace(as_array=lambda: np.zeros(4))


def _adapter(image_source):
    return ShinPX4Adapter(_Bridge(), image_source, _Controller())


# ------------------------------------------------------- typed, not bare

def test_the_frame_timeout_is_a_bridge_error():
    assert issubclass(SimulatorFrameTimeout, BridgeError)


@pytest.mark.parametrize("call", ["step", "reset"])
def test_a_stalled_camera_surfaces_as_infrastructure(call):
    def stalled():
        raise TimeoutError("no Shin benchmark camera frame received")

    adapter = _adapter(stalled)
    with pytest.raises(SimulatorFrameTimeout, match="delivered no frame"):
        adapter.step(np.zeros(4)) if call == "step" else adapter.reset(1)


def test_the_original_timeout_is_kept_as_the_cause():
    original = TimeoutError("no Shin benchmark camera frame received")

    def stalled():
        raise original

    with pytest.raises(SimulatorFrameTimeout) as caught:
        _adapter(stalled).step(np.zeros(4))
    assert caught.value.__cause__ is original


def test_a_healthy_camera_is_untouched():
    frame = np.zeros((4, 4), dtype=np.uint8)
    actor, state, _command = _adapter(lambda: frame).step(np.zeros(4))
    assert actor.image is frame and state == _state()


def test_only_a_frame_timeout_is_translated():
    """A camera that is misconfigured must not be dressed up as a stall."""
    def broken():
        raise ValueError("actor camera expected (320, 512), got (1, 1)")

    with pytest.raises(ValueError):
        _adapter(broken).step(np.zeros(4))


# --------------------------------------------- the retry actually owns it

def test_the_episode_retry_treats_it_like_a_gateway_timeout():
    """Same class of interruption, so the same bounded rebuild-and-retry."""
    import inspect

    body = inspect.getsource(recurrent_train.collect_episode_resilient)
    assert "SimulatorFrameTimeout" in body
    marker = body[body.index("recoverable = ("):]
    marker = marker[:marker.index("\n            recover")]
    for name in ("EntryResetError", "GatewayTimeout", "SimulatorFrameTimeout"):
        assert name in marker, name


def test_a_stalled_camera_is_retried_and_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def collect(_env, _model, _method, seed, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SimulatorFrameTimeout("the actor camera delivered no frame")
        return [{"reward": 0.0}], {"paper_success": 1, "seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = SimpleNamespace(
        cfg=SimpleNamespace(external={"episode_recoveries": 2}),
        recover_infrastructure=lambda: True,
        bridge=SimpleNamespace())
    rows, metric = recurrent_train.collect_episode_resilient(
        env, object(), "shin_se_fixed", 20121)
    assert calls["n"] == 2 and metric["seed"] == 20121 and rows


def test_the_retry_budget_still_bounds_a_camera_that_never_returns(monkeypatch):
    """A permanently wrong topic must still fail, not retry forever."""
    def collect(*_args, **_kwargs):
        raise SimulatorFrameTimeout("the actor camera delivered no frame")

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = SimpleNamespace(
        cfg=SimpleNamespace(external={"episode_recoveries": 2}),
        recover_infrastructure=lambda: True,
        bridge=SimpleNamespace())
    with pytest.raises(SimulatorFrameTimeout):
        recurrent_train.collect_episode_resilient(
            env, object(), "shin_se_fixed", 20121)


def test_a_gateway_timeout_keeps_working_as_before(monkeypatch):
    calls = {"n": 0}

    def collect(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise GatewayTimeout("PX4 gateway timeout after 20.00 s")
        return [{"reward": 0.0}], {"paper_success": 0}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = SimpleNamespace(
        cfg=SimpleNamespace(external={"episode_recoveries": 2}),
        recover_infrastructure=lambda: True,
        bridge=SimpleNamespace())
    recurrent_train.collect_episode_resilient(env, object(), "m", 1)
    assert calls["n"] == 2
