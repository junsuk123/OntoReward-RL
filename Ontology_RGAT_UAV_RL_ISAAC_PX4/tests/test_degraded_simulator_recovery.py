"""A degraded PX4 must be named, and must not consume the retry budget.

PX4 SITL that has run for hours starts failing preflight -- high accelerometer
bias, attitude failure -- and refuses ARM. ``ExternalStack.restart`` exists for
exactly that, but ``stop`` terminates only what the run started, so a simulator
the run merely adopted survives the cycle and is re-adopted. Retrying then
reproduces the identical failure once per bounded attempt, at a full entry
budget each, which is how a run spends a quarter of an hour to learn nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ontology_rgat import stack as stack_module
from ontology_rgat.bridge import ArmingRefused, BridgeError, EntryResetError
from ontology_rgat.ppo.recurrent_train import collect_episode_resilient


class _Env:
    def __init__(self, error, *, recoveries=2):
        self.cfg = SimpleNamespace(
            external=SimpleNamespace(reset_recoveries=recoveries))
        self.error = error
        self.attempts = 0
        self.recoveries = 0

    def recover_infrastructure(self):
        self.recoveries += 1
        return True


def _collect_raising(env, monkeypatch, error):
    def collect(*_args, **_kwargs):
        env.attempts += 1
        raise error
    monkeypatch.setattr(
        "ontology_rgat.ppo.recurrent_train.collect_episode", collect)


@pytest.fixture(autouse=True)
def _clear_stack():
    previous = stack_module.current()
    yield
    stack_module.current(previous)


def test_arming_refusal_on_an_adopted_simulator_fails_at_the_first_attempt(
        monkeypatch):
    stack_module.current(SimpleNamespace(adopted_simulator=True))
    env = _Env(ArmingRefused("PX4 refused to arm for 25 s"))
    _collect_raising(env, monkeypatch, env.error)
    with pytest.raises(ArmingRefused, match="adopted the simulator"):
        collect_episode_resilient(env, object(), "shin_se_fixed", 1)
    assert env.attempts == 1
    assert env.recoveries == 0


def test_arming_refusal_is_retried_when_the_run_owns_the_simulator(monkeypatch):
    """A restart can genuinely replace an owned SITL, so it is worth one."""
    stack_module.current(SimpleNamespace(adopted_simulator=False))
    env = _Env(ArmingRefused("PX4 refused to arm for 25 s"), recoveries=2)
    _collect_raising(env, monkeypatch, env.error)
    with pytest.raises(ArmingRefused):
        collect_episode_resilient(env, object(), "shin_se_fixed", 1)
    assert env.attempts == 3
    assert env.recoveries == 2


def test_an_ordinary_entry_failure_still_retries_on_an_adopted_simulator(
        monkeypatch):
    """Only the arming refusal is hopeless; a bad reset may well succeed next."""
    stack_module.current(SimpleNamespace(adopted_simulator=True))
    env = _Env(EntryResetError("PX4 did not hold the entry pose"), recoveries=2)
    _collect_raising(env, monkeypatch, env.error)
    with pytest.raises(EntryResetError):
        collect_episode_resilient(env, object(), "shin_se_fixed", 1)
    assert env.attempts == 3


def test_the_refusal_message_names_the_remedy(monkeypatch):
    stack_module.current(SimpleNamespace(adopted_simulator=True))
    env = _Env(ArmingRefused("PX4 refused to arm for 25 s"))
    _collect_raising(env, monkeypatch, env.error)
    with pytest.raises(ArmingRefused) as excinfo:
        collect_episode_resilient(env, object(), "shin_se_fixed", 1)
    message = str(excinfo.value)
    assert "restarting cannot replace the degraded PX4" in message
    assert "Stop the running Isaac/PX4 processes" in message


def test_can_replace_simulator_requires_an_owned_registered_stack():
    stack_module.current(None)
    assert stack_module.can_replace_simulator() is False
    stack_module.current(SimpleNamespace(adopted_simulator=True))
    assert stack_module.can_replace_simulator() is False
    stack_module.current(SimpleNamespace(adopted_simulator=False))
    assert stack_module.can_replace_simulator() is True


def test_arming_refusal_stays_an_entry_reset_error_for_existing_handlers():
    assert issubclass(ArmingRefused, EntryResetError)
    assert issubclass(ArmingRefused, BridgeError)
