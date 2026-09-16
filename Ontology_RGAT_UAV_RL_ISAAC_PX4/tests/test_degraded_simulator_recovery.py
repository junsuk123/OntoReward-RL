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


# --------------------------------------------------------------------------
# An aborted Isaac Sim must name its own crash.
#
# Isaac's stdout is buffered, so a Kit abort discards it and the stack log of
# the failed startup is empty -- which is exactly when the operator needs it.
# Kit's own session log survives the abort and records the assertion, so the
# failure report reads that instead of printing an empty tail.

_CRASHED_KIT_LOG = """\
2026-09-16T09:05:57Z [99ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash]  assertionCausedCrash = '1'
2026-09-16T09:05:57Z [114ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash]  lastAssertionCondition = 'm_owner == pthread_self()'
2026-09-16T09:05:57Z [114ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash]  lastAssertionFile = '../../../include/carb/thread/Mutex.h'
2026-09-16T09:05:57Z [115ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash]  lastAssertionLine = '158'
2026-09-16T09:05:57Z [116ms] [Warning] [carb.crashreporter-breakpad.plugin] [crash]  lastAssertionMessage = 'unlock() called by non-owning thread'
"""


def test_a_kit_abort_is_reported_from_kit_own_log(tmp_path, monkeypatch):
    kit_log = tmp_path / "kit_20260916_180537.log"
    kit_log.write_text(_CRASHED_KIT_LOG, encoding="utf-8")
    isaac_log = tmp_path / "isaac.log"
    isaac_log.write_text("", encoding="utf-8")   # the abort lost the buffer
    monkeypatch.setattr(stack_module, "_kit_session_log", lambda since: kit_log)

    entry = {"name": "isaac", "pid": 4321, "log": isaac_log, "started": 0.0,
             "process": SimpleNamespace(poll=lambda: -6)}
    report = stack_module.ExternalStack._startup_failure(
        "Isaac Sim + PX4 SITL", entry)

    assert "isaac (pid 4321) was killed by SIGABRT" in report
    assert "unlock() called by non-owning thread" in report
    assert "Mutex.h:158" in report
    assert str(kit_log) in report
    assert "aborted before its stdout was flushed" in report
    # The operator must not be told a benchmark bug caused an Isaac crash.
    assert "not a benchmark fault" in report


def test_a_clean_startup_exit_is_reported_without_a_kit_assertion(tmp_path,
                                                                  monkeypatch):
    kit_log = tmp_path / "kit_20260916_180624.log"
    kit_log.write_text("2026-09-16T09:06:24Z [1ms] [Info] [carb] all fine\n",
                       encoding="utf-8")
    isaac_log = tmp_path / "isaac.log"
    isaac_log.write_text("Isaac Python not found. Set ISAACSIM_PATH.\n",
                         encoding="utf-8")
    monkeypatch.setattr(stack_module, "_kit_session_log", lambda since: kit_log)

    entry = {"name": "isaac", "pid": 99, "log": isaac_log, "started": 0.0,
             "process": SimpleNamespace(poll=lambda: 1)}
    report = stack_module.ExternalStack._startup_failure("Isaac Sim + PX4 SITL",
                                                         entry)
    assert "exited with status 1" in report
    assert "Isaac Python not found" in report
    assert "not a benchmark fault" not in report


def test_the_failed_attempt_log_survives_the_relaunch(tmp_path):
    from ontology_rgat.stack import ExternalStack

    stack = ExternalStack.__new__(ExternalStack)
    stack.log_dir = tmp_path
    stack.root = tmp_path
    stack.managed = []
    (tmp_path / "isaac.log").write_text("first attempt output\n", encoding="utf-8")
    stack._launch("isaac", ["true"])
    assert (tmp_path / "isaac.previous.log").read_text() == "first attempt output\n"
    # A second relaunch keeps one slot, never an unbounded pile.
    (tmp_path / "isaac.log").write_text("second attempt output\n", encoding="utf-8")
    stack._launch("isaac", ["true"])
    assert (tmp_path / "isaac.previous.log").read_text() == "second attempt output\n"
    assert not list(tmp_path.glob("isaac.previous.previous*"))


def test_the_dead_process_is_named_even_when_another_one_is_waited_on(tmp_path,
                                                                     monkeypatch):
    # ``_wait_for`` waits on Isaac's banner, but the agent or a gateway can be
    # the process that died. Blaming Isaac there sends the operator to the
    # wrong log.
    from ontology_rgat.stack import ExternalStack, StackError

    stack = ExternalStack.__new__(ExternalStack)
    stack.log_dir = tmp_path
    stack.managed = [{"name": "agent", "pid": 7, "log": tmp_path / "agent.log",
                      "started": 0.0,
                      "process": SimpleNamespace(poll=lambda: 2)}]
    (tmp_path / "agent.log").write_text("agent refused the port\n", encoding="utf-8")
    monkeypatch.setattr(stack_module, "_kit_session_log", lambda since: None)

    with pytest.raises(StackError) as excinfo:
        stack._wait_for(lambda: False, 5.0, "Isaac Sim + PX4 SITL",
                        tmp_path / "isaac.log")
    assert "agent (pid 7) exited with status 2" in str(excinfo.value)
    assert "agent refused the port" in str(excinfo.value)
