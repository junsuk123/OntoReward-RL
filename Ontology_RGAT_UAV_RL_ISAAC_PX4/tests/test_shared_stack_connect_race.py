"""A worker must not open a bridge into a peer's half-finished restart.

Parallel arms share one Isaac/PX4 stack. When one worker cycles it, the
gateway is stopped for the minutes Isaac needs to reload the world. A second
worker that opens a bridge in that window sends its hello to nothing, and the
120 s setup budget -- sized for a boot already under way, not for a full
rebuild -- expires on a simulator that was never going to answer. That killed a
48-episode run at the moment the second arm started training, which is also the
moment where retrying is free: no episode, reward or label exists yet.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from ontology_rgat import stack as stack_module
from ontology_rgat.bridge import GatewayTimeout
from ontology_rgat.ppo.recurrent_train import open_live_env_resilient
from ontology_rgat.stack import ExternalStack


@pytest.fixture(autouse=True)
def _clear_stack():
    previous = stack_module.current()
    yield
    stack_module.current(previous)


def _bare_stack() -> ExternalStack:
    stack = ExternalStack.__new__(ExternalStack)
    stack._restart_lock = threading.RLock()
    stack._shutdown_requested = False
    stack.timeouts = {"agent": 30.0, "isaac": 600.0, "gateway": 60.0}
    stack.generation = 0
    stack.managed = []
    stack.adopted_simulator = False
    return stack


def test_wait_for_restart_blocks_until_the_cycle_finishes():
    stack = _bare_stack()
    released = threading.Event()

    def restart():
        with stack._restart_lock:
            time.sleep(0.2)
            stack.generation += 1
            released.set()

    worker = threading.Thread(target=restart)
    worker.start()
    time.sleep(0.05)                      # let the restarting thread take the lock
    assert stack.wait_for_restart(timeout=5.0) is True
    assert released.is_set(), "wait_for_restart returned mid-restart"
    worker.join()


def test_wait_for_restart_gives_up_rather_than_blocking_forever():
    stack = _bare_stack()
    # The lock is reentrant, so the waiter has to be a different thread to see
    # it held -- which is also the only case that occurs in the pipeline.
    stack._restart_lock.acquire()
    try:
        outcome: list[bool] = []
        probe = threading.Thread(
            target=lambda: outcome.append(stack.wait_for_restart(timeout=0.1)))
        probe.start()
        probe.join()
        assert outcome == [False]
    finally:
        stack._restart_lock.release()


def test_the_environment_waits_for_a_peer_restart_before_saying_hello():
    """The whole point: no hello is sent while the gateway is stopped."""
    stack = _bare_stack()
    stack_module.current(stack)
    order: list[str] = []

    def restart():
        with stack._restart_lock:
            order.append("restart-start")
            time.sleep(0.2)
            stack.generation += 1
            order.append("restart-done")

    worker = threading.Thread(target=restart)
    worker.start()
    time.sleep(0.05)

    from ontology_rgat.benchmarks.live_env import LiveShinEnvironment

    environment = LiveShinEnvironment.__new__(LiveShinEnvironment)
    environment.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.02), benchmark_control={})
    environment.image_source = object()

    def bridge(cfg):
        order.append("hello")
        return SimpleNamespace(cfg=cfg)

    import ontology_rgat.benchmarks.live_env as live_env

    original_bridge = live_env.PX4Bridge
    original_adapter = live_env.ShinPX4Adapter
    live_env.PX4Bridge = bridge
    live_env.ShinPX4Adapter = lambda *a, **k: SimpleNamespace()
    try:
        environment._connect()
    finally:
        live_env.PX4Bridge = original_bridge
        live_env.ShinPX4Adapter = original_adapter
    worker.join()

    assert order == ["restart-start", "restart-done", "hello"]
    # The bridge belongs to the simulator that is now running, not the one the
    # peer tore down, so a later fault of its own is not charged to this worker.
    assert environment._stack_generation == 1


def test_opening_the_environment_survives_one_shared_stack_interruption(capsys):
    stack = _bare_stack()
    restarts: list[int] = []
    stack.restart_if_generation = lambda generation: (
        restarts.append(int(generation)) or True)
    stack_module.current(stack)
    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise GatewayTimeout("PX4 gateway timeout after 120.00 s (hello seq=1).")
        return "environment"

    assert open_live_env_resilient(factory, "shin_se_onto_rgat_recovery") == "environment"
    assert attempts["count"] == 2
    assert restarts == [0]
    assert "could not open the live environment" in capsys.readouterr().out


def test_a_simulator_that_never_comes_back_still_fails_the_run():
    """The retry is bounded: a dead stack must not loop rebuilding itself."""
    stack = _bare_stack()
    stack.restart_if_generation = lambda generation: True
    stack_module.current(stack)
    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        raise GatewayTimeout("PX4 gateway timeout after 120.00 s (hello seq=1).")

    with pytest.raises(GatewayTimeout):
        open_live_env_resilient(factory, "shin_se_fixed", attempts=3)
    assert attempts["count"] == 3


def test_a_failed_hello_releases_the_local_udp_port():
    """Otherwise the retry cannot bind the port the dead bridge still holds."""
    import socket

    from ontology_rgat.bridge import PX4Bridge

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    cfg = SimpleNamespace(
        external=SimpleNamespace(
            local_host="127.0.0.1", local_port=free_port,
            gateway_host="127.0.0.1", gateway_port=free_port + 1,
            protocol_version=1, timeout=0.05, setup_timeout=0.05,
            target="sitl"),
        rl=SimpleNamespace(collective_span=1.0, max_roll_pitch=0.5,
                           max_yaw_rate=1.0),
        sim=SimpleNamespace(dt=0.02),
        get=lambda key, default=None: {"external": {"enabled": True}}.get(
            key, default))

    with pytest.raises(GatewayTimeout):
        PX4Bridge(cfg)
    # Binding again is the observable proof that the first socket was closed.
    second = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        second.bind(("127.0.0.1", free_port))
    finally:
        second.close()
