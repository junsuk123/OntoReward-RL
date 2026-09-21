"""Two workers must never fly one UAV/UGV pair at the same time.

On 2026-09-21 the FOV-risk collector was handed pair 1 while the baseline PPO
arm flew replica pairs 0 and 1. Both bridges bound local UDP port 14653 -- the
bridge sets SO_REUSEADDR, so the second bind raised nothing -- and the kernel
delivered each gateway reply to one of them. The other saw only hello and
state timeouts, each of which rebuilt the shared simulator for every pair:
26 rebuilds in four hours, 21 collected episodes, and the arm dead each run.
"""
from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from ontology_rgat.bridge import BridgeError, GatewayTimeout, PX4Bridge
from run_three_pipeline import (_balanced_training_pair_assignment,
                                _reward_design_pair_indices,
                                _training_pair_replicas)

METHODS = ["shin_se_fixed", "shin_se_onto_rgat_recovery"]


def test_reward_design_stays_off_every_replica_pair_of_the_early_arm():
    for replicate in range(len(METHODS)):
        replicas = _training_pair_replicas(METHODS, 4, replicate)
        primary, _ = _balanced_training_pair_assignment(METHODS, 4, replicate)
        free = _reward_design_pair_indices(
            [METHODS[0]], replicas, primary, 4)
        # The arm flies two pairs; the primary alone used to count as occupied.
        assert len(replicas[METHODS[0]]) == 2
        assert set(free).isdisjoint(replicas[METHODS[0]])
        assert len(free) == 2
    replicas = _training_pair_replicas(METHODS, 4, 0)
    primary, _ = _balanced_training_pair_assignment(METHODS, 4, 0)
    assert _reward_design_pair_indices([METHODS[0]], replicas, primary, 4) == [2, 3]


def test_reward_design_with_two_pairs_takes_the_other_arm_pair():
    replicas = _training_pair_replicas(METHODS, 2, 0)
    primary, _ = _balanced_training_pair_assignment(METHODS, 2, 0)
    assert _reward_design_pair_indices([METHODS[0]], replicas, primary, 2) == [1]


def test_reward_design_refuses_when_every_pair_is_flown():
    replicas = _training_pair_replicas(METHODS, 4, 0)
    primary, _ = _balanced_training_pair_assignment(METHODS, 4, 0)
    with pytest.raises(RuntimeError, match="unoccupied physical pair"):
        _reward_design_pair_indices(METHODS, replicas, primary, 4)


# ------------------------------------------------------------ bridge guard

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _cfg(local_port: int, gateway_port: int, pair_index: int = 1):
    return SimpleNamespace(
        external=SimpleNamespace(
            local_host="127.0.0.1", local_port=local_port,
            gateway_host="127.0.0.1", gateway_port=gateway_port,
            protocol_version=1, timeout=0.05, setup_timeout=0.5,
            target="sitl", pair_index=pair_index),
        rl=SimpleNamespace(collective_span=1.0, max_roll_pitch=0.5,
                           max_yaw_rate=1.0),
        sim=SimpleNamespace(dt=0.02),
        get=lambda key, default=None: {"external": {"enabled": True}}.get(
            key, default))


class _AnsweringGateway:
    """Replies to every request with a bare state, like a live gateway."""

    def __init__(self, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.05)
        self.requests: list[dict] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                raw, peer = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            request = json.loads(raw.decode("utf-8"))
            self.requests.append(request)
            self.socket.sendto(json.dumps({
                "v": 1, "type": "state", "seq": len(self.requests),
                "ack_seq": request["seq"], "extra": {}}).encode("utf-8"), peer)

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1.0)
        self.socket.close()


def test_a_second_bridge_on_a_live_pair_port_fails_before_it_says_hello():
    local_port, gateway_port = _free_port(), _free_port()
    gateway = _AnsweringGateway(gateway_port)
    try:
        first = PX4Bridge(_cfg(local_port, gateway_port))
        try:
            hellos = len(gateway.requests)
            with pytest.raises(BridgeError, match="pair 1") as caught:
                PX4Bridge(_cfg(local_port, gateway_port))
            assert not isinstance(caught.value, GatewayTimeout)
            # Refused locally: the live session's gateway never heard from it.
            assert len(gateway.requests) == hellos
        finally:
            first.close()
        # Closing the live bridge returns the pair to whoever flies it next.
        PX4Bridge(_cfg(local_port, gateway_port)).close()
    finally:
        gateway.close()


def test_a_failed_hello_gives_the_pair_port_back_as_well():
    local_port, gateway_port = _free_port(), _free_port()
    with pytest.raises(GatewayTimeout):
        PX4Bridge(_cfg(local_port, gateway_port))
    gateway = _AnsweringGateway(gateway_port)
    try:
        PX4Bridge(_cfg(local_port, gateway_port)).close()
    finally:
        gateway.close()
