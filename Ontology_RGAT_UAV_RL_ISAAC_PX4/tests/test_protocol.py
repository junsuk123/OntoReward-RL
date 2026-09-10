import json
import socket

import pytest

from ontology_rgat_px4.protocol import (
    ProtocolError,
    VehicleSample,
    decode,
    encode,
    validate_action,
    validate_goto,
)
from ontology_rgat_px4.udp_server import DatagramServer


def test_protocol_roundtrip():
    message = {"v": 1, "type": "action", "seq": 7, "action": [0.0, -1.0, 1.0, 0.2]}
    assert decode(encode(message))["seq"] == 7
    assert validate_action(message) == (0.0, -1.0, 1.0, 0.2)


@pytest.mark.parametrize("action", ([0, 0, 0], [0, 0, 0, 1.01], [0, float("nan"), 0, 0]))
def test_invalid_actions(action):
    with pytest.raises(ProtocolError):
        validate_action({"action": action})


def test_json_nan_is_never_emitted():
    with pytest.raises(ValueError):
        encode({"v": 1, "type": "state", "seq": 1, "bad": float("nan")})


def test_vehicle_sample_schema():
    message = VehicleSample(estimator_valid=True).to_message(1, 2, 1)
    assert message["frame"] == "ENU_FLU"
    assert message["ack_seq"] == 1
    assert len(message["quaternion_wxyz"]) == 4


def test_offboard_command_is_versioned():
    assert decode(encode({"v": 1, "type": "enable_offboard", "seq": 9}))["seq"] == 9


def test_async_reply_can_be_pinned_to_its_requesting_peer():
    """A read-only probe must not steal a delayed flight-control reply."""
    received = []
    server = DatagramServer("127.0.0.1", 0, 1, received.append)
    controller = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    controller.settimeout(0.2)
    probe.settimeout(0.05)
    endpoint = server.socket.getsockname()
    try:
        controller.sendto(encode({"v": 1, "type": "state", "seq": 1}), endpoint)
        server.poll()
        controller_peer = controller.getsockname()
        probe.sendto(encode({"v": 1, "type": "state", "seq": 2}), endpoint)
        server.poll()

        server.send({"v": 1, "type": "ack", "seq": 3, "ack_seq": 1},
                    peer=controller_peer)
        reply, _ = controller.recvfrom(4096)
        assert decode(reply)["ack_seq"] == 1
        with pytest.raises(TimeoutError):
            probe.recvfrom(4096)
    finally:
        controller.close()
        probe.close()
        server.close()


def test_goto_is_a_versioned_command():
    message = {"v": 1, "type": "goto", "seq": 3, "position": [1.0, -2.0, 4.0], "yaw": 0.2}
    assert decode(encode(message))["type"] == "goto"
    request = validate_goto(message)
    assert request.position_enu == (1.0, -2.0, 4.0)
    assert request.yaw_enu_rad == 0.2


@pytest.mark.parametrize("position", (
    [0.0, 0.0, 0.0],        # on the ground
    [0.0, 0.0, -1.0],       # below the ground
    [0.0, 0.0, 99.0],       # above the city
    [200.0, 0.0, 4.0],      # beyond the block the deck laps
    [float("inf"), 0.0, 4.0],
))
def test_goto_refuses_positions_outside_the_city(position):
    with pytest.raises(ProtocolError):
        validate_goto({"position": position})


@pytest.mark.parametrize("hold_s", (0.0, -1.0, 600.0, float("nan"), "soon"))
def test_goto_hold_is_bounded(hold_s):
    with pytest.raises(ProtocolError):
        validate_goto({"position": [0.0, 0.0, 4.0], "hold_s": hold_s})
