import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest

import ontology_rgat.bridge as bridge_module
from ontology_rgat.bridge import (BridgeError, EntryResetError, PX4Bridge,
                                  PX4EstimatorInvalid, PX4Failsafe,
                                  pacing_anchor_us)
from ontology_rgat_px4.protocol import (
    ProtocolError,
    VehicleSample,
    decode,
    encode,
    validate_action,
    validate_goto,
)
from ontology_rgat_px4.udp_server import DatagramServer
from ontology_rgat_px4.ros2_gateway import (ContinuousPx4Clock,
                                            action_age_seconds,
                                            advance_velocity_position_target,
                                            advance_pad_contact_latch,
                                            bounded_position_update,
                                            effective_px4_landed,
                                            failsafe_detail)


def test_failsafe_detail_classifies_only_benign_sitl_link_loss_as_recoverable():
    offboard = type("Flags", (), {
        "offboard_control_signal_lost": True,
        "manual_control_signal_lost": True,
        "gcs_connection_lost": True,
        "battery_warning": 0,
    })()
    detail = failsafe_detail(offboard, target="sitl")
    assert "offboard_control_signal_lost" in detail["reasons"]
    assert detail["recoverable_infrastructure"] is True
    assert failsafe_detail(offboard, target="hardware")[
        "recoverable_infrastructure"] is False

    # PX4 may clear the offboard bit before VehicleStatus.failsafe. The
    # autonomous SITL missing-input flags left in that callback window must
    # retain the same recovery classification observed in the real run.
    status_clear_race = type("Flags", (), {
        "auto_mission_missing": True,
        "manual_control_signal_lost": True,
        "gcs_connection_lost": True,
        "battery_warning": 0,
    })()
    detail = failsafe_detail(status_clear_race, target="sitl")
    assert detail["recoverable_infrastructure"] is True
    assert failsafe_detail(status_clear_race, target="hardware")[
        "recoverable_infrastructure"] is False

    no_flags = type("Flags", (), {"battery_warning": 0})()
    assert failsafe_detail(no_flags, target="sitl")[
        "recoverable_infrastructure"] is False

    hard = type("Flags", (), {
        "offboard_control_signal_lost": True,
        "local_position_invalid": True,
        "battery_warning": 0,
    })()
    detail = failsafe_detail(hard, target="sitl")
    assert detail["recoverable_infrastructure"] is False

    battery = type("Flags", (), {"battery_warning": 2})()
    detail = failsafe_detail(battery, target="sitl")
    assert "battery_warning_2" in detail["reasons"]
    assert detail["recoverable_infrastructure"] is False


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


def test_pad_contact_only_latches_after_takeoff_clearance():
    latched, clear = advance_pad_contact_latch(False, False, False, True)
    assert not latched and not clear       # parked, disarmed contact
    latched, clear = advance_pad_contact_latch(latched, clear, True, True, True)
    assert not latched and not clear       # armed but not off the roof yet
    latched, clear = advance_pad_contact_latch(latched, clear, True, False, True)
    assert not latched and not clear       # one takeoff bounce is not airborne
    latched, clear = advance_pad_contact_latch(
        latched, clear, True, False, False, False)
    assert not latched and not clear       # land flag alone precedes clearance
    latched, clear = advance_pad_contact_latch(
        latched, clear, True, False, False, True)
    assert not latched and clear           # takeoff has physically cleared it
    latched, clear = advance_pad_contact_latch(
        latched, clear, True, False, True, True)
    assert not latched and clear           # PX4 landed is not pad contact
    latched, clear = advance_pad_contact_latch(latched, clear, True, True, False)
    assert latched and clear               # the next contact is touchdown
    latched, clear = advance_pad_contact_latch(latched, clear, False, False)
    assert latched and clear               # sticky through motor disarm/bounce


def test_delayed_px4_landed_flag_is_not_ground_contact_in_air():
    assert not effective_px4_landed(True, True, [0.0, 0.0, 4.3])
    assert effective_px4_landed(True, True, [0.0, 0.0, 0.08])
    assert effective_px4_landed(True, False, None)


def test_sitl_deadman_uses_px4_lockstep_time_but_hardware_uses_wall_time():
    # One 20 ms simulated control period can take hundreds of wall milliseconds
    # under a rendered/lockstep Isaac run and must not cancel offboard control.
    assert action_age_seconds("sitl", 1_500_000_000, 1_000_000_000,
                              2_020_000, 2_000_000) == pytest.approx(0.02)
    assert action_age_seconds("hardware", 1_500_000_000, 1_000_000_000,
                              2_020_000, 2_000_000) == pytest.approx(0.5)
    # A newly delivered DDS batch can jump the PX4 stamp although this action
    # was received only 20 ms ago; that is not an offboard-loss condition.
    assert action_age_seconds("sitl", 1_020_000_000, 1_000_000_000,
                              9_424_000, 2_000_000) == pytest.approx(0.02)
    # When the client really stops, both clocks age and the smaller still trips.
    assert action_age_seconds("sitl", 3_000_000_000, 1_000_000_000,
                              4_000_000, 2_000_000) == pytest.approx(2.0)


def test_control_pacing_reanchors_after_a_missed_deadline():
    assert pacing_anchor_us(2_020_000, 2_016_000) == 2_020_000
    assert pacing_anchor_us(2_020_000, 2_300_000) == 2_300_000


def test_xrce_clock_rebase_preserves_simulated_time_deltas():
    epoch = 1_789_147_094_000_000
    clock = ContinuousPx4Clock(max_forward_gap_us=1_000_000)

    assert clock.update(epoch) == (epoch, False)
    assert clock.update(epoch + 20_000) == (epoch + 20_000, False)
    # Timesync loses its epoch offset, then reacquires it. Neither domain
    # switch may be billed as flight time.
    assert clock.update(2_000_000) == (epoch + 20_000, True)
    assert clock.update(2_020_000) == (epoch + 40_000, False)
    assert clock.update(epoch + 60_000) == (epoch + 40_000, True)
    assert clock.update(epoch + 80_000) == (epoch + 60_000, False)
    assert clock.discontinuities == 2


def test_bridge_pacing_reanchors_instead_of_failing_on_raw_clock_reset():
    bridge = object.__new__(PX4Bridge)
    bridge.control_period_us = 100_000
    bridge.last_px4_time_us = 1_789_147_094_000_000
    bridge.cfg = SimpleNamespace(timeout=.05)
    replies = iter([{"px4_time_us": 2_050_000},
                    {"px4_time_us": 2_100_000}])
    bridge.transact = lambda *_args, **_kwargs: next(replies)
    bridge.validate_state = lambda state: state

    state = bridge.pace_to_control_period({"px4_time_us": 2_000_000})

    assert state["px4_time_us"] == 2_100_000
    assert bridge.last_px4_time_us == 2_100_000


def test_bridge_pacing_still_rejects_a_genuine_stall():
    bridge = object.__new__(PX4Bridge)
    bridge.control_period_us = 100_000
    bridge.last_px4_time_us = 2_000_000
    bridge.cfg = SimpleNamespace(timeout=0.0)
    bridge.transact = lambda *_args, **_kwargs: {"px4_time_us": 2_000_000}
    bridge.validate_state = lambda state: state

    with pytest.raises(BridgeError, match="advanced only 0.0 ms"):
        bridge.pace_to_control_period({"px4_time_us": 2_000_000})


def test_reset_sends_motion_and_initial_condition_scales_separately():
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        wind_scale=1.0, pad_scale=.35, gnss_scale=1.0,
        reset_settle=0.0, auto_arm=False)
    sent = []
    bridge.transact = lambda kind, fields, expected: sent.append(
        (kind, fields, expected)) or {"status": "reset_complete"}
    bridge.wait_valid_state = lambda: {"estimator_valid": True}
    bridge.last_reset_ack = {}

    bridge.reset(12, scenario="circle", initial_condition_scale=0.0)

    kind, fields, expected = sent[0]
    assert kind == "reset" and expected == ("ack",)
    assert fields["pad_scale"] == pytest.approx(.35)
    assert fields["initial_condition_scale"] == pytest.approx(0.0)


def test_reset_rejects_invalid_initial_condition_scale_before_transmit():
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(wind_scale=1.0, pad_scale=.35, gnss_scale=1.0)
    bridge.transact = lambda *_args, **_kwargs: pytest.fail(
        "invalid reset must not be transmitted")
    with pytest.raises(BridgeError, match="initial-condition curriculum"):
        bridge.reset(12, initial_condition_scale=2.0)


def test_entry_gate_tolerates_brief_marker_dropout(monkeypatch):
    class Clock:
        value = -0.2

        def monotonic(self):
            self.value += 0.2
            return self.value

    clock = Clock()
    monkeypatch.setattr(bridge_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    states = iter([
        {"armed": True, "marker_quality": 0.5},
        {"armed": True, "marker_quality": 0.0},
        {"armed": True, "marker_quality": 0.0},
    ])
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        entry_timeout=10.0, arm_retry=2.0, entry_tolerance=0.5,
        entry_speed_tolerance=0.2, require_pad_in_view=True,
        entry_marker_memory=2.0, entry_settle=1.0)
    bridge.get_state = lambda: next(states)
    bridge.entry_state = lambda _state: (np.zeros(3), 0.0)

    state = bridge.wait_at_entry(np.zeros(3))

    assert state["marker_quality"] == 0.0


def test_entry_gate_uses_px4_time_for_marker_memory_and_settling(monkeypatch):
    class SlowRenderedClock:
        value = -5.0

        def monotonic(self):
            self.value += 5.0
            return self.value

    clock = SlowRenderedClock()
    monkeypatch.setattr(bridge_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    states = iter([
        {"armed": True, "marker_quality": 0.5, "px4_time_us": 0},
        {"armed": True, "marker_quality": 0.0, "px4_time_us": 250_000},
        {"armed": True, "marker_quality": 0.0, "px4_time_us": 500_000},
    ])
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        entry_timeout=100.0, arm_retry=2.0, entry_tolerance=0.5,
        entry_speed_tolerance=0.2, require_pad_in_view=True,
        entry_marker_memory=2.0, entry_settle=0.5)
    bridge.get_state = lambda: next(states)
    bridge.entry_state = lambda _state: (np.zeros(3), 0.0)

    state = bridge.wait_at_entry(np.zeros(3))

    assert state["px4_time_us"] == 500_000


def test_entry_gate_aborts_immediately_after_pad_contact(monkeypatch):
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        entry_timeout=90.0, arm_retry=2.0, entry_tolerance=0.5,
        entry_speed_tolerance=0.2, require_pad_in_view=True,
        entry_marker_memory=2.0, entry_settle=1.0)
    bridge.get_state = lambda: {
        "armed": False, "marker_quality": 0.0,
        "extra": {"pad_contact": True},
    }

    with pytest.raises(EntryResetError, match="contacted the pad"):
        bridge.wait_at_entry(np.zeros(3))


def test_entry_gate_propagates_px4_failsafe(monkeypatch):
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(entry_timeout=90.0, arm_retry=2.0)
    bridge.get_state = lambda: (_ for _ in ()).throw(
        PX4Failsafe(["offboard_control_signal_lost"], recoverable=True))

    with pytest.raises(PX4Failsafe, match="offboard_control_signal_lost"):
        bridge.wait_at_entry(np.zeros(3))


def _valid_bridge_state(*, armed, recoverable_failsafe):
    return {
        "position": [0.0, 0.0, 1.0], "velocity": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "acceleration": [0.0, 0.0, 0.0], "wind": [0.0, 0.0, 0.0],
        "aero_force": [0.0, 0.0, 0.0], "marker_quality": 1.0,
        "estimator_valid": True, "position_frame": "pad", "frame": "ENU_FLU",
        "pad": {}, "battery": {}, "gnss": {}, "armed": bool(armed),
        "extra": {
            "px4_failsafe": True,
            "px4_failsafe_detail": {
                "reasons": ["offboard_control_signal_lost"],
                "recoverable_infrastructure": bool(recoverable_failsafe),
            },
        },
    }


def test_bridge_ignores_only_disarmed_recoverable_sitl_link_clear_race():
    bridge = object.__new__(PX4Bridge)
    bridge.expected = {}

    state = bridge.validate_state(_valid_bridge_state(
        armed=False, recoverable_failsafe=True))
    assert state["extra"]["ignored_disarmed_link_failsafe"] is True

    with pytest.raises(PX4Failsafe):
        bridge.validate_state(_valid_bridge_state(
            armed=True, recoverable_failsafe=True))
    with pytest.raises(PX4Failsafe):
        bridge.validate_state(_valid_bridge_state(
            armed=False, recoverable_failsafe=False))


def test_physical_pad_contact_disarms_before_stopping_offboard():
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(outcome_settle_timeout=0.01)
    bridge.last_state = {
        "landed": True,
        "armed": True,
        "extra": {"pad_contact": True, "land_detector_authoritative": True},
    }
    calls = []
    bridge.disable_offboard = lambda: calls.append("offboard")
    bridge.disarm = lambda: calls.append("disarm")
    bridge.get_state = lambda: {"armed": False}

    assert bridge.stop_after_outcome()
    assert calls == ["disarm", "offboard"]


def test_bridge_waits_out_one_transient_estimator_invalid_sample(monkeypatch):
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(timeout=.1, estimator_warmup=.1)
    validations = iter([PX4EstimatorInvalid("temporary"), {"valid": True}])

    def validate(_state):
        result = next(validations)
        if isinstance(result, Exception):
            raise result
        return result

    bridge.validate_state = validate
    bridge.transact = lambda *_args, **_kwargs: {"sample": 2}

    assert bridge.validate_state_with_estimator_grace(
        {"sample": 1}) == {"valid": True}


def test_airborne_reset_hold_is_bounded_and_keeps_offboard_alive():
    bridge = object.__new__(PX4Bridge)
    bridge.last_state = {
        "position": [12.0, 0.0, 12.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "truth": {"valid": True, "position": [12.0, 0.0, 12.0]},
    }
    calls = []
    bridge.transact = lambda kind, payload, expected: calls.append(
        (kind, payload, expected)) or {"status": "goto_started"}

    bridge.hold_for_next_airborne_reset()

    kind, payload, expected = calls[0]
    assert kind == "goto" and expected == ("ack",)
    assert np.linalg.norm(payload["position"][:2]) == pytest.approx(9.0)
    assert payload["position"][2] == pytest.approx(8.0)
    assert payload["frame"] == "pad" and payload["hold_s"] == 120.0


def test_optical_position_update_is_bounded_around_dr_prediction():
    corrected = bounded_position_update([1.0, 2.0, 3.0], [11.0, 2.0, 3.0], 0.5)
    assert corrected == pytest.approx([1.5, 2.0, 3.0])
    assert bounded_position_update([1.0, 2.0, 3.0], [1.1, 2.0, 3.0], 0.5) \
        == pytest.approx([1.1, 2.0, 3.0])


def test_velocity_position_target_holds_integrates_and_obeys_world_bounds():
    held = advance_velocity_position_target(
        [2.0, 3.0, 4.5], [0.0, 0.0, 0.0], 0.1,
        floor_z_m=0.0, ceiling_z_m=12.0, world_radius_m=20.0)
    assert held == pytest.approx([2.0, 3.0, 4.5])
    moved = advance_velocity_position_target(
        held, [1.0, -2.0, -0.5], 2.0,
        floor_z_m=0.0, ceiling_z_m=12.0, world_radius_m=20.0)
    assert moved == pytest.approx([4.0, -1.0, 3.5])
    bounded = advance_velocity_position_target(
        [9.0, 0.0, 0.1], [10.0, 0.0, -10.0], 1.0,
        floor_z_m=0.05, ceiling_z_m=12.0, world_radius_m=10.0)
    assert bounded == pytest.approx([10.0, 0.0, 0.05])


@pytest.mark.parametrize("kwargs", (
    {"reference": [0.0, 0.0], "velocity_enu": [0.0, 0.0, 0.0], "dt_s": 0.1},
    {"reference": [0.0, 0.0, 1.0], "velocity_enu": [0.0, 0.0, 0.0], "dt_s": -0.1},
))
def test_velocity_position_target_rejects_invalid_updates(kwargs):
    with pytest.raises(ValueError):
        advance_velocity_position_target(
            **kwargs, floor_z_m=0.0, ceiling_z_m=12.0,
            world_radius_m=20.0)


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
