import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest

import ontology_rgat.bridge as bridge_module
from ontology_rgat.bridge import (BridgeError, EntryResetError, PX4Bridge,
                                  PX4EstimatorInvalid, PX4Failsafe,
                                  pacing_anchor_us)
from ontology_rgat.initialization import camera_centered_hover_offset
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
    bridge.transact = lambda kind, fields, expected, **kwargs: sent.append(
        (kind, fields, expected, kwargs)) or {"status": "reset_complete"}
    bridge.wait_valid_state = lambda: {"estimator_valid": True}
    bridge.last_reset_ack = {}

    bridge.reset(12, scenario="circle", initial_condition_scale=0.0)

    kind, fields, expected, options = sent[0]
    assert kind == "reset" and expected == ("ack",)
    # The reset may be the first request after a restart, so it is allowed to
    # outlast a simulator boot.
    assert options["timeout"] == pytest.approx(2.0)
    assert fields["pad_scale"] == pytest.approx(.35)
    assert fields["initial_condition_scale"] == pytest.approx(0.0)


def test_reset_rejects_invalid_initial_condition_scale_before_transmit():
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(wind_scale=1.0, pad_scale=.35, gnss_scale=1.0)
    bridge.transact = lambda *_args, **_kwargs: pytest.fail(
        "invalid reset must not be transmitted")
    with pytest.raises(BridgeError, match="initial-condition curriculum"):
        bridge.reset(12, initial_condition_scale=2.0)


def _entry_gate_bridge(states, *, require_pad_in_view=True):
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        entry_timeout=10.0, arm_retry=2.0, entry_tolerance=0.5,
        entry_speed_tolerance=0.2, require_pad_in_view=require_pad_in_view,
        entry_settle=0.5, entry_frame="pad", entry_view_margin=0.85,
        landing_camera={"resolution": [512, 320], "horizontal_fov_deg": 90.0,
                        "pitch_down_deg": 60.0,
                        "mount_translation_flu_m": [0.0, 0.0, -0.16]})
    bridge.get_state = lambda: next(states)
    return bridge


def test_entry_gate_opens_on_geometry_alone_and_ignores_marker_quality(
        monkeypatch, capsys):
    """One definition of initial visibility: the pad centre is in the frustum.

    The old gate accepted "a recent ArUco fix OR geometry", which put two
    incompatible meanings of "visible" into the same experiment. A 0.32 m tag
    is about ten pixels from 7.5 m, so the detector never fired there anyway.
    """
    class Clock:
        value = -0.2

        def monotonic(self):
            self.value += 0.2
            return self.value

    monkeypatch.setattr(bridge_module.time, "monotonic", Clock().monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    entry = camera_centered_hover_offset(7.55)
    states = iter([
        {"armed": True, "marker_quality": 0.0, "px4_time_us": 1_000_000 * n,
         "position": entry.tolist(), "velocity": [0.0, 0.0, 0.0],
         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "position_frame": "pad"}
        for n in range(6)
    ])
    bridge = _entry_gate_bridge(states)

    state = bridge.wait_at_entry(entry)

    assert state["marker_quality"] == 0.0
    assert state["px4_time_us"] == 1_000_000
    # No detector diagnostic is printed any more: there is no detector.
    assert "ArUco" not in capsys.readouterr().out


def test_entry_gate_rejects_a_recent_marker_fix_outside_the_camera_frame(
        monkeypatch):
    """A detection can no longer stand in for geometry, even a perfect one."""
    class Clock:
        value = -1.0

        def monotonic(self):
            self.value += 1.0
            return self.value

    monkeypatch.setattr(bridge_module.time, "monotonic", Clock().monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    # Hovering ahead of the deck: the forward/down camera looks away from it.
    entry = np.array([3.0, 0.0, 4.5])

    def states():
        n = 0
        while True:
            n += 1
            yield {"armed": True, "marker_quality": 1.0,
                   "px4_time_us": 1_000_000 * n, "position": entry.tolist(),
                   "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())

    with pytest.raises(EntryResetError,
                       match="geometric pad-centre view offset 2.11"):
        bridge.wait_at_entry(entry)


def test_entry_gate_uses_px4_time_for_settling(monkeypatch):
    class SlowRenderedClock:
        value = -5.0

        def monotonic(self):
            self.value += 5.0
            return self.value

    clock = SlowRenderedClock()
    monkeypatch.setattr(bridge_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    entry = camera_centered_hover_offset(4.5)
    states = iter([
        {"armed": True, "marker_quality": 0.0, "px4_time_us": stamp,
         "position": entry.tolist(), "velocity": [0.0, 0.0, 0.0],
         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "position_frame": "pad"}
        for stamp in (0, 250_000, 500_000)
    ])
    bridge = _entry_gate_bridge(states)
    bridge.cfg.entry_timeout = 100.0

    state = bridge.wait_at_entry(entry)

    assert state["px4_time_us"] == 500_000


def test_entry_gate_geometry_is_not_consulted_for_world_frame_entries(monkeypatch):
    class Clock:
        value = -1.0

        def monotonic(self):
            self.value += 1.0
            return self.value

    monkeypatch.setattr(bridge_module.time, "monotonic", Clock().monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    entry = camera_centered_hover_offset(4.5)

    def states():
        n = 0
        while True:
            n += 1
            yield {"armed": True, "marker_quality": 0.0,
                   "px4_time_us": 1_000_000 * n, "position": entry.tolist(),
                   "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_frame = "world"

    with pytest.raises(EntryResetError,
                       match="geometric pad-centre view offset n/a"):
        bridge.wait_at_entry(entry)


def test_entry_gate_aborts_immediately_after_pad_contact(monkeypatch):
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    bridge = object.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(
        entry_timeout=90.0, arm_retry=2.0, entry_tolerance=0.5,
        entry_speed_tolerance=0.2, require_pad_in_view=True,
        entry_settle=1.0)
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
    assert payload["frame"] == "pad" and payload["hold_s"] == 900.0


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


@pytest.mark.parametrize("hold_s", (0.0, -1.0, 1200.0, float("nan"), "soon"))
def test_goto_hold_is_bounded(hold_s):
    with pytest.raises(ProtocolError):
        validate_goto({"position": [0.0, 0.0, 4.0], "hold_s": hold_s})


def test_the_entry_timeout_names_the_condition_that_blocked_the_streak(
        monkeypatch):
    """A flickering limit cycle must not be reported as a compliant sample.

    The vehicle sits exactly on the entry pose and the pad is centred, but its
    speed alternates either side of the tolerance. Every individual bound is
    met at some point and the final sample looks healthy, yet the settle streak
    never completes -- which is what the operator has to be told.
    """
    class Clock:
        value = -0.2

        def monotonic(self):
            self.value += 0.2
            return self.value

    monkeypatch.setattr(bridge_module.time, "monotonic", Clock().monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        index = 0
        while True:
            index += 1
            # 0.5 s of simulated time per sample, so a 0.5 s settle would close
            # on two consecutive compliant samples if the speed ever allowed it.
            yield {"armed": True, "px4_time_us": 500_000 * index,
                   "position": entry.tolist(),
                   "velocity": [0.0, 0.0, 0.0 if index % 2 else 0.9],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_timeout = 3.0
    with pytest.raises(EntryResetError) as excinfo:
        bridge.wait_at_entry(entry)
    message = str(excinfo.value)
    assert "last sample" in message
    assert "longest hold" in message
    assert "speed was out of tolerance" in message
    assert "worst speed 0.90 m/s" in message
    # The last sample itself is compliant, so the old message would have shown
    # three healthy numbers and no cause at all.
    assert "speed 0.00 m/s" in message


def test_a_genuinely_settled_entry_reports_no_blocking_condition(monkeypatch):
    class Clock:
        value = -0.2

        def monotonic(self):
            self.value += 0.2
            return self.value

    monkeypatch.setattr(bridge_module.time, "monotonic", Clock().monotonic)
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _seconds: None)
    entry = camera_centered_hover_offset(7.55)
    states = iter([
        {"armed": True, "px4_time_us": 1_000_000 * n, "position": entry.tolist(),
         "velocity": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
         "position_frame": "pad"}
        for n in range(6)])
    bridge = _entry_gate_bridge(states)
    assert bridge.wait_at_entry(entry)["px4_time_us"] == 1_000_000


def _clock(step=0.2):
    class Clock:
        value = -step

        def monotonic(self):
            self.value += step
            return self.value
    return Clock().monotonic


def test_a_vehicle_that_never_arms_is_reported_as_an_arming_refusal(monkeypatch):
    """PX4 refusing to arm must not be reported as a failure to hold station.

    A parked vehicle's position offset is constant and large, so the old
    message blamed the pose and sent the operator after the wrong thing while
    the real cause -- command 400 rejected -- sat in the state's extra block.
    """
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock())
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        index = 0
        while True:
            index += 1
            yield {"armed": False, "px4_time_us": 100_000 * index,
                   "position": [6.06, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad",
                   "extra": {"last_command": [400, 1]}}

    bridge = _entry_gate_bridge(states())
    # The gate keeps requesting ARM during the grace window; the transport is
    # not under test here.
    bridge.transact = lambda *args, **kwargs: {}
    bridge.cfg.entry_timeout = 90.0
    bridge.cfg.entry_arm_grace = 20.0
    with pytest.raises(EntryResetError, match="refused to arm"):
        bridge.wait_at_entry(entry)


def test_the_arming_refusal_aborts_long_before_the_entry_budget(monkeypatch):
    """Eight bounded retries of a full budget is a quarter hour of nothing."""
    elapsed = []
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock())
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        index = 0
        while True:
            index += 1
            elapsed.append(index)
            yield {"armed": False, "px4_time_us": 100_000 * index,
                   "position": [6.06, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad",
                   "extra": {"last_command": [400, 1]}}

    bridge = _entry_gate_bridge(states())
    bridge.transact = lambda *args, **kwargs: {}
    bridge.cfg.entry_timeout = 99.0
    bridge.cfg.entry_arm_grace = 20.0
    with pytest.raises(EntryResetError, match="refused to arm"):
        bridge.wait_at_entry(entry)
    # The clock advances 0.2 s per sample, so the grace window is ~100 samples
    # and the full budget would have been ~495.
    assert len(elapsed) < 200


def test_an_accepted_arm_command_does_not_trip_the_refusal_path(monkeypatch):
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock())
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)
    states = iter([
        {"armed": True, "px4_time_us": 1_000_000 * n, "position": entry.tolist(),
         "velocity": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
         "position_frame": "pad", "extra": {"last_command": [400, 0]}}
        for n in range(8)])
    bridge = _entry_gate_bridge(states)
    assert bridge.wait_at_entry(entry)["px4_time_us"] == 1_000_000


def test_the_setpoint_is_re_aimed_when_only_the_deck_is_out_of_frame(monkeypatch):
    """Holding the commanded offset with the pad out of view is recoverable."""
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock())
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    # Hovering ahead of the deck: position and speed are fine, the forward/down
    # camera simply does not contain it.
    entry = np.array([3.0, 0.0, 4.5])
    attempts = []

    def states():
        index = 0
        while True:
            index += 1
            yield {"armed": True, "px4_time_us": 1_000_000 * index,
                   "position": entry.tolist(), "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_timeout = 40.0
    bridge.cfg.entry_view_retries = 2
    with pytest.raises(EntryResetError) as excinfo:
        bridge.wait_at_entry(entry, reissue=attempts.append)
    # Bounded, and reported rather than retried silently for ever.
    assert attempts == [1, 2]
    assert "re-aimed 2x" in str(excinfo.value)
    assert "view was out of tolerance" in str(excinfo.value)


def test_no_re_aim_happens_while_the_vehicle_is_still_travelling(monkeypatch):
    """Re-sending the setpoint mid-transit would restart the approach."""
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock())
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = np.array([3.0, 0.0, 4.5])
    attempts = []

    def states():
        index = 0
        while True:
            index += 1
            yield {"armed": True, "px4_time_us": 1_000_000 * index,
                   # Far from the target: offset, not view, is what blocks.
                   "position": [20.0, 0.0, 4.5], "velocity": [3.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_timeout = 30.0
    with pytest.raises(EntryResetError):
        bridge.wait_at_entry(entry, reissue=attempts.append)
    assert attempts == []


# --------------------------------------------------------------------------
# The entry setpoint chases a deck that is driving away from it.

def test_entry_setpoint_feeds_the_deck_velocity_forward():
    from ontology_rgat_px4.ros2_gateway import entry_feedforward_velocity

    # A position-only setpoint leaves PX4 to build the whole chase velocity
    # out of position error, which is a standing lag of v/MPC_XY_P.
    assert entry_feedforward_velocity(np.array([0.6, 0.0, 0.0])) == pytest.approx(
        [0.6, 0.0, 0.0])
    assert entry_feedforward_velocity(np.zeros(3)) == pytest.approx([0.0, 0.0, 0.0])


def test_entry_feedforward_refuses_an_untrustworthy_deck_twist():
    from ontology_rgat_px4.ros2_gateway import entry_feedforward_velocity

    # Above the rover's configured ceiling the sample is corrupt, and flying
    # it would carry the vehicle away from the deck it waits over. Falling
    # back to position-only is the previous, safe behaviour.
    assert entry_feedforward_velocity(np.array([50.0, 0.0, 0.0])) is None
    assert entry_feedforward_velocity(np.array([np.nan, 0.0, 0.0])) is None
    assert entry_feedforward_velocity(np.array([0.0, 0.0])) is None
    assert entry_feedforward_velocity(np.array([0.3, 0.0, 0.0]),
                                      max_speed_m_s=0.0) is None


def test_the_entry_lag_the_feedforward_removes_is_larger_than_the_tolerance():
    """The arithmetic that makes this a fix and not a preference.

    PX4's position loop settles a ramp input at ``v / MPC_XY_P``. The deck is
    configured for up to 0.60 m/s, the stock horizontal position gain is
    0.95 1/s, and the entry gate admits 0.90 m -- so the lag alone eats most
    of the tolerance before the rover turns or its seeded speed perturbation
    fires.
    """
    from config_loader import load_config
    from ontology_rgat.config import default_config
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    system = load_config(root / "config/shin2026-system.yaml")
    deck_speed = float(system["pad"]["speed_range_m_s"][1])
    tolerance = float(default_config().external["entry_tolerance"])
    mpc_xy_p = 0.95
    assert deck_speed / mpc_xy_p > 0.6 * tolerance


# --------------------------------------------------------------------------
# Setup traffic outlasts a simulator boot; flight steps do not.

class _CountingBridge:
    """A bridge whose socket never answers, to read back the budget used."""

    def __init__(self, timeout, setup_timeout):
        self.cfg = SimpleNamespace(timeout=timeout, setup_timeout=setup_timeout,
                                   protocol_version=1, gateway_host="127.0.0.1",
                                   gateway_port=1, local_host="127.0.0.1")
        self.sequence = 0
        self.sent = []

        class _Socket:
            def __init__(self, outer):
                self.outer = outer

            def sendto(self, payload, address):
                self.outer.sent.append(payload)

            def recv(self, size):
                raise socket.timeout()

        self.socket = _Socket(self)

    _setup_timeout = PX4Bridge._setup_timeout
    transact = PX4Bridge.transact


def _elapsed_budget(kind, **kwargs):
    import time as _time

    bridge = _CountingBridge(0.05, 0.4)
    started = _time.monotonic()
    with pytest.raises(BridgeError) as excinfo:
        bridge.transact(kind, {}, ("ack",), **kwargs)
    return _time.monotonic() - started, str(excinfo.value)


def test_a_flight_step_keeps_the_short_control_budget():
    elapsed, message = _elapsed_budget("action")
    assert elapsed < 0.3
    assert "after 0.05 s" in message


def test_setup_traffic_waits_out_a_booting_simulator():
    elapsed, message = _elapsed_budget("reset", timeout=0.4)
    assert elapsed >= 0.4
    assert "after 0.40 s" in message


def test_setup_timeout_is_never_shorter_than_the_control_budget():
    bridge = _CountingBridge(3.0, 1.0)
    assert bridge._setup_timeout() == 3.0
    bridge = _CountingBridge(2.0, 120.0)
    assert bridge._setup_timeout() == 120.0
    # A configuration that predates the setting keeps the old behaviour.
    bridge.cfg = SimpleNamespace(timeout=2.0)
    assert bridge._setup_timeout() == 2.0


def test_the_reconnect_and_reset_use_the_boot_budget():
    import inspect

    source = inspect.getsource(bridge_module.PX4Bridge)
    for call in ('self.transact("hello"', 'self.transact("reset"'):
        start = source.index(call)
        assert "_setup_timeout()" in source[start:start + 400], (
            f"{call} must outlast a simulator boot")
    # The bounded re-aim inside the entry deadline must not: one slow reply
    # there would consume the entry window itself.
    entry = source[source.index("def send_goto"):]
    assert entry[:entry.index("send_goto(")].count("_setup_timeout") == 0


def test_a_slow_stage_does_not_shorten_the_entry_manoeuvre(monkeypatch):
    """The entry budget is simulated time, so render load cannot shrink it.

    The settle streak has always been measured on PX4's clock. While the
    budget around it was wall time, adding a second rendered landing camera
    silently shortened the manoeuvre the gate was asking for: on the two-pair
    city stage a 99 s wall budget bought only a fraction of the simulated
    seconds the same number bought on a flat plane, and vehicles that were
    converging normally were cut off mid-settle -- reported as "longest hold
    0.98 s of 1.00 s" with every tolerance met at the final sample. Restarting
    a simulator for that also destroyed the other pair's episode.

    Here one simulated second costs twenty wall seconds. The vehicle is at the
    entry pose from the first sample and must still be handed over.
    """
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock(1.0))
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        index = 0
        while True:
            index += 1
            yield {"armed": True, "px4_time_us": 50_000 * index,
                   "position": entry.tolist(), "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_settle = 1.0
    bridge.cfg.entry_sim_budget = 60.0
    # Twenty-one samples at 1.0 s of wall clock each: the old wall-only gate
    # expired at ten with the vehicle sitting exactly on the entry pose.
    bridge.cfg.entry_timeout = 240.0

    state = bridge.wait_at_entry(entry)

    assert state["px4_time_us"] >= 1_000_000


def test_the_entry_budget_that_expired_names_its_clock(monkeypatch):
    """"Out of simulated time" and "out of wall time" need opposite responses.

    A vehicle that never converged is a vehicle; a stage that stopped
    advancing its clock is a simulator. The message has to say which, or the
    operator restarts the wrong thing.
    """
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock(0.01))
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        index = 0
        while True:
            index += 1
            yield {"armed": True, "px4_time_us": 1_000_000 * index,
                   "position": (entry + np.array([4.0, 0.0, 0.0])).tolist(),
                   "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_sim_budget = 5.0
    bridge.cfg.entry_timeout = 240.0
    with pytest.raises(EntryResetError) as excinfo:
        bridge.wait_at_entry(entry)
    message = str(excinfo.value)
    assert "5.0 simulated s" in message
    # The wall guard was nowhere near expiry, so it must not be the headline.
    assert "240.0 s wall" not in message
    assert "offset was out of tolerance" in message


def test_a_stage_that_stops_advancing_still_hits_the_wall_guard(monkeypatch):
    """A frozen PX4 clock cannot hold a worker open forever."""
    monkeypatch.setattr(bridge_module.time, "monotonic", _clock(1.0))
    monkeypatch.setattr(bridge_module.time, "sleep", lambda _s: None)
    entry = camera_centered_hover_offset(7.55)

    def states():
        while True:
            # The simulator answers, but its clock never moves.
            yield {"armed": True, "px4_time_us": 7_000_000,
                   "position": (entry + np.array([4.0, 0.0, 0.0])).tolist(),
                   "velocity": [0.0, 0.0, 0.0],
                   "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "position_frame": "pad"}

    bridge = _entry_gate_bridge(states())
    bridge.cfg.entry_sim_budget = 60.0
    bridge.cfg.entry_timeout = 12.0
    with pytest.raises(EntryResetError) as excinfo:
        bridge.wait_at_entry(entry)
    message = str(excinfo.value)
    assert "12.0 s wall" in message
    assert "slower than real time" in message
