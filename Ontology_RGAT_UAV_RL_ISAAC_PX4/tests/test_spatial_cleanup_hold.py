"""Post-policy cleanup must not hold the last direct acceleration in flight."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ontology_rgat.bridge import PX4Bridge


def gateway_method(name, namespace):
    path = Path(__file__).resolve().parents[1] / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4/ros2_gateway.py"
    method = next(n for n in ast.walk(ast.parse(path.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_gateway_replaces_acceleration_before_sending_land():
    calls = []
    node = SimpleNamespace(last_command_seq=0, sample=SimpleNamespace(landed=False),
        _prepare_spatial_landing_hold=lambda: calls.append("brake_hold") or True,
        _vehicle_command=lambda command: calls.append(command),
        _send_ack=lambda *args: calls.append("ack"))
    method = gateway_method("_on_udp", {
        "Any": object, "VehicleCommand": SimpleNamespace(VEHICLE_CMD_NAV_LAND=21)})
    method(node, {"type": "disarm", "seq": 1})
    assert calls == ["brake_hold", 21, "ack"]


def test_explicit_disable_clears_the_temporary_spatial_cleanup_hold():
    node=SimpleNamespace(last_command_seq=0, goto_target_enu=np.array([1.,2.,3.]),
        sample=SimpleNamespace(extra={"spatial_cleanup_hold":{"active":True}}),
        _send_ack=lambda *args:None)
    gateway_method("_on_udp", {"Any":object})(node,{"type":"disable_offboard","seq":1})
    assert node.goto_target_enu is None and not node.sample.extra["spatial_cleanup_hold"]["active"]
    assert not node.offboard_enabled and node.last_action_ns==0


@pytest.mark.parametrize("nav,armed,retain", [(14,2,True),(18,2,False),(14,1,False)])
def test_gateway_autonomously_releases_cleanup_hold_after_mode_change(nav,armed,retain):
    node=SimpleNamespace(sample=SimpleNamespace(extra={"spatial_cleanup_hold":{"active":True}}),
        offboard_nav_state=14, goto_target_enu=np.array([1.,2.,3.]), px4_failsafe_detail={},
        _report_failsafe=lambda:None, _publish_flight_state=lambda:None)
    message=SimpleNamespace(nav_state=nav,arming_state=armed)
    gateway_method("_on_status",{})(node,message)
    assert node.sample.extra["spatial_cleanup_hold"]["active"] is retain
    assert (node.goto_target_enu is not None) is retain


@pytest.mark.parametrize("target,interface,nav,allowed", [
    ("sitl", "spatial_acceleration", 14, True),
    ("hardware", "spatial_acceleration", 14, False),
    ("sitl", "velocity_yaw_rate", 14, False),
    ("sitl", "spatial_acceleration", 18, False),
])
def test_cleanup_hold_is_own_state_only_and_never_requests_offboard(target, interface, nav, allowed):
    calls = []
    node = SimpleNamespace(command_interface=interface, offboard_nav_state=14,
        sample=SimpleNamespace(nav_state=nav, quaternion_enu_flu_wxyz=[1,0,0,0], extra={}),
        px4_world_position=np.array([1.,2.,3.]), last_action_ns=100,
        last_action_px4_time_us=50, velocity_acceleration_enu=[0,0,2.],
        offboard_enabled=True, _publish_position_setpoint=lambda:calls.append("position"),
        _publish_flight_state=lambda:calls.append("flight"))
    method = gateway_method("_prepare_spatial_landing_hold", {
        "cfg":SimpleNamespace(target=target), "np":np,
        "now_ns":lambda:200, "yaw_from_quat_wxyz":lambda q:0.})
    assert method(node) is allowed
    if allowed:
        assert calls == ["position", "flight"]
        assert node.last_action_ns == node.last_action_px4_time_us == 0
        assert node.velocity_acceleration_enu is None
        assert not node.offboard_enabled and not node.goto_pad_relative
        np.testing.assert_array_equal(node.goto_target_enu, [1.,2.,3.])
        assert not node.sample.extra["spatial_cleanup_hold"]["policy_transition"]
    else:
        assert not calls and node.last_action_ns == 100


def test_bridge_keeps_braking_stream_until_fresh_land_mode_then_disarms(monkeypatch):
    bridge = PX4Bridge.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(target="sitl")
    bridge.last_state = {"extra":{"control_mapping":{"interface":"spatial_acceleration"}}}
    calls=[]
    bridge.disable_offboard=lambda:calls.append("disable")
    bridge.disarm=lambda:calls.append("brake_then_land_or_ground_disarm")
    states=iter([dict(armed=True,landed=False,nav_state=14),
                 dict(armed=True,landed=False,nav_state=18),
                 dict(armed=True,landed=True,nav_state=18),
                 dict(armed=False,landed=True,nav_state=18)])
    def state():
        item=next(states)
        calls.append((item["armed"],item["landed"],item["nav_state"]))
        return item
    bridge.get_state=state
    monkeypatch.setattr("ontology_rgat.bridge.time.sleep",lambda _:None)
    assert bridge.stop_after_outcome(timeout=1.)
    assert calls == ["brake_then_land_or_ground_disarm", (True,False,14),
        (True,False,18), "disable", (True,True,18),
        "brake_then_land_or_ground_disarm", (False,True,18)]


def test_bridge_retries_lost_land_until_fresh_mode_confirmation(monkeypatch):
    bridge = PX4Bridge.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(target="sitl")
    bridge.last_state = {"extra":{"control_mapping":{"interface":"spatial_acceleration"}}}
    clock = [0.]
    requests = []
    disabled = []
    bridge.disarm = lambda: requests.append(clock[0])
    bridge.disable_offboard = lambda: disabled.append(clock[0])
    states = iter([(True,False,14),(True,False,14),(True,False,14),
                   (True,False,18),(True,False,18),(False,True,18)])
    def state():
        clock[0] += .6
        armed, landed, nav = next(states)
        return dict(armed=armed,landed=landed,nav_state=nav)
    bridge.get_state = state
    monkeypatch.setattr("ontology_rgat.bridge.time.monotonic",lambda:clock[0])
    monkeypatch.setattr("ontology_rgat.bridge.time.sleep",lambda _:None)
    assert bridge.stop_after_outcome(timeout=10.)
    assert requests == [0.,1.2]  # Not retried after AUTO.LAND confirmation.
    assert disabled == [2.4]


def test_bridge_lost_land_retries_do_not_extend_cleanup_deadline(monkeypatch):
    bridge = PX4Bridge.__new__(PX4Bridge)
    bridge.cfg = SimpleNamespace(target="sitl")
    bridge.last_state = {"extra":{"control_mapping":{"interface":"spatial_acceleration"}}}
    clock = [0.]
    requests = []
    bridge.disarm = lambda: requests.append(clock[0])
    bridge.disable_offboard = lambda: pytest.fail("Still OFFBOARD: keep braking heartbeat")
    bridge.get_state = lambda: dict(armed=True,landed=False,nav_state=14)
    monkeypatch.setattr("ontology_rgat.bridge.time.monotonic",lambda:clock[0])
    def advance(_):
        clock[0] += .5
    monkeypatch.setattr("ontology_rgat.bridge.time.sleep",advance)
    assert not bridge.stop_after_outcome(timeout=3.)
    assert requests == [0.,1.,2.] and clock[0] == 3.
