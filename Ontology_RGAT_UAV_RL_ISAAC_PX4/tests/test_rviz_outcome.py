from types import SimpleNamespace

import numpy as np

from ontology_rgat.viz.rviz import _outcome_style
from ontology_rgat.viz.rviz import RvizPublisher


def test_rviz_outcome_colours_are_unambiguous():
    assert _outcome_style("running") is None

    success, green = _outcome_style("success")
    failure, red = _outcome_style("unsafe_touchdown")
    timeout, timeout_red = _outcome_style("timeout")
    uncertain, amber = _outcome_style("unconfirmed_success")

    assert success == "LANDING SUCCESS" and green[1] > green[0]
    assert failure == "LANDING FAILED" and red[0] > red[1]
    assert timeout == "LANDING FAILED" and timeout_red == red
    assert uncertain == "RESULT UNCONFIRMED"
    assert amber[0] > amber[1] > amber[2]


class _Message:
    ADD = 0
    CUBE = 1
    SPHERE = 2
    ARROW = 3
    CYLINDER = 4
    LINE_LIST = 5
    TEXT_VIEW_FACING = 6

    def __getattr__(self, name):
        value = _Message()
        setattr(self, name, value)
        return value


class _MarkerArray:
    def __init__(self):
        self.markers = []


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_terminal_scene_publishes_banner_hit_and_miss_line():
    rviz = RvizPublisher.__new__(RvizPublisher)
    rviz.cfg = SimpleNamespace(criteria=SimpleNamespace(xy=0.35))
    rviz.opt = SimpleNamespace(pad_frame="landing_pad")
    rviz.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 0)))
    rviz.m = {"Marker": _Message, "MarkerArray": _MarkerArray, "Point": _Message}
    rviz.scene_pub = _Publisher()

    log = SimpleNamespace(
        x=[np.zeros(13)], wind=[np.zeros(3)], aero_force=[np.zeros(3)],
        phi=[np.nan], t=[2.0], pad_speed=[1.5], closing_speed=[0.2],
        marker_quality=[0.8], gnss_quality=[0.7], hover_seconds_left=[12.0])
    cur = SimpleNamespace(meas={"pos": np.array([0.5, 0.0, 0.05])})
    info = {
        "status": "unsafe_touchdown",
        "diag": {"ground_truth": {
            "valid": True, "position": np.array([0.5, 0.0, 0.05])}},
    }

    rviz._publish_scene(log, cur, info)

    markers = rviz.scene_pub.messages[-1].markers
    outcome = [marker for marker in markers if marker.ns == "landing_outcome"]
    assert len(outcome) == 3
    assert outcome[0].text.startswith("LANDING FAILED")
    assert "0.50 m" in outcome[0].text
    assert outcome[0].color.r > outcome[0].color.g
