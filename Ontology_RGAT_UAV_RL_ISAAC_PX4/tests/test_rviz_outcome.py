import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from ontology_rgat.viz.rviz import _outcome_style
from ontology_rgat.viz.rviz import RvizPublisher, RvizPublisherGroup
from ontology_rgat.rgat.fov_graph import (FOV_FEATURE_NAMES,
                                          FOVSemanticObservation,
                                          build_fov_graph)


ROOT = Path(__file__).resolve().parents[1]


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


def test_rviz_keeps_map_fixed_and_follows_the_landing_pad():
    profile = (ROOT / "rviz" / "ontology_rgat.rviz").read_text(encoding="utf-8")
    assert "Fixed Frame: map" in profile
    assert profile.count("Target Frame: landing_pad") == 3
    assert "Name: Deck track (pad-relative)" in profile
    assert profile.count("Depth: 1") == 7
    assert profile.count("Reliability Policy: Best Effort") == 7


def test_parallel_rviz_layout_has_exactly_two_primary_pairs_and_cameras():
    profile = (ROOT / "rviz" / "ontology_rgat_parallel.rviz").read_text(
        encoding="utf-8")
    assert "Fixed Frame: map" in profile
    for index in range(2):
        assert f"/landing_rl/pair_{index}/scene" in profile
        assert f"/landing_rl/pair_{index}/ontology" in profile
        assert f"/landing_rl/pair_{index}/uav_path" in profile
        assert f"/landing_rl/pair_{index}/pad_path" in profile
        assert (f"/landing_pair_{index}/uav/perception/landing_camera/annotated"
                in profile)
        assert f"Target Frame: landing_pad_{index}" in profile
    assert "Pair 1 · Baseline Shin SE fixed" in profile
    assert "Pair 2 · Proposed + Ontology-R-GAT FOV" in profile
    assert "No SE" not in profile
    assert "pair_2" not in profile
    assert "landing_pad_2" not in profile


def test_rviz_group_routes_reset_and_step_by_method():
    class Fake:
        def __init__(self):
            self.cleared = 0
            self.steps = []
            self.potential = None

        def clear_trails(self):
            self.cleared += 1

        def publish_benchmark_step(self, **kwargs):
            self.steps.append(kwargs)

        def close(self):
            pass

    first, second = Fake(), Fake()
    group = RvizPublisherGroup({"a": first, "b": second})
    group.clear_trails(method="b")
    group.publish_benchmark_step(method="a", state={}, scenario="x", step=1,
                                 dt=.1, geometric_in_fov=True, status="running")
    group.potential = "frozen"
    assert first.cleared == 0 and second.cleared == 1
    assert first.steps[0]["method"] == "a" and not second.steps
    assert first.potential == second.potential == "frozen"

    # Crossover evaluation follows the physical pair even when the method was
    # initially associated with another publisher.
    group.clear_trails(method="b", pair_index=0)
    group.publish_benchmark_step(
        method="b", pair_index=0, state={}, scenario="y", step=2,
        dt=.1, geometric_in_fov=True, status="running")
    assert first.cleared == 1 and second.cleared == 1
    assert first.steps[-1]["method"] == "b"
    assert first.steps[-1]["pair_index"] == 0


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


def test_benchmark_scene_and_telemetry_show_proposed_fov_branch():
    rviz = RvizPublisher.__new__(RvizPublisher)
    rviz.cfg = SimpleNamespace(criteria=SimpleNamespace(xy=0.35))
    rviz.opt = SimpleNamespace(
        pad_frame="landing_pad_1", publish_rate_hz=10.0, trail_length=20,
        deck_size_m=(1.5, 1.5), route_waypoints_enu_m=(),
        publish_ontology_graph=True, graph_origin_pad_m=(0.0, -2.6, 1.6),
        graph_scale_m=0.42)
    rviz.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 0)))
    rviz.m = {
        "Marker": _Message, "MarkerArray": _MarkerArray, "Point": _Message,
        "String": _Message, "Path": _Message, "PoseStamped": _Message,
    }
    rviz.scene_pub = _Publisher()
    rviz.graph_pub = _Publisher()
    rviz.telemetry_pub = _Publisher()
    rviz.uav_path_pub = _Publisher()
    rviz.pad_path_pub = _Publisher()
    rviz._uav_trail = []
    rviz._pad_trail = []
    rviz._counter = 0
    rviz._graph_every = 5
    rviz.potential = None
    rviz._broadcast_tf = lambda *_args: None
    rviz._publish_path = lambda *_args: None
    graph = build_fov_graph(
        FOVSemanticObservation(*([0.5] * len(FOV_FEATURE_NAMES))))

    rviz.publish_benchmark_step(
        state={"pad": {"position": [0.0, 0.0, 0.0]},
               "truth": {"valid": True, "position": [0.1, -0.2, 1.3]},
               "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        method="shin_se_onto_rgat_recovery", scenario="circle", step=1, dt=0.1,
        geometric_in_fov=True, status="running", reward=-0.07,
        reward_parts={"active_perception": -0.01, "fov_margin": 0.3,
                      "predicted_fov_unavailability": 0.6,
                      "ontology_fov_reward": -0.06},
        semantic_graph=graph)

    scene = rviz.scene_pub.messages[-1].markers
    hud = next(marker for marker in scene if marker.type == _Message.TEXT_VIEW_FACING)
    assert "PROPOSED · Shin + Ontology-R-GAT FOV" in hud.text
    assert "COMMON Shin" in hud.text
    assert "P(loss<=1s)=0.60" in hud.text
    assert "r_onto=-0.060" in hud.text
    assert rviz.graph_pub.messages[-1].markers
    telemetry = json.loads(rviz.telemetry_pub.messages[-1].data)
    assert telemetry["ontology_fov_branch_enabled"] is True
    assert telemetry["active_perception_reward"] == -0.01
    assert telemetry["predicted_fov_unavailability"] == 0.6
    assert telemetry["ontology_fov_reward"] == -0.06
