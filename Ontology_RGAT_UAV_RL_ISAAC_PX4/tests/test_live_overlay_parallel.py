import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

import live_overlay


class _Draw:
    def __init__(self):
        self.lines = None
        self.points = None

    def clear_lines(self):
        pass

    def clear_points(self):
        pass

    def draw_lines(self, starts, ends, colors, widths):
        self.lines = (starts, ends, colors, widths)

    def draw_points(self, positions, colors, sizes):
        self.points = (positions, colors, sizes)


class _Node:
    def __init__(self):
        self.subscriptions = []

    def create_subscription(self, *args):
        self.subscriptions.append(args)
        return args


def test_parallel_overlay_draws_all_pairs_and_resets_only_requested_pair(monkeypatch):
    draw = _Draw()
    monkeypatch.setattr(live_overlay, "_acquire", lambda: draw)
    node = _Node()
    overlay = live_overlay.LiveOverlay(
        node, telemetry_topics=[f"/landing_rl/pair_{i}/telemetry" for i in range(3)])
    pairs = [
        ((0.0, 0.0, 3.0), (0.0, 0.0, 0.0)),
        ((5.0, 0.0, 3.0), (5.0, 0.0, 0.0)),
        ((10.0, 0.0, 3.0), (10.0, 0.0, 0.0)),
    ]
    overlay.update_many(pairs)
    overlay.update_many(pairs)

    assert len(node.subscriptions) == 3
    assert len(draw.points[0]) == 6
    assert all(len(overlay._pair_uav_trails[index]) == 2 for index in range(3))

    overlay.reset(1)
    assert len(overlay._pair_uav_trails[0]) == 2
    assert 1 not in overlay._pair_uav_trails
    assert len(overlay._pair_uav_trails[2]) == 2

    overlay.update_many(pairs)
    assert len(overlay._pair_uav_trails[0]) == 3
    assert len(overlay._pair_uav_trails[1]) == 1
    assert len(overlay._pair_uav_trails[2]) == 3
