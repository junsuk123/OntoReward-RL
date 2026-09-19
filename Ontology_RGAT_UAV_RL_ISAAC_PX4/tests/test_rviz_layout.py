import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_rviz_layout import build  # noqa: E402


def _groups(layout):
    return [d for d in layout["Visualization Manager"]["Displays"]
            if d.get("Class") == "rviz_common/Group"]


@pytest.mark.parametrize("pairs", [1, 2, 3, 4, 6])
def test_every_flying_pair_gets_its_own_displays(pairs):
    """A run sized by the machine must not show only the first two pairs."""
    layout = build(pairs)
    groups = _groups(layout)
    assert len(groups) == pairs
    for index in range(pairs):
        text = json.dumps(groups[index])
        assert f"/landing_rl/pair_{index}/scene" in text
        assert f"/landing_pair_{index}/uav/perception/landing_camera/annotated" in text


def test_no_pair_subscribes_to_another_pairs_topics():
    layout = build(4)
    for index, group in enumerate(_groups(layout)):
        found = set(re.findall(r"/landing_rl/pair_(\d+)/", json.dumps(group)))
        assert found == {str(index)}


def test_pairs_are_labelled_in_the_blocks_the_learner_assigns_them_in():
    """_training_pair_replicas splits {arm0: [0,1], arm1: [2,3]}, not alternating."""
    names = [group["Name"] for group in _groups(build(4))]
    assert "Baseline" in names[0] and "Baseline" in names[1]
    assert "Proposed" in names[2] and "Proposed" in names[3]


def test_the_transform_display_lists_every_pairs_frames():
    layout = build(4)
    tf = next(d for d in layout["Visualization Manager"]["Displays"]
              if isinstance(d.get("Frames"), dict))
    for index in range(4):
        assert tf["Frames"][f"landing_pad_{index}"] == {"Value": True}
        assert tf["Frames"][f"uav_body_{index}"] == {"Value": True}
    assert "landing_pad_4" not in tf["Frames"]


def test_a_zero_pair_layout_is_refused():
    with pytest.raises(ValueError):
        build(0)
