import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac_sim"))

from operator_overlay import overlay_wanted  # noqa: E402


def test_overlay_is_skipped_when_nothing_is_watching():
    """The main loop must not draw a diagnostic frame nobody receives."""
    assert overlay_wanted(True, 0) is False


def test_overlay_is_drawn_for_an_attached_operator_tool():
    assert overlay_wanted(True, 1) is True
    assert overlay_wanted(True, 3) is True


def test_the_configuration_switch_beats_an_attached_subscriber():
    """A timed run pays for nothing, even with RViz open."""
    assert overlay_wanted(False, 5) is False
    assert overlay_wanted(False, 0) is False
    assert overlay_wanted(False, None) is False


def test_an_unknown_subscriber_count_keeps_the_operator_view():
    """Losing the view silently is worse than paying for it."""
    assert overlay_wanted(True, None) is True
