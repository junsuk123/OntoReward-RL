"""Every launcher must refresh the gateway mirror before it boots a stack.

The gateway the stack runs is the ASCII-path mirror's copy, not this
repository's. A stale mirror starts cleanly and only then refuses the protocol
check, so the cost of forgetting is an Isaac boot plus the two relaunches
``ExternalStack`` makes before it gives up.

On 2026-09-22 a new deck scenario was added to ``protocol.py`` and ``./run.sh``
failed exactly that way: ``scripts/run_two_pipeline.sh`` -- the launcher a bare
``./run.sh`` uses -- was the only one of the three that skipped the sync the
other two have always done.
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Since 2026-11-20 the one launcher is ontology_rgat.live_stack (the shell launchers of the
# archived pipelines are in archive/non_orl_20261120.tar.zst).
LAUNCHER = ROOT / "python" / "ontology_rgat" / "live_stack.py"


def test_the_launcher_refreshes_the_gateway_before_starting():
    source = LAUNCHER.read_text(encoding="utf-8")
    check = source.index('"scripts/sync_gateway.sh"), "--check"')
    assert check < source.index("stack.start()"), "the gateway is synced after the stack starts"


def test_the_launcher_actually_syncs_when_the_check_fails():
    """``--check`` alone only reports; the repair call has to follow it."""
    source = LAUNCHER.read_text(encoding="utf-8")
    body = source[source.index('"--check"'):source.index("stack = ExternalStack(")]
    assert "if check.returncode:" in body
    assert 'subprocess.run([str(ROOT / "scripts/sync_gateway.sh")], check=True)' in body


def test_the_deck_vocabulary_the_gateway_validates_matches_the_simulator():
    """What the stale mirror got wrong: the two scenario lists must agree."""
    import sys
    sys.path.insert(0, str(ROOT / "isaac_sim"))
    from pad_motion import BENCHMARK_SCENARIOS

    protocol = (ROOT / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4"
                / "protocol.py").read_text(encoding="utf-8")
    missing = [deck for deck in BENCHMARK_SCENARIOS
               if f'"{deck}"' not in protocol]
    assert missing == [], (
        f"the gateway would reject decks the simulator can drive: {missing}")
