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
LAUNCHERS = ("run_two_pipeline.sh", "run_three_pipeline.sh",
             "run_shin2026_benchmark.sh")


# The path is quoted in the scripts, so the call reads
# ``"$workspace_root/scripts/sync_gateway.sh" --check``.
CHECK = re.compile(r'sync_gateway\.sh"?\s+--check')


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_launcher_refreshes_the_gateway_before_starting(name):
    script = _script(name)
    found = CHECK.search(script)
    assert found is not None, (
        f"{name} can start a gateway that does not match the repository")
    launch = script.index("exec ") if "exec " in script else len(script)
    assert found.start() < launch, (
        f"{name} syncs after it has already handed off")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_launcher_actually_syncs_when_the_check_fails(name):
    """``--check`` alone only reports; the repair call has to follow it."""
    script = _script(name)
    found = CHECK.search(script)
    assert found is not None, f"{name} never checks the gateway mirror at all"
    body = script[found.start():]
    body = body[:body.index("fi") + 2]
    repairs = [line for line in body.splitlines()
               if "sync_gateway.sh" in line and "--check" not in line]
    assert repairs, f"{name} detects a stale mirror and then uses it anyway"


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
