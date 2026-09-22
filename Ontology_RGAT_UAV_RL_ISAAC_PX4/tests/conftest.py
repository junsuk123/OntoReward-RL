import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))
# The learner package. Added after the gateway path so a name that exists in
# both resolves to the gateway's, which is what the ROS node actually imports.
sys.path.insert(1, str(ROOT / "python"))


def default_experiment_config() -> Path:
    """The experiment config a bare ``./run.sh`` runs.

    Read from the runner's own argparse default rather than named here, so an
    invariant checked against "the shipped configuration" follows the headline
    experiment when it changes. It moved from the six-deck two-arm comparison
    to the three-arm burst comparison on 2026-09-22, and every test that had
    hard-coded the old filename silently stopped guarding the default.
    """
    source = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    match = re.search(
        r'default=ROOT / "(config/experiments/[^"]+)"\s*\n\s*if primary_only',
        source)
    assert match, "the primary --config default moved; update this helper"
    return ROOT / match.group(1)
