import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))
# The learner package. Added after the gateway path so a name that exists in
# both resolves to the gateway's, which is what the ROS node actually imports.
sys.path.insert(1, str(ROOT / "python"))
