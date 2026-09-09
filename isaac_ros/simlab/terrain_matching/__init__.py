"""GNSS-denied, ontology-guided terrain visual matching research pipeline.

The package deliberately has no Isaac Sim or ROS dependency.  Camera frames can
come from the procedural generator today and from a ROS adapter later, while the
feature, reasoning, matching, and evaluation interfaces remain unchanged.
"""

from .pipeline import ExperimentRunner
from .schema import TerrainVisualObservation

__all__ = ["ExperimentRunner", "TerrainVisualObservation"]
