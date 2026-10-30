"""Current-observation 3D landing path (``spatial_current_semantics_v1``).

Separate from ``minimal/``, ``spatial/``, ``landing/`` and ``direct_policy/``:
it shares no checkpoint, hash or eligibility with them.  The identifiers are
defined here, not in ``contracts``, so importing the package stays free of
torch -- every ``direct_policy`` submodule import runs that package's
``__init__``, which loads its torch models.
"""
from __future__ import annotations

SCHEMA_ID = "spatial_current_semantics_v1"
OBSERVATION_VERSION = "spatial-current-observation/3"
FEATURE_VERSION = "spatial-current-features/1"
GRAPH_VERSION = "spatial-current-graph/1"
ACTION_VERSION = "spatial-current-action/1"
MODEL_VERSION = "spatial-current-models/1"

__all__ = ["ACTION_VERSION", "FEATURE_VERSION", "GRAPH_VERSION", "MODEL_VERSION",
           "OBSERVATION_VERSION", "SCHEMA_ID"]
