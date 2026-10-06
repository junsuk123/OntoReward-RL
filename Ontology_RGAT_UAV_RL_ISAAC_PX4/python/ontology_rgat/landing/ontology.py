"""The typed ontology, extensible by what the task physically contains.

The planar reference declares nine nodes over five relations. Replicating those
nine per axis is a correct DIMENSIONAL extension but not a complete one: 3D
introduces phenomena the planar task has no element for, and a node that does
not exist cannot be attended to. Worse, the quantities describing those
phenomena sit in the packet, which only ``ppo_vector_canonical`` reads -- the
flat and R-GAT arms see the graph alone. Leaving them unmodelled hands the
BASELINE exclusive information in 3D that it did not have in 2D, which is the
opposite of what the comparison is supposed to test.

Extensions are opt-in and append; the nine base nodes keep their indices, so
``DescentEligibility`` stays at 7 for the actor's descent gate and the planar
schema hash is untouched.

Each extension here is justified by a measurement, not by anticipation:

``DisturbanceEstimate``
    3D samples a constant per-episode external force of up to 0.75 N per axis
    -- 0.5 m/s^2 against a 2.5 m/s^2 authority. The causal observer in
    ``spatial.core.Estimator`` recovers it from own state alone (mean error
    0.030 m/s^2 against a true magnitude of 0.490, per-axis correlation
    0.998-0.999). The planar plant has no external force at all.

``MeasurementLatency``
    3D delivers optical solves 75 ms after capture, so the capture-aligned
    measurement and the estimator's current bearing stop coinciding. The
    planar task has no transport delay.

Yaw was considered and REJECTED: measured drift over 2400 steps was 0.03 deg
mean and 0.29 deg maximum, so there is nothing for a node to represent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

#: Upstream's nine, in their declared order. Indices are load-bearing.
BASE_NODES: tuple[str, ...] = (
    "PadVisibility", "PadMotion", "DroneTranslation", "DroneAttitude",
    "RelativeTracking", "TrackingCorrection", "ViewRecovery",
    "DescentEligibility", "LandingInhibit",
)
RELATION_NAMES: tuple[str, ...] = (
    "informs", "affects_visibility", "supports", "inhibits", "self")
BASE_EDGES: tuple[tuple[str, str, str], ...] = (
    ("PadMotion", "informs", "RelativeTracking"),
    ("DroneTranslation", "informs", "RelativeTracking"),
    ("DroneAttitude", "affects_visibility", "PadVisibility"),
    ("DroneTranslation", "affects_visibility", "PadVisibility"),
    ("PadMotion", "informs", "TrackingCorrection"),
    ("RelativeTracking", "informs", "TrackingCorrection"),
    ("PadVisibility", "informs", "ViewRecovery"),
    ("DroneAttitude", "informs", "ViewRecovery"),
    ("RelativeTracking", "informs", "ViewRecovery"),
    ("PadVisibility", "supports", "DescentEligibility"),
    ("RelativeTracking", "informs", "DescentEligibility"),
    ("DroneTranslation", "informs", "DescentEligibility"),
    ("DroneAttitude", "informs", "DescentEligibility"),
    ("PadVisibility", "informs", "LandingInhibit"),
    ("RelativeTracking", "informs", "LandingInhibit"),
    ("DroneTranslation", "informs", "LandingInhibit"),
    ("LandingInhibit", "inhibits", "DescentEligibility"),
)
#: Four grouped readouts over the base nodes.
BASE_READOUT_GROUPS: tuple[tuple[int, ...], ...] = ((0, 6), (1, 4, 5), (2, 3), (7, 8))


@dataclass(frozen=True)
class Extension:
    name: str
    node_class: str
    edges: tuple[tuple[str, str, str], ...]
    #: Index of the base readout group this node joins, so the four-group
    #: residual readout keeps its shape.
    readout_group: int


EXTENSIONS: tuple[Extension, ...] = (
    Extension(
        "DisturbanceEstimate", "MotionEstimate",
        (("DroneTranslation", "informs", "DisturbanceEstimate"),
         ("DisturbanceEstimate", "informs", "RelativeTracking"),
         ("DisturbanceEstimate", "informs", "TrackingCorrection"),
         ("DisturbanceEstimate", "inhibits", "DescentEligibility")),
        readout_group=1),
    Extension(
        "MeasurementLatency", "ObservationState",
        (("PadVisibility", "informs", "MeasurementLatency"),
         ("MeasurementLatency", "affects_visibility", "PadVisibility"),
         ("MeasurementLatency", "informs", "RelativeTracking"),
         ("MeasurementLatency", "inhibits", "DescentEligibility")),
        readout_group=0),
)
EXTENSION_NAMES: tuple[str, ...] = tuple(item.name for item in EXTENSIONS)

#: The planar task has neither an external force nor a transport delay.
PLANAR_EXTENSIONS: tuple[str, ...] = ()
#: The spatial task has both.
SPATIAL_EXTENSIONS: tuple[str, ...] = EXTENSION_NAMES


@dataclass(frozen=True)
class GraphSchema:
    node_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    declared_edges: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str, str], ...]
    readout_groups: tuple[tuple[int, ...], ...]
    extensions: tuple[str, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_names)


def _resolve(extensions) -> tuple[Extension, ...]:
    chosen = tuple(extensions)
    unknown = [name for name in chosen if name not in EXTENSION_NAMES]
    if unknown:
        raise ValueError(f"unknown ontology extension(s): {unknown}")
    # Declaration order, not call order, so the schema is a function of the set.
    return tuple(item for item in EXTENSIONS if item.name in chosen)


def schema(extensions=PLANAR_EXTENSIONS) -> GraphSchema:
    """Node/relation/edge structure for a task with these extensions."""
    chosen = _resolve(extensions)
    nodes = BASE_NODES + tuple(item.name for item in chosen)
    declared = BASE_EDGES + tuple(
        edge for item in chosen for edge in item.edges)
    for source, relation, target in declared:
        if source not in nodes or target not in nodes:
            raise ValueError(f"edge {(source, relation, target)} names an absent node")
        if relation not in RELATION_NAMES:
            raise ValueError(f"unknown relation {relation!r}")
    groups = [list(group) for group in BASE_READOUT_GROUPS]
    for offset, item in enumerate(chosen):
        groups[item.readout_group].append(len(BASE_NODES) + offset)
    edges = declared + tuple((node, "self", node) for node in nodes)
    return GraphSchema(nodes, RELATION_NAMES, declared, edges,
                       tuple(tuple(group) for group in groups),
                       tuple(item.name for item in chosen))


def schema_hash(graph: GraphSchema, *, feature_channels, version, decoding) -> str:
    """Digest of the structure a checkpoint was trained against."""
    return hashlib.sha256(json.dumps({
        "version": version, "nodes": graph.node_names,
        "relations": graph.relation_names, "edges": graph.edges,
        "features": tuple(feature_channels), "groups": graph.readout_groups,
        "packet_decoding": decoding,
    }, sort_keys=True).encode()).hexdigest()
