"""Seven-node deterministic graph over the registered policy observation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import GraphSchema, graph_schema
from .observation import PLANAR_NAMES, SPATIAL_NAMES


@dataclass(frozen=True)
class GraphState:
    features: np.ndarray                 # [nodes, channels]
    source: np.ndarray                   # [edges]
    target: np.ndarray
    relation: np.ndarray
    schema: GraphSchema

    def matlab_flatten(self) -> np.ndarray:
        """MATLAB ``X(:)`` for X=[channels,nodes]."""
        return np.asarray(self.features, dtype=np.float64).reshape(-1)


def topology(schema: GraphSchema | None = None):
    schema = schema or graph_schema(2)
    nodes = {name: index for index, name in enumerate(schema.nodes)}
    relations = {name: index for index, name in enumerate(schema.relations)}
    source = np.asarray([nodes[edge[0]] for edge in schema.edges], dtype=np.int64)
    target = np.asarray([nodes[edge[1]] for edge in schema.edges], dtype=np.int64)
    relation = np.asarray([relations[edge[2]] for edge in schema.edges], dtype=np.int64)
    return source, target, relation


def _put(features, node, primary, signed_value, secondary, validity, age):
    features[node] = (primary, signed_value, secondary, validity, age,
                      (node + 1) / features.shape[0])


def planar_graph(vector, *, camera_pitch_offset_rad=-math.pi / 6) -> GraphState:
    """Port of ``landing2d.graphstate.observationGraph``.

    ``vector`` is already normalized by the registered 12D transform.  Gamma
    contributes only the static optical-axis slope.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (12,):
        raise ValueError("planar graph expects the registered 12D vector")
    value = dict(zip(PLANAR_NAMES, vector))
    ex, height = value["relative_x"], value["relative_height"]
    rvx, pad_vx = value["relative_vx"], value["ugv_vx"]
    vz = value["drone_vz"]
    sin_pitch, cos_pitch = value["drone_sinTheta"], value["drone_cosTheta"]
    rate = value["drone_pitchRate"]
    vision_updated, vision_age = value["ugv_visionUpdated"], value["ugv_visionAge"]
    nav_valid, nav_age = value["drone_navigationValid"], value["drone_navigationAge"]
    slope = math.tan(-camera_pitch_offset_rad)
    cross_track = ex - slope * height
    closure_error = rvx - slope * vz
    features = np.zeros((7, 6), dtype=np.float64)
    _put(features, 0, max(abs(cross_track), abs(height)), cross_track, height,
         vision_updated, vision_age)
    _put(features, 1, abs(closure_error), closure_error, rvx,
         vision_updated, vision_age)
    _put(features, 2, abs(pad_vx), pad_vx, 0, vision_updated, vision_age)
    _put(features, 3, abs(vz), vz, height, nav_valid, nav_age)
    _put(features, 4, max(abs(sin_pitch), abs(rate)), sin_pitch, rate,
         nav_valid, nav_age)
    _put(features, 5, vision_updated, 1 - vision_age, cos_pitch, 1, vision_age)
    _put(features, 6, nav_valid, 1 - nav_age, 0, 1, nav_age)
    features = np.clip(features, -1, 1)
    schema = graph_schema(2)
    source, target, relation = topology(schema)
    return GraphState(features, source, target, relation, schema)


def spatial_graph(vector, *, camera_pitch_offset_rad=-math.pi / 6) -> GraphState:
    """21D spatial extension preserving all x/y channels instead of norms."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (21,):
        raise ValueError("spatial graph expects the registered 21D vector")
    v = dict(zip(SPATIAL_NAMES, vector))
    slope = math.tan(-camera_pitch_offset_rad)
    cross_x = v["relative_x"] - slope * v["relative_height"]
    cross_y = v["relative_y"]
    close_x = v["relative_vx"] - slope * v["drone_vz"]
    close_y = v["relative_vy"]
    # [primary, signed_x, secondary_x, signed_y, secondary_y, validity, age, type]
    f = np.zeros((7, 8), dtype=np.float64)
    def put(i, primary, sx, x2, sy, y2, valid, age):
        f[i] = (primary, sx, x2, sy, y2, valid, age, (i + 1) / 7)
    put(0, max(abs(cross_x), abs(cross_y), abs(v["relative_height"])),
        cross_x, v["relative_height"], cross_y, v["relative_height"],
        v["ugv_visionUpdated"], v["ugv_visionAge"])
    put(1, max(abs(close_x), abs(close_y)), close_x, v["relative_vx"],
        close_y, v["relative_vy"], v["ugv_visionUpdated"], v["ugv_visionAge"])
    put(2, max(abs(v["ugv_vx"]), abs(v["ugv_vy"])), v["ugv_vx"], 0,
        v["ugv_vy"], 0, v["ugv_visionUpdated"], v["ugv_visionAge"])
    put(3, abs(v["drone_vz"]), v["drone_vz"], v["relative_height"], 0, 0,
        v["drone_navigationValid"], v["drone_navigationAge"])
    put(4, max(abs(v["drone_sinPhi"]), abs(v["drone_sinTheta"]),
               abs(v["drone_rollRate"]), abs(v["drone_pitchRate"])),
        v["drone_sinTheta"], v["drone_pitchRate"], v["drone_sinPhi"],
        v["drone_rollRate"], v["drone_navigationValid"], v["drone_navigationAge"])
    put(5, v["ugv_visionUpdated"], 1-v["ugv_visionAge"],
        v["drone_cosTheta"], v["drone_sinPsi"], v["drone_cosPsi"], 1,
        v["ugv_visionAge"])
    put(6, v["drone_navigationValid"], 1-v["drone_navigationAge"],
        v["drone_yawRate"], 0, 0, 1, v["drone_navigationAge"])
    f = np.clip(f, -1, 1)
    schema = graph_schema(3)
    source, target, relation = topology(schema)
    return GraphState(f, source, target, relation, schema)


def shuffled_relations(schema: GraphSchema, seed: int) -> np.ndarray:
    """Fixed edge-relation shuffle preserving self edges and relation counts."""
    _, _, canonical = topology(schema)
    self_id = schema.relations.index("self")
    mutable = np.flatnonzero(canonical != self_id)
    if len(mutable) < 2:
        raise ValueError("graph has no non-self relations to shuffle")
    rng = np.random.default_rng(seed)
    for _ in range(128):
        candidate = canonical.copy()
        candidate[mutable] = rng.permutation(canonical[mutable])
        if not np.array_equal(candidate, canonical):
            # Reject a mere global relation-name permutation: it preserves the
            # equality pattern among edges and is not an edge-level ablation.
            global_rename = all(
                np.all(candidate[canonical == r] == candidate[canonical == r][0])
                for r in set(canonical[mutable]))
            if not global_rename:
                return candidate
    raise ValueError("seed produced no valid edge-level relation shuffle")
