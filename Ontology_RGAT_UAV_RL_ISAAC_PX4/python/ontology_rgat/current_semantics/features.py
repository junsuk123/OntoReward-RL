"""Common phi44 features and the 8x10 node input of spatial_current_semantics_v1 (guide 8.1).

Every value is a pure function of the current 21D vector and the fixed config:
no history, previous action, integral or simulator truth.  Raw channels are
copied bit-exact into the first slots of their node, so raw21 is recoverable
by name from either layout.  Every derived channel goes through the one
bounded map ``B(x; s) = x/(|x|+s)`` with a registered SI scale, and the MLP arm
reads the same 44 numbers the graph arms read.  Without a pad estimate the
estimate-derived entries are the neutral 0, and the validity flags in the same
vector say why, so a 0 never reads as alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import FEATURE_VERSION
from .contracts import (ESTIMATE_CHANNELS, ESTIMATE_VALIDITY_RULE, FEATURE_DIMENSION,
                        NODE_INPUT_DIMENSION, NODE_ORDER, OBSERVATION_CHANNELS,
                        OBSERVATION_DIMENSION, OBSERVATION_NAMES, CurrentSemanticsConfig,
                        contact_rate_rad_s, contact_tilt_rad)
from .geometry import NEUTRAL, CameraKinematics, camera_kinematics
from .observation import CurrentPhysical, decode


NUMERIC_SLOTS = NODE_INPUT_DIMENSION - 1
NODE_ID_COLUMN = NUMERIC_SLOTS

ALWAYS, ESTIMATE_VALID, FRONT_VALID = "always", "estimate_valid", "front_valid"
VALIDITY_RULES = MappingProxyType({
    ALWAYS: "own navigation, attitude or a validity flag; computed for every vector",
    ESTIMATE_VALID: ESTIMATE_VALIDITY_RULE + "; derived entries are then the neutral 0",
    FRONT_VALID: "estimate valid and optical depth Z > camera.z_near_m; otherwise the neutral 0",
})

# x/(|x|+s) rounds to exactly +-1 once |x|/s exceeds ~2**53; the registered map
# is open, so it saturates one ulp short and 1.0 stays a flag/sentinel value.
OPEN_BOUND = float(np.nextafter(1.0, 0.0))
BOUNDED_RULE = ("B(x; s) = x/(|x|+s), s > 0 in the unit of x, saturated at +-(1 - 2**-53) "
                "so float64 rounding never closes the interval; raw channels are copied, "
                "never re-mapped")


def bounded(x: float, scale: float) -> float:
    value = x / (abs(x) + scale)
    return min(max(value, -OPEN_BOUND), OPEN_BOUND)


@dataclass(frozen=True)
class FeatureSpec:
    """One phi44 entry.

    ``transform`` is an observation rule (signed | trig | flag | age); every
    derived entry is ``signed``, i.e. ``B``.  ``scale_key`` names where its SI
    scale is registered: ``observation.<channel>`` is fixed by the 21D
    contract, any other key is read from the config, so in the config-free
    ``FEATURES`` that ``scale`` is None and ``feature_registry(config)``
    fills it.  Outside ``valid_when`` the entry is the neutral 0.
    """
    name: str
    node: str
    unit: str
    transform: str
    scale: float | None
    source: str
    scale_key: str | None
    valid_when: str


_CHANNELS = MappingProxyType({channel.name: channel for channel in OBSERVATION_CHANNELS})
_Q, _G, _M, _A, _P, _O, _C, _L = NODE_ORDER
_IMAGE, _IMAGE_RATE = "feature_scales.image_coordinate", "feature_scales.image_rate_per_s"


def _raw(name: str) -> FeatureSpec:
    channel = _CHANNELS[name]
    return FeatureSpec(name=name, node=channel.node, unit=channel.unit,
                       transform=channel.transform, scale=channel.scale,
                       source=f"raw {name}: {channel.meaning}",
                       scale_key=None if channel.scale is None else f"observation.{name}",
                       valid_when=ESTIMATE_VALID if name in ESTIMATE_CHANNELS else ALWAYS)


def _derived(name: str, node: str, unit: str, source: str, scale_key: str | None,
             valid_when: str, transform: str = "signed") -> FeatureSpec:
    fixed = None
    if scale_key is not None and scale_key.startswith("observation."):
        fixed = _CHANNELS[scale_key.split(".", 1)[1]].scale
    return FeatureSpec(name=name, node=node, unit=unit, transform=transform, scale=fixed,
                       source=source, scale_key=scale_key, valid_when=valid_when)


FEATURES = (
    *(_raw(name) for name in ("ugv_visionUpdated", "ugv_visionAge",
                              "drone_navigationValid", "drone_navigationAge")),
    *(_raw(name) for name in ("relative_x", "relative_y", "relative_height")),
    *(_raw(name) for name in ("relative_vx", "relative_vy", "ugv_vx", "ugv_vy", "drone_vz")),
    _derived("own_vx", _M, "m/s", "B(vgx - vrx; s): own horizontal velocity, pad minus "
             "relative", "observation.ugv_vx", ESTIMATE_VALID),
    _derived("own_vy", _M, "m/s", "B(vgy - vry; s): own horizontal velocity, pad minus "
             "relative", "observation.ugv_vy", ESTIMATE_VALID),
    *(_raw(name) for name in ("drone_sinPhi", "drone_cosPhi", "drone_sinTheta",
                              "drone_cosTheta", "drone_sinPsi", "drone_cosPsi",
                              "drone_rollRate", "drone_pitchRate", "drone_yawRate")),
    _derived("approach_ex", _P, "m", "B(ex; s)", "feature_scales.approach_position_m",
             ESTIMATE_VALID),
    _derived("approach_ey", _P, "m", "B(ey; s)", "feature_scales.approach_position_m",
             ESTIMATE_VALID),
    _derived("approach_vx", _P, "m/s", "B(vrx; s)", "feature_scales.approach_velocity_m_s",
             ESTIMATE_VALID),
    _derived("approach_vy", _P, "m/s", "B(vry; s)", "feature_scales.approach_velocity_m_s",
             ESTIMATE_VALID),
    _derived("approach_closure", _P, "m^2/s", "B(ex*vrx + ey*vry; s): > 0 while the "
             "horizontal error grows, d|e|^2/dt = 2 e.v",
             "feature_scales.approach_closure_m2_s", ESTIMATE_VALID),
    _derived("view_xi", _O, "1", "B(xi; s), xi = 2u/(W-1) - 1", _IMAGE, FRONT_VALID),
    _derived("view_eta", _O, "1", "B(eta; s), eta = 2v/(H-1) - 1", _IMAGE, FRONT_VALID),
    _derived("view_xi_rate", _O, "1/s", "B(xi_dot; s), xi_dot = 2 u_dot/(W-1)",
             _IMAGE_RATE, FRONT_VALID),
    _derived("view_eta_rate", _O, "1/s", "B(eta_dot; s), eta_dot = 2 v_dot/(H-1)",
             _IMAGE_RATE, FRONT_VALID),
    _derived("view_margin_left", _O, "1", "B(1 + xi; s)", _IMAGE, FRONT_VALID),
    _derived("view_margin_right", _O, "1", "B(1 - xi; s)", _IMAGE, FRONT_VALID),
    _derived("view_margin_top", _O, "1", "B(1 + eta; s)", _IMAGE, FRONT_VALID),
    _derived("view_margin_bottom", _O, "1", "B(1 - eta; s)", _IMAGE, FRONT_VALID),
    _derived("view_front_valid", _O, "bool", "1 if estimate valid and Z > camera.z_near_m",
             None, ALWAYS, transform="flag"),
    _derived("contact_ex", _C, "m", "B(ex; s)", "contact.pad_radius_m", ESTIMATE_VALID),
    _derived("contact_ey", _C, "m", "B(ey; s)", "contact.pad_radius_m", ESTIMATE_VALID),
    _derived("contact_vx", _C, "m/s", "B(vrx; s)", "contact.xy_speed_m_s", ESTIMATE_VALID),
    _derived("contact_vy", _C, "m/s", "B(vry; s)", "contact.xy_speed_m_s", ESTIMATE_VALID),
    _derived("contact_vz", _C, "m/s", "B(vz; s): own vertical velocity, pad vz assumed 0; "
             "own navigation, so it needs no pad estimate", "contact.z_speed_m_s", ALWAYS),
    _derived("contact_tilt", _C, "rad", "B(contracts.contact_tilt_rad(roll, pitch); s)",
             "contact.tilt_rad", ALWAYS),
    _derived("contact_rate", _C, "rad/s", "B(contracts.contact_rate_rad_s(p, q); s)",
             "contact.rate_rad_s", ALWAYS),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURES)
NODE_FEATURES = MappingProxyType({node: tuple(spec.name for spec in FEATURES if spec.node == node)
                                  for node in NODE_ORDER})


def _frozen(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


_RAW_SLOTS = _frozen([index for index, spec in enumerate(FEATURES) if spec.name in _CHANNELS],
                     np.int64)
_RAW_SOURCES = _frozen([OBSERVATION_NAMES.index(FEATURES[index].name) for index in _RAW_SLOTS],
                       np.int64)
_DERIVED = tuple((index, spec) for index, spec in enumerate(FEATURES)
                 if spec.name not in _CHANNELS)
_NODE_ROW = _frozen([NODE_ORDER.index(spec.node) for spec in FEATURES], np.int64)
_NODE_SLOT = _frozen([NODE_FEATURES[spec.node].index(spec.name) for spec in FEATURES], np.int64)
_RAW_FLAT_INDEX = _frozen([FEATURE_NAMES.index(name) for name in OBSERVATION_NAMES], np.int64)
NODE_IDS = _frozen([(index + 1) / len(NODE_ORDER) for index in range(len(NODE_ORDER))],
                   np.float64)

if len(FEATURES) != FEATURE_DIMENSION or len(set(FEATURE_NAMES)) != FEATURE_DIMENSION:
    raise AssertionError(f"phi needs {FEATURE_DIMENSION} unique features")
if sorted(FEATURES[index].name for index in _RAW_SLOTS) != sorted(OBSERVATION_NAMES):
    raise AssertionError("every raw21 channel must appear exactly once")
if list(_NODE_ROW) != sorted(_NODE_ROW):
    raise AssertionError("features must be grouped in NODE_ORDER")
for _node, _names in NODE_FEATURES.items():
    if len(_names) > NUMERIC_SLOTS:
        raise AssertionError(f"{_node} has more than {NUMERIC_SLOTS} numeric slots")
    _kinds = [name in _CHANNELS for name in _names]
    if _kinds != sorted(_kinds, reverse=True):
        raise AssertionError(f"{_node}: raw channels must take the first slots")
for _spec in FEATURES:
    if _spec.name in _CHANNELS and _spec.node != _CHANNELS[_spec.name].node:
        raise AssertionError(f"{_spec.name} is registered on {_CHANNELS[_spec.name].node}")
    if (_spec.scale_key is None) != (_spec.transform in ("trig", "flag")):
        raise AssertionError(f"{_spec.name}: scale presence contradicts its transform")
    if _spec.valid_when not in VALIDITY_RULES:
        raise AssertionError(f"{_spec.name}: unknown validity {_spec.valid_when!r}")
if _CHANNELS["relative_vx"].scale != _CHANNELS["ugv_vx"].scale \
        or _CHANNELS["relative_vy"].scale != _CHANNELS["ugv_vy"].scale:
    raise AssertionError("own velocity needs one scale shared by pad and relative velocity")
del _node, _names, _kinds, _spec


def _scale(spec: FeatureSpec, config: CurrentSemanticsConfig) -> float | None:
    if spec.scale_key is None or spec.scale is not None:
        return spec.scale
    section, name = spec.scale_key.split(".", 1)
    return float(getattr(getattr(config, section), name))


def feature_registry(config: CurrentSemanticsConfig) -> tuple[FeatureSpec, ...]:
    """FEATURES with every configurable scale resolved; what the builders apply."""
    return tuple(replace(spec, scale=_scale(spec, config)) for spec in FEATURES)


def _scales(config: CurrentSemanticsConfig) -> tuple[float | None, ...]:
    return tuple(_scale(spec, config) for _, spec in _DERIVED)


def _quantities(phys: CurrentPhysical, camera: CameraKinematics) -> dict[str, float]:
    left, right, top, bottom = camera.margins
    return {
        "own_vx": phys.own_vx, "own_vy": phys.own_vy,
        "approach_ex": phys.ex, "approach_ey": phys.ey,
        "approach_vx": phys.vrx, "approach_vy": phys.vry,
        "approach_closure": phys.ex * phys.vrx + phys.ey * phys.vry,
        "view_xi": camera.xi, "view_eta": camera.eta,
        "view_xi_rate": camera.xi_dot, "view_eta_rate": camera.eta_dot,
        "view_margin_left": left, "view_margin_right": right,
        "view_margin_top": top, "view_margin_bottom": bottom,
        "view_front_valid": 1.0 if camera.front_valid else 0.0,
        "contact_ex": phys.ex, "contact_ey": phys.ey,
        "contact_vx": phys.vrx, "contact_vy": phys.vry, "contact_vz": phys.vz,
        "contact_tilt": float(contact_tilt_rad(phys.roll, phys.pitch)),
        "contact_rate": float(contact_rate_rad_s(phys.p, phys.q)),
    }


def _derived_values(phys: CurrentPhysical, camera: CameraKinematics,
                    scales: tuple[float | None, ...]) -> dict[str, float]:
    quantities = _quantities(phys, camera)
    available = {ALWAYS: True, ESTIMATE_VALID: phys.estimate_valid,
                 FRONT_VALID: camera.front_valid}
    values = {}
    for (_, spec), scale in zip(_DERIVED, scales):
        x = quantities[spec.name] if available[spec.valid_when] else NEUTRAL
        values[spec.name] = x if spec.transform == "flag" else bounded(x, scale)
    return values


def derived_features(phys: CurrentPhysical, camera: CameraKinematics,
                     config: CurrentSemanticsConfig) -> dict[str, float]:
    """The non-raw entries by name, B-mapped and gated, from decoded physical values.

    ``camera`` must be ``camera_kinematics(phys, config.calibration)``.
    """
    return _derived_values(phys, camera, _scales(config))


def _assemble(raw: np.ndarray, derived: Mapping[str, float]) -> np.ndarray:
    flat = np.empty(FEATURE_DIMENSION, dtype=np.float64)
    flat[_RAW_SLOTS] = raw[_RAW_SOURCES]
    for index, spec in _DERIVED:
        flat[index] = derived[spec.name]
    return flat


def _layout(flat: np.ndarray) -> np.ndarray:
    nodes = np.zeros(flat.shape[:-1] + (len(NODE_ORDER), NODE_INPUT_DIMENSION), dtype=np.float64)
    nodes[..., _NODE_ROW, _NODE_SLOT] = flat
    nodes[..., NODE_ID_COLUMN] = NODE_IDS
    return nodes


@dataclass(frozen=True, eq=False)
class FeatureFrame:
    """One decision's features.  The arrays are read-only so a frame stays a
    value; copy them (``torch.tensor``) before handing them to a mutating consumer."""
    flat: np.ndarray
    nodes: np.ndarray
    raw: np.ndarray
    camera: CameraKinematics
    estimate_valid: bool


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def build_features(vector: Any, config: CurrentSemanticsConfig) -> FeatureFrame:
    """phi44 and the 8x10 node input of one current 21D vector; 24D is refused by decode."""
    raw = np.array(vector, dtype=np.float64)
    phys = decode(raw)
    camera = camera_kinematics(phys, config.calibration)
    flat = _assemble(raw, _derived_values(phys, camera, _scales(config)))
    return FeatureFrame(flat=_read_only(flat), nodes=_read_only(_layout(flat)),
                        raw=_read_only(raw), camera=camera, estimate_valid=phys.estimate_valid)


def build_feature_batch(array: Any,
                        config: CurrentSemanticsConfig) -> tuple[np.ndarray, np.ndarray]:
    """(flat (N, 44), nodes (N, 8, 10)), row for row identical to ``build_features``."""
    batch = np.asarray(array, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != OBSERVATION_DIMENSION:
        raise ValueError(f"batch must have shape (N, {OBSERVATION_DIMENSION}), got "
                         f"{batch.shape}; the stable 24D profile is excluded")
    scales = _scales(config)
    flat = np.empty((batch.shape[0], FEATURE_DIMENSION), dtype=np.float64)
    for row, vector in enumerate(batch):
        phys = decode(vector)
        camera = camera_kinematics(phys, config.calibration)
        flat[row] = _assemble(vector, _derived_values(phys, camera, scales))
    return flat, _layout(flat)


def raw_from_flat(flat: Any) -> np.ndarray:
    values = np.asarray(flat, dtype=np.float64)
    if values.shape[-1:] != (FEATURE_DIMENSION,):
        raise ValueError(f"flat features need a trailing dimension {FEATURE_DIMENSION}, "
                         f"got {values.shape}")
    return values[..., _RAW_FLAT_INDEX]


def raw_from_nodes(nodes: Any) -> np.ndarray:
    values = np.asarray(nodes, dtype=np.float64)
    if values.shape[-2:] != (len(NODE_ORDER), NODE_INPUT_DIMENSION):
        raise ValueError(f"node input needs trailing shape ({len(NODE_ORDER)}, "
                         f"{NODE_INPUT_DIMENSION}), got {values.shape}")
    return values[..., _NODE_ROW[_RAW_FLAT_INDEX], _NODE_SLOT[_RAW_FLAT_INDEX]]


def manifest_part(config: CurrentSemanticsConfig) -> dict[str, Any]:
    """The feature registry for ``contracts.contract_manifest`` (hashed into feature_hash)."""
    return {
        "version": FEATURE_VERSION,
        "dimension": FEATURE_DIMENSION,
        "bounded_map": BOUNDED_RULE,
        "features": feature_registry(config),
        "node_order": NODE_ORDER,
        "node_features": NODE_FEATURES,
        "node_layout": {"shape": (len(NODE_ORDER), NODE_INPUT_DIMENSION),
                        "numeric_slots": NUMERIC_SLOTS, "id_column": NODE_ID_COLUMN,
                        "id_value": "(index + 1) / 8", "padding": 0.0,
                        "raw_channels": "first slots of their node, copied bit-exact"},
        "validity": VALIDITY_RULES,
        "neutral": NEUTRAL,
    }
