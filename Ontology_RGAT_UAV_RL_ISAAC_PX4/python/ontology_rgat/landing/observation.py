"""Normalize and decode the causal packet against the axis-generic spec.

Both routes previously open-coded the ``x / (|x| + scale)`` normalization and,
separately, its inverse for the graph builder -- the planar route in
``two_axis/contracts.py`` and ``ontology_v28.py``, the spatial route in
``spatial/core.py`` and ``spatial/context.py``. Four hand-maintained copies of
one pair of functions, with the scales repeated in each. Here they are one pair
driven by ``landing.packet.FIELD_SPECS``, so a scale cannot differ between the
forward and inverse direction or between dimensions.

The caller supplies raw physical quantities keyed by emitted field name; where
each number comes from is genuinely route-specific (planar pitch versus spatial
thrust-axis tilt) and stays with the route.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .packet import SPATIAL_AXES, field_specs, resolve_scale


def _scales(axes, half_fov, prolonged_loss_s, mission_limit_s, extras):
    """Per-emitted-field numeric scale, or ``None`` where unscaled."""
    axes = tuple(axes)
    if isinstance(half_fov, Mapping):
        per_axis = {axis: float(half_fov[axis]) for axis in axes}
    else:
        per_axis = {axis: float(half_fov) for axis in axes}
    resolved = {}
    for name, spec, axis in field_specs(axes, extras=extras):
        resolved[name] = resolve_scale(
            spec,
            half_fov=per_axis[axis] if axis is not None else max(per_axis.values()),
            prolonged_loss_s=prolonged_loss_s, mission_limit_s=mission_limit_s)
    return resolved


def normalize(raw: Mapping[str, float], axes=SPATIAL_AXES, *, half_fov,
              prolonged_loss_s=3.0, mission_limit_s=70.0,
              extras=False) -> np.ndarray:
    """Raw physical quantities -> the packet vector, in declared field order."""
    axes = tuple(axes)
    scales = _scales(axes, half_fov, prolonged_loss_s, mission_limit_s, extras)
    values = np.empty(len(scales), dtype=np.float32)
    for index, (name, spec, _) in enumerate(field_specs(axes, extras=extras)):
        if name not in raw:
            raise KeyError(f"causal packet field {name!r} was not supplied")
        value = float(raw[name])
        if not np.isfinite(value):
            raise ValueError(f"nonfinite causal packet field {name!r}")
        scale = scales[name]
        if scale is None:
            values[index] = value
        elif spec.scale == "mission_limit":
            values[index] = float(np.clip(value / scale, 0.0, 1.0))
        else:
            values[index] = value / (abs(value) + scale)
    return values


def decode(values, axes=SPATIAL_AXES, *, half_fov, prolonged_loss_s=3.0,
           mission_limit_s=70.0, extras=False) -> dict[str, float]:
    """The packet vector -> raw physical quantities, keyed by field name.

    Exact inverse of :func:`normalize` for the smooth-signed fields. The graph
    builder works in physical units, so this is what feeds it; sharing the
    scales with the forward direction is the point of this module.
    """
    axes = tuple(axes)
    scales = _scales(axes, half_fov, prolonged_loss_s, mission_limit_s, extras)
    values = np.asarray(values, dtype=float)
    specs = field_specs(axes, extras=extras)
    if values.shape != (len(specs),):
        raise ValueError(
            f"packet has {values.shape} values, expected {(len(specs),)} for "
            f"axes {axes}")
    raw = {}
    for index, (name, spec, _) in enumerate(specs):
        scale = scales[name]
        value = float(values[index])
        if scale is None or spec.scale == "mission_limit":
            raw[name] = value
        else:
            # x = s*n/(1-|n|); the guard keeps a saturated channel finite.
            raw[name] = scale * value / max(1.0 - abs(value), 1e-9)
    return raw
