"""The causal packet as a function of how many horizontal axes the task has.

The 2D reference packet is 26 fields. Decomposed by whether a quantity belongs
to a horizontal axis or is shared across them, it is exactly
``12 x 1 + 14 = 26`` -- so the dimensional extension is not a design choice,
it is a substitution: ``12 x 2 + 14 = 38`` for 3D.

This matters because the spatial packets were built by accretion instead, one
ladder rung at a time: 35 fields (v5), 39 (v6), 43 (v7), 47 (v9). None of them
is 38, and each added whatever the previous rung turned out to lack. Three of
the quantities the 2D reference graph reads -- the predicted bearing, the
predicted field-of-view margin and the acceleration uncertainty -- arrived only
at rungs 6 and 7, so the default spatial contract could not build the reference
context at all and filled the nine rows positionally instead.

``planar_name`` records what upstream MATLAB calls each field at one axis.
``test_one_axis_spec_reproduces_the_reference_registry`` pins the 1-axis
expansion of this spec onto the frozen 2D reference registry, so the derivation
is checked rather than asserted.

Nothing here reads simulator truth, outcomes, labels or contact state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

#: Scale for the ``x / (|x| + scale)`` smooth-signed normalization, or one of
#: the symbolic scales resolved from the task configuration.
Scale = float | str | None
HALF_FOV = "half_fov"
PROLONGED_LOSS = "prolonged_loss"
MISSION_LIMIT = "mission_limit"
#: Unscaled fields carry their registry normalization verbatim rather than a
#: scale. Inferring it from the source was wrong for detectionConfidence, which
#: the derivation test caught at once.
IDENTITY = "identity"
BINARY = "binary"
PROBABILITY = "probability_0_1"
UNSCALED = frozenset({IDENTITY, BINARY, PROBABILITY})


@dataclass(frozen=True)
class FieldSpec:
    name: str
    planar_name: str
    scale: Scale
    source: str
    per_axis: bool
    #: False marks a field the 2D reference does not have. Extras are opt-in
    #: and are excluded from the derivation test, so adding one can never
    #: quietly redefine the ported contract.
    reference: bool = True


def _axis(name, planar_name, scale, source, *, reference=True):
    return FieldSpec(name, planar_name, scale, source, True, reference)


def _shared(name, scale, source, *, reference=True):
    return FieldSpec(name, name, scale, source, False, reference)


#: Declared in the 2D reference registry's order. Each entry expands to one
#: field per horizontal axis when ``per_axis`` is set, so a one-axis expansion
#: reproduces that registry exactly and more axes extend it in place.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    _shared("h", 8.0, "own_state_adapter"),
    _axis("ownVelocity", "vx", 10.0, "own_state_adapter"),
    _shared("vz", 1.5, "own_state_adapter"),
    _axis("sinTilt", "sinTheta", IDENTITY, "own_state_adapter"),
    _axis("cosTilt", "cosTheta", IDENTITY, "own_state_adapter"),
    _axis("tiltRate", "pitchRate", math.pi / 2, "own_state_adapter"),
    _axis("relativePosition", "exEstimate", 3.0, "causal_pad_track"),
    _axis("relativeVelocity", "relativeVxEstimate", 3.0, "causal_pad_track"),
    _axis("padVelocity", "padVxEstimate", 10.0, "causal_pad_track"),
    _axis("padAcceleration", "padAxEstimate", 2.0, "causal_pad_track"),
    _shared("positionStd", 5.0, "causal_pad_track"),
    _shared("velocityStd", 5.0, "causal_pad_track"),
    _shared("accelerationStd", 3.0, "causal_pad_track"),
    _shared("trackInitialized", BINARY, "causal_pad_track"),
    _shared("detected", BINARY, "metric_marker_detector"),
    _axis("measuredBearing", "measuredBearing", HALF_FOV, "metric_marker_detector"),
    _shared("bearingValid", BINARY, "metric_marker_detector"),
    _shared("detectionConfidence", PROBABILITY, "metric_marker_detector"),
    _shared("timeSinceLastDetection", PROLONGED_LOSS, "causal_pad_track"),
    _axis("predictedBearing", "predictedBearing", HALF_FOV,
          "causal_pad_track_projection"),
    _axis("predictedFovMargin", "predictedFovMargin", HALF_FOV,
          "causal_pad_track_projection"),
    _shared("remainingMissionTime", MISSION_LIMIT, "public_mission_clock"),
    _axis("previousAction", "previousNormalizedActionX", IDENTITY, "policy_memory"),
    _shared("previousNormalizedActionZ", IDENTITY, "policy_memory"),
    _shared("landingInhibited", BINARY, "common_safety_supervisor"),
    _shared("abortRequested", BINARY, "common_safety_supervisor"),
    # --- Non-reference extras, opt-in. ---
    # With a nonzero camera transport delay the capture-aligned measurement and
    # the estimator's current bearing stop coinciding. The reference graph reads
    # the capture-aligned one (``measuredBearing``); the vector arm still needs
    # the current estimate, and the delay itself must be observable.
    _axis("estimatedBearing", "estimatedBearing", HALF_FOV,
          "causal_pad_track_projection", reference=False),
    _shared("opticalTransportAge", 1.0, "metric_marker_detector", reference=False),
)

#: ENU horizontal axes. One axis is the planar reference task, two is spatial.
PLANAR_AXES: tuple[str, ...] = ("x",)
SPATIAL_AXES: tuple[str, ...] = ("x", "y")

#: Sources that may never appear in a packet, whatever the axis count.
#: Same list the planar registry declares.
FORBIDDEN_SOURCES: tuple[str, ...] = (
    "future_pad_command", "phase_id", "phase_switch_time",
    "true_pad_acceleration", "current_hidden_pad_state", "episode_outcome",
    "reward_value", "teacher_action",
)


def field_names(axes=SPATIAL_AXES, *, extras=False) -> tuple[str, ...]:
    """Packet field order for a task with these horizontal axes."""
    axes = tuple(axes)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("horizontal axes must be a non-empty unique sequence")
    names: list[str] = []
    for spec in FIELD_SPECS:
        if not spec.reference and not extras:
            continue
        if not spec.per_axis:
            names.append(spec.name)
        elif len(axes) == 1:
            # One axis reproduces upstream's own names, so the planar registry
            # stays byte-identical to the ported MATLAB contract.
            names.append(spec.planar_name)
        else:
            names.extend(f"{spec.name}_{axis}" for axis in axes)
    return tuple(names)


def field_specs(axes=SPATIAL_AXES, *,
                extras=False) -> tuple[tuple[str, FieldSpec, str | None], ...]:
    """``(emitted name, spec, axis)`` triples in packet order."""
    axes = tuple(axes)
    rows = []
    for spec in FIELD_SPECS:
        if not spec.reference and not extras:
            continue
        if not spec.per_axis:
            rows.append((spec.name, spec, None))
        elif len(axes) == 1:
            rows.append((spec.planar_name, spec, axes[0]))
        else:
            rows.extend((f"{spec.name}_{axis}", spec, axis) for axis in axes)
    return tuple(rows)


def normalization(spec: FieldSpec) -> str:
    """The registry's declared normalization string for a field."""
    if spec.scale in UNSCALED:
        return spec.scale
    if spec.scale == MISSION_LIMIT:
        return "remaining_seconds_over_mission_limit"
    if spec.scale == HALF_FOV:
        return "signed_x_over_abs_x_plus_half_fov"
    if spec.scale == PROLONGED_LOSS:
        return "signed_x_over_abs_x_plus_prolonged_loss"
    return f"signed_x_over_abs_x_plus_{spec.scale}"


def registry(axes=SPATIAL_AXES, *, extras=False,
             clock="monotonic_simulation_time_seconds") -> dict:
    """Declarative registry for this axis count, with its own digest."""
    axes = tuple(axes)
    fields = [{"name": name, "size": 1, "source": spec.source,
               "normalization": normalization(spec)}
              for name, spec, _ in field_specs(axes, extras=extras)]
    payload = {
        "schema": f"ontology_rgat.causal_packet/axes-{len(axes)}"
                  + ("-with-transport-extras" if extras else ""),
        "axes": list(axes),
        "clock": clock,
        "fields": fields,
        "forbidden_sources": list(FORBIDDEN_SOURCES),
        "relative_convention": "pad-minus-uav",
        "graph": f"reference-9x12-per-axis x{len(axes)}",
    }
    payload["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def resolve_scale(spec: FieldSpec, *, half_fov, prolonged_loss_s, mission_limit_s):
    """Numeric scale for a field; ``None`` for the unscaled ones."""
    if spec.scale in UNSCALED:
        return None
    if spec.scale == HALF_FOV:
        return float(half_fov)
    if spec.scale == PROLONGED_LOSS:
        return float(prolonged_loss_s)
    if spec.scale == MISSION_LIMIT:
        return float(mission_limit_s)
    return float(spec.scale)


@dataclass(frozen=True)
class HeadGeometry:
    """Policy-head shape implied by the axis count.

    These four numbers were 2D literals spread across the model defaults
    (``packet_dim=26, action_dim=2, descent_axis=1, graph_planes=1``) and a
    schema-conditioned expression in the spatial trainer
    (``graph_planes=2 if config.reference_context else 1``). One horizontal
    axis gives the planar head back exactly.
    """

    packet_dim: int
    action_dim: int
    descent_axis: int
    graph_planes: int


def head_geometry(axes=SPATIAL_AXES, *, extras=False) -> HeadGeometry:
    """Actor/critic shape for a task with these horizontal axes.

    The action is one command per horizontal axis plus the vertical one, so the
    descent component is always the last, and the graph is one plane per axis.
    """
    axes = tuple(axes)
    return HeadGeometry(
        packet_dim=len(field_names(axes, extras=extras)),
        action_dim=len(axes) + 1,
        descent_axis=len(axes),
        graph_planes=len(axes))
