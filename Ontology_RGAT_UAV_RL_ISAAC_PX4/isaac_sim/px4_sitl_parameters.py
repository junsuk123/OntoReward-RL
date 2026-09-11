#!/usr/bin/env python3
"""Validate PX4 SITL parameters and render the per-run startup wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex
from typing import Any, Mapping


@dataclass(frozen=True)
class Px4Parameter:
    name: str
    value: int | float


def configured_px4_parameters(values: Mapping[str, Any] | None) -> tuple[Px4Parameter, ...]:
    """Validate config values and encode them for MAVLink ``PARAM_SET``."""
    parameters: list[Px4Parameter] = []
    for raw_name, value in (values or {}).items():
        name = str(raw_name).strip().upper()
        try:
            encoded_name = name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"PX4 parameter name must be ASCII: {name!r}") from exc
        if not name or len(encoded_name) > 16:
            raise ValueError(f"PX4 parameter name must contain 1-16 ASCII bytes: {name!r}")
        if isinstance(value, bool):
            raise ValueError(f"PX4 parameter {name} must be an integer or float, not bool")
        if isinstance(value, int):
            if not -(2**31) <= value < 2**31:
                raise ValueError(f"PX4 integer parameter {name} is outside int32 range")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"PX4 float parameter {name} must be finite")
        else:
            raise ValueError(f"PX4 parameter {name} must be an integer or float")
        parameters.append(Px4Parameter(name, value))
    return tuple(parameters)


def px4_rc_script(base_script: str | Path,
                  parameters: tuple[Px4Parameter, ...]) -> str:
    """Source PX4's stock rcS, then apply this experiment's overrides."""
    base = str(Path(base_script).resolve())
    if "\n" in base or "\r" in base:
        raise ValueError("PX4 rcS path must not contain a newline")
    lines = ["#!/bin/sh", f". {shlex.quote(base)}"]
    if parameters:
        lines.append('echo "INFO  [ontology_rgat] applying autonomous SITL parameters"')
    for parameter in parameters:
        value = str(parameter.value) if isinstance(parameter.value, int) \
            else format(parameter.value, ".9g")
        lines.append(f"param set {parameter.name} {value}")
    return "\n".join(lines) + "\n"
