from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GatewayConfig:
    protocol_version: int
    control_hz: float
    state_timeout_s: float
    action_timeout_s: float
    target: str
    bind_host: str
    matlab_host: str
    gateway_port: int
    matlab_port: int
    namespace: str
    target_system: int
    target_component: int
    source_system: int
    source_component: int
    hover_thrust: float
    collective_span: float
    max_roll_pitch_rad: float
    max_yaw_rate_rad_s: float
    offboard_prestream_count: int
    estimator_warmup_s: float
    marker_pose_drives_policy: bool

    @classmethod
    def from_mapping(cls, data: dict[str, Any], target: str | None = None) -> "GatewayConfig":
        import math

        system = data["system"]
        network = data["network"]
        px4 = data["px4"]
        vision = data.get("vision", {})
        resolved_target = target or system["target"]
        if resolved_target not in {"sitl", "hardware"}:
            raise ValueError("target must be 'sitl' or 'hardware'")
        return cls(
            protocol_version=int(system["protocol_version"]),
            control_hz=float(system["control_hz"]),
            state_timeout_s=float(system["state_timeout_s"]),
            action_timeout_s=float(system["action_timeout_s"]),
            target=resolved_target,
            bind_host=str(network["bind_host"]),
            matlab_host=str(network["matlab_host"]),
            gateway_port=int(network["gateway_port"]),
            matlab_port=int(network["matlab_port"]),
            namespace=str(px4["namespace"]).rstrip("/"),
            target_system=int(px4["target_system"]),
            target_component=int(px4["target_component"]),
            source_system=int(px4["source_system"]),
            source_component=int(px4["source_component"]),
            hover_thrust=float(px4["hover_thrust"]),
            collective_span=float(px4["collective_span"]),
            max_roll_pitch_rad=math.radians(float(px4["max_roll_pitch_deg"])),
            max_yaw_rate_rad_s=math.radians(float(px4["max_yaw_rate_deg_s"])),
            offboard_prestream_count=int(px4["offboard_prestream_count"]),
            estimator_warmup_s=float(px4["estimator_warmup_s"]),
            marker_pose_drives_policy=bool(vision.get("pose_source_for_policy", False)),
        )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return data


def load_gateway_config(path: str | Path, target: str | None = None) -> GatewayConfig:
    return GatewayConfig.from_mapping(load_yaml(path), target=target)

