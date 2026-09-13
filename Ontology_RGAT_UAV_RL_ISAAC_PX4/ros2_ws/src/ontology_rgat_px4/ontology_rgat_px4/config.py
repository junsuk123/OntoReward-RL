from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .battery import BatteryConfig


@dataclass(frozen=True)
class GatewayConfig:
    protocol_version: int
    control_hz: float
    state_timeout_s: float
    action_timeout_s: float
    sitl_action_timeout_s: float
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
    pair_index: int
    pair_count: int
    topic_root: str
    hover_thrust: float
    collective_span: float
    max_roll_pitch_rad: float
    max_yaw_rate_rad_s: float
    offboard_prestream_count: int
    estimator_warmup_s: float
    start_airborne: bool
    marker_pose_drives_policy: bool
    # Seconds over which a camera/PX4 position handover is faded out, so
    # losing the pad from frame does not step the state the policy flies on.
    vision_handover_tau_s: float
    marker_pose_max_step_m: float
    marker_pose_reacquire_error_m: float
    marker_fusion_max_correction_m: float
    pad_motion: str
    pad_deck_height_m: float
    world_xy_limit_m: float
    max_altitude_m: float
    # How far from the origin the deck's own route can reach. The deck drives a
    # lap of a city block, so the guard rail on a world-frame setpoint is the
    # city and not the pad-relative arena.
    world_radius_m: float
    # The geodetic datum the simulator's world ENU frame is pinned at. PX4's
    # local frame is pinned wherever its EKF happened to initialise -- the
    # spawn point -- so without these two the gateway cannot tell the frames
    # apart, and it mixes a PX4-local position with a world-frame deck pose.
    # Time constants for the cooperative deck track and the covariance-aware
    # pad-relative navigation filter.
    deck_track_tau_s: float
    pad_track_tau_s: float
    map_latitude_deg: float
    map_longitude_deg: float
    map_altitude_m: float
    gnss_enabled: bool
    gnss_dr_enter_quality: float
    battery: BatteryConfig

    @property
    def pad_is_static(self) -> bool:
        return self.pad_motion == "static"

    @classmethod
    def from_mapping(cls, data: dict[str, Any], target: str | None = None) -> "GatewayConfig":
        import math

        system = data["system"]
        network = data["network"]
        px4 = data["px4"]
        isaac = data.get("isaac", {}) or {}
        vision = data.get("vision", {})
        pad = data.get("pad", {}) or {}
        landing = data.get("landing", {}) or {}
        urban = data.get("urban", {}) or {}
        gnss = data.get("gnss", {}) or {}
        block = [float(v) for v in pad.get(
            "route_size_m", urban.get("block_size_m", (0.0, 0.0)))]
        waypoints = pad.get("route_waypoints_enu_m", ())
        if str(pad.get("motion", "static")).lower() == "waypoints" and waypoints:
            # Imported-world routes use absolute world coordinates rather than
            # a rectangle centred at [0, 0]. The safety radius must contain
            # those coordinates or it turns every legal setpoint into a command
            # toward the world origin.
            route_reach = max(
                math.hypot(float(point[0]), float(point[1]))
                for point in waypoints if len(point) >= 2
            )
        else:
            route_reach = 0.5 * math.hypot(*block) if len(block) == 2 else 0.0
        resolved_target = target or system["target"]
        if resolved_target not in {"sitl", "hardware"}:
            raise ValueError("target must be 'sitl' or 'hardware'")
        return cls(
            protocol_version=int(system["protocol_version"]),
            control_hz=float(system["control_hz"]),
            state_timeout_s=float(system["state_timeout_s"]),
            action_timeout_s=float(system["action_timeout_s"]),
            sitl_action_timeout_s=float(system.get(
                "sitl_action_timeout_s", system["action_timeout_s"])),
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
            pair_index=0,
            pair_count=1,
            topic_root="",
            hover_thrust=float(px4["hover_thrust"]),
            collective_span=float(px4["collective_span"]),
            max_roll_pitch_rad=math.radians(float(px4["max_roll_pitch_deg"])),
            max_yaw_rate_rad_s=math.radians(float(px4["max_yaw_rate_deg_s"])),
            offboard_prestream_count=int(px4["offboard_prestream_count"]),
            estimator_warmup_s=float(px4["estimator_warmup_s"]),
            start_airborne=bool(isaac.get("start_airborne", False)),
            marker_pose_drives_policy=bool(vision.get("pose_source_for_policy", False)),
            vision_handover_tau_s=float(vision.get("handover_tau_s", 0.6)),
            marker_pose_max_step_m=float(vision.get("pose_max_step_m", 0.75)),
            marker_pose_reacquire_error_m=float(
                vision.get("pose_reacquire_error_m", 2.0)),
            marker_fusion_max_correction_m=float(
                vision.get("fusion_max_correction_m", 0.5)),
            pad_motion=str(pad.get("motion", "static")).lower(),
            pad_deck_height_m=float(pad.get("deck_height_m", 0.0)),
            world_xy_limit_m=float(landing.get("world_xy_limit_m", 12.0)),
            max_altitude_m=float(landing.get("max_altitude_m", 12.0)),
            # The lap, plus the pad-relative slack the vehicle is allowed on
            # top of it. Falls back to the pad-relative limit when there is no
            # city, which is the fixed-pad control condition.
            world_radius_m=float(landing.get(
                "world_radius_m",
                route_reach + float(landing.get("world_xy_limit_m", 12.0))
                if route_reach > 0.0 else landing.get("world_xy_limit_m", 12.0))),
            deck_track_tau_s=float(pad.get("broadcast_track_tau_s", 1.0)),
            pad_track_tau_s=float(pad.get("entry_track_tau_s", 2.5)),
            map_latitude_deg=float((urban.get("origin") or {}).get("latitude", 0.0)),
            map_longitude_deg=float((urban.get("origin") or {}).get("longitude", 0.0)),
            map_altitude_m=float((urban.get("origin") or {}).get("altitude", 0.0)),
            gnss_enabled=bool(gnss.get("enabled", False)),
            gnss_dr_enter_quality=float(gnss.get("dr_enter_quality", 0.45)),
            battery=BatteryConfig.from_mapping(data),
        )


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a gateway YAML and recursively merge its optional ``extends``."""
    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"cyclic config extends chain at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"extends must be a non-empty path: {config_path}")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return _merge(load_yaml(parent_path, seen), data)


def load_gateway_config(path: str | Path, target: str | None = None) -> GatewayConfig:
    return GatewayConfig.from_mapping(load_yaml(path), target=target)
