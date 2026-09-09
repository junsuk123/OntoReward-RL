"""Dataclasses describing a scene.

Every class exposes ``from_dict`` which rejects unknown keys, so a typo in the
YAML fails at load time instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Type, TypeVar

from simlab.scenarios.environments import ENVIRONMENT_KEYS
from simlab.scenarios.occlusion import OCCLUDER_SET_KEYS

T = TypeVar("T")

#: The three tracking-failure situations this project reproduces. Each one runs
#: as its own episode: a full simulator start, its own data collection, and a
#: complete shutdown before the next begins.
SCENARIO_KEYS = ("mutual_occlusion", "structural_occlusion", "sensor_dropout")

#: Cameras a dropout may be scheduled on.
GATEABLE_CAMERAS = ("front_near", "front_far", "satellite_nadir")

Vec3 = Tuple[float, float, float]
Resolution = Tuple[int, int]


def _check_keys(cls: Type[Any], data: Mapping[str, Any]) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}; expected {sorted(known)}"
        )


def _vec3(value: Sequence[float], label: str) -> Vec3:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"{label}: expected 3 numbers, got {len(values)}")
    return values  # type: ignore[return-value]


def _range(value: Sequence[float], label: str) -> Tuple[float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 2 or values[0] <= 0 or values[1] < values[0]:
        raise ValueError(f"{label}: expected [low, high] with 0 < low <= high, got {value!r}")
    return values  # type: ignore[return-value]


def _resolution(value: Sequence[int], label: str) -> Resolution:
    values = tuple(int(v) for v in value)
    if len(values) != 2 or any(v <= 0 for v in values):
        raise ValueError(f"{label}: expected 2 positive integers, got {value!r}")
    return values  # type: ignore[return-value]


@dataclass
class AppConfig:
    """Kit application and stepping settings."""

    headless: bool = False
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    duration_s: float = 0.0  # 0 or less = run until the window is closed

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        _check_keys(cls, data)
        cfg = cls(**data)
        if cfg.physics_dt <= 0 or cfg.rendering_dt <= 0:
            raise ValueError("app: physics_dt and rendering_dt must be positive")
        return cfg

    @property
    def steps_per_second(self) -> float:
        return 1.0 / self.physics_dt


@dataclass
class WorldConfig:
    ground_plane: bool = True
    dome_light_intensity: float = 1000.0
    stage_units_in_meters: float = 1.0
    environment_type: str = "empty"
    environment_usd: str | None = None
    environment_prim_path: str = "/World/Environment"
    #: Which look ``procedural_urban`` wears; see simlab.scenarios.environments.
    environment: str = "urban_day"
    #: Layout/occluder jitter. None = derive from the scenario seed, so an
    #: episode YAML always pins the exact geometry the labels were built from.
    environment_seed: int | None = None
    #: Override the preset's occluder loadout; None = whatever the preset picks.
    occluder_set: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldConfig":
        _check_keys(cls, data)
        cfg = cls(**data)
        if cfg.environment_type not in {"empty", "procedural_urban"}:
            raise ValueError("world.environment_type must be empty or procedural_urban")
        if cfg.environment_usd and not cfg.environment_prim_path.startswith("/World/"):
            raise ValueError("world.environment_prim_path must be below /World")
        if cfg.environment not in ENVIRONMENT_KEYS:
            raise ValueError(
                f"world.environment must be one of {list(ENVIRONMENT_KEYS)}, got {cfg.environment!r}"
            )
        if cfg.occluder_set is not None and cfg.occluder_set not in OCCLUDER_SET_KEYS:
            raise ValueError(
                f"world.occluder_set must be one of {list(OCCLUDER_SET_KEYS)}, got {cfg.occluder_set!r}"
            )
        return cfg


@dataclass
class ControllerConfig:
    """Which algorithm drives the UGV, and its keyword arguments."""

    name: str = "square_loop"
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControllerConfig":
        _check_keys(cls, data)
        return cls(name=data.get("name", "square_loop"), params=dict(data.get("params", {})))


@dataclass
class UGVConfig:
    model: str = "nova_carter"  # key into simlab.sim.assets.ROBOTS
    prim_path: str = "/World/Robots/UGV"
    name: str = "ugv"
    spawn: Vec3 | None = None  # None = the model's default spawn height
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UGVConfig":
        _check_keys(cls, data)
        payload = dict(data)
        controller = ControllerConfig.from_dict(payload.pop("controller", {}) or {})
        spawn = payload.pop("spawn", None)
        return cls(
            controller=controller,
            spawn=_vec3(spawn, "ugv.spawn") if spawn is not None else None,
            **payload,
        )


@dataclass
class PersonConfig:
    """One character: which asset, where it starts, where it walks."""

    asset: str  # key into simlab.sim.assets.CHARACTERS, or a full USD subpath
    start: Vec3 = (0.0, 0.0, 0.0)
    waypoints: List[Vec3] = field(default_factory=list)
    yaw_deg: float = -90.0
    idle_s: float = 1.0  # pause at each waypoint

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PersonConfig":
        _check_keys(cls, data)
        payload = dict(data)
        if "asset" not in payload:
            raise ValueError("people.agents[]: 'asset' is required")
        start = payload.pop("start", (0.0, 0.0, 0.0))
        waypoints = payload.pop("waypoints", [])
        return cls(
            start=_vec3(start, "people.agents[].start"),
            waypoints=[_vec3(w, "people.agents[].waypoints[]") for w in waypoints],
            **payload,
        )


@dataclass
class PeopleConfig:
    """omni.anim.people settings plus the character roster."""

    root_prim: str = "/World/Characters"
    command_file: str = "people_commands.txt"
    navmesh: bool = False  # needs a baked navmesh; off = straight-line walking
    dynamic_avoidance: bool = False
    loop: str = "inf"
    count: int | None = None  # None = use every agent listed
    agents: List[PersonConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PeopleConfig":
        _check_keys(cls, data)
        payload = dict(data)
        agents = [PersonConfig.from_dict(a) for a in payload.pop("agents", [])]
        cfg = cls(agents=agents, **payload)
        if cfg.count is not None and cfg.count < 0:
            raise ValueError("people.count must be >= 0")
        return cfg

    def active_agents(self) -> List[PersonConfig]:
        """The roster after applying ``count``."""
        if self.count is None:
            return list(self.agents)
        return list(self.agents[: self.count])


@dataclass
class DroneTeamConfig:
    """Visual type and roster size for one drone team."""

    count: int
    model: str
    color: Vec3
    label: str
    cruise_altitude: float = 3.2
    asset_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], label: str) -> "DroneTeamConfig":
        _check_keys(cls, data)
        payload = dict(data)
        payload.setdefault("label", label)
        if "count" not in payload or "model" not in payload or "color" not in payload:
            raise ValueError(f"drones.{label}: count, model and color are required")
        color = _vec3(payload.pop("color"), f"drones.{label}.color")
        cfg = cls(color=color, **payload)
        if cfg.count < 0:
            raise ValueError(f"drones.{label}.count must be >= 0")
        if cfg.cruise_altitude <= 0:
            raise ValueError(f"drones.{label}.cruise_altitude must be positive")
        if cfg.asset_scale <= 0:
            raise ValueError(f"drones.{label}.asset_scale must be positive")
        return cfg


@dataclass
class DronesConfig:
    """Two-team quadrotors following a camera-aligned crossing shuttle."""

    enabled: bool = False
    root_prim: str = "/World/Drones"
    friendly: DroneTeamConfig = field(
        default_factory=lambda: DroneTeamConfig(4, "iris", (0.05, 0.3, 1.0), "Ally", asset_scale=1.35)
    )
    enemy: DroneTeamConfig = field(
        default_factory=lambda: DroneTeamConfig(
            2, "quadcopter", (1.0, 0.08, 0.03), "Enemy", asset_scale=1.5
        )
    )
    takeoff_duration_s: float = 4.0
    activity_half_length: float = 5.5
    safety_radius: float = 1.4
    max_speed: float = 2.2
    motion_mode: str = "crossing_shuttle"
    shuttle_leg_duration_s: float = 11.0
    crossing_offset_limit: float = 3.5
    crossing_time_jitter: float = 0.1
    camera_depth_separation_m: float = 1.8
    trajectory_seed: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DronesConfig":
        _check_keys(cls, data)
        payload = dict(data)
        defaults = cls()
        friendly_data = payload.pop("friendly", None)
        enemy_data = payload.pop("enemy", None)
        friendly = (
            DroneTeamConfig.from_dict(friendly_data, "friendly")
            if friendly_data is not None
            else defaults.friendly
        )
        enemy = (
            DroneTeamConfig.from_dict(enemy_data, "enemy")
            if enemy_data is not None
            else defaults.enemy
        )
        cfg = cls(friendly=friendly, enemy=enemy, **payload)
        if cfg.motion_mode != "crossing_shuttle":
            raise ValueError("drones.motion_mode must be crossing_shuttle")
        positive = (
            "takeoff_duration_s", "activity_half_length", "safety_radius", "max_speed",
            "shuttle_leg_duration_s", "camera_depth_separation_m",
        )
        for name in positive:
            if getattr(cfg, name) <= 0:
                raise ValueError(f"drones.{name} must be positive")
        if not 0.0 <= cfg.crossing_offset_limit < cfg.activity_half_length:
            raise ValueError(
                "drones.crossing_offset_limit must be in [0, activity_half_length)"
            )
        if not 0.0 <= cfg.crossing_time_jitter < 0.4:
            raise ValueError("drones.crossing_time_jitter must be in [0, 0.4)")
        if cfg.camera_depth_separation_m < cfg.safety_radius:
            raise ValueError("drones.camera_depth_separation_m must be >= safety_radius")
        return cfg


@dataclass
class CameraOpticsConfig:
    resolution: Resolution = (1920, 1080)
    focal_length_mm: float = 35.0
    horizontal_aperture_mm: float = 36.0
    clipping_range: Tuple[float, float] = (0.1, 1000.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], label: str) -> "CameraOpticsConfig":
        _check_keys(cls, data)
        payload = dict(data)
        resolution = _resolution(payload.pop("resolution", (1920, 1080)), f"{label}.resolution")
        clipping = tuple(float(v) for v in payload.pop("clipping_range", (0.1, 1000.0)))
        if len(clipping) != 2 or clipping[0] <= 0 or clipping[1] <= clipping[0]:
            raise ValueError(f"{label}.clipping_range must be [positive_near, larger_far]")
        cfg = cls(resolution=resolution, clipping_range=clipping, **payload)
        if cfg.focal_length_mm <= 0 or cfg.horizontal_aperture_mm <= 0:
            raise ValueError(f"{label}: focal length and aperture must be positive")
        return cfg


@dataclass
class FrontCameraRigConfig:
    near_position: Vec3 = (12.0, -1.5, 4.0)
    view_direction: Vec3 = (-1.0, 0.0, 0.0)
    baseline_direction: Vec3 = (0.0, 1.0, 0.0)
    separation_m: float = 3.0
    # Positive values rotate the rendered image clockwise while looking along
    # the configured optical axis. Both parallel cameras use the same roll.
    image_rotation_clockwise_deg: float = 90.0
    optics: CameraOpticsConfig = field(default_factory=CameraOpticsConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrontCameraRigConfig":
        _check_keys(cls, data)
        payload = dict(data)
        optics = CameraOpticsConfig.from_dict(payload.pop("optics", {}) or {}, "cameras.front.optics")
        near = _vec3(payload.pop("near_position", (12.0, -1.5, 4.0)), "cameras.front.near_position")
        direction = _vec3(payload.pop("view_direction", (-1.0, 0.0, 0.0)), "cameras.front.view_direction")
        baseline = _vec3(
            payload.pop("baseline_direction", (0.0, 1.0, 0.0)),
            "cameras.front.baseline_direction",
        )
        cfg = cls(
            near_position=near,
            view_direction=direction,
            baseline_direction=baseline,
            optics=optics,
            **payload,
        )
        if cfg.separation_m <= 0:
            raise ValueError("cameras.front.separation_m must be positive")
        if sum(v * v for v in cfg.view_direction) < 1e-12:
            raise ValueError("cameras.front.view_direction cannot be zero")
        if sum(v * v for v in cfg.baseline_direction) < 1e-12:
            raise ValueError("cameras.front.baseline_direction cannot be zero")
        dot = sum(a * b for a, b in zip(cfg.view_direction, cfg.baseline_direction))
        if abs(dot) > 1e-9:
            raise ValueError(
                "cameras.front.baseline_direction must be normal to view_direction"
            )
        return cfg


@dataclass
class SatelliteCameraConfig:
    position: Vec3 = (0.0, 0.0, 20.0)
    target: Vec3 = (0.0, 0.0, 2.5)
    optics: CameraOpticsConfig = field(
        default_factory=lambda: CameraOpticsConfig((2048, 2048), 35.0, 36.0)
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SatelliteCameraConfig":
        _check_keys(cls, data)
        payload = dict(data)
        optics_data = payload.pop("optics", None)
        optics = (
            CameraOpticsConfig.from_dict(optics_data or {}, "cameras.satellite.optics")
            if optics_data is not None
            else cls().optics
        )
        position = _vec3(payload.pop("position", (0.0, 0.0, 20.0)), "cameras.satellite.position")
        target = _vec3(payload.pop("target", (0.0, 0.0, 2.5)), "cameras.satellite.target")
        if position == target:
            raise ValueError("cameras.satellite.position and target must differ")
        return cls(position=position, target=target, optics=optics, **payload)


@dataclass
class CamerasConfig:
    enabled: bool = True
    root_prim: str = "/World/Sensors"
    front: FrontCameraRigConfig = field(default_factory=FrontCameraRigConfig)
    satellite: SatelliteCameraConfig = field(default_factory=SatelliteCameraConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CamerasConfig":
        _check_keys(cls, data)
        payload = dict(data)
        front = FrontCameraRigConfig.from_dict(payload.pop("front", {}) or {})
        satellite = SatelliteCameraConfig.from_dict(payload.pop("satellite", {}) or {})
        return cls(front=front, satellite=satellite, **payload)


@dataclass
class MutualOcclusionConfig:
    """Two or more aircraft that overlap and swap places in the same image.

    Separation is guaranteed structurally rather than by a reactive rule: every
    aircraft owns a lane on the camera's optical axis and neighbouring lanes are
    at least ``drones.safety_radius`` apart, so aircraft may share an image
    position -- which is the point -- without ever sharing a 3-D position.
    """

    lane_pitch_m: float | None = None  # None = drones.safety_radius * 1.05
    formation_pitch_m: float = 0.7  # lateral spread inside one team's formation
    leg_scale_range: Tuple[float, float] = (0.92, 1.08)
    #: Ally i and enemy i aim at the same crossing point, so each pass produces a
    #: real image-plane merge instead of two independent transits.
    pair_teams: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutualOcclusionConfig":
        _check_keys(cls, data)
        payload = dict(data)
        scale = payload.pop("leg_scale_range", (0.92, 1.08))
        cfg = cls(leg_scale_range=_range(scale, "scenarios.mutual_occlusion.leg_scale_range"), **payload)
        if cfg.lane_pitch_m is not None and cfg.lane_pitch_m <= 0:
            raise ValueError("scenarios.mutual_occlusion.lane_pitch_m must be positive")
        if cfg.formation_pitch_m < 0:
            raise ValueError("scenarios.mutual_occlusion.formation_pitch_m must be >= 0")
        return cfg


@dataclass
class StructuralOcclusionConfig:
    """Aircraft disappear behind a structure and come back out the other side."""

    occluder_set: str | None = None  # None = the environment preset's loadout
    lane_pitch_m: float | None = None
    #: Wider than the mutual-occlusion range so aircraft enter their occluders at
    #: different times and each track dies alone.
    leg_scale_range: Tuple[float, float] = (0.8, 1.2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuralOcclusionConfig":
        _check_keys(cls, data)
        payload = dict(data)
        scale = payload.pop("leg_scale_range", (0.8, 1.2))
        cfg = cls(
            leg_scale_range=_range(scale, "scenarios.structural_occlusion.leg_scale_range"),
            **payload,
        )
        if cfg.occluder_set is not None and cfg.occluder_set not in OCCLUDER_SET_KEYS:
            raise ValueError(
                "scenarios.structural_occlusion.occluder_set must be one of "
                f"{list(OCCLUDER_SET_KEYS)}"
            )
        if cfg.lane_pitch_m is not None and cfg.lane_pitch_m <= 0:
            raise ValueError("scenarios.structural_occlusion.lane_pitch_m must be positive")
        return cfg


@dataclass
class SensorDropoutConfig:
    """The aircraft keep flying; the observation stops arriving."""

    cameras: List[str] = field(default_factory=lambda: ["front_near", "front_far"])
    dropout_count: int = 6
    min_dropout_s: float = 0.4
    max_dropout_s: float = 2.5
    #: False = one camera goes dark at a time, so the other still sees the truth.
    simultaneous: bool = False
    #: Isolated single-frame losses on top of the scheduled blackouts.
    frame_drop_probability: float = 0.02

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensorDropoutConfig":
        _check_keys(cls, data)
        cfg = cls(**data)
        unknown = [name for name in cfg.cameras if name not in GATEABLE_CAMERAS]
        if unknown:
            raise ValueError(
                f"scenarios.sensor_dropout.cameras: unknown {unknown}; "
                f"expected a subset of {list(GATEABLE_CAMERAS)}"
            )
        if not cfg.cameras:
            raise ValueError("scenarios.sensor_dropout.cameras must not be empty")
        if cfg.dropout_count < 0:
            raise ValueError("scenarios.sensor_dropout.dropout_count must be >= 0")
        if not 0 < cfg.min_dropout_s <= cfg.max_dropout_s:
            raise ValueError(
                "scenarios.sensor_dropout: expected 0 < min_dropout_s <= max_dropout_s"
            )
        if not 0.0 <= cfg.frame_drop_probability < 0.5:
            raise ValueError(
                "scenarios.sensor_dropout.frame_drop_probability must be in [0, 0.5)"
            )
        return cfg


@dataclass
class ScenarioPlanEntry:
    """One line of the episode plan: a scenario flown in an environment."""

    scenario: str
    environment: str
    duration_s: float | None = None  # None = scenarios.episode_duration_s
    seed: int | None = None  # None = derived from scenarios.seed and the index
    repeat: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioPlanEntry":
        _check_keys(cls, data)
        payload = dict(data)
        if "scenario" not in payload or "environment" not in payload:
            raise ValueError("scenarios.plan[]: 'scenario' and 'environment' are required")
        cfg = cls(**payload)
        if cfg.scenario not in SCENARIO_KEYS:
            raise ValueError(
                f"scenarios.plan[].scenario must be one of {list(SCENARIO_KEYS)}, got {cfg.scenario!r}"
            )
        if cfg.environment not in ENVIRONMENT_KEYS:
            raise ValueError(
                f"scenarios.plan[].environment must be one of {list(ENVIRONMENT_KEYS)}, "
                f"got {cfg.environment!r}"
            )
        if cfg.duration_s is not None and cfg.duration_s <= 0:
            raise ValueError("scenarios.plan[].duration_s must be positive")
        if cfg.repeat < 1:
            raise ValueError("scenarios.plan[].repeat must be >= 1")
        return cfg


@dataclass
class ScenariosConfig:
    """The episode plan, and the one episode this process is flying.

    ``plan`` is read by the orchestrator, which writes one fully resolved scene
    YAML per episode and sets ``active``/``seed``/``episode_id`` in it. Every
    process of that episode -- simulator, collector, tracker, detector -- loads
    the same file, which is why the collector can reconstruct the blackout
    schedule and the occluder positions without talking to the simulator.
    """

    enabled: bool = False
    active: str | None = None
    seed: int | None = None  # None = a fresh random plan on every run
    episode_id: str = ""
    run_dir: str = "artifacts/scenarios"
    #: Where each episode's collected frames land; one session directory each.
    dataset_root: str = "artifacts/perception/raw"
    episode_duration_s: float = 60.0
    #: Events start only after takeoff has settled, so no episode opens on a
    #: blackout that would leave the track with nothing to initialise from.
    event_start_s: float = 8.0
    #: Dead time between episodes, so DDS discovery from the previous run is
    #: gone before the next simulator starts.
    cooldown_s: float = 8.0
    plan: List[ScenarioPlanEntry] = field(default_factory=list)
    mutual_occlusion: MutualOcclusionConfig = field(default_factory=MutualOcclusionConfig)
    structural_occlusion: StructuralOcclusionConfig = field(
        default_factory=StructuralOcclusionConfig
    )
    sensor_dropout: SensorDropoutConfig = field(default_factory=SensorDropoutConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenariosConfig":
        _check_keys(cls, data)
        payload = dict(data)
        plan = [ScenarioPlanEntry.from_dict(entry) for entry in payload.pop("plan", []) or []]
        cfg = cls(
            plan=plan,
            mutual_occlusion=MutualOcclusionConfig.from_dict(
                payload.pop("mutual_occlusion", {}) or {}
            ),
            structural_occlusion=StructuralOcclusionConfig.from_dict(
                payload.pop("structural_occlusion", {}) or {}
            ),
            sensor_dropout=SensorDropoutConfig.from_dict(payload.pop("sensor_dropout", {}) or {}),
            **payload,
        )
        if cfg.active is not None and cfg.active not in SCENARIO_KEYS:
            raise ValueError(
                f"scenarios.active must be one of {list(SCENARIO_KEYS)} or null, got {cfg.active!r}"
            )
        if cfg.enabled and cfg.active is None:
            raise ValueError("scenarios.enabled needs scenarios.active to name the scenario")
        if cfg.episode_duration_s <= 0:
            raise ValueError("scenarios.episode_duration_s must be positive")
        if not 0.0 <= cfg.event_start_s < cfg.episode_duration_s:
            raise ValueError(
                "scenarios.event_start_s must be in [0, episode_duration_s)"
            )
        if cfg.cooldown_s < 0:
            raise ValueError("scenarios.cooldown_s must be >= 0")
        return cfg

    def episodes(self) -> List["ScenarioPlanEntry"]:
        """The plan with ``repeat`` expanded, in the order it will be flown."""
        expanded: List[ScenarioPlanEntry] = []
        for entry in self.plan:
            expanded.extend([entry] * entry.repeat)
        return expanded


@dataclass
class TelemetryConfig:
    report_every_s: float = 2.0  # 0 or less disables periodic pose reports

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TelemetryConfig":
        _check_keys(cls, data)
        return cls(**data)


@dataclass
class SensorConfig:
    """Lidar and camera publishing for Nav2 / perception.

    ``prim`` values are relative to the robot prim; leave a section disabled if
    the model has no such sensor.
    """

    lidar_enabled: bool = True
    lidar_prim: str = "chassis_link/sensors/front_RPLidar/RPLidar_S2E"
    lidar_topic: str = "scan"
    lidar_frame: str = "front_lidar"

    camera_enabled: bool = True
    camera_prim: str = "chassis_link/sensors/front_hawk/left/camera_left"
    camera_topic: str = "front/rgb"
    camera_info_topic: str = "front/camera_info"
    camera_frame: str = "front_camera"
    camera_width: int = 640
    camera_height: int = 480

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensorConfig":
        _check_keys(cls, data)
        cfg = cls(**data)
        if cfg.camera_width <= 0 or cfg.camera_height <= 0:
            raise ValueError("sensors: camera_width/height must be positive")
        return cfg


@dataclass
class Ros2Config:
    """ROS 2 bridge wiring.

    When ``enabled`` and ``drive_from_cmd_vel``, the OmniGraph owns the wheels
    and the in-process controller is skipped -- the algorithm runs as an
    external ROS 2 node instead.
    """

    enabled: bool = False
    drive_from_cmd_vel: bool = True
    domain_id: int | None = None  # None = take ROS_DOMAIN_ID from the environment
    node_namespace: str = ""
    clock_topic: str = "clock"
    cmd_vel_topic: str = "cmd_vel"
    odom_topic: str = "odom"
    tf_topic: str = "tf"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    person_frame_prefix: str = "person"
    max_linear_speed: float = 1.0  # m/s clamp applied in the graph
    max_angular_speed: float = 2.0  # rad/s clamp applied in the graph
    publish_people_tf: bool = True
    map_frame: str = "map"
    publish_swarm_tf: bool = True
    publish_swarm_cameras: bool = True
    swarm_image_scale: float = 0.5
    sensors: SensorConfig = field(default_factory=SensorConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ros2Config":
        _check_keys(cls, data)
        payload = dict(data)
        sensors = SensorConfig.from_dict(payload.pop("sensors", {}) or {})
        cfg = cls(sensors=sensors, **payload)
        if cfg.domain_id is not None and not 0 <= cfg.domain_id <= 232:
            raise ValueError("ros2.domain_id must be between 0 and 232")
        if not 0 < cfg.swarm_image_scale <= 1:
            raise ValueError("ros2.swarm_image_scale must be in (0, 1]")
        return cfg

    def person_frame(self, index: int) -> str:
        """Frame id for the ``index``-th person (0-based), matching Person_NN."""
        return f"{self.person_frame_prefix}_{index + 1:02d}"


@dataclass
class SceneConfig:
    """Root of the configuration tree."""

    app: AppConfig = field(default_factory=AppConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    ugv: UGVConfig = field(default_factory=UGVConfig)
    people: PeopleConfig = field(default_factory=PeopleConfig)
    drones: DronesConfig = field(default_factory=DronesConfig)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    scenarios: ScenariosConfig = field(default_factory=ScenariosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneConfig":
        _check_keys(cls, data)
        return cls(
            app=AppConfig.from_dict(data.get("app", {}) or {}),
            world=WorldConfig.from_dict(data.get("world", {}) or {}),
            ugv=UGVConfig.from_dict(data.get("ugv", {}) or {}),
            people=PeopleConfig.from_dict(data.get("people", {}) or {}),
            drones=DronesConfig.from_dict(data.get("drones", {}) or {}),
            cameras=CamerasConfig.from_dict(data.get("cameras", {}) or {}),
            scenarios=ScenariosConfig.from_dict(data.get("scenarios", {}) or {}),
            telemetry=TelemetryConfig.from_dict(data.get("telemetry", {}) or {}),
            ros2=Ros2Config.from_dict(data.get("ros2", {}) or {}),
        )
