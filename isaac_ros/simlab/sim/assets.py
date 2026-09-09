"""Catalog of Isaac Sim asset paths, and resolution against the asset root.

The catalog is plain data, so this module is importable without Isaac Sim
running; only :func:`asset_root` touches ``isaacsim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class RobotSpec:
    """Everything the sim layer needs to drive a wheeled base."""

    usd: str  # path under the asset root
    wheel_hints: Tuple[str, str]  # (left, right) joint names
    wheel_radius: float  # m
    wheel_base: float  # m, distance between the drive wheels
    spawn_height: float  # m, z at spawn so the base clears the ground


@dataclass(frozen=True)
class DroneAssetSpec:
    """Official Isaac Sim aerial robot asset and its ontology identity."""

    usd: str
    manufacturer: str
    platform: str
    nominal_width_m: float


ROBOTS: Dict[str, RobotSpec] = {
    "nova_carter": RobotSpec(
        usd="/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        wheel_hints=("joint_wheel_left", "joint_wheel_right"),
        wheel_radius=0.14,
        wheel_base=0.413,
        spawn_height=0.15,
    ),
    "jetbot": RobotSpec(
        usd="/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
        wheel_hints=("left_wheel_joint", "right_wheel_joint"),
        wheel_radius=0.0325,
        wheel_base=0.1125,
        spawn_height=0.03,
    ),
}

DRONES: Dict[str, DroneAssetSpec] = {
    "crazyflie": DroneAssetSpec(
        usd="/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        manufacturer="Bitcraze",
        platform="Crazyflie 2.X",
        nominal_width_m=0.11,
    ),
    "quadcopter": DroneAssetSpec(
        usd="/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd",
        manufacturer="NVIDIA",
        platform="Isaac Sim Quadcopter",
        nominal_width_m=0.85,
    ),
    "iris": DroneAssetSpec(
        usd="assets/external/PegasusSimulator/Iris/iris.usd",
        manufacturer="3D Robotics",
        platform="Iris Quadrotor (Pegasus Simulator)",
        nominal_width_m=0.53,
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHARACTERS: Dict[str, str] = {
    "F_Business_02": "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "F_Medical_01": "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd",
    "M_Medical_01": "/Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "male_adult_construction_01_new": (
        "/Isaac/People/Characters/male_adult_construction_01_new/"
        "male_adult_construction_01_new.usd"
    ),
    "female_adult_police_01_new": (
        "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
    ),
}

# Shared skeleton + animation graph rig that every character binds to.
BIPED_SETUP = "/Isaac/People/Characters/Biped_Setup.usd"


def robot_spec(model: str) -> RobotSpec:
    try:
        return ROBOTS[model]
    except KeyError:
        raise ValueError(f"unknown ugv model {model!r}; available: {sorted(ROBOTS)}") from None


def drone_spec(model: str) -> DroneAssetSpec:
    try:
        return DRONES[model]
    except KeyError:
        raise ValueError(f"unknown drone model {model!r}; available: {sorted(DRONES)}") from None


def drone_asset_url(spec: DroneAssetSpec, isaac_assets_root: str) -> str:
    """Resolve official Isaac paths or repository-vendored external assets."""
    if spec.usd.startswith("/"):
        return isaac_assets_root + spec.usd
    path = (PROJECT_ROOT / spec.usd).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"drone asset is missing: {path}")
    return str(path)


def character_usd(asset: str) -> str:
    """Accept either a catalog key or a literal path under the asset root."""
    if asset.startswith("/"):
        return asset
    try:
        return CHARACTERS[asset]
    except KeyError:
        raise ValueError(
            f"unknown character {asset!r}; available: {sorted(CHARACTERS)} "
            "(or pass a path starting with '/')"
        ) from None


def asset_root() -> str:
    """Resolve the Isaac Sim asset root (local Nucleus or the public S3 mirror)."""
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Could not resolve the Isaac Sim assets root. "
            "Check network access to the Omniverse content server."
        )
    return root
