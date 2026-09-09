"""Everything both sides of an episode must agree on, derived from one config.

The simulator builds the stage from :func:`scene_boxes`; the dataset collector,
in another process, calls the same function on the same episode YAML to know
what stands between each camera and each aircraft. Keeping the derivation here
-- rather than reading the live USD stage -- is what makes offline labelling and
online rendering describe the same world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from simlab.config.schema import SceneConfig
from simlab.scenarios.environments import environment, environment_boxes
from simlab.scenarios.occlusion import Box
from simlab.sim.assets import drone_spec

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RosterEntry:
    """One aircraft as the perception side knows it, before any detection."""

    drone_id: str
    team: str
    model: str
    formation: str
    diameter_m: float

    @property
    def radius_m(self) -> float:
        return self.diameter_m * 0.5


def environment_seed(scene: SceneConfig) -> int:
    """Layout seed: explicit if set, otherwise the episode seed, otherwise 0."""
    if scene.world.environment_seed is not None:
        return int(scene.world.environment_seed)
    if scene.scenarios.seed is not None:
        return int(scene.scenarios.seed)
    return 0


def occluder_set(scene: SceneConfig) -> str | None:
    """Which sight-line occluders this episode gets.

    Scenarios are kept separable: only the structural-occlusion episodes put
    structures in front of the cameras. A mutual-occlusion episode where an
    aircraft also vanishes behind a billboard cannot answer whether the tracker
    survived the *merge*, and the same goes for a dropout episode. An explicit
    ``world.occluder_set`` always wins, for when that mix is what you want.
    """
    if scene.world.occluder_set is not None:
        return scene.world.occluder_set
    if not scene.scenarios.enabled:
        return None  # the environment preset's own loadout
    if scene.scenarios.active == "structural_occlusion":
        return scene.scenarios.structural_occlusion.occluder_set
    return "none"


def scene_boxes(scene: SceneConfig) -> List[Box]:
    """Every static box of this episode; empty when the world has no scenery."""
    if scene.world.environment_type != "procedural_urban":
        return []
    return environment_boxes(
        environment(scene.world.environment),
        seed=environment_seed(scene),
        occluder_set=occluder_set(scene),
    )


def camera_positions(scene: SceneConfig) -> Dict[str, Vec3]:
    """World positions of the three sensors, keyed by the ID used on the topics."""
    front = scene.cameras.front
    baseline = front.baseline_direction
    length = sum(component * component for component in baseline) ** 0.5 or 1.0
    far = tuple(
        front.near_position[axis] + baseline[axis] / length * front.separation_m
        for axis in range(3)
    )
    return {
        "front_near": tuple(float(v) for v in front.near_position),  # type: ignore[dict-item]
        "front_far": far,  # type: ignore[dict-item]
        "satellite_nadir": tuple(float(v) for v in scene.cameras.satellite.position),  # type: ignore[dict-item]
    }


def drone_roster(scene: SceneConfig) -> Dict[str, RosterEntry]:
    """Track IDs to identity, in the same order the simulator spawns them."""
    roster: Dict[str, RosterEntry] = {}
    for team_name, team in (("friendly", scene.drones.friendly), ("enemy", scene.drones.enemy)):
        prefix = "".join(ch if ch.isalnum() else "_" for ch in team.label).strip("_")
        prefix = prefix or ("Ally" if team_name == "friendly" else "Enemy")
        diameter = drone_spec(team.model).nominal_width_m * team.asset_scale
        for index in range(team.count):
            drone_id = f"{prefix}_{index + 1:02d}"
            roster[drone_id] = RosterEntry(
                drone_id=drone_id,
                team=team_name,
                model=team.model,
                formation=team.label,
                diameter_m=diameter,
            )
    return roster
