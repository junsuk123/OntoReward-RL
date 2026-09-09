"""Per-aircraft flight plans for each scenario.

The shuttle itself does not change between scenarios -- a straight traversal of
the activity bar with a randomized crossing point somewhere in the middle. What
changes is who shares a crossing point with whom, how fast each aircraft flies,
and how deep along the camera's optical axis it sits.

Lanes are the safety mechanism. Every aircraft owns a lane on the camera axis
and neighbouring lanes are at least ``drones.safety_radius`` apart, so two
aircraft can occupy the same pixel -- which is the entire point of the mutual
occlusion scenario -- while never coming within the safety radius in 3-D. No
reactive avoidance rule is needed, and none can be defeated by an unlucky seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from simlab.config.schema import DronesConfig, ScenariosConfig
from simlab.scenarios.timeline import stable_unit


@dataclass(frozen=True)
class LanePlan:
    """One aircraft's assignment for the whole episode."""

    team: str
    team_index: int
    lane_x: float  # depth along the front cameras' optical axis
    start_side: float  # +1 or -1: which end of the activity bar it starts at
    leg_duration_s: float
    crossing_group: int  # aircraft sharing a group aim at the same crossing point
    formation: str
    formation_offset_y: float
    half_length: float

    def as_dict(self) -> dict:
        return {
            "team": self.team,
            "team_index": self.team_index,
            "lane_x": round(self.lane_x, 3),
            "start_side": self.start_side,
            "leg_duration_s": round(self.leg_duration_s, 3),
            "crossing_group": self.crossing_group,
            "formation": self.formation,
            "formation_offset_y": round(self.formation_offset_y, 3),
        }


def speed_scale_bound(drones: DronesConfig) -> float:
    """Largest leg-speed multiplier that still respects ``drones.max_speed``.

    The fastest stretch of a shuttle leg is the run into a crossing point placed
    at the far end of its range, taken in the shortest half of the leg.
    """
    reach = drones.activity_half_length + drones.crossing_offset_limit
    share = max(0.05, 0.5 - drones.crossing_time_jitter)
    fastest = reach / (share * drones.shuttle_leg_duration_s)
    return max(0.5, drones.max_speed / fastest) if fastest > 0 else 1.0


def _speed_scale(drones: DronesConfig, low: float, high: float, seed: int, index: int) -> float:
    bound = speed_scale_bound(drones)
    low = min(max(0.5, low), bound)
    high = min(max(low, high), bound)
    if high <= low:
        return low
    return low + (high - low) * stable_unit(seed, 7717, index)


def lane_plans(
    drones: DronesConfig, scenarios: ScenariosConfig | None, seed: int
) -> List[LanePlan]:
    """One plan per aircraft, in roster order (every friendly, then every enemy)."""
    teams = (("friendly", drones.friendly), ("enemy", drones.enemy))
    scenario = scenarios.active if scenarios is not None and scenarios.enabled else None
    if scenario is None:
        return _default_plans(drones, teams)

    if scenario == "mutual_occlusion":
        tuning = scenarios.mutual_occlusion
        lane_pitch = tuning.lane_pitch_m or drones.safety_radius * 1.05
        formation_pitch = tuning.formation_pitch_m
        low, high = tuning.leg_scale_range
        pair_teams = tuning.pair_teams
    elif scenario == "structural_occlusion":
        tuning = scenarios.structural_occlusion
        lane_pitch = tuning.lane_pitch_m or drones.safety_radius * 1.05
        formation_pitch = 0.0
        low, high = tuning.leg_scale_range
        pair_teams = False
    else:  # sensor_dropout: plain, predictable flight so the gap is the variable
        lane_pitch = drones.safety_radius * 1.05
        formation_pitch = 0.0
        low = high = 1.0
        pair_teams = True

    plans: List[LanePlan] = []
    global_index = 0
    for team_name, team in teams:
        centre = (team.count - 1) * 0.5
        for index in range(team.count):
            side = -1.0 if team_name == "friendly" else 1.0
            offset = (index - centre) * formation_pitch
            group = index if pair_teams else global_index
            # The speed follows the crossing group, not the aircraft: a pair that
            # aims at the same crossing point must also arrive there at the same
            # moment, or it never merges in the image.
            scale = _speed_scale(drones, low, high, seed, group)
            plans.append(
                LanePlan(
                    team=team_name,
                    team_index=index,
                    lane_x=side * (drones.camera_depth_separation_m * 0.5 + index * lane_pitch),
                    start_side=side,
                    leg_duration_s=drones.shuttle_leg_duration_s / scale,
                    crossing_group=group,
                    formation=team.label,
                    formation_offset_y=offset,
                    half_length=max(1.0, drones.activity_half_length - abs(offset)),
                )
            )
            global_index += 1
    return plans


def _default_plans(drones: DronesConfig, teams) -> List[LanePlan]:
    """The plain two-aircraft crossing shuttle used when no scenario is active.

    Kept byte-identical to the pre-scenario behaviour: sides alternate by global
    roster index and everyone shares one crossing schedule.
    """
    plans: List[LanePlan] = []
    global_index = 0
    for team_name, team in teams:
        for index in range(team.count):
            side = -1.0 if global_index % 2 == 0 else 1.0
            plans.append(
                LanePlan(
                    team=team_name,
                    team_index=index,
                    lane_x=side * drones.camera_depth_separation_m * 0.5,
                    start_side=side,
                    leg_duration_s=drones.shuttle_leg_duration_s,
                    crossing_group=0,
                    formation=team.label,
                    formation_offset_y=0.0,
                    half_length=drones.activity_half_length,
                )
            )
            global_index += 1
    return plans
