"""Deterministic events, semantic managers, and quantitative change truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import EpochConfig, TemporalConfig
from .managers import BuildingChangeManager,LightingManager,RoadChangeManager,SeasonManager,VegetationChangeManager,WeatherManager


@dataclass(frozen=True)
class TemporalEvent:
    event_id: str
    virtual_year: float
    target_prim: str
    event_type: str
    parameters: dict[str, Any]
    change_ground_truth: float


class TemporalEventScheduler:
    def __init__(self, configured: tuple[dict[str, Any], ...], seed: int):
        self.seed = seed
        self.events = tuple(sorted((TemporalEvent(**item) for item in configured), key=lambda e: (e.virtual_year, e.event_id)))
        if len({e.event_id for e in self.events}) != len(self.events): raise ValueError("event IDs must be unique")
    def through(self, year: float) -> tuple[TemporalEvent, ...]: return tuple(e for e in self.events if e.virtual_year <= year)
    def replay_record(self) -> list[dict[str, Any]]: return [asdict(e) | {"random_seed": self.seed} for e in self.events]


@dataclass
class EnvironmentState:
    epoch: str
    virtual_year: float
    tau: float
    season: str
    weather: str
    vegetation_change_score: float = 0
    building_change_score: float = 0
    road_appearance_change_score: float = 0
    road_geometry_change_score: float = 0
    season_change_score: float = 0
    weather_change_score: float = 0
    illumination_change_score: float = 0
    environment_change_score: float = 0
    structural_change_score: float = 0
    appearance_change_score: float = 0
    dynamic_environment_score: float = 0
    tree_scale: float = 1
    visible_tree_count: int = 0
    building_variant: str = "EmptyLot"
    road_width_scale: float = 1
    road_marking_visibility: float = 1
    lighting_scale: float = 1
    applied_event_ids: list[str] = field(default_factory=list)


class EnvironmentStateFactory:
    def __init__(self, cfg: TemporalConfig, scheduler: TemporalEventScheduler):
        self.cfg,self.scheduler=cfg,scheduler;self.vegetation=VegetationChangeManager(cfg.changes);self.buildings=BuildingChangeManager(cfg.changes);self.roads=RoadChangeManager(cfg.changes);self.seasons=SeasonManager();self.weather=WeatherManager();self.lighting=LightingManager()

    def create(self, epoch: EpochConfig) -> EnvironmentState:
        c = self.cfg.changes; tau = epoch.year/self.cfg.time.maximum_year
        events = self.scheduler.through(epoch.year); event_types = {e.event_type for e in events}
        state = EnvironmentState(epoch.name, epoch.year, tau, epoch.season, epoch.weather)
        self.vegetation.apply(state,tau);self.buildings.apply(state,event_types);self.roads.apply(state,tau);self.seasons.apply(state);self.weather.apply(state);self.lighting.apply(state)
        state.structural_change_score = .6*state.building_change_score + .4*state.road_geometry_change_score
        state.appearance_change_score = np.mean([state.vegetation_change_score, state.road_appearance_change_score, state.season_change_score, state.weather_change_score, state.illumination_change_score]).item()
        allowed = {"vegetation", "building", "road_appearance", "road_geometry", "season", "weather", "illumination"}
        unknown = set(epoch.component_overrides) - allowed
        if unknown: raise ValueError(f"{epoch.name}: unknown component overrides {sorted(unknown)}")
        for key, value in epoch.component_overrides.items():
            if not 0 <= value <= 1: raise ValueError(f"{epoch.name}.{key} outside [0, 1]")
            setattr(state, f"{key}_change_score", float(value))
        if epoch.component_overrides:
            state.tree_scale = c.tree_scale_range[0] + state.vegetation_change_score*(c.tree_scale_range[1]-c.tree_scale_range[0])
            state.visible_tree_count = c.tree_initial_count + round(c.tree_new_count*state.vegetation_change_score)
            state.building_variant = "Completed" if state.building_change_score >= .6 else ("Construction" if state.building_change_score >= .25 else "EmptyLot")
            state.road_width_scale = 1 + .35*state.road_geometry_change_score
            state.road_marking_visibility = max(.25, 1-.7*state.road_appearance_change_score)
            state.lighting_scale = max(.38, 1-.48*state.illumination_change_score)
        state.structural_change_score = .6*state.building_change_score + .4*state.road_geometry_change_score
        state.appearance_change_score = np.mean([state.vegetation_change_score, state.road_appearance_change_score, state.season_change_score, state.weather_change_score, state.illumination_change_score]).item()
        components = {"vegetation": state.vegetation_change_score, "building": state.building_change_score, "road_appearance": state.road_appearance_change_score, "road_geometry": state.road_geometry_change_score, "season": state.season_change_score, "weather": state.weather_change_score, "illumination": state.illumination_change_score}
        state.environment_change_score = float(sum(self.cfg.score_weights[k]*v for k,v in components.items()))
        state.applied_event_ids = [e.event_id for e in events]
        return state
