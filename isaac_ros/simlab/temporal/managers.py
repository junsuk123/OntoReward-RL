"""Independent semantic change managers operating on one mutable state record."""

from __future__ import annotations


class VegetationChangeManager:
    def __init__(self,changes):self.changes=changes
    def apply(self,state,tau):
        c=self.changes;state.tree_scale=c.tree_scale_range[0]+tau*(c.tree_scale_range[1]-c.tree_scale_range[0]);state.visible_tree_count=c.tree_initial_count+round(c.tree_new_count*tau);state.vegetation_change_score=min(1.,.65*tau+.35*(state.visible_tree_count-c.tree_initial_count)/max(1,c.tree_new_count))


class BuildingChangeManager:
    def __init__(self,changes):self.changes=changes
    def apply(self,state,event_types):
        c=self.changes
        if state.virtual_year>=c.building_renovate_year:state.building_variant,state.building_change_score="Renovated",1.
        elif state.virtual_year>=c.building_complete_year:state.building_variant,state.building_change_score="Completed",.78
        elif "BUILDING_CONSTRUCTION_START" in event_types:state.building_variant,state.building_change_score="Construction",.42


class RoadChangeManager:
    def __init__(self,changes):self.changes=changes
    def apply(self,state,tau):
        c=self.changes;state.road_appearance_change_score=.65 if state.virtual_year>=c.road_resurface_year else .35*tau;state.road_marking_visibility=max(.25,1-.7*tau) if state.virtual_year<c.road_resurface_year else .95
        if state.virtual_year>=c.road_expand_year:state.road_geometry_change_score,state.road_width_scale=.9,1.35


class SeasonManager:
    DISTANCE={"spring":.15,"summer":0.,"autumn":.72,"winter":1.}
    def apply(self,state):state.season_change_score=self.DISTANCE[state.season]


class WeatherManager:
    DISTANCE={"clear":0.,"cloudy":.35,"fog":.75,"rain":.65,"snow":1.}
    def apply(self,state):state.weather_change_score=self.DISTANCE[state.weather];state.dynamic_environment_score=state.weather_change_score


class LightingManager:
    def apply(self,state):state.illumination_change_score=min(1.,.55*state.season_change_score+.45*state.weather_change_score);state.lighting_scale=max(.38,1-.48*state.illumination_change_score)
