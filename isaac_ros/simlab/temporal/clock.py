"""Virtual environmental clock, independent of vehicle physics time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentTime:
    simulation_seconds: float
    virtual_day: float
    virtual_month: float
    virtual_year: float
    normalized_environment_time: float


class EnvironmentTimeManager:
    def __init__(self, days_per_real_second: float, maximum_year: float):
        if days_per_real_second <= 0 or maximum_year <= 0: raise ValueError("time scale and maximum year must be positive")
        self.days_per_real_second = days_per_real_second; self.maximum_year = maximum_year
        self.simulation_seconds = 0.0; self.virtual_day = 0.0; self.playing = False

    @property
    def state(self) -> EnvironmentTime:
        year = self.virtual_day / 365.0
        return EnvironmentTime(self.simulation_seconds, self.virtual_day, self.virtual_day / (365/12), year, min(1.0, year/self.maximum_year))

    def play(self) -> None: self.playing = True
    def pause(self) -> None: self.playing = False
    def reset(self) -> None: self.simulation_seconds = 0; self.virtual_day = 0; self.playing = False
    def update(self, physics_seconds: float) -> EnvironmentTime:
        if physics_seconds < 0: raise ValueError("physics time cannot run backwards")
        self.simulation_seconds += physics_seconds
        if self.playing: self.virtual_day = min(self.maximum_year*365, self.virtual_day + physics_seconds*self.days_per_real_second)
        return self.state
    def advance_one_day(self): self.virtual_day = min(self.maximum_year*365, self.virtual_day+1); return self.state
    def advance_one_month(self): self.virtual_day = min(self.maximum_year*365, self.virtual_day+365/12); return self.state
    def advance_one_year(self): self.virtual_day = min(self.maximum_year*365, self.virtual_day+365); return self.state
    def jump_to_epoch(self, virtual_year: float) -> EnvironmentTime:
        if not 0 <= virtual_year <= self.maximum_year: raise ValueError("epoch outside configured range")
        self.virtual_day = virtual_year*365; return self.state
    def set_time_scale(self, days_per_real_second: float) -> None:
        if days_per_real_second <= 0: raise ValueError("time scale must be positive")
        self.days_per_real_second = days_per_real_second
