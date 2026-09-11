"""Reproducible benchmark interfaces kept separate from the urban experiment."""

from .shin2026 import (ActorObservation, CriticObservation, RewardSignals,
                       ShinBenchmarkConfig, assert_actor_payload_safe,
                       default_shin2026_config)

__all__ = [
    "ActorObservation", "CriticObservation", "RewardSignals",
    "ShinBenchmarkConfig", "assert_actor_payload_safe",
    "default_shin2026_config",
]
