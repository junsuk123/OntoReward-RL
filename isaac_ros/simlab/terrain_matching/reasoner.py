"""External-rule, explainable matchability reasoning and simple baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from simlab.utils.paths import PROJECT_ROOT
from .schema import TerrainVisualObservation


class MatchabilityReasoner:
    def __init__(self, rules_path: str | Path):
        path = Path(rules_path)
        if not path.is_absolute(): path = PROJECT_ROOT / path
        with path.open(encoding="utf-8") as stream: self.document = yaml.safe_load(stream)
        self.base_score = float(self.document["base_score"])
        self.rules = list(self.document["rules"])
        self.class_thresholds = self.document["classes"]

    @staticmethod
    def _matches(value: float, operator: str, threshold: float) -> bool:
        return value >= threshold if operator == ">=" else value <= threshold

    def evaluate(self, obs: TerrainVisualObservation, excluded_groups: set[str] | None = None) -> float:
        excluded_groups = excluded_groups or set()
        score, trace = self.base_score, []
        for rule in self.rules:
            if rule.get("group") in excluded_groups: continue
            metric = rule["metric"]; value = float(obs.normalized.get(metric, 0.0))
            fired = self._matches(value, rule["operator"], float(rule["threshold"]))
            contribution = float(rule["contribution"]) if fired else 0.0
            score += contribution
            trace.append({"rule": rule["id"], "metric": metric, "value": round(value, 6), "condition": f"{rule['operator']} {rule['threshold']}", "fired": fired, "contribution": contribution, "explanation": rule["explanation"]})
        score = min(1.0, max(0.0, score))
        low, high = float(self.class_thresholds["low_max"]), float(self.class_thresholds["high_min"])
        obs.visual_matchability = score
        obs.matchability_class = "LowMatchability" if score <= low else ("HighMatchability" if score >= high else "MediumMatchability")
        obs.reasoning_trace = trace
        obs.low_texture_suspected = obs.normalized["texture_entropy"] < 0.35 and obs.corner_density < 0.15
        obs.repetitive_pattern_suspected = obs.repetitiveness_score > 0.45
        obs.dynamic_texture_suspected = obs.flow_valid_ratio > 0.5 and obs.flow_inlier_ratio < 0.4
        obs.validate()
        return score

    @staticmethod
    def baseline_scores(obs: TerrainVisualObservation) -> dict[str, float]:
        terrain_prior = {"road": .50, "grass": .42, "urban_building": .78, "water": .22}
        n = obs.normalized
        handcrafted = .22*n["feature_count"] + .20*n["texture_entropy"] + .18*n["feature_distribution_uniformity"] + .16*n.get("flow_inlier_ratio", 0) + .12*n["blur_quality"] + .12*(1-n["repetitiveness_score"])
        return {"B0_Fixed_ORB": .5, "B1_Feature_Count": n["feature_count"], "B2_Terrain_Label": terrain_prior.get(obs.terrain_type_if_available or "", .5), "B3_Fixed_Handcrafted_Feature_Weights": float(handcrafted)}
