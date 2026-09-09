"""One-command experiment orchestration and reproducible research outputs."""

from __future__ import annotations

import csv
import json
import shutil
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# The shared perception venv exposes Ubuntu's optional numexpr/bottleneck and
# mpl_toolkits alongside newer user-site pandas/matplotlib packages.  This
# pipeline uses neither the optional dataframe accelerators nor Axes3D.  Hide
# only those known compatibility notices; all algorithm/runtime warnings remain.
warnings.filterwarnings(
    "ignore",
    message=r"Unable to import Axes3D\..*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version .*",
    category=UserWarning,
    module=r"pandas\..*",
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from simlab.utils.paths import PROJECT_ROOT
from .config import ExperimentConfig
from .matching import FeatureMatcher, GroundTruthLedger, ReferenceKeyframe, ReferenceKeyframeDatabase
from .reasoner import MatchabilityReasoner
from .vision import OpticalFlowAnalyzer, TerrainSceneGenerator, VisualFeatureExtractor


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        if config.deterministic:
            cv2.setRNGSeed(config.seed); cv2.setNumThreads(1)
        self.rng = np.random.default_rng(config.seed)
        self.generator = TerrainSceneGenerator(config.image.size, config.seed)
        self.features = VisualFeatureExtractor(config.image)
        self.flow = OpticalFlowAnalyzer(config.optical_flow)
        self.reasoner = MatchabilityReasoner(config.rules_file)
        self.matcher = FeatureMatcher(config.matching)

    def _run_directory(self) -> Path:
        root = Path(self.cfg.output_directory)
        if not root.is_absolute(): root = PROJECT_ROOT / root
        target = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        suffix = 1
        while target.exists(): target = root / f"{target.name}_{suffix:02d}"; suffix += 1
        (target / "figures").mkdir(parents=True)
        return target

    @staticmethod
    def _ontology_snapshot(path: Path, rules: list[dict[str, Any]]) -> None:
        lines = ["@prefix tv: <https://example.org/terrain-vision#> .", "tv:VisualObservation a tv:OntologyClass .", "tv:VisualMatchability a tv:OntologyClass ."]
        for rule in rules:
            lines.append(f"tv:{rule['id']} a tv:ReasoningRule ; tv:observesMetric \"{rule['metric']}\" ; tv:contribution \"{rule['contribution']}\" .")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> Path:
        out = self._run_directory(); database = ReferenceKeyframeDatabase(); truth = GroundTruthLedger()
        references: dict[str, np.ndarray] = {}
        for index, terrain in enumerate(self.cfg.terrain_classes):
            image = self.generator.generate(terrain)
            obs, kp, desc = self.features.extract(f"reference_{terrain}", image, 0.0, terrain)
            key = f"kf_{terrain}"
            database.add(ReferenceKeyframe(key, image, tuple(kp), desc, dict(obs.normalized)))
            truth.reference(key, (index * 100.0, 0.0)); references[terrain] = image

        schema_rows, match_rows, traces = [], [], []
        frame_index = 0
        for repetition in range(self.cfg.repetitions):
            for terrain_index, terrain in enumerate(self.cfg.terrain_classes):
                reference = references[terrain]
                for condition in self.cfg.conditions:
                    frame_id = f"r{repetition:02d}_{terrain}_{condition}"
                    query = self.generator.perturb(reference, condition, self.rng)
                    obs, kp, desc = self.features.extract(frame_id, query, frame_index / 10.0, terrain)
                    self.flow.update(reference, query, obs); score = self.reasoner.evaluate(obs)
                    truth.query(frame_id, (terrain_index * 100.0 + 1.0, 0.0), f"kf_{terrain}")
                    baseline = self.matcher.localize(kp, desc, database, guided=False)
                    guided = self.matcher.localize(kp, desc, database, score, guided=True)
                    evaluated = truth.evaluate(frame_id, guided, self.cfg.matching.correct_tolerance_m)
                    schema_rows.append(obs.to_dict())
                    row = {"frame_id": frame_id, "ground_truth_terrain": terrain, "condition": condition, "visual_matchability": score, "predicted_matchability_class": obs.matchability_class, **asdict(guided), **evaluated}
                    row.update({f"score_{k}": v for k, v in self.reasoner.baseline_scores(obs).items()})
                    row["baseline_selected_reference"] = baseline.selected_reference_keyframe
                    row["baseline_correct"] = baseline.selected_reference_keyframe == f"kf_{terrain}"
                    match_rows.append(row); traces.append({"frame_id": frame_id, "score": score, "trace": obs.reasoning_trace})
                    frame_index += 1

        flat_schema = []
        for source_row in schema_rows:
            row = dict(source_row)
            normalized = row.pop("normalized"); trace = row.pop("reasoning_trace")
            flat_schema.append({**row, **{f"normalized_{k}": v for k, v in normalized.items()}, "reasoning_trace": json.dumps(trace)})
        schema_df, matching_df = pd.DataFrame(flat_schema), pd.DataFrame(match_rows)
        schema_df.to_csv(out / "frame_level_schema.csv", index=False)
        (out / "frame_level_schema.json").write_text(json.dumps(schema_rows, indent=2), encoding="utf-8")
        matching_df.to_csv(out / "matching_results.csv", index=False)
        matching_df[["frame_id", "selected_reference_keyframe", "translation_error", "rotation_error", "is_correct_place_match"]].to_csv(out / "localization_results.csv", index=False)
        with (out / "reasoning_trace.jsonl").open("w", encoding="utf-8") as stream:
            for record in traces: stream.write(json.dumps(record) + "\n")

        methods = [
            {"method": "B0_Fixed_ORB", "metric": "relocalization_success_rate", "value": matching_df["baseline_correct"].mean()},
            {"method": "P_Ontology_Guided", "metric": "relocalization_success_rate", "value": matching_df["is_correct_place_match"].mean()},
        ]
        for name in ("B1_Feature_Count", "B2_Terrain_Label", "B3_Fixed_Handcrafted_Feature_Weights"):
            methods.append({"method": name, "metric": "spearman_score_vs_inlier_ratio", "value": float(matching_df[f"score_{name}"].corr(matching_df["matching_inlier_ratio"], method="spearman"))})
        methods.append({"method": "Ablation_Same_Features_Without_Ontology", "metric": "spearman_score_vs_inlier_ratio", "value": methods[-1]["value"]})
        success = matching_df["is_correct_place_match"].astype(bool)
        attempted = matching_df["selected_reference_keyframe"].notna()
        false_places = attempted & ~success
        n = len(matching_df); success_rate = float(success.mean())
        ci_half = 1.96 * float(np.sqrt(success_rate * (1-success_rate) / max(1, n)))
        summary_rows = [
            {"metric": "relocalization_success_rate", "value": success_rate},
            {"metric": "relocalization_failure_rate", "value": 1-success_rate},
            {"metric": "false_place_match_rate_per_attempt", "value": float(false_places.sum() / max(1, attempted.sum()))},
            {"metric": "unmatched_query_rate", "value": float((~attempted).mean())},
            {"metric": "success_rate_95ci_low", "value": max(0.0, success_rate-ci_half)},
            {"metric": "success_rate_95ci_high", "value": min(1.0, success_rate+ci_half)},
            {"metric": "matchability_inlier_pearson", "value": matching_df["visual_matchability"].corr(matching_df["matching_inlier_ratio"])},
            {"metric": "matchability_inlier_spearman", "value": matching_df["visual_matchability"].corr(matching_df["matching_inlier_ratio"], method="spearman")},
            {"metric": "roc_auc_predicting_match_success", "value": float(roc_auc_score(success, matching_df["visual_matchability"])) if success.nunique() == 2 else float("nan")},
            {"metric": "median_attempted_position_error_m", "value": matching_df.loc[attempted, "translation_error"].median()},
            {"metric": "position_rmse_m", "value": float(np.sqrt(np.mean(np.square(matching_df.loc[attempted, "translation_error"])))) if attempted.any() else float("nan")},
        ]
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out / "summary_metrics.csv", index=False)
        pd.DataFrame(methods).to_csv(out / "ablation_metrics.csv", index=False)
        schema_df.groupby("terrain_type_if_available").mean(numeric_only=True).to_csv(out / "terrain_statistics.csv")
        calibration = matching_df.assign(score_bin=pd.cut(matching_df.visual_matchability, bins=np.linspace(0, 1, 6), include_lowest=True)).groupby("score_bin", observed=False).agg(mean_predicted_matchability=("visual_matchability", "mean"), observed_match_success=("is_correct_place_match", "mean"), sample_count=("frame_id", "count")).reset_index()
        calibration.to_csv(out / "calibration_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(matching_df.visual_matchability, matching_df.matching_inlier_ratio, c=pd.Categorical(matching_df.ground_truth_terrain).codes); ax.set(xlabel="Ontology visual matchability", ylabel="RANSAC inlier ratio", title="Predicted vs. observed matching reliability"); fig.tight_layout(); fig.savefig(out / "figures" / "matchability_vs_inlier_ratio.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); matching_df.boxplot(column="visual_matchability", by="ground_truth_terrain", ax=ax); fig.suptitle(""); ax.set_title("Terrain-wise visual matchability"); fig.tight_layout(); fig.savefig(out / "figures" / "terrain_matchability.png", dpi=180); plt.close(fig)

        with (out / "resolved_config.yaml").open("w", encoding="utf-8") as stream: yaml.safe_dump(asdict(self.cfg), stream, sort_keys=False)
        rules_path = Path(self.cfg.rules_file); rules_path = rules_path if rules_path.is_absolute() else PROJECT_ROOT / rules_path
        shutil.copy2(rules_path, out / "ontology_rules.yaml")
        self._ontology_snapshot(out / "ontology_snapshot.ttl", self.reasoner.rules)
        (out / "run_log.txt").write_text(f"seed={self.cfg.seed}\nframes={len(matching_df)}\nsuccess_rate={matching_df['is_correct_place_match'].mean():.6f}\n", encoding="utf-8")
        return out
