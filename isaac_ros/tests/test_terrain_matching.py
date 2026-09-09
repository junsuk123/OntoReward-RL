"""Unit and integration coverage for the GNSS-denied terrain pipeline."""

import inspect
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np

from simlab.terrain_matching.config import ExperimentConfig, ImageConfig, load_experiment_config
from simlab.terrain_matching.matching import FeatureMatcher, GroundTruthLedger, ReferenceKeyframe, ReferenceKeyframeDatabase
from simlab.terrain_matching.pipeline import ExperimentRunner
from simlab.terrain_matching.reasoner import MatchabilityReasoner
from simlab.terrain_matching.schema import TerrainVisualObservation
from simlab.terrain_matching.vision import OpticalFlowAnalyzer, TerrainSceneGenerator, VisualFeatureExtractor


class SchemaAndReasoningTest(unittest.TestCase):
    def test_schema_rejects_invalid_normalized_metric(self):
        obs = TerrainVisualObservation("f", 0, 1, 1, 1.2, 0, 0, 0, 0, 127, 10)
        with self.assertRaises(ValueError): obs.validate()

    def test_score_is_bounded_and_traceable(self):
        obs = TerrainVisualObservation("f", 0, 6, 200, .2, .2, 300, .8, .1, 127, 500)
        obs.normalized = {"texture_entropy": .8, "corner_density": .2, "feature_count": .8, "feature_distribution_uniformity": .8, "repetitiveness_score": .1, "brightness_quality": 1, "blur_quality": 1, "flow_valid_ratio": .9, "flow_inlier_ratio": .9, "flow_direction_consistency": .9}
        score = MatchabilityReasoner("config/ontology_rules.yaml").evaluate(obs)
        self.assertLessEqual(score, 1); self.assertGreaterEqual(score, 0)
        self.assertTrue(any(item["fired"] for item in obs.reasoning_trace))

    def test_ground_truth_fields_do_not_exist_in_algorithm_records(self):
        forbidden = {"position_ground_truth", "ground_truth_position", "gnss", "correct_reference"}
        self.assertFalse(forbidden & {field.name for field in fields(TerrainVisualObservation)})
        self.assertFalse(forbidden & {field.name for field in fields(ReferenceKeyframe)})
        source = inspect.getsource(FeatureMatcher).lower()
        self.assertNotIn("groundtruth", source); self.assertNotIn("gnss", source)


class VisionIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.generator = TerrainSceneGenerator(256, 7)
        self.extractor = VisualFeatureExtractor(ImageConfig(size=256, orb_features=500))

    def test_known_transform_produces_geometric_inliers(self):
        source = self.generator.generate("urban_building")
        query = self.generator.perturb(source, "normal", np.random.default_rng(7))
        ref_obs, ref_kp, ref_desc = self.extractor.extract("ref", source, 0)
        query_obs, query_kp, query_desc = self.extractor.extract("query", query, 1)
        database = ReferenceKeyframeDatabase(); database.add(ReferenceKeyframe("kf", source, tuple(ref_kp), ref_desc, ref_obs.normalized))
        result = FeatureMatcher(load_experiment_config().matching).localize(query_kp, query_desc, database)
        self.assertGreater(result.geometric_inlier_count, 10)
        self.assertLess(result.reprojection_rmse, 2.0)

    def test_low_texture_water_is_less_feature_rich_than_urban(self):
        water, _, _ = self.extractor.extract("water", self.generator.generate("water"), 0)
        urban, _, _ = self.extractor.extract("urban", self.generator.generate("urban_building"), 0)
        self.assertLess(water.feature_count, urban.feature_count)


class EndToEndTest(unittest.TestCase):
    def test_minimal_run_writes_research_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(load_experiment_config(), terrain_classes=("road", "urban_building"), conditions=("normal", "blur"), output_directory=directory)
            output = ExperimentRunner(cfg).run()
            expected = {"resolved_config.yaml", "frame_level_schema.csv", "frame_level_schema.json", "ontology_snapshot.ttl", "reasoning_trace.jsonl", "matching_results.csv", "localization_results.csv", "summary_metrics.csv", "ablation_metrics.csv", "terrain_statistics.csv", "run_log.txt"}
            self.assertTrue(expected <= {item.name for item in output.iterdir()})
            self.assertTrue((output / "figures" / "matchability_vs_inlier_ratio.png").is_file())
            self.assertIn("Ablation_Same_Features_Without_Ontology", (output / "ablation_metrics.csv").read_text())


if __name__ == "__main__": unittest.main()
