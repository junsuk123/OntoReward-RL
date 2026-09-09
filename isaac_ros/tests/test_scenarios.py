"""The three scenarios: isolation, safety, determinism, and their ground truth."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from simlab.algorithms.swarm import SwarmController
from simlab.config import load_config
from simlab.config.schema import SCENARIO_KEYS, SceneConfig
from simlab.ml.lineage import (
    pending_raw_sessions,
    record_trained_sessions,
    untrained_sessions,
)
from simlab.scenarios.environments import ENVIRONMENT_KEYS, environment, environment_boxes
from simlab.scenarios.episode import build_plan, write_episode_config
from simlab.scenarios.motion import lane_plans, speed_scale_bound
from simlab.scenarios.occlusion import (
    Box,
    lane_occlusion_windows,
    occluder_boxes,
    segment_hits_box,
    structure_visibility,
)
from simlab.scenarios.projection import SceneProjector, classify
from simlab.scenarios.scene import camera_positions, drone_roster, scene_boxes
from simlab.scenarios.timeline import ScenarioTimeline

FLEETS = ((1, 1), (3, 2), (4, 4))


def episode_config(scenario, environment_key="urban_day", seed=4242, allies=2, enemies=2):
    scene = load_config()
    scene.scenarios.enabled = True
    scene.scenarios.active = scenario
    scene.scenarios.seed = seed
    scene.world.environment = environment_key
    scene.world.environment_seed = seed
    scene.drones.friendly.count = allies
    scene.drones.enemy.count = enemies
    return scene


class FlightSafetyTest(unittest.TestCase):
    """Aircraft may share a pixel; they may never share a position."""

    def test_every_scenario_and_fleet_keeps_its_separation(self):
        for scenario in SCENARIO_KEYS:
            for allies, enemies in FLEETS:
                scene = episode_config(scenario, allies=allies, enemies=enemies)
                cfg = scene.drones
                controller = SwarmController(cfg, scene.scenarios)
                dt = 1.0 / 60.0
                separation, speed, offset = math.inf, 0.0, 0.0
                for step in range(round(90.0 / dt)):
                    states = controller.step(step * dt, dt)
                    for index, state in enumerate(states):
                        offset = max(offset, abs(state.position[1]))
                        speed = max(speed, math.dist((0.0, 0.0, 0.0), state.velocity))
                        for other in states[index + 1:]:
                            separation = min(
                                separation, math.dist(state.position, other.position)
                            )
                label = f"{scenario} {allies}v{enemies}"
                self.assertGreater(separation, cfg.safety_radius, label)
                self.assertLessEqual(speed, cfg.max_speed + 0.02, label)
                self.assertLessEqual(offset, cfg.activity_half_length + 1e-6, label)

    def test_lanes_are_at_least_a_safety_radius_apart(self):
        scene = episode_config("mutual_occlusion", allies=4, enemies=4)
        plans = lane_plans(scene.drones, scene.scenarios, seed=1)
        lanes = sorted(plan.lane_x for plan in plans)
        gaps = [second - first for first, second in zip(lanes, lanes[1:])]
        self.assertTrue(gaps)
        self.assertGreaterEqual(min(gaps), scene.drones.safety_radius)

    def test_speed_bound_respects_the_configured_maximum(self):
        scene = load_config()
        bound = speed_scale_bound(scene.drones)
        reach = scene.drones.activity_half_length + scene.drones.crossing_offset_limit
        share = 0.5 - scene.drones.crossing_time_jitter
        fastest = bound * reach / (share * scene.drones.shuttle_leg_duration_s)
        self.assertLessEqual(fastest, scene.drones.max_speed + 1e-9)

    def test_no_scenario_reproduces_the_plain_shuttle(self):
        scene = load_config()
        plans = lane_plans(scene.drones, scene.scenarios, seed=99)
        depth = scene.drones.camera_depth_separation_m * 0.5
        self.assertEqual([-depth, depth], [plan.lane_x for plan in plans])
        self.assertEqual({0}, {plan.crossing_group for plan in plans})
        self.assertEqual(
            {scene.drones.shuttle_leg_duration_s}, {plan.leg_duration_s for plan in plans}
        )


class MutualOcclusionTest(unittest.TestCase):
    def test_paired_aircraft_share_a_crossing_point(self):
        scene = episode_config("mutual_occlusion", allies=2, enemies=2)
        controller = SwarmController(scene.drones, scene.scenarios)
        groups = {}
        for state in controller.states:
            groups.setdefault(state.plan.crossing_group, []).append(state)
        pair = groups[0]
        self.assertEqual({"friendly", "enemy"}, {state.team for state in pair})
        leg = 0
        crossing_y, fraction = controller.crossing_parameters(leg, 0)
        t = scene.drones.takeoff_duration_s + (leg + fraction) * pair[0].plan.leg_duration_s
        controller.step(t, 1.0 / 60.0)
        for state in pair:
            self.assertAlmostEqual(state.position[1], crossing_y, places=6)

    def test_a_nearer_aircraft_hides_the_one_behind_it(self):
        scene = episode_config("mutual_occlusion")
        projector = SceneProjector(scene, ("front_near",))
        roster = list(drone_roster(scene))
        near, far = roster[2], roster[0]  # an enemy in front of an ally
        positions = {far: (-2.37, 1.0, 3.2), near: (0.9, 1.0, 3.2)}
        results = {p.drone_id: p for p in projector.observe("front_near", positions, 960, 540)}
        self.assertEqual(0, results[near].depth_rank)
        self.assertEqual(1, results[far].depth_rank)
        self.assertGreater(results[far].coverage, 0.5)
        self.assertEqual(near, results[far].covered_by)
        self.assertEqual("occluded_by_drone", classify(results[far], 0.35))
        self.assertEqual("visible", classify(results[near], 0.35))

    def test_separated_aircraft_are_both_visible(self):
        scene = episode_config("mutual_occlusion")
        projector = SceneProjector(scene, ("front_near",))
        roster = list(drone_roster(scene))
        positions = {roster[0]: (-0.9, -4.0, 3.2), roster[2]: (0.9, 4.0, 3.2)}
        results = projector.observe("front_near", positions, 960, 540)
        self.assertEqual(2, len(results))
        for projection in results:
            self.assertEqual("visible", classify(projection, 0.35))


class StructuralOcclusionTest(unittest.TestCase):
    def test_a_box_on_the_sight_line_hides_the_target(self):
        wall = Box("Wall", (6.0, 0.0, 3.0), (0.6, 4.0, 6.0), kind="occluder")
        camera = (12.0, 0.0, 4.0)
        self.assertTrue(segment_hits_box(camera, (0.0, 0.0, 3.2), wall))
        hidden = structure_visibility(camera, (0.0, 0.0, 3.2), 0.35, [wall])
        self.assertEqual(0.0, hidden.ratio)
        self.assertEqual("Wall", hidden.blocker)
        clear = structure_visibility(camera, (0.0, -8.0, 3.2), 0.35, [wall])
        self.assertEqual(1.0, clear.ratio)
        self.assertIsNone(clear.blocker)

    def test_flat_scenery_never_blocks_a_sight_line(self):
        road = Box("Road", (0.0, 0.0, 0.02), (80.0, 9.0, 0.04), kind="road")
        visible = structure_visibility((12.0, 0.0, 4.0), (0.0, 0.0, 3.2), 0.35, [road])
        self.assertEqual(1.0, visible.ratio)

    def test_every_occluder_set_hides_the_flight_lane(self):
        for key in ("urban_screens", "container_yard", "gantry_wall", "sparse_masts"):
            boxes = occluder_boxes(key, seed=11)
            windows = [
                (name, start, end)
                for name, start, end in lane_occlusion_windows(
                    (12.0, -1.5, 4.0), boxes, -0.9, 3.2
                )
                if end >= -5.5 and start <= 5.5
            ]
            self.assertTrue(windows, f"{key} hides nothing along the lane")

    def test_structural_episodes_are_the_only_ones_with_occluders(self):
        for scenario in SCENARIO_KEYS:
            scene = episode_config(scenario, environment_key="urban_day")
            occluders = [box for box in scene_boxes(scene) if box.kind == "occluder"]
            if scenario == "structural_occlusion":
                self.assertTrue(occluders, scenario)
            else:
                self.assertFalse(occluders, f"{scenario} must keep a clear sight line")

    def test_an_occluded_aircraft_is_classified_and_not_labelled(self):
        scene = episode_config("structural_occlusion")
        projector = SceneProjector(scene, ("front_near",))
        boxes = projector.boxes
        camera = camera_positions(scene)["front_near"]
        drone_id = list(drone_roster(scene))[0]
        hidden = None
        for name, start, end in lane_occlusion_windows(camera, boxes, -0.9, 3.2):
            middle = (start + end) * 0.5
            if -5.5 <= middle <= 5.5:
                hidden = (name, middle)
                break
        self.assertIsNotNone(hidden, "the urban_day loadout hides no part of the lane")
        blocker, y = hidden
        results = projector.observe("front_near", {drone_id: (-0.9, y, 3.2)}, 960, 540)
        self.assertEqual(1, len(results))
        self.assertEqual("occluded_by_structure", classify(results[0], 0.35))
        self.assertEqual(blocker, results[0].blocker)


class SensorDropoutTest(unittest.TestCase):
    def test_the_schedule_is_reproducible_from_the_config_alone(self):
        scene = episode_config("sensor_dropout")
        first = ScenarioTimeline.from_scene(scene)
        second = ScenarioTimeline.from_scene(load_config_like(scene))
        self.assertEqual(first.gaps, second.gaps)
        self.assertEqual(first.episode_id, second.episode_id)

    def test_gaps_are_bounded_and_never_overlap_on_one_camera(self):
        scene = episode_config("sensor_dropout")
        timeline = ScenarioTimeline.from_scene(scene)
        dropout = scene.scenarios.sensor_dropout
        self.assertEqual(dropout.dropout_count, len(timeline.gaps))
        for gap in timeline.gaps:
            self.assertIn(gap.camera, dropout.cameras)
            self.assertGreaterEqual(gap.start_s, scene.scenarios.event_start_s)
            self.assertLess(gap.end_s, scene.scenarios.episode_duration_s)
            self.assertGreaterEqual(gap.duration_s, dropout.min_dropout_s - 1e-9)
            self.assertLessEqual(gap.duration_s, dropout.max_dropout_s + 1e-9)
        for camera in dropout.cameras:
            windows = sorted(
                (gap.start_s, gap.end_s) for gap in timeline.gaps if gap.camera == camera
            )
            for (_, end), (start, _) in zip(windows, windows[1:]):
                self.assertLessEqual(end, start)

    def test_one_camera_keeps_watching_while_the_other_is_dark(self):
        scene = episode_config("sensor_dropout")
        scene.scenarios.sensor_dropout.simultaneous = False
        timeline = ScenarioTimeline.from_scene(scene)
        for gap in timeline.gaps:
            middle = (gap.start_s + gap.end_s) * 0.5
            others = [name for name in timeline.cameras if name != gap.camera]
            self.assertTrue(any(timeline.observing(name, middle) for name in others))
            self.assertFalse(timeline.observing(gap.camera, middle))
            self.assertAlmostEqual(
                timeline.gap_elapsed(gap.camera, middle), middle - gap.start_s, places=6
            )

    def test_the_other_scenarios_schedule_no_blackouts(self):
        for scenario in ("mutual_occlusion", "structural_occlusion"):
            timeline = ScenarioTimeline.from_scene(episode_config(scenario))
            self.assertEqual((), timeline.gaps)
            self.assertFalse(timeline.frame_dropped("front_near", 17))

    def test_frame_drops_are_deterministic_and_close_to_their_rate(self):
        scene = episode_config("sensor_dropout")
        scene.scenarios.sensor_dropout.frame_drop_probability = 0.1
        timeline = ScenarioTimeline.from_scene(scene)
        dropped = [step for step in range(4000) if timeline.frame_dropped("front_near", step)]
        repeat = [step for step in range(4000) if timeline.frame_dropped("front_near", step)]
        self.assertEqual(dropped, repeat)
        self.assertAlmostEqual(len(dropped) / 4000.0, 0.1, delta=0.02)


class EnvironmentTest(unittest.TestCase):
    def test_every_preset_builds_scenery_clear_of_the_flight_corridor(self):
        for key in ENVIRONMENT_KEYS:
            boxes = environment_boxes(environment(key), seed=3)
            self.assertTrue(boxes, key)
            for box in boxes:
                if box.kind != "building":
                    continue
                low, high = box.minimum, box.maximum
                clear_of_lane = high[1] < -6.0 or low[1] > 6.0 or high[0] < -6.0 or low[0] > 6.0
                self.assertTrue(clear_of_lane, f"{key}: {box.name} sits in the arena")

    def test_the_layout_follows_its_seed(self):
        first = environment_boxes(environment("urban_day"), seed=1)
        again = environment_boxes(environment("urban_day"), seed=1)
        other = environment_boxes(environment("urban_day"), seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_the_plan_covers_all_three_scenarios_in_several_environments(self):
        scene = load_config()
        episodes = scene.scenarios.episodes()
        self.assertEqual(set(SCENARIO_KEYS), {entry.scenario for entry in episodes})
        for scenario in SCENARIO_KEYS:
            environments = {e.environment for e in episodes if e.scenario == scenario}
            self.assertGreaterEqual(len(environments), 2, scenario)


class EpisodePlanTest(unittest.TestCase):
    def test_each_episode_gets_its_own_frozen_config(self):
        scene = load_config()
        plans = build_plan(scene, "20260101_000000", base_seed=5)
        self.assertEqual(len(scene.scenarios.episodes()), len(plans))
        self.assertEqual(len(plans), len({plan.episode_id for plan in plans}))
        self.assertEqual(len(plans), len({plan.seed for plan in plans}))
        with tempfile.TemporaryDirectory() as directory:
            for plan in plans[:3]:
                target = Path(directory) / plan.episode_id / "episode.yaml"
                object.__setattr__(plan, "config_path", target)
                write_episode_config(plan, "configs/default.yaml")
                written = load_config(target)
                self.assertTrue(written.scenarios.enabled)
                self.assertEqual(plan.scenario, written.scenarios.active)
                self.assertEqual(plan.seed, written.scenarios.seed)
                self.assertEqual(plan.environment, written.world.environment)
                self.assertEqual(plan.episode_id, written.scenarios.episode_id)
                self.assertTrue(written.app.headless)
                self.assertTrue(written.ros2.enabled)


class UnlearnedDataTest(unittest.TestCase):
    """Training is triggered by data the deployed weights have not seen."""

    def test_sessions_move_from_pending_to_ingested_to_trained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, dataset, lineage = root / "raw", root / "dataset", root / "lineage"
            for name in ("episode_a", "episode_b"):
                (raw / name).mkdir(parents=True)
                (raw / name / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
            (raw / "empty").mkdir()  # no manifest: not a collection session
            self.assertEqual(
                ["episode_a", "episode_b"],
                [path.name for path in pending_raw_sessions(raw, dataset)],
            )

            (dataset / "sessions").mkdir(parents=True)
            (dataset / "sessions" / "episode_a.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                ["episode_b"], [path.name for path in pending_raw_sessions(raw, dataset)]
            )
            self.assertEqual(["episode_a"], untrained_sessions(dataset, lineage))

            model = root / "best.pt"
            model.touch()
            record_trained_sessions(lineage, ["episode_a"], model)
            self.assertEqual([], untrained_sessions(dataset, lineage))
            (dataset / "sessions" / "episode_b.json").write_text("{}", encoding="utf-8")
            self.assertEqual(["episode_b"], untrained_sessions(dataset, lineage))
            payload = json.loads((lineage / "trained_sessions.json").read_text())
            self.assertEqual(["episode_a"], payload["sessions"])


def load_config_like(scene):
    """Rebuild a scene from another one's values, the way a second process would."""
    return SceneConfig.from_dict(
        {
            "world": {"environment": scene.world.environment},
            "scenarios": {
                "enabled": True,
                "active": scene.scenarios.active,
                "seed": scene.scenarios.seed,
                "episode_duration_s": scene.scenarios.episode_duration_s,
                "event_start_s": scene.scenarios.event_start_s,
                "sensor_dropout": {
                    "cameras": list(scene.scenarios.sensor_dropout.cameras),
                    "dropout_count": scene.scenarios.sensor_dropout.dropout_count,
                    "min_dropout_s": scene.scenarios.sensor_dropout.min_dropout_s,
                    "max_dropout_s": scene.scenarios.sensor_dropout.max_dropout_s,
                    "simultaneous": scene.scenarios.sensor_dropout.simultaneous,
                },
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
