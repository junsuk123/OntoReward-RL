import math
import unittest

from simlab.algorithms.swarm import SwarmController
from simlab.config import load_config
from simlab.config.schema import DronesConfig


class SwarmControllerTest(unittest.TestCase):
    def test_camera_geometry_and_identity_config(self):
        cfg = load_config()
        self.assertEqual((12.0, -1.5, 4.0), cfg.cameras.front.near_position)
        self.assertEqual((-1.0, 0.0, 0.0), cfg.cameras.front.view_direction)
        self.assertEqual((0.0, 1.0, 0.0), cfg.cameras.front.baseline_direction)
        self.assertEqual(3.0, cfg.cameras.front.separation_m)
        self.assertEqual(90.0, cfg.cameras.front.image_rotation_clockwise_deg)
        self.assertEqual((0.0, 0.0, 20.0), cfg.cameras.satellite.position)
        self.assertEqual((2048, 2048), cfg.cameras.satellite.optics.resolution)
        self.assertEqual("procedural_urban", cfg.world.environment_type)
        self.assertIsNone(cfg.world.environment_usd)
        self.assertFalse(cfg.world.ground_plane)

    def test_default_roster_has_two_requested_teams(self):
        controller = SwarmController(load_config().drones)
        cfg = load_config().drones
        self.assertEqual(cfg.friendly.count, sum(s.team == "friendly" for s in controller.states))
        self.assertEqual(cfg.enemy.count, sum(s.team == "enemy" for s in controller.states))
        self.assertEqual({"iris"}, {s.model for s in controller.states if s.team == "friendly"})
        self.assertEqual(
            {"quadcopter"}, {s.model for s in controller.states if s.team == "enemy"}
        )
        self.assertEqual(1.5, load_config().drones.enemy.asset_scale)

    def test_sixty_second_patrol_stays_bounded_and_separated(self):
        cfg = load_config().drones
        controller = SwarmController(cfg)
        dt = 1.0 / 60.0
        minimum_separation = math.inf
        maximum_activity_offset = 0.0
        maximum_speed = 0.0
        for step in range(round(60.0 / dt)):
            states = controller.step(step * dt, dt)
            for index, state in enumerate(states):
                maximum_activity_offset = max(maximum_activity_offset, abs(state.position[1]))
                maximum_speed = max(maximum_speed, math.dist((0.0, 0.0, 0.0), state.velocity))
                for other in states[index + 1 :]:
                    minimum_separation = min(
                        minimum_separation, math.dist(state.position, other.position)
                    )
        self.assertLessEqual(maximum_activity_offset, cfg.activity_half_length)
        self.assertGreater(minimum_separation, cfg.safety_radius)
        self.assertLessEqual(maximum_speed, cfg.max_speed + 0.02)
        self.assertTrue(all(state.position[2] > 1.0 for state in states))

    def test_random_crossing_shuttle_is_normal_to_camera_axis(self):
        scene = load_config()
        cfg = scene.drones
        cfg.trajectory_seed = 1234
        controller = SwarmController(cfg)
        camera_axis = scene.cameras.front.view_direction
        activity_axis = (0.0, 1.0, 0.0)
        self.assertEqual(scene.cameras.front.baseline_direction, activity_axis)
        self.assertAlmostEqual(sum(a * b for a, b in zip(camera_axis, activity_axis)), 0.0)
        self.assertEqual(activity_axis[2], 0.0)

        crossings = [controller.crossing_parameters(leg) for leg in range(5)]
        self.assertGreater(len({round(y, 3) for y, _ in crossings}), 1)
        for leg, (crossing_y, fraction) in enumerate(crossings):
            t = cfg.takeoff_duration_s + (leg + fraction) * cfg.shuttle_leg_duration_s
            states = controller.step(t, 1.0 / 60.0)
            self.assertAlmostEqual(states[0].position[1], crossing_y, places=6)
            self.assertAlmostEqual(states[1].position[1], crossing_y, places=6)
            self.assertAlmostEqual(states[0].position[2], states[1].position[2], places=6)
            self.assertGreaterEqual(
                math.dist(states[0].position, states[1].position), cfg.safety_radius
            )


if __name__ == "__main__":
    unittest.main()
