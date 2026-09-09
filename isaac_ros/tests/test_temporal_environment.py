import csv
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from simlab.temporal.batch import TemporalBatchRunner
from simlab.temporal.clock import EnvironmentTimeManager
from simlab.temporal.config import load_temporal_config
from simlab.temporal.renderer import TemporalSceneRenderer
from simlab.temporal.state import EnvironmentStateFactory,TemporalEventScheduler
from simlab.temporal.trajectory import FixedWingTrajectoryManager


class TemporalClockTest(unittest.TestCase):
    def test_environment_clock_is_independent_and_controllable(self):
        clock=EnvironmentTimeManager(365,10);clock.update(2);self.assertEqual(0,clock.state.virtual_year)
        clock.play();clock.update(2);self.assertEqual(2,clock.state.virtual_year);clock.pause();clock.update(4);self.assertEqual(2,clock.state.virtual_year)
        clock.jump_to_epoch(5);self.assertEqual(5,clock.state.virtual_year);clock.reset();self.assertEqual(0,clock.state.virtual_year)


class TemporalStateTest(unittest.TestCase):
    def setUp(self):
        self.cfg=load_temporal_config();self.scheduler=TemporalEventScheduler(self.cfg.events,self.cfg.random_seed);self.factory=EnvironmentStateFactory(self.cfg,self.scheduler)
    def test_same_seed_reproduces_events_and_images(self):
        first=TemporalEventScheduler(self.cfg.events,self.cfg.random_seed).replay_record();second=TemporalEventScheduler(self.cfg.events,self.cfg.random_seed).replay_record();self.assertEqual(first,second)
        state=self.factory.create(self.cfg.epochs[2]);a=TemporalSceneRenderer(self.cfg).render_world(state);b=TemporalSceneRenderer(self.cfg).render_world(state);self.assertTrue(np.array_equal(a,b))
    def test_reset_state_and_camera_trajectory_are_exact(self):
        year0=self.factory.create(self.cfg.epochs[0]);again=self.factory.create(self.cfg.epochs[0]);self.assertEqual(year0,again)
        poses=TemporalSceneRenderer(self.cfg).camera_poses();self.assertEqual(self.cfg.camera.spiral_samples,len(poses))
        self.assertEqual(len(poses),len(FixedWingTrajectoryManager(self.cfg.camera).poses()))

    def test_fixed_wing_spiral_expands_and_stays_inside_large_map(self):
        poses=FixedWingTrajectoryManager(self.cfg.camera).poses();cx,cy=self.cfg.camera.spiral_center_xy_m;radii=[np.hypot(p.x_m-cx,p.y_m-cy) for p in poses]
        self.assertTrue(all(a<b for a,b in zip(radii,radii[1:])));self.assertAlmostEqual(self.cfg.camera.spiral_start_radius_m,radii[0]);self.assertAlmostEqual(self.cfg.camera.spiral_end_radius_m,radii[-1]);self.assertLess(radii[-1],self.cfg.map.extent_m/2)
        self.assertGreater(len({round(p.yaw_deg,1) for p in poses}),10)
    def test_semantic_cases_have_similar_global_but_different_structural_change(self):
        appearance=self.factory.create(next(e for e in self.cfg.epochs if e.name=="CaseAppearance"));structural=self.factory.create(next(e for e in self.cfg.epochs if e.name=="CaseStructural"))
        self.assertLess(abs(appearance.environment_change_score-structural.environment_change_score),.03);self.assertGreater(structural.structural_change_score-appearance.structural_change_score,.8)
    def test_camera_output_changes_across_epochs(self):
        renderer=TemporalSceneRenderer(self.cfg);a=renderer.render_world(self.factory.create(self.cfg.epochs[0]));b=renderer.render_world(self.factory.create(self.cfg.epochs[3]));self.assertGreater(float(np.mean(np.abs(a.astype(float)-b.astype(float)))),20)


class TemporalBatchTest(unittest.TestCase):
    def test_end_to_end_outputs_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=replace(load_temporal_config(),output_directory=directory);out=TemporalBatchRunner(cfg).run()
            expected={"resolved_config.yaml","environment_timeline.csv","environment_events.jsonl","environment_change_ground_truth.csv","camera_poses.csv","matching_results.csv","summary_metrics.csv","reference_images","query_images","figures","epoch_usd_layers"}
            self.assertTrue(expected<={p.name for p in out.iterdir()})
            self.assertTrue((out/"reference_images"/"spiral_flight.mp4").is_file())
            with (out/"camera_poses.csv").open() as stream:rows=list(csv.DictReader(stream))
            by_epoch={}
            for row in rows:by_epoch.setdefault(row["epoch"],[]).append(tuple(row[k] for k in ("x_m","y_m","z_m","yaw_deg")))
            self.assertEqual(1,len({tuple(v) for v in by_epoch.values()}))

    def test_isaac_adapter_imports_no_omni_module_before_launch(self):
        import inspect
        from simlab.temporal import isaac_capture
        prefix=inspect.getsource(isaac_capture).split("class IsaacTemporalScene",1)[0]
        self.assertNotIn("import omni",prefix);self.assertNotIn("from pxr",prefix)


if __name__=="__main__":unittest.main()
