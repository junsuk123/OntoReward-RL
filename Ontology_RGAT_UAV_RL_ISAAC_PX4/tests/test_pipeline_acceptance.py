"""Regressions found by actual CLI and UDP integration runs."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import copy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("warmup", [0, 2])
def test_explicit_value_warmup_is_common_and_recorded(tmp_path, monkeypatch, warmup):
    import json
    import run_spatial_pipeline as pipeline
    jobs=[]
    monkeypatch.setattr(pipeline, "train_arm", lambda **job:jobs.append(job))
    monkeypatch.setattr(sys, "argv", ["pipeline", "--stage", "train", "--output", str(tmp_path),
        "--iterations", "3", "--value-warmup-iterations", str(warmup)])
    assert pipeline.main() == 0
    assert len(jobs)==3 and all(job["hyper"].value_warmup_iterations==warmup for job in jobs)
    assert json.loads((tmp_path/"plan.json").read_text())["value_warmup_iterations_override"]==warmup


def test_read_only_status_recognizes_spatial_pipeline_without_shell_false_positives():
    spec = importlib.util.spec_from_file_location('runtime_audit', ROOT/'tools/runtime_audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.relevant_programs(['python', '-u', '/repo/python/run_spatial_pipeline.py']) == ['run_spatial_pipeline.py']
    assert audit.relevant_programs(['python', '/repo/isaac_sim/landing_world.py']) == ['landing_world.py']
    assert audit.relevant_programs(['bash', '-lc', 'python /repo/isaac_sim/landing_world.py']) == []
    assert audit.relevant_programs(['rg', 'run_spatial_pipeline.py']) == []
    assert audit.relevant_programs([]) == []


def test_actual_test_range_can_be_fresh_without_changing_paired_validation():
    from run_spatial_pipeline import isaac_test_seeds
    assert isaac_test_seeds(12000, 2) == (12000, 12001)
    assert isaac_test_seeds(13000, 2) == (13000, 13001)
    for start, count in [(2000, 2), (-1, 2), (13000, 0), (13000, 1001), (2**31-1, 2)]:
        with pytest.raises(ValueError):
            isaac_test_seeds(start, count)


def test_actual_acceptance_requires_proposed_landing_not_only_baseline_success():
    from ontology_rgat.spatial.validation import actual_acceptance
    from ontology_rgat.two_axis.models import POLICY_MODES
    rows=[]
    for mode in POLICY_MODES:
        rows.append(dict(mode=mode,seed=1,backend='isaac',signature={'schema':'same'},
            metrics=dict(episodes=1,landing_rate=1.,safe_abort_rate=0.,
                         task_timeout_rate=0.,unsafe_rate=0.,
                         mean_abs_relation_residual=[.001,0.,0.],
                         rows=[{'seed':12000,'status':'SUCCESS','stop_confirmed':True}])))
    assert actual_acceptance(rows,[1])['passes']
    assert not actual_acceptance(rows[:2],[1])['passes']
    broken=copy.deepcopy(rows)
    broken[-1]['metrics'].update(landing_rate=0.,task_timeout_rate=1.)
    broken[-1]['metrics']['rows'][0]['status']='TASK_TIMEOUT'
    assert not actual_acceptance(broken,[1])['passes']
    for mutation in ('unpaired','zero_relation','false_rate','duplicate','local','signature',
                     'unconfirmed_stop'):
        broken=copy.deepcopy(rows)
        if mutation=='unpaired': broken[-1]['metrics']['rows'][0]['seed']=12001
        elif mutation=='zero_relation': broken[-1]['metrics']['mean_abs_relation_residual']=[0.,0.,0.]
        elif mutation=='false_rate': broken[-1]['metrics']['rows'][0]['status']='UNSAFE_CONTACT'
        elif mutation=='duplicate': broken[-1]['mode']=broken[0]['mode']
        elif mutation=='local': broken[-1]['backend']='local'
        elif mutation=='unconfirmed_stop': broken[-1]['metrics']['rows'][0].pop('stop_confirmed')
        else: broken[-1]['signature']['schema']='different'
        assert not actual_acceptance(broken,[1])['passes'],mutation
    assert not actual_acceptance([],[])['passes']


def test_safe_abort_label_requires_separate_terminal_hold_evidence():
    from ontology_rgat.spatial.validation import actual_acceptance
    from ontology_rgat.two_axis.models import POLICY_MODES
    rows = [dict(mode=mode,seed=1,backend='isaac',signature={'schema':'same'},
                 metrics=dict(episodes=2,landing_rate=.5,safe_abort_rate=.5,
                              task_timeout_rate=0.,unsafe_rate=0.,
                              mean_abs_relation_residual=[.001,0.,0.],
                              rows=[dict(seed=12000,status='SUCCESS',stop_confirmed=True),
                                    dict(seed=12001,status='SAFE_ABORT',stop_confirmed=True)]))
            for mode in POLICY_MODES]
    result = actual_acceptance(rows,[1])
    assert result['no_unsafe_outcomes'] and not result['passes']
    assert not result['all_abort_terminal_holds_verified']
    for row in rows:
        row['metrics']['rows'][1]['terminal_hold_audit']={'hold_verified':True}
    assert actual_acceptance(rows,[1])['passes']


def test_actual_infrastructure_error_cannot_leave_old_acceptance(tmp_path):
    import json
    from run_spatial_pipeline import incomplete_isaac_report
    target = tmp_path / "isaac_acceptance.json"
    target.write_text('{"passes": true}')
    with pytest.raises(RuntimeError, match="DDS"):
        with incomplete_isaac_report(tmp_path):
            raise RuntimeError("DDS unavailable")
    status = json.loads(target.read_text())
    assert not status["passes"] and not status["complete_matrix"]
    assert status["state"] == "incomplete"
    assert status["no_unsafe_outcomes"] is None


def test_checkpoint_wait_never_promotes_an_incomplete_progress_file(tmp_path):
    from run_spatial_pipeline import wait_checkpoint_summary
    (tmp_path / "progress.json").write_text('{"completed_nominal_episodes":100}')
    with pytest.raises(RuntimeError, match="completed training summary"):
        wait_checkpoint_summary(tmp_path)
    (tmp_path / "summary.json").write_text('{"selected_checkpoint":"checkpoint_best.pt"}')
    assert wait_checkpoint_summary(tmp_path)["selected_checkpoint"] == "checkpoint_best.pt"


def test_nadir_camera_keeps_two_horizontal_entry_axes():
    from ontology_rgat.initialization import camera_centered_hover_offset, constrain_camera_visible_entry
    np.testing.assert_allclose(camera_centered_hover_offset(3., 90.), [0, 0, 3.], atol=1e-12)
    target = np.array([.3, -.2, 3.])
    np.testing.assert_allclose(constrain_camera_visible_entry(
        target, 0., pitch_down_deg=90., image_size=(640,480)), target)


def test_fake_goto_satisfies_physical_entry_without_erasing_sensor_bias():
    spec = importlib.util.spec_from_file_location('fake_gateway', ROOT/'tools/fake_gateway.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from ontology_rgat.bridge import PX4Bridge
    fake = module.Fake()
    target = np.array([.5, -.2, 4.])
    fake.goto(target, 'pad')
    state = fake.state(1, 1)
    position, speed = PX4Bridge.entry_state(state)
    np.testing.assert_allclose(position, target)
    assert speed == 0
    assert np.linalg.norm(np.asarray(state['position'])-position) > 1.


def test_tuning_ties_and_common_selection_are_worker_order_independent(tmp_path, monkeypatch):
    import run_two_axis_pipeline as pipeline
    from ontology_rgat.two_axis.training import PPOHyperparameters
    from concurrent.futures import Future
    class Pool:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def submit(self, _, job):
            f = Future()
            f.set_result(dict(arm=job['arm'], trial=job['trial'], overrides=job['overrides'],
                             score=2., probe={'landing_rate':0.}, final_difficulty=1.))
            return f
    monkeypatch.setattr(pipeline, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(pipeline, 'as_completed', lambda futures: reversed(list(futures)))
    result = pipeline.stage_tune(tmp_path, PPOHyperparameters(iterations=1),
                                 workers=3, threads=1, seed=11)
    assert result['best_common']['trial'] == 0
    assert all(row['trial']==0 for row in result['best_per_arm'].values())


def test_cli_returns_failure_when_aggregate_fairness_fails(tmp_path, monkeypatch):
    import run_two_axis_pipeline as pipeline
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--stage', 'aggregate', '--output', str(tmp_path)])
    monkeypatch.setattr(pipeline, 'stage_aggregate',
                        lambda _: {'fairness': {'passes':False}, 'arms':{}})
    assert pipeline.main() == 2


@pytest.mark.parametrize("fault", ["estimator", "failsafe"])
def test_landing_cleanup_survives_transient_invalid_estimator(monkeypatch, fault):
    from ontology_rgat.bridge import PX4Bridge, PX4EstimatorInvalid, PX4Failsafe
    bridge = PX4Bridge.__new__(PX4Bridge)
    bridge.last_state = {}
    calls = []
    bridge.disable_offboard = lambda: calls.append("disable")
    bridge.disarm = lambda: calls.append("land")
    states = iter([None, {"landed": True, "armed": False}])
    def state():
        value = next(states)
        if value is None:
            if fault == "failsafe":
                raise PX4Failsafe(["offboard_control_signal_lost"], recoverable=True)
            raise PX4EstimatorInvalid("temporary DDS reconnect")
        return value
    bridge.get_state = state
    monkeypatch.setattr("ontology_rgat.bridge.time.sleep",lambda _: None)
    assert bridge.stop_after_outcome(timeout=.1)
    assert calls == ["disable", "land"]


def test_cleanup_disarms_only_after_fresh_ground_landing(monkeypatch):
    from ontology_rgat.bridge import PX4Bridge
    bridge = PX4Bridge.__new__(PX4Bridge)
    bridge.last_state = {}
    calls = []
    bridge.disable_offboard = lambda: calls.append("disable")
    bridge.disarm = lambda: calls.append("disarm_or_land")
    states = iter([{"landed": False, "armed": True},
                   {"landed": True, "armed": True},
                   {"landed": True, "armed": False}])
    def state():
        value = next(states)
        calls.append((value['landed'], value['armed']))
        return value
    bridge.get_state = state
    monkeypatch.setattr("ontology_rgat.bridge.time.sleep", lambda _: None)
    assert bridge.stop_after_outcome(timeout=1.)
    assert calls == ["disable", "disarm_or_land", (False, True),
                     (True, True), "disarm_or_land", (True, False)]
