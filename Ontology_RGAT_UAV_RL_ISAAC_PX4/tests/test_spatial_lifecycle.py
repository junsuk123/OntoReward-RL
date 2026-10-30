from types import SimpleNamespace
import json
import pytest

from ontology_rgat.bridge import (ArmingRefused, CleanupStopUnconfirmed,
                                  PX4Failsafe)
from ontology_rgat.spatial.lifecycle import (recover_failed_cleanup,
                                             recover_interrupted_episode,
                                             recover_refused_reset)


def test_terminal_and_failed_cleanup_remain_distinct_in_durable_audit(tmp_path, monkeypatch):
    from ontology_rgat.spatial.lifecycle import record_flight_event
    stack = SimpleNamespace(log_dir=tmp_path/'stack')
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    record_flight_event(dict(phase='terminal_before_cleanup',seed=2001,status='SAFE_ABORT'))
    record_flight_event(dict(phase='cleanup_finished',seed=2001,stop_confirmed=False))
    rows=[json.loads(line) for line in (tmp_path/'flight_terminal.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[0]['status']=='SAFE_ABORT' and not rows[1]['stop_confirmed']
    assert all('observed_at_utc' in row for row in rows)


@pytest.mark.parametrize('adopted,budget', [(True,2),(False,0)])
def test_reset_recovery_never_takes_over_an_adopted_or_unapproved_stack(monkeypatch, adopted, budget):
    stack = SimpleNamespace(adopted_simulator=adopted, spatial_recovery_budget=budget)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    assert not recover_refused_reset(ArmingRefused('refused'),seed=42,
                                    release=lambda:pytest.fail('must not release'))


def test_reset_recovery_budget_is_run_wide_and_failed_attempt_is_audited(tmp_path, monkeypatch):
    calls = []
    stack = SimpleNamespace(adopted_simulator=False, spatial_recovery_budget=1,
        log_dir=tmp_path/'stack', restart=lambda:calls.append('restart') or True)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    error = ArmingRefused('preflight accelerometer bias')
    assert recover_refused_reset(error, seed=123, release=lambda:calls.append('release'))
    assert not recover_refused_reset(error, seed=124, release=lambda:pytest.fail('budget'))
    assert calls == ['release','restart']
    rows=[json.loads(x) for x in (tmp_path/'reset_recovery.jsonl').read_text().splitlines()]
    assert [r['state'] for r in rows] == ['restarting','restarted']
    assert all(r['seed']==123 and r['transitions_collected']==0 and
               r['counted_as_rl_episode'] is False for r in rows)


def test_reset_recovery_does_not_retry_policy_time_failures():
    with pytest.raises(TypeError):
        recover_refused_reset(RuntimeError('policy disarmed'), seed=42, release=lambda:None)


def test_recoverable_policy_time_failsafe_restarts_same_seed_and_audits_discard(
        tmp_path, monkeypatch):
    calls = []
    stack = SimpleNamespace(adopted_simulator=False, spatial_recovery_budget=2,
        spatial_recoveries_used=0, log_dir=tmp_path/'stack',
        restart=lambda:calls.append('restart') or True)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    error = PX4Failsafe(['gcs_connection_lost'], recoverable=True)
    assert recover_interrupted_episode(
        error, seed=100088, transitions_collected=17)
    assert calls == ['restart'] and stack.spatial_recoveries_used == 1
    rows = [json.loads(x) for x in
            (tmp_path/'episode_recovery.jsonl').read_text().splitlines()]
    assert [row['state'] for row in rows] == ['restarting', 'restarted']
    assert all(row['seed'] == 100088 and row['transitions_discarded'] == 17
               and row['counted_as_rl_episode'] is False for row in rows)


def test_hard_policy_time_failsafe_is_never_recovered(tmp_path, monkeypatch):
    stack = SimpleNamespace(adopted_simulator=False, spatial_recovery_budget=2,
        spatial_recoveries_used=0, log_dir=tmp_path/'stack',
        restart=lambda:pytest.fail('must not restart'))
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    assert not recover_interrupted_episode(
        PX4Failsafe(['battery_warning_2'], recoverable=False),
        seed=100088, transitions_collected=2)


def test_failed_cleanup_restart_preserves_completed_episode(tmp_path, monkeypatch):
    calls = []
    stack = SimpleNamespace(adopted_simulator=False, spatial_recovery_budget=2,
        spatial_recoveries_used=0, log_dir=tmp_path/'stack',
        restart=lambda:calls.append('restart') or True)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    error = CleanupStopUnconfirmed('cleanup timeout')
    assert recover_failed_cleanup(
        error, seed=100031, transitions_collected=670)
    rows = [json.loads(x) for x in
            (tmp_path/'cleanup_recovery.jsonl').read_text().splitlines()]
    assert calls == ['restart']
    assert [row['state'] for row in rows] == ['restarting', 'restarted']
    assert all(row['counted_as_rl_episode'] is True
               and row['transitions_discarded'] == 0
               and row['transitions_collected'] == 670 for row in rows)


def test_failed_cleanup_is_attempted_once_even_when_backend_closes(monkeypatch):
    from ontology_rgat.spatial.environment import IsaacBackend
    calls = []
    backend = IsaacBackend.__new__(IsaacBackend)
    backend.seed = 100031
    backend.finished = False
    backend.cleanup_attempted = False
    backend.bridge = SimpleNamespace(
        stop_after_outcome=lambda timeout:calls.append(('stop', timeout)) or False,
        close=lambda:calls.append('close'))
    assert backend.finish() is False
    assert backend.finish() is False
    with pytest.raises(CleanupStopUnconfirmed, match='not confirmed'):
        backend.close()
    assert calls == [('stop', 180.0), 'close']


def test_new_stack_context_cannot_reset_run_wide_recovery_allowance(tmp_path,monkeypatch):
    from ontology_rgat.spatial.lifecycle import used_reset_recoveries
    calls=[]
    def fresh():
        return SimpleNamespace(adopted_simulator=False,spatial_recovery_budget=1,
            spatial_recoveries_used=used_reset_recoveries(tmp_path),log_dir=tmp_path/'stack',
            restart=lambda:calls.append('restart') or True)
    assert used_reset_recoveries(tmp_path)==0
    first=fresh()
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current',lambda:first)
    assert recover_refused_reset(ArmingRefused('before training'),seed=1,release=lambda:None)
    second=fresh()
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current',lambda:second)
    assert second.spatial_recoveries_used==1
    assert not recover_refused_reset(ArmingRefused('before test'),seed=2,
                                    release=lambda:pytest.fail('must not restart'))
    assert calls==['restart']


def test_recovery_usage_counts_starts_once_and_fails_closed_on_corruption(tmp_path):
    from ontology_rgat.spatial.lifecycle import used_reset_recoveries
    path=tmp_path/'reset_recovery.jsonl'
    path.write_text('\n'.join(json.dumps(dict(attempt=1,state=state)) for state in
                             ['restarting','restarted','restarting','failed'])+'\n')
    assert used_reset_recoveries(tmp_path)==2
    path.write_text('{"attempt":')
    with pytest.raises(ValueError,match='journal'):
        used_reset_recoveries(tmp_path)


def test_pre_policy_entry_contact_can_consume_owned_recovery(tmp_path, monkeypatch):
    from ontology_rgat.bridge import EntryResetError
    stack = SimpleNamespace(adopted_simulator=False, spatial_recovery_budget=1,
        log_dir=tmp_path/'stack', restart=lambda:True)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current', lambda:stack)
    assert recover_refused_reset(EntryResetError('contact during entry'),
                                seed=4, release=lambda:None)


def test_final_handover_refusal_is_typed_only_before_any_policy_transition(monkeypatch):
    from ontology_rgat.bridge import EntryResetError
    from ontology_rgat.spatial.environment import IsaacBackend
    backend=IsaacBackend.__new__(IsaacBackend);backend.seed=12
    events=[]
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.record_flight_event',events.append)
    state={'armed':False,'nav_state':4}
    with pytest.raises(EntryResetError,match='no|not an RL episode'):
        backend.assert_policy_flight(state,pre_policy=True)
    assert events[0]['transitions_collected']==0
    assert events[0]['phase']=='entry_refused_before_policy'
    with pytest.raises(RuntimeError) as failure:
        backend.assert_policy_flight(state)
    assert not isinstance(failure.value,EntryResetError)


def test_explicit_cold_episodes_require_confirmed_stop_and_preserve_every_outcome(tmp_path,monkeypatch):
    from ontology_rgat.spatial.lifecycle import prepare_isolated_episode,record_confirmed_stop
    calls=[]
    stack=SimpleNamespace(adopted_simulator=False,spatial_isolate_episodes=True,
        log_dir=tmp_path/'stack',restart=lambda:calls.append('restart') or True)
    monkeypatch.setattr('ontology_rgat.spatial.lifecycle.current',lambda:stack)
    assert not prepare_isolated_episode(seed=20,release=lambda:calls.append('close'))
    with pytest.raises(RuntimeError,match='confirmed'):
        prepare_isolated_episode(seed=21,release=lambda:pytest.fail('still active'))
    record_confirmed_stop(True)
    assert prepare_isolated_episode(seed=21,release=lambda:calls.append('close'))
    assert calls==['close','restart'] and stack.spatial_episode_count==2
    row=json.loads((tmp_path/'episode_isolation.jsonl').read_text())
    assert row['seed']==21 and row['transitions_discarded']==0
    assert row['reason']=='scheduled_episode_isolation'
    record_confirmed_stop(True)
    stack.adopted_simulator=True
    with pytest.raises(RuntimeError,match='adopted'):
        prepare_isolated_episode(seed=22,release=lambda:pytest.fail('not owned'))
