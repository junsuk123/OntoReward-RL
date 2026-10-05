"""DDS discovery may lag UDP readiness; identity mismatches stay fail-closed."""
from unittest.mock import Mock

import pytest

from ontology_rgat.spatial import runtime_contract as runtime


def packet(**fields):
    return {'extra': {'spatial_clock': fields}}


def ready():
    return packet(valid=True, profile_sha256='profile', source_sha256='source')


@pytest.fixture
def timer(monkeypatch):
    now = [0.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    return now


def wait(bridge, **kwargs):
    return runtime.wait_runtime_identity(bridge, profile_sha256='profile',
                                         source_sha256='source', **kwargs)


def test_missing_then_stale_clock_waits_without_reset_or_arm(timer):
    bridge = Mock()
    bridge.transact.side_effect = [{}, packet(valid=False, profile_sha256='profile',
                                           source_sha256='source'), ready()]
    result = wait(bridge)
    assert result['attempts'] == 3 and result['wall_seconds'] == pytest.approx(.1)
    assert result['read_only'] and not result['reset_or_arm_sent']
    assert bridge.method_calls == [
        ('transact', ('state', {}, ('state',)), {'timeout': 8.})]*3


@pytest.mark.parametrize('key', ['profile_sha256', 'source_sha256'])
@pytest.mark.parametrize('valid', [False, True])
def test_reported_mismatch_never_waits_or_retries(timer, key, valid):
    state = ready()
    state['extra']['spatial_clock'].update({key: 'different', 'valid': valid})
    bridge = Mock()
    bridge.transact.return_value = state
    with pytest.raises(ValueError, match='stale'):
        wait(bridge)
    assert bridge.transact.call_count == 1 and timer[0] == 0


def test_missing_identity_is_bounded_and_never_arms(timer):
    bridge = Mock()
    bridge.transact.return_value = {}
    with pytest.raises(ValueError, match='unavailable.*no reset/arm'):
        wait(bridge, timeout=.12)
    assert timer[0] == pytest.approx(.12)
    assert all(call[0] == 'transact' for call in bridge.method_calls)
    assert all(0 < call.kwargs['timeout'] <= .12 for call in bridge.transact.call_args_list)


def test_late_matching_reply_does_not_extend_deadline(timer):
    bridge = Mock()
    def late(*args, **kwargs):
        timer[0] = 1.
        return ready()
    bridge.transact.side_effect = late
    with pytest.raises(ValueError, match='unavailable'):
        wait(bridge, timeout=.1)
    assert bridge.transact.call_count == 1


@pytest.mark.parametrize('timeout', [0., -1., 31., float('inf'), float('nan')])
def test_invalid_budget_is_rejected_before_query(timeout):
    bridge = Mock()
    with pytest.raises(ValueError, match='timeout'):
        wait(bridge, timeout=timeout)
    bridge.transact.assert_not_called()


def test_malformed_identity_is_rejected(timer):
    bridge = Mock()
    bridge.transact.return_value = {'extra': {'spatial_clock': ['invalid']}}
    with pytest.raises(ValueError, match='malformed'):
        wait(bridge)
