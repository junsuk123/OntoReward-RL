"""Explicit, bounded recovery of pre-policy entry failures on owned SITL."""
from datetime import datetime, timezone
import json
import time

from ..stack import current


def used_reset_recoveries(output):
    """Preserve the run-wide allowance across train/evaluate stack contexts.

Each restart has a start and a final event; count starts, not JSONL lines.
An interrupted journal fails closed instead of silently granting new retries.
"""
    path = output/'reset_recovery.jsonl'
    if not path.exists():
        return 0
    starts, highest = 0, 0
    try:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            attempt = row['attempt']
            if (type(attempt) is not int or attempt <= 0 or
                    row['state'] not in ('restarting','restarted','failed')):
                raise ValueError('invalid recovery event')
            starts += row['state'] == 'restarting'
            highest = max(highest,attempt)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('reset recovery journal is incomplete or invalid') from exc
    # Count old repeated attempt IDs conservatively as separate consumed starts.
    return max(starts,highest)


def record_flight_event(event):
    """Durable evaluator/cleanup audit, including failed terminal cleanup."""
    stack = current()
    if stack is None or getattr(stack, 'log_dir', None) is None:
        return
    payload = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), **event)
    with (stack.log_dir.parent/'flight_terminal.jsonl').open('a') as handle:
        handle.write(json.dumps(payload, allow_nan=False)+'\n')


def prepare_isolated_episode(*, seed, release):
    """Opt-in cold PX4/EKF episode boundary after confirmed prior cleanup.

    Scheduled isolation is not failure recovery and never discards a rollout.
    It is forbidden on adopted stacks or while any prior flight is unconfirmed.
    """
    stack=current()
    if stack is None or not getattr(stack,'spatial_isolate_episodes',False):
        return False
    if stack.adopted_simulator:
        raise RuntimeError('episode isolation cannot restart an adopted simulator')
    count=getattr(stack,'spatial_episode_count',0)
    if not getattr(stack,'spatial_last_stop_confirmed',True):
        raise RuntimeError('previous flight stop must be confirmed before episode isolation')
    restarted=False
    if count:
        event=dict(seed=int(seed),previous_episodes=count,reason='scheduled_episode_isolation',
                   transitions_discarded=0,started_utc=datetime.now(timezone.utc).isoformat())
        started=time.monotonic()
        try:
            release()
            if not stack.restart():
                raise RuntimeError('owned episode isolation did not replace simulator')
            restarted=True
            event['state']='restarted'
        except BaseException as exc:
            event.update(state='failed',error=repr(exc))
            raise
        finally:
            event['wall_seconds']=time.monotonic()-started
            with (stack.log_dir.parent/'episode_isolation.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
    stack.spatial_episode_count=count+1
    stack.spatial_last_stop_confirmed=False
    return restarted


def record_confirmed_stop(confirmed):
    stack=current()
    if stack is not None and getattr(stack,'spatial_isolate_episodes',False):
        stack.spatial_last_stop_confirmed=bool(confirmed)


def recover_refused_reset(error, *, seed, release):
    """No transitions exist yet; retry the SAME seed, never relabel failure.

    The caller must catch EntryResetError specifically. Adopted simulators are
    never replaced, and the opt-in budget is shared across the entire run.
    """
    from ..bridge import EntryResetError
    if not isinstance(error, EntryResetError):
        raise TypeError("recovery is restricted to pre-policy entry failure")
    stack = current()
    if stack is None or stack.adopted_simulator:
        return False
    used = getattr(stack, 'spatial_recoveries_used', 0)
    if used >= getattr(stack, 'spatial_recovery_budget', 0):
        return False
    stack.spatial_recoveries_used = used+1
    event = dict(seed=int(seed), attempt=used+1, error_type=type(error).__name__,
        error=str(error), transitions_collected=0, counted_as_rl_episode=False,
        started_utc=datetime.now(timezone.utc).isoformat(), state='restarting')
    path = stack.log_dir.parent/'reset_recovery.jsonl'
    def persist():
        with path.open('a') as handle:
            handle.write(json.dumps(event)+'\n')
    persist()
    started = time.monotonic()
    print(f"[spatial reset recovery] seed={seed}, owned restart {used+1}/"
          f"{stack.spatial_recovery_budget}; no RL transition collected", flush=True)
    try:
        release()
        replaced = stack.restart()
        if not replaced:
            raise RuntimeError('owned spatial reset recovery did not replace simulator')
        event['state'] = 'restarted'
        return True
    except BaseException as exc:
        event.update(state='failed', recovery_error=repr(exc))
        raise
    finally:
        event['wall_seconds'] = time.monotonic()-started
        persist()
