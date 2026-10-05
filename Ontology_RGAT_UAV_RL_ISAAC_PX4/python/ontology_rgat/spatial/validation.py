"""Conservative actual-flight integration gate, not a performance-parity test."""
import math

from ..two_axis.models import POLICY_MODES


def actual_acceptance(rows, training_seeds):
    expected = {(mode, seed) for seed in training_seeds for mode in POLICY_MODES}
    keys = [(r['mode'], r['seed']) for r in rows]
    complete = bool(expected) and len(keys) == len(expected) and set(keys) == expected
    complete = complete and all(r['backend'] == 'isaac' for r in rows)
    signatures = [r['signature'] for r in rows]
    common_contract = bool(signatures) and all(s == signatures[0] for s in signatures)
    episode_sets = [tuple(e['seed'] for e in r['metrics']['rows']) for r in rows]
    paired = bool(episode_sets) and bool(episode_sets[0]) and all(
        ids == episode_sets[0] and len(ids) == len(set(ids)) for ids in episode_sets)
    valid_rates = True
    no_unsafe = bool(rows)
    confirmed_stops = bool(rows)
    verified_aborts = bool(rows)
    for row in rows:
        metrics = row['metrics']
        episodes = metrics['rows']
        count = len(episodes)
        if not count or metrics['episodes'] != count:
            valid_rates = False
            continue
        statuses = [e['status'] for e in episodes]
        confirmed_stops &= all(e.get('stop_confirmed') is True for e in episodes)
        verified_aborts &= all(
            e.get('terminal_hold_audit', {}).get('hold_verified') is True
            for e in episodes if e['status'] == 'SAFE_ABORT')
        for field, status in [('landing_rate','SUCCESS'),('safe_abort_rate','SAFE_ABORT'),
                              ('task_timeout_rate','TASK_TIMEOUT')]:
            valid_rates &= math.isclose(metrics[field],statuses.count(status)/count,
                                        abs_tol=1e-12)
        unsafe = sum(s not in ('SUCCESS','SAFE_ABORT','TASK_TIMEOUT') for s in statuses)
        valid_rates &= math.isclose(metrics['unsafe_rate'],unsafe/count,abs_tol=1e-12)
        no_unsafe &= unsafe == 0
    proposed = [r for r in rows if r['mode'] == 'ppo_ontology_rgat']
    proposed_landing = bool(proposed) and all(r['metrics']['landing_rate'] > 0 for r in proposed)
    relation_active = bool(proposed) and all(
        math.sqrt(sum(float(x)**2 for x in r['metrics']['mean_abs_relation_residual'])) > 1e-8
        for r in proposed)
    result = dict(
        complete_matrix=complete, common_contract=common_contract,
        paired_episode_seeds=paired, outcome_rates_consistent=valid_rates,
        no_unsafe_outcomes=no_unsafe,
        all_episode_stops_confirmed=confirmed_stops,
        all_abort_terminal_holds_verified=verified_aborts,
        landing_observed=any(r['metrics']['landing_rate'] > 0 for r in rows),
        proposed_landing_observed=proposed_landing, proposed_relation_active=relation_active,
        backend='actual-isaac-px4', performance_parity_claim=False, state='completed',
        acceptance_version='spatial-actual-integration/3',
        note='Small integration sample; hold checks are terminal snapshots, not sustained stability or superiority.')
    result['passes'] = all((complete, common_contract, paired, valid_rates, no_unsafe,
                            confirmed_stops, verified_aborts, proposed_landing, relation_active))
    return result
