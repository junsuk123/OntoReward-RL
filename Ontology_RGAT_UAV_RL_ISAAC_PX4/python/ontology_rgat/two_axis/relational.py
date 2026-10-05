"""Validation-only, raw-policy-preserving R-GAT activation (upstream v2.8).

Nonzero weights are necessary, not evidence of useful relational reasoning.
Only real PPO gradients produce a candidate; no epsilon/jitter is injected.
The guard can reject every scale, in which case the anchor is preserved.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RelationGuardOptions:
    scale_grid: tuple[float, ...] = (1., .75, .5, .25, .1, .05, .02, .01, .005, .002, .001)
    mean_return_tolerance: float = .25
    selection_score_tolerance: float = .25
    rate_tolerance: float = 0.
    min_residual_norm: float = 1e-6
    max_residual_norm: float = .005
    active_tolerance: float = 1e-10

    def __post_init__(self):
        if (not self.scale_grid or any(not math.isfinite(v) or not 0 < v <= 1
                                      for v in self.scale_grid)
                or any(a < b for a, b in zip(self.scale_grid, self.scale_grid[1:]))):
            raise ValueError("relation scale grid must be descending in (0,1]")
        values = asdict(self)
        values.pop("scale_grid")
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("guard tolerances must be finite and nonnegative")
        if self.max_residual_norm < self.min_residual_norm:
            raise ValueError("invalid relation residual trust region")


@torch.no_grad()
def relational_path_audit(agent, tolerance=1e-10):
    required = (agent.mode == "ppo_ontology_rgat"
                and hasattr(agent.actor, "relational_delta"))
    report = {"required": required, "parameter_path_active": False,
              "tolerance": float(tolerance)}
    if not required:
        return report
    for role in ("actor", "critic"):
        head = getattr(agent, role)
        report[f"{role}_readout_norm"] = float(head.encoder.readout.weight.norm())
        report[f"{role}_head_norm"] = float(head.residual.weight.norm())
    report["parameter_path_active"] = all(
        report[f"{role}_{part}_norm"] > tolerance
        for role in ("actor", "critic") for part in ("readout", "head"))
    return report


def assert_raw_preserved(anchor, candidate):
    if (anchor.mode != candidate.mode or not relational_path_audit(anchor)["required"]
            or not relational_path_audit(candidate)["required"]):
        raise ValueError("relation guard requires two compatible grouped R-GAT agents")
    for role in ("actor", "critic"):
        left, right = getattr(anchor, role), getattr(candidate, role)
        a, b = left.raw.state_dict(), right.raw.state_dict()
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError(f"relation-only adaptation changed protected {role} raw path")
    if not torch.equal(anchor.actor.log_std, candidate.actor.log_std):
        raise ValueError("relation-only adaptation changed protected log_std")


@torch.no_grad()
def scale_relational_readout(candidate, scale):
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("relation readout scale must be in [0,1]")
    trial = copy.deepcopy(candidate)
    for head in (trial.actor, trial.critic):
        head.encoder.readout.weight.mul_(scale)
        head.encoder.readout.bias.mul_(scale)
    return trial


def guard_relational_candidate(anchor, candidate, *, evaluator, options=None):
    """Largest acceptable scale on a caller-supplied *validation* evaluator.

    The production entry point below binds the evaluator to VALIDATION_SEEDS.
    A callback makes the guard's failure and rejection paths directly testable.
    Neither input model is modified, including when every candidate is rejected.
    """
    options = options or RelationGuardOptions()
    assert_raw_preserved(anchor, candidate)
    base = evaluator(anchor)
    required = ("landing_rate", "unsafe_rate", "safe_abort_rate", "task_timeout_rate",
                "mean_return", "selection_score")
    if not all(math.isfinite(float(base[k])) for k in required):
        raise ValueError("anchor validation metrics must be finite")
    report = {"accepted": False, "selected_scale": 0., "anchor_info": base,
              "selected_info": base, "options": asdict(options), "trials": []}
    selected = anchor
    for scale in options.scale_grid:
        trial = scale_relational_readout(candidate, scale)
        info = evaluator(trial)
        delta = np.asarray(info["mean_abs_relation_residual"], dtype=float)
        residual_norm = float(np.linalg.norm(delta))
        finite = (delta.shape == (anchor.actor.log_std.numel(),) and np.isfinite(delta).all()
                  and np.all(delta >= 0)
                  and all(math.isfinite(float(info[k])) for k in required))
        tol = options.rate_tolerance + 1e-12
        outcome_ok = (finite and info["landing_rate"] >= base["landing_rate"] - tol
                      and all(info[k] <= base[k] + tol for k in
                              ("unsafe_rate", "safe_abort_rate", "task_timeout_rate")))
        score_ok = (finite and info["mean_return"] >= base["mean_return"]
                    - options.mean_return_tolerance - 1e-12
                    and info["selection_score"] >= base["selection_score"]
                    - options.selection_score_tolerance - 1e-12)
        audit = relational_path_audit(trial, options.active_tolerance)
        residual_ok = (finite and options.min_residual_norm <= residual_norm
                       <= options.max_residual_norm)
        accepted = bool(outcome_ok and score_ok and residual_ok and audit["parameter_path_active"])
        report["trials"].append({"scale": scale, "validation": info, "audit": audit,
                                 "residual_norm": residual_norm,
                                 "outcome_guard": bool(outcome_ok),
                                 "score_guard": bool(score_ok),
                                 "residual_guard": bool(residual_ok), "accepted": accepted})
        if accepted:
            selected = trial
            report.update(accepted=True, selected_scale=scale, selected_info=info)
            break
    report["selected_audit"] = relational_path_audit(selected, options.active_tolerance)
    return selected, report


def ensure_relational_path(anchor, config, hyper, *, seed, validation_seeds):
    """Repair only a selected nominal checkpoint; never read the test split.

    Uses the existing fixed-decision Python PPO collector, an explicit remaining
    difference from MATLAB's six complete episodes per repair iteration.
    """
    from .curriculum import CurriculumScheduler
    from .environment import TwoAxisLandingEnv
    from .training import (EpisodeDrivenCollector, PPOTrainer, VALIDATION_SEEDS,
                           evaluate_policy)

    if (not validation_seeds or len(set(validation_seeds)) != len(validation_seeds)
            or not set(validation_seeds).issubset(VALIDATION_SEEDS)):
        raise ValueError("relation activation accepts only the declared validation split")
    before = relational_path_audit(anchor)
    report = {"attempted": False, "changed": False, "environment_steps": 0,
              "before_audit": before, "after_audit": before,
              "validation_seeds": list(validation_seeds), "history": []}
    if not before["required"] or before["parameter_path_active"]:
        return anchor, report
    candidate = copy.deepcopy(anchor)
    for head in (candidate.actor, candidate.critic):
        head.set_adaptation(config.ontology, True)
    repair_hyper = replace(hyper, iterations=config.ontology.activation_iterations,
                           decisions_per_iteration=config.ontology.activation_decisions)
    trainer = PPOTrainer(candidate, repair_hyper)
    nominal = replace(config, curriculum=replace(config.curriculum, enabled=False))
    scheduler = CurriculumScheduler(nominal.curriculum, difficulty=1.)
    training_seed = 8_100_000 + 10_000 * int(seed)
    collector = EpisodeDrivenCollector(TwoAxisLandingEnv(nominal, perturbations=True),
                                       scheduler, base_seed=training_seed)
    report.update(attempted=True, train_seed_base=training_seed,
                  hyperparameters=asdict(repair_hyper))
    for _ in range(repair_hyper.iterations):
        rollout = collector.collect(candidate, repair_hyper.decisions_per_iteration)
        report["history"].append(trainer.update(
            rollout, discount_time_constant_s=config.timing.discount_time_constant_s))
        report["environment_steps"] += len(rollout)
    assert_raw_preserved(anchor, candidate)
    selected, guard = guard_relational_candidate(
        anchor, candidate, evaluator=lambda agent: evaluate_policy(
            agent, config, seeds=validation_seeds))
    report.update(changed=guard["accepted"], guard=guard,
                  after_audit=relational_path_audit(selected))
    return selected, report
