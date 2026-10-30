"""Guide 10 G4: the common-PPO learnability pilot on the calibrated local plant.

One arm (``g4_pilot.arm``, Raw by registration), each model seed trained with
the registered PPO for ``updates`` x ``episodes_per_update`` complete episodes
under a training-only, success-gated curriculum, validated at difficulty 1.0 with BOTH the
deterministic mean and the sampled policy, and judged once on held-out test
seeds against the criteria registered before the run.  Every number lands in
the run root; nothing is selected on the test seeds.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from . import SCHEMA_ID
from .contracts import CurrentSemanticsConfig, contract_manifest, load_config


# ------------------------------------------------------------------ worker

_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str) -> None:
    torch.set_num_threads(1)
    _WORKER["config"] = load_config(config_path)


def _model(config: CurrentSemanticsConfig, arm: str, state: dict[str, Any]):
    from .models import CurrentActorCritic
    model = CurrentActorCritic(arm, config, seed=0)
    model.load_state_dict(state)
    model.eval()
    return model


def fly_policy(backend: Any, model: Any, config: CurrentSemanticsConfig, *, seed: int,
               deterministic: bool, generator: torch.Generator,
               record: bool = False, trace: bool = False) -> dict[str, Any]:
    """One complete episode of a frozen policy on an open backend (local or Isaac).

    The policy reads the current 21D vector only.  ``record`` adds the
    evaluator-only terminal truth and per-episode extremes for reports, and
    ``trace`` one row per decision (decoded estimate, truth, action); nothing
    recorded reaches the policy.
    """
    from .features import build_features
    from .models import tensors_from_frames
    from .observation import decode

    observations, latents, rewards, dts = [], [], [], []
    reference, tilts, rows = 0.0, [], []
    observation, info = backend.reset(int(seed))
    observations.append(observation)
    while True:
        raw, flat, nodes = tensors_from_frames([build_features(observation, config)])
        with torch.no_grad():
            out = model.act(raw, flat, nodes, deterministic=deterministic, generator=generator)
        latent = out["latent"][0].double().numpy()
        step = backend.step(out["action"][0].double().numpy(), latent=latent)
        latents.append(latent); rewards.append(step.reward); dts.append(step.transition_s)
        reference += step.info["reward_components"]["reference_reward"]
        if record:
            roll, pitch = step.info["evaluator_only"]["roll_pitch_rad"][:2]
            tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, math.cos(roll)
                                                                  * math.cos(pitch))))))
        if trace:
            seen = decode(observation)
            truth = step.info["evaluator_only"]
            rows.append({
                "t": float(step.elapsed_s), "action": step.trace.policy_accel,
                "applied": step.trace.adapter_limited_accel,
                "measured": step.trace.measured_accel,
                "measured_over_s": float(step.trace.measured_over_s),
                "estimate": {"valid": seen.estimate_valid, "ex": seen.ex, "ey": seen.ey,
                             "h": seen.h, "vrx": seen.vrx, "vry": seen.vry, "vz": seen.vz,
                             "vision_updated": seen.vision_updated,
                             "vision_age_s": seen.vision_age_s},
                "truth_r": truth["relative_position_own_minus_pad_m"],
                "truth_v": truth["relative_velocity_own_minus_pad_m_s"],
                "truth_roll_pitch_deg": [math.degrees(a) for a in truth["roll_pitch_rad"][:2]],
                "reward": float(step.reward)})
        observation = step.observation
        observations.append(observation)
        if step.done:
            break
    episode = {"seed": int(seed), "refused": None, "deterministic": bool(deterministic),
               "status": step.status, "terminal": True, "success": step.status == "SUCCESS",
               "return": float(np.sum(rewards)), "reference_return": float(reference),
               "decisions": len(rewards), "observations": np.asarray(observations),
               "latents": np.asarray(latents), "rewards": np.asarray(rewards),
               "dt_s": np.asarray(dts)}
    if "scenario" in info:  # evaluator-only composition draw
        episode["scenario"] = info["scenario"]
    if "evaluation_dropout_s" in info:  # the evaluation cell's blackout window
        episode["evaluation_dropout_s"] = info["evaluation_dropout_s"]
    if record:
        episode.update(elapsed_s=float(step.elapsed_s), termination=step.info["termination"],
                       terminal_truth=step.info["evaluator_only"],
                       max_tilt_deg=float(max(tilts)),
                       handover_wait_decisions=info.get("handover_wait_decisions"))
    if trace:
        episode["trace"] = rows
    return episode


def run_episode(config: CurrentSemanticsConfig, arm: str, state: dict[str, Any], *,
                seed: int, difficulty: float, deterministic: bool,
                action_seed: int, train: bool = False,
                split: str = "in_distribution", factors: Any = None,
                dropout: Any = None) -> dict[str, Any]:
    """One complete episode on the calibrated local plant with a frozen policy, in the
    task-v5 scenario ``split``.  ``train`` adds the registered training randomisation;
    validation and test never do.  ``factors`` (scenario.PlantFactors) and ``dropout``
    (contracts.VisionBlackout) are the development evaluation cells' conditions."""
    from .backend import make_local_backend

    model = _model(config, arm, state)
    generator = torch.Generator().manual_seed(int(action_seed))
    backend = make_local_backend(
        config, difficulty=difficulty,
        randomization=(config.g4.training_randomization if train else None),
        scenario=split, factors=factors, dropout=dropout)
    try:
        try:
            episode = fly_policy(backend, model, config, seed=seed,
                                 deterministic=deterministic, generator=generator)
        except RuntimeError as exc:  # pre-policy handover refusal: setup, not RL
            if backend.active:
                raise
            return {"seed": int(seed), "refused": str(exc)}
    finally:
        backend.close()
    episode["difficulty"] = float(difficulty)
    return episode


def _work(task: tuple) -> dict[str, Any]:
    arm, state, seed, difficulty, deterministic, action_seed, train = task
    return run_episode(_WORKER["config"], arm, state, seed=seed, difficulty=difficulty,
                       deterministic=deterministic, action_seed=action_seed, train=train)


# ------------------------------------------------------------------ helpers

class Curriculum:
    """``contracts.SuccessGatedCurriculum`` as a state machine: promote by ``step``
    once the mean sampled training success of the last ``window_updates`` updates
    at the current difficulty reaches ``promote_at_success``; never demote."""

    def __init__(self, spec: Any):
        self.spec = spec
        self.difficulty = float(spec.start_difficulty)
        self.window: list[float] = []

    def record(self, success: float | None) -> bool:
        """Log one update's success at the current difficulty; True when promoted."""
        if self.difficulty >= 1.0:
            return False
        self.window.append(0.0 if success is None else float(success))
        recent = self.window[-self.spec.window_updates:]
        if (len(recent) == self.spec.window_updates
                and float(np.mean(recent)) >= self.spec.promote_at_success):
            self.difficulty = float(min(1.0, round(self.difficulty + self.spec.step, 9)))
            self.window = []
            return True
        return False


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    flown = [e for e in episodes if not e.get("refused")]
    statuses: dict[str, int] = {}
    for e in flown:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    return {"episodes": len(flown), "refused": len(episodes) - len(flown),
            "success": (float(np.mean([e["success"] for e in flown])) if flown else None),
            "mean_return": (float(np.mean([e["return"] for e in flown])) if flown else None),
            "mean_decisions": (float(np.mean([e["decisions"] for e in flown]))
                               if flown else None),
            "statuses": statuses}


def _write(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


# ------------------------------------------------------------------ pilot

def evaluate(pool: Any, arm: str, state: dict[str, Any], seeds: tuple[int, ...], *,
             deterministic: bool, offset: int) -> dict[str, Any]:
    tasks = [(arm, state, seed, 1.0, deterministic, seed + offset, False) for seed in seeds]
    episodes = list(pool.map(_work, tasks))
    return {**summarize(episodes),
            "per_seed": [{"seed": e["seed"], "status": e.get("status"),
                          "refused": e.get("refused")} for e in episodes]}


def train_seed(config: CurrentSemanticsConfig, config_path: Path, model_seed: int,
               root: Path, pool: Any, *, log: Callable[[str], None],
               updates: int | None = None) -> dict[str, Any]:
    from .features import build_feature_batch
    from .models import CurrentActorCritic, save_checkpoint
    from .ppo import PPOTrainer, build_batch
    from ..landing.terminal import REFERENCE_DISCOUNT_TAU_S

    g4 = config.g4
    updates = g4.updates if updates is None else int(updates)
    manifest = contract_manifest(config)
    root.mkdir(parents=True, exist_ok=True)
    history, validation = root / "history.jsonl", root / "validation.jsonl"
    for path in (history, validation):
        path.unlink(missing_ok=True)
    torch.manual_seed(0)  # nothing should read it; models use private generators
    model = CurrentActorCritic(g4.arm, config, seed=model_seed)
    trainer = PPOTrainer(model, g4.ppo)
    generator = torch.Generator().manual_seed(1_000_003 + model_seed)
    features = lambda vectors: build_feature_batch(vectors, config)
    best, best_key = None, None
    curriculum, at_full = Curriculum(g4.curriculum), 0
    started = time.monotonic()
    for update in range(updates):
        difficulty = curriculum.difficulty
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        seeds = [g4.train_seed(model_seed, update, e) for e in range(g4.episodes_per_update)]
        episodes = list(pool.map(_work, [(g4.arm, state, s, difficulty, False, s + 7_000_000,
                                          True) for s in seeds]))
        flown = [e for e in episodes if not e.get("refused")]
        batch, info = build_batch(model, flown, settings=g4.ppo,
                                  tau_s=REFERENCE_DISCOUNT_TAU_S, feature_batch=features)
        stats = trainer.update(batch, generator=generator)
        summary = summarize(episodes)
        promoted = curriculum.record(summary["success"])
        row = {"update": update, "difficulty": difficulty, **summary, **info, **stats,
               "promoted_to": curriculum.difficulty if promoted else None,
               "wall_s": time.monotonic() - started}
        _append(history, row)
        if promoted:
            log(f"seed {model_seed} promoted to difficulty {curriculum.difficulty:.1f} "
                f"after update {update}")
        if update % 10 == 0 or update == updates - 1:
            log(f"seed {model_seed} update {update}/{updates} d={difficulty:.2f} "
                f"success={row['success']:.2f} return={row['mean_return']:.1f} "
                f"kl={row['max_kl']} acc/rej={stats['accepted_steps']}/{stats['rejected_steps']} "
                f"log_std={np.round(stats['log_std'], 2).tolist()}")
        last = update == updates - 1
        at_full += difficulty >= 1.0
        if difficulty >= 1.0 and (at_full % g4.validate_every == 0 or last):
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            det = evaluate(pool, g4.arm, state, g4.validation_seeds, deterministic=True,
                           offset=11_000_000)
            smp = evaluate(pool, g4.arm, state, g4.validation_seeds, deterministic=False,
                           offset=13_000_000)
            _append(validation, {"update": update, "deterministic": det, "sampled": smp})
            key = (det["success"] or 0.0, smp["success"] or 0.0, det["mean_return"] or -math.inf)
            log(f"seed {model_seed} validation @ {update}: det {det['success']:.2f} "
                f"sampled {smp['success']:.2f} return {det['mean_return']:.1f}")
            if best_key is None or key > best_key:
                best_key, best = key, update
                save_checkpoint(root / "best.pt", model, manifest=manifest,
                                extra={"update": update, "model_seed": model_seed})
    save_checkpoint(root / "final.pt", model, manifest=manifest,
                    extra={"update": updates - 1, "model_seed": model_seed})
    return {"model_seed": model_seed, "updates": updates, "best_update": best,
            "best_validation_key": list(best_key) if best_key else None,
            "final_training_difficulty": curriculum.difficulty,
            "updates_at_full_difficulty": at_full,
            "wall_s": time.monotonic() - started}


def test_checkpoint(config: CurrentSemanticsConfig, path: Path, pool: Any) -> dict[str, Any]:
    from .models import CurrentActorCritic, load_checkpoint
    model = CurrentActorCritic(config.g4.arm, config, seed=0)
    meta = load_checkpoint(path, model, manifest=contract_manifest(config))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"checkpoint": str(path), "update": meta["extra"].get("update"),
            "deterministic": evaluate(pool, config.g4.arm, state, config.g4.test_seeds,
                                      deterministic=True, offset=17_000_000),
            "sampled": evaluate(pool, config.g4.arm, state, config.g4.test_seeds,
                                deterministic=False, offset=19_000_000)}


def run_pilot(config_path: Path, root: Path, *, log: Callable[[str], None] = print,
              updates: int | None = None, workers: int | None = None) -> dict[str, Any]:
    """Train every registered model seed, test each selected checkpoint, judge G4.

    ``updates``/``workers`` overrides exist for tests only; a G4 verdict is
    written only when the registered budget ran (``protocol_complete``).
    """
    config = load_config(config_path)
    g4 = config.g4
    manifest = contract_manifest(config)
    # A ProcessPoolExecutor, not multiprocessing.Pool: a worker that dies raises
    # BrokenProcessPool here instead of being respawned forever (a silent hang).
    count = g4.workers if workers is None else int(workers)
    seeds_out, tests = [], {}
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker,
                             initargs=(str(config_path),)) as pool:
        for model_seed in g4.model_seeds:
            seed_root = Path(root) / f"seed{model_seed}"
            seeds_out.append(train_seed(config, config_path, model_seed, seed_root, pool,
                                        log=log, updates=updates))
            best = seed_root / "best.pt"
            tests[str(model_seed)] = (test_checkpoint(config, best, pool) if best.exists()
                                      else {"checkpoint": None,
                                            "note": "no checkpoint reached difficulty 1.0"})
    criteria = g4.criteria
    verdicts = {}
    for key, test in tests.items():
        det = (test.get("deterministic") or {}).get("success")
        smp = (test.get("sampled") or {}).get("success")
        verdicts[key] = {"deterministic_success": det, "sampled_success": smp,
                         "passed": (det is not None and smp is not None
                                    and det >= criteria.deterministic_success_min
                                    and smp >= criteria.sampled_success_min)}
    complete = updates is None
    report = {"schema_id": SCHEMA_ID, "stage": "G4 learnability pilot", "arm": g4.arm,
              "protocol": g4.protocol,
              "backend": "spatial_local_v12", "protocol_complete": complete,
              "hashes": {key: manifest[key] for key in ("g4_hash", "algorithm_hash", "task_hash",
                                                        "calibration_hash", "g3_hash")},
              "criteria": {"deterministic_success_min": criteria.deterministic_success_min,
                           "sampled_success_min": criteria.sampled_success_min},
              "seeds": seeds_out, "tests": tests, "verdicts": verdicts,
              "learnability_passed": complete and all(v["passed"] for v in verdicts.values()),
              "note": ("Raw PPO only, local plant; a learnability pilot, not a comparison "
                       "of arms and not an Isaac result")}
    return report
