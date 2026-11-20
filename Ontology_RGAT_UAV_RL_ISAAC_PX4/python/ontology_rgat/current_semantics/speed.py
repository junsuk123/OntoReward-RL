"""learning-speed/1: does the ontology-structured policy (ORL, ``orl/1``) reach the task's
full difficulty in fewer PPO updates than PPO on the raw observation and the untyped GNN,
under the base self-navigation, i.e. with NO navigation estimator in the loop?

User decision 2026-11-20 ("1"): the kinematic WLS estimator (internal name WMON) belongs to
separate research and must not carry this study's contribution; under the base navigation
and under CA-KF the registered R-GAT arm learned exactly as fast as PPO, while ORL --
the same ontology used as the policy's STRUCTURE, no raw bypass -- reached difficulty 1.0
first on 5 of 5 development seeds against every arm (mean update 70 vs 107-109; status doc
20.3).  That observation was exploratory (the metric was read after the fact), so this
protocol tests it on fresh model seeds with the decision rule fixed in the plan file before
any of its models trains.

Procedure.  Each training seed trains the proposed arm and every comparison arm with the
G4-passed recipe of the base contract (budget per update, curriculum, PPO settings,
training randomisation), the same episode seeds for all arms of a seed, and stops at the
first update trained at difficulty 1.0 (``pilot.train_seed(stop_at_full=True)``) or at the
cap.  No checkpoint is selected and nothing is evaluated: the event is the whole outcome.

Decision rule (``RULE``, hashed into the registration).  Per comparison arm X, the paired
difference per training seed is ln T(ORL) - ln T(X), T = first update at difficulty 1.0,
a model still below 1.0 at the cap counted as the cap.  ORL is FASTER than X when the upper
bound of the two-sided paired-t interval at 1 - alpha/2 (Bonferroni over the two
comparisons) is below zero, SLOWER when the lower bound is above zero, otherwise not
established.  A censored comparison model makes its difference a bound in X's favour
(conservative); a censored ORL model makes it a bound in ORL's favour, so any censored ORL
model blocks the FASTER verdict for every comparison.  The exact sign test and the ratio of
means are reported, never decisive.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import yaml

from . import development as dev
from . import orl
from .contracts import (AUDITED_SEED_RANGES, DIAGNOSTIC_SEED_RANGES, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, load_config)

# Isaac environment seeds the development's Isaac flights spent.
SPENT_ISAAC_SEEDS = (("isaac-development environments", 994_101, 994_130),)


# ----------------------------------------------------- helpers (from confirmation.py, archived
# 2026-11-20 with every non-ORL route; copied verbatim so the protocol's behaviour is unchanged)

def _alpha(value: Any, where: str) -> float:
    alpha = dev._number(value, where, low=0.0)
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"{where} must lie in (0, 0.5)")
    return alpha


CLAIM = "claim"   # one per model directory: the pid training it


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def claim(directory: Path) -> bool:
    """Atomically claim one model for this process.  A claim whose pid is gone is cleared,
    together with the dead run's checkpoints, so a model is trained from scratch exactly once
    by whichever process holds it (``train_seed`` restarts its own history files)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CLAIM
    for _ in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            text = path.read_text(encoding="utf-8").split()
            pid = int(text[0]) if text and text[0].isdigit() else -1
            if pid > 0 and _alive(pid):
                return pid == os.getpid()
            path.unlink(missing_ok=True)
            continue
        os.write(handle, f"{os.getpid()} {dt.datetime.now().isoformat()}\n".encode())
        os.close(handle)
        for stale in ("best.pt", "final.pt"):
            (directory / stale).unlink(missing_ok=True)
        return True
    return False


def release(directory: Path) -> None:
    (directory / CLAIM).unlink(missing_ok=True)


def paired_interval(diffs: list[float], confidence: float) -> dict[str, Any]:
    """Two-sided paired-t interval of the mean of ``diffs`` (one per training seed)."""
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    if n < 2 or not math.isfinite(se):
        return {"n": n, "mean": mean, "sd": sd, "confidence": confidence, "low": None,
                "high": None, "t": None, "p": None, "per_seed": d.tolist()}
    q = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    if se == 0.0:
        t, p = (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    else:
        t = mean / se
        p = float(2.0 * stats.t.sf(abs(t), n - 1))
    return {"n": n, "mean": mean, "sd": sd, "confidence": confidence,
            "low": mean - q * se, "high": mean + q * se, "t": t, "p": p,
            "per_seed": d.tolist()}


PROTOCOL = "learning-speed/1"
DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "learning_speed_v1.yaml"
FOLDER = "learning_speed"                 # <root>/learning_speed/
REGISTRATION = "learning_speed_registration.json"
RESULT = "learning_speed.json"
FULL = 1.0
COMPARABLE = ("raw_ppo", "untyped_gnn_ppo", "semantic_mlp_ppo", "ontology_rgat_ppo")
RULE = (
    "per comparison arm X and training seed: d = ln T(proposed) - ln T(X), T = first update "
    "trained at difficulty 1.0, censored at the cap = the cap; proposed FASTER than X iff the "
    "upper bound of the two-sided paired-t interval of mean d at 1 - family_alpha/2 "
    "(Bonferroni over the comparisons) is < 0 AND no proposed-arm model is censored; SLOWER "
    "iff the lower bound is > 0; otherwise not established.  Sign test and ratio of means "
    "are reported, never decisive.")
# Training episode seeds other registered plans own (base + model seed * 1e6 + update * 12
# + episode): confirmation/1 (model seeds 20-39) and its miniature and CA-KF replication.
SPENT_TRAINING_SEEDS = (("confirmation/1, confirmation-mini/1, replication-mini/1 training",
                         120_000_000, 139_007_199),)
SPENT_MODEL_SEEDS = frozenset({0, 1, *range(10, 15), *range(20, 40)})


@dataclass(frozen=True)
class SpeedPlan:
    protocol: str
    proposed: str                       # an ORL version, e.g. "orl/1"
    comparisons: tuple[str, ...]        # registered arm names
    training_seeds: tuple[int, ...]
    train_seed_base: int
    cap: int
    workers: int
    family_alpha: float

    @property
    def proposed_arm(self) -> str:
        return orl.ARMS[self.proposed]

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.proposed_arm, *self.comparisons)

    def train_seed(self, model_seed: int, update: int, episode: int, per: int) -> int:
        return self.train_seed_base + model_seed * 1_000_000 + update * per + episode

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "proposed_arm": self.proposed_arm}


def load_plan(path: str | Path, config: CurrentSemanticsConfig) -> SpeedPlan:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = dev._section(data, ("protocol", "proposed", "comparisons", "training_seeds",
                               "train_seed_base", "cap", "workers", "family_alpha"),
                        "learning_speed")
    if data["protocol"] != PROTOCOL:
        raise ValueError(f"learning_speed.protocol must be {PROTOCOL!r}")
    if data["proposed"] not in orl.ARMS:
        raise ValueError(f"learning_speed.proposed must be one of {sorted(orl.ARMS)}")
    comparisons = tuple(data["comparisons"])
    if not comparisons or len(set(comparisons)) != len(comparisons) \
            or not set(comparisons) <= set(COMPARABLE):
        raise ValueError(f"learning_speed.comparisons must be distinct arms of {COMPARABLE}")
    seeds = dev._seeds(data["training_seeds"], "learning_speed.training_seeds")
    if len(seeds) < 3:
        raise ValueError("a paired-t comparison needs at least three training seeds")
    plan = SpeedPlan(
        protocol=PROTOCOL, proposed=data["proposed"], comparisons=comparisons,
        training_seeds=seeds,
        train_seed_base=int(dev._number(data["train_seed_base"], "train_seed_base", low=0)),
        cap=int(dev._number(data["cap"], "cap", low=1)),
        workers=int(dev._number(data["workers"], "workers", low=1)),
        family_alpha=_alpha(data["family_alpha"], "family_alpha"))
    _check_contract(config)
    _audit(plan, config)
    return plan


def _check_contract(config: CurrentSemanticsConfig) -> None:
    """learning-speed/1 runs on the base self-navigation only: no navigation estimator,
    above all not the kinematic WLS one, may sit in the policy's observation path."""
    if config.navigation is not None:
        raise ValueError("learning-speed/1 runs on the base navigation contract only "
                         "(spatial_view_contact_v1.yaml); this config carries a navigation "
                         "estimator")


def _audit(plan: SpeedPlan, config: CurrentSemanticsConfig) -> None:
    """No training seed of this plan may lie in any recorded or registered use."""
    g4 = config.g4
    per = g4.episodes_per_update
    if set(plan.training_seeds) & (SPENT_MODEL_SEEDS | set(g4.model_seeds)):
        raise ValueError("learning-speed training seeds reuse a G4, development or "
                         "confirmation model seed")
    used = [(name, first, last) for name, first, last, _ in AUDITED_SEED_RANGES]
    used += list(DIAGNOSTIC_SEED_RANGES) + list(dev.SPENT_DEVELOPMENT_SEEDS)
    used += list(SPENT_ISAAC_SEEDS) + list(SPENT_TRAINING_SEEDS)
    used += [("g4 pilot training", *dev._PILOT_SEEDS)]
    taken = (set(g4.validation_seeds) | set(g4.test_seeds) | set(config.g3.pilot_seeds)
             | set(config.isaac_eval.seeds))
    blocks = [(s, plan.train_seed(s, 0, 0, per), plan.train_seed(s, plan.cap - 1, per - 1, per))
              for s in plan.training_seeds]
    for seed, low, high in blocks:
        for name, first, last in used:
            if low <= last and first <= high:
                raise ValueError(f"learning-speed seed {seed} episodes {low}-{high} overlap "
                                 f"{name}")
        if any(low <= s <= high for s in taken):
            raise ValueError(f"learning-speed seed {seed} episodes {low}-{high} overlap a "
                             "registered seed")
    spans = sorted((a, b) for _, a, b in blocks)
    if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
        raise ValueError("learning-speed seed blocks overlap each other")


def arm_config(config: CurrentSemanticsConfig, plan: SpeedPlan, arm: str
               ) -> CurrentSemanticsConfig:
    """The base contract's G4 recipe re-pointed at one arm, this plan's seeds and its cap."""
    return replace(config, g4=replace(config.g4, arm=arm, model_seeds=plan.training_seeds,
                                      train_seed_base=plan.train_seed_base,
                                      updates=plan.cap))


def plan_hash(plan: SpeedPlan, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    manifest = contract_manifest(config)
    return digest({"plan": plan.to_dict(), "rule": RULE, "orl_hash": orl.HASHES[plan.proposed],
                   "contract": {k: manifest[k] for k in dev.BOUND_HASHES}})


def out_dir(root: Path) -> Path:
    return Path(root) / FOLDER


def label(arm: str) -> str:
    return arm.replace(":", "_")


def model_keys(plan: SpeedPlan) -> tuple[str, ...]:
    """Seed-major, so the arms of one seed train close together in time."""
    return tuple(f"{s}/{arm}" for s in plan.training_seeds for arm in plan.arms)


def model_dir(root: Path, key: str) -> Path:
    seed, arm = key.split("/", 1)
    return out_dir(root) / f"seed{seed}" / label(arm)


def resolved(plan: SpeedPlan, config: CurrentSemanticsConfig) -> dict[str, Any]:
    n = len(model_keys(plan))
    return {"protocol": PROTOCOL, "plan_hash": plan_hash(plan, config),
            "proposed_arm": plan.proposed_arm, "comparisons": list(plan.comparisons),
            "training_seeds": list(plan.training_seeds), "models": n, "cap": plan.cap,
            "episodes_per_update": config.g4.episodes_per_update,
            "max_training_episodes": n * plan.cap * config.g4.episodes_per_update,
            "event": "first update trained at difficulty 1.0", "rule": RULE,
            "family_alpha": plan.family_alpha, "evaluates": False, "starts_isaac": False}


# ------------------------------------------------------------- registration

def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    """Write the registration before any model trains; an existing one must match."""
    from .experiment import write_json
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    payload = {**resolved(plan, config), "plan_file": str(Path(plan_path).resolve())}
    path = out_dir(root) / REGISTRATION
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored.pop("registered_utc", None)
        if stored.get("plan_hash") != payload["plan_hash"]:
            return 3, {"stage": "register-speed", "status": "REFUSED",
                       "reason": f"{path} registers another plan"}
        return 0, {"stage": "register-speed", "status": "ALREADY_REGISTERED", **payload}
    if any(out_dir(root).glob("seed*/*/history.jsonl")):
        return 3, {"stage": "register-speed", "status": "REFUSED",
                   "reason": "models were trained before the registration; archive them"}
    payload = {**payload, "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return 0, {"stage": "register-speed", "status": "REGISTERED", **payload}


def check_registration(plan: SpeedPlan, config: CurrentSemanticsConfig, root: Path
                       ) -> str | None:
    path = out_dir(root) / REGISTRATION
    if not path.exists():
        return f"{PROTOCOL} is not registered ({path}); run --stage register-speed first"
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("plan_hash") != plan_hash(plan, config):
        return f"{path} registers another plan or contract; the plan file or code changed"
    return None


# ------------------------------------------------------------------ training

def first_full(rows: list[dict[str, Any]]) -> int | None:
    return next((int(r["update"]) for r in rows if float(r["difficulty"]) >= FULL), None)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _summary(path: Path, digest: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("plan_hash") != digest or data.get("complete") is not True:
        raise RuntimeError(f"{path} belongs to another run; archive it")
    return data


def run(config_path: Path, plan_path: Path, root: Path, *,
        log: Callable[[str], None] = print, workers: int | None = None) -> tuple[int, dict]:
    """Train every registered model nobody holds; any number of processes share them."""
    import torch
    from .experiment import write_json
    from .pilot import _init_worker, train_seed
    if torch.get_num_threads() != 1:
        return 3, {"stage": "speed", "status": "REFUSED",
                   "reason": f"torch uses {torch.get_num_threads()} threads; training is "
                             "deterministic only single-threaded (run through run.sh)"}
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    problem = check_registration(plan, config, root)
    if problem:
        return 3, {"stage": "speed", "status": "REFUSED", "reason": problem}
    digest = plan_hash(plan, config)
    keys = model_keys(plan)
    pending = [k for k in keys if _summary(model_dir(root, k) / "summary.json", digest) is None]
    log(f"speed: {len(keys) - len(pending)} of {len(keys)} models done")
    done_here: list[str] = []
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=int(workers or plan.workers),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(config_path),)) as pool:
            for key in pending:
                directory = model_dir(root, key)
                if _summary(directory / "summary.json", digest) is not None \
                        or not claim(directory):
                    continue
                seed_text, arm = key.split("/", 1)
                acfg = arm_config(config, plan, arm)
                log(f"speed: {key} trains to difficulty 1.0 or update {plan.cap}")
                dev._status(directory, "training")
                trained = train_seed(acfg, config_path, int(seed_text), directory, pool,
                                     log=log, updates=plan.cap, stop_at_full=True)
                rows = _rows(directory / "history.jsonl")
                event = first_full(rows)
                write_json(directory / "summary.json", {
                    "protocol": PROTOCOL, "plan_hash": digest, "complete": True, "model": key,
                    "arm": arm, "training_seed": int(seed_text), "cap": plan.cap,
                    "first_full_update": event, "censored": event is None,
                    "final_difficulty": float(rows[-1]["difficulty"]),
                    "updates_run": len(rows), "training": trained})
                release(directory)
                dev._status(directory, "done")
                done_here.append(key)
                log(f"speed: {key} first difficulty-1.0 update "
                    f"{event if event is not None else f'none by {plan.cap}'}")
    return 0, {"stage": "speed", "protocol": PROTOCOL, "trained_here": done_here,
               "wall_s": time.monotonic() - started}


# ------------------------------------------------------------------ analysis

def compare(plan: SpeedPlan, proposed: dict[int, dict[str, Any]],
            other: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The registered comparison of one arm pair (``RULE``)."""
    from scipy import stats
    seeds = sorted(set(proposed) & set(other))
    t = lambda r: plan.cap if r["censored"] else r["first_full_update"]
    diffs = [math.log(t(proposed[s])) - math.log(t(other[s])) for s in seeds]
    interval = paired_interval(diffs, 1.0 - plan.family_alpha / len(plan.comparisons))
    proposed_censored = sum(proposed[s]["censored"] for s in seeds)
    if interval["high"] is not None and interval["high"] < 0.0 and not proposed_censored:
        verdict = "faster"
    elif interval["low"] is not None and interval["low"] > 0.0:
        verdict = "slower"
    else:
        verdict = "not established"
    wins = sum(d < 0 for d in diffs)
    losses = sum(d > 0 for d in diffs)
    untied = wins + losses
    return {
        "verdict": verdict, "log_ratio": interval,
        "ratio": math.exp(interval["mean"]),
        "ratio_interval": ([math.exp(interval["low"]), math.exp(interval["high"])]
                           if interval["low"] is not None else None),
        "proposed_first": wins, "other_first": losses, "ties": len(diffs) - untied,
        "sign_test_p": float(stats.binomtest(wins, untied, 0.5).pvalue) if untied else None,
        "proposed_censored": proposed_censored,
        "other_censored": sum(other[s]["censored"] for s in seeds),
        "per_seed": [{"seed": s, "proposed": t(proposed[s]), "other": t(other[s]),
                      "proposed_censored": proposed[s]["censored"],
                      "other_censored": other[s]["censored"]} for s in seeds]}


def report(config_path: Path, plan_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    problem = check_registration(plan, config, root)
    if problem:
        raise ValueError(problem)
    digest = plan_hash(plan, config)
    records: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in plan.arms}
    pending = []
    for key in model_keys(plan):
        seed, arm = key.split("/", 1)
        summary = _summary(model_dir(root, key) / "summary.json", digest)
        if summary is None:
            pending.append(key)
        else:
            records[arm][int(seed)] = summary
    arms = {}
    for arm, by_seed in records.items():
        times = [plan.cap if r["censored"] else r["first_full_update"] for r in by_seed.values()]
        arms[arm] = {"models": len(by_seed),
                     "censored": sum(r["censored"] for r in by_seed.values()),
                     "first_full_update": {s: r["first_full_update"]
                                           for s, r in sorted(by_seed.items())},
                     "mean_update_censored_at_cap": float(np.mean(times)) if times else None,
                     "median_update_censored_at_cap": float(np.median(times)) if times else None}
    complete = not pending
    comparisons = ({other: compare(plan, records[plan.proposed_arm], records[other])
                    for other in plan.comparisons} if complete else None)
    verdict = None
    if comparisons is not None:
        verdicts = {k: v["verdict"] for k, v in comparisons.items()}
        verdict = {"per_comparison": verdicts,
                   "proposed_faster_than_every_comparison":
                       all(v == "faster" for v in verdicts.values())}
    return {"protocol": PROTOCOL, "plan_hash": digest, "complete": complete,
            "pending": pending, "proposed_arm": plan.proposed_arm, "cap": plan.cap,
            "rule": RULE, "arms": arms, "comparisons": comparisons, "verdict": verdict}
