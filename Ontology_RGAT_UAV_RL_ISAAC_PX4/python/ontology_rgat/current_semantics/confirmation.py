"""Guide 11.4-11.6: the independent confirmation of a passed development gate.

``confirmation/1`` repeats the development procedure of a root whose development gate
PASSED -- the same four arms, PPO budget, curriculum, checkpoint selection and ten
evaluation cells, all read from that root's registered development plan -- on fresh
training, validation and evaluation seeds, and judges it by a decision rule fixed in the
plan file before any confirmation model trains (``confirmation_v1.yaml``).

The unit is the training seed.  Each seed's four arms meet the same environments, so a
seed contributes one paired difference per comparison; a 100-environment cell is never
counted as 100 replicates.

* Local family (guide 11.5 "확인"): the primary metric is each model's mean success over
  the ten cells.  The two primary comparisons (R-GAT minus Semantic MLP, R-GAT minus
  Untyped GNN) each need the lower bound of a two-sided paired-t interval at
  1 - alpha/2 (Bonferroni) above zero.  Only then is the practical comparison (R-GAT
  minus Raw PPO) read, at 1 - alpha.  Guards against Raw: ID and four-OOD success
  interval lower bounds >= -2 pp, OOD unsafe interval upper bound <= +1 pp.
* Isaac family (guide 11.6.5): every confirmation model flies the same registered Isaac
  cases deterministically, and the same comparisons are made on each model's Isaac
  success rate.  Risk (UNSAFE_CONTACT + SAFETY_ENVELOPE_VIOLATION) is a secondary family,
  Holm-adjusted.

A model that never selects a checkpoint counts as zero success and is never dropped
(the development rule).  ``register`` binds the plan, the development registration and
verdict, and the contract hashes; ``run`` and the Isaac runner refuse anything else.
Training can be split into shards (one process per shard, each under its own lock); the
verdict is written by whichever process finds every model complete.

``confirmation-mini/1`` (user, 2026-11-19: "miniature size validation") is the same procedure
and decision rule at a smaller size: fewer fresh seeds, a shorter PPO budget (``updates``,
same 12 episodes per update and PPO settings for every arm), fewer environments per cell and
fewer Isaac cases.  It writes to its own folder, so the two protocols never mix.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import yaml

from . import SCHEMA_ID
from . import development as dev
from . import isaac_development as idev
from .contracts import (AUDITED_SEED_RANGES, DIAGNOSTIC_SEED_RANGES, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, load_config)

PROTOCOL = "confirmation/1"
MINI_PROTOCOL = "confirmation-mini/1"
# protocol -> (its Isaac protocol, its folder under the run root)
PROTOCOLS = {PROTOCOL: ("isaac-confirmation/1", "confirmation"),
             MINI_PROTOCOL: ("isaac-confirmation-mini/1", "confirmation_mini")}
ISAAC_PROTOCOL = PROTOCOLS[PROTOCOL][0]
FULL_PATH = PROJECT_ROOT / "config" / "current_semantics" / "confirmation_v1.yaml"
DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "confirmation_mini_v1.yaml"
REGISTRATION = "confirmation_registration.json"
RESULT = "confirmation.json"
ISAAC_OUT = "isaac"                       # <root>/confirmation/isaac/
CHECKPOINTS = "checkpoints.json"          # sha256 of every model, recorded at its first flight
LOCAL_REFERENCE = "local_reference.jsonl"
PROPOSED = "ontology_rgat_ppo"
RISK_STATUSES = ("UNSAFE_CONTACT", "SAFETY_ENVELOPE_VIOLATION")
GUARD_REFERENCE = "raw_ppo"   # guide 11.5: the ID / OOD / unsafe guards are against Raw PPO
AMENDMENT = "confirmation_amendment.json"
AMENDMENT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "confirmation_mini_amendment_v1.yaml"
# Isaac environment seeds the development's Isaac flights spent.
SPENT_ISAAC_SEEDS = (("isaac-development environments", 994_101, 994_130),)


@dataclass(frozen=True)
class Analysis:
    primary_comparisons: tuple[str, ...]
    primary_family_alpha: float
    practical_comparison: str | None
    practical_alpha: float
    guard_alpha: float
    id_success_ci_low_min: float
    four_ood_success_ci_low_min: float
    ood_unsafe_ci_high_max: float
    risk_family_alpha: float


@dataclass(frozen=True)
class ConfirmationPlan:
    protocol: str
    development_protocol: str
    models: dev.DevelopmentPlan        # the development plan with the confirmation seeds
    analysis: Analysis
    isaac: idev.IsaacDevelopmentPlan   # conditions and fresh environment seeds
    updates: int                       # PPO updates per model (12 complete episodes each)
    # set by an amendment (``effective_plan``): the hash every model summary carries is the
    # REGISTERED plan's, while the arm set and decision rule are the amended ones
    registered_hash: str | None = None

    @property
    def folder(self) -> str:
        return PROTOCOLS[self.protocol][1]

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "development_protocol": self.development_protocol,
                "models": self.models.to_dict(), "analysis": asdict(self.analysis),
                "isaac": self.isaac.to_dict(), "updates": self.updates}


# ------------------------------------------------------------------- loading

def _alpha(value: Any, where: str) -> float:
    alpha = dev._number(value, where, low=0.0)
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"{where} must lie in (0, 0.5)")
    return alpha


def load_plan(path: str | Path, config: CurrentSemanticsConfig) -> ConfirmationPlan:
    """The confirmation plan over the config's registered development plan."""
    raw_data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("confirmation plan must be a mapping")
    optional = {k: raw_data.pop(k) for k in ("updates", "episodes_per_cell") if k in raw_data}
    data = dev._section(raw_data, ("protocol", "development_protocol", "training_seeds",
                                   "train_seed_base", "validation_seeds", "workers",
                                   "env_seed_base", "analysis", "isaac"), "confirmation")
    if data["protocol"] not in PROTOCOLS:
        raise ValueError(f"confirmation.protocol must be one of {tuple(PROTOCOLS)}")
    updates = int(dev._number(optional.get("updates", config.g4.updates), "updates", low=1))
    if updates > config.g4.updates:
        raise ValueError(f"confirmation.updates may shorten, never extend, the registered "
                         f"budget of {config.g4.updates} updates")
    development = dev.load_development(dev.default_plan_path(config), config)
    if data["development_protocol"] != development.protocol:
        raise ValueError("confirmation.development_protocol must name the registered "
                         f"development protocol {development.protocol!r}")
    seeds = dev._seeds(data["training_seeds"], "confirmation.training_seeds")
    if len(seeds) < 3:
        raise ValueError("a paired-t confirmation needs at least three training seeds")
    models = replace(
        development, protocol=PROTOCOL, training_seeds=seeds,
        train_seed_base=int(dev._number(data["train_seed_base"], "train_seed_base", low=0)),
        validation_seeds=dev._seeds(data["validation_seeds"], "confirmation.validation_seeds"),
        workers=int(dev._number(data["workers"], "workers", low=1)),
        env_seed_base=int(dev._number(data["env_seed_base"], "env_seed_base", low=0)),
        episodes_per_cell=int(dev._number(optional.get("episodes_per_cell",
                                                       development.episodes_per_cell),
                                          "episodes_per_cell", low=1)))
    raw = dev._section(data["analysis"], tuple(Analysis.__dataclass_fields__),
                       "confirmation.analysis")
    analysis = Analysis(
        primary_comparisons=tuple(raw["primary_comparisons"]),
        primary_family_alpha=_alpha(raw["primary_family_alpha"], "primary_family_alpha"),
        practical_comparison=raw["practical_comparison"],
        practical_alpha=_alpha(raw["practical_alpha"], "practical_alpha"),
        guard_alpha=_alpha(raw["guard_alpha"], "guard_alpha"),
        id_success_ci_low_min=float(raw["id_success_ci_low_min"]),
        four_ood_success_ci_low_min=float(raw["four_ood_success_ci_low_min"]),
        ood_unsafe_ci_high_max=float(raw["ood_unsafe_ci_high_max"]),
        risk_family_alpha=_alpha(raw["risk_family_alpha"], "risk_family_alpha"))
    # Guide 11.3: two primary comparisons, then the practical one against Raw.
    if analysis.primary_comparisons != ("semantic_mlp_ppo", "untyped_gnn_ppo") \
            or analysis.practical_comparison != "raw_ppo":
        raise ValueError("confirmation.analysis must compare R-GAT with semantic_mlp_ppo and "
                         "untyped_gnn_ppo (primary) and raw_ppo (practical), guide 11.3")
    isaac_data = dev._section(data["isaac"], ("protocol", "mode", "conditions"),
                              "confirmation.isaac")
    isaac_protocol = PROTOCOLS[data["protocol"]][0]
    if isaac_data["protocol"] != isaac_protocol or isaac_data["mode"] != "deterministic":
        raise ValueError(f"confirmation.isaac must be {isaac_protocol!r}, deterministic")
    conditions = []
    for name, cell_seeds in (isaac_data["conditions"] or {}).items():
        if name not in dev.CELLS or models.cell(name).optical_noise_scale != 1.0:
            raise ValueError(f"isaac condition {name!r} is not a flyable development cell")
        conditions.append((name, dev._seeds(cell_seeds, f"confirmation.isaac.{name}")))
    if not conditions:
        raise ValueError("confirmation.isaac.conditions must not be empty")
    isaac = idev.IsaacDevelopmentPlan(conditions=tuple(conditions), protocol=isaac_protocol,
                                      development_protocol=data["protocol"])
    if len(set(isaac.seeds)) != len(isaac.seeds):
        raise ValueError("confirmation.isaac seeds must be distinct across conditions")
    plan = ConfirmationPlan(data["protocol"], development.protocol, models, analysis, isaac,
                            updates)
    _audit(plan, development, config)
    return plan


def budget_config(config: CurrentSemanticsConfig, plan: ConfirmationPlan
                  ) -> CurrentSemanticsConfig:
    """The main contract with the plan's PPO budget: only the number of updates may differ
    (confirmation-mini/1); every arm, PPO setting and episode count per update is kept.
    Checkpoints are saved and loaded under this config's manifest."""
    if plan.updates == config.g4.updates:
        return config
    return replace(config, g4=replace(config.g4, updates=plan.updates))


def _blocks(plan: dev.DevelopmentPlan, config: CurrentSemanticsConfig,
            updates: int | None = None) -> list[tuple[str, int, int]]:
    per = config.g4.episodes_per_update
    updates = config.g4.updates if updates is None else updates
    out = [("validation", s, s) for s in plan.validation_seeds]
    out += [("training", plan.train_seed(s, 0, 0, per), plan.train_seed(s, updates - 1,
                                                                         per - 1, per))
            for s in plan.training_seeds]
    out.append(("evaluation", plan.eval_seed(0, 0, 0),
                plan.eval_seed(len(plan.training_seeds) - 1, len(dev.CELLS) - 1,
                               plan.episodes_per_cell - 1)))
    return out


def _audit(plan: ConfirmationPlan, development: dev.DevelopmentPlan,
           config: CurrentSemanticsConfig) -> None:
    """Guide 11.4: no confirmation seed may lie in any recorded or registered use, and the
    confirmation never reuses a development or G4 model seed."""
    g4 = config.g4
    used = [(name, first, last) for name, first, last, _ in AUDITED_SEED_RANGES]
    used += list(DIAGNOSTIC_SEED_RANGES) + list(dev.SPENT_DEVELOPMENT_SEEDS)
    used += list(SPENT_ISAAC_SEEDS) + [("g4 pilot training", *dev._PILOT_SEEDS)]
    used += [(f"development {label}", a, b) for label, a, b in _blocks(development, config)]
    taken = (set(g4.validation_seeds) | set(g4.test_seeds) | set(config.g3.pilot_seeds)
             | set(config.isaac_eval.seeds))
    models = plan.models
    if set(models.training_seeds) & (set(g4.model_seeds) | set(development.training_seeds)):
        raise ValueError("confirmation training seeds reuse a G4 or development model seed")
    if models.episodes_per_cell > 1000 or len(models.training_seeds) > 1000:
        raise ValueError("evaluation seed layout overflows its blocks")
    blocks = _blocks(models, config, plan.updates) + [("isaac", s, s) for s in plan.isaac.seeds]
    for label, low, high in blocks:
        for name, first, last in used:
            if low <= last and first <= high:
                raise ValueError(f"confirmation {label} seeds {low}-{high} overlap {name}")
        if any(low <= s <= high for s in taken):
            raise ValueError(f"confirmation {label} seeds {low}-{high} overlap a registered "
                             "seed")
    spans = sorted((a, b) for _, a, b in blocks)
    if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
        raise ValueError("confirmation seed blocks overlap each other")


def plan_hash(plan: ConfirmationPlan, config: CurrentSemanticsConfig) -> str:
    if plan.registered_hash is not None:
        return plan.registered_hash
    from ..direct_policy.contracts import digest
    manifest = contract_manifest(config)
    development = dev.load_development(dev.default_plan_path(config), config)
    return digest({"plan": plan.to_dict(),
                   "development_plan_hash": dev.plan_hash(development, config),
                   "contract": {k: manifest[k] for k in dev.BOUND_HASHES}})


def out_dir(root: Path, plan: ConfirmationPlan) -> Path:
    return Path(root) / plan.folder


def model_root(root: Path, plan: ConfirmationPlan, key: str) -> Path:
    """``<root>/<folder>/development/seed<s>/<arm>``: the layout
    ``isaac_development.checkpoint_path`` reads, so its flight code serves both stages."""
    seed, arm = key.split("/")
    return out_dir(root, plan) / "development" / f"seed{seed}" / arm


def model_keys(plan: ConfirmationPlan) -> tuple[str, ...]:
    return tuple(f"{s}/{arm}" for s in plan.models.training_seeds for arm in plan.models.arms)


def resolved(plan: ConfirmationPlan, config: CurrentSemanticsConfig) -> dict[str, Any]:
    g4 = config.g4
    n = len(model_keys(plan))
    return {"protocol": plan.protocol, "plan_hash": plan_hash(plan, config),
            "training_seeds": list(plan.models.training_seeds), "models": n,
            "updates": plan.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": n * plan.updates * g4.episodes_per_update,
            "validation_seeds": len(plan.models.validation_seeds),
            "episodes_per_cell": plan.models.episodes_per_cell,
            "evaluation_episodes": n * len(dev.CELLS) * plan.models.episodes_per_cell,
            "isaac_episodes": n * len(plan.isaac.seeds), "analysis": asdict(plan.analysis),
            "isaac_conditions": plan.isaac.to_dict()["conditions"]}


# ------------------------------------------------------------- registration

def blocker(config: CurrentSemanticsConfig, root: Path) -> str | None:
    """None when this root's development procedure is complete with a PASSED gate."""
    development = dev.load_development(dev.default_plan_path(config), config)
    problem = dev.check_registration(config, development, root)
    if problem:
        return problem
    path = Path(root) / dev.RESULT
    if not path.exists():
        return "the development stage has not finished (no development.json)"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("plan_hash") != dev.plan_hash(development, config):
        return "development.json belongs to another development plan"
    if not result.get("procedure_complete"):
        return "the development procedure is not complete"
    if not (result.get("verdict") or {}).get("development_gate_passed"):
        return "confirmation opens only after a PASSED development gate (guide 11.5)"
    return None


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    refused = {"stage": "register-confirmation", "status": "REFUSED"}
    problem = blocker(config, root)
    if problem:
        return 3, {**refused, "reason": problem}
    digest_ = plan_hash(plan, config)
    path = out_dir(root, plan) / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest_:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {**refused, "reason": f"{path} registers another plan "
                                        f"({old.get('plan_hash')}); archive it first"}
    if (out_dir(root, plan) / "development").exists():
        return 3, {**refused, "reason": "confirmation models exist without a registration"}
    manifest = contract_manifest(config)
    development = json.loads((Path(root) / dev.RESULT).read_text(encoding="utf-8"))
    return 0, {"schema_id": SCHEMA_ID, "stage": "register-confirmation",
               "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_file": str(plan_path), "plan": plan.to_dict(), **resolved(plan, config),
               "contract": {k: manifest[k] for k in dev.BOUND_HASHES},
               "development": {"plan_hash": development.get("plan_hash"),
                               "development_gate_passed": True},
               "seed_audit": "no confirmation seed lies in AUDITED_SEED_RANGES, "
                             "DIAGNOSTIC_SEED_RANGES, the G4 pilots' or the development's "
                             "seed blocks, or any registered seed (confirmation._audit)",
               "note": "registered before any confirmation model trained or flew"}


def check_registration(config: CurrentSemanticsConfig, plan: ConfirmationPlan,
                       root: Path) -> str | None:
    problem = blocker(config, root)
    if problem:
        return problem
    path = out_dir(root, plan) / REGISTRATION
    if not path.exists():
        return "no confirmation registration: run --stage register-confirmation first"
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("plan_hash") != plan_hash(plan, config):
        return ("the confirmation plan or the contract differs from the registration "
                f"({old.get('plan_hash')}); confirm refuses any override")
    return None


# ---------------------------------------------------------------- amendment

def load_amendment(path: str | Path, plan: ConfirmationPlan) -> dict[str, Any]:
    """An amendment narrows a registered plan before any evaluation exists: a subset of the
    arms (the proposed arm and Raw PPO always kept) and the decision rule over them.  Seeds,
    budget, cells, Isaac cases and the guards are the registered plan's."""
    data = dev._section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                        ("amendment", "reason", "arms", "analysis"), "amendment")
    arms = tuple(data["arms"])
    if not set(arms) <= set(plan.models.arms) or PROPOSED not in arms \
            or GUARD_REFERENCE not in arms or len(set(arms)) != len(arms):
        raise ValueError("amendment.arms must be distinct registered arms including "
                         f"{PROPOSED} and {GUARD_REFERENCE}")
    raw = dev._section(data["analysis"], tuple(Analysis.__dataclass_fields__),
                       "amendment.analysis")
    analysis = Analysis(
        primary_comparisons=tuple(raw["primary_comparisons"]),
        primary_family_alpha=_alpha(raw["primary_family_alpha"], "primary_family_alpha"),
        practical_comparison=raw["practical_comparison"],
        practical_alpha=_alpha(raw["practical_alpha"], "practical_alpha"),
        guard_alpha=_alpha(raw["guard_alpha"], "guard_alpha"),
        id_success_ci_low_min=float(raw["id_success_ci_low_min"]),
        four_ood_success_ci_low_min=float(raw["four_ood_success_ci_low_min"]),
        ood_unsafe_ci_high_max=float(raw["ood_unsafe_ci_high_max"]),
        risk_family_alpha=_alpha(raw["risk_family_alpha"], "risk_family_alpha"))
    compared = set(analysis.primary_comparisons) | ({analysis.practical_comparison} - {None})
    if not analysis.primary_comparisons or compared != set(arms) - {PROPOSED} \
            or analysis.practical_comparison in analysis.primary_comparisons:
        raise ValueError("amendment.analysis must compare the proposed arm with every other "
                         "kept arm exactly once")
    if (analysis.guard_alpha, analysis.id_success_ci_low_min,
            analysis.four_ood_success_ci_low_min, analysis.ood_unsafe_ci_high_max) != (
            plan.analysis.guard_alpha, plan.analysis.id_success_ci_low_min,
            plan.analysis.four_ood_success_ci_low_min, plan.analysis.ood_unsafe_ci_high_max):
        raise ValueError("an amendment may not move the registered guards")
    return {"amendment": data["amendment"], "reason": data["reason"], "arms": list(arms),
            "analysis": asdict(analysis)}


def amend(config_path: Path, plan_path: Path, amendment_path: Path,
          root: Path) -> tuple[int, dict[str, Any]]:
    """Register an amendment; refused once any model has been evaluated or any Isaac
    episode flown, so no amendment can follow a result."""
    from ..direct_policy.contracts import digest
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    refused = {"stage": "amend-confirmation", "status": "REFUSED"}
    problem = check_registration(config, plan, root)
    if problem:
        return 3, {**refused, "reason": problem}
    body = load_amendment(amendment_path, plan)
    path = out_dir(root, plan) / AMENDMENT
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("amendment_hash") == digest(body):
            return 0, {**old, "status": "ALREADY_AMENDED"}
        return 3, {**refused, "reason": f"{path} already amends this plan"}
    evaluated = sorted(str(p.parent.relative_to(out_dir(root, plan)))
                       for p in (out_dir(root, plan) / "development").glob("seed*/*/summary.json"))
    if evaluated or isaac_rows(root, plan):
        return 3, {**refused, "reason": "a model has already been evaluated or flown "
                                        f"({evaluated[:3]}); an amendment must precede results"}
    progress = {}
    for history in sorted((out_dir(root, plan) / "development").glob("seed*/*/history.jsonl")):
        rows = history.read_text(encoding="utf-8").splitlines()
        progress[str(history.parent.relative_to(out_dir(root, plan) / "development"))] = len(rows)
    return 0, {"schema_id": SCHEMA_ID, "stage": "amend-confirmation", "status": "AMENDED",
               "amended_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_hash": plan_hash(plan, config), "amendment_file": str(amendment_path),
               "amendment_hash": digest(body), **body,
               "evaluated_models_at_amendment": 0, "isaac_episodes_at_amendment": 0,
               "training_updates_at_amendment": progress,
               "note": ("no model had been evaluated and no Isaac episode flown when this "
                        "amendment was registered; only training-curriculum progress existed")}


def effective_plan(plan: ConfirmationPlan, config: CurrentSemanticsConfig,
                   root: Path) -> ConfirmationPlan:
    """The registered plan narrowed by its amendment, if one is registered.  Model summaries
    keep carrying the registered plan hash."""
    path = out_dir(root, plan) / AMENDMENT
    if not path.exists():
        return plan
    record = json.loads(path.read_text(encoding="utf-8"))
    registered = plan_hash(plan, config)
    if record.get("plan_hash") != registered:
        raise RuntimeError(f"{path} amends another plan ({record.get('plan_hash')})")
    analysis = Analysis(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in record["analysis"].items()})
    models = replace(plan.models, arms=tuple(a for a in plan.models.arms if a in record["arms"]))
    return replace(plan, models=models, analysis=analysis, registered_hash=registered)


# ---------------------------------------------------------------------- run

def shard_keys(plan: ConfirmationPlan, shard: tuple[int, int] | None) -> tuple[str, ...]:
    """Models of one shard: every n-th model of the seed-major order, starting at k.  With
    four shards each shard is one arm, so all shards finish a training seed together and the
    Isaac runner can fly it while the next seed trains."""
    if shard is None:
        return model_keys(plan)
    k, n = shard
    if not 0 <= k < n:
        raise ValueError("--shard must be K/N with 0 <= K < N")
    return tuple(key for i, key in enumerate(model_keys(plan)) if i % n == k)


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


def _summary(path: Path, digest: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("plan_hash") != digest or summary.get("complete") is not True:
        raise RuntimeError(f"{path} belongs to another run; archive it")
    return summary


def completed_models(plan: ConfirmationPlan, config: CurrentSemanticsConfig,
                     root: Path) -> dict[str, dict[str, Any]]:
    digest = plan_hash(plan, config)
    out = {}
    for key in model_keys(plan):
        summary = _summary(model_root(root, plan, key) / "summary.json", digest)
        if summary is not None:
            out[key] = summary
    return out


def run(config_path: Path, plan_path: Path, root: Path, *,
        log: Callable[[str], None] = print,
        shard: tuple[int, int] | None = None,
        workers: int | None = None, dynamic: bool = False) -> tuple[int, dict[str, Any]]:
    """Train, select and evaluate this shard's models (resumable per model).  ``workers``
    sizes this process's rollout pool only: every episode is seeded and the pool preserves
    order, so the result does not depend on it (the registered plan keeps its own value).
    ``dynamic`` lets any number of processes share the models: each claims the next model
    nobody holds (``claim``), in seed-major order, so seeds finish in order and no process
    idles while another policy's slower models remain."""
    from .models import CurrentActorCritic, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    problem = check_registration(config, plan, root)
    if problem:
        return 3, {"stage": "confirm", "status": "REFUSED", "reason": problem}
    plan = effective_plan(plan, config, root)
    digest = plan_hash(plan, config)
    models_plan = plan.models
    keys = shard_keys(plan, shard)
    index = {s: i for i, s in enumerate(models_plan.training_seeds)}
    started = time.monotonic()
    pending = [k for k in keys
               if _summary(model_root(root, plan, k) / "summary.json", digest) is None]
    mode = "dynamic" if dynamic else f"shard {shard or 'all'}"
    log(f"confirm: {mode}: {len(keys) - len(pending)} of {len(keys)} models already done")
    trained_here: list[str] = []
    if pending:
        with ProcessPoolExecutor(max_workers=int(workers or models_plan.workers),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(config_path),)) as pool:
            for key in pending:
                seed_text, arm = key.split("/")
                training_seed = int(seed_text)
                directory = model_root(root, plan, key)
                if _summary(directory / "summary.json", digest) is not None:
                    continue                      # finished by another process meanwhile
                if not claim(directory):
                    if dynamic:
                        continue                  # another live process holds it
                    raise RuntimeError(f"{key} is claimed by another live process")
                acfg = dev.arm_config(budget_config(config, plan), models_plan, arm)
                log(f"confirm: training {key}")
                dev._status(directory, "training")
                trained = train_seed(acfg, config_path, training_seed, directory, pool, log=log)
                best = directory / "best.pt"
                summary = {"plan_hash": digest, "complete": True, "training": trained,
                           "pool_workers": int(workers or models_plan.workers),
                           "validation": dev._validation_of(directory, trained["best_update"]),
                           "checkpoint": str(best) if best.exists() else None}
                if best.exists():
                    model = CurrentActorCritic(arm, acfg, seed=0)
                    load_checkpoint(best, model, manifest=contract_manifest(acfg))
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    log(f"confirm: evaluating {key} on {len(models_plan.cells)} cells")
                    summary["cells"] = dev.evaluate_model(
                        pool, models_plan, arm, state, index[training_seed],
                        progress=lambda name, i, root_=directory: dev._status(
                            root_, "evaluating", cell=name, cell_index=i,
                            cells=len(models_plan.cells)))
                else:
                    summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                            "timeout": 0.0, "statuses": {},
                                            "note": "no checkpoint reached difficulty 1.0"}
                                        for c in dev.CELLS}
                from .experiment import write_json
                write_json(directory / "summary.json", summary)
                release(directory)
                trained_here.append(key)
                dev._status(directory, "done")
                mean = float(np.mean([summary["cells"][c]["success"] for c in dev.CELLS]))
                log(f"confirm: {key} ten-cell success {mean:.3f}")
    return 0, {"stage": "confirm", "protocol": plan.protocol, "plan_hash": digest,
               "shard": list(shard) if shard else None, "dynamic": dynamic,
               "models": list(keys), "trained_here": trained_here,
               "wall_s": time.monotonic() - started}


# --------------------------------------------------------------- statistics

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


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise ``alpha``."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    rejected, m = {k: False for k in pvalues}, len(order)
    for i, key in enumerate(order):
        if pvalues[key] is None or pvalues[key] > alpha / (m - i):
            break
        rejected[key] = True
    return rejected


def _comparisons(plan: ConfirmationPlan, metric: dict[str, dict[int, float]]
                 ) -> dict[str, Any]:
    """Primary (Bonferroni) and practical intervals of R-GAT minus each arm on ``metric``
    (arm -> training seed -> value); the practical one opens after both primaries pass."""
    a = plan.analysis
    seeds = plan.models.training_seeds
    diff = lambda arm: [metric[PROPOSED][s] - metric[arm][s] for s in seeds]
    primary_conf = 1.0 - a.primary_family_alpha / len(a.primary_comparisons)
    primary = {}
    for arm in a.primary_comparisons:
        interval = paired_interval(diff(arm), primary_conf)
        primary[arm] = {**interval, "passed": interval["low"] is not None
                        and interval["low"] > 0.0}
    both = all(v["passed"] for v in primary.values())
    practical_out = {}
    if a.practical_comparison is not None:
        practical = paired_interval(diff(a.practical_comparison), 1.0 - a.practical_alpha)
        practical_out[a.practical_comparison] = {
            **practical, "opened": both,
            "passed": both and practical["low"] is not None and practical["low"] > 0.0}
    return {"primary": primary, "primary_passed": both, "practical": practical_out,
            "means": {arm: float(np.mean([metric[arm][s] for s in seeds])) for arm in metric}}


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def judge_local(plan: ConfirmationPlan, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Guide 11.5 "확인" on ``models[f"{seed}/{arm}"]`` = {validation, cells}."""
    a, mp = plan.analysis, plan.models
    seeds, arms = mp.training_seeds, mp.arms
    g = mp.gates

    def rate(seed, arm, cell, key="success"):
        return models[f"{seed}/{arm}"]["cells"][cell][key]

    def metric(cells, key="success"):
        return {arm: {s: _mean([rate(s, arm, c, key) for c in cells]) for s in seeds}
                for arm in arms}

    ten = metric(dev.CELLS)
    decision = _comparisons(plan, ten)
    guard_conf = 1.0 - a.guard_alpha
    raw = GUARD_REFERENCE
    guard_metric = {"id_success_vs_raw": metric(dev.ID_CELLS),
                    "four_ood_success_vs_raw": metric(dev.FOUR_OOD),
                    "ood_unsafe_vs_raw": metric(dev.OOD_CELLS, "unsafe")}
    guards = {}
    for name, m in guard_metric.items():
        interval = paired_interval([m[PROPOSED][s] - m[raw][s] for s in seeds], guard_conf)
        if name == "ood_unsafe_vs_raw":
            ok = interval["high"] is not None and interval["high"] <= a.ood_unsafe_ci_high_max
        else:
            bound = (a.id_success_ci_low_min if name == "id_success_vs_raw"
                     else a.four_ood_success_ci_low_min)
            ok = interval["low"] is not None and interval["low"] >= bound
        guards[name] = {**interval, "passed": bool(ok)}
    # Secondary, never decisive: the guide's original primary metric and per-cell means.
    secondary = {"composition_ood_success": _comparisons(plan, metric(("composition_ood",)))}
    validation = {}
    for key, model in models.items():
        v = model.get("validation") or {}
        det, smp = v.get("deterministic_success"), v.get("sampled_success")
        validation[key] = {"deterministic_success": det, "sampled_success": smp,
                           "checkpoint": model.get("checkpoint") is not None,
                           "passed": (det is not None and smp is not None
                                      and det >= g.validation_deterministic_success_min
                                      and smp >= g.validation_sampled_success_min)}
    passed = decision["primary_passed"] and all(x["passed"] for x in guards.values())
    return {"primary_metric": "ten_cell_mean_success", **decision, "guards": guards,
            "confirmation_passed": bool(passed),
            "practical_passed": (None if a.practical_comparison is None else bool(
                passed and decision["practical"][a.practical_comparison]["passed"])),
            "secondary": secondary,
            "learnability": {"validated": sum(v["passed"] for v in validation.values()),
                             "with_checkpoint": sum(v["checkpoint"]
                                                    for v in validation.values()),
                             "models": len(validation), "per_model": validation},
            "success_by_arm_and_cell": {arm: {c: _mean([rate(s, arm, c) for s in seeds])
                                              for c in dev.CELLS} for arm in arms},
            "unsafe_by_arm_and_cell": {arm: {c: _mean([rate(s, arm, c, "unsafe")
                                                       for s in seeds]) for c in dev.CELLS}
                                       for arm in arms},
            "ten_cell_success_by_seed": {arm: {str(s): ten[arm][s] for s in seeds}
                                         for arm in arms}}


def judge_isaac(plan: ConfirmationPlan, rows: list[dict[str, Any]],
                excluded: dict[str, str]) -> dict[str, Any]:
    """The same decision rule on each model's Isaac success rate over its registered cases
    (a model without a checkpoint scores zero and no risk, never dropped); risk is the
    Holm-adjusted secondary family."""
    a, mp = plan.analysis, plan.models
    seeds, arms = mp.training_seeds, mp.arms
    cases = len(plan.isaac.seeds)
    by: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(row["model"], []).append(row)

    def rate(key, test):
        own = by.get(key, [])
        if key in excluded:
            return 0.0
        if len(own) != cases:
            raise ValueError(f"{key} has {len(own)} of {cases} Isaac verdicts")
        return sum(test(r["status"]) for r in own) / cases

    success = {arm: {s: rate(f"{s}/{arm}", lambda st: st == "SUCCESS") for s in seeds}
               for arm in arms}
    risk = {arm: {s: rate(f"{s}/{arm}", lambda st: st in RISK_STATUSES) for s in seeds}
            for arm in arms}
    decision = _comparisons(plan, success)
    others = [arm for arm in arms if arm != PROPOSED]
    risk_intervals = {arm: paired_interval([risk[PROPOSED][s] - risk[arm][s] for s in seeds],
                                           0.95) for arm in others}
    rejected = holm({arm: v["p"] for arm, v in risk_intervals.items()}, a.risk_family_alpha)
    risk_family = {arm: {**v, "holm_rejected": rejected[arm],
                         "lower_risk": bool(rejected[arm] and v["mean"] < 0.0)}
                   for arm, v in risk_intervals.items()}
    statuses: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    for row in rows:
        counts = statuses[row["arm"]]
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    by_condition = {cell: {arm: _mean([
        sum(r["status"] == "SUCCESS" for r in by.get(f"{s}/{arm}", [])
            if r["condition"] == cell) / len(cell_seeds) for s in seeds])
        for arm in arms} for cell, cell_seeds in plan.isaac.conditions}
    return {"primary_metric": "isaac_success_rate", **decision,
            "isaac_passed": bool(decision["primary_passed"]),
            "practical_passed": (None if a.practical_comparison is None else bool(
                decision["primary_passed"]
                and decision["practical"][a.practical_comparison]["passed"])),
            "risk_family": risk_family,
            "risk_means": {arm: _mean([risk[arm][s] for s in seeds]) for arm in arms},
            "statuses_by_arm": statuses,
            "success_by_condition": by_condition,
            "flights_per_arm": {arm: sum(len(by.get(f"{s}/{arm}", [])) for s in seeds)
                                for arm in arms},
            "excluded_models": excluded,
            "success_by_seed": {arm: {str(s): success[arm][s] for s in seeds} for arm in arms},
            "risk_by_seed": {arm: {str(s): risk[arm][s] for s in seeds} for arm in arms}}


# -------------------------------------------------------------------- Isaac

def isaac_dir(root: Path, plan: ConfirmationPlan) -> Path:
    return out_dir(root, plan) / ISAAC_OUT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isaac_rows(root: Path, plan: ConfirmationPlan) -> list[dict[str, Any]]:
    stream = isaac_dir(root, plan) / "episodes.jsonl"
    if not stream.exists():
        return []
    return [json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ready_seeds(plan: ConfirmationPlan, config: CurrentSemanticsConfig,
                root: Path) -> list[int]:
    """Training seeds whose four models are complete (trained, selected, evaluated)."""
    done = completed_models(plan, config, root)
    return [s for s in plan.models.training_seeds
            if all(f"{s}/{arm}" in done for arm in plan.models.arms)]


def bind_checkpoints(plan: ConfirmationPlan, root: Path, seeds: list[int]
                     ) -> tuple[dict[str, Any], dict[str, str]]:
    """Record each model's checkpoint sha256 the first time its seed is flown and refuse a
    checkpoint that changed since; models without best.pt are excluded (zero success)."""
    from .experiment import write_json
    path = isaac_dir(root, plan) / CHECKPOINTS
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "checkpoints": {}, "excluded": {}}
    for seed in seeds:
        for arm in plan.models.arms:
            key = f"{seed}/{arm}"
            best = model_root(root, plan, key) / "best.pt"
            if not best.exists():
                record["excluded"].setdefault(
                    key, "no best.pt: the model selected no checkpoint (zero success)")
                continue
            digest = _sha256(best)
            known = record["checkpoints"].get(key)
            if known is None:
                record["checkpoints"][key] = {"path": str(best), "sha256": digest,
                                              "bound_utc": dt.datetime.now(
                                                  dt.timezone.utc).isoformat(timespec="seconds")}
            elif known["sha256"] != digest:
                raise RuntimeError(f"confirmation checkpoint {key} changed since it was bound")
    isaac_dir(root, plan).mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record["checkpoints"], record["excluded"]


def bind_models(plan: ConfirmationPlan, root: Path, keys: list[str]
                ) -> tuple[dict[str, Any], dict[str, str]]:
    """``bind_checkpoints`` for individual COMPLETED models (summary.json written, so their
    best.pt is final): a model can fly as soon as it is evaluated, without waiting for the
    other arms of its seed, and a model still training is never bound or excluded."""
    from .experiment import write_json
    path = isaac_dir(root, plan) / CHECKPOINTS
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "checkpoints": {}, "excluded": {}}
    for key in keys:
        directory = model_root(root, plan, key)
        if not (directory / "summary.json").exists():
            raise RuntimeError(f"{key} is not complete; only evaluated models are bound")
        best = directory / "best.pt"
        if not best.exists():
            record["excluded"].setdefault(
                key, "no best.pt: the model selected no checkpoint (zero success)")
            continue
        digest = _sha256(best)
        known = record["checkpoints"].get(key)
        if known is None:
            record["checkpoints"][key] = {"path": str(best), "sha256": digest,
                                          "bound_utc": dt.datetime.now(
                                              dt.timezone.utc).isoformat(timespec="seconds")}
        elif known["sha256"] != digest:
            raise RuntimeError(f"confirmation checkpoint {key} changed since it was bound")
    isaac_dir(root, plan).mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record["checkpoints"], record["excluded"]


def model_order(plan: ConfirmationPlan, keys: list[str]) -> list[tuple[str, int, str]]:
    """The registered cases of ``keys``, model by model, condition by condition."""
    return [(cell, env, key) for key in keys for cell, envs in plan.isaac.conditions
            for env in envs]


def seed_order(plan: ConfirmationPlan, seeds: list[int], excluded: dict[str, str]
               ) -> tuple[list[tuple[str, int, str]], tuple[str, ...]]:
    """(cases, model keys) of the given training seeds: seed by seed, and inside a seed
    condition by condition, environment-major (every arm meets an environment before the
    next one starts)."""
    order, keys = [], []
    for seed in seeds:
        own = tuple(f"{seed}/{arm}" for arm in plan.models.arms
                    if f"{seed}/{arm}" not in excluded)
        keys += own
        order += [(cell, env, key) for cell, envs in plan.isaac.conditions for env in envs
                  for key in own]
    return order, tuple(keys)


# ------------------------------------------------------------------- report

def report(config_path: Path, plan_path: Path, root: Path, *,
           local_reference: list[dict[str, Any]] | None = None,
           isaac_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    plan = effective_plan(load_plan(plan_path, config), config, root)
    models = completed_models(plan, config, root)
    keys = model_keys(plan)
    local_complete = len(models) == len(keys)
    rows = isaac_rows(root, plan)
    path = isaac_dir(root, plan) / CHECKPOINTS
    excluded = (json.loads(path.read_text(encoding="utf-8")).get("excluded", {})
                if path.exists() else {})
    expected = sum(len(plan.isaac.seeds) for k in keys if k not in excluded)
    flown = {r["model"] for r in rows}
    bound = flown | set(excluded)
    isaac_complete = (local_complete and len(rows) == expected
                      and all(k in bound for k in keys))
    paired = None
    if local_reference:
        local = {(r["condition"], r["seed"], r["model"]): r["status"] for r in local_reference}
        pairs = [(r["status"], local.get((r["condition"], r["seed"], r["model"])))
                 for r in rows]
        paired = {"cases": len(pairs),
                  "both_success": sum(a == "SUCCESS" and b == "SUCCESS" for a, b in pairs),
                  "isaac_only_success": sum(a == "SUCCESS" and b != "SUCCESS"
                                            for a, b in pairs),
                  "local_only_success": sum(a != "SUCCESS" and b == "SUCCESS"
                                            for a, b in pairs),
                  "both_fail": sum(a != "SUCCESS" and b != "SUCCESS" for a, b in pairs)}
    manifest = contract_manifest(config)
    return {"schema_id": SCHEMA_ID, "stage": "confirmation report", "protocol": plan.protocol,
            "plan_hash": plan_hash(plan, config),
            "hashes": {k: manifest[k] for k in dev.BOUND_HASHES},
            "local": {"complete": local_complete, "models_done": len(models),
                      "models": len(keys),
                      "verdict": judge_local(plan, models) if local_complete else None},
            "isaac": {"complete": isaac_complete, "episodes": len(rows),
                      "expected_episodes": expected, **(isaac_meta or {}),
                      "verdict": (judge_isaac(plan, rows, excluded) if isaac_complete
                                  else None),
                      "paired_with_local": paired},
            "updates_per_model": plan.updates,
            "note": (f"{plan.protocol}: fresh seeds, decision rule registered before training "
                     "(confirmation_registration.json); a failed comparison reads 'not "
                     "confirmed', never 'equivalent' (guide 13)")}


def local_reference(config_path: Path, plan: ConfirmationPlan, root: Path,
                    keys: tuple[str, ...], workers: int = 8) -> list[dict[str, Any]]:
    """The registered Isaac cases of ``keys`` on the calibrated local plant (the development
    cells' own code path), so every Isaac verdict has a paired local one."""
    from .pilot import _init_worker
    config = budget_config(load_config(config_path), plan)
    plan = effective_plan(plan, load_config(config_path), root)
    models = idev.load_models(config, plan.models, out_dir(root, plan), keys)
    tasks, index = [], []
    for cell_name, seed, key in idev.episode_order(plan.isaac, tuple(models)):
        state = {k: v.detach().clone() for k, v in models[key][0].state_dict().items()}
        tasks.append((key.split("/")[1], state, seed, asdict(plan.models.cell(cell_name)),
                      plan.models.dropout_start_s, seed + 9_000_000))
        index.append((cell_name, seed, key))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        episodes = list(pool.map(dev._cell_work, tasks, chunksize=4))
    return [{"condition": c, "seed": int(s), "model": k, "arm": k.split("/")[1],
             "training_seed": int(k.split("/")[0]), "status": e.get("status")}
            for (c, s, k), e in zip(index, episodes)]
