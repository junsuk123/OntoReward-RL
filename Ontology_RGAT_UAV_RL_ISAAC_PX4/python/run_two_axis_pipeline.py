#!/usr/bin/env python3
"""End-to-end pipeline for the primary two-axis comparison.

Stages
------
``train``     PPO from a random initialisation for every (arm, seed).
``evaluate``  Selected checkpoints on the held-out validation and test splits.
``aggregate`` Cross-seed tables, paired per-seed differences, fairness report.
``all``       All three, in order.

The environment/reward/evaluation are common. v2.8 graph pretraining, staged
adaptation and selection margin are declared extra factors. Aggregation derives
fairness from recorded signatures, seeds and eligibility, not launcher claims.

No behaviour cloning, no teacher, no PN supervision, no privileged input.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from ontology_rgat.two_axis.config import load_config, REFERENCE_CONFIG_PATH
from ontology_rgat.two_axis.artifacts import json_text
from ontology_rgat.two_axis.models import POLICY_MODES
from ontology_rgat.two_axis.training import (PPOHyperparameters, SEARCH_SPACE,
                                             TEST_SEEDS, TUNING_SEEDS,
                                             VALIDATION_SEEDS, evaluate_policy,
                                             load_agent,
                                             measure_inference_time_s,
                                             train_arm, tuning_score)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/two_axis_reference_v28_active_port2"


def _run_dir(output: Path, arm: str, seed: int) -> Path:
    return Path(output) / "runs" / f"{arm}__seed{seed}"


def _train_one(payload: dict) -> dict:
    """Worker entry point; threads are capped so parallel arms do not thrash."""
    import torch
    torch.set_num_threads(int(payload["threads"]))
    hyper = PPOHyperparameters(**payload["hyper"])
    return train_arm(payload["arm"], seed=payload["seed"], hyper=hyper,
                     output_dir=Path(payload["output_dir"]),
                     config=payload["config"],
                     progress_every=payload["progress_every"])


def _tune_one(payload: dict) -> dict:
    import torch
    torch.set_num_threads(int(payload["threads"]))
    hyper = PPOHyperparameters(**payload["hyper"])
    summary = train_arm(payload["arm"], seed=payload["seed"], hyper=hyper,
                        output_dir=Path(payload["output_dir"]),
                        config=payload["config"],
                        progress_every=10 ** 9)
    agent, _ = load_agent(Path(payload["output_dir"]) / "checkpoint_final.pt",
                          payload["config"])
    probe = evaluate_policy(agent, payload["config"], seeds=TUNING_SEEDS,
                            difficulty=1.0)
    return {"arm": payload["arm"], "trial": payload["trial"],
            "overrides": payload["overrides"], "probe": probe,
            "final_difficulty": summary["final_difficulty"],
            "training_rates_last_400": summary["training_rates_last_400"],
            "score": tuning_score(summary, probe)}


def stage_tune(output: Path, hyper: PPOHyperparameters, *, workers: int,
               threads: int, seed: int, config=None) -> dict:
    """Same grid, same budget, same scoring rule for every arm."""
    jobs = []
    for arm in POLICY_MODES:
        for trial, overrides in enumerate(SEARCH_SPACE):
            merged = {**asdict(hyper), **overrides}
            jobs.append({
                "arm": arm, "trial": trial, "overrides": overrides,
                "seed": seed, "hyper": merged, "threads": threads,
                "config": config or load_config(),
                "output_dir": str(Path(output) / "tuning"
                                  / f"{arm}__trial{trial}")})
    print(f"tuning: {len(SEARCH_SPACE)} identical trials x {len(POLICY_MODES)} "
          f"arms = {len(jobs)} short runs of {hyper.iterations} iterations",
          flush=True)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_tune_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"[tune] {row['arm']} trial={row['trial']} "
                  f"score={row['score']:.3f} "
                  f"d={row['final_difficulty']:.2f} "
                  f"probe_land={row['probe']['landing_rate']:.3f} "
                  f"({len(results)}/{len(jobs)})", flush=True)
    best: dict[str, dict] = {}
    for arm in POLICY_MODES:
        subset = [r for r in results if r["arm"] == arm]
        winner = max(subset, key=lambda r: (r["score"], -r["trial"]))
        best[arm] = {"trial": winner["trial"], "overrides": winner["overrides"],
                     "score": winner["score"], "probe": winner["probe"]}
        print(f"[tune:best] {arm} -> trial {winner['trial']} "
              f"{winner['overrides']} score={winner['score']:.3f}", flush=True)
    # One pooled winner for the matched-hyperparameter comparison. Per-arm
    # winners remain diagnostic, never silently change the primary experiment.
    pooled = [{"trial": trial, "overrides": overrides,
               "score": float(np.mean([r["score"] for r in results
                                        if r["trial"] == trial]))}
              for trial, overrides in enumerate(SEARCH_SPACE)]
    common = max(pooled, key=lambda r: (r["score"], -r["trial"]))
    payload = {"search_space": list(SEARCH_SPACE), "trials": sorted(
                   results, key=lambda r: (r["arm"], r["trial"])),
               "best_common": common, "pooled_trials": pooled,
               "tie_break": "smallest_trial_index",
               "best_per_arm": best, "tuning_seeds": TUNING_SEEDS,
               "budget_per_arm": len(SEARCH_SPACE),
               "note": ("identical grid, identical trial count and identical "
                        "scoring for every arm; scored on tuning seeds that "
                        "appear in neither validation nor test")}
    (Path(output) / "tuning.json").write_text(
        json_text(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def stage_train(output: Path, seeds: list[int], hyper: PPOHyperparameters, *,
                workers: int, threads: int, progress_every: int,
                per_arm: dict[str, dict] | None = None, config=None) -> list[dict]:
    jobs = [{"arm": arm, "seed": seed,
             "hyper": {**asdict(hyper), **((per_arm or {}).get(arm, {}))},
             "output_dir": str(_run_dir(output, arm, seed)),
             "threads": threads, "progress_every": progress_every,
             "config": config or load_config()}
            for seed in seeds for arm in POLICY_MODES]
    print(f"training {len(jobs)} runs "
          f"({len(POLICY_MODES)} arms x {len(seeds)} seeds), "
          f"{workers} workers x {threads} threads", flush=True)
    summaries: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_train_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            summary = future.result()
            summaries.append(summary)
            print(f"[done] {job['arm']} seed={job['seed']} "
                  f"{summary['training_time_s']/60:.1f} min "
                  f"difficulty={summary['final_difficulty']:.2f} "
                  f"({len(summaries)}/{len(jobs)})", flush=True)
    print(f"training wall clock {(time.perf_counter()-started)/60:.1f} min",
          flush=True)
    return summaries


def stage_evaluate(output: Path, seeds: list[int], config=None) -> list[dict]:
    """Score the selected checkpoint of every run on both held-out splits."""
    config = config or load_config()
    if (Path(output)/"evaluation.json").exists():
        raise FileExistsError("evaluation already exists; preserve the locked held-out result")
    rows: list[dict] = []
    for seed in seeds:
        for arm in POLICY_MODES:
            directory = _run_dir(output, arm, seed)
            summary = json.loads((directory / "summary.json").read_text())
            selected = summary.get("selected_checkpoint")
            # Only a checkpoint that reached difficulty 1.0 is selectable; if
            # none did, the run is reported as ineligible rather than scored
            # with a policy that never saw the nominal task.
            name = selected.get("checkpoint", "checkpoint_best.pt") if selected else "checkpoint_final.pt"
            if name not in {"checkpoint_best.pt", "checkpoint_final.pt", "checkpoint_relational.pt"}:
                raise ValueError(f"unexpected selected checkpoint: {name}")
            path = directory / name
            agent, payload = load_agent(path, config)
            row = {
                "arm": arm, "seed": seed,
                "checkpoint": path.name,
                "checkpoint_eligible": bool(selected is not None),
                "selected_iteration": (selected or {}).get("iteration"),
                "curriculum_difficulty_at_selection":
                    payload["curriculum_difficulty"],
                "reached_nominal": summary["reached_nominal"],
                "training_time_s": summary["training_time_s"],
                "environment_steps": summary["environment_steps"],
                "total_environment_steps_including_pretraining": summary.get(
                    "total_environment_steps_including_pretraining", summary["environment_steps"]),
                "relation_activation": summary.get("relation_activation", {}),
                "parameter_count": summary["parameter_count"]["total"],
                "training_rates": summary["training_rates_overall"],
                "signature": summary["signature"],
                "hyperparameters": summary["hyperparameters"],
                "experiment": config.experiment,
                "validation_seeds": VALIDATION_SEEDS,
                "test_seeds": TEST_SEEDS,
                "comparison_factors": summary.get("comparison_factors", ["unrecorded"]),
                "inference_time_s": measure_inference_time_s(agent, config),
                "validation": evaluate_policy(
                    agent, config, seeds=VALIDATION_SEEDS, difficulty=1.0),
                "test": (evaluate_policy(agent, config, seeds=TEST_SEEDS, difficulty=1.0)
                         if selected else None),
            }
            rows.append(row)
            if config.ontology.schema == "compact_context_graph_v3_grouped":
                from ontology_rgat.two_axis.environment import TwoAxisLandingEnv
                from ontology_rgat.two_axis.models_v28 import relational_contribution
                probe,_ = TwoAxisLandingEnv(config).reset(seed=VALIDATION_SEEDS[0])
                row["relation_output_probe"] = relational_contribution(agent,*agent.tensors(probe))
            print(f"[eval] {arm} seed={seed} "
                  f"val={row['validation']['landing_rate']:.3f} "
                  f"test={row['test']['landing_rate'] if row['test'] else 'ineligible'} "
                  f"eligible={row['checkpoint_eligible']}", flush=True)
    (Path(output) / "evaluation.json").write_text(
        json_text(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def fairness_report(rows: list[dict]) -> dict:
    """Re-derive the single-factor claim from what the runs recorded."""
    def unique(key):
        return {json.dumps(row[key], sort_keys=True) for row in rows}

    signatures = unique("signature")
    hyper = unique("hyperparameters")
    report = {
        "identical_experiment_signature": len(signatures) == 1,
        "identical_hyperparameters": len(hyper) == 1,
        "identical_validation_seeds": bool(rows) and all("validation_seeds" in r for r in rows)
            and len({tuple(r["validation_seeds"]) for r in rows}) == 1,
        "identical_test_seeds": bool(rows) and all("test_seeds" in r for r in rows)
            and len({tuple(r["test_seeds"]) for r in rows}) == 1,
        "all_checkpoints_eligible": bool(rows) and all(r.get("checkpoint_eligible", False) for r in rows),
        "complete_arm_seed_matrix": bool(rows) and len(rows) == len({r["seed"] for r in rows})*len(POLICY_MODES)
            and len({(r["arm"], r["seed"]) for r in rows}) == len(rows)
            and {r["arm"] for r in rows} == set(POLICY_MODES),
        "evaluation_difficulty": 1.0,
        "arms": sorted({row["arm"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "behaviour_cloning": False,
        "teacher_or_pn_supervision": False,
        "declared_factors": sorted({factor for row in rows for factor in
                                    row.get("comparison_factors", ["unrecorded"])}),
    }
    report["passes"] = bool(report["identical_experiment_signature"]
                            and report["identical_hyperparameters"]
                            and report["identical_validation_seeds"]
                            and report["identical_test_seeds"]
                            and report["all_checkpoints_eligible"]
                            and report["complete_arm_seed_matrix"])
    report["single_factor_claim_allowed"] = report["passes"] and report["declared_factors"] == ["state_representation"]
    return report


def stage_aggregate(output: Path) -> dict:
    rows = json.loads((Path(output) / "evaluation.json").read_text())
    if not fairness_report(rows)["complete_arm_seed_matrix"]:
        raise ValueError("aggregation requires a complete, unique three-arm seed matrix")
    metrics = ("landing_rate", "safe_abort_rate", "task_timeout_rate",
               "unsafe_rate", "fov_capture_rate",
               "supervisor_intervention_rate", "mean_landing_time_s")
    table: dict[str, dict] = {}
    for arm in POLICY_MODES:
        subset = [row for row in rows if row["arm"] == arm]
        entry: dict = {
            "seeds": [row["seed"] for row in subset],
            "parameter_count": subset[0]["parameter_count"],
            "eligible_checkpoints": sum(row["checkpoint_eligible"] for row in subset),
            "reached_nominal": sum(row["reached_nominal"] for row in subset),
            "training_time_s_mean": float(np.mean(
                [row["training_time_s"] for row in subset])),
            "inference_time_s_mean": float(np.mean(
                [row["inference_time_s"] for row in subset])),
        }
        for split in ("validation", "test"):
            for metric in metrics:
                values = np.array([row[split][metric] if row[split] is not None else np.nan for row in subset],
                                  dtype=float)
                finite = values[np.isfinite(values)]
                entry[f"{split}_{metric}_mean"] = (
                    float(finite.mean()) if finite.size else float("nan"))
                entry[f"{split}_{metric}_std"] = (
                    float(finite.std(ddof=1)) if finite.size > 1 else 0.0)
        table[arm] = entry
    # Paired per-seed differences against the canonical vector baseline: the
    # seed is shared, so the pairing removes seed variance from the comparison.
    baseline = "ppo_vector_canonical"
    paired: dict[str, dict] = {}
    for arm in POLICY_MODES:
        if arm == baseline:
            continue
        differences = []
        for seed in sorted({row["seed"] for row in rows}):
            a = next(r for r in rows if r["arm"] == arm and r["seed"] == seed)
            b = next(r for r in rows if r["arm"] == baseline and r["seed"] == seed)
            differences.append(a["test"]["landing_rate"] - b["test"]["landing_rate"]
                               if a["test"] and b["test"] else float("nan"))
        values = np.array(differences, dtype=float)
        paired[f"{arm}_minus_{baseline}"] = {
            "per_seed": differences,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    payload = {
        "experiment": rows[0].get("experiment", "unrecorded"),
        "arms": table,
        "paired_test_landing_rate_differences": paired,
        "fairness": fairness_report(rows),
        "claim": "Report only measured outcomes; consult fairness and declared factors before comparing arms.",
    }
    (Path(output) / "aggregate.json").write_text(
        json_text(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REFERENCE_CONFIG_PATH)
    parser.add_argument("--stage",
                        choices=("tune", "train", "evaluate", "aggregate", "all"),
                        required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--decisions", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluation-every", type=int, default=50)
    # Checkpoint selection ranks on this validation score. At 12 episodes a
    # landing rate carries roughly +/-14 points of standard error, which is
    # wider than the differences between arms, so the selector picks noise.
    parser.add_argument("--evaluation-episodes", type=int,
                        default=len(VALIDATION_SEEDS))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tune-iterations", type=int, default=250)
    parser.add_argument("--tune-seed", type=int, default=11)
    parser.add_argument("--use-tuned", action="store_true",
                        help="apply tuning.json's common pooled winner to all arms")
    args = parser.parse_args()
    config = load_config(args.config)
    if min(args.workers, args.threads, args.progress_every) <= 0 or any(s < 0 for s in args.seeds):
        parser.error("workers, threads and progress interval must be positive; seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds would overwrite the same run directory")

    hyper = PPOHyperparameters(iterations=args.iterations,
                               decisions_per_iteration=args.decisions,
                               evaluation_every=args.evaluation_every,
                               evaluation_episodes=args.evaluation_episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "arms": list(POLICY_MODES), "seeds": args.seeds,
        "hyperparameters": asdict(hyper),
        "config": config.canonical_dict, "config_sha256": config.sha256,
        "config_path": str(args.config.resolve()),
        "validation_seeds": VALIDATION_SEEDS, "test_seeds": TEST_SEEDS,
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous.get("config_sha256") != config.sha256 or previous.get("seeds") != args.seeds:
            parser.error("output belongs to a different/unversioned experiment; choose a fresh --output")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    if args.stage in ("tune", "all"):
        stage_tune(args.output,
                   PPOHyperparameters(iterations=args.tune_iterations,
                                      decisions_per_iteration=args.decisions,
                                      evaluation_every=10 ** 9),
                   workers=args.workers, threads=args.threads,
                   seed=args.tune_seed, config=config)
    per_arm = None
    tuning_path = args.output / "tuning.json"
    if (args.use_tuned or args.stage == "all") and tuning_path.exists():
        tuning = json.loads(tuning_path.read_text())
        if "best_common" not in tuning:
            parser.error("tuning artifact predates common deterministic selection; rerun tune")
        per_arm = {arm: tuning["best_common"]["overrides"] for arm in POLICY_MODES}
        print(f"applying common tuned hyperparameters: {tuning['best_common']}", flush=True)
    if args.stage in ("train", "all"):
        stage_train(args.output, args.seeds, hyper, workers=args.workers,
                    threads=args.threads, progress_every=args.progress_every,
                    per_arm=per_arm, config=config)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(args.output, args.seeds, config=config)
    if args.stage in ("aggregate", "all"):
        payload = stage_aggregate(args.output)
        print(json.dumps(payload["fairness"], indent=2))
        for arm, row in payload["arms"].items():
            print(f"{arm:<22} test_landing="
                  f"{row['test_landing_rate_mean']:.3f}"
                  f"+-{row['test_landing_rate_std']:.3f} "
                  f"val={row['validation_landing_rate_mean']:.3f} "
                  f"params={row['parameter_count']}")
        if not payload["fairness"]["passes"]:
            print("Acceptance failed; see aggregate.json fairness diagnostics.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
