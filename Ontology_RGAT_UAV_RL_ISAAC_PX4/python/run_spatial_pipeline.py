#!/usr/bin/env python3
"""Explicit spatial causal CLI: train, reload/evaluate, actual Isaac evaluation.

Bare root launcher remains read-only. Isaac execution is opt-in, owns only
processes it starts, and cannot silently adopt a different/live experiment.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import signal
import multiprocessing
import time
import torch

from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.environment import SpatialLandingEnv, IsaacBackend
from ontology_rgat.spatial.training import train_arm, load_agent, evaluate, TEST_SEEDS
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.validation import actual_acceptance
from ontology_rgat.spatial.lifecycle import used_reset_recoveries
from ontology_rgat.two_axis.models import POLICY_MODES
from ontology_rgat.two_axis.training import PPOHyperparameters
from ontology_rgat.two_axis.artifacts import json_text
from ontology_rgat.config import default_config
from ontology_rgat.stack import ExternalStack, current as current_stack

ROOT = Path(__file__).resolve().parents[1]


def isaac_test_seeds(start, episodes):
    """Explicit paired holdout range, separate from validation seeds 2000/2001."""
    if not 10000 <= start < 2**31 or not 1 <= episodes <= 1000:
        raise ValueError("Isaac test start must be >=10000 and episodes in [1,1000]")
    if start + episodes > 2**31:
        raise ValueError("Isaac test seed range exceeds signed 32-bit identifiers")
    return tuple(range(start, start + episodes))


def train_local_job(job):
    torch.set_num_threads(1)
    return train_arm(**job)


def wait_checkpoint_summary(directory, wait_seconds=0):
    """Wait only for a completed training summary, never deploy progress files."""
    path = directory / "summary.json"
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"completed training summary unavailable: {path}") from exc
            time.sleep(0.5)


@contextmanager
def incomplete_isaac_report(output):
    """Never leave an old pass/missing report after a failed actual run."""
    path = output / "isaac_acceptance.json"
    status = dict(complete_matrix=False, no_unsafe_outcomes=None,
                  landing_observed=None, passes=False, state="running",
                  backend="actual-isaac-px4", performance_parity_claim=False)
    path.write_text(json_text(status, indent=2) + "\n")
    try:
        yield
    except BaseException as exc:
        status.update(state="incomplete", error_type=type(exc).__name__, error=str(exc))
        path.write_text(json_text(status, indent=2) + "\n")
        raise


@contextmanager
def live_stack(output, *, adopt=False, headless=False, schema="spatial-causal-rgat/5",
               reset_recoveries=0, isolate_episodes=False):
    if isolate_episodes and adopt:
        raise ValueError('fresh-stack episodes require ownership, never adoption')
    if not 0 <= reset_recoveries <= 5:
        raise ValueError('reset recovery budget must be in [0,5]')
    profile = Path(deployment_profile(schema)["path"])
    with open("/tmp/ontology_rgat_flight_pipeline.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another flight pipeline owns the stack")
        # Inspect program tokens, never shell/search text containing names.
        active = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                tokens = (entry / "cmdline").read_bytes().decode().split("\0")
            except (OSError, UnicodeError):
                continue
            if any(Path(t).name == "landing_world.py" for t in tokens[:3]):
                active.append(tokens)
        if active and not adopt:
            raise RuntimeError(
                "Isaac already active; refusing takeover without --adopt-stack"
            )
        if active and not all(str(profile) in tokens for tokens in active):
            raise RuntimeError("active Isaac uses a different profile")
        check = subprocess.run(
            [str(ROOT / "scripts/sync_gateway.sh"), "--check"],
            capture_output=True,
            text=True,
        )
        if check.returncode:
            if active:
                raise RuntimeError(
                    "active gateway source is stale; stop owned stack before rebuild"
                )
            subprocess.run([str(ROOT / "scripts/sync_gateway.sh")], check=True)
        stack = ExternalStack(
            default_config("quick"),
            config_path=profile,
            headless=headless,
            viewport_pair_index=0,
            startup_attempts=2,
            log_dir=output / "stack",
        )
        stack.spatial_recovery_budget = int(reset_recoveries)
        stack.spatial_recoveries_used = used_reset_recoveries(output)
        stack.spatial_isolate_episodes=bool(isolate_episodes)
        stack.spatial_episode_count=0
        stack.spatial_last_stop_confirmed=True
        previous_stack = current_stack()
        try:
            stack.start()
            current_stack(stack)
            yield stack
        finally:
            try:
                stack.stop()
            finally:
                current_stack(previous_stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "isaac", "all"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[811])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--training-backend",
        choices=["local", "isaac"],
        default="local",
        help="Isaac collects actual PX4 rollouts and validates on the same backend",
    )
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--episodes-per-iteration", type=int, default=0,
                        help="Opt-in complete-episode batches; replaces the decision budget")
    parser.add_argument("--curriculum-loss-timeout-start", type=float, default=None,
                        help="TRAIN ONLY: reference starts at 12 s, converges to nominal 3 s")
    parser.add_argument("--evaluation-every", type=int, default=None)
    parser.add_argument("--value-warmup-iterations", type=int, default=None,
                        help="Explicit common critic-only warmup override (preset default is retained if omitted)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Common PPO credit-assignment trace, recorded for every arm")
    parser.add_argument("--nominal-only", action="store_true",
                        help="Disable training curriculum; evaluation is always nominal")
    parser.add_argument(
        "--ppo-preset",
        choices=["python-default", "reference-v28-scratch", "reference-v28-episodic"],
        default="python-default",
        help="Explicit common optimizer/exploration settings; recorded per run",
    )
    parser.add_argument("--horizon", type=float, default=70.0)
    parser.add_argument("--contract-version",
                        choices=["3", "4", "5", "6", "7", "8", "9", "10",
                                 "spatial-reference/1", "spatial-reference/2",
                                 "spatial-reference/3", "spatial-reference/4",
                                 "spatial-reference/5", "spatial-reference/6",
                                 "spatial-reference/7", "reference"],
                        default="reference")
    parser.add_argument("--initialize-v4-weights", action="store_true",
                        help="Explicit v4 to v5 PPO transfer; eligibility is NOT transferred")
    # The proposed arm differs from ppo_semantic_flat by exactly one term, and
    # that term is frozen with a zero-initialised readout for the first
    # `adaptation_warmup_fraction` of the budget. At the shipped 0.90 it is
    # trainable for the last 10 % of iterations, and a run that stops early
    # never activates it at all -- measured 2026-10-05, the two arms were
    # numerically identical for 8607/8607 iterations of spatial_long_nominal.
    # Exposed so the staged schedule can be swept instead of edited.
    parser.add_argument("--adaptation-warmup-fraction", type=float, default=None,
                        help="override ontology.adaptation_warmup_fraction; "
                             "lower gives the relational path more of the budget")
    parser.add_argument("--activation-iterations", type=int, default=2)
    parser.add_argument("--activation-decisions", type=int, default=128)
    parser.add_argument("--isaac-episodes", type=int, default=1)
    parser.add_argument("--isaac-seed-start", type=int, default=12000,
                        help="Paired actual holdout seed range; use a fresh range for confirmation")
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=0.,
                        help="Bounded wait for completed training summaries; never uses progress")
    parser.add_argument("--adopt-stack", action="store_true")
    parser.add_argument("--fresh-stack-per-episode", action="store_true",
                        help="Owned Isaac only: cold PX4/EKF state after every confirmed stop")
    parser.add_argument("--reset-recoveries", type=int, choices=range(6), default=0,
                        help="Opt-in total owned-stack arming-refusal restarts; never replaces adopted Isaac")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent local training processes only; Isaac is serialized",
    )
    parser.add_argument(
        "--curriculum-angular-scales",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("TILT", "RATE"),
        help="TRAIN-ONLY angular tolerance ramps; nominal/test/Isaac unchanged",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Fine-tune eligible same-contract checkpoints in a NEW output",
    )
    # PPO regresses on the action the policy PROPOSED. The supervisor replaces
    # axes independently, so a per-step flag is the wrong granularity; see
    # tests/test_per_axis_intervention_masking.py. Opt-in: every recorded
    # result predates it.
    parser.add_argument(
        "--intervention-masking",
        choices=["none", "per_axis"],
        default="none",
        help="drop supervisor-replaced AXES from the actor objective",
    )
    # A clone whose MEAN lands 95.8 % lands 0.0 % sampled at the shipped
    # sigma 0.333, so no PPO batch contains a success to learn from.
    parser.add_argument(
        "--final-log-std", type=float, default=None,
        help="anneal the exploration ceiling down to this log_std",
    )
    parser.add_argument(
        "--log-std-anneal-fraction", type=float, default=0.5,
        help="fraction of the run spent annealing before the floor is held",
    )
    parser.add_argument(
        "--state-dependent-log-std", action="store_true",
        help="let the actor set exploration per state instead of one global sigma",
    )
    # The shipped early stop checks target_kl one minibatch late, so the first
    # actor step of every iteration is unconstrained; at small sigma that one
    # step is several sigmas wide (PPOHyperparameters.enforce_target_kl).
    parser.add_argument(
        "--enforce-target-kl", action="store_true",
        help="revert any actor step whose measured KL exceeds the target and "
             "adapt the actor learning rate to the trust region",
    )
    args = parser.parse_args()
    if args.value_warmup_iterations is not None and args.value_warmup_iterations < 0:
        parser.error("value warmup iterations must be nonnegative")
    try:
        actual_seeds = isaac_test_seeds(args.isaac_seed_start, args.isaac_episodes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.fresh_stack_per_episode and args.adopt_stack:
        parser.error('fresh-stack episodes cannot adopt an operator-owned stack')
    if args.curriculum_loss_timeout_start is not None and (
        not 3. <= args.curriculum_loss_timeout_start <= 30.
        or args.training_backend != "local" or args.nominal_only):
        parser.error("loss-timeout curriculum requires local curriculum training and [3,30] s")
    if not 0 <= args.episodes_per_iteration <= 64:
        parser.error("episodes per iteration must be in [0,64]")
    if args.ppo_preset == "reference-v28-episodic" and not args.episodes_per_iteration:
        parser.error("reference episodic preset requires --episodes-per-iteration")
    if not 0 <= args.gae_lambda <= 1:
        parser.error("GAE lambda must be in [0,1]")
    if not 0 <= args.checkpoint_wait_seconds <= 3600:
        parser.error("checkpoint wait must be in [0,3600] seconds")
    if args.initialize_v4_weights and (
        not args.initialize_from or args.contract_version != "5"):
        parser.error("v4 weights require --initialize-from and contract version 5")
    if args.workers < 1 or (args.training_backend == "isaac" and args.workers != 1):
        parser.error("workers must be positive; actual Isaac uses exactly one worker")
    if any(not 1 <= scale <= 4 for scale in args.curriculum_angular_scales):
        parser.error("angular curriculum scales must be in [1,4]")
    if args.training_backend == "isaac" and args.curriculum_angular_scales != [
        1.0,
        1.0,
    ]:
        parser.error("angular curriculum is local training only")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or args.isaac_episodes < 1
    ):
        parser.error("unique nonnegative seeds and positive episode budget required")
    torch.set_num_threads(1)
    deployment = deployment_profile(schema_for(args.contract_version))
    cfg = replace(
        SpatialConfig(),
        schema=schema_for(args.contract_version),
        horizon=args.horizon,
        isaac_profile_sha256=deployment["sha256"],
    )
    if args.adaptation_warmup_fraction is not None:
        if not 0.0 <= args.adaptation_warmup_fraction < 1.0:
            parser.error("adaptation warmup fraction must be in [0,1)")
        cfg = replace(cfg, ontology=replace(
            cfg.ontology,
            adaptation_warmup_fraction=args.adaptation_warmup_fraction))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "signature": cfg.signature,
        "config": asdict(cfg),
        "training_seeds": args.seeds,
        "deployment": deployment,
        "training_backend": args.training_backend,
        "fresh_stack_per_episode": args.fresh_stack_per_episode,
        "ppo_preset": args.ppo_preset,
        "episodes_per_iteration": args.episodes_per_iteration,
        "curriculum_loss_timeout_start": args.curriculum_loss_timeout_start,
        "curriculum_angular_scales": args.curriculum_angular_scales,
        "workers": args.workers,
        "initialize_from": str(args.initialize_from.resolve())
        if args.initialize_from
        else None,
        "intervention_masking": args.intervention_masking,
        "final_log_std": args.final_log_std,
        "log_std_anneal_fraction": args.log_std_anneal_fraction,
        "state_dependent_log_std": args.state_dependent_log_std,
        "nominal_only": args.nominal_only,
        "gae_lambda": args.gae_lambda,
        "value_warmup_iterations_override": args.value_warmup_iterations,
        "initialize_v4_weights": args.initialize_v4_weights,
        "publication_claim_allowed": False,
        "note": "spatial integration profile, not reference performance parity",
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if (
            previous["signature"] != cfg.signature
            or previous["training_seeds"] != args.seeds
        ):
            parser.error(
                "output belongs to a different spatial contract or seed matrix"
            )
        if (
            args.stage in ("train", "all")
            and previous["training_backend"] != args.training_backend
        ):
            parser.error("output belongs to a different training backend")
    else:
        plan_path.write_text(json_text(plan, indent=2) + "\n")
    if args.stage in ("train", "all"):
        preset = (
            {}
            if args.ppo_preset == "python-default"
            else dict(
                epochs=8,
                actor_lr=5e-4,
                entropy_coefficient=0.0025,
                initial_log_std=-1.1,
                value_warmup_iterations=2,
            )
        )
        if args.value_warmup_iterations is not None:
            preset["value_warmup_iterations"] = args.value_warmup_iterations
        hyper = PPOHyperparameters(
            iterations=args.iterations,
            decisions_per_iteration=args.decisions,
            evaluation_every=args.evaluation_every or max(1, args.iterations // 2),
            evaluation_episodes=2,
            gae_lambda=args.gae_lambda,
            advantage_normalization=("rollout" if args.ppo_preset == "reference-v28-episodic"
                                     else "minibatch"),
            intervention_masking=args.intervention_masking,
            final_log_std=args.final_log_std,
            log_std_anneal_fraction=args.log_std_anneal_fraction,
            enforce_target_kl=args.enforce_target_kl,
            **preset,
        )

        def train_all():
            jobs = []
            backend_kwargs = dict(curriculum_enabled=not args.nominal_only)
            if args.training_backend == "isaac":
                backend_kwargs = dict(
                    env_factory=lambda c: SpatialLandingEnv(c, IsaacBackend(c)),
                    training_backend="isaac-px4-spatial",
                    curriculum_enabled=False,
                )
            for seed in args.seeds:
                for mode in POLICY_MODES:
                    initial_checkpoint = None
                    if args.initialize_from:
                        source = args.initialize_from / "runs" / f"{mode}__seed{seed}"
                        if args.initialize_from.resolve() == output.resolve():
                            raise ValueError(
                                "fine-tuning must preserve its source output"
                            )
                        source_summary = json.loads(
                            (source / "summary.json").read_text()
                        )
                        name = source_summary.get("selected_checkpoint")
                        if not name:
                            raise ValueError(
                                "initialization source has no eligible checkpoint"
                            )
                        initial_checkpoint = source / name
                    jobs.append(
                        dict(
                            mode=mode,
                            cfg=cfg,
                            seed=seed,
                            hyper=hyper,
                            output=output / "runs" / f"{mode}__seed{seed}",
                            activation_iterations=args.activation_iterations,
                            activation_decisions=args.activation_decisions,
                            episodes_per_iteration=args.episodes_per_iteration,
                            curriculum_loss_timeout_start=args.curriculum_loss_timeout_start,
                            state_dependent_log_std=args.state_dependent_log_std,
                            initial_checkpoint=initial_checkpoint,
                            allow_v4_initialization=args.initialize_v4_weights,
                            curriculum_angular_scales=tuple(
                                args.curriculum_angular_scales
                            ),
                            **backend_kwargs,
                        )
                    )
            if args.workers == 1:
                for job in jobs:
                    train_arm(**job)
            else:
                # Pool's context terminates its OWN local children on an
                # interrupt, unlike Executor.__exit__ which waits for all
                # submitted long runs. No Isaac process enters this pool.
                with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
                    pool.map(train_local_job, jobs)

        if args.training_backend == "isaac":
            with live_stack(output, adopt=args.adopt_stack, headless=args.headless, schema=cfg.schema,
                            reset_recoveries=args.reset_recoveries,
                            isolate_episodes=args.fresh_stack_per_episode):
                train_all()
        else:
            train_all()
    rows = []

    def run_evaluation(backend):
        for seed in args.seeds:
            # Proposed model last, so the final live view is the proposed pair.
            for mode in POLICY_MODES:
                directory = output / "runs" / f"{mode}__seed{seed}"
                summary = wait_checkpoint_summary(directory, args.checkpoint_wait_seconds)
                name = summary["selected_checkpoint"]
                if not name:
                    raise RuntimeError(f"{mode} has no completed nominal checkpoint")
                agent, meta = load_agent(directory / name, cfg)
                if not meta.get("eligible"):
                    raise RuntimeError("ineligible spatial checkpoint")
                kwargs = {}
                if backend == "isaac":
                    # Each fully trained checkpoint must pass before its own
                    # first reset/arming. This permits a bounded pipeline with
                    # independent local training still finishing another arm.
                    preflight = evaluate(agent, cfg)
                    if preflight["unsafe_rate"] > 0:
                        raise RuntimeError(
                            f"{mode}: nominal local preflight failed; Isaac flight refused")
                    kwargs["env_factory"] = lambda c: SpatialLandingEnv(
                        c, IsaacBackend(c)
                    )
                    seeds = actual_seeds
                else:
                    seeds = TEST_SEEDS
                trace_path = output / f"{backend}_{mode}_seed{seed}.jsonl"
                with trace_path.open("x") as trace:

                    def record(arm, episode, step, action, info, packet=None):
                        trace.write(
                            json_text(
                                {
                                    "mode": arm,
                                    "seed": episode,
                                    "step": step,
                                    "action": action.tolist(),
                                    "packet": packet,
                                    "info": info,
                                }
                            )
                            + "\n"
                        )
                        trace.flush()

                    metrics = evaluate(agent, cfg, seeds=seeds, trace=record, **kwargs)
                row = {
                    "mode": mode,
                    "seed": seed,
                    "signature": cfg.signature,
                    "checkpoint": name,
                    "checkpoint_sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                    "episode_seeds": list(seeds),
                    "cold_stack_per_episode": bool(backend == "isaac" and args.fresh_stack_per_episode),
                    "backend": backend,
                    "metrics": metrics,
                }
                rows.append(row)
                (output / f"{backend}_evaluation.json").write_text(
                    json_text(rows, indent=2) + "\n"
                )
                print(f"[spatial {backend}] {mode} seed={seed}: {metrics}", flush=True)

    if args.stage in ("evaluate", "all"):
        run_evaluation("local")
    if args.stage in ("isaac", "all"):
        # Recheck nominal validation before allocating the real simulated
        # vehicle. A selectable checkpoint need not be safe to deploy.
        with incomplete_isaac_report(output):
            rows = []
            with live_stack(output, adopt=args.adopt_stack, headless=args.headless, schema=cfg.schema,
                            reset_recoveries=args.reset_recoveries,
                            isolate_episodes=args.fresh_stack_per_episode):
                run_evaluation("isaac")
        acceptance = actual_acceptance(rows, args.seeds)
        (output / "isaac_acceptance.json").write_text(
            json_text(acceptance, indent=2) + "\n"
        )
        if not acceptance["passes"]:
            return 2
    return 0


def request_shutdown(*_):
    # Unwind env/stack finally blocks so a terminated run cannot leave the
    # last acceleration command driving an armed vehicle.
    raise KeyboardInterrupt("spatial shutdown requested")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_shutdown)
    raise SystemExit(main())
