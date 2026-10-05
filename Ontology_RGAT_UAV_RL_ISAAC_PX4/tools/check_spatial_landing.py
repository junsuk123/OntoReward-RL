#!/usr/bin/env python3
"""Isolated causal flight-feasibility diagnostic, NEVER a PPO teacher.

Use --backend isaac explicitly for real simulated flight. This diagnostic
does not import a policy, write checkpoints, or constitute learned success.
The same measurement boundary, supervisor, plant and evaluator are exercised.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import SpatialLandingEnv, IsaacBackend


def diagnostic_optical_fault(measurement, elapsed_s, blackout_after_s):
    """Diagnostic-only input loss; never alter physics, own telemetry or truth."""
    if blackout_after_s is not None and elapsed_s >= blackout_after_s:
        return replace(measurement,optical_position=None,confidence=0.)
    return measurement


def run(cfg, seed, backend, trace, *, integral_gain=0., blackout_after_s=None):
    env = SpatialLandingEnv(cfg, IsaacBackend(cfg) if backend == "isaac" else None)
    try:
        env.reset(seed=seed)
        original_update=env.estimator.update
        if blackout_after_s is not None:
            env.estimator.update=lambda measurement: original_update(diagnostic_optical_fault(
                measurement,measurement.time_s-env.start,blackout_after_s))
        reference = env.estimator.own.own_velocity.copy()
        integral_xy = np.zeros(2)
        previous_time = env.estimator.last_t
        for step in range(int(cfg.horizon / cfg.dt) * 3 + 5):
            est = env.estimator
            if cfg.direct_acceleration:
                reference = est.own.own_velocity.copy()
            elapsed = max(0., est.last_t-previous_time)
            previous_time = est.last_t
            # Optional diagnostic-only integral rejects constant external
            # forces using optical/EKF estimates. Never exported to PPO.
            if est.age < .5 and not env.safety.abort:
                integral_xy = np.clip(integral_xy+elapsed*est.r[:2], -2., 2.)
            desired = est.pad_v.copy()
            desired[:2] -= 0.65 * est.r[:2] + integral_gain*integral_xy
            aligned = (
                np.linalg.norm(est.r[:2]) < 0.12 and np.linalg.norm(est.rv[:2]) < 0.2
            )
            # A constant, sub-limit sink crosses the camera's near-field blind
            # interval within the bounded terminal-coast allowance. Flaring
            # toward zero above contact strands a camera-only vehicle blind.
            desired[2] = -0.25 if aligned else 0.0
            action = np.clip(
                (desired - reference) / (0.6 * np.asarray(cfg.max_acceleration)), -1, 1
            )
            _, reward, done, _, info = env.step(action)
            reference += np.asarray(info["applied_acceleration_m_s2"]) * cfg.dt
            row = {
                "seed": seed,
                "step": step,
                "controller": "isolated-causal-feasibility",
                "learned_policy": False,
                "diagnostic_integral_gain": integral_gain,
                "diagnostic_optical_blackout_after_s": blackout_after_s,
                "reward": reward,
                "action": action.tolist(),
                "info": info,
            }
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            if step % 20 == 0 or done:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "status": info["status"],
                            "estimated": info["estimated_relative_position"],
                            "truth": info["truth_relative_position"],
                            "age": info["optical_age_s"],
                        }
                    ),
                    flush=True,
                )
            if done:
                return info["status"]
        raise RuntimeError("diagnostic exceeded its bounded episode")
    finally:
        env.close()


def main():
    # Let a bounded runner unwind env.close()/live_stack on termination.
    def request_shutdown(signum, _frame):
        raise KeyboardInterrupt(f"diagnostic shutdown requested ({signum})")

    signal.signal(signal.SIGTERM, request_shutdown)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "isaac"], default="local")
    parser.add_argument("--seed", type=int, default=14000)
    parser.add_argument("--episodes", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--integral-gain", type=float, default=0.,
                        help="Isolated causal diagnostic only; never a PPO target")
    parser.add_argument("--horizon", type=float, default=45.0)
    parser.add_argument("--optical-blackout-after-s",type=float,
                        help="DIAGNOSTIC ONLY: hide optical updates after this simulation time; no physics/own-state changes")
    parser.add_argument("--start-stack", action="store_true",
                        help="Own a new stack; otherwise explicitly adopt the operator's stack")
    parser.add_argument("--reset-recoveries", type=int, choices=range(6), default=0)
    parser.add_argument("--contract-version", choices=["5", "6", "7", "8", "9", "10"], default="5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.integral_gain <= 1:
        parser.error('diagnostic integral gain must be in [0,1]')
    if args.optical_blackout_after_s is not None and not (
            math.isfinite(args.optical_blackout_after_s)
            and 0 < args.optical_blackout_after_s < args.horizon):
        parser.error('diagnostic blackout time must be finite and inside the mission')
    cfg = replace(SpatialConfig(), horizon=args.horizon,
                  schema=f"spatial-causal-rgat/{args.contract_version}")
    args.output.mkdir(parents=True, exist_ok=True)

    def execute():
        result = {
            "backend": args.backend,
            "seed": args.seed,
            "signature": cfg.signature,
            "requested_episodes": args.episodes,
            "episodes": [],
            "complete": False,
            "learned_policy": False,
            "ppo_success_claim": False,
            "diagnostic_integral_gain": args.integral_gain,
            "diagnostic_optical_blackout_after_s": args.optical_blackout_after_s,
        }
        try:
            with (args.output / "feasibility.jsonl").open("x") as trace:
                for offset in range(args.episodes):
                    status = run(cfg, args.seed+offset, args.backend, trace,
                                 integral_gain=args.integral_gain,
                                 blackout_after_s=args.optical_blackout_after_s)
                    result["episodes"].append(dict(seed=args.seed+offset,status=status))
                    result["status"] = status
                    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            result["complete"] = True
            return 0 if all(e['status']=='SUCCESS' for e in result['episodes']) else 2
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.backend == "isaac":
        # Explicit adoption is required here because the operator starts and
        # owns the diagnostic stack separately. Lock/profile checks still run.
        from run_spatial_pipeline import live_stack

        with live_stack(args.output, adopt=not args.start_stack, schema=cfg.schema,
                        reset_recoveries=args.reset_recoveries):
            return execute()
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
