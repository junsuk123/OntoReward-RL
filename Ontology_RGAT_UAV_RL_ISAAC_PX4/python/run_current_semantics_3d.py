#!/usr/bin/env python3
"""``./run.sh current-semantics-3d``: the current-observation 3D study (guide 10-12).

Stages that would train or register a comparison are refused until the gates
open; ``--dry-run`` prints the resolved plan and creates nothing.  The Isaac
half of G3 (``--stage coupling --backend isaac``) and the Isaac evaluation of
the selected G4 checkpoints (``--stage isaac-eval``) and of the development models
(``--stage isaac-development``, guide 11.6) START an owned Isaac Sim + PX4 stack and
run only with ``--allow-isaac``; none adopts a running Isaac.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config
from ontology_rgat.current_semantics.experiment import (ARTIFACTS, DEFAULT_RUN_ROOT,
                                                       DEVELOPMENT_STAGES, REFUSED_STAGES,
                                                       fixture_checks,
                                                       gate_table, isaac_eval_state, owned,
                                                       refusal_reasons, report_states,
                                                       resolved_plan, run_root_for, run_smoke,
                                                       write_json)


ISAAC_DEVELOPMENT_STAGES = ("register-isaac-development", "isaac-development")
RELATION_SHUFFLE_STAGES = ("register-relation-shuffle", "relation-shuffle")
ORL_STAGES = ("orl-learnability", "register-orl-development", "orl-develop")
STAGES = ("audit", "fixtures", "smoke", "coupling", "learnability", "isaac-eval",
          "isaac-trace", "report", "dashboard", *DEVELOPMENT_STAGES,
          *ISAAC_DEVELOPMENT_STAGES, "isaac-handover-fix", *RELATION_SHUFFLE_STAGES,
          *ORL_STAGES, "monitor", *REFUSED_STAGES)
FLYING_STAGES = ("isaac-eval", "isaac-trace", "isaac-development", "isaac-handover-fix")
REFUSED_EXIT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./run.sh current-semantics-3d",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="default: the config's task root (task v5 "
                             f"{DEFAULT_RUN_ROOT.name}, task v6 {DEFAULT_RUN_ROOT.name}_task6); "
                             "relative paths resolve against the launch directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved plan; create nothing")
    parser.add_argument("--seed", type=int, help="smoke only (default 0)")
    parser.add_argument("--steps", type=int, help="smoke only (default 32)")
    parser.add_argument("--backend", choices=("local", "isaac"), help="coupling only")
    parser.add_argument("--allow-isaac", action="store_true",
                        help="coupling --backend isaac / isaac-eval only: start an owned "
                             "Isaac/PX4 stack")
    parser.add_argument("--headless", action="store_true",
                        help="coupling --backend isaac / isaac-eval only")
    parser.add_argument("--pairs", default=None,
                        help="isaac-trace only: comma-separated model_seed:seed pairs to "
                             "re-fly with per-decision traces, e.g. 1:994001,0:994008")
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="isaac-trace only: an archived G4 pilot root (holding g4_pilot.json) "
                             "whose selected checkpoints to re-fly; they may have been trained "
                             "on another task, every other contract hash must match")
    parser.add_argument("--plan", type=Path, default=None,
                        help="register-development/develop only: the development plan "
                             "(default config/current_semantics/development_v1.yaml); develop "
                             "refuses any plan that differs from the registration")
    parser.add_argument("--reset-recoveries", type=int, default=None,
                        help="coupling --backend isaac / isaac-eval only: owned restarts for "
                             "pre-policy entry failures (default 2)")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.stage != "smoke" and (args.seed is not None or args.steps is not None):
        parser.error("--seed/--steps apply to --stage smoke only")
    if args.stage != "coupling" and args.backend is not None:
        parser.error("--backend applies to --stage coupling only")
    flies = args.backend == "isaac" or args.stage in FLYING_STAGES
    if (args.pairs is not None) != (args.stage == "isaac-trace"):
        parser.error("--pairs is required by, and only by, --stage isaac-trace")
    if args.plan is not None and args.stage not in (*DEVELOPMENT_STAGES, "isaac-handover-fix",
                                                    *RELATION_SHUFFLE_STAGES, *ORL_STAGES[1:]):
        parser.error("--plan applies to --stage register-development/develop/"
                     "isaac-handover-fix/register-relation-shuffle/relation-shuffle only")
    if args.checkpoint_root is not None and args.stage != "isaac-trace":
        parser.error("--checkpoint-root applies to --stage isaac-trace only")
    if not flies and (args.headless or args.reset_recoveries is not None or args.allow_isaac):
        parser.error("--allow-isaac/--headless/--reset-recoveries apply to --stage coupling "
                     "--backend isaac and the flying stages (isaac-eval, isaac-trace, "
                     "isaac-development) only")
    if args.stage == "coupling" and args.backend is None:
        parser.error("--stage coupling needs --backend local|isaac")
    if args.stage == "smoke" and args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _isaac_coupling(config, root: Path, extra: dict) -> dict:
    """The Isaac half of G3 inside an owned stack (run_spatial_pipeline.live_stack:
    flight lock, refuses an already-running Isaac, confirmed stop on exit)."""
    from ontology_rgat.current_semantics.backend import make_isaac_backend
    from ontology_rgat.current_semantics.coupling import run_isaac_checks
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    from run_spatial_pipeline import live_stack
    profile = Path(deployment_profile("spatial-reference/12")["path"])
    stack_root = root / "isaac_stack"
    stack_root.mkdir(parents=True, exist_ok=True)
    with live_stack(stack_root, adopt=False, headless=extra["headless"],
                    schema="spatial-reference/12", reset_recoveries=extra["reset_recoveries"],
                    profile_path=profile):
        backend = make_isaac_backend(config, allow_isaac=True)
        try:
            return run_isaac_checks(config, backend)
        finally:
            backend.close()


def _selected_checkpoints(config, root: Path) -> dict[str, Path]:
    """best.pt of every model seed that G4 tested, from ``root``'s own g4_pilot/ (an
    archived pilot keeps its checkpoints there; the path recorded in its g4_pilot.json
    may now hold a LATER pilot's file).  The checkpoint's stored update must equal the
    update that pilot tested, or nothing flies."""
    import torch
    pilot = json.loads((root / ARTIFACTS["learnability"]).read_text(encoding="utf-8"))
    chosen, wrong = {}, []
    for seed, test in pilot["tests"].items():
        path = root / "g4_pilot" / f"seed{seed}" / "best.pt"
        if not test.get("checkpoint") or not path.exists():
            wrong.append(f"{seed}: no selected checkpoint")
            continue
        stored = torch.load(path, map_location="cpu", weights_only=True)["extra"].get("update")
        if stored != test.get("update"):
            wrong.append(f"{seed}: {path} holds update {stored}, the pilot tested {test.get('update')}")
            continue
        chosen[seed] = path
    if wrong:
        raise SystemExit(f"isaac: refusing the selected checkpoints of {root}: {wrong}")
    return chosen


def _pairs(text: str, config=None, models=()) -> list[tuple[str, int]]:
    if text == "all":  # every registered isaac_eval seed for every checkpoint, seed-major
        return [(model, seed) for seed in config.isaac_eval.seeds for model in models]
    pairs = []
    for item in text.split(","):
        model, _, seed = item.strip().partition(":")
        if not model or not seed.isdigit():
            raise SystemExit(f"isaac-trace: --pairs item {item!r} is not model_seed:seed")
        pairs.append((model, int(seed)))
    return pairs


def _development(args, config_path: Path, root: Path) -> int:
    """register-development / develop (guide 11): local plant only, never Isaac."""
    from ontology_rgat.current_semantics.development import (
        default_plan_path, load_development, register, resolved, run_development)
    config = load_config(config_path)
    plan_path = (args.plan or default_plan_path(config)).expanduser().resolve()
    plan = load_development(plan_path, config)
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True,
                             extra={"plan_file": str(plan_path), **resolved(plan, config)}))
        return 0
    with owned(root):
        if args.stage == "register-development":
            status, payload = register(config_path, plan_path, root)
            path = root / ARTIFACTS["register_development"]
            if payload.get("status") == "REGISTERED":
                write_json(path, payload)
        else:
            # The live dashboard comes up with the run (guide 11 four-arm comparison);
            # a daemon thread, so it ends with this process.
            from ontology_rgat.current_semantics.dashboard import export, serve
            try:
                _, url = serve(root, plan_path=plan_path)
                print(f"develop: live dashboard {url}", flush=True)
            except OSError as exc:
                print(f"develop: dashboard not started ({exc})", flush=True)
            status, payload = run_development(config_path, plan_path, root,
                                              log=lambda line: print(line, flush=True))
            path = root / ARTIFACTS["develop"]
            if payload.get("status") != "REFUSED":
                write_json(path, payload)
                print(f"develop: dashboard snapshot {export(root)}", flush=True)
    if payload.get("status") in ("REFUSED", "ALREADY_REGISTERED"):
        _print({k: v for k, v in payload.items() if k in ("stage", "status", "reason",
                                                           "plan_hash")})
        return status
    print(f"{args.stage}: wrote {path} (exit {status})")
    return status


def _relation_shuffle(args, config_path: Path, root: Path) -> int:
    """register-relation-shuffle / relation-shuffle (guide 8.4, status doc 19): a fifth,
    relation-shuffled R-GAT arm of development/1; local plant only, never Isaac."""
    from ontology_rgat.current_semantics import relation_control as rc
    config = load_config(config_path)
    plan_path = (args.plan or rc.DEFAULT_PLAN_PATH).expanduser().resolve()
    try:
        plan, development = rc.load_plan(plan_path, config)
    except ValueError as exc:
        _print({"stage": args.stage, "status": "REFUSED", "dry_run": args.dry_run,
                "reason": str(exc)})
        return REFUSED_EXIT
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True,
                             extra={"plan_file": str(plan_path),
                                    **rc.resolved(plan, development, config),
                                    "blocked_by": rc.base_blocker(config, development, root),
                                    "will_write": ([str(root / rc.REGISTRATION)]
                                                   if args.stage == "register-relation-shuffle"
                                                   else [str(root / rc.RESULT),
                                                         str(root / rc.OUT_DIR)])}))
        return 0
    with owned(root):
        if args.stage == "register-relation-shuffle":
            status, payload = rc.register(config_path, plan_path, root)
            path = root / rc.REGISTRATION
            if payload.get("status") == "REGISTERED":
                write_json(path, payload)
        else:
            status, payload = rc.run(config_path, plan_path, root,
                                     log=lambda line: print(line, flush=True))
            path = root / rc.RESULT
            if payload.get("status") != "REFUSED":
                write_json(path, payload)
    if payload.get("status") in ("REFUSED", "ALREADY_REGISTERED"):
        _print({k: v for k, v in payload.items() if k in ("stage", "status", "reason",
                                                           "plan_hash")})
        return status
    print(f"{args.stage}: wrote {path} (exit {status})")
    return status


def _orl(args, config_path: Path, root: Path) -> int:
    """orl-learnability (status doc 20): the G4 learnability protocol of this config with
    the ORL arm, in ``<root>/orl`` under its own lock; local plant only."""
    from ontology_rgat.current_semantics import orl
    from ontology_rgat.current_semantics.pilot import run_pilot
    out = root / "orl"
    config = load_config(config_path)
    if args.stage != "orl-learnability":
        return _orl_development(args, config_path, root)
    path = out / "orl_learnability.json"
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True, extra={
            "arm": orl.ORL_ARM, "orl_hash": orl.ORL_HASH, "spec": orl.spec_dict(),
            "model_seeds": list(config.g4.model_seeds), "updates": config.g4.updates,
            "episodes_per_update": config.g4.episodes_per_update,
            "will_write": [str(path), str(out / "g4_pilot")], "trains": True,
            "starts_isaac": False}))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with owned(out):
        report = run_pilot(config_path, out / "g4_pilot", arm=orl.ORL_ARM,
                           log=lambda line: print(line, flush=True))
        report["orl"] = {"hash": orl.ORL_HASH, "spec": orl.spec_dict()}
        write_json(path, report)
    print(f"{args.stage}: wrote {path} (learnability_passed {report['learnability_passed']})")
    return 0


def _orl_development(args, config_path: Path, root: Path) -> int:
    """register-orl-development / orl-develop (status doc 20): ORL as a paired arm of
    development/1; local plant only, in ``<root>/orl`` under its own lock."""
    from ontology_rgat.current_semantics import orl_development as od
    config = load_config(config_path)
    plan_path = (args.plan or od.DEFAULT_PLAN_PATH).expanduser().resolve()
    try:
        plan, development = od.load_plan(plan_path, config)
    except ValueError as exc:
        _print({"stage": args.stage, "status": "REFUSED", "dry_run": args.dry_run,
                "reason": str(exc)})
        return REFUSED_EXIT
    out = root / od.OUT
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True, extra={
            "plan_file": str(plan_path), **od.resolved(plan, development, config),
            "blocked_by": od.blocker(config, development, root),
            "will_write": ([str(out / od.REGISTRATION)] if args.stage == od.STAGES[0]
                           else [str(out / od.RESULT), str(out / "development")])}))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with owned(out):
        if args.stage == od.STAGES[0]:
            status, payload = od.register(config_path, plan_path, root)
            path = out / od.REGISTRATION
            if payload.get("status") == "REGISTERED":
                write_json(path, payload)
        else:
            status, payload = od.run(config_path, plan_path, root,
                                     log=lambda line: print(line, flush=True))
            path = out / od.RESULT
            if payload.get("status") != "REFUSED":
                write_json(path, payload)
    if payload.get("status") in ("REFUSED", "ALREADY_REGISTERED"):
        _print({k: v for k, v in payload.items() if k in ("stage", "status", "reason",
                                                           "plan_hash")})
        return status
    print(f"{args.stage}: wrote {path} (exit {status})")
    return status


def _fly_development_models(config, plan, dev_plan, root: Path, out: Path, *, headless: bool,
                            reset_recoveries: int, resume: bool, order=None,
                            label: str = "isaac-development"):
    """The development models on an owned Isaac/PX4 stack (live_stack: flight lock, refuses
    an already-running Isaac, confirmed stop on exit), one flight and one backend wrapper
    per condition.  Returns (idev.run's record, the flight)."""
    from dataclasses import asdict
    from ontology_rgat.current_semantics import development as dev
    from ontology_rgat.current_semantics import isaac_development as idev
    from ontology_rgat.current_semantics.backend import (CurrentSpatialBackend,
                                                         make_isaac_backend)
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    from run_spatial_pipeline import live_stack
    profile = Path(deployment_profile("spatial-reference/12")["path"])
    stack_root = out / "isaac_stack"
    stack_root.mkdir(parents=True, exist_ok=True)
    wrappers: list = []

    def make(cell):  # one Isaac flight, one wrapper per condition
        factors, dropout = dev.cell_conditions(config, asdict(cell), dev_plan.dropout_start_s)
        if not wrappers:
            backend = make_isaac_backend(config, allow_isaac=True, split=cell.split,
                                         factors=factors, dropout=dropout)
        else:
            backend = CurrentSpatialBackend(wrappers[0].flight, config, name="isaac_v12",
                                            scientific=False, split=cell.split,
                                            factors=factors, dropout=dropout)
        wrappers.append(backend)
        return backend

    with live_stack(stack_root, adopt=False, headless=headless,
                    schema="spatial-reference/12",
                    reset_recoveries=reset_recoveries, profile_path=profile):
        try:
            flown = idev.run(config, plan, dev_plan, root, make, out,
                             log=lambda line: print(line, flush=True),
                             resume=resume, order=order, label=label)
        finally:
            error = None
            for backend in sorted(wrappers, key=lambda w: not w.active):
                try:
                    backend.close()
                except Exception as exc:  # keep closing; report the first failure
                    error = error or exc
            if error is not None:
                raise error
    return flown, (wrappers[0].flight if wrappers else None)


def _isaac_handover_fix(args, config_path: Path, root: Path) -> int:
    """isaac-handover-fix (status doc 16): identical isaac-development/1 cases re-flown after
    the handover-kick fix, registered (cases, recorded verdicts, checkpoints, Isaac source)
    before the first episode flies, judged against the plan's criteria."""
    from ontology_rgat.current_semantics import isaac_handover_fix as fix
    plan_path = (args.plan or fix.DEFAULT_PATH).expanduser().resolve()
    problem, ctx = fix.prepare(config_path, plan_path, root)
    problem = problem or fix.check_registration(ctx, root)
    registration, result, out_dir = fix.files(fix.load_plan(plan_path).protocol)
    previous = root / result
    if not problem and previous.exists() and json.loads(
            previous.read_text(encoding="utf-8")).get("complete"):
        problem = f"this protocol's flight is complete ({previous})"
    if problem:
        _print({"stage": args.stage, "status": "REFUSED", "reason": problem})
        return REFUSED_EXIT
    out = root / out_dir
    cases = ctx["cases"]
    extra = {"plan_file": str(plan_path), "plan": ctx["plan"].to_dict(),
             "plan_hash": ctx["plan_hash"], "out_dir": str(out),
             "cases": [f"{c} {s} {k}: before {ctx['before'][(c, s, k)]['status']}, "
                       f"local {ctx['local'][(c, s, k)]['status']}" for c, s, k in cases],
             "episodes": len(cases),
             "recorded_success": {
                 "before": sum(ctx["before"][c]["status"] == "SUCCESS" for c in cases),
                 "local": sum(ctx["local"][c]["status"] == "SUCCESS" for c in cases)},
             "registered": (root / registration).exists(),
             "resume": (out / "episodes.jsonl").exists(),
             "starts_owned_stack": True, "adopts_running_isaac": False,
             "headless": args.headless,
             "reset_recoveries": 2 if args.reset_recoveries is None else args.reset_recoveries}
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True, extra=extra))
        return 0
    with owned(root):
        if not extra["registered"]:
            write_json(root / registration, fix.registration(ctx, plan_path))
        flown, flight = _fly_development_models(
            ctx["config"], ctx["reference"], ctx["dev_plan"], root, out,
            headless=args.headless, reset_recoveries=extra["reset_recoveries"],
            resume=extra["resume"], order=cases, label=args.stage)
        summary = fix.summarize(ctx["plan"], cases, flown["rows"], ctx["before"],
                                ctx["local"], out / "traces")
        payload = {"schema_id": fix.SCHEMA_ID, "stage": args.stage,
                   "protocol": ctx["plan"].protocol,
                   "plan_hash": ctx["plan_hash"], "backend": getattr(flight, "name", None),
                   "profile_sha256": getattr(flight, "profile_sha256", None),
                   "stopped": flown["stopped"], "interruptions": flown["interruptions"],
                   "wall_s": flown["wall_s"], **summary}
        write_json(previous, payload)
    print(f"{args.stage}: wrote {previous} (fix_verified {payload['fix_verified']})")
    return 0 if payload["complete"] else 1


def _isaac_development(args, config_path: Path, root: Path) -> int:
    """register-isaac-development / isaac-development (guide 11.6): every development model
    on the registered Isaac cases, after a PASSED development gate, with a paired local
    reference on the identical cases."""
    from ontology_rgat.current_semantics import development as dev
    from ontology_rgat.current_semantics import isaac_development as idev
    config = load_config(config_path)
    dev_plan = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    plan = idev.load_plan(idev.DEFAULT_PATH, config, dev_plan)
    keys = idev.model_keys(dev_plan)
    out = root / idev.OUT_DIR
    summary = {"plan_file": str(idev.DEFAULT_PATH), "plan": plan.to_dict(),
               "models": list(keys), "episodes": len(plan.seeds) * len(keys)}
    if args.stage == "register-isaac-development":
        if args.dry_run:
            _print(resolved_plan(config_path, args.stage, root, dry_run=True,
                                 extra={**summary, "starts_isaac": False}))
            return 0
        with owned(root):
            status, payload = idev.register(config_path, idev.DEFAULT_PATH,
                                            dev.DEFAULT_DEVELOPMENT_PATH, root)
            path = root / ARTIFACTS["register_isaac_development"]
            if payload.get("status") == "REGISTERED":
                write_json(path, payload)
        _print({k: v for k, v in payload.items()
                if k in ("stage", "status", "reason", "plan_hash", "episodes")})
        return status
    problem = idev.check_registration(config, plan, dev_plan, root)
    if problem:
        _print({"stage": args.stage, "status": "REFUSED", "reason": problem})
        return REFUSED_EXIT
    previous = root / ARTIFACTS["isaac_development"]
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8")).get(
            "protocol_complete"):
        _print({"stage": args.stage, "status": "REFUSED",
                "reason": f"this registration's Isaac flight is complete ({previous})"})
        return REFUSED_EXIT
    extra = {**summary, "out_dir": str(out),
             "resume": (out / "episodes.jsonl").exists(),
             "local_reference": str(out / idev.LOCAL_REFERENCE),
             "starts_owned_stack": True, "adopts_running_isaac": False,
             "headless": args.headless,
             "reset_recoveries": 2 if args.reset_recoveries is None else args.reset_recoveries}
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True, extra=extra))
        return 0
    with owned(root):
        out.mkdir(parents=True, exist_ok=True)
        local_path = out / idev.LOCAL_REFERENCE
        if local_path.exists():
            local = [json.loads(line) for line in
                     local_path.read_text(encoding="utf-8").splitlines()]
        else:  # deterministic, so its order relative to the flight does not matter
            print("isaac-development: local reference on the identical cases", flush=True)
            local = idev.local_reference(config_path, plan, dev_plan, root)
            local_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in local),
                                  encoding="utf-8")
        flown, flight = _fly_development_models(config, plan, dev_plan, root, out,
                                                headless=extra["headless"],
                                                reset_recoveries=extra["reset_recoveries"],
                                                resume=extra["resume"])
        payload = idev.report(config, plan, dev_plan, flown,
                              backend=getattr(flight, "name", None),
                              profile_sha256=getattr(flight, "profile_sha256", None),
                              local=local)
        write_json(previous, payload)
    status = 0 if payload["protocol_complete"] else 1
    print(f"{args.stage}: wrote {previous} (exit {status})")
    return status


def _isaac_eval(config, root: Path, extra: dict) -> dict:
    """The selected G4 checkpoints inside an owned stack (same rules as the G3 half).
    With ``extra["pairs"]`` it is the diagnostic re-flight: traces only, no verdict."""
    from ontology_rgat.current_semantics.backend import make_isaac_backend
    from ontology_rgat.current_semantics.isaac_eval import run_isaac_evaluation
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    from run_spatial_pipeline import live_stack
    checkpoints = _selected_checkpoints(config, Path(extra.get("checkpoint_root") or root))
    profile = Path(deployment_profile("spatial-reference/12")["path"])
    out = Path(extra["out_dir"])
    stack_root = out / "isaac_stack"
    stack_root.mkdir(parents=True, exist_ok=True)
    with live_stack(stack_root, adopt=False, headless=extra["headless"],
                    schema="spatial-reference/12", reset_recoveries=extra["reset_recoveries"],
                    profile_path=profile):
        backend = make_isaac_backend(config, allow_isaac=True)
        try:
            return run_isaac_evaluation(config, backend, checkpoints, out,
                                        log=lambda line: print(line, flush=True),
                                        pairs=extra.get("pairs"),
                                        resume=bool(extra.get("resume")),
                                        foreign_task=bool(extra.get("checkpoint_root")))
        finally:
            backend.close()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    config_path = args.config.expanduser().resolve()
    requested = None if args.run_root is None else args.run_root.expanduser()
    if requested is not None and not requested.is_absolute():
        requested = Path.cwd() / requested
    try:
        root = run_root_for(load_config(config_path), requested)
    except ValueError as exc:
        parser.error(str(exc))

    if args.stage == "monitor":  # read-only: every task root, process and log; no lock
        from ontology_rgat.current_semantics.monitor import RESULTS, serve as serve_monitor
        if args.dry_run:
            _print({"stage": "monitor", "dry_run": True, "reads": str(RESULTS),
                    "serves": "http://127.0.0.1:8780/ (next free port up to 8799)",
                    "writes": [], "trains": False, "starts_isaac": False})
            return 0
        server, url = serve_monitor(background=False)
        print(f"monitor: {url} (read-only view of {RESULTS}; Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    if args.stage == "dashboard":  # read-only: no lock, no gate, never trains or flies
        from ontology_rgat.current_semantics.dashboard import serve
        if args.dry_run:
            _print({"stage": "dashboard", "dry_run": True, "run_root": str(root),
                    "serves": "http://127.0.0.1:8773/ (next free port up to 8792)",
                    "writes": [], "trains": False, "starts_isaac": False})
            return 0
        server, url = serve(root, background=False)
        print(f"dashboard: {url} (read-only view of {root}; Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    if args.stage in REFUSED_STAGES:
        _print({"stage": args.stage, "status": "REFUSED", "dry_run": args.dry_run,
                "reason": "confirmation opens only after a passed development gate (guide "
                          "11.4-11.5) and is not implemented yet"})
        return REFUSED_EXIT
    if args.stage in DEVELOPMENT_STAGES:
        config = load_config(config_path)
        gates = gate_table(config, root)
        if refusal_reasons(gates):
            _print({"stage": args.stage, "status": "REFUSED", "dry_run": args.dry_run,
                    "reason": "no development run may be registered or started while any of "
                              "G0-G4 is open (guide 10-11)",
                    "refusal_reasons": refusal_reasons(gates)})
            return REFUSED_EXIT
        return _development(args, config_path, root)
    if args.stage in RELATION_SHUFFLE_STAGES:
        return _relation_shuffle(args, config_path, root)
    if args.stage in ORL_STAGES:
        return _orl(args, config_path, root)
    if args.stage in FLYING_STAGES and not args.allow_isaac:
        _print({"stage": args.stage, "status": "REFUSED", "allow_isaac": False,
                "reason": "the Isaac evaluation starts an owned Isaac Sim + PX4 stack and runs "
                          "only with --allow-isaac, on the user's approval; no stack was "
                          "contacted"})
        return REFUSED_EXIT
    if args.stage in ISAAC_DEVELOPMENT_STAGES:
        return _isaac_development(args, config_path, root)
    if args.stage == "isaac-handover-fix":
        return _isaac_handover_fix(args, config_path, root)
    if args.stage == "coupling" and args.backend == "isaac" and not args.allow_isaac:
        _print({"stage": "coupling", "backend": "isaac", "status": "REFUSED",
                "allow_isaac": False,
                "reason": "the Isaac half of G3 starts an owned Isaac Sim + PX4 stack and runs "
                          "only with --allow-isaac, on the user's approval; no stack was "
                          "contacted"})
        return REFUSED_EXIT

    if args.stage == "learnability":
        config = load_config(config_path)
        gates = gate_table(config, root)
        unmet = [g for g in ("G2", "G3") if gates[g]["status"] != "passed"]
        if unmet:
            _print({"stage": "learnability", "status": "REFUSED",
                    "reason": f"the G4 pilot needs G2 and G3 passed; not passed: {unmet}",
                    "refusal_reasons": refusal_reasons(gates)})
            return REFUSED_EXIT

    if args.stage in FLYING_STAGES:
        config = load_config(config_path)
        gates = gate_table(config, root)
        if gates["G4"]["status"] != "passed":
            _print({"stage": args.stage, "status": "REFUSED",
                    "reason": "the Isaac evaluation flies the selected checkpoints of a PASSED "
                              f"G4 pilot; G4 is {gates['G4']['status']}",
                    "refusal_reasons": refusal_reasons(gates)})
            return REFUSED_EXIT

    extra = {}
    if args.stage == "isaac-eval":
        # An interrupted flight of THIS protocol (same hashes) is continued, never
        # restarted; a completed one is never flown again.
        from ontology_rgat.current_semantics.isaac_eval import HASH_KEYS
        from ontology_rgat.current_semantics.contracts import contract_manifest
        previous = root / ARTIFACTS["isaac_eval"]
        if previous.exists():
            report = json.loads(previous.read_text(encoding="utf-8"))
            manifest = contract_manifest(config)
            same = report.get("hashes") == {k: manifest[k] for k in HASH_KEYS}
            if same and report.get("protocol_complete"):
                _print({"stage": "isaac-eval", "status": "REFUSED",
                        "reason": "this protocol's Isaac evaluation is already complete "
                                  f"({previous}); register a new one to fly again"})
                return REFUSED_EXIT
            if not same:
                _print({"stage": "isaac-eval", "status": "REFUSED",
                        "reason": f"{previous} belongs to another contract; archive it first"})
                return REFUSED_EXIT
    if args.stage in FLYING_STAGES:
        source = (args.checkpoint_root.expanduser().resolve()
                  if args.checkpoint_root is not None else root)
        selected = _selected_checkpoints(config, source)
        pairs = (_pairs(args.pairs, config, tuple(selected)) if args.stage == "isaac-trace"
                 else None)
        if pairs is not None:  # refuse a bad pair before any stack starts
            from ontology_rgat.current_semantics.isaac_eval import check_pairs
            try:
                pairs = check_pairs(config, pairs, selected)
            except ValueError as exc:
                raise SystemExit(f"isaac-trace: {exc}") from None
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extra = {"checkpoints": {k: str(v) for k, v in selected.items()},
                 "checkpoint_root": (str(source) if args.checkpoint_root is not None
                                     else None),
                 "out_dir": str(root / "isaac_eval" if pairs is None
                                else root / "isaac_trace" / stamp),
                 "pairs": pairs, "verdict": pairs is None,
                 "resume": (pairs is None and (root / ARTIFACTS["isaac_eval"]).exists()),
                 "seeds": list(config.isaac_eval.seeds), "mode": "deterministic",
                 "episodes": (len(pairs) if pairs else
                              len(config.isaac_eval.seeds) * len(config.g4.model_seeds)),
                 "criterion": {"deterministic_success_min":
                               config.isaac_eval.deterministic_success_min},
                 "starts_owned_stack": True, "adopts_running_isaac": False,
                 "headless": args.headless,
                 "reset_recoveries": 2 if args.reset_recoveries is None
                 else args.reset_recoveries}
    if args.stage == "learnability":
        g4 = config.g4
        extra = {"arm": g4.arm, "model_seeds": list(g4.model_seeds), "updates": g4.updates,
                 "episodes_per_update": g4.episodes_per_update,
                 "training_episodes": g4.updates * g4.episodes_per_update * len(g4.model_seeds),
                 "validation_seeds": len(g4.validation_seeds), "test_seeds": len(g4.test_seeds),
                 "workers": g4.workers, "backend": "spatial_local_v12",
                 "trains": True, "starts_isaac": False}
    if args.stage == "smoke":
        extra = {"seed": 0 if args.seed is None else args.seed,
                 "steps": 32 if args.steps is None else args.steps}
    if args.stage == "coupling":
        extra = {"backend": args.backend}
        if args.backend == "isaac":
            config = load_config(config_path)
            extra.update(isaac_seeds=list(config.g3.isaac.seeds),
                         isaac_episodes=3 * len(config.g3.isaac.seeds),
                         starts_owned_stack=True, adopts_running_isaac=False,
                         headless=args.headless,
                         reset_recoveries=2 if args.reset_recoveries is None
                         else args.reset_recoveries)
    if args.dry_run:
        _print(resolved_plan(config_path, args.stage, root, dry_run=True, extra=extra))
        return 0

    config = load_config(config_path)
    if args.stage == "report":
        found = {stage: str(root / name) for stage, name in ARTIFACTS.items()
                 if (root / name).exists()}
        _print({"stage": "report", "run_root": str(root), "artifacts": found,
                "gates": (gates := gate_table(config, root)),
                "report_states": report_states(gates, isaac_eval_state(config, root))})
        return 0

    with owned(root):
        if args.stage == "audit":
            payload = resolved_plan(config_path, "audit", root, dry_run=False)
            status = 0
        elif args.stage == "fixtures":
            checks = fixture_checks(config)
            payload = {"stage": "fixtures", "checks": checks,
                       "all_passed": all(check["passed"] for check in checks)}
            status = 0 if payload["all_passed"] else 1
        elif args.stage == "smoke":
            payload = run_smoke(config, seed=extra["seed"], steps=extra["steps"])
            status = 0 if payload["all_finite"] else 1
        elif args.stage == "learnability":
            from ontology_rgat.current_semantics.pilot import run_pilot
            payload = run_pilot(config_path, root / "g4_pilot",
                                log=lambda line: print(line, flush=True))
            status = 0
        elif args.stage in FLYING_STAGES:
            payload = _isaac_eval(config, root, extra)
            flown = sum(r["deterministic"]["episodes"] for r in payload["results"].values())
            status = (0 if payload["protocol_complete"] else 1) if extra["pairs"] is None \
                else (0 if payload["stopped"] is None and flown == len(extra["pairs"]) else 1)
        elif args.backend == "local":
            from ontology_rgat.current_semantics.coupling import run_local_checks
            payload = run_local_checks(config)
            status = 0 if payload["local_checks_passed"] else 1
        else:
            payload = _isaac_coupling(config, root, extra)
            status = 0 if payload["isaac_checks_passed"] else 1
        artifact = ("coupling_isaac" if args.backend == "isaac" else
                    "isaac_eval" if args.stage == "isaac-eval" else args.stage)
        path = (Path(extra["out_dir"]) / "report.json" if args.stage == "isaac-trace"
                else root / ARTIFACTS[artifact])
        write_json(path, payload)
    print(f"{args.stage}: wrote {path} (exit {status})")
    return status


if __name__ == "__main__":
    sys.exit(main())
