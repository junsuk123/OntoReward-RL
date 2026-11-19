"""Live monitor of everything the current-semantics study is doing now, read-only.

The development dashboard (``dashboard.py``) follows ONE registered four-arm run.  This
page answers "what is running and how far is it" across the whole study:

  * processes  -- every live ``run_current_semantics_3d.py`` stage, design script
                  (``development_design/*.py``) and Isaac/PX4 stack process;
  * runs       -- every training directory with a ``history.jsonl`` under a task root
                  (G4 pilots, design-only Raw runs, development models): update,
                  difficulty, rolling success, validation, ETA, and whether it is
                  running, stalled or done;
  * tasks      -- per task root (v5 / v6 / v7): the G4 verdict, the design
                  calibration verdict, the development registration and result, and
                  the Isaac flights on record;
  * confirmation -- the registered confirmation runs (``confirmation*/``): models trained
                  and evaluated, the amendment's arm set and rule, Isaac flights per arm,
                  interruptions and, once written, the verdict;
  * encoding   -- for a graph-arm run (R-GAT or untyped GNN), its newest checkpoint flown
                  for one probe episode with the encoder traced: per-edge attention per hop,
                  node-state norms and the 8-D LandingContext readout per decision
                  (``encoding_view.py``, ``/api/encoding``), refreshed whenever the run writes
                  a newer checkpoint;
  * logs       -- the tails of the logs written in the last hour;
  * machine    -- load, memory and GPU.

``snapshot()`` only reads files and ``ps``/``nvidia-smi`` output; nothing is written,
locked or started.  ``serve`` publishes ``/`` and ``/api/state`` on 127.0.0.1 by default
(``--host`` binds another address, e.g. this machine's Tailscale address, so the page is
reachable from the tailnet only);
``./run.sh current-semantics-3d --stage monitor`` runs it in the foreground.
"""
from __future__ import annotations

import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from .contracts import PROJECT_ROOT
from .dashboard import _cached, _finite, _json, _jsonl, _thin, UNSAFE

RESULTS = PROJECT_ROOT / "results" / "current_semantics_3d"
TASK_LABELS = {"spatial-current-task/5": "과제 v5 (조합: 방향 보류)",
               "spatial-current-task/6": "과제 v6 (조합: 상대 기하 보류)",
               "spatial-current-task/7": "과제 v7 (결합: 가장자리·빠른 패드)"}
# A run whose history has not grown for this long and has no final.pt is "stalled".
STALL_S = 300.0
LOG_WINDOW_S = 3600.0
# Runs and task verdicts untouched for longer than this are past records and are not shown
# (live runs and registered confirmation runs always are).
RECENT_S = 24 * 3600.0
PROCESS_KEYS = ("run_current_semantics_3d.py", "development_design/", "isaac-sim", "kit/python",
                "px4", "MicroXRCEAgent", "gateway")
_GPU: dict[str, Any] = {"t": 0.0, "value": None}


# ------------------------------------------------------------------ pieces

def _rolling(values: list, window: int = 10) -> list:
    out, queue = [], []
    for v in values:
        queue.append(v)
        if len(queue) > window:
            queue.pop(0)
        good = [x for x in queue if x is not None]
        out.append(sum(good) / len(good) if good else None)
    return out


CONFIRMATION_DIRS = {"confirmation_mini": "축소 확인 실험", "confirmation": "확인 실험"}


def _run_kind(rel: Path) -> str:
    parts = rel.parts
    if parts[0] in CONFIRMATION_DIRS:
        return f"{CONFIRMATION_DIRS[parts[0]]} · {parts[-1]}"
    if parts[0] == "g4_pilot":
        return "G4 pilot"
    if parts[0] == "development":
        return "development"
    if parts[0] == "development_design":
        return "design (등록 아님)"
    if parts[0] == "relation_shuffle":
        return "관계 섞기 대조군"
    if parts[0] in ("orl", "orl2", "orl3", "orl4", "orl5"):
        name = "ORL" if parts[0] == "orl" else f"ORL/{parts[0][3:]}"
        if parts[1] == "control":  # orl/4-/5's rule control (MLP + view rule)
            name, parts = f"{name} 대조군(MLP+규칙)", parts[1:]
        return f"{name} " + ("G4 pilot" if parts[1] == "g4_pilot" else parts[1])
    return parts[0]


def _planned_updates(task_root: Path, run_dir: Path) -> int:
    """600 is the registered budget of every G4 pilot, design run and development model; a
    confirmation registers its own (``updates``)."""
    rel = run_dir.relative_to(task_root).parts
    if rel and rel[0] in CONFIRMATION_DIRS:
        reg = _cached(task_root / rel[0] / "confirmation_registration.json", _json) or {}
        return int(reg.get("updates") or 600)
    for candidate in (task_root / "development_registration.json",):
        data = _cached(candidate, _json) if candidate.exists() else None
        if data and data.get("updates") and "development" in run_dir.parts:
            return int(data["updates"])
    return 600


_RUN_CACHE: dict[Path, tuple[tuple, dict[str, Any]]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def run_state(results: Path, task_root: Path, run_dir: Path, now: float) -> dict[str, Any]:
    """``_run_state`` cached until one of the run's files changes; only the age-dependent
    fields (state, seconds since the last update, ETA) are recomputed on every call."""
    key = tuple(_mtime(run_dir / name) for name in ("history.jsonl", "validation.jsonl",
                                                     "final.pt", "best.pt", "status.json",
                                                     "summary.json", "claim"))
    hit = _RUN_CACHE.get(run_dir)
    if hit is None or hit[0] != key:
        hit = (key, _run_state(results, task_root, run_dir, now))
        _RUN_CACHE[run_dir] = hit
    state = dict(hit[1])
    age = None if not key[0] else now - key[0]
    state["seconds_since_update"] = age
    if state["state"] in ("running", "stalled"):
        state["state"] = "running" if age is not None and age < STALL_S else "stalled"
    update, wall, planned = state["update"], state["wall_s"], state["planned"]
    state["eta_s"] = (wall / (update + 1) * max(planned - update - 1, 0)
                      if state["state"] == "running" and update and wall else None)
    return state


def _run_state(results: Path, task_root: Path, run_dir: Path, now: float) -> dict[str, Any]:
    rows = _cached(run_dir / "history.jsonl", _jsonl) or []
    vrows = _cached(run_dir / "validation.jsonl", _jsonl) or []
    planned = _planned_updates(task_root, run_dir)
    last = rows[-1] if rows else {}
    try:
        age = now - (run_dir / "history.jsonl").stat().st_mtime
    except FileNotFoundError:
        age = None
    final = (run_dir / "final.pt").exists()
    phase = (_cached(run_dir / "status.json", _json) or {}).get("phase")
    claim = run_dir / "claim"
    excluded = claim.exists() and "excluded-arm-holder" in claim.read_text(errors="replace")
    if excluded:
        state = "excluded"
    elif phase == "evaluating" and not (run_dir / "summary.json").exists():
        state = "evaluating"
    else:
        state = ("done" if final else "running" if age is not None and age < STALL_S
                 else "stalled" if rows else "pending")
    update = last.get("update")
    wall = _finite(last.get("wall_s"))
    eta = None
    if state == "running" and update is not None and wall and update > 0:
        eta = wall / (update + 1) * max(planned - update - 1, 0)
    statuses = [r.get("statuses") or {} for r in rows]
    unsafe = [sum(v for k, v in s.items() if k in UNSAFE) / max(int(r.get("episodes") or 0), 1)
              for s, r in zip(statuses, rows)]
    success = [_finite(r.get("success")) for r in rows]
    full = next((r.get("update") for r in rows if (r.get("difficulty") or 0) >= 1.0), None)
    best = None
    if vrows:
        keyed = [((v.get("deterministic") or {}).get("success") or 0,
                  (v.get("sampled") or {}).get("success") or 0, v.get("update")) for v in vrows]
        best = max(keyed)
    log_std = last.get("log_std") or []
    return {
        "path": str(run_dir.relative_to(results)),
        "kind": _run_kind(run_dir.relative_to(task_root)),
        "state": state, "update": update, "planned": planned,
        "difficulty": _finite(last.get("difficulty")),
        "success_10": _rolling(success)[-1] if success else None,
        "return_10": _rolling([_finite(r.get("mean_return")) for r in rows])[-1] if rows else None,
        "log_std": sum(log_std) / len(log_std) if log_std else None,
        "accepted": last.get("accepted_steps"), "rejected": last.get("rejected_steps"),
        "full_difficulty_at": full, "seconds_since_update": age, "eta_s": eta,
        "wall_s": wall, "best_validation": None if best is None else
        {"deterministic": best[0], "sampled": best[1], "update": best[2]},
        "has_best": (run_dir / "best.pt").exists(),
        "series": {
            "update": _thin([r.get("update") for r in rows]),
            "success": _thin(_rolling(success)),
            "difficulty": _thin([_finite(r.get("difficulty")) for r in rows]),
            "unsafe": _thin(_rolling(unsafe)),
            "val_update": [v.get("update") for v in vrows],
            "val_det": [(v.get("deterministic") or {}).get("success") for v in vrows],
            "val_smp": [(v.get("sampled") or {}).get("success") for v in vrows]},
    }


def _history_dirs(task_root: Path) -> list[Path]:
    """Live-layout training directories only; archived pilots (g4_pilot_v*) are skipped."""
    patterns = ("g4_pilot/seed*/history.jsonl", "development/seed*/*/history.jsonl",
                "development_design/*/seed*/history.jsonl",
                "relation_shuffle/seed*/shuffle*/history.jsonl",
                "orl/g4_pilot/seed*/history.jsonl", "orl/development/seed*/history.jsonl",
                "orl2/g4_pilot/seed*/history.jsonl", "orl2/development/seed*/history.jsonl",
                "orl3/g4_pilot/seed*/history.jsonl", "orl3/development/seed*/history.jsonl",
                "orl4/g4_pilot/seed*/history.jsonl", "orl4/development/seed*/history.jsonl",
                "orl4/control/g4_pilot/seed*/history.jsonl",
                "orl4/control/development/seed*/history.jsonl",
                "orl5/g4_pilot/seed*/history.jsonl", "orl5/development/seed*/history.jsonl",
                "orl5/control/g4_pilot/seed*/history.jsonl",
                "orl5/control/development/seed*/history.jsonl",
                "confirmation_mini/development/seed*/*/history.jsonl",
                "confirmation/development/seed*/*/history.jsonl")
    found = []
    for pattern in patterns:
        found += [p.parent for p in task_root.glob(pattern)]
    return sorted(set(found))


def _verdicts(results: Path, task: str, root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"task": task, "label": TASK_LABELS.get(task, task),
                           "root": str(root.relative_to(results)) if root.exists() else None}
    if not root.exists():
        return out
    g4 = _cached(root / "g4_pilot.json", _json)
    if g4:
        out["g4"] = {"protocol": g4.get("protocol"), "passed": g4.get("learnability_passed"),
                     "tests": {k: [(t.get("deterministic") or {}).get("success"),
                                   (t.get("sampled") or {}).get("success")]
                               for k, t in (g4.get("tests") or {}).items()}}
    elif (root / "g4_pilot").exists():
        out["g4"] = {"protocol": None, "passed": None, "running": True}
    calibrations = []
    for path in sorted((root / "development_design").glob("calibration*.json")):
        data = _cached(path, _json) or {}
        passed = data.get("passed", data.get("composition_passed"))
        calibrations.append({"file": path.name, "passed": passed,
                             "summary": {k: data.get(k) for k in
                                         ("held", "trained", "composition_gap", "ood_levels")
                                         if k in data}})
    out["calibrations"] = calibrations
    reg = _cached(root / "development_registration.json", _json)
    result = _cached(root / "development.json", _json)
    out["development"] = None if reg is None else {
        "protocol": reg.get("protocol"), "registered_utc": reg.get("registered_utc"),
        "complete": (result or {}).get("procedure_complete"),
        "gate": ((result or {}).get("verdict") or {}).get("development_gate_passed")}
    shuffle_reg = _cached(root / "relation_shuffle_registration.json", _json)
    shuffle = _cached(root / "relation_shuffle.json", _json)
    verdict = (shuffle or {}).get("verdict") or {}
    out["relation_shuffle"] = None if shuffle_reg is None else {
        "protocol": shuffle_reg.get("protocol"), "models": shuffle_reg.get("models"),
        "complete": (shuffle or {}).get("procedure_complete"),
        "verdict": verdict.get("semantic_placement"),
        "delta": (verdict.get("primary") or {}).get("mean"),
        "ci95": (verdict.get("primary") or {}).get("ci95")}
    flights = []
    for path in sorted(root.glob("isaac_*.json")):
        if path.name.endswith("registration.json"):
            continue
        data = _cached(path, _json) or {}
        success = data.get("success")
        flights.append({"file": path.name, "protocol": data.get("protocol"),
                        "complete": data.get("complete", data.get("protocol_complete")),
                        "verified": data.get("fix_verified", data.get("isaac_eval_passed")),
                        "success": success if isinstance(success, dict) else None,
                        "episodes": data.get("episodes")})
    out["isaac"] = flights
    return out


def label(args: str) -> str:
    """The script and its arguments with every path shortened to its last component."""
    tokens = args.split()
    start = next((i for i, t in enumerate(tokens) if t.endswith(".py")), 0)
    return " ".join(t.rstrip("/").rsplit("/", 1)[-1] if "/" in t else t
                    for t in tokens[start:])[:200]


def parse_processes(text: str, me: int = -1) -> list[dict[str, Any]]:
    """``ps -eo pid=,ppid=,etimes=,pcpu=,rss=,args=`` output -> the study's processes, each
    with its spawned workers (``spawn_main`` children, whose command line does not name
    the job) folded into ``workers`` / ``worker_cpu``.  Shell wrappers are dropped."""
    rows = {}
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        pid, ppid, etimes, pcpu, rss, args = parts
        rows[int(pid)] = {"pid": int(pid), "ppid": int(ppid), "elapsed_s": int(etimes),
                          "cpu": float(pcpu), "rss_mb": int(rss) / 1024.0, "args": args[:400],
                          "label": label(args)}
    keep = {pid: r for pid, r in rows.items()
            if pid != me and any(k in r["args"] for k in PROCESS_KEYS)
            and not r["args"].startswith(("/bin/bash -c", "bash -c", "ps ", "pgrep"))
            and "spawn_main" not in r["args"]}
    for r in rows.values():
        if "spawn_main" in r["args"] or "multiprocessing.resource_tracker" in r["args"]:
            parent = keep.get(r["ppid"])
            if parent is not None:
                parent["workers"] = parent.get("workers", 0) + ("spawn_main" in r["args"])
                parent["worker_cpu"] = parent.get("worker_cpu", 0.0) + r["cpu"]
                parent["rss_mb"] += r["rss_mb"]
    return sorted(keep.values(), key=lambda r: -r["elapsed_s"])


def processes() -> list[dict[str, Any]]:
    try:
        text = subprocess.run(["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,rss=,args="],
                              capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_processes(text, os.getpid())


def machine() -> dict[str, Any]:
    out: dict[str, Any] = {"cpus": os.cpu_count()}
    try:
        out["load"] = list(os.getloadavg())
    except OSError:
        out["load"] = None
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
        out["mem_used_gb"] = (info["MemTotal"] - info["MemAvailable"]) / 2**20
        out["mem_total_gb"] = info["MemTotal"] / 2**20
    except (OSError, KeyError, ValueError):
        pass
    now = time.monotonic()
    if now - _GPU["t"] > 10.0:
        try:
            text = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=2).stdout.strip().splitlines()
            util, used, total = (float(x) for x in text[0].split(","))
            _GPU["value"] = {"util": util, "mem_used_gb": used / 1024, "mem_total_gb": total / 1024}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            _GPU["value"] = None
        _GPU["t"] = now
    out["gpu"] = _GPU["value"]
    return out


def _encoding_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Graph-arm runs that have a checkpoint to trace: live confirmation runs first."""
    from .encoding_view import GRAPH_ARMS
    out = []
    for r in runs:
        name = Path(r["path"]).name
        if name in GRAPH_ARMS and r.get("has_best") is not None:
            final = (RESULTS / r["path"] / "final.pt").exists()
            if r.get("has_best") or final:
                out.append({"path": r["path"], "arm": name, "state": r["state"],
                            "kind": r["kind"], "task": r.get("task"),
                            "confirmation": "confirmation" in r["path"]})
    live = {"running": 0, "evaluating": 0, "stalled": 1}
    out.sort(key=lambda r: (not r["confirmation"], live.get(r["state"], 2),
                            r["arm"] != "ontology_rgat_ppo", r["path"]))
    return out


def _isaac_stack_alive() -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(Path(t).name == "landing_world.py" for t in tokens[:3]):
            return True
    return False


def confirmation_state(results: Path, task: str, root: Path) -> list[dict[str, Any]]:
    """Every registered confirmation run of one root, as the monitor shows it."""
    out = []
    for folder, title in CONFIRMATION_DIRS.items():
        base = root / folder
        reg = _cached(base / "confirmation_registration.json", _json) if base.exists() else None
        if not reg:
            continue
        amendment = _cached(base / "confirmation_amendment.json", _json)
        plan = reg.get("plan") or {}
        arms = (amendment or {}).get("arms") or plan.get("models", {}).get("arms") or []
        seeds = reg.get("training_seeds") or []
        analysis = (amendment or {}).get("analysis") or reg.get("analysis") or {}
        conditions = (plan.get("isaac") or {}).get("conditions") or {}
        cases = sum(len(v) for v in conditions.values())
        models = []
        for seed in seeds:
            for arm in arms:
                d = base / "development" / f"seed{seed}" / arm
                summary = _cached(d / "summary.json", _json) if (d / "summary.json").exists() else None
                rows = _cached(d / "history.jsonl", _jsonl) if (d / "history.jsonl").exists() else []
                phase = (_cached(d / "status.json", _json) or {}).get("phase")
                cells = (summary or {}).get("cells") or {}
                ten = ([c.get("success") for c in cells.values() if c.get("success") is not None]
                       if cells else [])
                val = (summary or {}).get("validation") or {}
                models.append({"seed": seed, "arm": arm,
                               "state": ("done" if summary else "evaluating" if phase == "evaluating"
                                         else "training" if rows else "pending"),
                               "update": len(rows), "difficulty": (rows[-1].get("difficulty")
                                                                    if rows else None),
                               "ten_cell_success": sum(ten) / len(ten) if ten else None,
                               "best_update": val.get("best_update"),
                               "validation_det": val.get("deterministic_success")})
        isaac_rows = _cached(base / "isaac" / "episodes.jsonl", _jsonl) \
            if (base / "isaac" / "episodes.jsonl").exists() else []
        per_arm = {}
        for arm in arms:
            own = [r for r in isaac_rows or [] if r.get("arm") == arm]
            per_arm[arm] = {"flown": len(own),
                            "success": sum(r.get("status") == "SUCCESS" for r in own),
                            "risk": sum(r.get("status") in ("UNSAFE_CONTACT",
                                                            "SAFETY_ENVELOPE_VIOLATION")
                                        for r in own),
                            "other": sum(r.get("status") not in ("SUCCESS", "UNSAFE_CONTACT",
                                                                 "SAFETY_ENVELOPE_VIOLATION")
                                         for r in own)}
        interruptions = _cached(base / "isaac" / "interruptions.jsonl", _jsonl) \
            if (base / "isaac" / "interruptions.jsonl").exists() else []
        result = _cached(base / "confirmation.json", _json) if (base / "confirmation.json").exists() \
            else None

        def verdict(part):
            v = ((result or {}).get(part) or {}).get("verdict")
            if not v:
                return None
            rows_ = {}
            for arm, iv in {**(v.get("primary") or {}), **(v.get("practical") or {})}.items():
                rows_[arm] = {k: iv.get(k) for k in ("mean", "low", "high", "confidence", "passed")}
            return {"means": v.get("means"), "comparisons": rows_,
                    "passed": v.get("confirmation_passed", v.get("isaac_passed"))}
        out.append({
            "task": task, "folder": str(base.relative_to(results)), "title": title,
            "protocol": reg.get("protocol"), "registered_utc": reg.get("registered_utc"),
            "amendment": None if not amendment else {
                "amended_utc": amendment.get("amended_utc"), "arms": amendment.get("arms"),
                "reason": amendment.get("reason")},
            "arms": arms, "seeds": seeds, "updates": reg.get("updates"),
            "primary_comparisons": analysis.get("primary_comparisons"),
            "models": models,
            "models_done": sum(m["state"] == "done" for m in models),
            "isaac": {"expected": cases * len(models), "flown": len(isaac_rows or []),
                      "per_arm": per_arm, "stack_alive": _isaac_stack_alive(),
                      "last": [{k: r.get(k) for k in ("condition", "seed", "model", "status",
                                                      "elapsed_s", "wall_s")}
                               for r in (isaac_rows or [])[-8:]][::-1],
                      "interruptions": (interruptions or [])[-5:]},
            "verdict": {"local": verdict("local"), "isaac": verdict("isaac")}})
    return out


def _tail(path: Path, lines: int = 8) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - 8192, 0))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [line[:300] for line in text.splitlines()[-lines:]]


def recent_logs(results: Path, roots: list[Path], now: float) -> list[dict[str, Any]]:
    out = []
    for root in roots:
        for path in (list(root.glob("*.log")) + list(root.glob("development_design/*.log"))
                     + [q for d in CONFIRMATION_DIRS for q in root.glob(f"{d}/*.log")]):
            try:
                age = now - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= LOG_WINDOW_S:
                out.append({"path": str(path.relative_to(results)), "age_s": age,
                            "tail": _tail(path)})
    return sorted(out, key=lambda r: r["age_s"])


# ------------------------------------------------------------------ state

def snapshot(results: Path | None = None) -> dict[str, Any]:
    from .experiment import NAVIGATION_ROOT_TAG, TASK_RUN_ROOTS
    results = RESULTS if results is None else Path(results)
    now = time.time()
    roots = {task: (results / root.name) for task, root in TASK_RUN_ROOTS.items()}
    # observation-profile /4 roots (experiment.config_run_root), shown under their task
    for task, root in list(roots.items()):
        for nav_root in sorted(results.glob(f"{root.name}{NAVIGATION_ROOT_TAG}*")):
            if nav_root.is_dir():
                tag = nav_root.name.split(NAVIGATION_ROOT_TAG, 1)[1]
                roots[f"{task} obs/4 {tag}"] = nav_root
    runs = []
    for task, root in roots.items():
        if not root.exists():
            continue
        for run_dir in _history_dirs(root):
            try:
                state = run_state(results, root, run_dir, now)
            except (OSError, ValueError):
                continue
            state["task"] = task
            runs.append(state)
    runs = [r for r in runs if r["state"] in ("running", "evaluating")
            or "confirmation" in r["path"]
            or (r.get("seconds_since_update") is not None
                and r["seconds_since_update"] <= RECENT_S)]
    active_tasks = {r["task"] for r in runs}
    for task, root in roots.items():
        if root.exists() and any(now - _mtime(f) <= RECENT_S for f in root.iterdir()
                                 if f.is_file()):
            active_tasks.add(task)
    order = {"running": 0, "evaluating": 0, "stalled": 1, "pending": 2, "done": 3, "excluded": 4}
    runs.sort(key=lambda r: (order.get(r["state"], 9), r["path"]))
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "results": str(results),
        "machine": machine(),
        "processes": processes(),
        "runs": runs,
        "tasks": [_verdicts(results, task, root) for task, root in roots.items()
                  if task in active_tasks],
        "recent_window_h": RECENT_S / 3600.0,
        "encoding_runs": _encoding_runs(runs),
        "confirmations": [c for task, root in roots.items() if root.exists()
                          for c in confirmation_state(results, task, root)],
        "logs": recent_logs(results, [r for r in roots.values() if r.exists()], now),
    }


# ------------------------------------------------------------------ server

def _handler(results: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib interface
            route = self.path.split("?")[0].rstrip("/")
            if route in ("", "/index.html"):
                body, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif route == "/api/encoding":
                from urllib.parse import parse_qs, urlsplit
                from .encoding_view import view
                query = parse_qs(urlsplit(self.path).query)
                try:
                    payload = view(results or RESULTS, query.get("run", [""])[0],
                                   query.get("probe", ["nominal"])[0])
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                  default=lambda o: None).encode("utf-8")
                kind = "application/json; charset=utf-8"
            elif route == "/api/state":
                body = json.dumps(snapshot(results), ensure_ascii=False, allow_nan=False,
                                  default=lambda o: None).encode("utf-8")
                kind = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return
    return Handler


def serve(*, host: str = "127.0.0.1", port: int = 8780, results: Path | None = None,
          background: bool = True,
          open_browser: bool | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Start the monitor on ``port`` .. ``port + 19`` (the first free one)."""
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), _handler(results))
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"no free port in {port}-{port + 19}")
    server.daemon_threads = True
    url = f"http://{host}:{server.server_address[1]}/"
    if open_browser is None:
        open_browser = not os.environ.get("CS3D_DASHBOARD_NO_BROWSER")
    if open_browser:
        def _open():
            try:
                import webbrowser
                webbrowser.open_new_tab(url)
            except Exception:  # noqa: BLE001 - headless: the URL is printed
                pass
        threading.Thread(target=_open, daemon=True).start()
    if background:
        threading.Thread(target=server.serve_forever, daemon=True, name="cs3d-monitor").start()
    return server, url


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Study Monitor</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d232b;--muted:#68717d;--line:#e3e6ea;--grid:#eef0f3;
--a:#2f6fdb;--b:#0f9d6e;--c:#d98a0b;--bad:#c2410c;--good:#0f9d6e;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#191d23;--ink:#e6e9ee;--muted:#9aa3ad;
--line:#2a3038;--grid:#20252c;--a:#6b9ef5;--b:#34c38f;--c:#f0a73a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
header{padding:14px 16px 6px;display:flex;flex-wrap:wrap;gap:10px 16px;align-items:baseline}
h1{font-size:18px;margin:0;font-weight:650}.sub{color:var(--muted);font-size:12px}
main{padding:0 16px 24px;display:grid;gap:12px;grid-template-columns:repeat(12,minmax(0,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
.card h2{font-size:12px;margin:0 0 8px;font-weight:650;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.s12{grid-column:span 12}.s6{grid-column:span 6}.s4{grid-column:span 4}
@media (max-width:1100px){.s6,.s4{grid-column:span 12}}
.kv{display:flex;flex-wrap:wrap;gap:18px}.kv b{display:block;font-size:17px}.kv span{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:left;font-size:13px;vertical-align:top}
th{color:var(--muted);font-weight:600}td.n,th.n{text-align:right}
.wrap{overflow-x:auto}
.pill{display:inline-block;white-space:nowrap;flex:none;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;background:var(--grid)}
.running{background:#dbeafe;color:#1e40af}.done{background:#d1fae5;color:#065f46}
.stalled{background:#fee2e2;color:#991b1b}.pending{background:var(--grid);color:var(--muted)}
.evaluating{background:#ede9fe;color:#5b21b6}.excluded{background:var(--grid);color:var(--muted);text-decoration:line-through}
@media (prefers-color-scheme:dark){.running{background:#1e3a5f;color:#bfdbfe}.done{background:#064e3b;color:#a7f3d0}
.stalled{background:#5f1e1e;color:#fecaca}}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}.mid{color:var(--warn);font-weight:600}
.runs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.run{border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
.run .t{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.run .p{font-weight:600;word-break:break-all;font-size:13px}
.bar{height:7px;background:var(--grid);border-radius:4px;overflow:hidden;margin:6px 0}.bar>i{display:block;height:100%;background:var(--a)}
.legend{display:flex;gap:12px;font-size:11px;color:var(--muted)}.legend i{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}
pre{margin:4px 0 0;padding:8px;background:var(--grid);border-radius:6px;font-size:11.5px;line-height:1.35;
white-space:pre-wrap;word-break:break-all;max-height:170px;overflow:auto}
svg text{fill:var(--muted);font-size:10px}.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}
.empty{color:var(--muted);font-size:13px}
.enc-bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:6px}
.enc-bar select,.enc-bar button{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
.enc{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:12px;margin-top:6px}
@media (max-width:900px){.enc{grid-template-columns:minmax(0,1fr)}}
</style></head><body>
<header><h1>현재 상태 모니터</h1><span class="sub" id="meta">불러오는 중…</span></header>
<main>
<section class="card s12"><h2>확인 실험 (실시간)</h2><div id="conf"></div></section>
<section class="card s12"><h2>온톨로지 R-GAT 인코딩 (실시간, 최신 체크포인트)</h2>
<div class="enc-bar"><label>실행 <select id="encRun"></select></label>
<label>프로브 <select id="encProbe"><option value="nominal">기본 조건</option><option value="blackout">시야 차단 1–3 s</option></select></label>
<label>홉 <select id="encHop"><option value="1">2 (판독 직전)</option><option value="0">1</option></select></label>
<button id="encPlay">▶ 재생</button><input id="encStep" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
<span class="sub" id="encT"></span></div>
<div class="sub" id="encMeta">실행을 고르면 그 실행의 최신 체크포인트로 프로브 에피소드 하나를 돌려 인코더 내부를 기록합니다.</div>
<div class="enc"><div><svg id="encGraph" viewBox="0 0 660 445" width="100%"></svg><div class="legend" id="encLegend"></div></div>
<div><div class="sub">LandingContext 판독 g<sub>t</sub> (8차원, 현재 시점)</div><svg id="encBars" viewBox="0 0 320 150" width="100%"></svg>
<div class="sub">g<sub>t</sub> 시간 변화 (행 = 차원, 열 = 결정 시점, 파랑 음수 · 빨강 양수)</div><svg id="encHeat" viewBox="0 0 320 120" width="100%" preserveAspectRatio="none" style="height:120px"></svg>
<div class="sub">정책이 본 패드 추정 높이 h와 시야 나이</div><svg id="encCtx" viewBox="0 0 320 90" width="100%"></svg></div></div>
<div class="sub" id="encNote"></div></section>
<section class="card s12"><h2>기계</h2><div class="kv" id="machine"></div></section>
<section class="card s12"><h2>학습 실행 (실행 중 + 최근 24시간)</h2><div class="runs" id="runs"></div></section>
<section class="card s12"><h2>실행 중 프로세스</h2><div class="wrap" id="procs"></div></section>
<section class="card s12"><h2>과제별 판정 (최근 24시간 안에 바뀐 과제)</h2><div class="wrap" id="tasks"></div></section>
<section class="card s12"><h2>최근 로그 (1시간 안에 쓰인 것)</h2><div id="logs"></div></section>
</main>
<script>
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=x=>x==null?"–":(100*x).toFixed(1)+"%";
function dur(s){if(s==null)return"–";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
 return h?`${h}시간 ${m}분`:m?`${m}분 ${x}초`:`${x}초`}
const STATE={running:"학습 중",evaluating:"평가 중",done:"완료",stalled:"멈춤?",pending:"대기",excluded:"제외",training:"학습 중"};
const ARM={ontology_rgat_ppo:"Ontology R-GAT (제안)",raw_ppo:"PPO (raw obs.)",untyped_gnn_ppo:"Untyped GNN",semantic_mlp_ppo:"Semantic MLP"};
function conf(C){if(!C.length)return `<div class="empty">등록된 확인 실험이 없습니다.</div>`;
 return C.map(c=>{const n=c.models.length,done=c.models_done,I=c.isaac;
  const bar=(a,b)=>`<div class="bar"><i style="width:${b?(100*a/b).toFixed(1):0}%"></i></div>`;
  const head=`<div class="kv"><div><span>프로토콜</span><b>${esc(c.protocol)}</b></div><div><span>정책</span><b>${c.arms.map(a=>esc(ARM[a]||a)).join(" · ")}</b></div>
   <div><span>학습 시드</span><b>${c.seeds.length}개 (${c.seeds[0]}–${c.seeds[c.seeds.length-1]})</b></div><div><span>모델당 업데이트</span><b>${c.updates}</b></div>
   <div><span>모델 완료</span><b>${done} / ${n}</b></div><div><span>Isaac 비행</span><b>${I.flown} / ${I.expected} ${I.stack_alive?'<span class="pill running">스택 실행 중</span>':'<span class="pill pending">스택 꺼짐</span>'}</b></div></div>
   <div class="sub">모델 학습·평가</div>${bar(done+c.models.filter(m=>m.state!=="done").reduce((s,m)=>s+Math.min(m.update/(c.updates||1),1),0),n)}
   <div class="sub">Isaac 비행</div>${bar(I.flown,I.expected)}
   ${c.amendment?`<div class="sub">정정 ${esc(c.amendment.amended_utc)}: 정책 ${c.amendment.arms.map(a=>esc(ARM[a]||a)).join(", ")}</div>`:""}`;
  const seeds=[...new Set(c.models.map(m=>m.seed))];
  const grid=`<div class="wrap"><table><tr><th>시드</th>${c.arms.map(a=>`<th>${esc(ARM[a]||a)}</th>`).join("")}</tr>`+seeds.map(s=>`<tr><td>${s}</td>`+c.arms.map(a=>{const m=c.models.find(x=>x.seed===s&&x.arm===a);
    if(!m)return"<td>–</td>";const st=m.state==="training"?"running":m.state;
    const body=m.state==="done"?`10셀 ${pct(m.ten_cell_success)} · 검증 det ${pct(m.validation_det)} @${m.best_update}`:m.state==="pending"?"":`${m.update}/${c.updates} · 난이도 ${m.difficulty==null?"–":m.difficulty.toFixed(1)}`;
    return `<td><span class="pill ${st}">${STATE[m.state]||m.state}</span> <span class="sub">${body}</span></td>`}).join("")+"</tr>").join("")+"</table></div>";
  const isaac=`<div class="wrap"><table><tr><th>정책</th><th class="n">비행</th><th class="n">성공</th><th class="n">위험 판정</th><th class="n">그 밖의 실패</th></tr>`+
   c.arms.map(a=>{const r=I.per_arm[a]||{};return `<tr><td>${esc(ARM[a]||a)}</td><td class="n">${r.flown||0}</td><td class="n">${r.success||0}${r.flown?` (${pct(r.success/r.flown)})`:""}</td><td class="n">${r.risk||0}</td><td class="n">${r.other||0}</td></tr>`}).join("")+"</table></div>"+
   (I.last.length?`<div class="sub" style="margin-top:6px">최근 비행</div><pre>${esc(I.last.map(r=>`${r.condition} ${r.seed} ${r.model} -> ${r.status} (${(r.elapsed_s||0).toFixed(1)} s)`).join("\n"))}</pre>`:"")+
   (I.interruptions.length?`<div class="sub">인프라 중단 ${I.interruptions.length}건 (판정된 에피소드는 다시 비행하지 않음)</div><pre>${esc(I.interruptions.map(x=>x.error).join("\n"))}</pre>`:"");
  const V=c.verdict,vt=(name,v)=>!v?"":`<div class="sub" style="margin-top:8px">${name} 판정: ${v.passed?'<span class="ok">확인됨</span>':'<span class="no">확인되지 않음</span>'}</div><table><tr><th>R-GAT − 비교 정책</th><th class="n">평균</th><th class="n">구간</th><th>판정</th></tr>`+
   Object.entries(v.comparisons).map(([a,x])=>`<tr><td>${esc(ARM[a]||a)}</td><td class="n">${pct(x.mean)}</td><td class="n">[${pct(x.low)}, ${pct(x.high)}] ${(100*x.confidence).toFixed(1)}%</td><td>${x.passed?'<span class="ok">통과</span>':'<span class="no">미통과</span>'}</td></tr>`).join("")+"</table>";
  return head+`<div style="margin-top:8px">${grid}</div><div style="margin-top:8px">${isaac}</div>`+vt("로컬",V.local)+vt("Isaac",V.isaac)+
   `<div class="sub" style="margin-top:6px">완료 모델의 10셀 성공률은 개별 모델 값이며 비교 판정이 아닙니다. 판정은 모든 모델과 비행이 끝난 뒤 사전 등록 규칙으로 계산합니다.</div>`}).join("<hr>")}
function spark(r){const s=r.series,W=320,H=110,L=30,R=6,T=6,B=16;const xs=s.update.filter(x=>x!=null);
 const x1=Math.max(r.planned||600,...xs,10);const X=x=>L+x/x1*(W-L-R),Y=y=>T+(1-y)*(H-T-B);
 let g="";for(const y of [0,.5,1])g+=`<line x1="${L}" x2="${W-R}" y1="${Y(y)}" y2="${Y(y)}" stroke="${css('--grid')}"/><text x="${L-4}" y="${Y(y)+3}" text-anchor="end">${y*100}%</text>`;
 g+=`<text x="${W-R}" y="${H-3}" text-anchor="end">${x1} update</text>`;
 const line=(ys,col,w)=>{let d="",pen=false;s.update.forEach((x,i)=>{const y=ys[i];if(y==null||x==null){pen=false;return}
   d+=(pen?"L":"M")+X(x).toFixed(1)+","+Y(Math.max(0,Math.min(1,y))).toFixed(1);pen=true});
   return d?`<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}"/>`:""};
 let p=line(s.difficulty,css('--c'),1.4)+line(s.unsafe,css('--bad'),1)+line(s.success,css('--a'),2);
 s.val_update.forEach((u,i)=>{if(s.val_det[i]!=null)p+=`<circle cx="${X(u)}" cy="${Y(s.val_det[i])}" r="2.4" fill="${css('--b')}"/>`});
 return `<svg width="100%" viewBox="0 0 ${W} ${H}">${g}${p}</svg>`}
function render(S){
 document.getElementById("conf").innerHTML=conf(S.confirmations||[]);
 encOptions(S);
 document.getElementById("meta").textContent=`갱신 ${new Date(S.generated_utc).toLocaleTimeString()} · 5초마다 · 읽기 전용`;
 const m=S.machine,g=m.gpu;
 document.getElementById("machine").innerHTML=[["CPU 부하 (1/5/15분)",m.load?m.load.map(x=>x.toFixed(1)).join(" / ")+` · ${m.cpus}코어`:"–"],
  ["메모리",m.mem_total_gb?`${m.mem_used_gb.toFixed(1)} / ${m.mem_total_gb.toFixed(0)} GB`:"–"],
  ["GPU",g?`${g.util.toFixed(0)}% · ${g.mem_used_gb.toFixed(1)} / ${g.mem_total_gb.toFixed(0)} GB`:"–"],
  ["학습 중 실행",S.runs.filter(r=>r.state==="running").length],["프로세스",S.processes.length]]
  .map(([k,v])=>`<div><span>${k}</span><b>${v}</b></div>`).join("");
 const runs=S.runs.filter(r=>r.state!=="done").concat(S.runs.filter(r=>r.state==="done").slice(0,12));
 document.getElementById("runs").innerHTML=runs.length?runs.map(r=>{const frac=r.update==null?0:(r.update+1)/r.planned;
  const bv=r.best_validation;return `<div class="run"><div class="t"><span class="p">${esc(r.path)}</span><span class="pill ${r.state}">${STATE[r.state]||r.state}</span></div>
  <div class="sub">${esc(r.kind)} · ${esc((r.task||"").replace("spatial-current-",""))}</div>
  <div class="bar"><i style="width:${(100*Math.min(frac,1)).toFixed(1)}%"></i></div>
  <div class="kv" style="gap:12px"><div><span>update</span><b>${r.update==null?"–":r.update+1} / ${r.planned}</b></div>
  <div><span>난이도</span><b>${r.difficulty==null?"–":r.difficulty.toFixed(1)}</b></div>
  <div><span>성공률(10)</span><b>${pct(r.success_10)}</b></div>
  <div><span>최고 검증 det/smp</span><b>${bv?`${pct(bv.deterministic)} / ${pct(bv.sampled)}`:"–"}</b></div>
  <div><span>${r.state==="running"?"남은 시간":"마지막 갱신"}</span><b>${r.state==="running"?dur(r.eta_s):dur(r.seconds_since_update)+" 전"}</b></div></div>
  ${spark(r)}<div class="legend"><span><i style="background:${css('--a')}"></i>학습 성공률(10)</span><span><i style="background:${css('--c')}"></i>난이도</span>
  <span><i style="background:${css('--bad')}"></i>unsafe(10)</span><span><i style="background:${css('--b')};height:6px;width:6px;border-radius:3px"></i>검증 det</span></div>
  <div class="sub">d=1.0 도달 ${r.full_difficulty_at==null?"–":"update "+r.full_difficulty_at} · log σ ${r.log_std==null?"–":r.log_std.toFixed(2)} · KL 수락/거부 ${r.accepted??"–"}/${r.rejected??"–"}</div></div>`}).join("")
  :`<div class="empty">학습 디렉터리가 없습니다.</div>`;
 const P=S.processes;document.getElementById("procs").innerHTML=P.length?`<table><tr><th>pid</th><th class="n">경과</th><th class="n">CPU</th><th class="n">worker</th><th class="n">메모리</th><th>명령</th></tr>`+
  P.map(p=>`<tr><td>${p.pid}</td><td class="n">${dur(p.elapsed_s)}</td><td class="n">${(p.cpu+(p.worker_cpu||0)).toFixed(0)}%</td><td class="n">${p.workers||"–"}</td><td class="n">${p.rss_mb.toFixed(0)} MB</td><td class="code" title="${esc(p.args)}">${esc(p.label)}</td></tr>`).join("")+"</table>"
  :`<div class="empty">실행 중인 study 프로세스가 없습니다.</div>`;
 const yes=v=>v===true?`<span class="ok">통과</span>`:v===false?`<span class="no">실패</span>`:`<span class="mid">진행/없음</span>`;
 document.getElementById("tasks").innerHTML=`<table><tr><th>과제</th><th>G4 pilot</th><th>설계 판정</th><th>development</th><th>Isaac 비행</th></tr>`+
  S.tasks.map(t=>{if(!t.root)return `<tr><td>${esc(t.label)}</td><td colspan="4" class="sub">결과 폴더 없음</td></tr>`;
   const g4=t.g4?`${esc(t.g4.protocol||"학습 중")} ${t.g4.running?"":yes(t.g4.passed)}<br><span class="sub">${Object.entries(t.g4.tests||{}).map(([k,v])=>`seed ${k}: ${pct(v[0])}/${pct(v[1])}`).join(" · ")}</span>`:"–";
   const cal=(t.calibrations||[]).map(c=>`${esc(c.file)} ${yes(c.passed)}`).join("<br>")||"–";
   const rs=t.relation_shuffle;const d=(t.development?`${esc(t.development.protocol)} · ${t.development.complete?"완료 · gate "+(t.development.gate?"통과":"미통과"):"등록됨"}`:"–")
    +(rs?`<br>${esc(rs.protocol)} · ${rs.complete?`판정: ${esc(rs.verdict)} <span class="sub">Δ ${pct(rs.delta)} [${pct(rs.ci95[0])}, ${pct(rs.ci95[1])}]</span>`:`등록됨 · 모델 ${rs.models}`}`:"");
   const isaac=(t.isaac||[]).map(f=>`${esc(f.file)} ${f.verified==null?"":yes(f.verified)}${f.success?` <span class="sub">${f.success.after??""}${f.episodes?"/"+f.episodes:""}</span>`:""}`).join("<br>")||"–";
   return `<tr><td>${esc(t.label)}<br><span class="sub">${esc(t.root)}</span></td><td>${g4}</td><td>${cal}</td><td>${d}</td><td>${isaac}</td></tr>`}).join("")+"</table>";
 document.getElementById("logs").innerHTML=S.logs.length?S.logs.map(l=>`<div style="margin-bottom:10px"><span class="code">${esc(l.path)}</span> <span class="sub">${dur(l.age_s)} 전</span><pre>${esc(l.tail.join("\n"))}</pre></div>`).join("")
  :`<div class="empty">최근 1시간 안에 쓰인 로그가 없습니다.</div>`;
}

const REL={qualifiesEstimate:"#2a78d6",informsApproach:"#eb6834",affectsVisibility:"#1baf7a",conditionsContact:"#eda100",informsLanding:"#e87ba4",self:"#9a9a9a",related:"#7a7a7a"};
const NPOS={ObservationQuality:[90,55],RelativeGeometry:[90,160],MotionState:[90,265],AttitudeState:[90,370],ApproachSituation:[340,110],ViewGeometry:[340,215],ContactCondition:[340,320],LandingContext:[570,215]};
const SHORT={ObservationQuality:"Q",RelativeGeometry:"G",MotionState:"M",AttitudeState:"A",ApproachSituation:"P",ViewGeometry:"O",ContactCondition:"C",LandingContext:"L"};
let ENC=null,encIdx=0,encTimer=null,encKey="",encMtime=null;
function encOptions(S){const sel=document.getElementById("encRun"),cur=sel.value;const R=S.encoding_runs||[];
 const html=R.map(r=>`<option value="${esc(r.path)}">${r.confirmation?"[확인] ":""}${esc(r.path.split("/").slice(-3).join("/"))} · ${STATE[r.state]||r.state}</option>`).join("");
 if(sel.dataset.h!==html){sel.innerHTML=html||`<option value="">체크포인트가 있는 그래프 정책 실행이 없습니다</option>`;sel.dataset.h=html;if(cur&&R.some(r=>r.path===cur))sel.value=cur}
 if(!encKey&&R.length)loadEnc()}
async function loadEnc(force){const run=document.getElementById("encRun").value,probe=document.getElementById("encProbe").value;if(!run)return;
 const key=run+"|"+probe;if(key!==encKey){ENC=null;encIdx=0;document.getElementById("encMeta").textContent="프로브 에피소드 계산 중… (몇 초 걸립니다)"}
 encKey=key;try{const r=await fetch(`api/encoding?run=${encodeURIComponent(run)}&probe=${probe}`,{cache:"no-store"});const d=await r.json();
  if(key!==encKey)return;if(d.error){document.getElementById("encMeta").textContent="오류: "+d.error;return}
  if(!d.checkpoint){ENC=null;document.getElementById("encMeta").textContent=d.note;drawEnc();return}
  if(ENC&&ENC.checkpoint_mtime===d.checkpoint_mtime&&!force)return;
  const changed=ENC&&ENC.checkpoint_mtime!==d.checkpoint_mtime;ENC=d;if(!changed)encIdx=0;encIdx=Math.min(encIdx,d.steps.length-1);
  const sl=document.getElementById("encStep");sl.max=d.steps.length-1;sl.value=encIdx;
  const when=new Date(d.checkpoint_mtime*1000).toLocaleTimeString();
  document.getElementById("encMeta").innerHTML=`${esc(ARM[d.arm]||d.arm)} · 체크포인트 <b>${esc(d.checkpoint)}</b> (update ${d.checkpoint_update??"–"}, ${when} 저장)${changed?' <span class="pill running">새 체크포인트</span>':""} · 프로브 ${esc(d.probe_cell)} 시드 ${d.probe_seed} → <b>${esc(d.status)}</b>, ${d.decisions}회 결정, 계산 ${d.compute_s}s`;
  document.getElementById("encNote").textContent=d.note+". 간선 굵기 = 어텐션 α(같은 노드로 들어오는 간선끼리 합 1), 노드 진하기 = 노드 상태 크기 ‖h‖.";
  drawEnc()}catch(e){document.getElementById("encMeta").textContent="인코딩을 불러오지 못했습니다"}}
function mix(a,b,t){const p=x=>[1,3,5].map(i=>parseInt(x.slice(i,i+2),16));const A=p(a),B=p(b);return`rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(",")})`}
function drawEnc(){const G=document.getElementById("encGraph"),B=document.getElementById("encBars"),H=document.getElementById("encHeat"),C=document.getElementById("encCtx");
 if(!ENC){[G,B,H,C].forEach(x=>x.innerHTML="");document.getElementById("encT").textContent="";return}
 const d=ENC,st=d.steps[encIdx],hop=Math.min(+document.getElementById("encHop").value,d.hops-1),al=st.alpha[hop],nm=st.norm[hop+1];
 document.getElementById("encT").textContent=`t = ${st.t.toFixed(1)} s · ${encIdx+1}/${d.steps.length}`;
 const ink=css('--ink'),muted=css('--muted'),panel=css('--panel'),line=css('--line');
 let defs="<defs>"+Object.entries(REL).map(([k,c])=>`<marker id="m-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="${c}"/></marker>`).join("")+"</defs>";
 let e="",n="";d.edges.forEach((ed,i)=>{const a=al[i],c=REL[ed.relation]||"#888",[x0,y0]=NPOS[ed.source],[x1,y1]=NPOS[ed.target],w=(1+11*a).toFixed(1),op=(0.25+0.75*a).toFixed(2);
  if(ed.source===ed.target){e+=`<circle cx="${x0}" cy="${y0}" r="${(33+5.5*a).toFixed(1)}" fill="none" stroke="${c}" stroke-width="${w}" opacity="${op}"><title>${ed.source} self α=${a.toFixed(3)}</title></circle>`;return}
  const dx=x1-x0,dy=y1-y0,L=Math.hypot(dx,dy),ux=dx/L,uy=dy/L,sx=x0+ux*30,sy=y0+uy*30,tx=x1-ux*33,ty=y1-uy*33,mx=(sx+tx)/2-uy*14,my=(sy+ty)/2+ux*14;
  e+=`<path d="M${sx},${sy} Q${mx},${my} ${tx},${ty}" fill="none" stroke="${c}" stroke-width="${w}" opacity="${op}" marker-end="url(#m-${ed.relation})"><title>${ed.source} → ${ed.target} (${ed.relation}) α=${a.toFixed(3)}</title></path>`;
  if(ed.target===d.readout)e+=`<text x="${mx}" y="${my-6}" text-anchor="middle" style="font-size:11px;font-weight:600;fill:${c}">α ${a.toFixed(2)}</text>`});
 const nmax=Math.sqrt(8);d.nodes.forEach((name,i)=>{const[x,y]=NPOS[name],v=Math.min(nm[i]/nmax,1),fill=mix("#f1eef4","#8e3f6c",v);
  n+=`<circle cx="${x}" cy="${y}" r="27" fill="${fill}" stroke="${ink}" stroke-width="1.2"><title>${name} ‖h‖=${nm[i].toFixed(3)}</title></circle>
  <text x="${x}" y="${y+5}" text-anchor="middle" style="font-size:16px;font-weight:700;fill:${v>0.55?"#fff":"#1d232b"}">${SHORT[name]}</text>
  <text x="${x}" y="${y+53}" text-anchor="middle" style="font-size:10px;fill:${muted}">${name}</text>`});
 G.innerHTML=defs+e+n+`<text x="90" y="440" text-anchor="middle" style="font-size:11px">관측 상태</text><text x="340" y="440" text-anchor="middle" style="font-size:11px">착륙 상황</text><text x="570" y="440" text-anchor="middle" style="font-size:11px">결정 맥락</text><text x="570" y="290" text-anchor="middle" style="font-size:10px">자기 루프 = 노드 둘레 고리</text>`;
 document.getElementById("encLegend").innerHTML=[...new Set(d.edges.map(x=>x.relation))].map(r=>`<span><i style="background:${REL[r]||"#888"};height:4px"></i>${r}</span>`).join("");
 const gmax=Math.max(1e-6,...d.steps.flatMap(s=>s.g.map(Math.abs)));
 let b=`<line x1="20" x2="310" y1="75" y2="75" stroke="${line}"/>`;st.g.forEach((v,i)=>{const x=24+i*36,h=Math.abs(v)/gmax*62,y=v>=0?75-h:75;
  b+=`<rect x="${x}" y="${y}" width="28" height="${Math.max(h,0.5)}" rx="3" fill="${v>=0?"#d14a2f":"#2a78d6"}"><title>g${i+1} = ${v.toFixed(4)}</title></rect><text x="${x+14}" y="148" text-anchor="middle">g${i+1}</text>`});
 B.innerHTML=b+`<text x="4" y="12">±${gmax.toFixed(3)}</text>`;
 const T=d.steps.length,cw=320/T;let hm="";for(let i=0;i<8;i++)d.steps.forEach((s,j)=>{const v=s.g[i]/gmax;
  hm+=`<rect x="${(j*cw).toFixed(2)}" y="${i*15}" width="${(cw+0.3).toFixed(2)}" height="15" fill="${v>=0?mix("#f2f2f2","#d14a2f",v):mix("#f2f2f2","#2a78d6",-v)}"/>`});
 H.innerHTML=hm+`<rect x="${(encIdx*cw).toFixed(2)}" y="0" width="${Math.max(cw,1.5).toFixed(2)}" height="120" fill="none" stroke="${ink}" stroke-width="1.5"/>`;
 const hs=d.steps.map(s=>s.seen.h),hmax=Math.max(0.5,...hs.filter(x=>x!=null));let pth="",pen=false;
 hs.forEach((h,j)=>{if(h==null){pen=false;return}pth+=(pen?"L":"M")+(j*cw).toFixed(1)+","+(80-h/hmax*70).toFixed(1);pen=true});
 let blind="";d.steps.forEach((s,j)=>{if(s.seen.vision_age_s==null||s.seen.vision_age_s>0.5)blind+=`<rect x="${(j*cw).toFixed(2)}" y="0" width="${(cw+0.3).toFixed(2)}" height="84" fill="#eda100" opacity="0.18"/>`});
 C.innerHTML=blind+`<path d="${pth}" fill="none" stroke="${css('--a')}" stroke-width="1.6"/><line x1="${encIdx*cw}" x2="${encIdx*cw}" y1="0" y2="86" stroke="${ink}"/><text x="2" y="10">h ≤ ${hmax.toFixed(1)} m · 주황 = 시야 0.5 s 이상 끊김</text>`}
function encStep(){if(!ENC)return;encIdx=(encIdx+1)%ENC.steps.length;document.getElementById("encStep").value=encIdx;drawEnc()}
document.getElementById("encRun").onchange=()=>{encKey="";loadEnc(true)};
document.getElementById("encProbe").onchange=()=>{encKey="";loadEnc(true)};
document.getElementById("encHop").onchange=drawEnc;
document.getElementById("encStep").oninput=e=>{encIdx=+e.target.value;drawEnc()};
document.getElementById("encPlay").onclick=e=>{if(encTimer){clearInterval(encTimer);encTimer=null;e.target.textContent="▶ 재생"}else{encTimer=setInterval(encStep,100);e.target.textContent="❚❚ 정지"}};
setInterval(()=>{if(encKey)loadEnc()},15000);
async function poll(){try{const r=await fetch("api/state",{cache:"no-store"});if(r.ok)render(await r.json())}
 catch(e){document.getElementById("meta").textContent="서버에 연결할 수 없습니다 (모니터가 종료됐을 수 있습니다)"}}
poll();setInterval(poll,5000);
</script></body></html>
"""
