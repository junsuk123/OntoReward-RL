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
  * logs       -- the tails of the logs written in the last hour;
  * machine    -- load, memory and GPU.

``snapshot()`` only reads files and ``ps``/``nvidia-smi`` output; nothing is written,
locked or started.  ``serve`` publishes ``/`` and ``/api/state`` on 127.0.0.1;
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


def _run_kind(rel: Path) -> str:
    parts = rel.parts
    if parts[0] == "g4_pilot":
        return "G4 pilot"
    if parts[0] == "development":
        return "development"
    if parts[0] == "development_design":
        return "design (등록 아님)"
    if parts[0] == "relation_shuffle":
        return "관계 섞기 대조군"
    if parts[0] in ("orl", "orl2", "orl3"):
        name = "ORL" if parts[0] == "orl" else f"ORL/{parts[0][3:]}"
        return f"{name} " + ("G4 pilot" if parts[1] == "g4_pilot" else parts[1])
    return parts[0]


def _planned_updates(task_root: Path, run_dir: Path) -> int:
    """600 is the registered budget of every G4 pilot, design run and development model."""
    for candidate in (task_root / "development_registration.json",):
        data = _cached(candidate, _json) if candidate.exists() else None
        if data and data.get("updates") and "development" in run_dir.parts:
            return int(data["updates"])
    return 600


def run_state(results: Path, task_root: Path, run_dir: Path, now: float) -> dict[str, Any]:
    rows = _cached(run_dir / "history.jsonl", _jsonl) or []
    vrows = _cached(run_dir / "validation.jsonl", _jsonl) or []
    planned = _planned_updates(task_root, run_dir)
    last = rows[-1] if rows else {}
    try:
        age = now - (run_dir / "history.jsonl").stat().st_mtime
    except FileNotFoundError:
        age = None
    final = (run_dir / "final.pt").exists()
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
                "orl3/g4_pilot/seed*/history.jsonl", "orl3/development/seed*/history.jsonl")
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
        for path in list(root.glob("*.log")) + list(root.glob("development_design/*.log")):
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
    from .experiment import TASK_RUN_ROOTS
    results = RESULTS if results is None else Path(results)
    now = time.time()
    roots = {task: (results / root.name) for task, root in TASK_RUN_ROOTS.items()}
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
    order = {"running": 0, "stalled": 1, "pending": 2, "done": 3}
    runs.sort(key=lambda r: (order.get(r["state"], 9), r["path"]))
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "results": str(results),
        "machine": machine(),
        "processes": processes(),
        "runs": runs,
        "tasks": [_verdicts(results, task, root) for task, root in roots.items()],
        "logs": recent_logs(results, [r for r in roots.values() if r.exists()], now),
    }


# ------------------------------------------------------------------ server

def _handler(results: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib interface
            route = self.path.split("?")[0].rstrip("/")
            if route in ("", "/index.html"):
                body, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
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
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;background:var(--grid)}
.running{background:#dbeafe;color:#1e40af}.done{background:#d1fae5;color:#065f46}
.stalled{background:#fee2e2;color:#991b1b}.pending{background:var(--grid);color:var(--muted)}
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
</style></head><body>
<header><h1>현재 상태 모니터</h1><span class="sub" id="meta">불러오는 중…</span></header>
<main>
<section class="card s12"><h2>기계</h2><div class="kv" id="machine"></div></section>
<section class="card s12"><h2>학습 실행</h2><div class="runs" id="runs"></div></section>
<section class="card s12"><h2>실행 중 프로세스</h2><div class="wrap" id="procs"></div></section>
<section class="card s12"><h2>과제별 판정</h2><div class="wrap" id="tasks"></div></section>
<section class="card s12"><h2>최근 로그 (1시간 안에 쓰인 것)</h2><div id="logs"></div></section>
</main>
<script>
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=x=>x==null?"–":(100*x).toFixed(1)+"%";
function dur(s){if(s==null)return"–";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
 return h?`${h}시간 ${m}분`:m?`${m}분 ${x}초`:`${x}초`}
const STATE={running:"학습 중",done:"완료",stalled:"멈춤?",pending:"대기"};
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
async function poll(){try{const r=await fetch("api/state",{cache:"no-store"});if(r.ok)render(await r.json())}
 catch(e){document.getElementById("meta").textContent="서버에 연결할 수 없습니다 (모니터가 종료됐을 수 있습니다)"}}
poll();setInterval(poll,5000);
</script></body></html>
"""
