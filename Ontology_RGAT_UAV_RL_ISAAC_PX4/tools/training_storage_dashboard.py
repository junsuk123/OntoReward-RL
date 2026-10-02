#!/usr/bin/env python3
"""What training data this project has on disk, and how fresh it is.

The live dashboard on :8770 shows the flight in progress. This one answers the
other question -- what has been *stored*, and is it still the thing the current
configuration would produce:

  * the detached run, if one is flying
  * PPO checkpoints and how far each arm has trained
  * the per-arm training history, the only record of the learning curve
  * teacher demonstration sets, keyed by the fingerprint that decides reuse
  * the accumulated episode datastore
  * the log tree, which is where the disk goes

Fingerprints matter here and are shown rather than hidden. A demonstration set
or a checkpoint is reused only when its key matches what the configuration now
hashes to, so "4/4 successes" under a stale key is not a warm start -- it is
1.5 hours of flying about to be repeated. The 2026-09-25 envelope change did
exactly that, and nothing on screen said so beforehand.

    python3 tools/training_storage_dashboard.py            # serve on :8771
    python3 tools/training_storage_dashboard.py --port N
    python3 tools/training_storage_dashboard.py --once     # print JSON and exit

Read-only: it opens the datastore with ``mode=ro`` and never writes anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
LOGS = PROJECT / "logs"

# Freshness bands, in seconds. "live" is shorter than the slowest episode so a
# file that stops moving during a run is visible as stale rather than current.
LIVE_S = 15 * 60
RECENT_S = 6 * 60 * 60


def _age_state(seconds: float | None) -> str:
    if seconds is None:
        return "missing"
    if seconds <= LIVE_S:
        return "live"
    if seconds <= RECENT_S:
        return "recent"
    return "cold"


def _stat(path: Path) -> dict | None:
    try:
        info = path.stat()
    except OSError:
        return None
    age = time.time() - info.st_mtime
    return {"path": str(path), "name": path.name, "bytes": info.st_size,
            "modified": info.st_mtime, "age_s": age, "state": _age_state(age)}


def _tree_bytes(root: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                pass
    return total


def _rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _number(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------- the live run
def live_run() -> dict:
    pid_file = LOGS / "run_detached.pid"
    log_pointer = LOGS / "run_detached.log.path"
    state: dict = {"running": False}
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return state
    try:
        os.kill(pid, 0)
    except OSError:
        state["last_pid"] = pid
        return state
    state["running"] = True
    state["pid"] = pid
    try:
        state["uptime_s"] = int(subprocess.check_output(
            ["ps", "-o", "etimes=", "-p", str(pid)]).decode().strip())
    except Exception:
        state["uptime_s"] = None
    try:
        log = Path(log_pointer.read_text().strip())
        state["log"] = str(log)
        tail = log.read_text(errors="replace").splitlines()[-4000:]
        state["stage"] = _stage_from(tail)
        state["progress"] = _progress_from(tail)
    except OSError:
        pass
    return state


def _stage_from(lines: list[str]) -> str:
    for line in reversed(lines):
        if "episode" in line and "/1008" in line:
            return "PPO training"
        if "training teacher flight" in line:
            return "teacher demonstrations"
        if "keypoint" in line.lower() and "validation" in line.lower():
            return "keypoint validation"
        if "External stack ready" in line:
            return "stack up"
    return "starting"


def _progress_from(lines: list[str]) -> dict:
    seen: dict[str, int] = {}
    total = 0
    for line in lines:
        if " episode " not in line or "/" not in line:
            continue
        parts = line.split()
        for index, token in enumerate(parts):
            if token == "episode" and index and "/" in parts[index + 1]:
                arm = parts[index - 1]
                try:
                    done, total_s = parts[index + 1].split("/")
                    seen[arm] = max(seen.get(arm, 0), int(done))
                    total = max(total, int(total_s))
                except ValueError:
                    pass
    return {"arms": seen, "per_arm_total": total}


# ---------------------------------------------------------------- the data
def experiments() -> list[dict]:
    """Every experiment output tree, newest first."""
    found = []
    for candidate in sorted(RESULTS.glob("*/*")):
        if not (candidate / "models").is_dir():
            continue
        info = _stat(candidate / "models") or {}
        found.append({
            "name": f"{candidate.parent.name}/{candidate.name}",
            "root": str(candidate),
            "bytes": _tree_bytes(candidate),
            "age_s": info.get("age_s"),
            "state": _age_state(info.get("age_s")),
        })
    return sorted(found, key=lambda item: item.get("age_s") or 1e18)


def arms(root: Path) -> list[dict]:
    models = root / "models"
    out = []
    for arm_dir in sorted(p for p in models.glob("*") if p.is_dir()):
        if arm_dir.name == "shared":
            continue
        history = arm_dir / f"{arm_dir.name}_training.csv"
        rows = _rows(history)
        live = _stat(arm_dir / f"{arm_dir.name}.pt")
        best = _stat(arm_dir / f"{arm_dir.name}.best.pt")
        archived = [p for p in arm_dir.glob("*.incompatible-*.pt")
                    if "_training" not in p.name]
        entry = {
            "arm": arm_dir.name,
            "checkpoint": live,
            "best": best,
            "archived": len(archived),
            "history": _stat(history),
            "episodes": len(rows),
        }
        if rows:
            successes = sum(_number(r, "paper_success") for r in rows)
            tail = rows[-80:]
            entry.update({
                "last_episode": rows[-1].get("episode"),
                "successes": int(successes),
                "success_rate": successes / len(rows),
                "recent_success_rate": (
                    sum(_number(r, "paper_success") for r in tail) / len(tail)),
                "curriculum": _number(rows[-1], "curriculum"),
                "series": _series(rows),
            })
        out.append(entry)
    return out


def _series(rows: list[dict], points: int = 90) -> list[dict]:
    """Downsampled learning curve: one point per window, medians and rates."""
    if not rows:
        return []
    width = max(1, len(rows) // points)
    series = []
    for start in range(0, len(rows), width):
        window = rows[start:start + width]
        if len(window) < max(2, width // 2):
            continue
        lateral = sorted(_number(r, "touchdown_lateral_error") for r in window)
        visible = sorted(_number(r, "visible_keypoint_fraction_mean")
                         for r in window)
        series.append({
            "episode": int(_number(window[-1], "episode")),
            "success_rate": sum(_number(r, "paper_success")
                                for r in window) / len(window),
            "lateral_m": lateral[len(lateral) // 2],
            "visible": visible[len(visible) // 2],
        })
    return series


def demonstrations(root: Path) -> list[dict]:
    shared = root / "models/shared"
    training = root / "training"
    sets = {}
    for artifact in shared.glob("teacher_demonstrations_*.pt"):
        key = artifact.stem.replace("teacher_demonstrations_", "")
        sets.setdefault(key, {})["artifact"] = _stat(artifact)
    for attempts in training.glob("teacher_attempts_*.csv"):
        key = attempts.stem.replace("teacher_attempts_", "")
        rows = _rows(attempts)
        flown = [r for r in rows if r.get("status") != "infrastructure_failure"]
        sets.setdefault(key, {})["attempts"] = {
            "file": _stat(attempts),
            "rows": len(rows),
            "flights": len(flown),
            "accepted": int(sum(_number(r, "paper_success") for r in rows)),
            "teacher": (rows[-1].get("teacher") if rows else None),
        }
    out = []
    for key, value in sets.items():
        stamps = [part.get("age_s") for part in
                  (value.get("artifact"), (value.get("attempts") or {}).get("file"))
                  if part]
        out.append({"fingerprint": key, "age_s": min(stamps) if stamps else None,
                    "state": _age_state(min(stamps) if stamps else None), **value})
    return sorted(out, key=lambda item: item.get("age_s") or 1e18)


def datastore() -> dict:
    path = RESULTS / "datastore/collected.sqlite3"
    info = _stat(path)
    if info is None:
        return {"present": False}
    tables = []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        try:
            names = [row[0] for row in connection.execute(
                "select name from sqlite_master where type='table' order by name")]
            for name in names:
                try:
                    count = connection.execute(
                        f'select count(*) from "{name}"').fetchone()[0]
                except sqlite3.Error:
                    count = None
                tables.append({"table": name, "rows": count})
        finally:
            connection.close()
    except sqlite3.Error as error:
        return {"present": True, "file": info, "error": str(error)}
    sidecars = [_stat(path.with_name(path.name + suffix))
                for suffix in ("-wal", "-shm")]
    return {"present": True, "file": info, "tables": tables,
            "sidecars": [s for s in sidecars if s]}


def log_storage() -> dict:
    runs = []
    if (LOGS / "runs").is_dir():
        for run in sorted((LOGS / "runs").glob("run-*")):
            stack = run / "stack"
            ulogs = list(run.glob("px4/**/*.ulg")) + list(run.glob("px4/**/*.ulg.gz"))
            generations = len(list(stack.glob("*.[0-9][0-9][0-9].log"))) if stack.is_dir() else 0
            info = _stat(stack if stack.is_dir() else run)
            runs.append({
                "run": run.name, "bytes": _tree_bytes(run),
                "ulogs": len(ulogs), "generations": generations,
                "age_s": info.get("age_s") if info else None,
                "state": _age_state(info.get("age_s") if info else None),
            })
    usage = os.statvfs(PROJECT)
    return {
        "runs": sorted(runs, key=lambda item: item["run"], reverse=True),
        "total_bytes": _tree_bytes(LOGS) if LOGS.is_dir() else 0,
        "disk_free_bytes": usage.f_bavail * usage.f_frsize,
        "disk_total_bytes": usage.f_blocks * usage.f_frsize,
    }


def snapshot() -> dict:
    trees = experiments()
    primary = Path(trees[0]["root"]) if trees else None
    return {
        "generated": time.time(),
        "project": str(PROJECT),
        "run": live_run(),
        "experiments": trees,
        "primary": trees[0]["name"] if trees else None,
        "arms": arms(primary) if primary else [],
        "demonstrations": demonstrations(primary) if primary else [],
        "datastore": datastore(),
        "logs": log_storage(),
    }


PAGE = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training storage</title>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f4f3f0; --line:#e3e2dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#78776f;
  --series-1:#2a78d6; --series-2:#eb6834;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --line:#383835;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --series-1:#3987e5; --series-2:#d95926;
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --line:#383835;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --series-1:#3987e5; --series-2:#d95926;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--text-secondary);margin:34px 0 10px;font-weight:600}
.sub{color:var(--text-muted);margin:0 0 4px;font-size:13px}
.tiles{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.tile{background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{font-size:12px;color:var(--text-secondary)}
.tile .v{font-size:23px;font-weight:600;letter-spacing:-.02em;margin-top:2px;
  font-variant-numeric:tabular-nums}
.tile .n{font-size:12px;color:var(--text-muted);margin-top:1px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;font-size:12px;color:var(--text-secondary);
  padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td.wrap{white-space:normal}
tr:last-child td{border-bottom:none}
.num{text-align:right}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.scroll{overflow-x:auto;background:var(--surface-2);border:1px solid var(--line);border-radius:10px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.s-live .dot{background:var(--good)} .s-live{color:var(--good)}
.s-recent .dot{background:var(--warning)} .s-recent{color:var(--text-secondary)}
.s-cold .dot{background:var(--serious)} .s-cold{color:var(--text-muted)}
.s-missing .dot{background:var(--critical)} .s-missing{color:var(--critical)}
.legend{display:flex;gap:16px;align-items:center;margin:0 0 8px;font-size:12px;
  color:var(--text-secondary)}
.legend i{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:6px}
.charts{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
figure{margin:0;background:var(--surface-2);border:1px solid var(--line);
  border-radius:10px;padding:12px 14px 8px}
figcaption{font-size:12px;color:var(--text-secondary);margin-bottom:6px}
svg{display:block;width:100%;height:150px;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);
  border:1px solid var(--line);border-radius:7px;padding:6px 9px;font-size:12px;
  box-shadow:0 4px 14px rgba(0,0,0,.14);opacity:0;transition:opacity .08s}
footer{margin-top:40px;color:var(--text-muted);font-size:12px}
</style>
<main>
  <h1>Training storage</h1>
  <p class="sub" id="where"></p>
  <div id="app"></div>
  <footer id="foot"></footer>
</main>
<div class="tip" id="tip"></div>
<script>
const SERIES = ["var(--series-1)", "var(--series-2)"];
const fmtB = b => b==null ? "—" :
  b >= 1e9 ? (b/1e9).toFixed(1)+" GB" : b >= 1e6 ? (b/1e6).toFixed(0)+" MB" :
  b >= 1e3 ? (b/1e3).toFixed(0)+" kB" : b+" B";
const fmtAge = s => s==null ? "—" :
  s < 90 ? Math.round(s)+"s ago" : s < 5400 ? Math.round(s/60)+"m ago" :
  s < 172800 ? (s/3600).toFixed(1)+"h ago" : (s/86400).toFixed(1)+"d ago";
const esc = s => String(s==null?"":s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
// Status is never colour alone: every badge carries its word.
const badge = st => `<span class="badge s-${st}"><span class="dot"></span>${st}</span>`;

function tiles(items){
  return `<div class="tiles">${items.map(t => `<div class="tile">
    <div class="k">${esc(t.k)}</div><div class="v">${esc(t.v)}</div>
    <div class="n">${t.n||""}</div></div>`).join("")}</div>`;
}
function table(head, rows){
  if(!rows.length) return `<p class="sub">nothing stored yet</p>`;
  return `<div class="scroll"><table><thead><tr>${
    head.map(h => `<th class="${h.num?"num":""}">${esc(h.t)}</th>`).join("")
  }</tr></thead><tbody>${rows.map(r => `<tr>${
    r.map((c,i) => `<td class="${head[i].num?"num ":""}${head[i].mono?"mono ":""}${head[i].wrap?"wrap":""}">${c}</td>`).join("")
  }</tr>`).join("")}</tbody></table></div>`;
}

// One measure per chart, one y-axis, always. Two arms = two series, legend
// present and both direct-labelled at their last point.
function smoothed(series, key, span){
  if(!span || span < 2) return series;
  const half = Math.floor(span/2);
  return series.map((p,i) => {
    const lo = Math.max(0, i-half), hi = Math.min(series.length, i+half+1);
    let sum = 0;
    for(let j=lo;j<hi;j++) sum += series[j][key];
    return {...p, [key]: sum/(hi-lo)};
  });
}
function chart(id, caption, armSeries, key, format, domainMax, span){
  const W=560, H=150, P={t:12,r:64,b:22,l:40};
  armSeries = armSeries.map(a => ({...a, series: smoothed(a.series, key, span)}));
  const all = armSeries.flatMap(a => a.series);
  if(!all.length) return "";
  const xs = all.map(p => p.episode), ys = all.map(p => p[key]);
  const x0=Math.min(...xs), x1=Math.max(...xs, x0+1);
  const y1 = domainMax!=null ? Math.max(domainMax, ...ys) : Math.max(...ys, 1e-9);
  const X = e => P.l + (e-x0)/(x1-x0) * (W-P.l-P.r);
  const Y = v => H-P.b - (v/y1) * (H-P.t-P.b);
  const ticks = [0, y1/2, y1];
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(caption)}">`;
  ticks.forEach(t => { svg += `<line x1="${P.l}" x2="${W-P.r}" y1="${Y(t)}" y2="${Y(t)}"
      stroke="var(--line)" stroke-width="1"/>
    <text x="${P.l-7}" y="${Y(t)+4}" text-anchor="end" font-size="10.5"
      fill="var(--text-muted)">${format(t)}</text>`; });
  svg += `<text x="${P.l}" y="${H-5}" font-size="10.5" fill="var(--text-muted)">ep ${x0}</text>
    <text x="${W-P.r}" y="${H-5}" text-anchor="end" font-size="10.5" fill="var(--text-muted)">ep ${x1}</text>`;
  armSeries.forEach((a,i) => {
    const d = a.series.map((p,j) => `${j?"L":"M"}${X(p.episode).toFixed(1)},${Y(p[key]).toFixed(1)}`).join("");
    svg += `<path d="${d}" fill="none" stroke="${SERIES[i]}" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>`;
    const last = a.series[a.series.length-1];
    svg += `<circle cx="${X(last.episode)}" cy="${Y(last[key])}" r="3.5" fill="${SERIES[i]}"
      stroke="var(--surface-2)" stroke-width="2"/>
      <text x="${X(last.episode)+8}" y="${Y(last[key])+4}" font-size="11"
        fill="var(--text-secondary)">${format(last[key])}</text>`;
  });
  svg += `<rect id="${id}-hit" x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}"
     fill="transparent"/><line id="${id}-cross" y1="${P.t}" y2="${H-P.b}" stroke="var(--text-muted)"
     stroke-width="1" opacity="0"/></svg>`;
  const meta = JSON.stringify({x0,x1,P,W,key,arms:armSeries.map(a=>({arm:a.arm,series:a.series}))});
  return `<figure data-chart='${meta.replace(/'/g,"&#39;")}' id="${id}">
    <figcaption>${esc(caption)}</figcaption>${svg}</figure>`;
}

function wireHover(){
  const tip = document.getElementById("tip");
  document.querySelectorAll("figure[data-chart]").forEach(fig => {
    const meta = JSON.parse(fig.dataset.chart);
    const svg = fig.querySelector("svg");
    const cross = fig.querySelector('[id$="-cross"]');
    const hit = fig.querySelector('[id$="-hit"]');
    const move = ev => {
      const box = svg.getBoundingClientRect();
      const vx = (ev.clientX - box.left) / box.width * meta.W;
      const frac = (vx - meta.P.l) / (meta.W - meta.P.l - meta.P.r);
      const ep = Math.round(meta.x0 + frac * (meta.x1 - meta.x0));
      cross.setAttribute("x1", vx); cross.setAttribute("x2", vx);
      cross.setAttribute("opacity", "1");
      const near = a => a.series.reduce((b,p) =>
        Math.abs(p.episode-ep) < Math.abs(b.episode-ep) ? p : b, a.series[0]);
      tip.innerHTML = `<b>episode ${ep}</b>` + meta.arms.map((a,i) => {
        const p = near(a);
        const v = meta.key === "success_rate" ? (100*p[meta.key]).toFixed(1)+"%"
                : p[meta.key].toFixed(2);
        return `<br><span style="color:${SERIES[i]}">●</span> ${esc(a.arm)}: ${v}`;
      }).join("");
      tip.style.opacity = 1;
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - 220) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
    };
    hit.addEventListener("mousemove", move);
    hit.addEventListener("mouseleave", () => {
      tip.style.opacity = 0; cross.setAttribute("opacity", "0"); });
  });
}

function render(s){
  document.getElementById("where").textContent =
    `${s.project} · primary ${s.primary || "—"}`;
  const r = s.run, out = [];

  out.push(`<h2>Run</h2>` + tiles([
    {k:"detached run", v: r.running ? "flying" : "stopped",
     n: r.running ? `pid ${r.pid} · up ${fmtAge(r.uptime_s).replace(" ago","")}`
                  : (r.last_pid ? `last pid ${r.last_pid}` : "")},
    {k:"stage", v: r.stage || "—", n: r.log ? r.log.split("/").pop() : ""},
    ...Object.entries((r.progress||{}).arms||{}).map(([arm,done]) =>
      ({k:arm, v:`${done}`, n:`of ${(r.progress||{}).per_arm_total||"?"} episodes`})),
  ]));

  const withSeries = s.arms.filter(a => (a.series||[]).length);
  if(withSeries.length){
    out.push(`<h2>Learning curve</h2>`);
    out.push(`<div class="legend">` + withSeries.map((a,i) =>
      `<span><i style="background:${SERIES[i]}"></i>${esc(a.arm)}</span>`).join("") +
      `<span style="color:var(--text-muted)">medians over ~${
        Math.max(1, Math.round(withSeries[0].episodes/withSeries[0].series.length))
      }-episode windows</span></div>`);
    out.push(`<div class="charts">` +
      chart("c1","Success rate — rolling", withSeries, "success_rate",
            v => (100*v).toFixed(0)+"%", 0.06, 9) +
      chart("c2","Touchdown lateral error (m) — lower is better", withSeries,
            "lateral_m", v => v.toFixed(1)) +
      chart("c3","Pad kept in view (mean visible keypoint fraction)", withSeries,
            "visible", v => v.toFixed(2), 1) +
      `</div>`);
  }

  out.push(`<h2>Checkpoints and history</h2>` + table(
    [{t:"arm"},{t:"episodes",num:true},{t:"successes",num:true},
     {t:"recent 80",num:true},{t:"curriculum",num:true},{t:"checkpoint"},
     {t:"best"},{t:"archived",num:true},{t:"history written"}],
    s.arms.map(a => [
      esc(a.arm),
      a.last_episode ?? "—",
      a.successes!=null ? `${a.successes} (${(100*a.success_rate).toFixed(1)}%)` : "—",
      a.recent_success_rate!=null ? (100*a.recent_success_rate).toFixed(1)+"%" : "—",
      a.curriculum!=null ? a.curriculum.toFixed(2) : "—",
      a.checkpoint ? `${badge(a.checkpoint.state)} ${fmtB(a.checkpoint.bytes)} · ${fmtAge(a.checkpoint.age_s)}` : badge("missing"),
      a.best ? `${fmtB(a.best.bytes)} · ${fmtAge(a.best.age_s)}` : badge("missing"),
      a.archived || 0,
      a.history ? fmtAge(a.history.age_s) : badge("missing"),
    ])));

  out.push(`<h2>Teacher demonstrations</h2>
    <p class="sub">Reused only while the fingerprint matches what the configuration
    now hashes to. A set under a superseded key is not a warm start.</p>` + table(
    [{t:"fingerprint",mono:true},{t:"teacher"},{t:"flights",num:true},
     {t:"accepted",num:true},{t:"set size",num:true},{t:"state"},{t:"written"}],
    s.demonstrations.map(d => {
      const at = d.attempts || {};
      return [esc(d.fingerprint), esc(at.teacher || "—"),
        at.flights ?? "—", at.accepted ?? "—",
        d.artifact ? fmtB(d.artifact.bytes) : badge("missing"),
        badge(d.state), fmtAge(d.age_s)];
    })));

  const ds = s.datastore;
  out.push(`<h2>Accumulated datastore</h2>` + (ds.present
    ? tiles([{k:"collected.sqlite3", v:fmtB(ds.file.bytes), n:fmtAge(ds.file.age_s)},
             ...(ds.tables||[]).map(t => ({k:t.table, v:(t.rows??"?").toLocaleString(), n:"rows"}))])
    : `<p class="sub">no datastore yet</p>`));

  const lg = s.logs;
  out.push(`<h2>Log storage</h2>
    <p class="sub">One directory per simulator boot, not per run: a run that
    rebuilds the stack 61 times leaves 61 of them.</p>` + tiles([
    {k:"logs tree", v:fmtB(lg.total_bytes), n:`${lg.runs.length} simulator boot(s)`},
    {k:"disk free", v:fmtB(lg.disk_free_bytes),
     n:`of ${fmtB(lg.disk_total_bytes)} · ${(100*(1-lg.disk_free_bytes/lg.disk_total_bytes)).toFixed(0)}% used`},
  ]) + (() => {
    const shown = lg.runs.slice(0, 10);
    const rest = lg.runs.slice(10);
    const restBytes = rest.reduce((a,b) => a + b.bytes, 0);
    return table(
      [{t:"boot"},{t:"size",num:true},{t:"PX4 ulogs",num:true},
       {t:"stack generations",num:true},{t:"state"},{t:"written"}],
      shown.map(x => [esc(x.run), fmtB(x.bytes), x.ulogs, x.generations,
                      badge(x.state), fmtAge(x.age_s)])
      .concat(rest.length ? [[`<span style="color:var(--text-muted)">${rest.length} older boot(s)</span>`,
        fmtB(restBytes), rest.reduce((a,b)=>a+b.ulogs,0), "", "", ""]] : []));
  })());

  document.getElementById("app").innerHTML = out.join("");
  document.getElementById("foot").textContent =
    "read-only · refreshed " + new Date(s.generated*1000).toLocaleTimeString();
  wireHover();
}

async function tick(){
  try{ render(await (await fetch("api/state", {cache:"no-store"})).json()); }
  catch(e){ document.getElementById("foot").textContent = "refresh failed: " + e; }
}
tick(); setInterval(tick, 10000);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):                                # noqa: N802 - stdlib API
        if self.path.rstrip("/") in ("", "/index.html"):
            body, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif self.path.startswith("/api/state"):
            body = json.dumps(snapshot()).encode("utf-8")
            kind = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):                    # noqa: D401 - quiet server
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--once", action="store_true",
                        help="print the snapshot as JSON and exit")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(snapshot(), indent=2))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"training storage dashboard: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
