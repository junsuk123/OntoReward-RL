"""Live dashboard of the four-arm development run (guide 11), read-only.

``snapshot(root)`` reads what the development stage writes under the run root --
``development_registration.json``, ``development/seed*/<arm>/{status.json,
history.jsonl, validation.jsonl, summary.json}`` and ``development.json`` -- and
never writes anything.  ``serve`` publishes it on a local HTTP server
(``/`` page, ``/api/state`` JSON); ``--stage develop`` starts it in a daemon
thread when training begins and ``--stage dashboard`` serves it on its own.
``export`` writes a self-contained HTML copy with the state embedded.  The gate
shown before every seed is complete is a provisional preview computed from the
complete seeds only; the registered verdict is ``development.json``.
"""
from __future__ import annotations

from dataclasses import replace
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

MAX_POINTS = 300
UNSAFE = {"UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT", "SAFETY_ENVELOPE_VIOLATION",
          "MISSED_PAD_CONTACT"}
_CACHE: dict[str, tuple[tuple[float, int], Any]] = {}
_LOCK = threading.Lock()


def _cached(path: Path, parse):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    key = (stat.st_mtime, stat.st_size)
    with _LOCK:
        hit = _CACHE.get(str(path))
        if hit and hit[0] == key:
            return hit[1]
    try:
        value = parse(path)
    except (OSError, ValueError):
        return hit[1] if hit else None  # a file mid-write: keep the last good read
    with _LOCK:
        _CACHE[str(path)] = (key, value)
    return value


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:  # a line being appended
                break
    return rows


def _thin(values: list, limit: int = MAX_POINTS) -> list:
    if len(values) <= limit:
        return values
    step = len(values) / limit
    picked = [values[int(i * step)] for i in range(limit - 1)]
    return picked + [values[-1]]


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def _history(rows: list[dict]) -> dict[str, list]:
    out = {"update": [], "difficulty": [], "success": [], "mean_return": [], "log_std": [],
           "mean_decisions": [], "unsafe": [], "wall_s": []}
    for r in rows:
        episodes = max(int(r.get("episodes") or 0), 1)
        statuses = r.get("statuses") or {}
        log_std = r.get("log_std") or []
        out["update"].append(r.get("update"))
        out["difficulty"].append(_finite(r.get("difficulty")))
        out["success"].append(_finite(r.get("success")))
        out["mean_return"].append(_finite(r.get("mean_return")))
        out["log_std"].append(_finite(sum(log_std) / len(log_std)) if log_std else None)
        out["mean_decisions"].append(_finite(r.get("mean_decisions")))
        out["unsafe"].append(sum(v for k, v in statuses.items() if k in UNSAFE) / episodes)
        out["wall_s"].append(_finite(r.get("wall_s")))
    return out


def _plan(root: Path, plan_path: Path | None):
    """(plan dict, plan object or None).  The object is used for the gate preview."""
    registration = _cached(root / "development_registration.json", _json)
    plan_dict = (registration or {}).get("plan")
    plan_obj = None
    try:
        from .contracts import load_config
        from .development import DEFAULT_DEVELOPMENT_PATH, load_development
        path = Path((registration or {}).get("plan_file") or plan_path or DEFAULT_DEVELOPMENT_PATH)
        plan_obj = load_development(path, load_config())
        if plan_dict is None:
            plan_dict = plan_obj.to_dict()
    except Exception:  # noqa: BLE001 - the dashboard must render whatever exists
        pass
    return registration, plan_dict, plan_obj


def snapshot(root: str | Path, *, plan_path: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    registration, plan, plan_obj = _plan(root, plan_path)
    plan = plan or {}
    arms = list(plan.get("arms") or [])
    seeds = list(plan.get("training_seeds") or [])
    cells = [c["name"] for c in plan.get("cells") or []]
    updates = int((registration or {}).get("updates") or 600)
    dev = root / "development"
    models, done_models, now = {}, {}, time.time()
    for s in seeds:
        for arm in arms:
            key = f"{s}/{arm}"
            mroot = dev / f"seed{s}" / arm
            status = _cached(mroot / "status.json", _json) or {}
            rows = _cached(mroot / "history.jsonl", _jsonl) or []
            vrows = _cached(mroot / "validation.jsonl", _jsonl) or []
            summary = _cached(mroot / "summary.json", _json)
            history = _history(rows)
            last = rows[-1] if rows else {}
            if summary is not None:
                state = "done"
            elif status.get("phase") == "evaluating":
                state = "evaluating"
            elif rows or status.get("phase") == "training":
                state = "training"
            else:
                state = "pending"
            try:
                age = now - (mroot / "history.jsonl").stat().st_mtime if rows else None
            except FileNotFoundError:
                age = None
            model = {
                "state": state, "phase": status,
                "update": last.get("update"), "difficulty": last.get("difficulty"),
                "seconds_since_update": age,
                "wall_s": last.get("wall_s"),
                "history": {k: _thin(v) for k, v in history.items()},
                "validation": {
                    "update": [r.get("update") for r in vrows],
                    "deterministic": [(r.get("deterministic") or {}).get("success") for r in vrows],
                    "sampled": [(r.get("sampled") or {}).get("success") for r in vrows]},
            }
            if summary is not None:
                model["best_update"] = (summary.get("training") or {}).get("best_update")
                model["validation_selected"] = summary.get("validation")
                model["cells"] = {name: {k: c.get(k) for k in ("success", "unsafe", "timeout",
                                                               "episodes")}
                                  for name, c in (summary.get("cells") or {}).items()}
                done_models[key] = summary
            models[key] = model
    total = len(seeds) * len(arms)
    done = sum(m["state"] == "done" for m in models.values())
    updates_done = sum(min((m["update"] + 1) if m["update"] is not None else 0, updates)
                       for m in models.values())
    current = next((k for k, m in models.items() if m["state"] in ("training", "evaluating")),
                   None)
    # ETA from the finished models' wall time (training only; evaluation adds ~1-2 min each).
    finished_walls = [m["wall_s"] for m in models.values() if m["state"] == "done" and m["wall_s"]]
    eta = None
    if finished_walls:
        per = sum(finished_walls) / len(finished_walls)
        remaining = total - done
        if current and models[current]["state"] == "training" and models[current]["update"]:
            remaining -= models[current]["update"] / updates
        eta = max(remaining, 0.0) * (per + 90.0)
    aggregate = {arm: {} for arm in arms}
    for arm in arms:
        for cell in cells:
            vals = [models[f"{s}/{arm}"]["cells"][cell] for s in seeds
                    if models[f"{s}/{arm}"].get("cells") and cell in models[f"{s}/{arm}"]["cells"]]
            succ = [v["success"] for v in vals if v.get("success") is not None]
            uns = [v["unsafe"] for v in vals if v.get("unsafe") is not None]
            aggregate[arm][cell] = {"success": sum(succ) / len(succ) if succ else None,
                                    "unsafe": sum(uns) / len(uns) if uns else None,
                                    "seeds": len(succ)}
    complete = [s for s in seeds if all(f"{s}/{a}" in done_models for a in arms)]
    preview = None
    if plan_obj is not None and complete:
        try:
            from .development import judge
            subset = replace(plan_obj, training_seeds=tuple(complete))
            preview = judge(subset, {k: v for k, v in done_models.items()
                                     if int(k.split("/")[0]) in complete})
            preview = {k: preview[k] for k in ("learnability_passed", "composition_margins",
                                                "guards", "development_gate_passed")}
        except Exception as exc:  # noqa: BLE001
            preview = {"error": str(exc)}
    final = _cached(root / "development.json", _json)
    if registration is None:
        overall = "not_registered"
    elif final is not None:
        overall = "complete" if final.get("procedure_complete") else "finished_incomplete"
    elif current is not None:
        overall = "running"
    elif done == total and total:
        overall = "judging"
    else:
        overall = "registered_waiting"
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_root": str(root), "overall": overall,
        "registration": None if registration is None else {
            k: registration.get(k) for k in ("protocol", "plan_hash", "registered_utc", "models",
                                             "updates", "episodes_per_update",
                                             "training_episodes", "evaluation_episodes")},
        "arms": arms, "proposed_arm": plan.get("proposed_arm"), "training_seeds": seeds,
        "cells": cells, "gates": plan.get("gates"),
        "progress": {"models_done": done, "models_total": total, "updates_done": updates_done,
                     "updates_total": total * updates, "current": current, "eta_s": eta,
                     "seeds_complete": complete},
        "models": models, "aggregate": aggregate, "gate_preview": preview,
        "verdict": (final or {}).get("verdict"),
    }


# ------------------------------------------------------------------ server

def _handler(root: Path, plan_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib interface
            if self.path.split("?")[0].rstrip("/") in ("", "/index.html"):
                body = PAGE.replace("__EMBEDDED_STATE__", "null").encode("utf-8")
                kind = "text/html; charset=utf-8"
            elif self.path.startswith("/api/state"):
                body = json.dumps(snapshot(root, plan_path=plan_path), ensure_ascii=False,
                                  allow_nan=False, default=lambda o: None).encode("utf-8")
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

        def log_message(self, *args):  # keep the training log clean
            return
    return Handler


def serve(root: str | Path, *, host: str = "127.0.0.1", port: int = 8773,
          plan_path: Path | None = None, background: bool = True,
          open_browser: bool | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Start the dashboard; tries ``port`` .. ``port + 19``.  ``open_browser`` None opens
    one unless CS3D_DASHBOARD_NO_BROWSER is set."""
    root = Path(root)
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), _handler(root, plan_path))
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
            except Exception:  # noqa: BLE001 - a headless machine just prints the URL
                pass
        threading.Thread(target=_open, daemon=True).start()
    if background:
        threading.Thread(target=server.serve_forever, daemon=True,
                         name="cs3d-dashboard").start()
    return server, url


def export(root: str | Path, out: str | Path | None = None) -> Path:
    """A self-contained HTML copy with the current state embedded (no server needed)."""
    root = Path(root)
    out = Path(out) if out is not None else root / "development" / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = json.dumps(snapshot(root), ensure_ascii=False, allow_nan=False,
                       default=lambda o: None).replace("</", "<\\/")
    out.write_text(PAGE.replace("__EMBEDDED_STATE__", state), encoding="utf-8")
    return out


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Four-Arm Development</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d232b;--muted:#68717d;--line:#e3e6ea;--grid:#eef0f3;
--raw:#7c8592;--mlp:#2f6fdb;--untyped:#d98a0b;--rgat:#0f9d6e;--bad:#c2410c;--good:#0f9d6e;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#191d23;--ink:#e6e9ee;--muted:#9aa3ad;
--line:#2a3038;--grid:#20252c;--raw:#9aa3ad;--mlp:#6b9ef5;--untyped:#f0a73a;--rgat:#34c38f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
header{padding:16px 20px 8px;display:flex;flex-wrap:wrap;gap:12px;align-items:baseline}
h1{font-size:18px;margin:0;font-weight:650}.sub{color:var(--muted);font-size:12px}
.pill{padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;background:var(--grid)}
.pill.running{background:#dbeafe;color:#1e40af}.pill.complete{background:#d1fae5;color:#065f46}
@media (prefers-color-scheme:dark){.pill.running{background:#1e3a5f;color:#bfdbfe}.pill.complete{background:#064e3b;color:#a7f3d0}}
main{padding:0 20px 24px;display:grid;gap:14px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.card h2{font-size:13px;margin:0 0 10px;font-weight:650;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.span12{grid-column:span 12}.span7{grid-column:span 7}.span5{grid-column:span 5}.span6{grid-column:span 6}
@media (max-width:1000px){.span7,.span5,.span6{grid-column:span 12}}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--rgat)}
.kv{display:flex;flex-wrap:wrap;gap:18px;margin-top:8px}.kv b{display:block;font-size:18px}.kv span{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:center;font-size:13px}
th{color:var(--muted);font-weight:600}td.l,th.l{text-align:left}
.chip{display:inline-block;min-width:96px;padding:5px 6px;border-radius:6px;font-size:12px;background:var(--grid)}
.chip.training{outline:2px solid var(--mlp)}.chip.evaluating{outline:2px solid var(--untyped)}
.chip.done{background:transparent;border:1px solid var(--line)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:4px 0 6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
svg text{fill:var(--muted);font-size:11px}.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.prop{background:color-mix(in srgb,var(--rgat) 10%,transparent)}
.note{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body>
<header><h1>네 비교군 development — 실시간</h1><span id="overall" class="pill">…</span>
<span class="sub" id="meta"></span></header>
<main>
<section class="card span12"><h2>진행</h2><div class="bar"><i id="pbar" style="width:0"></i></div>
<div class="kv" id="kv"></div></section>
<section class="card span5"><h2>모델 20개 (학습 seed × 비교군)</h2><div id="grid"></div></section>
<section class="card span7"><h2>학습 곡선</h2>
<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
<select id="metric">
<option value="success">학습 표본 성공률 (10-update 이동평균)</option>
<option value="val_det">검증 성공률 deterministic</option>
<option value="val_smp">검증 성공률 sampled</option>
<option value="mean_return">학습 return (이동평균)</option>
<option value="difficulty">curriculum 난이도</option>
<option value="unsafe">학습 unsafe 비율 (이동평균)</option>
<option value="log_std">log σ</option></select>
<span class="sub">가는 선: 학습 seed별, 굵은 선: 비교군 평균</span></div>
<div class="legend" id="legend"></div><div id="chart"></div></section>
<section class="card span7"><h2>평가 10셀 — 성공률 (완료된 seed 평균)</h2><div id="cells"></div>
<div class="note">셀당 100 episode, deterministic. 괄호는 완료된 학습 seed 수와 unsafe 비율.</div></section>
<section class="card span5"><h2>Development gate (가이드 11.5)</h2><div id="gate"></div></section>
</main>
<script>
const EMBEDDED = __EMBEDDED_STATE__;
const ARM = {raw_ppo:["Raw","--raw"],semantic_mlp_ppo:["Semantic MLP","--mlp"],
  untyped_gnn_ppo:["Untyped GNN","--untyped"],ontology_rgat_ppo:["Ontology R-GAT","--rgat"]};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = x => x==null ? "–" : (100*x).toFixed(1)+"%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let S = null;
function rolling(v, w){const out=[];let sum=0,n=0,q=[];for(const x of v){q.push(x);if(x!=null){sum+=x;n++}
 if(q.length>w){const y=q.shift();if(y!=null){sum-=y;n--}}out.push(n?sum/n:null)}return out}
function fmtT(s){if(s==null)return "–";s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
 return h?`${h}시간 ${m}분`:`${m}분`}
function series(model, metric){
  if(metric==="val_det"||metric==="val_smp"){const v=model.validation;
    const y=metric==="val_det"?v.deterministic:v.sampled;return v.update.map((u,i)=>[u,y[i]])}
  const h=model.history, raw=h[metric]||[];
  const y=["success","mean_return","unsafe"].includes(metric)?rolling(raw,10):raw;
  return h.update.map((u,i)=>[u,y[i]])}
function meanSeries(list){const m=new Map();for(const s of list)for(const [x,y] of s){if(y==null)continue;
  const e=m.get(x)||[0,0];e[0]+=y;e[1]++;m.set(x,e)}
  return [...m.entries()].sort((a,b)=>a[0]-b[0]).map(([x,e])=>[x,e[0]/e[1]])}
function chart(el, lines, opt){
  const W=el.clientWidth||640,H=300,L=46,R=12,T=10,B=30;
  const pts=lines.flatMap(l=>l.points.filter(p=>p[1]!=null));
  if(!pts.length){el.innerHTML='<div class="note">아직 자료가 없습니다.</div>';return}
  let x0=0,x1=Math.max(10,...pts.map(p=>p[0]));let y0=opt.yMin??Math.min(...pts.map(p=>p[1])),
      y1=opt.yMax??Math.max(...pts.map(p=>p[1]));if(y1-y0<1e-9){y1+=1;y0-=1}
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  let g="";for(let i=0;i<=4;i++){const yy=y0+(y1-y0)*i/4;g+=`<line x1="${L}" x2="${W-R}" y1="${Y(yy)}" y2="${Y(yy)}" stroke="${css('--grid')}"/>`+
   `<text x="${L-6}" y="${Y(yy)+4}" text-anchor="end">${opt.pct?(100*yy).toFixed(0)+"%":yy.toFixed(Math.abs(y1-y0)<5?2:0)}</text>`}
  for(let i=0;i<=5;i++){const xx=x0+(x1-x0)*i/5;g+=`<text x="${X(xx)}" y="${H-10}" text-anchor="middle">${Math.round(xx)}</text>`}
  let p="";for(const l of lines){let d="",pen=false;for(const [x,y] of l.points){if(y==null){pen=false;continue}
    d+=(pen?"L":"M")+X(x).toFixed(1)+","+Y(y).toFixed(1);pen=true}
    if(d)p+=`<path d="${d}" fill="none" stroke="${l.color}" stroke-width="${l.width}" stroke-opacity="${l.opacity}" stroke-linejoin="round"/>`}
  el.innerHTML=`<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}">${g}${p}<text x="${(W+L)/2}" y="${H}" text-anchor="middle">update</text></svg>`}
function render(){
  if(!S)return;const p=S.progress,reg=S.registration||{};
  const o=document.getElementById("overall");const label={not_registered:"등록 전",registered_waiting:"등록됨 · 대기",
   running:"학습 중",judging:"판정 중",complete:"완료",finished_incomplete:"종료(불완전)"}[S.overall]||S.overall;
  o.textContent=label;o.className="pill "+(S.overall==="running"?"running":S.overall==="complete"?"complete":"");
  document.getElementById("meta").textContent=`${reg.protocol||""} · plan ${(reg.plan_hash||"").slice(0,12)} · 갱신 ${new Date(S.generated_utc).toLocaleTimeString()}`;
  document.getElementById("pbar").style.width=(p.updates_total?100*p.updates_done/p.updates_total:0)+"%";
  const cur=p.current?S.models[p.current]:null;
  const curTxt=cur?`${esc(p.current.split("/")[0])} · ${ARM[p.current.split("/")[1]]?.[0]||""} — ${cur.state==="evaluating"?
     `평가 ${cur.phase.cell_index+1}/${cur.phase.cells} (${esc(cur.phase.cell)})`:`update ${cur.update}/${reg.updates||600}, d=${(cur.difficulty??0).toFixed(1)}`}`:"–";
  document.getElementById("kv").innerHTML=[["완료 모델",`${p.models_done} / ${p.models_total}`],
   ["학습 update",`${p.updates_done.toLocaleString()} / ${p.updates_total.toLocaleString()}`],
   ["현재",curTxt],["완료된 학습 seed",`${p.seeds_complete.length} / ${S.training_seeds.length}`],
   ["남은 시간(추정)",fmtT(p.eta_s)]].map(([k,v])=>`<div><span>${k}</span><b>${v}</b></div>`).join("");
  // grid
  let t=`<table><tr><th class="l">seed</th>${S.arms.map(a=>`<th style="color:${css(ARM[a][1])}">${ARM[a][0]}</th>`).join("")}</tr>`;
  for(const s of S.training_seeds){t+=`<tr><td class="l">${s}</td>`;for(const a of S.arms){const m=S.models[`${s}/${a}`];
    let txt="대기";if(m.state==="training")txt=`학습 ${m.update}/${reg.updates||600}<br>d=${(m.difficulty??0).toFixed(1)}`;
    if(m.state==="evaluating")txt=`평가 ${m.phase.cell_index+1}/${m.phase.cells}`;
    if(m.state==="done"){const c=m.cells||{},v=m.validation_selected||{};txt=`comp-OOD ${pct(c.composition_ood?.success)}<br>val ${pct(v.deterministic_success)}`}
    t+=`<td><span class="chip ${m.state}">${txt}</span></td>`}t+="</tr>"}
  document.getElementById("grid").innerHTML=t+"</table>";
  // chart
  const metric=document.getElementById("metric").value;const lines=[];
  for(const a of S.arms){const col=css(ARM[a][1]);const per=S.training_seeds.map(s=>series(S.models[`${s}/${a}`],metric)).filter(x=>x.length);
    for(const pts of per)lines.push({points:pts,color:col,width:1,opacity:.35});
    if(per.length)lines.push({points:meanSeries(per),color:col,width:2.6,opacity:1})}
  document.getElementById("legend").innerHTML=S.arms.map(a=>`<span><i style="background:${css(ARM[a][1])}"></i>${ARM[a][0]}${a===S.proposed_arm?" (제안)":""}</span>`).join("");
  const isPct=["success","val_det","val_smp","unsafe","difficulty"].includes(metric);
  chart(document.getElementById("chart"),lines,{pct:isPct&&metric!=="difficulty",yMin:isPct?0:undefined,yMax:isPct?1:undefined});
  // cells
  let c=`<table><tr><th class="l">셀</th>${S.arms.map(a=>`<th class="${a===S.proposed_arm?"prop":""}" style="color:${css(ARM[a][1])}">${ARM[a][0]}</th>`).join("")}</tr>`;
  for(const cell of S.cells){c+=`<tr><td class="l">${cell}</td>`;for(const a of S.arms){const v=S.aggregate[a]?.[cell]||{};
    const x=v.success,bg=x==null?"transparent":`color-mix(in srgb, ${css('--rgat')} ${Math.round(70*x)}%, transparent)`;
    c+=`<td class="${a===S.proposed_arm?"prop":""}" style="background:${bg}">${pct(x)}<br><span class="sub">(${v.seeds||0}, unsafe ${pct(v.unsafe)})</span></td>`}c+="</tr>"}
  document.getElementById("cells").innerHTML=c+"</table>";
  // gate
  const gv=S.verdict||S.gate_preview;const gates=S.gates||{};let gh="";
  if(!gv){gh=`<div class="note">모든 비교군이 끝난 학습 seed가 생기면 미리보기를 계산합니다.</div>`}
  else if(gv.error){gh=`<div class="no">${esc(gv.error)}</div>`}
  else{gh+=`<div class="note" style="margin-top:0">${S.verdict?"등록된 판정 (development.json)":`미리보기 — 완료된 학습 seed ${p.seeds_complete.length}개만으로 계산`}</div>`;
    const m=gv.composition_margins||{},gd=gv.guards||{};
    gh+=`<table><tr><th class="l">항목</th><th>평균</th><th>기준</th><th></th></tr>`;
    const row=(k,v,crit)=>`<tr><td class="l">${k}</td><td>${v==null?"–":(100*v.mean).toFixed(1)+"%p"}</td><td>${crit}</td><td class="${v?.passed?"ok":"no"}">${v?(v.passed?"통과":"미달"):"–"}</td></tr>`;
    gh+=row("Composition: R-GAT − MLP",m.mlp,"> "+(100*(gates.composition_margin_min??0)).toFixed(0)+"%p");
    gh+=row("Composition: R-GAT − Untyped",m.untyped,"> "+(100*(gates.composition_margin_min??0)).toFixed(0)+"%p");
    gh+=row("Composition: R-GAT − Raw",m.raw,"> "+(100*(gates.composition_margin_min??0)).toFixed(0)+"%p");
    gh+=row("ID 성공률 vs Raw",gd.id_success_vs_raw,"≥ −"+(100*(gates.id_drop_max??0)).toFixed(0)+"%p");
    gh+=row("기존 4 OOD 평균 vs Raw",gd.four_ood_success_vs_raw,"≥ −"+(100*(gates.ood_mean_drop_max??0)).toFixed(0)+"%p");
    gh+=row("OOD unsafe vs Raw",gd.ood_unsafe_vs_raw,"≤ +"+(100*(gates.ood_unsafe_increase_max??0)).toFixed(0)+"%p");
    gh+=`<tr><td class="l">모든 모델 검증 기준</td><td></td><td>det ≥ ${pct(gates.validation_deterministic_success_min)}, smp ≥ ${pct(gates.validation_sampled_success_min)}</td><td class="${gv.learnability_passed?"ok":"no"}">${gv.learnability_passed?"통과":"미달"}</td></tr></table>`;
    gh+=`<p style="font-size:15px;margin:12px 0 0">Development gate: <span class="${gv.development_gate_passed?"ok":"no"}">${gv.development_gate_passed?"통과":"미통과"}</span>${S.verdict?"":" (미리보기)"}</p>`}
  document.getElementById("gate").innerHTML=gh;
}
async function poll(){try{const r=await fetch("api/state",{cache:"no-store"});if(r.ok){S=await r.json();render()}}catch(e){}}
document.getElementById("metric").addEventListener("change",render);
window.addEventListener("resize",render);
if(EMBEDDED){S=EMBEDDED;render()}else{poll();setInterval(poll,5000)}
</script></body></html>
"""
