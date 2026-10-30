"""Read-only live state and self-contained reports for MATLAB-port runs.

The training history is intentionally the source of truth.  The dashboard does
not infer outcomes from simulator stdout and never opens a checkpoint, so it is
safe to run while Isaac/PX4 owns the policy process.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any


UNSAFE = {"UNSAFE_CONTACT", "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION"}
METHOD_LABELS = {
    "ppo": "PPO",
    "onto_rgat_ppo": "Ontology R-GAT PPO",
    "shuffled_rgat_ppo": "Shuffled R-GAT PPO",
}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dimension(path: Path, manifest: dict[str, Any]) -> int | None:
    for part in reversed(path.parts):
        if part in {"2d", "3d"}:
            return int(part[0])
    value = manifest.get("dimension")
    return int(value) if value in (2, 3) else None


def _process_owns(run_root: Path) -> bool:
    needle = str(run_root.resolve())
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = entry.joinpath("cmdline").read_bytes().decode(
                "utf-8", errors="ignore").replace("\0", " ")
        except OSError:
            continue
        if needle in command and ("run_matlab_port" in command or "matlab-port" in command):
            return True
    return False


def _training_roots(run_root: Path) -> list[Path]:
    roots = {path.parent.parent for path in run_root.rglob("history.json")
             if path.parent.name in METHOD_LABELS}
    return sorted(roots, key=lambda path: str(path))


def _safe_mean(values) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _arm_state(training_root: Path, history_path: Path,
               fallback_updates: int | None,
               fallback_episodes: int | None) -> dict[str, Any]:
    history = _read_json(history_path, [])
    history = history if isinstance(history, list) else []
    manifest = _read_json(training_root / "training_manifest.json", {}) or {}
    method = history_path.parent.name
    planned_updates = manifest.get("updates", fallback_updates)
    episodes_per_update = manifest.get("episodes_per_update", fallback_episodes)
    if episodes_per_update is None and history:
        episodes_per_update = max((len(row.get("episodes", [])) for row in history), default=0)
    target_kl = manifest.get("target_kl", 0.02)
    series = []
    totals: Counter[str] = Counter()
    returns: list[float] = []
    wall_seconds: list[float] = []
    samples = accepted = rejected = 0
    validations = []
    for row in history:
        episodes = row.get("episodes", [])
        counts = Counter(str(episode.get("status", "UNKNOWN")) for episode in episodes)
        episode_returns = [float(episode.get("return", 0.0)) for episode in episodes]
        count = len(episodes)
        totals.update(counts)
        returns.extend(episode_returns)
        wall_seconds.extend(float(episode.get("wall_seconds", 0.0))
                            for episode in episodes
                            if float(episode.get("wall_seconds", 0.0)) > 0.0)
        samples += int(row.get("samples", 0))
        accepted += int(row.get("accepted_steps", 0))
        rejected += int(row.get("rejected_steps", 0))
        point = {
            "update": int(row.get("update", len(series) + 1)),
            "episodes": count,
            "mean_return": _safe_mean(episode_returns),
            "success_rate": counts["SUCCESS"] / count if count else 0.0,
            "timeout_rate": counts["TASK_TIMEOUT"] / count if count else 0.0,
            "unsafe_rate": sum(counts[name] for name in UNSAFE) / count if count else 0.0,
            "post_step_kl": float(row.get("post_step_kl", 0.0)),
            "actor_lr": float(row.get("actor_lr", 0.0)),
            "accepted_steps": int(row.get("accepted_steps", 0)),
            "rejected_steps": int(row.get("rejected_steps", 0)),
        }
        series.append(point)
        validation = row.get("validation")
        if isinstance(validation, dict):
            validations.append({
                "update": point["update"],
                "success_rate": float(validation.get("success_rate", 0.0)),
                "unsafe_rate": float(validation.get("unsafe_rate", 0.0)),
                "mean_return": float(validation.get("mean_return", 0.0)),
            })
    episode_count = sum(totals.values())
    completed_updates = max((point["update"] for point in series), default=0)
    episodes_planned = (int(planned_updates) * int(episodes_per_update)
                        if planned_updates is not None and episodes_per_update else None)
    mean_wall_seconds = _safe_mean(wall_seconds)
    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "dimension": _dimension(training_root, manifest),
        "training_root": str(training_root),
        "updates_completed": completed_updates,
        "planned_updates": planned_updates,
        "episodes_per_update": episodes_per_update,
        "episodes_complete": episode_count,
        "episodes_planned": episodes_planned,
        "progress": (completed_updates / int(planned_updates)
                     if planned_updates else None),
        "outcomes": dict(sorted(totals.items())),
        "success_rate": totals["SUCCESS"] / episode_count if episode_count else 0.0,
        "timeout_rate": totals["TASK_TIMEOUT"] / episode_count if episode_count else 0.0,
        "unsafe_rate": (sum(totals[name] for name in UNSAFE) / episode_count
                        if episode_count else 0.0),
        "mean_return": _safe_mean(returns),
        "mean_episode_wall_seconds": mean_wall_seconds,
        "estimated_training_wall_seconds_remaining": (
            max(0, episodes_planned - episode_count) * mean_wall_seconds
            if episodes_planned is not None and mean_wall_seconds is not None else None),
        "samples": samples,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "target_kl": float(target_kl),
        "max_post_step_kl": max((point["post_step_kl"] for point in series), default=0.0),
        "series": series,
        "validations": validations,
    }


def _evaluations(run_root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(run_root.rglob("evaluate.json")):
        payload = _read_json(path, {})
        if not isinstance(payload, dict) or not isinstance(payload.get("methods"), dict):
            continue
        relative = str(path.parent.relative_to(run_root))
        for method, row in payload["methods"].items():
            if not isinstance(row, dict):
                continue
            result.append({
                "stage": relative,
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "success_rate": row.get("success_rate"),
                "unsafe_rate": row.get("unsafe_rate"),
                "mean_return": row.get("mean_return"),
                "deterministic": row.get("deterministic"),
                "episodes": len(row.get("episodes", [])),
            })
    return result


def _recoveries(training_roots: list[Path]) -> dict[str, Any]:
    rows = []
    for root in training_roots:
        for kind, name in (("reset", "reset_recovery.jsonl"),
                           ("episode", "episode_recovery.jsonl"),
                           ("cleanup", "cleanup_recovery.jsonl")):
            for row in _read_jsonl(root / name):
                if row.get("state") not in {"restarted", "failed"}:
                    continue
                rows.append({"kind": kind, "training_root": str(root), **row})
    return {
        "count": len(rows),
        "reset_count": sum(row["kind"] == "reset" for row in rows),
        "episode_count": sum(row["kind"] == "episode" for row in rows),
        "cleanup_count": sum(row["kind"] == "cleanup" for row in rows),
        "events": rows[-20:],
    }


def _diagnosis(state: str, arms: list[dict[str, Any]], recoveries: dict[str, Any]) -> list[str]:
    notes = []
    if state == "INTERRUPTED":
        notes.append("학습 프로세스가 없고 계획된 update가 남아 있어, 현재 run은 중단 상태입니다.")
    for arm in arms:
        points = arm["series"]
        if not points:
            continue
        if len(points) < 10:
            notes.append(f"{arm['label']}: {len(points)} update만으로는 학습 추세를 확정할 수 없습니다.")
        remaining = arm.get("estimated_training_wall_seconds_remaining")
        if remaining is not None and remaining >= 86400:
            notes.append(
                f"{arm['label']}: 현재 episode wall time 기준, validation을 제외한 "
                f"학습만 약 {remaining/86400:.1f}일 남았습니다.")
        if arm["unsafe_rate"] == 0:
            notes.append(f"{arm['label']}: 저장된 {arm['episodes_complete']} episode에서 unsafe 종료는 0회입니다.")
        if arm["max_post_step_kl"] <= 1.5 * arm["target_kl"]:
            notes.append(f"{arm['label']}: post-step KL은 모두 1.5× target KL 이내입니다.")
        else:
            notes.append(f"{arm['label']}: post-step KL이 1.5× target KL을 넘은 update가 있습니다.")
    if recoveries["count"]:
        discarded = recoveries["reset_count"] + recoveries["episode_count"]
        if discarded:
            notes.append(f"reset/mid-episode recovery {discarded}회는 RL 종료 결과에 포함되지 않습니다.")
        if recoveries["cleanup_count"]:
            notes.append(
                f"post-policy cleanup recovery {recoveries['cleanup_count']}회는 "
                "완료된 RL 결과를 보존했습니다.")
    return notes


def snapshot(run_root: str | Path, *, planned_updates: int | None = None,
             episodes_per_update: int | None = None) -> dict[str, Any]:
    """Return compact chart data from a complete or interrupted run."""
    root = Path(run_root).expanduser().resolve()
    training_roots = _training_roots(root) if root.exists() else []
    arms = []
    for training_root in training_roots:
        for history in sorted(training_root.glob("*/history.json")):
            if history.parent.name in METHOD_LABELS:
                arms.append(_arm_state(training_root, history, planned_updates,
                                       episodes_per_update))
    running = _process_owns(root)
    complete = bool(arms) and all(
        arm["planned_updates"] is not None
        and arm["updates_completed"] >= int(arm["planned_updates"])
        for arm in arms)
    pipeline = _read_json(root / "pipeline_report.json", {}) or {}
    if running:
        state = "RUNNING"
    elif pipeline.get("status") == "PASSED" or complete:
        state = "COMPLETE"
    elif arms:
        state = "INTERRUPTED"
    else:
        state = "WAITING"
    recoveries = _recoveries(training_roots)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "state": state,
        "arms": arms,
        "evaluations": _evaluations(root) if root.exists() else [],
        "recoveries": recoveries,
        "pipeline": {
            "status": pipeline.get("status"),
            "workflow": pipeline.get("workflow"),
            "stages": _read_json(root / "pipeline_progress.json", []),
        },
        "diagnosis": _diagnosis(state, arms, recoveries),
    }


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MATLAB-port training dashboard</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#111827;--muted:#667085;--line:#d7dde5;
--blue:#2878d0;--orange:#e36b32;--green:#179b70;--red:#d74646;--purple:#7b61c9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif}
main{max-width:1500px;margin:auto;padding:24px}.head{display:flex;justify-content:space-between;
gap:16px;align-items:end}h1{font-size:24px;margin:0}h2{font-size:16px;margin:0 0 12px}
.sub{color:var(--muted);font-size:12px}.pill{display:inline-block;padding:5px 10px;border-radius:99px;
font-weight:700;background:#e8edf3}.RUNNING{color:#087655;background:#dff5ec}.COMPLETE{color:#087655;
background:#dff5ec}.INTERRUPTED{color:#a52a2a;background:#fde8e8}.grid{display:grid;
grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}.panel{background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 1px 2px #1018280a}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span12{grid-column:span 12}
.metric{font-size:26px;font-weight:750;margin-top:4px}.bar{height:8px;background:#edf0f3;border-radius:9px;
overflow:hidden;margin-top:10px}.bar i{display:block;height:100%;background:var(--blue)}canvas{width:100%;height:230px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid var(--line);
text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.good{color:var(--green)}.bad{color:var(--red)}ul{margin:0;padding-left:20px}.empty{color:var(--muted);padding:24px 0}
@media(max-width:900px){.span3,.span4,.span6{grid-column:span 12}.head{align-items:start;flex-direction:column}}
</style></head><body><main><div class="head"><div><h1>MATLAB-port 학습 · 결과 분석</h1>
<div class="sub" id="root"></div></div><div><span id="state" class="pill"></span>
<div class="sub" id="updated"></div></div></div><div id="app"></div></main><script>
const EMBEDDED_STATE=__EMBEDDED_STATE__;
const colors=['#2878d0','#e36b32','#179b70','#7b61c9'];
const pct=v=>v==null?'\u2014':(100*v).toFixed(1)+'%'; const num=(v,d=2)=>v==null?'\u2014':Number(v).toFixed(d);
const duration=v=>v==null?'\u2014':v>=86400?(v/86400).toFixed(1)+' d':v>=3600?(v/3600).toFixed(1)+' h':(v/60).toFixed(0)+' min';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function lineChart(canvas, arms, key, opts={}){const c=document.getElementById(canvas);if(!c)return;
 const dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
 const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);const pad={l:48,r:20,t:18,b:32};
 let vals=arms.flatMap(a=>a.series.map(p=>p[key]).filter(v=>v!=null&&isFinite(v)));
 if(!vals.length){x.fillStyle='#667085';x.fillText('\uB370\uC774\uD130 \uC5C6\uC74C',pad.l,pad.t+20);return}
 let xmax=Math.max(1,...arms.flatMap(a=>a.series.map(p=>p.update))), ymin=opts.min??Math.min(...vals),ymax=opts.max??Math.max(...vals);
 if(ymax===ymin){ymax+=1;ymin-=1} const X=v=>pad.l+(w-pad.l-pad.r)*v/xmax;
 const Y=v=>pad.t+(h-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));x.strokeStyle='#d7dde5';x.fillStyle='#667085';x.font='11px system-ui';
 for(let i=0;i<=4;i++){let yy=pad.t+(h-pad.t-pad.b)*i/4;x.beginPath();x.moveTo(pad.l,yy);x.lineTo(w-pad.r,yy);x.stroke();
  let v=ymax-(ymax-ymin)*i/4;x.fillText(opts.percent?(100*v).toFixed(0)+'%':opts.scientific?v.toExponential(1):v.toFixed(opts.digits??1),3,yy+4)}
 arms.forEach((a,j)=>{x.strokeStyle=colors[j%colors.length];x.lineWidth=2;x.beginPath();let started=false;
  a.series.forEach(p=>{let v=p[key];if(v==null||!isFinite(v))return;let xx=X(p.update),yy=Y(v);started?x.lineTo(xx,yy):x.moveTo(xx,yy);started=true});x.stroke();
  x.fillStyle=colors[j%colors.length];x.fillRect(pad.l+j*170,h-14,10,3);x.fillStyle='#344054';x.fillText(a.label,pad.l+15+j*170,h-9)});
 x.fillStyle='#667085';x.fillText('update',w-55,h-8)}
function render(s){document.getElementById('root').textContent=s.run_root;let st=document.getElementById('state');st.textContent=s.state;st.className='pill '+s.state;
 window.__last=s;
 document.getElementById('updated').textContent='\uAC31\uC2E0 '+new Date(s.generated_at).toLocaleString();let arms=s.arms||[];
 let cards=arms.map(a=>`<section class="panel span3"><div class="sub">${esc(a.label)}</div><div class="metric">${a.updates_completed} / ${a.planned_updates??'?'}</div>
 <div>update \u00B7 ${a.episodes_complete.toLocaleString()} episode</div><div class="sub">\uD559\uC2B5\uB9CC \uC608\uC0C1 \uC794\uC5EC ${duration(a.estimated_training_wall_seconds_remaining)}</div><div class="bar"><i style="width:${Math.min(100,100*(a.progress??0))}%"></i></div></section>
 <section class="panel span3"><div class="sub">${esc(a.label)} outcome</div><div class="metric good">${pct(a.success_rate)}</div>
 <div>timeout ${pct(a.timeout_rate)} \u00B7 unsafe <span class="${a.unsafe_rate?'bad':'good'}">${pct(a.unsafe_rate)}</span></div></section>`).join('');
 let table=arms.length?`<table><thead><tr><th>arm</th><th>update</th><th>episodes</th><th>success</th><th>timeout</th><th>unsafe</th><th>mean return</th><th>max KL</th><th>accepted/rejected</th></tr></thead><tbody>${arms.map(a=>`<tr><td>${esc(a.label)}</td><td>${a.updates_completed}/${a.planned_updates??'?'}</td><td>${a.episodes_complete}</td><td>${pct(a.success_rate)}</td><td>${pct(a.timeout_rate)}</td><td>${pct(a.unsafe_rate)}</td><td>${num(a.mean_return)}</td><td>${Number(a.max_post_step_kl).toExponential(2)}</td><td>${a.accepted_steps}/${a.rejected_steps}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">\uD559\uC2B5 history\uAC00 \uC544\uC9C1 \uC5C6\uC2B5\uB2C8\uB2E4.</div>';
 let evals=s.evaluations||[];let et=evals.length?`<table><thead><tr><th>stage</th><th>arm</th><th>episodes</th><th>mode</th><th>success</th><th>unsafe</th><th>mean return</th></tr></thead><tbody>${evals.map(e=>`<tr><td>${esc(e.stage)}</td><td>${esc(e.label)}</td><td>${e.episodes}</td><td>${e.deterministic===true?'deterministic':e.deterministic===false?'sampled':'\u2014'}</td><td>${pct(e.success_rate)}</td><td>${pct(e.unsafe_rate)}</td><td>${num(e.mean_return)}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">\uD3C9\uAC00 \uACB0\uACFC\uB294 \uC544\uC9C1 \uC5C6\uC2B5\uB2C8\uB2E4.</div>';
 document.getElementById('app').innerHTML=`<div class="grid">${cards}<section class="panel span12"><h2>\uD604\uC7AC \uD559\uC2B5 \uC694\uC57D</h2>${table}</section>
 <section class="panel span6"><h2>update\uBCC4 mean return</h2><canvas id="ret"></canvas></section><section class="panel span6"><h2>update\uBCC4 sampled SUCCESS</h2><canvas id="succ"></canvas></section>
 <section class="panel span6"><h2>post-step KL</h2><canvas id="kl"></canvas></section><section class="panel span6"><h2>update\uBCC4 TASK_TIMEOUT</h2><canvas id="timeout"></canvas></section>
 <section class="panel span12"><h2>Validation / Isaac \uACB0\uACFC</h2>${et}</section><section class="panel span6"><h2>\uC790\uB3D9 \uD574\uC11D</h2><ul>${(s.diagnosis||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></section>
 <section class="panel span6"><h2>\uC778\uD504\uB77C recovery</h2><div class="metric">${s.recoveries.count}</div><div>reset ${s.recoveries.reset_count} \u00B7 mid-episode ${s.recoveries.episode_count} \u00B7 post-cleanup ${s.recoveries.cleanup_count}</div><div class="sub">post-cleanup\uC740 \uC644\uB8CC rollout\uC744 \uBCF4\uC874, \uB098\uBA38\uC9C0\uB294 policy outcome/gradient\uC5D0\uC11C \uC81C\uC678</div></section></div>`;
 lineChart('ret',arms,'mean_return',{digits:0});lineChart('succ',arms,'success_rate',{min:0,max:1,percent:true});lineChart('kl',arms,'post_step_kl',{min:0,scientific:true});lineChart('timeout',arms,'timeout_rate',{min:0,max:1,percent:true})}
async function tick(){try{let s=EMBEDDED_STATE??await(await fetch('/api/state',{cache:'no-store'})).json();render(s)}catch(e){document.getElementById('app').innerHTML='<p class="bad">'+esc(e)+'</p>'}}
tick();if(EMBEDDED_STATE===null)setInterval(tick,5000);else setInterval(()=>location.reload(),10000);
window.addEventListener('resize',()=>{if(window.__last)render(window.__last)})
</script></body></html>"""


def export_report(run_root: str | Path, *, planned_updates: int | None = None,
                  episodes_per_update: int | None = None) -> dict[str, Any]:
    """Write a browser-openable report and its compact machine-readable state."""
    root = Path(run_root).expanduser().resolve()
    state = snapshot(root, planned_updates=planned_updates,
                     episodes_per_update=episodes_per_update)
    output = root / "visualization"
    _atomic_json(output / "state.json", state)
    encoded = json.dumps(state, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    output.joinpath("index.html").write_text(
        PAGE.replace("__EMBEDDED_STATE__", encoded), encoding="utf-8")
    return state
