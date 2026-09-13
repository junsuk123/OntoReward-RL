"""A live training dashboard, served from the pipeline process.

A paper-scale run is hours of real-time flight, so the question "is this still
going somewhere" has to be answerable from a laptop, over SSH, with nothing
installed. That rules out a plotting window and it rules out a JavaScript
bundle: this serves one self-contained HTML page from the standard library and
draws its charts on a canvas, so it works with no network access and no CDN.

Bound to ``127.0.0.1`` by default. It exposes training telemetry, which is not
sensitive, but it also has no authentication, so putting it on a public
interface is a deliberate act and not the default. Forward it instead::

    ssh -N -L 8770:127.0.0.1:8770 <host>
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .live import STORE, LiveStore

__all__ = ["Dashboard"]


def _saved_reward_scalars(cfg) -> dict[str, Any]:
    """Restore the last committed design while a new long run is collecting data."""
    out: dict[str, Any] = {}
    design_path = Path(cfg.paths.models) / "rgat_fixed_reward_external.json"
    try:
        design = json.loads(design_path.read_text(encoding="utf-8"))
        if design.get("format") == "ontology_rgat.fixed_reward/1" and design.get("frozen"):
            out.update(reward_weights=design["weights"],
                       reward_ranges=design["physical_ranges"],
                       reward_task_constants=design.get("sparse_task_constants", {}),
                       reward_design_id=design["design_id"],
                       reward_weights_frozen=True)
    except (OSError, ValueError, KeyError, TypeError):
        return out

    acceptance_path = Path(cfg.paths.results) / "optimization_acceptance.json"
    try:
        report = json.loads(acceptance_path.read_text(encoding="utf-8"))
        # Never display an old policy's verdict beside a newer reward design.
        if report.get("reward_design_id") != out.get("reward_design_id"):
            return out
        reward = report["reward_optimization"]
        consistency = report["rgat_consistency"]
        out.update(
            reward_success_rate=reward["success_rate"],
            reward_success_min=reward["minimum"],
            reward_optimization_pass=reward["pass"],
            rgat_consistency_std=consistency["std"],
            rgat_consistency_max_std=consistency["max_std"],
            rgat_worst_case_success=consistency["worst_case"],
            rgat_consistency_pass=consistency["pass"],
            optimization_overall_pass=report["overall_pass"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return out


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3쌍 병렬 Ontology-RGAT 학습</title>
<style>
:root{color-scheme:light;--bg:#f2f2f2;--card:#ffffff;--ink:#262626;--muted:#666666;
--line:#b8b8b8;--grid:#d8d8d8;--accent:#0072BD;--good:#77AC30;
--warn:#D95319;--bad:#A2142F}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.45 Arial,Helvetica,sans-serif}
header{padding:11px 18px;background:#e8e8e8;border-bottom:1px solid #8c8c8c;display:flex;
gap:14px;align-items:baseline;flex-wrap:wrap;box-shadow:0 1px 2px rgba(0,0,0,.08)}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:0}
#stage{font-weight:600;color:var(--accent)}
#age{color:var(--muted);font-variant-numeric:tabular-nums;margin-left:auto}
main{padding:14px 16px;display:grid;gap:12px;
grid-template-columns:repeat(3,minmax(250px,1fr))}
.card{background:var(--card);border:1px solid #a6a6a6;border-radius:2px;padding:11px 13px;
box-shadow:0 1px 2px rgba(0,0,0,.06)}
.card h2{font-size:13px;margin:0 0 6px;text-align:center;color:var(--ink);font-weight:600}
canvas{width:100%;height:205px;display:block;background:#fff}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.tile{background:#fafafa;border:1px solid #b7b7b7;border-top:3px solid var(--accent);
border-radius:1px;padding:7px 9px}
.tile b{display:block;font-size:18px;font-variant-numeric:tabular-nums;font-weight:600}
.tile span{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:4px}
.wide{grid-column:1/-1}
[hidden]{display:none!important}
.contract-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:7px}
.contract-item{border:1px solid #b8b8b8;background:#fafafa;padding:7px 9px;min-height:52px}
.contract-item b,.contract-item span{display:block}.contract-item b{color:var(--accent);font-size:11px}
.contract-item span{font-size:12px}.contract-item.forbidden{border-left:4px solid var(--bad)}
.method-strip{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
.method-chip{border:1px solid #a8a8a8;border-left:4px solid var(--accent);padding:5px 8px;
background:white;font-variant-numeric:tabular-nums}.method-chip b,.method-chip small{display:block}
.method-chip small{color:var(--muted)}
.pair-grid,.pair-plot-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.pair-card{border:1px solid #a8a8a8;border-top:4px solid var(--accent);background:#fafafa;
padding:9px 10px;min-height:150px}.pair-head{display:flex;justify-content:space-between;
gap:8px;align-items:baseline;margin-bottom:6px}.pair-head b{font-size:13px}.pair-head span{
color:var(--muted);font-size:10px}.pair-metrics{display:grid;grid-template-columns:repeat(3,1fr);
gap:5px}.pair-metrics div{background:#fff;border:1px solid #d0d0d0;padding:4px 5px}
.pair-metrics b,.pair-metrics small{display:block}.pair-metrics b{font-size:13px;
font-variant-numeric:tabular-nums}.pair-metrics small{font-size:9px;color:var(--muted)}
.pair-links{margin-top:6px;color:var(--muted);font:9px/1.45 "Courier New",monospace;
overflow-wrap:anywhere}.pair-state{font-weight:600}.pair-state.running{color:var(--accent)}
.pair-state.success{color:var(--good)}.pair-state.failure,.pair-state.unsafe_touchdown{
color:var(--bad)}
.pair-plot{min-width:0;border:1px solid #b8b8b8;background:#fff;padding:7px}
.pair-plot h3{height:34px;margin:0 0 3px;font-size:11px;line-height:1.3;text-align:center}
.pair-plot canvas{height:170px}.pair-gates{display:flex;gap:3px;flex-wrap:wrap;margin-top:6px}
.pair-gates i{font-style:normal;font-size:9px;padding:1px 4px;border:1px solid #bbb;
background:#fff}.pair-gates i.pass{border-color:var(--good);color:#4a7620}
.pair-gates i.fail{border-color:var(--bad);color:var(--bad)}
.phase-status{display:flex;align-items:center;gap:10px;padding:8px 10px;border-left:5px solid var(--accent);
background:#f6f9fb}.phase-status b{font-size:13px}.phase-status span{color:var(--muted)}
.phase-status.training{border-left-color:var(--warn)}.phase-status.evaluation{border-left-color:var(--good)}
.benchmark-formula{margin-top:9px;padding:6px 9px;border:1px solid #b8b8b8;background:#f7f7f7;
font:12px/1.5 "Courier New",monospace;text-align:center}
.g3d{position:relative}
.g3d canvas{height:420px;cursor:grab;touch-action:none}
.g3d canvas.drag{cursor:grabbing}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px;
font-size:11px;color:var(--muted)}
.bar label{display:flex;gap:5px;align-items:center}
.bar input[type=range]{width:104px}
.bar button{font:inherit;font-size:11px;padding:2px 9px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}
.tip{position:absolute;pointer-events:none;z-index:2;background:var(--card);
border:1px solid var(--line);border-radius:7px;padding:5px 8px;font-size:11px;
line-height:1.4;box-shadow:0 3px 10px rgba(0,0,0,.22);white-space:nowrap;display:none}
.relbar{display:grid;grid-template-columns:auto 1fr auto;gap:5px 9px;align-items:center;
font-size:11px;color:var(--muted);margin-top:10px}
.relbar .track{background:var(--line);border-radius:3px;height:6px;overflow:hidden}
.relbar .track i{display:block;height:6px;border-radius:3px}
.relbar b{font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}
.relbar em{font-style:normal;opacity:.75}
.note{font-size:11px;color:var(--muted);margin-top:8px}
@media(max-width:780px){main{grid-template-columns:1fr}.pair-grid,.pair-plot-grid{
grid-template-columns:1fr}}
</style></head><body>
<header><h1 id="page-title">3쌍 병렬 비전 착륙 &middot; Isaac Sim + PX4 + PPO</h1>
<span id="stage">connecting</span><span id="detail"></span><span id="age"></span></header>
<main id="root"></main>
<script>
// MATLAB default color order (R2025a), shared with the PNG exporters.
const PALETTE=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'];
const BENCHMARK_METHODS=['shin_se','no_se','onto_no_se','shin2026','sparse',
  'manual_no_active','ontoreward','ontoreward_plus_active',
  'shin_se_fixed','shin_se_rgat_weight','no_se_fixed',
  'onto_rgat_adaptive_weight_no_se','onto_rgat_potential_pbrs_no_se',
  'mlp_adaptive_weight_no_se','gat_adaptive_weight_no_se','rgat_adaptive_weight_no_se'];
const BENCHMARK_TRAIN=BENCHMARK_METHODS.map(x=>'benchmark_train_'+x);
const BENCHMARK_EVAL=BENCHMARK_METHODS.map(x=>'benchmark_eval_'+x);
const BENCHMARK_STEP=BENCHMARK_METHODS.map(x=>'benchmark_step_'+x);
const CARDS=[
 {id:'tiles',title:null},
 {id:'phase_status',view:'benchmark',kind:'phase'},
 {id:'parallel_pairs',view:'benchmark',kind:'pairs',
  title:'동시 비행쌍 · 한 Isaac Sim 월드 / 독립 PX4·PPO'},
 {id:'parallel_live_reward',view:'benchmark',kind:'pairplots',plot:'reward',
  title:'실시간 보상 · pair별 독립 trajectory'},
 {id:'parallel_live_flight',view:'benchmark',kind:'pairplots',plot:'flight',
  title:'실시간 비행 상태 · pair별 독립 sensor'},
 {id:'parallel_live_method',view:'benchmark',kind:'pairplots',plot:'method',
  title:'방법론 고유 신호 · SE 오차 / 시각 관측 / R-GAT 의미 상태'},
 {id:'benchmark_contract',view:'benchmark',kind:'contract',
  title:'세 방법론 공통 RL 계약과 정보 경계'},
 {id:'benchmark_eval_success',view:'benchmark',title:'[현재 평가] 이동 성공률',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'paper_success',smooth:5,ymin:0,ymax:1},
 {id:'benchmark_eval_return',view:'benchmark',title:'[현재 평가] episode 누적 보상',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'episode_return',smooth:5},
 {id:'benchmark_eval_position',view:'benchmark',title:'[현재 평가] SE 위치 RMSE',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'position_rmse',smooth:5},
 {id:'benchmark_eval_scenario',view:'benchmark',kind:'evalbars',
  title:'[현재 평가] Scenario별 성공률'},
 {id:'benchmark_eval_velocity',view:'benchmark',title:'[현재 평가] SE 속도 RMSE',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'velocity_rmse',smooth:5},
 {id:'benchmark_eval_visual_loss',view:'benchmark',
  title:'[현재 평가] 표적 FOV 소실 구간의 추정 오차',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'visual_loss_estimation_error',smooth:5},
 {id:'benchmark_success',view:'benchmark',title:'[완료된 학습 기록] 이동 성공률',
  series:BENCHMARK_TRAIN,x:'episode',y:'paper_success',smooth:40,ymin:0,ymax:1},
 {id:'benchmark_return',view:'benchmark',
  title:'[완료된 학습 기록] 누적 보상 (진단값)',
  series:BENCHMARK_TRAIN,x:'episode',y:'episode_return',smooth:20},
 {id:'benchmark_curriculum',view:'benchmark',title:'UGV 운동 curriculum c',
  series:BENCHMARK_TRAIN,x:'episode',y:'curriculum',ymin:0,ymax:1},
 {id:'benchmark_action_scale',view:'benchmark',title:'UAV action-envelope curriculum',
  series:BENCHMARK_TRAIN,x:'episode',y:'action_envelope_scale',ymin:0,ymax:1},
 {id:'benchmark_position_rmse',view:'benchmark',title:'SE 위치 RMSE',
  series:BENCHMARK_TRAIN,x:'episode',y:'position_rmse',smooth:12},
 {id:'benchmark_velocity_rmse',view:'benchmark',title:'SE 속도 RMSE',
  series:BENCHMARK_TRAIN,x:'episode',y:'velocity_rmse',smooth:12},
 {id:'benchmark_aux',view:'benchmark',title:'정규화 6-state 보조 손실',
  series:BENCHMARK_TRAIN,x:'episode',y:'auxiliary_estimation_loss',smooth:12},
 {id:'benchmark_active_saturation',view:'benchmark',title:'Active reward 포화율',
  series:BENCHMARK_TRAIN,x:'episode',y:'active_reward_saturation_fraction',
  smooth:12,ymin:0,ymax:1},
 {id:'benchmark_adaptive_weights',view:'benchmark',
  title:'동결 adaptive reward 가중치 (episode 평균)',series:BENCHMARK_TRAIN,
  x:'episode',y:['adaptive_weight_1_mean','adaptive_weight_2_mean',
   'adaptive_weight_3_mean','adaptive_weight_4_mean','adaptive_weight_5_mean'],
  labels:['lateral','vertical','vz safety','undershoot','yaw']},
 {id:'benchmark_adaptive_latency',view:'benchmark',title:'Adaptive R-GAT 추론 지연',
  series:BENCHMARK_TRAIN,x:'episode',y:'adaptive_rgat_inference_latency_ms_mean',smooth:12},
 {id:'benchmark_policy_loss',view:'benchmark',title:'Recurrent PPO policy 손실',
  series:BENCHMARK_TRAIN,x:'episode',y:'ppo_loss',smooth:12},
 {id:'benchmark_value_loss',view:'benchmark',title:'Asymmetric critic value 손실',
  series:BENCHMARK_TRAIN,x:'episode',y:'value_loss',smooth:12},
 {id:'benchmark_entropy',view:'benchmark',title:'PPO policy entropy',
  series:BENCHMARK_TRAIN,x:'episode',y:'entropy',smooth:12},
 {id:'benchmark_kl',view:'benchmark',title:'PPO 근사 KL divergence',
  series:BENCHMARK_TRAIN,x:'episode',y:'kl_divergence',smooth:12},
 {id:'benchmark_lr',view:'benchmark',title:'실효 PPO learning rate',
  series:BENCHMARK_TRAIN,x:'episode',y:'effective_learning_rate'},
 {id:'benchmark_early_stop',view:'benchmark',title:'PPO KL 조기 종료율',
  series:BENCHMARK_TRAIN,x:'episode',y:'ppo_early_stop',smooth:12,ymin:0,ymax:1},
 {id:'benchmark_battery_used',view:'benchmark',title:'비행당 실제 pack 사용 에너지 (J)',
  series:BENCHMARK_TRAIN,x:'episode',y:'battery_energy_used_j',smooth:12},
 {id:'benchmark_battery_final',view:'benchmark',title:'종료 시 정규화 배터리 여유',
  series:BENCHMARK_TRAIN,x:'episode',y:'battery_reserve_final',smooth:12,ymin:0,ymax:1},
 {id:'benchmark_battery_depleted',view:'benchmark',title:'배터리 고갈 종료율',
  series:BENCHMARK_TRAIN,x:'episode',y:'battery_depleted',smooth:20,ymin:0,ymax:1},
 {id:'benchmark_reacquisition',view:'benchmark',title:'FOV 소실 후 재관측률',
  series:BENCHMARK_TRAIN,x:'episode',y:'visual_reacquisition_rate',smooth:12,ymin:0,ymax:1},
 {id:'benchmark_recovery_landing',view:'benchmark',title:'소실 → 재관측 → 착륙 성공률',
  series:BENCHMARK_TRAIN,x:'episode',y:'successful_recovery_landing',smooth:20,ymin:0,ymax:1},
 {id:'benchmark_unsafe_blind_descent',view:'benchmark',title:'저시인성 상태의 위험 하강률',
  series:BENCHMARK_TRAIN,x:'episode',y:'unsafe_descent_low_visibility_fraction',
  smooth:12,ymin:0,ymax:1},
 {id:'graph3d',view:'benchmark',kind:'graph',
  title:'Pair 3 · 학습된 ontology graph와 R-GAT attention'},
];
const LABELS={benchmark_step:'current episode',
 benchmark_train_shin_se:'A · Shin SE',benchmark_train_no_se:'B · No SE',
 benchmark_train_onto_no_se:'C · Onto No SE',
 benchmark_eval_shin_se:'A · Shin SE',benchmark_eval_no_se:'B · No SE',
 benchmark_eval_onto_no_se:'C · Onto No SE',
 benchmark_train_shin_se_fixed:'Shin SE · fixed',
 benchmark_train_shin_se_rgat_weight:'Shin SE · R-GAT weight',
 benchmark_train_no_se_fixed:'No SE · fixed',
 benchmark_train_onto_rgat_adaptive_weight_no_se:'Onto R-GAT adaptive · no SE',
 benchmark_train_onto_rgat_potential_pbrs_no_se:'Onto R-GAT potential PBRS · no SE',
 benchmark_eval_shin_se_fixed:'Shin SE · fixed',
 benchmark_eval_shin_se_rgat_weight:'Shin SE · R-GAT weight',
 benchmark_eval_no_se_fixed:'No SE · fixed',
 benchmark_eval_onto_rgat_adaptive_weight_no_se:'Onto R-GAT adaptive · no SE',
 benchmark_eval_onto_rgat_potential_pbrs_no_se:'Onto R-GAT potential PBRS · no SE',
 benchmark_train_shin2026:'Shin + active',benchmark_train_sparse:'Sparse',
 benchmark_train_manual_no_active:'Shin − active',benchmark_train_ontoreward:'OntoReward',
 benchmark_train_ontoreward_plus_active:'OntoReward + active',
 benchmark_eval_shin2026:'Shin + active',benchmark_eval_sparse:'Sparse',
 benchmark_eval_manual_no_active:'Shin − active',benchmark_eval_ontoreward:'OntoReward',
 benchmark_eval_ontoreward_plus_active:'OntoReward + active'};
const root=document.getElementById('root');
for(const c of CARDS){
  const el=document.createElement('section');
  el.className='card'+((c.id==='tiles'||['graph','contract','evalbars','pairs','pairplots','phase'].includes(c.kind))?' wide':'')
    +(c.kind==='graph'?' g3d':'');
  el.id='card-'+c.id;
  el.dataset.view=c.view||'common';
  el.hidden=c.view==='benchmark';
  if(c.id==='tiles'){el.innerHTML='<div class="tiles" id="tiles"></div>';}
  else if(c.kind==='phase'){el.innerHTML='<div class="phase-status" id="phase-status"></div>';}
  else if(c.kind==='pairs'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="pair-grid" id="parallel-pair-grid"></div>`;}
  else if(c.kind==='pairplots'){el.innerHTML=`<h2>${c.title}</h2><div class="pair-plot-grid">`
    +[0,1,2].map(index=>`<div class="pair-plot"><h3 id="pair-title-${c.id}-${index}">`
      +`Pair ${index+1} · 초기화 대기</h3><canvas id="cv-${c.id}-${index}"></canvas>`
      +`<div class="legend" id="lg-${c.id}-${index}"></div></div>`).join('')+'</div>';}
  else if(c.kind==='contract'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="contract-grid" id="benchmark-contract"></div>
    <div class="method-strip" id="benchmark-methods"></div>
    <div class="benchmark-formula">actor = &pi;(y[6:256], u<sub>UAV</sub>)
      &nbsp;&nbsp;|&nbsp;&nbsp; critic<sub>train</sub> = V(u<sub>UAV</sub>, s<sub>rel,true</sub>)
      &nbsp;&nbsp;|&nbsp;&nbsp; action = [v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>,
      &omega;<sub>yaw</sub>]</div>`;}
  else if(c.kind==='evalbars'){el.innerHTML=`<h2>${c.title}</h2>
    <canvas id="cv-${c.id}" style="height:270px"></canvas>
    <div class="legend" id="lg-${c.id}"></div>`;}
  else if(c.kind==='graph'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="bar"><span id="g3d-src">waiting for a graph</span>
      <label style="margin-left:auto"><input type="checkbox" id="g3d-auto" checked>spin</label>
      <label>hide weak edges<input type="range" id="g3d-floor" min="0" max="0.9"
        step="0.05" value="0"></label>
      <button id="g3d-reset">reset view</button></div>
    <canvas id="cv-graph3d"></canvas><div class="tip" id="g3d-tip"></div>
    <div class="relbar" id="rel-graph3d"></div>
    <div class="note">Drag to rotate, wheel to zoom, hover a node to isolate its links.
      Depth is distance from the raw semantic channels to SafeLanding; node size and
      colour are the channel's current activation; edge width and opacity are the
      second R-GAT layer's attention, which is learned importance and not causal
      proof. Self-loops are not drawn.</div>`;}
  else{el.innerHTML=`<h2>${c.title}</h2><canvas id="cv-${c.id}"></canvas>
    <div class="legend" id="lg-${c.id}"></div>`;}
  root.appendChild(el);
}
function smooth(v,w){if(w<2||v.length<2)return v;const o=[];let s=0;
  for(let i=0;i<v.length;i++){s+=v[i];if(i>=w)s-=v[i-w];o.push(s/Math.min(i+1,w));}return o;}
function draw(card,state){
  const cv=document.getElementById('cv-'+card.id);if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const css=getComputedStyle(document.body);
  const line=css.getPropertyValue('--grid').trim();
  const muted=css.getPropertyValue('--muted').trim();
  const lines=[];const yKeys=Array.isArray(card.y)?card.y:[card.y];
  const sources=card.series||[];
  for(const s of sources){
    const rows=state.series[s]||[];if(!rows.length)continue;
    for(const key of yKeys){
      const xs=[],ys=[];
      for(const r of rows){const yv=Number(r[key]),xv=Number(r[card.x]);
        if(!Number.isFinite(yv)||!Number.isFinite(xv))continue;
        xs.push(xv);ys.push(yv);}
      if(!xs.length)continue;
      const label=(card.kind==='pairplots')
        ?(card.labels?card.labels[yKeys.indexOf(key)]:key)
        :(card.labels&&yKeys.length>1)
        ?`${LABELS[s]||s} ${card.labels[yKeys.indexOf(key)]}`
        :(yKeys.length>1?(card.labels?card.labels[yKeys.indexOf(key)]:key):(LABELS[s]||s));
      const benchmarkMethod=s.replace(/^benchmark_(train|eval)_/,'');
      const benchmarkIndex=BENCHMARK_METHODS.indexOf(benchmarkMethod);
      const colorIndex=benchmarkIndex>=0?benchmarkIndex:
        (sources.length>1?sources.indexOf(s):yKeys.indexOf(key));
      lines.push({xs,ys:smooth(ys,card.smooth||1),label,
        color:PALETTE[Math.max(0,colorIndex)%PALETTE.length]});
    }
  }
  const lg=document.getElementById('lg-'+card.id);
  if(!lines.length){g.fillStyle=muted;g.font='12px sans-serif';
    g.fillText('no data yet',10,h/2);if(lg)lg.innerHTML='';return;}
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(const l of lines){for(let i=0;i<l.xs.length;i++){
    x0=Math.min(x0,l.xs[i]);x1=Math.max(x1,l.xs[i]);
    y0=Math.min(y0,l.ys[i]);y1=Math.max(y1,l.ys[i]);}}
  if(card.ymin!==undefined)y0=card.ymin;if(card.ymax!==undefined)y1=card.ymax;
  if(x1===x0){x0-=0.5;x1+=0.5;}if(y1===y0){y1+=0.5;y0-=0.5;}
  if(card.ymin===undefined&&card.ymax===undefined){const margin=(y1-y0)*0.06;
    y0-=margin;y1+=margin;}
  const pad={l:48,r:10,t:9,b:25};
  const px=v=>pad.l+(v-x0)/(x1-x0)*(w-pad.l-pad.r);
  const py=v=>h-pad.b-(v-y0)/(y1-y0)*(h-pad.t-pad.b);
  g.strokeStyle=line;g.lineWidth=.7;g.fillStyle=muted;g.font='10px Arial';
  g.setLineDash([1.5,2.5]);
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,y=py(v);
    g.beginPath();g.moveTo(pad.l,y);g.lineTo(w-pad.r,y);g.stroke();
    g.fillText(v.toFixed(Math.abs(v)>=100?0:2),4,y+3);}
  for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4,x=px(v);
    g.beginPath();g.moveTo(x,pad.t);g.lineTo(x,h-pad.b);g.stroke();}
  g.setLineDash([]);g.strokeStyle='#262626';g.lineWidth=.8;
  g.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
  g.fillStyle=muted;g.fillText(formatTick(x0),pad.l,h-7);
  g.textAlign='right';g.fillText(formatTick(x1),w-pad.r,h-7);g.textAlign='left';
  lines.forEach(l=>{g.strokeStyle=l.color;g.lineWidth=1.8;
    g.beginPath();for(let k=0;k<l.xs.length;k++){
      const X=px(l.xs[k]),Y=py(l.ys[k]);k?g.lineTo(X,Y):g.moveTo(X,Y);}g.stroke();});
  if(lg)lg.innerHTML=lines.map(l=>
    `<span><i style="background:${l.color}"></i>${escapeHTML(l.label)}</span>`).join('');
}
function formatTick(v){const a=Math.abs(v);return a>=1000?v.toExponential(1):
  (a>=100?Number(v).toFixed(0):a>=10?Number(v).toFixed(1):Number(v).toFixed(2));}
function drawPairPlots(card,state){
  const pairs=(state.scalars||{}).parallel_pair_status||[];
  for(let index=0;index<3;index++){
    const pair=pairs.find(item=>Number(item.index)===index)||pairs[index]||{};
    const method=String(pair.method||'');
    const title=document.getElementById(`pair-title-${card.id}-${index}`);
    if(title)title.textContent=`Pair ${index+1} · ${LABELS['benchmark_train_'+method]||method||'초기화 대기'}`;
    let spec;
    if(card.plot==='reward')spec={
      y:['reward','task','shape','active_perception'],
      labels:['전체','terminal task','PBRS shaping','active perception']};
    else if(card.plot==='flight')spec={
      y:['uav_speed_m_s','ugv_speed_m_s','in_fov','battery_reserve'],
      labels:['UAV 속도','UGV 속도','표적 FOV','배터리 여유'],ymin:0};
    else if(method.startsWith('shin_se')||method==='shin2026')spec={
      y:['position_error','velocity_error','estimation_loss','in_fov'],
      labels:['위치 오차','속도 오차','6-state 손실','표적 FOV'],ymin:0};
    else if(method.includes('rgat')||method.includes('onto'))spec={
      y:['phi','phi_next','shape','semantic_image_alignment','semantic_visual_loss_risk'],
      labels:['Phi(G_t)','Phi(G_t+1)','PBRS','image alignment','visual-loss risk']};
    else spec={
      y:['in_fov','lateral_progress','vertical_progress','vertical_speed_penalty'],
      labels:['표적 FOV','수평 progress','수직 progress','수직속도 penalty']};
    draw({...card,...spec,id:`${card.id}-${index}`,
      series:method?[`benchmark_step_${method}`]:[],x:'step'},state);
  }
}
function tiles(state){
  const s=state.scalars||{};const out=[];
  const add=(label,value)=>out.push(
    `<div class="tile"><b>${escapeHTML(value)}</b><span>${escapeHTML(label)}</span></div>`);
  const methods=s.benchmark_methods||[];
  const trained=methods.reduce((n,m)=>n+(state.series['benchmark_train_'+m]||[]).length,0);
  add('단계',state.stage.name);
  add('실험 phase',s.benchmark_phase||'initializing');
  add('UAV / UGV pair',Number(s.parallel_pair_count||3));
  add('실행 모드',s.benchmark_mode||'--');
  const trainingDone=trained>=Number(s.training_total||0)&&Number(s.training_total||0)>0;
  add('학습 checkpoint',`${trained} / ${s.training_total||0}${trainingDone?' · 완료':' · 진행'}`);
  add('현재 평가 진행',`${s.evaluation_completed||0} / ${s.evaluation_total||0}`);
  methods.forEach((method,index)=>{
    const rows=state.series['benchmark_eval_'+method]||[];
    const success=rows.length?rows.reduce((sum,row)=>sum+Number(row.paper_success||0),0)/rows.length:null;
    add(`Pair ${index+1} 평가`,`${rows.length}회 · ${success===null?'대기':(100*success).toFixed(1)+'%'}`);
  });
  if(s.rgat_dataset_episodes!==undefined){
    add('R-GAT 실제 비행 데이터',`${s.rgat_dataset_episodes} ep / ${s.rgat_dataset_samples||0}`);
    add('R-GAT 안전 접촉',`${s.rgat_dataset_successes||0} / ${s.rgat_dataset_episodes}`);}
  if(s.config_hash)add('설정 hash',String(s.config_hash).slice(0,10));
  if(s.reward_design_id)add('보상 설계',String(s.reward_design_id).slice(0,16));
  document.getElementById('tiles').innerHTML=out.join('');
}
function phasePanel(state){
  const s=state.scalars||{},phase=String(s.benchmark_phase||'initializing');
  const box=document.getElementById('phase-status');if(!box)return;
  box.className='phase-status '+phase;
  if(phase==='evaluation')box.innerHTML='<b>학습 완료 · 현재 paired evaluation 갱신 중</b>'+
    '<span>아래의 “완료된 학습 기록” 그래프와 96/96 checkpoint 값은 더 변하지 않습니다. '+
    '현재 변화는 pair별 live plot, 평가 진행 수와 “[현재 평가]” 그래프에서 확인하십시오.</span>';
  else if(phase==='training')box.innerHTML='<b>PPO 학습 진행 중</b>'+
    '<span>episode가 종료되고 optimizer와 checkpoint 기록이 완료될 때 학습 그래프가 증가합니다.</span>';
  else box.innerHTML='<b>실험 초기화 중</b><span>pair 연결 및 artifact 준비 상태를 확인하고 있습니다.</span>';
}
function escapeHTML(value){return String(value??'--').replace(/[&<>"']/g,c=>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function pairPanel(state){
  const box=document.getElementById('parallel-pair-grid');if(!box)return;
  const s=state.scalars||{},pairs=s.parallel_pair_status||s.parallel_pair_layout||[];
  box.innerHTML=[0,1,2].map(index=>{
    const pair=pairs.find(item=>Number(item.index)===index)||pairs[index]||{index:index};
    const method=pair.method||'--',rows=state.series['benchmark_step_'+method]||[];
    const live=rows.length?rows[rows.length-1]:{},xyz=pair.relative_xyz||null;
    const step=pair.step??live.step??0,status=String(pair.status||live.status||'waiting');
    const marker=pair.marker_visible===null||pair.marker_visible===undefined
      ?'대기':(pair.marker_visible?'관측':'소실');
    const reserve=Number(pair.battery_reserve??live.battery_reserve);
    const gate=pair.landing_gate||null;
    const pos=xyz&&xyz.length===3
      ?`${Number(xyz[0]).toFixed(2)}, ${Number(xyz[1]).toFixed(2)}, ${Number(xyz[2]).toFixed(2)}`:'--';
    return `<div class="pair-card" style="border-top-color:${PALETTE[index%PALETTE.length]}">`
      +`<div class="pair-head"><b>Pair ${index+1} · ${escapeHTML(LABELS['benchmark_train_'+method]||method)}</b>`
      +`<span class="pair-state ${escapeHTML(status)}">${escapeHTML(status.toUpperCase())}</span></div>`
      +`<div class="pair-metrics"><div><b>${escapeHTML(pair.episode??0)} / ${escapeHTML(step)}</b><small>${escapeHTML(pair.episode_kind||'episode')} / 스텝</small></div>`
      +`<div><b>${escapeHTML(marker)}</b><small>마커</small></div>`
      +`<div><b>${Number(pair.ugv_speed_m_s??live.ugv_speed_m_s??0).toFixed(2)} m/s</b><small>UGV 속도</small></div>`
      +`<div><b>${Number(pair.uav_speed_m_s??live.uav_speed_m_s??0).toFixed(2)} m/s</b><small>UAV 속도</small></div>`
      +`<div><b>${Number.isFinite(reserve)?(100*reserve).toFixed(1)+'%':'--'}</b><small>배터리 잔량</small></div>`
      +`<div><b>${escapeHTML(pos)}</b><small>패드 상대 XYZ (m)</small></div></div>`
      +(gate?`<div class="pair-gates">`+
        [['접촉','contact'],['위치','position'],['수직속도','vertical_speed'],
         ['수평상대속도','relative_horizontal_speed'],['자세','attitude'],['각속도','angular_rate']]
        .map(([label,key])=>{const value=gate[key],ready=value!==null&&value!==undefined;
          const pass=ready&&Number(value)===1;return `<i class="${ready?(pass?'pass':'fail'):''}">`
          +`${label} ${ready?(pass?'통과':'실패'):'대기'}</i>`;}).join('')+'</div>':'')
      +`<div class="pair-links">${escapeHTML(pair.phase||'waiting')} · ${escapeHTML(String(pair.scenario||'--').replaceAll('_',' '))}<br>`
      +`${escapeHTML(pair.activity||'')} ${escapeHTML(pair.activity_detail||'')}<br>`
      +`route ${Number(pair.route_phase_fraction||0).toFixed(2)} · PX4 ${escapeHTML(pair.px4_namespace)} · UDP ${escapeHTML(pair.gateway_port)}/${escapeHTML(pair.learner_port)}<br>`
      +`${escapeHTML(pair.rviz_namespace)} · ${escapeHTML(pair.camera_topic)}</div></div>`;
  }).join('');
}
function benchmarkPanel(state){
  const s=state.scalars||{},contract=s.actor_contract||{};
  const labels={camera:'Actor image',proprioception:'Actor proprioception',
    temporal:'Shared temporal backbone',state_estimation:'Optional state supervision',
    actor:'Deployment actor',critic:'Asymmetric critic',
    reward_side:'Reward-side ontology state',
    forbidden:'Hard information boundary'};
  document.getElementById('benchmark-contract').innerHTML=Object.entries(labels).map(([key,label])=>
    `<div class="contract-item ${key==='forbidden'?'forbidden':''}"><b>${label}</b>`+
    `<span>${escapeHTML(contract[key]||'--')}</span></div>`).join('');
  const methods=s.benchmark_methods||[];
  const methodChips=methods.map((method,index)=>{
    const rows=state.series['benchmark_train_'+method]||[],n=Math.min(50,rows.length);
    const success=n?rows.slice(-n).reduce((sum,row)=>sum+Number(row.paper_success||0),0)/n:null;
    // The run-level budget can include a Shin-only estimator warm-up, so it
    // cannot be divided evenly into an honest per-pipeline denominator.
    const progress=rows.length+' completed';
    const rate=success===null?'waiting':(100*success).toFixed(1)+'% success';
    return `<div class="method-chip" style="border-left-color:${PALETTE[index%PALETTE.length]}">`+
      `<b>${escapeHTML(LABELS['benchmark_train_'+method]||method)}</b>`+
      `<small>${progress} episodes · ${rate}</small></div>`;}).join('');
  document.getElementById('benchmark-methods').innerHTML=methodChips;
}
function drawEvaluationBars(card,state){
  const cv=document.getElementById('cv-'+card.id),lg=document.getElementById('lg-'+card.id);
  if(!cv)return;
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const methods=(state.scalars.benchmark_methods||[]).filter(method=>
    (state.series['benchmark_eval_'+method]||[]).length);
  const preferred=['training_random_walk','straight_8mps','linear_acceleration_wave',
    'circle','zigzag','u_turn','vertical_heave_boat'];
  const present=new Set();
  for(const method of methods)for(const row of state.series['benchmark_eval_'+method]||[])
    present.add(row.scenario);
  const scenarios=preferred.filter(x=>present.has(x));
  for(const value of present)if(!scenarios.includes(value))scenarios.push(value);
  if(!methods.length||!scenarios.length){g.fillStyle='#666';g.font='12px Arial';
    g.fillText('no paired evaluation data yet',12,h/2);lg.innerHTML='';return;}
  const pad={l:48,r:12,t:10,b:67},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
  g.strokeStyle='#D8D8D8';g.lineWidth=.7;g.setLineDash([1.5,2.5]);
  g.fillStyle='#666';g.font='10px Arial';
  for(let i=0;i<=4;i++){const y=pad.t+ph*(1-i/4);
    g.beginPath();g.moveTo(pad.l,y);g.lineTo(w-pad.r,y);g.stroke();
    g.fillText((i/4).toFixed(2),8,y+3);}
  g.setLineDash([]);
  const group=pw/scenarios.length,barWidth=Math.min(24,group*.76/methods.length);
  scenarios.forEach((scenario,si)=>{
    methods.forEach((method,mi)=>{
      const rows=(state.series['benchmark_eval_'+method]||[]).filter(r=>r.scenario===scenario);
      if(!rows.length)return;
      const mean=rows.reduce((sum,row)=>sum+Number(row.paper_success||0),0)/rows.length;
      const x=pad.l+group*si+(group-barWidth*methods.length)/2+mi*barWidth;
      const height=Math.max(0,Math.min(1,mean))*ph;
      g.fillStyle=PALETTE[BENCHMARK_METHODS.indexOf(method)%PALETTE.length];
      g.fillRect(x,pad.t+ph-height,Math.max(1,barWidth-1),height);
    });
    const label=scenario.replace('training_','').replaceAll('_',' ');
    g.save();g.translate(pad.l+group*(si+.5),h-pad.b+8);g.rotate(-Math.PI/7);
    g.fillStyle='#4d4d4d';g.textAlign='right';g.font='10px Arial';g.fillText(label,0,0);g.restore();
  });
  g.strokeStyle='#262626';g.lineWidth=.8;g.strokeRect(pad.l,pad.t,pw,ph);
  lg.innerHTML=methods.map(method=>{
    const index=BENCHMARK_METHODS.indexOf(method);
    return `<span><i style="background:${PALETTE[index%PALETTE.length]}"></i>`+
      `${escapeHTML(LABELS['benchmark_eval_'+method]||method)}</span>`;}).join('');
}
// ------------------------------------------------------------ 3D ontology
// Hand-rolled: the page has to work with no network, so there is no three.js
// to reach for. Fourteen nodes and forty edges is well inside what a painter's
// algorithm on a 2D canvas can do at 60 Hz, and doing it by hand is what lets
// the depth cue, the attention width and the hover isolation share one pass.
const REL_COLORS=['#c0392b','#0d8c4d','#1a59bf','#8a8f98'];
const G={yaw:-0.55,pitch:0.30,zoom:1,auto:true,floor:0,hover:-1,drag:null,
         data:null,dirty:true,pts:[],fit:0};
const clamp01=v=>Math.max(0,Math.min(1,v));
function riskColor(v){const t=clamp01(v);
  return [Math.round(255*(0.20+0.72*t)),Math.round(255*(0.70-0.50*t)),
          Math.round(255*(0.28-0.12*t))];}
function nodeRGB(n){
  if(n.role==='goal')return [242,199,68];
  return riskColor(n.role==='risk'?n.value:1-n.value);}
const rgba=(c,a)=>`rgba(${c[0]},${c[1]},${c[2]},${a})`;
function hexRGB(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),
  parseInt(h.slice(5,7),16)];}
const DIST=8.2;
function project(p,w,h){
  const cy=Math.cos(G.yaw),sy=Math.sin(G.yaw);
  const x=p[0]*cy-p[1]*sy, y=p[0]*sy+p[1]*cy;
  const cp=Math.cos(G.pitch),sp=Math.sin(G.pitch);
  const y2=y*cp-p[2]*sp, z2=y*sp+p[2]*cp;
  const d=y2+DIST, f=Math.min(w,h)*2.0/Math.max(d,0.4);
  return {x:x*f, y:-z2*f, d:d, f:f};
}
// Nearer is brighter: without this the ring behind the graph reads as the
// ring in front of it and the rotation stops telling you anything.
const fade=d=>Math.max(0.22,Math.min(1,1.45-(d-DIST+2.2)*0.30));
function graphDraw(){
  const cv=document.getElementById('cv-graph3d');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.clientHeight;
  if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){
    cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const css=getComputedStyle(document.body);
  const muted=css.getPropertyValue('--muted').trim();
  const ink=css.getPropertyValue('--ink').trim();
  const card=css.getPropertyValue('--card').trim();
  const data=G.data;G.pts=[];
  if(!data||!data.nodes||!data.nodes.length){
    g.fillStyle=muted;g.font='12px sans-serif';
    g.fillText('no graph published yet',12,h/2);return;}
  // Fit the rotated cloud to the card rather than trusting a fixed scale: the
  // graph is much wider than it is tall, so a scale that suits it end-on
  // wastes most of the canvas edge-on. Low-passed, or the spin would pump.
  const raw=data.nodes.map(n=>project(n.pos,w,h));
  let bx0=Infinity,bx1=-Infinity,by0=Infinity,by1=-Infinity;
  for(const q of raw){bx0=Math.min(bx0,q.x);bx1=Math.max(bx1,q.x);
    by0=Math.min(by0,q.y);by1=Math.max(by1,q.y);}
  const inset=56;   // the labels sit above the nodes and need the room
  const target=Math.min((w-2*inset)/Math.max(bx1-bx0,1),
                        (h-2*inset)/Math.max(by1-by0,1));
  G.fit=G.fit?G.fit*0.86+target*0.14:target;
  const k=G.fit*G.zoom;
  const ox=w/2-(bx0+bx1)/2*k, oy=h/2-(by0+by1)/2*k;
  const P=raw.map(q=>({x:q.x*k+ox, y:q.y*k+oy, d:q.d, f:q.f*k}));
  const radius=(n,q)=>Math.max(
    (n.role==='goal'?0.105:0.050+0.048*clamp01(n.value))*q.f, 6);
  let peak=0;
  for(const e of data.edges)if(e.s!==e.d&&e.a!==undefined)peak=Math.max(peak,e.a);
  const strength=e=>(e.a===undefined||peak<=0)?0.34:e.a/peak;
  const items=[];
  for(const e of data.edges){
    if(e.s===e.d)continue;
    const a=strength(e);
    if(a<G.floor)continue;
    const A=P[e.s],B=P[e.d];
    const lit=G.hover<0||e.s===G.hover||e.d===G.hover;
    items.push({d:(A.d+B.d)/2,draw:()=>{
      const rgb=hexRGB(REL_COLORS[e.r%REL_COLORS.length]);
      const alpha=(0.14+0.66*a)*fade((A.d+B.d)/2)*(lit?1:0.10);
      const dx=B.x-A.x,dy=B.y-A.y,len=Math.hypot(dx,dy)||1;
      const ux=dx/len,uy=dy/len;
      // Trim to the node discs so the arrowhead lands on the rim, not inside.
      const x0=A.x+ux*(A.f*0.052+3),y0=A.y+uy*(A.f*0.052+3);
      const x1=B.x-ux*(B.f*0.062+5),y1=B.y-uy*(B.f*0.062+5);
      g.strokeStyle=rgba(rgb,alpha);g.lineWidth=0.7+2.9*a*(lit?1:0.6);
      g.beginPath();g.moveTo(x0,y0);g.lineTo(x1,y1);g.stroke();
      const head=4+5*a;
      g.fillStyle=rgba(rgb,alpha);
      g.beginPath();g.moveTo(x1,y1);
      g.lineTo(x1-ux*head+uy*head*0.45,y1-uy*head-ux*head*0.45);
      g.lineTo(x1-ux*head-uy*head*0.45,y1-uy*head+ux*head*0.45);
      g.closePath();g.fill();}});
  }
  data.nodes.forEach((n,i)=>{
    const p=P[i],r=radius(n,p);
    G.pts.push({x:p.x,y:p.y,r:r,i:i});
    const lit=G.hover<0||G.hover===i;
    items.push({d:p.d,draw:()=>{
      const rgb=nodeRGB(n),al=fade(p.d)*(lit?1:0.30);
      const grad=g.createRadialGradient(p.x-r*0.35,p.y-r*0.4,r*0.15,p.x,p.y,r);
      grad.addColorStop(0,rgba([Math.min(255,rgb[0]+70),Math.min(255,rgb[1]+70),
                                Math.min(255,rgb[2]+70)],al));
      grad.addColorStop(1,rgba(rgb,al));
      g.fillStyle=grad;g.beginPath();g.arc(p.x,p.y,r,0,6.2832);g.fill();
      if(G.hover===i){g.strokeStyle=ink;g.lineWidth=1.4;g.stroke();}}});
  });
  items.sort((a,b)=>b.d-a.d);
  for(const it of items)it.draw();
  // Labels are placed near-to-far, so a node in front keeps the spot it wants
  // and one behind is nudged clear of it, and then drawn far-to-near so the
  // front of the graph still paints on top. Nothing is ever dropped: a label
  // that finds no free slot is drawn faded rather than silently lost.
  g.textAlign='center';g.font='600 10.5px ui-sans-serif,system-ui,sans-serif';
  const boxes=[],placed=[];
  const near=data.nodes.map((n,i)=>i).sort((a,b)=>P[a].d-P[b].d);
  for(const i of near){
    const n=data.nodes[i],p=P[i];
    const text=n.role==='goal'?n.name:`${n.name} ${n.value.toFixed(2)}`;
    const tw=g.measureText(text).width;
    const x=Math.max(tw/2+4,Math.min(w-tw/2-4,p.x));
    const base=p.y-radius(n,p)-6;
    let spot=null;
    for(const dy of [0,-17,17,-34,34,-51,51,-68,68]){
      const b={x0:x-tw/2-3,x1:x+tw/2+3,y0:base+dy-11,y1:base+dy+4};
      if(b.y0<2||b.y1>h-2)continue;
      if(boxes.some(o=>o.x0<b.x1&&b.x0<o.x1&&o.y0<b.y1&&b.y0<o.y1))continue;
      boxes.push(b);spot={x:x,y:base+dy,text:text,free:true};break;
    }
    placed[i]=spot||{x:x,y:base,text:text,free:false};
  }
  for(const i of near.slice().reverse()){
    const p=P[i],L=placed[i];
    // The depth cue belongs on the geometry: fading text as well leaves the
    // back half of the graph labelled with something nobody can read. Text
    // keeps a floor, and a halo in the card colour lifts it off the edges
    // crossing behind it.
    const al=Math.max(0.80,fade(p.d))*(G.hover<0||G.hover===i?1:0.25)
      *(L.free?1:0.5);
    g.lineWidth=3.5;g.lineJoin='round';
    g.strokeStyle=rgba(hexRGB(card),Math.min(1,al*1.15));
    g.strokeText(L.text,L.x,L.y);
    g.fillStyle=rgba(hexRGB(ink),al);g.fillText(L.text,L.x,L.y);
  }
  g.textAlign='left';
}
function graphLegend(){
  const box=document.getElementById('rel-graph3d');if(!box)return;
  const data=G.data;
  if(!data||!data.relations){box.innerHTML='';return;}
  const peak=Math.max(1e-9,...data.relations.map(r=>r.mean||0));
  // The self-relation usually carries the largest mean and never appears in
  // the picture, so the bar has to say why it has no edges to point at.
  const drawn=new Set(data.edges.filter(e=>e.s!==e.d).map(e=>e.r));
  box.innerHTML=data.relations.map((r,i)=>{
    const c=REL_COLORS[i%REL_COLORS.length];
    const m=r.mean;
    const tag=drawn.has(i)?'':' <em>(self-loops, not drawn)</em>';
    return `<span style="color:${c}">&#9632; ${r.name}${tag}</span>`
      +`<span class="track"><i style="width:${m===undefined?0:100*m/peak}%;`
      +`background:${c}"></i></span>`
      +`<b>${m===undefined?'&mdash;':m.toFixed(3)}</b>`;}).join('');
}
function graphTip(ev){
  const tip=document.getElementById('g3d-tip');
  const data=G.data;
  if(G.hover<0||!data){tip.style.display='none';return;}
  const n=data.nodes[G.hover];
  let best=null;
  for(const e of data.edges){
    if(e.s===e.d||e.a===undefined)continue;
    if(e.s!==G.hover&&e.d!==G.hover)continue;
    if(!best||e.a>best.a)best=e;
  }
  const rel=best?data.relations[best.r].name:null;
  tip.innerHTML=`<b>${n.name}</b><br>activation ${n.value.toFixed(3)}`
    +` &middot; ${n.role} &middot; depth ${n.layer}`
    +(best?`<br>strongest link: ${data.nodes[best.s].name} &rarr;`
       +` ${data.nodes[best.d].name} (${rel}, &alpha;=${best.a.toFixed(3)})`:'');
  const card=document.getElementById('card-graph3d').getBoundingClientRect();
  tip.style.display='block';
  tip.style.left=Math.min(ev.clientX-card.left+12,card.width-tip.offsetWidth-8)+'px';
  tip.style.top=(ev.clientY-card.top+12)+'px';
}
function graphBind(){
  const cv=document.getElementById('cv-graph3d');if(!cv)return;
  cv.addEventListener('pointerdown',e=>{
    G.drag={x:e.clientX,y:e.clientY};cv.classList.add('drag');
    cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointermove',e=>{
    if(G.drag){
      G.yaw+=(e.clientX-G.drag.x)*0.008;
      G.pitch=Math.max(-1.3,Math.min(1.3,G.pitch+(e.clientY-G.drag.y)*0.006));
      G.drag={x:e.clientX,y:e.clientY};G.hover=-1;G.dirty=true;
      document.getElementById('g3d-tip').style.display='none';return;}
    const r=cv.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top;
    let hit=-1,bestD=Infinity;
    for(const p of G.pts){
      const d=Math.hypot(p.x-mx,p.y-my);
      if(d<p.r+7&&d<bestD){bestD=d;hit=p.i;}}
    if(hit!==G.hover){G.hover=hit;G.dirty=true;}
    graphTip(e);});
  const release=e=>{if(G.drag){G.drag=null;cv.classList.remove('drag');}};
  cv.addEventListener('pointerup',release);
  cv.addEventListener('pointercancel',release);
  cv.addEventListener('pointerleave',()=>{
    G.hover=-1;G.dirty=true;document.getElementById('g3d-tip').style.display='none';});
  cv.addEventListener('wheel',e=>{
    e.preventDefault();
    G.zoom=Math.max(0.45,Math.min(3.2,G.zoom*(e.deltaY<0?1.12:0.89)));
    G.dirty=true;},{passive:false});
  document.getElementById('g3d-auto').addEventListener('change',
    e=>{G.auto=e.target.checked;});
  document.getElementById('g3d-floor').addEventListener('input',
    e=>{G.floor=parseFloat(e.target.value);G.dirty=true;});
  document.getElementById('g3d-reset').addEventListener('click',()=>{
    G.yaw=-0.55;G.pitch=0.30;G.zoom=1;G.dirty=true;});
  (function frame(){
    if(G.auto&&!G.drag&&G.hover<0){G.yaw+=0.0032;G.dirty=true;}
    if(G.dirty){G.dirty=false;graphDraw();}
    requestAnimationFrame(frame);})();
}
graphBind();

function applyProfile(state){
  const profile='benchmark';
  document.body.dataset.profile=profile;
  document.getElementById('page-title').textContent=
    '3쌍 병렬 비전 착륙 · Isaac Sim + PX4 + PPO';
  for(const c of CARDS){
    const el=document.getElementById('card-'+c.id);
    el.hidden=Boolean(c.view&&c.view!=='common'&&c.view!==profile);
  }
  if(profile==='benchmark'){
    const pipeline=state.scalars.current_pipeline||state.scalars.current_method||'';
    const active=Number(state.scalars.parallel_pair_count||1)>1
      ?(state.scalars.benchmark_methods||[]):[pipeline];
    const estimator=active.some(x=>x.startsWith('shin_se')||x==='shin2026');
    for(const id of ['benchmark_position_rmse','benchmark_velocity_rmse','benchmark_aux',
                     'benchmark_active_saturation',
                     'benchmark_eval_position','benchmark_eval_velocity',
                     'benchmark_eval_visual_loss']){
      const el=document.getElementById('card-'+id);if(el)el.hidden=!estimator;
    }
    const adaptive=active.some(x=>x.includes('adaptive_weight')||x==='shin_se_rgat_weight');
    for(const id of ['benchmark_adaptive_weights','benchmark_adaptive_latency']){
      const el=document.getElementById('card-'+id);if(el)el.hidden=!adaptive;
    }
  }
  return profile;
}
let lastRevision=-1,lastAt=0;
async function tick(){
  try{
    const r=await fetch('api/state',{cache:'no-store'});
    const state=await r.json();
    document.getElementById('stage').textContent=state.stage.name;
    document.getElementById('detail').textContent=state.stage.detail||'';
    if(state.revision!==lastRevision){
      lastRevision=state.revision;lastAt=Date.now();
      const profile=applyProfile(state);
      tiles(state);
      phasePanel(state);
      if(profile==='benchmark')pairPanel(state);
      for(const c of CARDS){
        if(c.id==='tiles'||c.kind==='graph'||c.kind==='contract'||c.kind==='pairs'||
           c.kind==='phase'||
           c.kind==='pairplots'||
           (c.view&&c.view!=='common'&&c.view!==profile))continue;
        c.kind==='evalbars'?drawEvaluationBars(c,state):draw(c,state);
      }
      for(const c of CARDS.filter(item=>item.kind==='pairplots'))drawPairPlots(c,state);
      benchmarkPanel(state);
      const gs=state.graph||null;
      const stamp=gs?`${gs.source||'graph'}${gs.attention?'':' (schema only, '
        +'R-GAT 학습 전)'}${gs.phi!==undefined?`  \u03a6=${gs.phi.toFixed(3)}`:''}`
        :'ontology graph 초기화 대기';
      document.getElementById('g3d-src').textContent=stamp;
      G.data=gs;G.dirty=true;graphLegend();
    }
  }catch(e){document.getElementById('stage').textContent='disconnected';}
  const age=lastAt?Math.round((Date.now()-lastAt)/1000):0;
  document.getElementById('age').textContent=lastAt?`updated ${age}s ago`:'';
}
tick();setInterval(tick,1500);
window.addEventListener('resize',()=>{lastRevision=-1;G.dirty=true;});
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    store: LiveStore = STORE
    protocol_version = "HTTP/1.1"

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                          # noqa: N802 - stdlib API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            payload = json.dumps(self.store.snapshot(), default=_jsonable).encode("utf-8")
            self._send(payload, "application/json")
        else:
            self.send_error(404)

    def log_message(self, *args: Any) -> None:
        """Silence the per-request logging; the poll would drown the run's output."""


def _jsonable(value: Any) -> Any:
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class Dashboard:
    """The HTTP server, on a daemon thread so it never holds the run open."""

    def __init__(self, cfg, store: LiveStore | None = None):
        self.cfg = cfg
        self.store = store or STORE
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        opt = self.cfg.viz.dashboard
        return f"http://{opt.host}:{opt.port}/"

    def start(self) -> "Dashboard | None":
        opt = self.cfg.viz.dashboard
        if not opt.enabled:
            return None
        self.store.set(
            reward_lambda=float(self.cfg.reward.pbrs["lambda"]),
            reward_gamma=float(self.cfg.reward.pbrs.gamma),
            reward_formula="r_sparse + lambda * (gamma * frozen_R_GAT(G') - frozen_R_GAT(G))",
            **_saved_reward_scalars(self.cfg),
        )
        handler = type("BoundHandler", (_Handler,), {"store": self.store})
        try:
            self.server = ThreadingHTTPServer((str(opt.host), int(opt.port)), handler)
        except OSError as exc:
            print(f"Live dashboard disabled: cannot bind {opt.host}:{opt.port} ({exc}).")
            return None
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.4}, daemon=True)
        self.thread.start()
        print(f"Live dashboard: {self.url}")
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def __enter__(self) -> "Dashboard":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
