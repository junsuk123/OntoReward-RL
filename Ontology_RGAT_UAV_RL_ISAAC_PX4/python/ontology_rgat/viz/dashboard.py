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
<title>2쌍 Shin baseline / Ontology-R-GAT FOV 비교</title>
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
.section{grid-column:1/-1;background:none;border:0;box-shadow:none;padding:9px 0 0}
.section h2{font-size:14px;margin:0 0 3px;text-align:left;
border-bottom:2px solid var(--accent);padding-bottom:4px;letter-spacing:.01em}
.section p{margin:5px 0 0;color:var(--muted);font-size:11px;max-width:105ch}
.tile.delta{border-top-color:var(--warn)}
.tile.delta b{font-variant-numeric:tabular-nums}
.tile small{display:block;color:var(--muted);font-size:9px;margin-top:2px}
/* run pipeline: a staged track whose state comes from the live stage name */
.runpipe{display:flex;gap:0;align-items:stretch;flex-wrap:nowrap;overflow-x:auto;
padding-bottom:4px}
.runstage{flex:1 1 0;min-width:132px;border:1px solid #b8b8b8;background:#fafafa;
padding:7px 9px;position:relative;margin-right:13px}
.runstage:last-child{margin-right:0}
.runstage::after{content:'';position:absolute;right:-13px;top:50%;width:13px;height:2px;
background:var(--line)}
.runstage:last-child::after{display:none}
.runstage b{display:block;font-size:11.5px;line-height:1.25;padding-right:34px}
.runstage span{display:block;color:var(--muted);font-size:9.5px;margin-top:3px}
.runstage em{display:block;font-style:normal;font-size:9.5px;margin-top:4px;
padding-top:4px;border-top:1px dotted #cfcfcf;color:#4d4d4d}
.runstage.done{border-left:4px solid var(--good);background:#f4f9ee}
.runstage.active{border-left:4px solid var(--accent);background:#fff;
box-shadow:0 0 0 2px rgba(0,114,189,.18)}
.runstage.pending{border-left:4px solid var(--line);opacity:.62}
.runstage .kind{position:absolute;top:6px;right:7px;font-size:8px;letter-spacing:.05em;
text-transform:uppercase;color:#8a8a8a}
.runstage.active .kind{color:var(--accent);font-weight:700}
.runnote{margin-top:8px;color:var(--muted);font-size:10.5px}
/* algorithm pipeline: one scalable diagram, no canvas */
.algowrap{width:100%;overflow-x:auto}
.algowrap svg{width:100%;min-width:900px;height:auto;display:block}
.algolimits{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px;font-size:10.5px;
color:var(--muted)}
.algolimits b{color:var(--bad);font-weight:600;margin-right:4px}
/* MDP: identical rows merge across both arms, the differing one splits */
.mdp{display:grid;grid-template-columns:150px 1fr 1fr;gap:5px;font-size:11.5px}
.mdp .hd{font-weight:700;padding:5px 7px;border-bottom:2px solid var(--line);font-size:11px}
.mdp .hd.a0{border-bottom-color:var(--accent)}
.mdp .hd.a1{border-bottom-color:var(--warn)}
.mdp .rk{padding:6px 7px;font-weight:600;background:#fafafa;border:1px solid #e0e0e0}
.mdp .same{grid-column:2/4;padding:6px 8px;border:1px solid #dcdcdc;background:#fbfbfb}
.mdp .diff{padding:6px 8px;border:1px solid #d6b08f;background:#fff7f0}
.mdp .diff.base{border-color:#bcd4e6;background:#f3f8fc}
.mdp .same .tag{display:inline-block;font-size:9px;color:var(--good);border:1px solid var(--good);
padding:0 4px;margin-right:6px;vertical-align:1px}
.mdp .diff .tag{display:inline-block;font-size:9px;color:var(--warn);border:1px solid var(--warn);
padding:0 4px;margin-right:6px;vertical-align:1px}
.mdp .nt{display:block;color:var(--muted);font-size:10px;margin-top:3px}
.mdpsum{margin-top:9px;font-size:11px;color:var(--muted)}
.mdpsum b{color:var(--warn)}
/* FOV readout status: three stages of the proposed arm's own preparation */
.fovwrap{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
.fovbox{border:1px solid #b8b8b8;background:#fafafa;padding:8px 10px;
border-top:3px solid var(--line)}
.fovbox.run{border-top-color:var(--accent);background:#fff}
.fovbox.done{border-top-color:var(--good);background:#f6faf1}
.fovbox h3{margin:0 0 5px;font-size:11.5px;display:flex;justify-content:space-between;
align-items:baseline;gap:8px}
.fovbox h3 i{font-style:normal;font-size:9px;text-transform:uppercase;
letter-spacing:.05em;color:#8a8a8a}
.fovbox.run h3 i{color:var(--accent);font-weight:700}
.fovbox.done h3 i{color:var(--good);font-weight:700}
.fovrow{display:flex;justify-content:space-between;gap:8px;font-size:11px;
padding:2px 0;border-bottom:1px dotted #e2e2e2}
.fovrow:last-child{border-bottom:0}
.fovrow b{font-variant-numeric:tabular-nums;font-weight:600}
.fovrow.warn b{color:var(--warn)}
.fovrow.bad b{color:var(--bad)}
.fovrow.good b{color:var(--good)}
.fovbar{height:7px;background:#e6e6e6;margin:5px 0 7px;position:relative}
.fovbar i{display:block;height:100%;background:var(--accent)}
.fovbar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--bad);
text-decoration:none}
.fovempty{color:var(--muted);font-size:11px;padding:6px 0}
[hidden]{display:none!important}
.contract-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:7px}
.contract-item{border:1px solid #b8b8b8;background:#fafafa;padding:7px 9px;min-height:52px}
.contract-item b,.contract-item span{display:block}.contract-item b{color:var(--accent);font-size:11px}
.contract-item span{font-size:12px}.contract-item.forbidden{border-left:4px solid var(--bad)}
.method-strip{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
.method-chip{border:1px solid #a8a8a8;border-left:4px solid var(--accent);padding:5px 8px;
background:white;font-variant-numeric:tabular-nums}.method-chip b,.method-chip small{display:block}
.method-chip small{color:var(--muted)}
.pair-grid,.pair-plot-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.pair-card{border:1px solid #a8a8a8;border-top:4px solid var(--accent);background:#fafafa;
padding:9px 10px;min-height:150px}.pair-head{display:flex;justify-content:space-between;
gap:8px;align-items:baseline;margin-bottom:6px}.pair-head b{font-size:13px}.pair-head span{
color:var(--muted);font-size:10px}.pair-metrics{display:grid;grid-template-columns:repeat(3,1fr);
gap:5px}.pair-metrics div{background:#fff;border:1px solid #d0d0d0;padding:4px 5px}
.pair-metrics b,.pair-metrics small{display:block}.pair-metrics b{font-size:13px;
font-variant-numeric:tabular-nums}.pair-metrics small{font-size:9px;color:var(--muted)}
.pair-rewards{display:grid;grid-template-columns:1fr 1.65fr;gap:5px;margin-top:6px}
.pair-reward{border:1px solid #c5c5c5;border-left:4px solid var(--accent);background:#fff;
padding:5px 7px}.pair-reward.added{border-left-color:var(--warn)}
.pair-reward.off{border-left-color:var(--line);color:var(--muted)}
.pair-reward b,.pair-reward small{display:block}.pair-reward b{font-size:11px;
font-variant-numeric:tabular-nums}.pair-reward small{font-size:9px;color:var(--muted)}
.pair-links{margin-top:6px;color:var(--muted);font:9px/1.45 "Courier New",monospace;
overflow-wrap:anywhere}.pair-state{font-weight:600}.pair-state.running{color:var(--accent)}
.pair-assignment{margin:-1px 0 7px;padding:3px 5px;border-left:3px solid var(--line);
background:#f1f1f1;color:var(--muted);font-size:10px}.pair-assignment.rotated{
border-left-color:var(--warn);color:var(--ink);font-weight:600}
.pair-state.success{color:var(--good)}.pair-state.failure,.pair-state.unsafe_touchdown{
color:var(--bad)}
.pair-state.complete{color:var(--good)}
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
.bar select{max-width:330px;font:inherit;font-size:11px;padding:2px 6px;border:1px solid var(--line);
background:var(--card);color:var(--ink)}
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
.module-flow{display:flex;gap:6px;align-items:stretch;overflow-x:auto;margin:9px 0 7px;
padding-bottom:2px}.module-box{min-width:155px;flex:1;border:1px solid var(--line);
border-top:3px solid var(--accent);background:#fafafa;padding:6px 8px}.module-box.head{
border-top-color:var(--warn)}.module-box b,.module-box small{display:block}.module-box b{font-size:11px}
.module-box small{font-size:9px;color:var(--muted)}.module-arrow{align-self:center;color:var(--muted);
font-size:18px}.graph-audit{display:grid;grid-template-columns:minmax(230px,.8fr) minmax(360px,1.4fr)
minmax(300px,1fr);gap:8px;margin-top:9px}.audit-panel{border:1px solid var(--line);
background:#fff;min-width:0}.audit-panel h3{font-size:11px;margin:0;padding:5px 7px;
background:#f1f1f1;border-bottom:1px solid var(--line)}.audit-scroll{max-height:245px;overflow:auto}
.audit-table{width:100%;border-collapse:collapse;font:9px/1.35 "Courier New",monospace;
font-variant-numeric:tabular-nums}.audit-table th{position:sticky;top:0;background:#f7f7f7;z-index:1}
.audit-table th,.audit-table td{padding:3px 5px;border-bottom:1px solid #e5e5e5;text-align:right;
white-space:nowrap}.audit-table th:nth-child(2),.audit-table td:nth-child(2){text-align:left}
.audit-empty{padding:12px;color:var(--muted);font-size:10px}
.note{font-size:11px;color:var(--muted);margin-top:8px}
@media(max-width:780px){main{grid-template-columns:1fr}.pair-grid,.pair-plot-grid{
grid-template-columns:1fr}.graph-audit{grid-template-columns:1fr}}
</style></head><body>
<header><h1 id="page-title">2쌍 Shin baseline / Ontology-R-GAT FOV 비교</h1>
<span id="stage">connecting</span><span id="detail"></span><span id="age"></span></header>
<main id="root"></main>
<script>
// MATLAB default color order (R2025a), shared with the PNG exporters.
const PALETTE=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'];
const BENCHMARK_METHODS=['shin_se_fixed','shin_se_onto_rgat_recovery'];
const BENCHMARK_TRAIN=BENCHMARK_METHODS.map(x=>'benchmark_train_'+x);
const BENCHMARK_EVAL=BENCHMARK_METHODS.map(x=>'benchmark_eval_'+x);
const BENCHMARK_STEP=BENCHMARK_METHODS.map(x=>'benchmark_step_'+x);
const PROPOSED_TRAIN=['benchmark_train_shin_se_onto_rgat_recovery'];
const CARDS=[
 {id:'tiles',title:null},
 {id:'phase_status',view:'benchmark',kind:'phase'},
 {id:'run_pipeline',view:'benchmark',kind:'runpipe',
  title:'시뮬레이션 · 학습 · 검증 파이프라인 (현재 위치 표시)'},

 {id:'head_performance',view:'benchmark',kind:'heading',
  title:'1 · 두 모델 성능',
  note:'인식·상태추정·actor·critic·action·제어기·환경·원문 보상항·종료가 동일하고, '
      +'동일 초기 checkpoint와 동일 PPO 예산을 쓴다. 두 곡선의 차이는 학습 전용 '
      +'온톨로지-R-GAT 보상 하나뿐이다.'},
 {id:'benchmark_eval_success',view:'benchmark',title:'[평가] 안전 착륙 성공률',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'paper_success',smooth:5,ymin:0,ymax:1},
 {id:'benchmark_eval_position',view:'benchmark',title:'[평가] 상대 위치 RMSE (m)',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'position_rmse',smooth:5},
 {id:'benchmark_eval_velocity',view:'benchmark',title:'[평가] 상대 속도 RMSE (m/s)',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'velocity_rmse',smooth:5},
 {id:'benchmark_eval_return',view:'benchmark',title:'[평가] episode 누적 보상',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'episode_return',smooth:5},
 {id:'benchmark_success',view:'benchmark',title:'[학습] 안전 착륙 성공률',
  series:BENCHMARK_TRAIN,x:'episode',y:'paper_success',smooth:40,ymin:0,ymax:1},
 {id:'benchmark_return',view:'benchmark',title:'[학습] 누적 보상 (진단값)',
  series:BENCHMARK_TRAIN,x:'episode',y:'episode_return',smooth:20},
 {id:'benchmark_eval_scenario',view:'benchmark',kind:'evalbars',
  title:'[평가] scenario별 성공률'},
 // Training now rotates one analytic deck motion per episode instead of
 // flying a single random walk, so the aggregate training curves above mix six
 // difficulties. This separates them.
 {id:'benchmark_train_scenario',view:'benchmark',kind:'evalbars',
  series:BENCHMARK_TRAIN,metric:'paper_success',
  empty:'아직 학습 scenario 데이터가 없다',
  title:'[학습] scenario별 성공률'},

 {id:'head_ontology',view:'benchmark',kind:'heading',
  title:'2 · 온톨로지-R-GAT의 영향력',
  note:'추가 보상은 비종료 step에서 r_paper(t) − λ·q(G_{t+1}) 하나다. q는 향후 1초 중 '
      +'패드 중심이 FOV 밖인 시간 비율의 기댓값이며 이진 확률이 아니다. 아래는 그 항이 '
      +'실제로 무엇을 바꿨는지와, 동결된 readout이 맞았는지를 나눠 본다.'},
 {id:'fov_status',view:'benchmark',kind:'fovstatus',
  title:'제안 arm 준비 상태 · FOV 데이터 수집 → readout 학습 → 동결'},
 {id:'fov_offline_training',view:'benchmark',
  title:'FOV readout 오프라인 학습 (epoch)',
  series:['fov_risk_training'],x:'epoch',
  y:['train_huber','validation_huber','validation_contract'],
  labels:['학습 Huber','검증 Huber','검증 규약 위반'],smooth:1},
 {id:'benchmark_fov_loss',view:'benchmark',title:'FOV 소실 시간 비율 (낮을수록 좋음)',
  series:BENCHMARK_TRAIN,x:'episode',y:'geometric_fov_loss_fraction',
  smooth:20,ymin:0,ymax:1},
 {id:'benchmark_reacquisition',view:'benchmark',title:'소실 후 재관측률',
  series:BENCHMARK_TRAIN,x:'episode',y:'geometric_fov_reacquisition_rate',
  smooth:12,ymin:0,ymax:1},
 {id:'benchmark_recovery_landing',view:'benchmark',title:'소실 → 재관측 → 착륙 성공률',
  series:BENCHMARK_TRAIN,x:'episode',y:'successful_recovery_landing',
  smooth:20,ymin:0,ymax:1},
 {id:'benchmark_unsafe_blind_descent',view:'benchmark',title:'저시인성 상태의 위험 하강률',
  series:BENCHMARK_TRAIN,x:'episode',y:'descent_during_low_keypoint_visibility_fraction',
  smooth:12,ymin:0,ymax:1},
 {id:'benchmark_eval_visual_loss',view:'benchmark',
  title:'[평가] FOV 소실 구간의 상태추정 오차',
  series:BENCHMARK_EVAL,x:'evaluation_index',y:'geometric_fov_loss_estimation_error',smooth:5},
 {id:'benchmark_fov_calibration',view:'benchmark',
  title:'readout 보정 · 예측 대 실측 FOV 비가용 비율',
  series:PROPOSED_TRAIN,x:'episode',y:['fov_predicted_mean','fov_actual_mean'],
  labels:['예측 q','실측 y'],smooth:12,ymin:0,ymax:1},
 {id:'benchmark_fov_prediction_error',view:'benchmark',
  title:'readout 오차 · MAE와 편향 (관측된 미래창만)',
  series:PROPOSED_TRAIN,x:'episode',y:['fov_prediction_mae','fov_prediction_bias'],
  labels:['MAE','편향(예측−실측)'],smooth:12},
 {id:'benchmark_onto_share',view:'benchmark',
  title:'추가 보상이 차지한 step 보상 크기 비중',
  series:PROPOSED_TRAIN,x:'episode',y:'ontology_fov_reward_share',smooth:12,ymin:0},
 {id:'benchmark_onto_sum',view:'benchmark',
  title:'episode당 추가 보상 합 −λ·Σq (항상 ≤ 0)',
  series:PROPOSED_TRAIN,x:'episode',y:'ontology_fov_reward_sum',smooth:12},
 {id:'benchmark_fov_risk',view:'benchmark',
  title:'현재 episode · 예측 FOV 비가용 비율과 추가 보상',
  series:['benchmark_step_shin_se_onto_rgat_recovery'],x:'step',
  y:['predicted_fov_unavailability','ontology_fov_reward'],
  labels:['예측 q (향후 1s FOV 밖 시간 비율)','추가 보상 −λ·q'],smooth:3},
 {id:'graph3d',view:'benchmark',kind:'graph',
  title:'FOV 온톨로지 → R-GAT 비가용 비율 readout'},

 {id:'head_run',view:'benchmark',kind:'heading',
  title:'3 · 실행 상태',
  note:'비교 대상이 아니라 실행이 건전한지 보는 값이다. 두 arm에 동일하게 적용되며, '
      +'여기서의 차이는 결론이 아니라 교란 가능성의 단서로만 읽는다.'},
 {id:'parallel_pairs',view:'benchmark',kind:'pairs',
  title:'동시 비행쌍 · 한 Isaac Sim 월드 / 독립 PX4·PPO'},
 {id:'parallel_live_perception',view:'benchmark',kind:'pairplots',plot:'perception',
  title:'실시간 기하 FOV 대 keypoint 인지 품질 · 두 arm 동일 정의'},
 {id:'parallel_live_reward',view:'benchmark',kind:'pairplots',plot:'reward',
  title:'실시간 보상 분해 · pair별 독립 trajectory'},
 {id:'benchmark_curriculum',view:'benchmark',title:'curriculum · UGV 운동 c와 action envelope',
  series:BENCHMARK_TRAIN,x:'episode',y:['curriculum','action_envelope_scale'],
  labels:['UGV 운동 c','action envelope'],ymin:0,ymax:1},
 {id:'benchmark_position_rmse',view:'benchmark',title:'[학습] 상대 위치 RMSE (m)',
  series:BENCHMARK_TRAIN,x:'episode',y:'position_rmse',smooth:12},
 {id:'benchmark_ppo_loss',view:'benchmark',title:'PPO policy / critic 손실',
  series:BENCHMARK_TRAIN,x:'episode',y:['ppo_loss','value_loss'],
  labels:['policy','value'],smooth:12},
 {id:'benchmark_explore',view:'benchmark',title:'PPO entropy / 근사 KL',
  series:BENCHMARK_TRAIN,x:'episode',y:['entropy','kl_divergence'],
  labels:['entropy','KL'],smooth:12},
 {id:'benchmark_active_saturation',view:'benchmark',
  title:'공통 active-perception 보상 포화율 (교란 감시)',
  series:BENCHMARK_TRAIN,x:'episode',y:'active_reward_saturation_fraction',
  smooth:12,ymin:0,ymax:1},
 {id:'benchmark_battery_depleted',view:'benchmark',
  title:'배터리 고갈 종료율 (원문에 없는 이 백엔드의 종료 조건)',
  series:BENCHMARK_TRAIN,x:'episode',y:'battery_depleted',smooth:20,ymin:0,ymax:1},
 {id:'benchmark_contract',view:'benchmark',kind:'contract',
  title:'두 방법론의 동일 Shin RL 계약과 추가 FOV 분기'},

 {id:'head_structure',view:'benchmark',kind:'heading',
  title:'4 · 알고리즘과 MDP 구조',
  note:'측정값이 아니라 설계다. 수치는 실행 중인 설정과 코드 상수에서 만들어지므로 '
      +'그래프 node를 바꾸거나 λ를 바꾸면 이 그림도 함께 바뀐다.'},
 {id:'algorithm_pipeline',view:'benchmark',kind:'algopipe',
  title:'제안 알고리즘 · 센서 → 온톨로지/R-GAT 그래프 → 보상 함수'},
 {id:'mdp_structure',view:'benchmark',kind:'mdp',
  title:'두 강화학습 에이전트의 State · Action · Environment · Reward'},
];

const LABELS={benchmark_step:'current episode',
 benchmark_train_shin_se_fixed:'Baseline · Shin SE fixed',
 benchmark_train_shin_se_onto_rgat_recovery:'Proposed · Shin + Ontology-R-GAT FOV',
 benchmark_eval_shin_se_fixed:'Baseline · Shin SE fixed',
 benchmark_eval_shin_se_onto_rgat_recovery:'Proposed · Shin + Ontology-R-GAT FOV'};
const root=document.getElementById('root');
for(const c of CARDS){
  const el=document.createElement('section');
  el.className=(c.kind==='heading'?'card section'
    :'card'+((c.id==='tiles'||['graph','contract','evalbars','pairs','pairplots','phase',
      'runpipe','algopipe','mdp','fovstatus']
      .includes(c.kind))?' wide':''))
    +(c.kind==='graph'?' g3d':'');
  el.id='card-'+c.id;
  el.dataset.view=c.view||'common';
  el.hidden=c.view==='benchmark';
  if(c.id==='tiles'){el.innerHTML='<div class="tiles" id="tiles"></div>';}
  else if(c.kind==='heading'){el.innerHTML=`<h2>${c.title}</h2>`
    +(c.note?`<p>${c.note}</p>`:'');}
  else if(c.kind==='phase'){el.innerHTML='<div class="phase-status" id="phase-status"></div>';}
  else if(c.kind==='fovstatus'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="fovwrap" id="fov-status"></div>`;}
  else if(c.kind==='runpipe'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="runpipe" id="run-pipeline"></div>
    <div class="runnote" id="run-pipeline-note"></div>`;}
  else if(c.kind==='algopipe'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="algowrap" id="algo-pipeline"></div>
    <div class="algolimits" id="algo-limits"></div>`;}
  else if(c.kind==='mdp'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="mdp" id="mdp-grid"></div>
    <div class="mdpsum" id="mdp-summary"></div>`;}
  else if(c.kind==='pairs'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="pair-grid" id="parallel-pair-grid"></div>`;}
  else if(c.kind==='pairplots'){el.innerHTML=`<h2>${c.title}</h2><div class="pair-plot-grid">`
    +[0,1].map(index=>`<div class="pair-plot"><h3 id="pair-title-${c.id}-${index}">`
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
    <div class="bar"><select id="g3d-select" aria-label="graph module"></select>
      <span id="g3d-src">waiting for a graph</span>
      <label style="margin-left:auto"><input type="checkbox" id="g3d-auto" checked>spin</label>
      <label>hide weak edges<input type="range" id="g3d-floor" min="0" max="0.9"
        step="0.05" value="0"></label>
      <button id="g3d-reset">reset view</button></div>
    <canvas id="cv-graph3d"></canvas><div class="tip" id="g3d-tip"></div>
    <div class="relbar" id="rel-graph3d"></div>
    <div class="module-flow" id="g3d-modules"></div>
    <div class="graph-audit">
      <section class="audit-panel"><h3>모든 node 값</h3><div class="audit-scroll" id="g3d-nodes"></div></section>
      <section class="audit-panel"><h3>모든 edge · attention-head 값</h3><div class="audit-scroll" id="g3d-edges"></div></section>
      <section class="audit-panel"><h3>모든 MLP 출력 head 값</h3><div class="audit-scroll" id="g3d-heads"></div></section>
    </div>
    <div class="note">Drag to rotate, wheel to zoom, hover a node to isolate its links.
      Depth is distance from the raw semantic channels to SafeLanding; node size and
      colour are the channel's current activation; edge width and opacity are the
      R-GAT message-passing coefficients. They are not reported as relation importance
      or causal evidence. Self-loops remain in the audit table even though the 3D canvas omits them.</div>`;}
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
      const named=card.labels?card.labels[yKeys.indexOf(key)]:key;
      const label=(card.kind==='pairplots')?named
        :(card.labels&&yKeys.length>1)
        // Prefix with the method only when more than one method is drawn;
        // otherwise every legend entry repeats the same long name.
        ?(sources.length>1?`${LABELS[s]||s} ${named}`:named)
        :(yKeys.length>1?named:(LABELS[s]||s));
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
function methodLabel(method){return LABELS['benchmark_train_'+method]||method||'초기화 대기';}
function drawPairPlots(card,state){
  const pairs=(state.scalars||{}).parallel_pair_status||[];
  for(let index=0;index<2;index++){
    const pair=pairs.find(item=>Number(item.index)===index)||pairs[index]||{};
    const assigned=String(pair.assigned_method||pair.method||'');
    const method=String(pair.active_method||assigned);
    const title=document.getElementById(`pair-title-${card.id}-${index}`);
    if(title){
      const assignedLabel=methodLabel(assigned),activeLabel=methodLabel(method);
      title.textContent=`물리 Pair ${index+1} · 현재 ${activeLabel}`+
        (method&&method!==assigned?` · 학습 배정 ${assignedLabel}`:'');}
    let spec;
    if(card.plot==='reward')spec={
      y:['reward','task','active_perception','ontology_fov_reward'],
      labels:['전체','terminal task','Shin active perception','Ontology FOV 추가']};
    else if(card.plot==='perception')spec={
      y:['geometric_in_fov','keypoint_confidence','visible_keypoint_fraction','fov_margin'],
      labels:['기하 FOV(패드 중심)','keypoint 신뢰도','keypoint 가시 비율','FOV margin'],
      ymin:0,ymax:1};
    else spec={
      y:['geometric_in_fov','lateral_progress','vertical_progress','vertical_speed_penalty'],
      labels:['기하 FOV(패드 중심)','수평 progress','수직 progress','수직속도 penalty']};
    draw({...card,...spec,id:`${card.id}-${index}`,
      series:[`benchmark_step_pair_${index}`],x:'step'},state);
  }
}
function meanOf(rows,key,tail){
  const use=tail?rows.slice(-tail):rows;
  const values=use.map(r=>Number(r[key])).filter(Number.isFinite);
  return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
}
function lastOf(rows,key){
  for(let i=rows.length-1;i>=0;i--){const v=Number(rows[i][key]);
    if(Number.isFinite(v))return v;}
  return null;
}
const pctText=v=>v===null?'--':(100*v).toFixed(1)+'%';
function deltaText(value,digits){
  if(value===null)return '--';
  const shown=Math.abs(value)<Math.pow(10,-digits)/2?0:value;
  return (shown>0?'+':shown<0?'−':'')+Math.abs(shown).toFixed(digits);
}
function tiles(state){
  const s=state.scalars||{},out=[];
  const add=(label,value,note,cls)=>out.push(
    `<div class="tile ${cls||''}"><b>${escapeHTML(value)}</b>`+
    `<span>${escapeHTML(label)}</span>`+
    (note?`<small>${escapeHTML(note)}</small>`:'')+`</div>`);
  const base='shin_se_fixed',prop='shin_se_onto_rgat_recovery';
  const evalRows=m=>state.series['benchmark_eval_'+m]||[];
  const trainRows=m=>state.series['benchmark_train_'+m]||[];
  const trained=BENCHMARK_METHODS.reduce((n,m)=>n+trainRows(m).length,0);

  add('단계',state.stage.name);
  add('실험 phase',s.benchmark_phase||'initializing');
  add('학습 checkpoint',`${trained} / ${s.training_total||0}`,
      trained>=Number(s.training_total||0)&&Number(s.training_total||0)>0?'완료':'진행 중');
  add('평가 진행',`${s.evaluation_completed||0} / ${s.evaluation_total||0}`);

  // Head to head. The delta is the whole point of the experiment, so it is a
  // tile of its own rather than something to read off two other tiles.
  const baseSuccess=meanOf(evalRows(base),'paper_success');
  const propSuccess=meanOf(evalRows(prop),'paper_success');
  add('Baseline 성공률',pctText(baseSuccess),`평가 ${evalRows(base).length}회`);
  add('Proposed 성공률',pctText(propSuccess),`평가 ${evalRows(prop).length}회`);
  add('Δ 성공률',
      baseSuccess===null||propSuccess===null?'--'
      :deltaText(100*(propSuccess-baseSuccess),1)+'%p',
      'Proposed − Baseline · 표본이 작으면 해석 금지','delta');

  const baseFov=meanOf(trainRows(base),'geometric_fov_loss_fraction',50);
  const propFov=meanOf(trainRows(prop),'geometric_fov_loss_fraction',50);
  add('Baseline FOV 소실률',pctText(baseFov),'최근 50 ep');
  add('Proposed FOV 소실률',pctText(propFov),'최근 50 ep');
  add('Δ FOV 소실률',
      baseFov===null||propFov===null?'--'
      :deltaText(100*(propFov-baseFov),1)+'%p',
      'Proposed − Baseline · 음수가 개선','delta');

  // What the added reward is doing, and whether it is believable.
  const share=lastOf(trainRows(prop),'ontology_fov_reward_share');
  const mae=lastOf(trainRows(prop),'fov_prediction_mae');
  const predicted=lastOf(trainRows(prop),'fov_predicted_mean');
  const actual=lastOf(trainRows(prop),'fov_actual_mean');
  add('추가 보상 비중',share===null?'--':(100*share).toFixed(1)+'%',
      'step 보상 크기 대비 −λ·q');
  add('readout MAE',mae===null?'--':mae.toFixed(3),
      predicted===null||actual===null?'관측된 미래창만'
      :`예측 ${predicted.toFixed(2)} / 실측 ${actual.toFixed(2)}`);
  // Only the FOV readout's own scalar. reward_design_id can still hold a
  // legacy fixed-reward design restored from disk, which is a different
  // artifact and must not appear under this label.
  if(s.fov_risk_design_id)
    add('FOV readout 설계',String(s.fov_risk_design_id).slice(0,12),'PPO 중 동결');
  if(s.config_hash)add('설정 hash',String(s.config_hash).slice(0,10),'두 arm 공통');
  document.getElementById('tiles').innerHTML=out.join('');
}
const KIND_LABEL={infra:'infra',gate:'검증',train:'학습',data:'데이터',
  eval:'평가',report:'보고'};
function fovRow(label,value,cls){
  return `<div class="fovrow ${cls||''}"><span>${escapeHTML(label)}</span>`
    +`<b>${escapeHTML(value)}</b></div>`;}
function fovStatusPanel(state){
  const box=document.getElementById('fov-status');if(!box)return;
  const s=state.scalars||{};
  const data=s.fov_dataset,train=s.fov_training,model=s.fov_model;
  const num=(v,d)=>Number.isFinite(Number(v))?Number(v).toFixed(d===undefined?3:d):'--';
  const cards=[];

  // 1. collection. The loop exits on target coverage, not on the episode
  // count, so the count alone would misreport how far along it is.
  let cls=data?(data.covered?'done':'run'):'';
  let body='';
  if(!data){body='<div class="fovempty">아직 시작하지 않았습니다.</div>';}
  else{
    const frac=Math.max(0,Math.min(1,Number(data.episodes)/Math.max(1,Number(data.maximum))));
    const minMark=Math.max(0,Math.min(1,Number(data.minimum)/Math.max(1,Number(data.maximum))));
    body=`<div class="fovbar"><i style="width:${(100*frac).toFixed(1)}%"></i>`
      +`<u style="left:${(100*minMark).toFixed(1)}%" title="최소 episode"></u></div>`
      +fovRow('episode',`${data.episodes} / 최소 ${data.minimum} · 상한 ${data.maximum}`)
      +fovRow('두 regime 확보',data.covered?'예':'아직',data.covered?'good':'warn')
      +fovRow('FOV 소실이 있던 episode',String(data.loss_episodes))
      +fovRow('지도 가능 sample',String(data.supervised_samples))
      +fovRow('꼬리 mask',String(data.masked_samples))
      +fovRow('평균 목표 y',num(data.target_mean))
      +fovRow('환경 step',String(data.environment_steps))
      +(data.cached?fovRow('출처','재사용된 캐시'):'');
  }
  cards.push(`<div class="fovbox ${cls}"><h3>1. 데이터 수집<i>`
    +(data?(data.covered?'완료':'진행'):'대기')+`</i></h3>${body}</div>`);

  // 2. offline training
  cls=model?'done':(train?'run':'');
  if(model)body=fovRow('epoch','완료')
      +fovRow('최종 검증 손실',num(train&&train.best_validation_loss,5))
      +fovRow('규약 위반',num(model.contract_violation,5),
        Number(model.contract_violation)>0.01?'warn':'good')
      +fovRow('학습/검증 episode',
        `${model.train_episodes} / ${model.validation_episodes}`)
      +fovRow('목적함수',String(model.loss||'--'));
  else if(train)body=`<div class="fovbar"><i style="width:`
      +`${(100*Math.max(0,Math.min(1,Number(train.epoch)/Math.max(1,Number(train.total_epochs))))).toFixed(1)}%"></i></div>`
      +fovRow('epoch',`${train.epoch} / ${train.total_epochs}`)
      +fovRow('검증 손실',num(train.validation_loss,5))
      +fovRow('최저 검증 손실',num(train.best_validation_loss,5),
        train.improved?'good':'')
      +fovRow('규약 위반',num(train.validation_contract,5));
  else body='<div class="fovempty">데이터 수집이 끝나면 시작합니다.</div>';
  cards.push(`<div class="fovbox ${cls}"><h3>2. readout 학습<i>`
    +(model?'동결됨':(train?'진행':'대기'))+`</i></h3>${body}</div>`);

  // 3. what the frozen readout is worth. The constant predictor sits next to
  // the RMSE because an error alone cannot say the model beat the mean.
  cls=model?'done':'';
  if(!model)body='<div class="fovempty">학습이 끝나면 표시합니다.</div>';
  else{
    const rmse=Number(model.rmse),base=Number(model.constant_predictor_rmse);
    const beats=Number.isFinite(rmse)&&Number.isFinite(base)&&rmse<base;
    body=fovRow('MAE',num(model.mae))
      +fovRow('RMSE',num(rmse))
      +fovRow('상수 예측기 RMSE',num(base))
      +fovRow('상수 예측기보다 나은가',
        Number.isFinite(rmse)&&Number.isFinite(base)?(beats?'예':'아니오'):'--',
        beats?'good':'bad')
      +fovRow('편향',num(model.bias))
      +fovRow('예측 / 실측 평균',
        `${num(model.prediction_mean,3)} / ${num(model.target_mean,3)}`)
      +fovRow('PPO 중 동결',model.frozen?'예':'아니오',model.frozen?'good':'bad')
      +fovRow('설계 id',String(model.design_id||'--').slice(0,16));
  }
  cards.push(`<div class="fovbox ${cls}"><h3>3. 동결 readout 검증<i>`
    +(model?'확정':'대기')+`</i></h3>${body}</div>`);
  box.innerHTML=cards.join('');
}
function runPipelinePanel(state){
  const box=document.getElementById('run-pipeline');if(!box)return;
  const spec=(state.scalars||{}).run_pipeline;
  if(!spec){box.innerHTML='';return;}
  const stages=spec.stages||[],aliases=spec.aliases||{};
  const raw=String(state.stage.name||'');
  const name=aliases[raw]||raw;
  let active=stages.findIndex(s=>s.id===name);
  if(active<0)active=stages.findIndex(s=>name&&(name.startsWith(s.id)||s.id.startsWith(name)));
  // The evaluation phase is authoritative even when a worker thread is
  // reporting its own per-pair stage name.
  if(String((state.scalars||{}).benchmark_phase||'')==='evaluation')
    active=stages.findIndex(s=>s.id==='paired evaluation');
  box.innerHTML=stages.map((stage,index)=>{
    const cls=active<0?'pending':index<active?'done':index===active?'active':'pending';
    return `<div class="runstage ${cls}">`
      +`<i class="kind">${escapeHTML(KIND_LABEL[stage.kind]||stage.kind)}</i>`
      +`<b>${index+1}. ${escapeHTML(stage.title)}`
      +(stage.optional?' <small style="color:#8a8a8a">(선택)</small>':'')+`</b>`
      +`<span>${escapeHTML(stage.detail)}</span>`
      +`<em>→ ${escapeHTML(stage.produces)}</em></div>`;}).join('');
  const note=document.getElementById('run-pipeline-note');
  if(note)note.textContent=spec.note||'';
}
function algoPipelinePanel(state){
  const box=document.getElementById('algo-pipeline');if(!box)return;
  const spec=(state.scalars||{}).algorithm_pipeline;
  if(!spec){box.innerHTML='<span style="color:#666">제안 파이프라인이 구성되면 표시됩니다.</span>';
    const empty=document.getElementById('algo-limits');if(empty)empty.innerHTML='';return;}
  const chain=spec.chain||[];
  const W=176,GAP=28,H=118,X0=14,Y=54;
  const width=X0*2+chain.length*W+(chain.length-1)*GAP;
  const offY=Y+H+74,OFFH=84,total=offY+OFFH+42;
  const parts=[];
  parts.push(`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
    markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5a5a5a"/></marker>
    <marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5"
    markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#D95319"/></marker></defs>`);
  const kept=spec.deployed_stages??2;
  parts.push(`<rect x="${X0}" y="12" width="11" height="11" fill="#0072BD"/>`);
  parts.push(`<text x="${X0+17}" y="22" font-size="11" fill="#262626">`
    +`배포에 남음 (앞 ${kept}단계)</text>`);
  parts.push(`<rect x="${X0+178}" y="12" width="11" height="11" fill="#D95319"/>`);
  parts.push(`<text x="${X0+195}" y="22" font-size="11" fill="#262626">`
    +`학습 전용 · 배포 시 제거</text>`);
  chain.forEach((node,index)=>{
    const x=X0+index*(W+GAP);
    const online=index<(spec.deployed_stages??2);
    const accent=online?'#0072BD':'#D95319';
    const faded=online?'#f3f8fc':'#fff7f0';
    parts.push(`<rect x="${x}" y="${Y}" width="${W}" height="${H}" rx="2"
      fill="${faded}" stroke="${accent}" stroke-width="1.2"/>`);
    parts.push(`<rect x="${x}" y="${Y}" width="${W}" height="4" fill="${accent}"/>`);
    parts.push(`<text x="${x+11}" y="${Y+24}" font-size="12.5" font-weight="700"
      fill="#262626">${escapeHTML(node.title)}</text>`);
    (node.lines||[]).forEach((line,li)=>parts.push(
      `<text x="${x+11}" y="${Y+43+li*16}" font-size="10.5" fill="#4d4d4d">`
      +`${escapeHTML(line)}</text>`));
    if(index<chain.length-1)parts.push(`<line x1="${x+W}" y1="${Y+H/2}"
      x2="${x+W+GAP-4}" y2="${Y+H/2}" stroke="#5a5a5a" stroke-width="1.4"
      marker-end="url(#ah)"/>`);
  });
  const offline=spec.offline||{};
  const target=Math.max(0,chain.findIndex(n=>n.id===(offline.into||'rgat')));
  const ox=X0,ow=X0+target*(W+GAP)+W-X0;
  parts.push(`<rect x="${ox}" y="${offY}" width="${ow}" height="${OFFH}" rx="2"
    fill="#fdf6ef" stroke="#D95319" stroke-width="1.2" stroke-dasharray="6 4"/>`);
  parts.push(`<text x="${ox+12}" y="${offY+22}" font-size="12" font-weight="700"
    fill="#8a3b12">${escapeHTML(offline.title||'오프라인 지도')}</text>`);
  (offline.lines||[]).forEach((line,li)=>parts.push(
    `<text x="${ox+12}" y="${offY+42+li*16}" font-size="10.5" fill="#6b4226">`
    +`${escapeHTML(line)}</text>`));
  const jx=X0+target*(W+GAP)+W/2;
  parts.push(`<line x1="${jx}" y1="${offY}" x2="${jx}" y2="${Y+H+6}"
    stroke="#D95319" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#ahd)"/>`);
  parts.push(`<text x="${jx+8}" y="${offY-8}" font-size="10" fill="#8a3b12">`
    +`학습 시에만 · 추론에는 들어가지 않음</text>`);
  const dep=spec.deployment||{};
  const depY=offY+OFFH+28;
  parts.push(`<text x="${X0}" y="${depY}" font-size="11" fill="#262626">`
    +`<tspan font-weight="700" fill="#77AC30">배포 유지</tspan>`
    +`<tspan dx="8">${escapeHTML((dep.keeps||[]).join(' · '))}</tspan></text>`);
  parts.push(`<text x="${X0+430}" y="${depY}" font-size="11" fill="#262626">`
    +`<tspan font-weight="700" fill="#A2142F">배포 제거</tspan>`
    +`<tspan dx="8">${escapeHTML((dep.drops||[]).join(' · '))}</tspan></text>`);
  box.innerHTML=`<svg viewBox="0 0 ${width} ${total}" role="img"
    aria-label="제안 알고리즘 파이프라인">${parts.join('')}</svg>`;
  const limits=document.getElementById('algo-limits');
  if(limits)limits.innerHTML=(spec.limits||[]).map(text=>
    `<span><b>주의</b>${escapeHTML(text)}</span>`).join('');
}
function mdpPanel(state){
  const box=document.getElementById('mdp-grid');if(!box)return;
  const spec=(state.scalars||{}).mdp_contract;
  if(!spec){box.innerHTML='';const s0=document.getElementById('mdp-summary');
    if(s0)s0.textContent='';return;}
  const arms=spec.arms||[],rows=spec.rows||[];
  const out=[`<div class="hd"></div>`];
  arms.forEach((arm,index)=>out.push(
    `<div class="hd a${index}">${escapeHTML(arm.label)}`
    +`<span class="nt">${escapeHTML(arm.role||'')}</span></div>`));
  rows.forEach(row=>{
    out.push(`<div class="rk">${escapeHTML(row.label)}</div>`);
    if(row.identical){
      // One cell spanning both arms: the claim is that there is nothing to
      // compare here, and a split cell would invite comparing it anyway.
      out.push(`<div class="same"><span class="tag">동일</span>`
        +`${escapeHTML(row.value)}`
        +(row.note?`<span class="nt">${escapeHTML(row.note)}</span>`:'')+`</div>`);
    }else{
      out.push(`<div class="diff base">${escapeHTML(row.value)}`
        +(row.note?`<span class="nt">${escapeHTML(row.note)}</span>`:'')+`</div>`);
      out.push(`<div class="diff"><span class="tag">차이</span>`
        +`${escapeHTML(row.value)}`
        +`<span class="nt">+ ${escapeHTML(row.delta||'')}</span></div>`);
    }
  });
  box.innerHTML=out.join('');
  const summary=document.getElementById('mdp-summary');
  if(summary)summary.innerHTML=
    `${spec.total_count}개 구조 항목 중 <b>${spec.total_count-spec.identical_count}개</b>만 다르다.`
    +' 나머지는 두 arm에서 같은 코드 경로를 쓴다.';
}
function phasePanel(state){
  const s=state.scalars||{},phase=String(s.benchmark_phase||'initializing');
  const box=document.getElementById('phase-status');if(!box)return;
  box.className='phase-status '+phase;
  if(phase==='evaluation')box.innerHTML='<b>학습 완료 · 현재 crossover paired evaluation 갱신 중</b>'+
    '<span>Crossover 평가에서는 정책이 물리 pair를 seed마다 순환합니다. 착륙 결과는 '+
    '“현재 정책”으로 표시된 방법에 귀속되며 “학습 배정” pair에 귀속되지 않습니다. '+
    '현재 변화는 pair별 live plot, 방법별 평가 진행 수와 “[현재 평가]” 그래프에서 확인하십시오.</span>';
  else if(phase==='training')box.innerHTML='<b>PPO 학습 진행 중</b>'+
    '<span>episode가 종료되고 optimizer와 checkpoint 기록이 완료될 때 학습 그래프가 증가합니다.</span>';
  else if(phase.includes('reward-design')||String(state.stage.name||'').includes('reward')
      ||String(state.stage.name||'').includes('FOV'))
    box.innerHTML='<b>FOV-risk R-GAT 데이터 수집/학습 중</b>'+
      '<span>제안 모델 전용 pair은 동결된 Shin baseline 정책으로 동일 simulator-domain 시각 전이를 수집합니다. baseline PPO는 다른 독립 pair에서 동시에 학습합니다.</span>';
  else box.innerHTML='<b>실험 초기화 중</b><span>pair 연결 및 artifact 준비 상태를 확인하고 있습니다.</span>';
}
function escapeHTML(value){return String(value??'--').replace(/[&<>"']/g,c=>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function pairPanel(state){
  const box=document.getElementById('parallel-pair-grid');if(!box)return;
  const s=state.scalars||{},pairs=s.parallel_pair_status||s.parallel_pair_layout||[];
  const pairCount=Math.max(1,Number(s.parallel_pair_count||pairs.length||2));
  box.innerHTML=Array.from({length:pairCount},(_,index)=>{
    const pair=pairs.find(item=>Number(item.index)===index)||pairs[index]||{index:index};
    const assigned=pair.assigned_method||pair.method||'--';
    const method=pair.active_method||assigned;
    const assignedLabel=methodLabel(assigned),activeLabel=methodLabel(method);
    const methodIndex=(s.benchmark_methods||[]).indexOf(method);
    const activeColor=PALETTE[(methodIndex>=0?methodIndex:index)%PALETTE.length];
    const rows=state.series['benchmark_step_pair_'+index]||[];
    const live=rows.length?rows[rows.length-1]:{},xyz=pair.relative_xyz||null;
    const step=pair.step??live.step??0,status=String(pair.status||live.status||'waiting');
    const geometricFov=pair.geometric_pad_center_in_fov;
    // Geometric frustum truth and neural perception quality are different
    // variables and are shown as such: a pad in frame that the encoder cannot
    // resolve is perception degradation, not an FOV loss.
    const fovLabel=geometricFov===null||geometricFov===undefined
      ?'대기':(geometricFov?'프레임 내':'프레임 이탈');
    const keypointConfidence=Number(
      pair.keypoint_confidence??live.keypoint_confidence);
    const visibleKeypoints=Number(
      pair.visible_keypoint_fraction??live.visible_keypoint_fraction);
    const pct=value=>Number.isFinite(value)?(100*value).toFixed(0)+'%':'--';
    const reserve=Number(pair.battery_reserve??live.battery_reserve);
    const activeReward=Number(pair.active_perception??live.active_perception);
    const fovMargin=Number(pair.fov_margin??live.fov_margin);
    const fovRisk=Number(pair.predicted_fov_unavailability??
      live.predicted_fov_unavailability);
    const ontoReward=Number(pair.ontology_fov_reward??live.ontology_fov_reward);
    const proposed=method==='shin_se_onto_rgat_recovery';
    const gate=pair.landing_gate||null;
    const pos=xyz&&xyz.length===3
      ?`${Number(xyz[0]).toFixed(2)}, ${Number(xyz[1]).toFixed(2)}, ${Number(xyz[2]).toFixed(2)}`:'--';
    return `<div class="pair-card" style="border-top-color:${activeColor}">`
      +`<div class="pair-head"><b>물리 Pair ${index+1} · 현재 정책: ${escapeHTML(activeLabel)}</b>`
      +`<span class="pair-state ${escapeHTML(status)}">${escapeHTML(status.toUpperCase())}</span></div>`
      +`<div class="pair-assignment ${method!==assigned?'rotated':''}">`
      +`학습 배정: ${escapeHTML(assignedLabel)}`
      +(String(pair.phase)==='evaluation'?' · crossover 평가로 정책 순환 중':'')+`</div>`
      +`<div class="pair-metrics"><div><b>${escapeHTML(pair.episode??0)} / ${escapeHTML(step)}</b><small>${escapeHTML(pair.episode_kind||'episode')} / 스텝</small></div>`
      +`<div><b>${escapeHTML(fovLabel)}</b><small>기하 FOV (패드 중심)</small></div>`
      +`<div><b>${escapeHTML(pct(keypointConfidence))} / ${escapeHTML(pct(visibleKeypoints))}</b>`
      +`<small>keypoint 신뢰도 / 가시 비율</small></div>`
      +`<div><b>${Number(pair.ugv_speed_m_s??live.ugv_speed_m_s??0).toFixed(2)} m/s</b><small>UGV 속도</small></div>`
      +`<div><b>${Number(pair.uav_speed_m_s??live.uav_speed_m_s??0).toFixed(2)} m/s</b><small>UAV 속도</small></div>`
      +`<div><b>${Number.isFinite(reserve)?(100*reserve).toFixed(1)+'%':'--'}</b><small>배터리 잔량</small></div>`
      +`<div><b>${escapeHTML(pos)}</b><small>패드 상대 XYZ (m)</small></div></div>`
      +`<div class="pair-rewards"><div class="pair-reward"><b>${Number.isFinite(activeReward)?activeReward.toFixed(3):'--'}</b>`
      +`<small>공통 Shin active-perception reward</small></div>`
      +(proposed
        ?`<div class="pair-reward added"><b>margin ${Number.isFinite(fovMargin)?fovMargin.toFixed(2):'--'} · q(1s 비가용 비율) ${Number.isFinite(fovRisk)?fovRisk.toFixed(2):'--'} · r ${Number.isFinite(ontoReward)?ontoReward.toFixed(3):'--'}</b>`
          +`<small>제안 모델에만 추가된 Ontology-R-GAT FOV-risk reward</small></div>`
        :`<div class="pair-reward added off"><b>OFF</b><small>Ontology-R-GAT FOV-risk 추가 분기</small></div>`)
      +`</div>`
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
  // Charts one metric per deck-motion scenario. Training rotates through the
  // same scenario list the evaluation uses, so the same grouping answers both
  // "which decks does the policy land on" and "which decks is it learning on".
  const names=card.series||BENCHMARK_EVAL;
  const seriesFor=method=>names.find(name=>name.endsWith(method));
  const metric=card.metric||'paper_success';
  const cv=document.getElementById('cv-'+card.id),lg=document.getElementById('lg-'+card.id);
  if(!cv)return;
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const methods=(state.scalars.benchmark_methods||[]).filter(method=>
    (state.series[seriesFor(method)]||[]).length);
  const preferred=['training_random_walk','straight_8mps','linear_acceleration_wave',
    'circle','zigzag','u_turn','vertical_heave_boat'];
  const present=new Set();
  for(const method of methods)for(const row of state.series[seriesFor(method)]||[])
    present.add(row.scenario);
  const scenarios=preferred.filter(x=>present.has(x));
  for(const value of present)if(!scenarios.includes(value))scenarios.push(value);
  if(!methods.length||!scenarios.length){g.fillStyle='#666';g.font='12px Arial';
    g.fillText(card.empty||'no paired evaluation data yet',12,h/2);
    lg.innerHTML='';return;}
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
      const rows=(state.series[seriesFor(method)]||[]).filter(r=>r.scenario===scenario);
      if(!rows.length)return;
      const mean=rows.reduce((sum,row)=>sum+Number(row[metric]||0),0)/rows.length;
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
      `${escapeHTML(LABELS[seriesFor(method)]||method)}</span>`;}).join('');
}
// ------------------------------------------------------------ 3D ontology
// Hand-rolled: the page has to work with no network, so there is no three.js
// to reach for. The compact fixed graph is well inside what a painter's
// algorithm on a 2D canvas can do at 60 Hz, and doing it by hand is what lets
// the depth cue, the attention width and the hover isolation share one pass.
const REL_COLORS=['#c0392b','#0d8c4d','#1a59bf','#8a8f98'];
const G={yaw:-0.55,pitch:0.30,zoom:1,auto:true,floor:0,hover:-1,drag:null,
         data:null,graphs:{},selected:'',manual:false,dirty:true,pts:[],fit:0};
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
    const text=`${n.name} ${n.value.toFixed(2)}`;
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
function graphNumber(value,digits=6){
  if(value===null||value===undefined||value==='')return '—';
  const number=Number(value);return Number.isFinite(number)?number.toFixed(digits):'—';
}
function graphTable(headers,rows){
  if(!rows.length)return '<div class="audit-empty">값이 아직 발행되지 않았습니다.</div>';
  return '<table class="audit-table"><thead><tr>'+headers.map(value=>
    `<th>${escapeHTML(value)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>
    '<tr>'+row.map(value=>`<td>${escapeHTML(String(value))}</td>`).join('')+
    '</tr>').join('')+'</tbody></table>';
}
function graphAudit(){
  const data=G.data||{},nodes=data.nodes||[],edges=data.edges||[],model=data.model||{};
  const nodeRows=nodes.map((node,index)=>[index,`${node.name} (${node.role})`,
    graphNumber(node.value),graphNumber(node.embedding_mean),graphNumber(node.embedding_l2)]);
  document.getElementById('g3d-nodes').innerHTML=graphTable(
    ['#','node','input','embed mean','embed L2'],nodeRows);

  const maxHeads=edges.reduce((maximum,edge)=>Math.max(maximum,(edge.heads||[]).length),0);
  const edgeRows=edges.map((edge,index)=>{
    const source=nodes[edge.s]?nodes[edge.s].name:String(edge.s);
    const target=nodes[edge.d]?nodes[edge.d].name:String(edge.d);
    const relation=(data.relations||[])[edge.r];
    const values=[index,`${source} → ${target}`,relation?relation.name:String(edge.r),
      graphNumber(edge.a)];
    for(let head=0;head<maxHeads;head++)values.push(graphNumber((edge.heads||[])[head]));
    return values;
  });
  document.getElementById('g3d-edges').innerHTML=graphTable(
    ['#','edge','relation','mean α',...Array.from({length:maxHeads},(_,i)=>`H${i+1} α`)],edgeRows);

  const outputHeads=model.output_heads||[],headRows=[];
  for(const head of outputHeads)for(const output of head.outputs||[]){
    const pre=output.logit!==undefined?output.logit:output.pre_activation;
    headRows.push([String(head.name||'head'),String(output.name||'output'),
      graphNumber(pre),graphNumber(output.value),graphNumber(output.hidden_mean),
      graphNumber(output.hidden_l2)]);
  }
  document.getElementById('g3d-heads').innerHTML=graphTable(
    ['MLP head','output','logit/raw','value','hidden mean','hidden L2'],headRows);

  const flow=document.getElementById('g3d-modules'),encoder=model.encoder||{};
  const boxes=[{name:'Ontology input',detail:`${nodes.length} nodes · ${edges.length} edges`}];
  for(const layer of encoder.layers||[])boxes.push({name:layer.name||'encoder layer',
    detail:`${layer.input_dim??'—'} → ${layer.output_dim??'—'} · ${layer.attention_heads??0} attention heads`});
  for(const head of outputHeads)boxes.push({name:head.name||'output head',head:true,
    detail:`${head.kind||'MLP'} · ${(head.outputs||[]).length} outputs`});
  flow.innerHTML=boxes.map((box,index)=>(index?'<span class="module-arrow">→</span>':'')+
    `<div class="module-box${box.head?' head':''}"><b>${escapeHTML(box.name)}</b>`+
    `<small>${escapeHTML(box.detail)}</small></div>`).join('');
}
function graphSnapshots(state){
  const incoming={...(state.graphs||{})};
  if(!Object.keys(incoming).length&&state.graph)incoming.legacy=state.graph;
  G.graphs=incoming;
  const keys=Object.keys(incoming).sort((left,right)=>{
    const a=incoming[left],b=incoming[right];
    return Number(a.pair_index??99)-Number(b.pair_index??99)||left.localeCompare(right);
  });
  const modeled=keys.find(key=>incoming[key]&&incoming[key].model);
  if(!keys.includes(G.selected)||(modeled&&!G.manual&&!(incoming[G.selected]||{}).model))
    G.selected=modeled||keys[0]||'';
  const select=document.getElementById('g3d-select');
  select.innerHTML=keys.map(key=>{const graph=incoming[key]||{};
    const pair=graph.pair_index===undefined?'':`Pair ${Number(graph.pair_index)+1} · `;
    const suffix=graph.model?` · ${graph.model.architecture||'R-GAT'}`:' · schema only';
    return `<option value="${escapeHTML(key)}"${key===G.selected?' selected':''}>`+
      `${escapeHTML(pair+(graph.method||key)+suffix)}</option>`;}).join('');
  select.disabled=keys.length<2;
  G.data=incoming[G.selected]||null;
  const graph=G.data;
  document.getElementById('g3d-src').textContent=graph
    ?`${graph.source||G.selected}${graph.attention?'':' · attention 대기'}`+
      `${graph.phi!==undefined?` · Φ=${Number(graph.phi).toFixed(3)}`:''}`
    :'ontology graph 초기화 대기';
  G.dirty=true;graphLegend();graphAudit();
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
       +` ${data.nodes[best.d].name} (${rel}, &alpha;=${best.a.toFixed(3)}`
       +`${best.heads&&best.heads.length?', heads=['+best.heads.map(value=>Number(value).toFixed(3)).join(', ')+']':''})`:'');
  const card=document.getElementById('card-graph3d').getBoundingClientRect();
  tip.style.display='block';
  tip.style.left=Math.min(ev.clientX-card.left+12,card.width-tip.offsetWidth-8)+'px';
  tip.style.top=(ev.clientY-card.top+12)+'px';
}
function graphBind(){
  const cv=document.getElementById('cv-graph3d');if(!cv)return;
  document.getElementById('g3d-select').addEventListener('change',event=>{
    G.selected=event.target.value;G.manual=true;G.data=G.graphs[G.selected]||null;
    G.hover=-1;G.fit=0;G.dirty=true;graphLegend();graphAudit();
    const graph=G.data;
    document.getElementById('g3d-src').textContent=graph
      ?`${graph.source||G.selected}${graph.attention?'':' · attention 대기'}`+
        `${graph.phi!==undefined?` · Φ=${Number(graph.phi).toFixed(3)}`:''}`
      :'ontology graph 초기화 대기';
  });
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
    '2쌍 Shin baseline / Ontology-R-GAT FOV 비교';
  const hasGraph=Boolean(state.graph)||Object.keys(state.graphs||{}).length>0;
  for(const c of CARDS){
    const el=document.getElementById('card-'+c.id);
    el.hidden=Boolean(c.view&&c.view!=='common'&&c.view!==profile)
      ||(c.kind==='graph'&&!hasGraph);
  }
  if(profile==='benchmark'){
    const pipeline=state.scalars.current_pipeline||state.scalars.current_method||'';
    const active=Number(state.scalars.parallel_pair_count||1)>1
      ?(state.scalars.benchmark_methods||[]):[pipeline];
    const estimator=active.some(x=>x.startsWith('shin_se')||x==='shin2026');
    for(const id of ['benchmark_position_rmse','benchmark_active_saturation',
                     'benchmark_eval_position','benchmark_eval_velocity',
                     'benchmark_eval_visual_loss']){
      const el=document.getElementById('card-'+id);if(el)el.hidden=!estimator;
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
      runPipelinePanel(state);
      fovStatusPanel(state);
      algoPipelinePanel(state);
      mdpPanel(state);
      if(profile==='benchmark')pairPanel(state);
      for(const c of CARDS){
        if(c.id==='tiles'||c.kind==='graph'||c.kind==='contract'||c.kind==='pairs'||
           c.kind==='phase'||c.kind==='heading'||c.kind==='runpipe'||
           c.kind==='fovstatus'||
           c.kind==='algopipe'||c.kind==='mdp'||
           c.kind==='pairplots'||
           (c.view&&c.view!=='common'&&c.view!==profile))continue;
        c.kind==='evalbars'?drawEvaluationBars(c,state):draw(c,state);
      }
      for(const c of CARDS.filter(item=>item.kind==='pairplots'))drawPairPlots(c,state);
      benchmarkPanel(state);
      graphSnapshots(state);
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
