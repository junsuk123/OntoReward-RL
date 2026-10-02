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
/* teacher demonstrations: progress on the left, the live flight on the right */
.teacher{display:grid;grid-template-columns:minmax(250px,1fr) 1.7fr;gap:10px}
.teacher .tprog{border:1px solid #b8b8b8;background:#fafafa;padding:8px 10px;min-width:0}
/* one live tile per physical pair flying the warm start */
.teacher .tgrid{display:grid;gap:8px;min-width:0;
grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.teacher .tgrid .tprog{padding:6px 8px}
.teacher .tgrid h4{margin:0 0 3px;font-size:11px;font-weight:700}
.teacher .tgrid h4 span{color:var(--muted);font-weight:400}
.teacher .tbar{height:10px;background:#e3e3e3;border:1px solid #b8b8b8;margin:6px 0}
.teacher .tbar i{display:block;height:100%;background:var(--good)}
.teacher table{width:100%;border-collapse:collapse;font-size:10.5px;margin-top:6px}
.teacher th,.teacher td{border-bottom:1px solid #e0e0e0;padding:2px 4px;text-align:right;
font-variant-numeric:tabular-nums}.teacher th:first-child,.teacher td:first-child{text-align:left}
.teacher .tlive{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:6px}
.teacher .tlive div{background:#fff;border:1px solid #d0d0d0;padding:4px 5px}
.teacher .tlive b{display:block;font-size:13px;font-variant-numeric:tabular-nums}
.teacher .tlive small{display:block;font-size:9px;color:var(--muted)}
.teacher .tmode{font-weight:700;color:var(--accent);margin-top:4px}
.teacher .tmode.lost{color:var(--warn)}.teacher .tmode.reacquired{color:var(--good)}
.teacher .tnote{color:var(--muted);font-size:10.5px;margin-top:4px}
/* trajectories: four full-width elevation plots, one stable physical pair per row */
.traj-grid{display:grid;grid-template-columns:1fr;gap:8px}
.traj-plot{border:1px solid #b8b8b8;background:#fff;padding:7px;min-width:0}
.traj-plot h3{margin:0 0 3px;font-size:11px;line-height:1.3;text-align:center;height:30px;
overflow:hidden}
.traj-plot canvas{height:260px;aspect-ratio:auto}.traj-plot .legend{justify-content:center}
.collection{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch}
.collection .cstage{border:1px solid #b8b8b8;background:#fff;padding:8px 11px;min-width:150px}
.collection .cstage b{display:block;font-size:13px}
.collection .cstage small{color:#6a6a6a}
.collection .cstage.active{border-color:#0072BD;border-left-width:4px}
.collection .cstage.done{border-color:#77AC30;border-left-width:4px}
.collection .cnote{flex:1 1 240px;color:#4a4a4a;font-size:11px;line-height:1.5}
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
.rgat-flow{position:relative;background:#101b35;border-color:#25365e;color:#e9f2ff;
overflow:hidden}.rgat-flow h2{color:#e9f2ff}.rgat-flow canvas{height:520px;background:#101b35;
cursor:grab;touch-action:none}.rgat-flow canvas.drag{cursor:grabbing}
.rf-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:7px;
font-size:10.5px;color:#a8badc}.rf-head b{color:#7de4ff;font-size:11.5px}.rf-pill{padding:2px 7px;
border:1px solid #3b568d;border-radius:999px;background:rgba(26,46,86,.75)}
.rf-pill.live{border-color:#24b8a8;color:#79f3d9}.rf-pill.wait{border-color:#b68339;color:#ffd182}
.rf-stats{display:grid;grid-template-columns:repeat(6,minmax(90px,1fr));gap:6px;margin-top:8px}
.rf-stat{border:1px solid #304874;background:rgba(16,28,55,.82);padding:5px 7px;min-width:0}
.rf-stat b{display:block;color:#f1f6ff;font-size:13px;font-variant-numeric:tabular-nums}
.rf-stat span{display:block;color:#91a8cf;font-size:8.5px;text-transform:uppercase;letter-spacing:.05em}
.rgat-flow .note{color:#9eb0d0}.rf-legend{display:flex;gap:13px;flex-wrap:wrap;margin-top:7px;
color:#9eb0d0;font-size:9.5px}.rf-legend i{display:inline-block;width:17px;height:2px;
vertical-align:middle;margin-right:4px}.rf-tip{color:#e9f2ff;background:rgba(9,19,40,.96);
border-color:#3e5f9d;box-shadow:0 5px 18px rgba(0,0,0,.45)}
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
.sensor-lineage{margin-top:10px;border:1px solid var(--line);background:#f8fafc;padding:8px 9px}
.sensor-lineage-head{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;margin-bottom:6px}
.sensor-lineage-head h3{font-size:12px;margin:0;color:var(--ink)}
.sensor-lineage-head span{font-size:10px;color:var(--muted)}
.sensor-lineage-head .boundary{margin-left:auto;color:#236c35;border:1px solid #77AC30;
background:#f4f9ee;padding:2px 6px;border-radius:9px}
.sensor-map{width:100%;overflow-x:auto;background:#fff;border:1px solid #d9dfe7}
.sensor-map svg{display:block;width:100%;min-width:980px;height:auto}
.sensor-map .prov-edge{fill:none;stroke:#8ba0b8;stroke-width:1.1;opacity:.30;
transition:opacity .12s,stroke-width .12s}.sensor-map .prov-edge.ontology{stroke:#D95319}
.sensor-map .prov-card{fill:#fff;stroke:#9faebf;stroke-width:1}
.sensor-map .prov-card.sensor{fill:#eef6fb;stroke:#5f98bc}
.sensor-map .prov-card.context{fill:#f8f4fc;stroke:#8a65a5}
.sensor-map .prov-card.ontology{fill:#fff7f0;stroke:#cf8755}
.sensor-map .prov-title{font:600 11px Arial,sans-serif;fill:#26384c}
.sensor-map .prov-value{font:10px "Courier New",monospace;fill:#365b72}
.sensor-map .prov-meta{font:8.5px Arial,sans-serif;fill:#718096}
.sensor-map .prov-column{font:700 10px Arial,sans-serif;letter-spacing:.08em;fill:#59697c}
.sensor-map .dim{opacity:.34}.sensor-map .focus{opacity:1!important;stroke-width:2.5!important}
.sensor-map .prov-node.focus .prov-card{stroke:#0072BD;stroke-width:2.2}
.sensor-lineage-legend{display:flex;gap:13px;flex-wrap:wrap;margin-top:6px;
font-size:9.5px;color:var(--muted)}.sensor-lineage-legend b{color:var(--ink)}
.note{font-size:11px;color:var(--muted);margin-top:8px}
@media(max-width:780px){main{grid-template-columns:1fr}.pair-grid,.pair-plot-grid{
grid-template-columns:1fr}.graph-audit{grid-template-columns:1fr}.rf-stats{
grid-template-columns:repeat(2,minmax(90px,1fr))}}
</style></head><body>
<header><h1 id="page-title">2쌍 Shin baseline / Ontology-R-GAT FOV 비교</h1>
<span id="stage">connecting</span><span id="detail"></span><span id="age"></span></header>
<main id="root"></main>
<script>
// MATLAB default color order (R2025a), shared with the PNG exporters.
const PALETTE=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'];
// The arms are whatever the run configured, not a fixed pair: the 2026-09-22
// comparison puts a non-learned visual servo beside the two PPO arms. These
// four lists are the ones every card holds a reference to, so syncArms()
// rewrites them in place when the run says what it is comparing.
// Frozen fallback for a state that arrives before the run has configured its
// arms. Reading the mutable list below instead would let a stale arm set
// resurrect itself -- a non-learned arm would come back as learned.
const DEFAULT_METHODS=Object.freeze(
  ['shin_se_fixed','shin_se_onto_rgat_state']);
const BENCHMARK_METHODS=DEFAULT_METHODS.slice();
const BENCHMARK_TRAIN=BENCHMARK_METHODS.map(x=>'benchmark_train_'+x);
const BENCHMARK_EVAL=BENCHMARK_METHODS.map(x=>'benchmark_eval_'+x);
const BENCHMARK_STEP=BENCHMARK_METHODS.map(x=>'benchmark_step_'+x);
let ARMS=BENCHMARK_METHODS.map(m=>({method:m,label:m,learned:true}));
function armsOf(state){
  const declared=(state.scalars||{}).benchmark_arms;
  if(Array.isArray(declared)&&declared.length)
    return declared.map(a=>({method:String(a.method),
      label:String(a.label||a.method),learned:a.learned!==false,
      ontology:a.ontology===true,
      // How the ontology takes part: 'state_representation' (current) or
      // 'additive_reward_term' (retired). Published by the run from its
      // pipeline spec so the panels never have to guess from an arm's name.
      ontology_role:a.ontology_role||null,
      graph_state_representation:a.graph_state_representation||null}));
  return ((state.scalars||{}).benchmark_methods||DEFAULT_METHODS)
    .map(m=>({method:String(m),label:String(m),learned:true,
      ontology_role:m.endsWith('_state')?'state_representation':null}));
}
function refill(list,next){list.length=0;for(const v of next)list.push(v);}
function syncArms(state){
  ARMS=armsOf(state);
  const learned=ARMS.filter(a=>a.learned).map(a=>a.method);
  refill(BENCHMARK_METHODS,ARMS.map(a=>a.method));
  // A non-learned arm has no training curve by construction, so it is absent
  // here rather than drawn as an empty series.
  refill(BENCHMARK_TRAIN,learned.map(m=>'benchmark_train_'+m));
  refill(BENCHMARK_EVAL,ARMS.map(a=>'benchmark_eval_'+a.method));
  refill(BENCHMARK_STEP,ARMS.map(a=>'benchmark_step_'+a.method));
  const ontology=ontologyArms().map(a=>a.method);
  if(ontology.length){
    refill(PROPOSED_TRAIN,ontology.map(m=>'benchmark_train_'+m));
    refill(PROPOSED_STEP,ontology.map(m=>'benchmark_step_'+m));
  }
}
function armLabel(method){
  const arm=ARMS.find(a=>a.method===String(method));
  return arm?arm.label:null;
}
// Whether this arm's reward carries the frozen R-GAT term. Published by the
// run from its pipeline spec, so a differently named ontology arm is labelled
// correctly instead of silently falling through to the baseline rendering.
function isOntologyArm(method){
  const arm=ARMS.find(a=>a.method===String(method));
  return !!(arm&&arm.ontology);
}
function seriesLabel(key){
  const m=/^benchmark_(train|eval)_(.+)$/.exec(String(key));
  return (m&&armLabel(m[2]))||LABELS[key]||key;
}
function learnedArms(){return ARMS.filter(a=>a.learned);}
function ontologyArms(){return ARMS.filter(a=>a.ontology&&a.learned);}
// HOW the ontology takes part, as the run's own pipeline spec reports it.
// 'state_representation' is the current method; 'additive_reward_term' is the
// retired one. Cards that only make sense for one of them declare the role
// they need and are hidden for the other, so a panel never describes a
// mechanism this run does not have.
function armRoles(){
  return new Set(ARMS.map(a=>a.ontology_role).filter(Boolean));
}
function hasRole(role){return !role||armRoles().has(role);}
// Series for the arm whose reward carries the frozen R-GAT term. Written as a
// literal id until 2026-09-22, which meant section 2 went blank for every
// ontology pipeline this repository declares except one -- and would have to
// be edited again for the next. Refilled from the run in ``syncArms``.
const PROPOSED_TRAIN=['benchmark_train_shin_se_onto_rgat_state'];
const PROPOSED_STEP=['benchmark_step_shin_se_onto_rgat_state'];
const CARDS=[
 {id:'tiles',title:null},
 {id:'phase_status',view:'benchmark',kind:'phase'},
 {id:'run_pipeline',view:'benchmark',kind:'runpipe',
  title:'시뮬레이션 · 학습 · 검증 파이프라인 (현재 위치 표시)'},
 {id:'collection_stage',view:'benchmark',kind:'collection',
  title:'수집 단계 · 데이터셋과 사용한 물리 pair (--stage)'},

 {id:'head_performance',view:'benchmark',kind:'heading',
  title:'1 · 두 모델 성능',
  note:'인식·상태추정·actor·critic·action·제어기·환경·보상·종료가 동일하고, 같은 '
      +'시연 집합과 같은 PPO 예산·seed를 쓴다. 두 곡선의 차이는 관측에 온톨로지 '
      +'상황 그래프가 들어가는지 하나뿐이다. 제어는 세 arm 모두 평면 3채널 '
      +'[a_fwd, a_z, tilt]이며 횡방향 속도와 요레이트는 항상 0이다.'},
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
  note:'온톨로지 그래프 G_t는 정책의 상태 표현이다. 9개 node · 4개 관계로 상황을 '
      +'요약해 R-GAT으로 부호화하고, 그래프 수준 읽기 g_t를 actor와 critic 입력에 '
      +'이어붙인다. 보상에는 전혀 들어가지 않으므로 두 arm의 보상은 완전히 같다. '
      +'아래는 그래프가 무엇을 보고 있는지와, 부호기가 실제로 동작하는지를 나눠 본다. '
      +'그래프 채널 자체는 두 arm 모두에 대해 계산·표시되며, 소비하는 쪽은 제안 arm뿐이다.'},
 {id:'fov_status',view:'benchmark',kind:'fovstatus',
  requiresRole:'additive_reward_term',
  title:'제안 arm 준비 상태 · FOV 데이터 수집 → readout 학습 → 동결'},
 {id:'fov_offline_training',view:'benchmark',
  requiresRole:'additive_reward_term',
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
 {id:'benchmark_fov_calibration',view:'benchmark',requiresRole:'additive_reward_term',
  title:'readout 보정 · 예측 대 실측 FOV 비가용 비율',
  series:PROPOSED_TRAIN,x:'episode',y:['fov_predicted_mean','fov_actual_mean'],
  labels:['예측 q','실측 y'],smooth:12,ymin:0,ymax:1},
 {id:'benchmark_fov_prediction_error',view:'benchmark',requiresRole:'additive_reward_term',
  title:'readout 오차 · MAE와 편향 (관측된 미래창만)',
  series:PROPOSED_TRAIN,x:'episode',y:['fov_prediction_mae','fov_prediction_bias'],
  labels:['MAE','편향(예측−실측)'],smooth:12},
 {id:'benchmark_onto_share',view:'benchmark',requiresRole:'additive_reward_term',
  title:'추가 보상이 차지한 step 보상 크기 비중',
  series:PROPOSED_TRAIN,x:'episode',y:'ontology_fov_reward_share',smooth:12,ymin:0},
 {id:'benchmark_onto_sum',view:'benchmark',requiresRole:'additive_reward_term',
  title:'episode당 추가 보상 합 −λ·Σq (항상 ≤ 0)',
  series:PROPOSED_TRAIN,x:'episode',y:'ontology_fov_reward_sum',smooth:12},
 {id:'benchmark_fov_risk',view:'benchmark',requiresRole:'additive_reward_term',
  title:'현재 episode · 예측 FOV 비가용 비율과 추가 보상',
  series:PROPOSED_STEP,x:'step',
  y:['predicted_fov_unavailability','ontology_fov_reward'],
  labels:['예측 q (향후 1s FOV 밖 시간 비율)','추가 보상 −λ·q'],smooth:3},
 {id:'onto_state_nodes',view:'benchmark',requiresRole:'state_representation',
  title:'현재 episode · 온톨로지 상황 그래프의 위험 node 값',
  series:BENCHMARK_STEP,x:'step',
  y:['onto_AlignmentError','onto_FOVMargin','onto_RelativeRange',
     'onto_MeasurementAge'],
  labels:['정렬 오차','FOV 여유 소진','상대 거리','관측 경과'],
  smooth:3,ymin:0,ymax:1},
 {id:'onto_state_support',view:'benchmark',requiresRole:'state_representation',
  title:'현재 episode · 지원 node와 중간 node',
  series:BENCHMARK_STEP,x:'step',
  y:['onto_PadVisibility','onto_TouchdownSafety','onto_DescentRate',
     'onto_TargetMotion'],
  labels:['패드 가시성','접지 안전도(중간)','하강 속도','표적 이동'],
  smooth:3,ymin:0,ymax:1},
 {id:'onto_graph_embedding',view:'benchmark',requiresRole:'state_representation',
  title:'현재 episode · 그래프 읽기 g_t의 크기 (부호기가 동작하는지)',
  series:PROPOSED_STEP,x:'step',
  y:['graph_embedding_norm','graph_embedding_max'],
  labels:['RMS |g_t|','최대 |g_t|'],smooth:3,ymin:0,ymax:1},
 {id:'planar_command',view:'benchmark',
  title:'현재 episode · 평면 엔벨로프가 실제로 보낸 명령',
  series:BENCHMARK_STEP,x:'step',
  y:['cmd_forward_m_s','cmd_vertical_m_s','cmd_lateral_m_s'],
  labels:['전후진 v [m/s]','상승·하강 v [m/s]','횡방향 v [m/s] (항상 0)'],smooth:3},
 {id:'planar_tilt',view:'benchmark',
  title:'현재 episode · 종방향 틸트 명령과 요레이트',
  series:BENCHMARK_STEP,x:'step',
  y:['cmd_tilt_deg','cmd_yaw_rate_deg_s'],
  labels:['틸트 [deg]','요레이트 [deg/s] (항상 0)'],smooth:3},
 {id:'graph3d',view:'benchmark',kind:'graph',
  title:'온톨로지 상황 그래프 G_t · node 값과 관계 (제안 arm은 이 그래프를 상태로 읽는다)'},
 {id:'rgat_neural_flow',view:'benchmark',kind:'rgatflow',requiresRole:'state_representation',
  title:'Ontology R-GAT · 실시간 message-passing 신경망'},

 {id:'head_run',view:'benchmark',kind:'heading',
  title:'3 · 실행 상태',
  note:'비교 대상이 아니라 실행이 건전한지 보는 값이다. 두 arm에 동일하게 적용되며, '
      +'여기서의 차이는 결론이 아니라 교란 가능성의 단서로만 읽는다.'},
 {id:'parallel_pairs',view:'benchmark',kind:'pairs',
  title:'동시 비행쌍 · 한 Isaac Sim 월드 / 독립 PX4·PPO'},
 {id:'teacher_demos',view:'benchmark',kind:'teacher',
  title:'교사 시연 비행 · 실시간 (behavior-cloning warm start, training-only)'},
 {id:'trajectories',view:'benchmark',kind:'trajectories',
  title:'실시간 궤적 · 절대 world ENU X / 고도 · 4개 pair 세로 배치 · 공통 고정 축'},
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
  title:'제안 알고리즘 · 센서 → 온톨로지 상황 그래프 → R-GAT → 정책 상태 g_t'},
 {id:'mdp_structure',view:'benchmark',kind:'mdp',
  title:'두 강화학습 에이전트의 State · Action · Environment · Reward'},
];

const LABELS={benchmark_step:'current episode',
 benchmark_train_shin_se_fixed:'Baseline · PPO (관측 벡터)',
 benchmark_train_shin_se_onto_rgat_state:'Proposed · PPO + 온톨로지 상황 그래프',
 benchmark_eval_shin_se_fixed:'Baseline · PPO (관측 벡터)',
 benchmark_eval_shin_se_onto_rgat_state:'Proposed · PPO + 온톨로지 상황 그래프'};
const root=document.getElementById('root');
for(const c of CARDS){
  const el=document.createElement('section');
  el.className=(c.kind==='heading'?'card section'
    :'card'+((c.id==='tiles'||['graph','rgatflow','contract','evalbars','pairs','pairplots','phase',
      'runpipe','algopipe','mdp','fovstatus','teacher','trajectories','collection']
      .includes(c.kind))?' wide':''))
    +(c.kind==='graph'?' g3d':'')+(c.kind==='rgatflow'?' rgat-flow':'');
  el.id='card-'+c.id;
  el.dataset.view=c.view||'common';
  if(c.requiresRole)el.dataset.role=c.requiresRole;
  el.hidden=c.view==='benchmark';
  if(c.id==='tiles'){el.innerHTML='<div class="tiles" id="tiles"></div>';}
  else if(c.kind==='heading'){el.innerHTML=`<h2>${c.title}</h2>`
    +(c.note?`<p>${c.note}</p>`:'');}
  else if(c.kind==='phase'){el.innerHTML='<div class="phase-status" id="phase-status"></div>';}
  else if(c.kind==='fovstatus'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="fovwrap" id="fov-status"></div>`;}
  else if(c.kind==='collection'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="collection" id="collection-stage"></div>`;}
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
  else if(c.kind==='teacher'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="teacher" id="teacher-demos"></div>`;}
  else if(c.kind==='trajectories'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="traj-grid" id="traj-grid"></div>`;}
  else if(c.kind==='pairplots'){el.innerHTML=`<h2>${c.title}</h2>`
    +`<div class="pair-plot-grid" id="pair-plot-grid-${c.id}"></div>`;}
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
    <section class="sensor-lineage">
      <div class="sensor-lineage-head"><h3>Sensor data → semantic/context → ontology node</h3>
        <span>Hover any box to isolate the exact construction path and transform.</span>
        <span class="boundary" id="g3d-sensor-boundary">onboard inputs only</span></div>
      <div class="sensor-map" id="g3d-sensor-map"></div>
      <div class="sensor-lineage-legend" id="g3d-sensor-legend"></div>
    </section>
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
  else if(c.kind==='rgatflow'){el.innerHTML=`<h2>${c.title}</h2>
    <div class="rf-head"><b id="rf-source">제안 arm graph 대기</b>
      <span class="rf-pill live">PPO와 함께 학습</span>
      <span class="rf-pill">state representation</span>
      <span class="rf-pill" id="rf-attention">attention telemetry 대기</span>
      <span style="margin-left:auto">drag: 회전 · wheel: 확대</span></div>
    <canvas id="cv-rgat-flow"></canvas><div class="tip rf-tip" id="rf-tip"></div>
    <div class="rf-stats" id="rf-stats"></div>
    <div class="rf-legend" id="rf-legend"></div>
    <div class="note">실제 G_t node·relation·현재 activation을 사용한 계산 구조 뷰다.
      입력 node 사이의 색 선은 선언된 ontology relation이고, R-GAT 내부의 조밀한 선은
      feature-channel message routing을 나타낸다. attention 값이 payload에 있을 때만 선 굵기에
      반영하며, 값이 없을 때는 구조를 균일하게 그린다. 시각적 연결은 인과 설명이 아니다.</div>`;}
  else{el.innerHTML=`<h2>${c.title}</h2><canvas id="cv-${c.id}"></canvas>
    <div class="legend" id="lg-${c.id}"></div>`;}
  root.appendChild(el);
}
// Cards are created after the browser's native hash-scroll pass. Repeat it
// once so a direct link to the live R-GAT panel lands on the requested card.
if(location.hash)requestAnimationFrame(()=>{
  const target=document.querySelector(location.hash);if(target)target.scrollIntoView();});
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
        ?(sources.length>1?`${seriesLabel(s)} ${named}`:named)
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
function methodLabel(method){
  return armLabel(method)||LABELS['benchmark_train_'+method]||method||'초기화 대기';}
function drawPairPlots(card,state){
  const s=state.scalars||{},pairs=s.parallel_pair_status||[];
  // One plot per physical pair. This was two, hardcoded, while the stage the
  // machine sizes itself to is four: pairs 3 and 4 flew the whole run without
  // appearing on either live panel.
  const count=Math.max(1,Number(s.parallel_pair_count||pairs.length||1));
  const grid=document.getElementById(`pair-plot-grid-${card.id}`);
  if(!grid)return;
  if(grid.childElementCount!==count){
    grid.innerHTML=Array.from({length:count},(_,index)=>
      `<div class="pair-plot"><h3 id="pair-title-${card.id}-${index}">`
      +`Pair ${index+1} · 초기화 대기</h3><canvas id="cv-${card.id}-${index}"></canvas>`
      +`<div class="legend" id="lg-${card.id}-${index}"></div></div>`).join('');
  }
  for(let index=0;index<count;index++){
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
  // Head to head is between the learned arms; a non-learned reference arm is
  // reported beside them rather than being one side of the delta.
  const learned=learnedArms();
  // The proposed side is the arm carrying the R-GAT reward term, not
  // whichever arm the run happens to list second.
  const proposedArm=ontologyArms()[0]||learned[1]||{};
  const prop=proposedArm.method||'shin_se_onto_rgat_state';
  const baseArm=learned.find(a=>a.method!==prop)||learned[0]||{};
  const base=baseArm.method||'shin_se_fixed';
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
function collectionPanel(state){
  const box=document.getElementById('collection-stage');if(!box)return;
  const s=state.scalars||{};
  const stage=String(s.collection_stage||'');
  const datasets=s.collection_datasets||[];
  const pairs=s.collection_pairs||[];
  if(!stage&&!datasets.length){
    box.innerHTML='<div class="cnote">수집 단계가 아직 보고되지 않았습니다. '
      +'<code>--stage collect</code>는 데이터만 모으고, <code>--stage train</code>은 '
      +'모은 데이터로 학습합니다. 기본값 <code>all</code>은 수집을 끝낸 뒤 학습합니다.</div>';
    return;}
  const collecting=stage==='collect'||stage==='all';
  const training=stage==='train'||stage==='all';
  const done=!!s.collection_complete;
  const cell=(label,detail,cls)=>`<div class="cstage ${cls}"><b>${escapeHTML(label)}</b>`
    +`<small>${detail}</small></div>`;
  const cards=[
    cell('1 · 수집',
      collecting?(done?'완료 · 비행 없음':'진행 중'):'이 실행에서는 생략',
      collecting?(done?'done':'active'):''),
    ...datasets.map(entry=>{
      const reused=Number(entry.reused_episodes),flown=Number(entry.flown_episodes);
      const provenance=Number.isFinite(reused)
        ?`재사용 ${reused} · 새로 비행 ${Number.isFinite(flown)?flown:'--'}`
        :'이번 실행 누적';
      return cell(escapeHTML(entry.name),
        `<b style="font-size:16px">${escapeHTML(entry.episodes)}</b> episode<br>${provenance}`,
        'done');}),
    cell('2 · 학습',
      training?'동결 readout → 전 arm PPO → 평가':'이 실행에서는 생략',
      training&&done?'active':(training?'':'')),
  ];
  box.innerHTML=cards.join('')
    +`<div class="cnote">수집은 PPO를 한 episode도 돌리지 않으므로 물리 pair `
    +`<b>${pairs.length?escapeHTML(pairs.join(', ')):'--'}</b>을(를) 모두 쓴다. `
    +`학습 단계는 수집을 위해 비행하지 않으며, 필요한 데이터셋이 없으면 `
    +`<code>--stage collect</code>를 지목하는 오류로 멈춘다.</div>`;
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
    const proposed=isOntologyArm(method);
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
const TEACHER_MODE={following:'패드 추종',aligned:'정렬 · 하강 대기',descending:'하강',
  lost:'시야 이탈',lost_climbing:'시야 이탈 · 상승 회복 중',reacquired:'재포착 · 재추종'};
function teacherTile(pair,index){
  // One live flight, on the physical pair flying it. The warm start runs on
  // every idle pair, so there is one of these per collecting pair.
  const live=pair.teacher||null;
  const num=(v,digits)=>Number.isFinite(Number(v))?Number(v).toFixed(digits):'--';
  const mode=live?String(live.mode||''):'';
  const cls=mode.startsWith('lost')?'lost':mode==='reacquired'?'reacquired':'';
  const fov=live?(live.geometric_in_fov===false?'프레임 이탈'
    :live.geometric_in_fov===true?'프레임 내':'--'):'--';
  // The deck this pair is on: the warm start rotates one per seed, so the
  // tiles differ from each other and from flight to flight.
  const deck=String(pair.scenario||'').replaceAll('_',' ');
  return `<div class="tprog"><h4>물리 Pair ${Number(index)+1}`
    +` <span>· seed ${escapeHTML(live?live.seed??'--':'--')}`
    +` · step ${escapeHTML(live?live.step??'--':'--')}`
    +(deck?` · ${escapeHTML(deck)}`:'')+`</span></h4>`
    +`<div class="tmode ${cls}">${escapeHTML(TEACHER_MODE[mode]||mode||'대기')}</div>`
    +`<div class="tlive"><div><b>${num(live&&live.altitude_m,2)} m</b><small>고도 (패드 기준)</small></div>`
    +`<div><b>${num(live&&live.lateral_error_m,2)} m</b><small>측방 오차</small></div>`
    +`<div><b>${num(live&&live.target_vz_m_s,2)} m/s</b><small>목표 vz (+ 상승)</small></div>`
    +`<div><b>${escapeHTML(fov)}</b><small>기하 FOV (패드 중심)</small></div>`
    +`<div><b>${escapeHTML(live?live.losses??0:0)}</b><small>시야 이탈 횟수</small></div>`
    +`<div><b>${num(live&&live.max_altitude_after_loss_m,2)} m</b><small>이탈 후 최고 고도</small></div>`
    +`</div></div>`;
}
function teacherPanel(state){
  const box=document.getElementById('teacher-demos');if(!box)return;
  const s=state.scalars||{},demo=s.teacher_demonstrations||null;
  const pairs=s.parallel_pair_status||[];
  const flying=pairs.filter(p=>p.teacher).map(p=>Number(p.index));
  if(!demo&&!flying.length){box.innerHTML='<div class="tprog">시연 단계 대기 중 · 저장된 시연이 '
    +'그대로 재사용되면 비행이 없어 이 패널은 비어 있습니다.</div>';return;}
  const d=demo||{};
  // Which physical pairs are collecting: what the stage published, else the
  // pairs that have live teacher telemetry, else the single reported pair.
  const declared=(d.collection_pairs||[]).map(Number).filter(Number.isFinite);
  const indices=declared.length?declared
    :(flying.length?flying:[Number(d.pair_index||0)]);
  const num=(v,digits)=>Number.isFinite(Number(v))?Number(v).toFixed(digits):'--';
  const ratio=d.required?Math.min(1,Number(d.accepted||0)/Number(d.required)):0;
  const rows=(d.recent||[]).slice().reverse();
  const table=rows.length?'<table><tr><th>seed</th><th>pair</th><th>덱</th><th>결과</th><th>step</th>'
    +'<th>착지 측방오차 m</th><th>FOV 손실 비율</th></tr>'
    +rows.map(r=>{const color=r.accepted?'var(--good)'
        :(r.status==='infrastructure_failure'?'var(--muted)':'var(--bad)');
      const label=r.accepted?'착륙 · 채택':(r.status==='infrastructure_failure'?'인프라 스킵':'실패');
      const pair=Number.isFinite(Number(r.pair_index))?Number(r.pair_index)+1:'--';
      const deck=String(r.scenario||'').replaceAll('_',' ');
      return `<tr><td>${escapeHTML(num(r.seed,0))}</td><td>${escapeHTML(pair)}</td>`
        +`<td>${escapeHTML(deck||'--')}</td>`
        +`<td style="color:${color}">${label}</td>`
        +`<td>${escapeHTML(num(r.steps,0))}</td><td>${escapeHTML(num(r.lateral_error_m,2))}</td>`
        +`<td>${Number.isFinite(Number(r.fov_loss_fraction))
          ?(100*Number(r.fov_loss_fraction)).toFixed(0)+'%':'--'}</td></tr>`;}).join('')
    +'</table>':'<div class="tnote">아직 완료된 비행이 없습니다.</div>';
  const collecting=indices.map(i=>Number(i)+1).join('·');
  // One deck or a rotation: six names on one line ran into the fields after
  // them, and the deck each pair is actually flying is on its own tile.
  const decks=String(d.scenario||'').split(' · ').filter(Boolean);
  const tiles=indices.map(index=>teacherTile(
    pairs.find(p=>Number(p.index)===Number(index))||{},index)).join('');
  box.innerHTML=`<div class="tprog"><b>${d.complete?'시연 확보 완료':'시연 비행 중'}`
    +` · 채택 ${escapeHTML(d.accepted??0)} / ${escapeHTML(d.required??'--')}</b>`
    +`<div class="tbar"><i style="width:${(100*ratio).toFixed(0)}%"></i></div>`
    +`<div class="tnote">비행 ${escapeHTML(d.flights??0)} / ${escapeHTML(d.max_flights??'--')}`
    +` · 인프라 스킵 ${escapeHTML(d.skips??0)} · 교사 ${escapeHTML(d.teacher||'--')}<br>`
    +`시나리오 ${escapeHTML(decks.length>1?`${decks.length}종 순환`
        :String(d.scenario||'--').replaceAll('_',' '))}`
    +` · 지문 ${escapeHTML(d.fingerprint||'--')}`
    +` · 수집 물리 Pair ${escapeHTML(collecting)} (${indices.length}대 동시)</div>${table}`
    +`<div class="tnote">교사 행동은 training-only label입니다. 학생에게는 image embedding과 `
    +`proprioception만 전달됩니다.</div></div>`
    +`<div class="tgrid">${tiles}</div>`;
}
function drawTrajectories(state){
  const grid=document.getElementById('traj-grid');if(!grid)return;
  const s=state.scalars||{},pairs=s.parallel_pair_status||[];
  const count=Math.max(1,Number(s.parallel_pair_count||pairs.length||1));
  if(grid.childElementCount!==count){
    grid.innerHTML=Array.from({length:count},(_,i)=>`<div class="traj-plot">`
      +`<h3 id="traj-title-${i}">Pair ${i+1}</h3><canvas id="cv-traj-${i}"></canvas>`
      +`<div class="legend" id="lg-traj-${i}"></div></div>`).join('');
  }
  for(let i=0;i<count;i++){
    const pair=pairs.find(p=>Number(p.index)===i)||{};
    const rows=state.series['benchmark_step_pair_'+i]||[];
    const uav=[],pad=[];
    for(const r of rows){
      if(Number.isFinite(Number(r.uav_x))&&Number.isFinite(Number(r.uav_y)))
        uav.push([Number(r.uav_x),Number(r.uav_y),Number(r.uav_z)]);
      if(Number.isFinite(Number(r.pad_x))&&Number.isFinite(Number(r.pad_y)))
        pad.push([Number(r.pad_x),Number(r.pad_y),Number(r.pad_z)]);}
    // The panel is the PHYSICAL pair and never moves: its canvas, its series
    // and its index are fixed for the whole run. What rotates is the policy
    // flying it -- during collection every pair flies the reward-design source
    // policy, and crossover evaluation cycles the arms across pairs seed by
    // seed -- so titling the panel by the ACTIVE policy made the panels read
    // as though their assignment kept swapping between them. Name the stable
    // identity first and mark a borrowed policy as borrowed, which is what
    // the pair cards already did and this panel did not.
    const assigned=String(pair.assigned_method||pair.method||'');
    const active=String(pair.active_method||assigned);
    const title=document.getElementById(`traj-title-${i}`);
    if(title)title.textContent=`물리 Pair ${i+1} · 배정 ${methodLabel(assigned)}`
      +(active&&active!==assigned?` · 현재 비행 ${methodLabel(active)}`:'')
      +` · ${String(pair.phase||'waiting')} · ${String(pair.episode_kind||'episode')} ${pair.episode??0}`
      +` · step ${pair.step??0} · ${String(pair.status||'')}`;
    drawTrajectory(document.getElementById(`cv-traj-${i}`),
      document.getElementById(`lg-traj-${i}`),uav,pad);
  }
}
// A fixed-size absolute viewport, re-anchored only when an episode's series is
// reset. The first pad sample is the episode's initial UGV position. Put that
// position on the leading boundary so carried world X cannot walk out of a
// global box over successive episodes. The shuttle eventually reverses, so
// select the leading boundary from its first observed movement direction.
// Plot Y is physical altitude (world ENU Z), whose configured entry range ends
// at 8 m; 0..10 m keeps the entire flight and removes impossible negative air.
const TRAJECTORY_VIEWPORT=Object.freeze({
  xLength:170,yMin:0,yMax:10,xTick:20,yTick:1});
function drawTrajectory(cv,lg,uav,pad){
  if(!cv)return;
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const css=getComputedStyle(document.body);
  const muted=css.getPropertyValue('--muted').trim(),gridColor=css.getPropertyValue('--grid').trim();
  const empty=!uav.length&&!pad.length;
  const U=uav.map(p=>[p[0],p[2]]),P=pad.map(p=>[p[0],p[2]]);
  const initialUgvX=P.length?P[0][0]:(U.length?U[0][0]:0);
  const directionSample=P.find((p,i)=>i>0&&Math.abs(p[0]-initialUgvX)>1e-4);
  const ugvDirection=(directionSample&&directionSample[0]<initialUgvX)?-1:1;
  const bound=ugvDirection<0
    ?{xMin:initialUgvX-TRAJECTORY_VIEWPORT.xLength,xMax:initialUgvX,
      yMin:TRAJECTORY_VIEWPORT.yMin,yMax:TRAJECTORY_VIEWPORT.yMax,
      xTick:TRAJECTORY_VIEWPORT.xTick,yTick:TRAJECTORY_VIEWPORT.yTick}
    :{xMin:initialUgvX,xMax:initialUgvX+TRAJECTORY_VIEWPORT.xLength,
      yMin:TRAJECTORY_VIEWPORT.yMin,yMax:TRAJECTORY_VIEWPORT.yMax,
      xTick:TRAJECTORY_VIEWPORT.xTick,yTick:TRAJECTORY_VIEWPORT.yTick};
  const xSpan=bound.xMax-bound.xMin,ySpan=bound.yMax-bound.yMin;
  const padL=37,padB=27,padT=8,padR=9;
  const availableW=Math.max(10,w-padL-padR),availableH=Math.max(10,h-padT-padB);
  // Every pair gets these same two scale factors and literal bounds. X and
  // altitude intentionally use the full wide rectangle; equal metric aspect
  // would compress the useful 0..10 m altitude band to a few pixels.
  const scaleX=availableW/xSpan,scaleY=availableH/ySpan;
  const W=availableW,H=availableH,ox=padL,oy=padT;
  const px=v=>ox+(v-bound.xMin)*scaleX;
  const py=v=>oy+H-(v-bound.yMin)*scaleY;
  g.strokeStyle=gridColor;g.lineWidth=.6;g.setLineDash([1.5,2.5]);
  g.fillStyle=muted;g.font='9px Arial';
  for(let v=bound.xMin;v<=bound.xMax;v+=bound.xTick){
    const X=px(v);g.beginPath();g.moveTo(X,oy);g.lineTo(X,oy+H);g.stroke();
    g.textAlign='center';g.fillText(Number(v.toFixed(1)),X,oy+H+13);}
  for(let v=bound.yMin;v<=bound.yMax;v+=bound.yTick){
    const Y=py(v);g.beginPath();g.moveTo(ox,Y);g.lineTo(ox+W,Y);g.stroke();
    g.textAlign='right';g.fillText(String(v),ox-5,Y+3);}
  g.setLineDash([]);g.strokeStyle='#262626';g.lineWidth=.8;g.strokeRect(ox,oy,W,H);
  g.fillStyle=muted;g.font='9px Arial';g.textAlign='center';
  g.fillText('world ENU X (m)',ox+W/2,h-2);
  g.save();g.translate(8,oy+H/2);g.rotate(-Math.PI/2);g.fillText('altitude / world ENU Z (m)',0,0);g.restore();
  g.save();g.beginPath();g.rect(ox,oy,W,H);g.clip();
  const poly=(arr,color,dash)=>{if(!arr.length)return;
    g.strokeStyle=color;g.lineWidth=1.8;g.setLineDash(dash);g.beginPath();
    arr.forEach((p,k)=>{const X=px(p[0]),Y=py(p[1]);k?g.lineTo(X,Y):g.moveTo(X,Y);});g.stroke();
    g.setLineDash([]);g.lineWidth=1.2;
    g.fillStyle='#fff';g.beginPath();g.arc(px(arr[0][0]),py(arr[0][1]),3.5,0,2*Math.PI);g.fill();g.stroke();
    const last=arr[arr.length-1];g.fillStyle=color;g.beginPath();
    g.arc(px(last[0]),py(last[1]),4.5,0,2*Math.PI);g.fill();};
  poly(P,PALETTE[1],[4,3]);
  poly(U,PALETTE[0],[]);
  if(U.length&&P.length){
    const u=U[U.length-1],p=P[P.length-1];
    g.strokeStyle='#262626';g.globalAlpha=.55;g.lineWidth=1;g.setLineDash([3,2]);
    g.beginPath();g.moveTo(px(u[0]),py(u[1]));g.lineTo(px(u[0]),py(p[1]));g.stroke();
    g.setLineDash([]);g.globalAlpha=1;g.fillStyle='#262626';g.font='9px Arial';
    g.textAlign='left';const altitudeGap=u[1]-p[1];
    g.fillText(`Δh ${altitudeGap.toFixed(2)} m`,px(u[0])+4,
      (py(u[1])+py(p[1]))/2-4);}
  g.restore();
  if(empty){g.fillStyle=muted;g.font='12px sans-serif';g.textAlign='center';
    g.fillText('no trajectory yet',ox+W/2,oy+H/2);}
  if(uav.length){const last=uav[uav.length-1],pz=pad.length?pad[pad.length-1][2]:null;
    g.fillStyle='#262626';g.font='10px Arial';g.textAlign='left';
    g.fillText(`UAV X ${last[0].toFixed(1)} m · 고도 ${last[2].toFixed(2)} m`+
      `${pz===null?'':` · 패드 위 ${(last[2]-pz).toFixed(2)} m`} · ${uav.length} step`,ox+4,oy+12);}
  if(lg)lg.innerHTML=`<span><i style="background:${PALETTE[0]}"></i>UAV</span>`
    +`<span><i style="background:${PALETTE[1]}"></i>착륙 패드</span>`
    +`<span>○ 시작 &nbsp;● 현재 &nbsp;· 고도 기준 측면 뷰 · `
    +`X ${bound.xMin.toFixed(1)}..${bound.xMax.toFixed(1)} m / `
    +`고도 ${bound.yMin}..${bound.yMax} m · 에피소드 초기 UGV 기준 · 모든 pair 동일 축척</span>`;
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
  // Display order only: a deck not listed here still gets its group, appended
  // below, so a new scenario needs no dashboard change to appear.
  const preferred=['straight_escape_burst_track','straight_escape_burst','training_random_walk',
    'training_random_walk_escape_burst','straight_8mps',
    'linear_acceleration_wave','circle','zigzag','u_turn',
    'vertical_heave_boat'];
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
         data:null,graphs:{},state:null,selected:'',manual:false,dirty:true,pts:[],fit:0};
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
const PROV_SEMANTICS=[
  ['centroid_x','Keypoint centroid x','normalized image','context'],
  ['centroid_y','Keypoint centroid y','normalized image','context'],
  ['raw_target_scale','Raw target scale','normalized image','context'],
  ['keypoint_confidence','Keypoint confidence','0..1','semantic'],
  ['visible_keypoint_fraction','Visible keypoint fraction','0..1','semantic'],
  ['image_alignment','Image alignment','0..1','semantic'],
  ['apparent_target_scale','Apparent target scale','0..1','semantic'],
  ['image_plane_motion_safety','Image motion safety','0..1','context'],
  ['scale_rate_safety','Scale-rate safety','0..1','context'],
  ['visibility_memory','Visibility memory','0..1','context'],
  ['reacquisition_trend','Reacquisition trend','0..1','context'],
  ['vertical_motion_safety','Vertical-motion safety','0..1','semantic'],
  ['attitude_stability','Attitude stability','0..1','semantic'],
  ['battery_risk','Battery risk','0..1','semantic'],
  ['visual_loss_risk','Visual-loss risk','0..1','context'],
  ['visual_loss_duration_s','Visual-loss duration','s','context']];
const PROV_SENSOR_LINKS=[
  ['landing_camera','centroid_x','confidence-weighted keypoint centroid'],
  ['landing_camera','centroid_y','confidence-weighted keypoint centroid'],
  ['landing_camera','raw_target_scale','RMS keypoint spread'],
  ['keypoint_encoder','keypoint_confidence','heatmap entropy + visibility'],
  ['keypoint_encoder','visible_keypoint_fraction','visible keypoints / 6'],
  ['keypoint_encoder','image_alignment','centroid distance from image centre'],
  ['keypoint_encoder','apparent_target_scale','reliability-weighted scale'],
  ['landing_camera','image_plane_motion_safety','centroid delta / dt'],
  ['temporal_context','image_plane_motion_safety','previous frame'],
  ['landing_camera','scale_rate_safety','scale delta / dt'],
  ['temporal_context','scale_rate_safety','previous frame'],
  ['keypoint_encoder','visibility_memory','current reliability'],
  ['temporal_context','visibility_memory','1.5 s exponential memory'],
  ['keypoint_encoder','reacquisition_trend','confidence recovery'],
  ['temporal_context','reacquisition_trend','previous confidence'],
  ['px4_odometry','vertical_motion_safety','exp(-|vz| / 0.6)'],
  ['px4_imu','attitude_stability','exp(-tilt / 22 deg)'],
  ['battery_monitor','battery_risk','1 - reserve'],
  ['keypoint_encoder','visual_loss_risk','keypoint dropout'],
  ['temporal_context','visual_loss_risk','loss duration / 2 s'],
  ['temporal_context','visual_loss_duration_s','consecutive dropout time']];
const PROV_ONTOLOGY_LINKS=[
  ['centroid_x','AlignmentError','bearing from camera nadir column'],
  ['vertical_motion_safety','DescentRate','recover |vz|, then soft saturation'],
  ['image_plane_motion_safety','TargetMotion','1 - motion safety'],
  ['centroid_x','FOVMargin','distance to horizontal image edge'],
  ['centroid_y','FOVMargin','distance to vertical image edge'],
  ['visual_loss_risk','MeasurementAge','bounded observation age'],
  ['raw_target_scale','RelativeRange','inverse apparent scale'],
  ['keypoint_confidence','PadVisibility','confidence × visible fraction'],
  ['visible_keypoint_fraction','PadVisibility','confidence × visible fraction'],
  ['visibility_memory','PadVisibility','dropout memory support'],
  ['visual_loss_risk','PadVisibility','suppresses stale memory']];
function graphPointFor(state,graph){
  if(!state||!graph)return {};
  const series=state.series||{},pair=Number(graph.pair_index);
  let rows=Number.isFinite(pair)?series[`benchmark_step_pair_${pair}`]:null;
  if(!rows||!rows.length)rows=series[`benchmark_step_${graph.method||''}`]||[];
  return rows.length?rows[rows.length-1]:{};
}
function fallbackSensorProvenance(state,graph){
  const point=graphPointFor(state,graph),value=name=>{
    if(name==='visual_loss_duration_s'){
      const risk=Number(point.semantic_visual_loss_risk);
      return Number.isFinite(risk)?Math.min(2,2*risk):null;}
    const raw=point[`semantic_${name}`];return Number.isFinite(Number(raw))?Number(raw):null;};
  const vertical=Math.max(1e-6,Math.min(1,Number(value('vertical_motion_safety')??1)));
  const sensors=[
    {id:'landing_camera',label:'Landing camera',kind:'sensor',readings:[
      {label:'alignment',value:value('image_alignment'),unit:'0..1'},
      {label:'apparent scale',value:value('apparent_target_scale'),unit:'0..1'}]},
    {id:'keypoint_encoder',label:'Frozen 6-keypoint encoder',kind:'inference',readings:[
      {label:'confidence',value:value('keypoint_confidence'),unit:'0..1'},
      {label:'visible',value:value('visible_keypoint_fraction'),unit:'fraction'}]},
    {id:'temporal_context',label:'Visual history',kind:'context',readings:[
      {label:'loss age',value:value('visual_loss_duration_s'),unit:'s'}]},
    {id:'px4_odometry',label:'PX4 local odometry',kind:'sensor',readings:[
      {label:'|vertical speed|',value:-.6*Math.log(vertical),unit:'m/s'}]},
    {id:'px4_imu',label:'PX4 IMU attitude',kind:'sensor',readings:[
      {label:'stability',value:value('attitude_stability'),unit:'0..1'}]},
    {id:'battery_monitor',label:'PX4 battery monitor',kind:'sensor',readings:[
      {label:'reserve',value:value('battery_risk')===null?null:1-value('battery_risk'),unit:'fraction'}]}];
  const semantics=PROV_SEMANTICS.map(([id,label,unit,kind])=>({
    id,label,unit,kind,value:value(id)}));
  const nodeNames=new Set((graph.nodes||[]).map(node=>String(node.name)));
  const connections=PROV_SENSOR_LINKS.map(([source,target,transform])=>({
    source,target,transform,stage:'sensor_to_semantic'}));
  for(const [source,target,transform] of PROV_ONTOLOGY_LINKS)if(nodeNames.has(target))
    connections.push({source,target,transform,stage:'semantic_to_ontology'});
  return {format:'ontology-rgat-sensor-provenance-v1-fallback',sensors,semantics,connections,
    boundary:['No simulator truth','No geometric FOV label','No relative-pose estimate',
      'Onboard camera/PX4 signals only']};
}
function provenanceValue(value,unit){
  const number=Number(value);if(!Number.isFinite(number))return '—';
  const digits=Math.abs(number)>=10?2:3;
  return `${number.toFixed(digits)}${unit&&unit!=='0..1'?' '+unit:''}`;
}
function graphSensorMap(state){
  const box=document.getElementById('g3d-sensor-map');if(!box)return;
  const graph=G.data;if(!graph){box.innerHTML='<div class="audit-empty">graph data 대기</div>';return;}
  const p=(graph.sensor_provenance&&graph.sensor_provenance.sensors)
    ?graph.sensor_provenance:fallbackSensorProvenance(state,graph);
  const sensors=p.sensors||[],semantics=p.semantics||[],nodes=graph.nodes||[];
  const links=(p.connections||[]).filter(link=>
    link.stage==='sensor_to_semantic'||nodes.some(node=>node.name===link.target));
  const width=1180,top=38,row=37;
  const height=Math.max(390,top+Math.max(semantics.length*row,sensors.length*82,nodes.length*52)+12);
  const columns={sensor:{x:18,w:245},semantic:{x:420,w:292},ontology:{x:895,w:260}};
  const positions=new Map();
  const spread=(items,column,step)=>{const total=(items.length-1)*step;
    const start=top+Math.max(0,(height-top-12-total)/2);
    items.forEach((item,index)=>positions.set(item.id||item.name,
      {x:column.x,y:start+index*step,w:column.w,h:step-7}));};
  spread(sensors,columns.sensor,82);spread(semantics,columns.semantic,row);
  spread(nodes.map(node=>({...node,id:node.name})),columns.ontology,52);
  const paths=links.map((link,index)=>{const A=positions.get(link.source),B=positions.get(link.target);
    if(!A||!B)return '';const x0=A.x+A.w,y0=A.y+A.h/2,x1=B.x,y1=B.y+B.h/2;
    const bend=(x1-x0)*.48,kind=link.stage==='semantic_to_ontology'?' ontology':'';
    return `<path class="prov-edge${kind}" data-link="${index}" data-source="${escapeHTML(link.source)}" `+
      `data-target="${escapeHTML(link.target)}" d="M${x0},${y0} C${x0+bend},${y0} ${x1-bend},${y1} ${x1},${y1}">`+
      `<title>${escapeHTML(link.transform||'direct')}</title></path>`;}).join('');
  const sensorCards=sensors.map(sensor=>{const q=positions.get(sensor.id),readings=sensor.readings||[];
    const detail=readings.map(reading=>`${reading.label}: ${provenanceValue(reading.value,reading.unit)}`).join(' · ');
    return `<g class="prov-node" data-entity="${escapeHTML(sensor.id)}"><rect class="prov-card sensor" `+
      `x="${q.x}" y="${q.y}" width="${q.w}" height="${q.h}" rx="5"/>`+
      `<text class="prov-title" x="${q.x+9}" y="${q.y+15}">${escapeHTML(sensor.label)}</text>`+
      `<text class="prov-value" x="${q.x+9}" y="${q.y+31}">${escapeHTML(detail.slice(0,42))}</text>`+
      `<text class="prov-meta" x="${q.x+9}" y="${q.y+45}">${escapeHTML(sensor.kind||'sensor')}</text>`+
      `<title>${escapeHTML(detail)}</title></g>`;}).join('');
  const semanticCards=semantics.map(channel=>{const q=positions.get(channel.id);
    return `<g class="prov-node" data-entity="${escapeHTML(channel.id)}"><rect class="prov-card context" `+
      `x="${q.x}" y="${q.y}" width="${q.w}" height="${q.h}" rx="4"/>`+
      `<text class="prov-title" x="${q.x+8}" y="${q.y+14}">${escapeHTML(channel.label)}</text>`+
      `<text class="prov-value" text-anchor="end" x="${q.x+q.w-8}" y="${q.y+14}">`+
      `${escapeHTML(provenanceValue(channel.value,channel.unit))}</text>`+
      `<text class="prov-meta" x="${q.x+8}" y="${q.y+26}">${escapeHTML(channel.kind||'semantic')}</text></g>`;}).join('');
  const ontologyCards=nodes.map(node=>{const q=positions.get(node.name);
    return `<g class="prov-node" data-entity="${escapeHTML(node.name)}"><rect class="prov-card ontology" `+
      `x="${q.x}" y="${q.y}" width="${q.w}" height="${q.h}" rx="5"/>`+
      `<text class="prov-title" x="${q.x+9}" y="${q.y+16}">${escapeHTML(node.name)}</text>`+
      `<text class="prov-value" text-anchor="end" x="${q.x+q.w-9}" y="${q.y+16}">`+
      `${escapeHTML(provenanceValue(node.value,'0..1'))}</text>`+
      `<text class="prov-meta" x="${q.x+9}" y="${q.y+31}">${escapeHTML(node.role||'node')}</text></g>`;}).join('');
  box.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" `+
    `aria-label="Live sensor to semantic context to ontology provenance map">`+
    `<text class="prov-column" x="18" y="19">1 · SENSOR / ONBOARD SIGNAL</text>`+
    `<text class="prov-column" x="420" y="19">2 · SEMANTIC + TEMPORAL CONTEXT</text>`+
    `<text class="prov-column" x="895" y="19">3 · ONTOLOGY G_t INPUT</text>`+
    `${paths}${sensorCards}${semanticCards}${ontologyCards}</svg>`;
  const boundary=document.getElementById('g3d-sensor-boundary');
  boundary.textContent=(p.boundary||[]).join(' · ')||'onboard inputs only';
  const legend=document.getElementById('g3d-sensor-legend');
  legend.innerHTML='<b>Blue</b> sensor/inference → context &nbsp; <b>Orange</b> context → ontology '
    +`&nbsp; ${p.format&&p.format.endsWith('-fallback')?'· existing-run compatibility view':''}`;
  const svg=box.querySelector('svg'),nodeEls=[...box.querySelectorAll('.prov-node')];
  const edgeEls=[...box.querySelectorAll('.prov-edge')];
  const focus=entity=>{const sensor=sensors.some(item=>item.id===entity);
    const semantic=semantics.some(item=>item.id===entity),active=new Set([entity]),activeLinks=[];
    for(const link of links){let hit=false;
      if(sensor&&link.source===entity){hit=true;active.add(link.target);}
      if(semantic&&(link.source===entity||link.target===entity)){hit=true;active.add(link.source);active.add(link.target);}
      if(!sensor&&!semantic&&link.target===entity){hit=true;active.add(link.source);}
      if(hit)activeLinks.push(link);}
    if(sensor)for(const link of links)if(active.has(link.source)&&link.stage==='semantic_to_ontology'){
      active.add(link.target);activeLinks.push(link);}
    if(!sensor&&!semantic)for(const link of links)if(active.has(link.target)&&link.stage==='sensor_to_semantic'){
      active.add(link.source);activeLinks.push(link);}
    nodeEls.forEach(el=>el.classList.toggle('dim',!active.has(el.dataset.entity)));
    edgeEls.forEach((el,index)=>{const link=links[index],on=activeLinks.includes(link);
      el.classList.toggle('dim',!on);el.classList.toggle('focus',on);});
    const chosen=nodeEls.find(el=>el.dataset.entity===entity);if(chosen)chosen.classList.add('focus');};
  nodeEls.forEach(el=>{el.addEventListener('pointerenter',()=>focus(el.dataset.entity));});
  svg.addEventListener('pointerleave',()=>{nodeEls.forEach(el=>el.classList.remove('dim','focus'));
    edgeEls.forEach(el=>el.classList.remove('dim','focus'));});
}
function graphSnapshots(state){
  G.state=state;
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
  G.dirty=true;graphLegend();graphAudit();graphSensorMap(state);
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
    G.hover=-1;G.fit=0;G.dirty=true;graphLegend();graphAudit();graphSensorMap(G.state);
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

// ---------------------------------------------------- dense R-GAT flow view
// This complements the compact ontology graph above. It expands the real
// nine-node G_t into the feature channels processed by the two R-GAT layers,
// then into graph readout g_t and the actor/critic consumers. Internal channel
// edges are an architecture view; only published alpha values may claim to be
// attention, and the panel says explicitly when those values are unavailable.
const RF={data:null,arch:{input:14,hidden1:32,hidden2:32,graph:32,heads:1},
  yaw:-0.45,pitch:0.08,zoom:1,auto:true,drag:null,dirty:true,pts:[],hover:null};
function rfArchitecture(state){
  const pipeline=((state.scalars||{}).algorithm_pipeline||{}).chain||[];
  const text=pipeline.flatMap(stage=>stage.lines||[]).join(' · ');
  const input=(text.match(/node 특징\s*(\d+)차원/)||[])[1];
  const arrows=[...text.matchAll(/(\d+)\s*→\s*(\d+)/g)];
  const graph=(text.match(/(?:^|·)\s*(\d+)차원/)||[])[1];
  RF.arch={input:Number(input||14),hidden1:Number(arrows[0]?.[2]||32),
    hidden2:Number(arrows[1]?.[2]||arrows[0]?.[2]||32),
    graph:Number(graph||32),heads:1};
}
function rfRing(count,x,radius,phase,kind,values,names){
  const nodes=[];
  for(let i=0;i<count;i++){
    const angle=phase+6.283185*i/Math.max(count,1);
    const band=count>16?((i%4)-1.5)*0.055:0;
    nodes.push({x:x+band,y:radius*Math.cos(angle),z:radius*Math.sin(angle),
      kind:kind,index:i,value:values?Number(values[i]||0):0.5,
      name:names?names[i]:`${kind} channel ${i+1}`});
  }
  return nodes;
}
function rfScene(){
  const data=RF.data;if(!data||!(data.nodes||[]).length)return null;
  const values=data.nodes.map(n=>Number(n.value||0));
  const names=data.nodes.map(n=>String(n.name));
  const input=rfRing(data.nodes.length,-3.35,1.72,0,'input',values,names);
  const h1=rfRing(RF.arch.hidden1,-1.45,1.82,0.35,'R-GAT 1');
  const h2=rfRing(RF.arch.hidden2,0.45,1.82,1.05,'R-GAT 2');
  const embed=rfRing(RF.arch.graph,2.28,1.32,0.65,'g_t');
  const output=[{x:3.75,y:-0.58,z:0,kind:'output',index:0,value:.8,name:'actor π(a|s)'},
    {x:3.75,y:.58,z:0,kind:'output',index:1,value:.8,name:'critic V(s)'}];
  const all=[...input,...h1,...h2,...embed,...output];
  const offsets={input:0,h1:input.length,h2:input.length+h1.length,
    embed:input.length+h1.length+h2.length,output:input.length+h1.length+h2.length+embed.length};
  const edges=[];
  const real=(data.edges||[]).filter(e=>e.s!==e.d);
  for(const e of real)edges.push({a:e.s,b:e.d,r:e.r,strength:e.a,kind:'ontology'});
  // Route each ontology node into several hidden channels. The relation comes
  // from one of that node's actual outgoing ontology edges; no synthetic alpha
  // is attached when the publisher did not provide one.
  input.forEach((node,i)=>{
    const outgoing=real.filter(e=>e.s===i);
    for(let j=0;j<Math.min(7,h1.length);j++){
      const source=outgoing[j%Math.max(outgoing.length,1)];
      edges.push({a:i,b:offsets.h1+(i*7+j*5)%h1.length,
        r:source?source.r:j%(data.relations||[]).length,
        strength:source?source.a:undefined,kind:'message'});
    }
  });
  h1.forEach((node,i)=>{for(let j=0;j<4;j++)edges.push({
    a:offsets.h1+i,b:offsets.h2+(i*5+j*9)%h2.length,r:j%4,
    strength:undefined,kind:'hidden'});});
  h2.forEach((node,i)=>{for(let j=0;j<3;j++)edges.push({
    a:offsets.h2+i,b:offsets.embed+(i*3+j*11)%embed.length,r:j%4,
    strength:undefined,kind:'readout'});});
  embed.forEach((node,i)=>{for(let j=0;j<2;j++)edges.push({
    a:offsets.embed+i,b:offsets.output+j,r:j,strength:undefined,kind:'consumer'});});
  return {nodes:all,edges:edges,offsets:offsets,counts:[input.length,h1.length,h2.length,
    embed.length,output.length]};
}
function rfProject(node,w,h){
  const cy=Math.cos(RF.yaw),sy=Math.sin(RF.yaw);
  const y=node.y*cy-node.z*sy,z=node.y*sy+node.z*cy;
  const cp=Math.cos(RF.pitch),sp=Math.sin(RF.pitch);
  const x=node.x*cp-y*sp,depth=node.x*sp+y*cp+8.5;
  const f=Math.min(w/9.6,h/5.3)*RF.zoom*8.5/Math.max(depth,2.5);
  return {x:w*.49+x*f,y:h*.54-z*f,d:depth,f:f};
}
function rfNodeColor(node){
  if(node.kind==='input'){
    const source=(RF.data.nodes||[])[node.index]||{};return nodeRGB(source);}
  if(node.kind==='R-GAT 1')return [30,176,239];
  if(node.kind==='R-GAT 2')return [47,213,196];
  if(node.kind==='g_t')return [181,229,80];
  return node.index===0?[255,167,61]:[221,98,231];
}
function rfDraw(now=0){
  const cv=document.getElementById('cv-rgat-flow');if(!cv)return;
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){
    cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  const bg=g.createLinearGradient(0,0,w,h);bg.addColorStop(0,'#111d38');
  bg.addColorStop(.55,'#172747');bg.addColorStop(1,'#0c1630');g.fillStyle=bg;g.fillRect(0,0,w,h);
  // Quiet floating diamonds supply depth without claiming to be graph nodes.
  for(let i=0;i<18;i++){
    const x=(i*137%997)/997*w,y=(i*83%521)/521*h+Math.sin(now*.0003+i)*7;
    const size=3+(i%5);g.save();g.translate(x,y);g.rotate(Math.PI/4+now*.00005*(i%3));
    g.fillStyle=`rgba(${45+i*9%150},${70+i*17%130},${150+i*23%100},.10)`;
    g.fillRect(-size,-size,size*2,size*2);g.restore();
  }
  const scene=rfScene();RF.pts=[];
  if(!scene){g.fillStyle='#9eb0d0';g.font='12px sans-serif';
    g.fillText('제안 arm의 ontology graph를 기다리는 중',16,h/2);return;}
  const P=scene.nodes.map(n=>rfProject(n,w,h));
  // Stage halos and titles make the dense cloud readable as a computation.
  const stages=[['ONTOLOGY G_t',0],['R-GAT 14→32',scene.offsets.h1],
    ['R-GAT 32→32 + residual',scene.offsets.h2],['READOUT g_t',scene.offsets.embed],
    ['PPO',scene.offsets.output]];
  g.textAlign='center';g.font='600 10px ui-sans-serif,system-ui,sans-serif';
  for(const [label,index] of stages){g.fillStyle='rgba(185,207,244,.82)';
    g.fillText(label,P[index].x,18);}
  g.globalCompositeOperation='lighter';
  const ordered=scene.edges.map((edge,index)=>({edge,index,
    d:(P[edge.a].d+P[edge.b].d)/2})).sort((a,b)=>b.d-a.d);
  for(const item of ordered){
    const e=item.edge,A=P[e.a],B=P[e.b];
    const relation=REL_COLORS[(e.r||0)%REL_COLORS.length];
    const rgb=hexRGB(relation);const hasAlpha=Number.isFinite(Number(e.strength));
    const alpha=hasAlpha?clamp01(Number(e.strength)):0.32;
    const base=e.kind==='ontology'?.28:e.kind==='consumer'?.18:.055;
    g.strokeStyle=rgba(rgb,base*(.45+.9*alpha));
    g.lineWidth=e.kind==='ontology'?1.1+2.5*alpha:.42+.75*alpha;
    g.beginPath();g.moveTo(A.x,A.y);
    const bend=(B.x-A.x)*.45;g.bezierCurveTo(A.x+bend,A.y,B.x-bend,B.y,B.x,B.y);g.stroke();
    // A sparse moving pulse makes message direction visible without turning
    // every one of the hundreds of architecture edges into visual noise.
    if(item.index%11===0){const t=(now*.00018+item.index*.071)%1;
      const mt=1-t,px=mt*mt*mt*A.x+3*mt*mt*t*(A.x+bend)+
        3*mt*t*t*(B.x-bend)+t*t*t*B.x;
      const py=mt*mt*mt*A.y+3*mt*mt*t*A.y+3*mt*t*t*B.y+t*t*t*B.y;
      g.fillStyle=rgba([170,235,255],.58);g.beginPath();g.arc(px,py,1.6,0,6.2832);g.fill();}
  }
  g.globalCompositeOperation='source-over';
  const nodeOrder=scene.nodes.map((node,index)=>({node,index,d:P[index].d})).sort((a,b)=>b.d-a.d);
  for(const item of nodeOrder){
    const n=item.node,p=P[item.index],rgb=rfNodeColor(n);
    const r=n.kind==='input'?5.2+5*clamp01(n.value):n.kind==='output'?10:3.2;
    const hovered=RF.hover===item.index;
    const grad=g.createRadialGradient(p.x-r*.3,p.y-r*.35,.2,p.x,p.y,r);
    grad.addColorStop(0,rgba([255,255,255],.95));grad.addColorStop(.28,rgba(rgb,.95));
    grad.addColorStop(1,rgba(rgb,.38));g.fillStyle=grad;g.beginPath();g.arc(p.x,p.y,r,0,6.2832);g.fill();
    if(hovered){g.strokeStyle='#fff';g.lineWidth=1.5;g.stroke();}
    RF.pts.push({x:p.x,y:p.y,r:r+5,index:item.index,node:n});
  }
  g.textAlign='left';
  // Input labels are the live ontology channels; internal channel labels are
  // intentionally omitted because their vectors are not published per unit.
  g.font='9.5px ui-sans-serif,system-ui,sans-serif';
  scene.nodes.slice(0,scene.counts[0]).forEach((n,i)=>{
    const p=P[i];g.fillStyle='rgba(226,238,255,.9)';
    g.fillText(`${n.name} ${n.value.toFixed(2)}`,p.x+9,p.y+3);});
  for(const i of [0,1]){const index=scene.offsets.output+i,p=P[index];
    g.fillStyle='#f4f7ff';g.font='600 10px ui-sans-serif,system-ui,sans-serif';
    g.fillText(scene.nodes[index].name,p.x+13,p.y+3);}
}
function rfSnapshot(state){
  rfArchitecture(state);
  const methods=new Set(ontologyArms().map(arm=>arm.method));
  const candidates=Object.values(state.graphs||{}).filter(graph=>
    graph&&methods.has(String(graph.method||''))&&(graph.nodes||[]).length);
  candidates.sort((a,b)=>Number(b.time||0)-Number(a.time||0));
  RF.data=candidates[0]||null;RF.dirty=true;
  const source=document.getElementById('rf-source');
  source.textContent=RF.data?(RF.data.source||RF.data.method||'ontology graph'):
    '제안 arm graph 대기';
  const attention=document.getElementById('rf-attention');
  attention.textContent=RF.data&&RF.data.attention?'live attention α':'attention 미발행 · 구조 표시';
  attention.className='rf-pill '+(RF.data&&RF.data.attention?'live':'wait');
  const stats=document.getElementById('rf-stats');
  const values=[
    [RF.data?(RF.data.nodes||[]).length:'—','ontology nodes'],
    [RF.data?(RF.data.relations||[]).length:'—','relation types'],
    [RF.data?(RF.data.edges||[]).length:'—','declared edges'],
    [`${RF.arch.input}→${RF.arch.hidden1}→${RF.arch.hidden2}`,'R-GAT channels'],
    [RF.arch.graph,'graph state g_t'],['2','actor / critic encoders']];
  stats.innerHTML=values.map(value=>`<div class="rf-stat"><b>${value[0]}</b>`+
    `<span>${value[1]}</span></div>`).join('');
  const legend=document.getElementById('rf-legend');
  legend.innerHTML=(RF.data?.relations||[]).map((relation,index)=>
    `<span><i style="background:${REL_COLORS[index%REL_COLORS.length]}"></i>`+
    `${escapeHTML(relation.name)}</span>`).join('')+
    '<span>● 크기/색 = 현재 ontology activation</span>';
}
function rfTip(ev){
  const tip=document.getElementById('rf-tip');
  if(RF.hover===null){tip.style.display='none';return;}
  const hit=RF.pts.find(point=>point.index===RF.hover);if(!hit)return;
  const node=hit.node;tip.innerHTML=`<b>${escapeHTML(node.name)}</b><br>`+
    `${escapeHTML(node.kind)} · channel ${node.index+1}`+
    (node.kind==='input'?`<br>live activation ${node.value.toFixed(3)}`:'');
  const card=document.getElementById('card-rgat_neural_flow').getBoundingClientRect();
  tip.style.display='block';tip.style.left=Math.min(ev.clientX-card.left+12,
    card.width-tip.offsetWidth-8)+'px';tip.style.top=(ev.clientY-card.top+12)+'px';
}
function rfBind(){
  const cv=document.getElementById('cv-rgat-flow');if(!cv)return;
  cv.addEventListener('pointerdown',event=>{RF.drag={x:event.clientX,y:event.clientY};
    RF.auto=false;cv.classList.add('drag');cv.setPointerCapture(event.pointerId);});
  cv.addEventListener('pointermove',event=>{
    if(RF.drag){RF.yaw+=(event.clientX-RF.drag.x)*.008;
      RF.pitch=Math.max(-.5,Math.min(.5,RF.pitch+(event.clientY-RF.drag.y)*.003));
      RF.drag={x:event.clientX,y:event.clientY};RF.dirty=true;return;}
    const rect=cv.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
    let best=null,distance=Infinity;for(const point of RF.pts){const d=Math.hypot(point.x-x,point.y-y);
      if(d<point.r&&d<distance){distance=d;best=point.index;}}
    RF.hover=best;rfTip(event);RF.dirty=true;});
  const release=()=>{RF.drag=null;cv.classList.remove('drag');};
  cv.addEventListener('pointerup',release);cv.addEventListener('pointercancel',release);
  cv.addEventListener('pointerleave',()=>{release();RF.hover=null;
    document.getElementById('rf-tip').style.display='none';});
  cv.addEventListener('wheel',event=>{event.preventDefault();RF.zoom=Math.max(.55,
    Math.min(2.4,RF.zoom*(event.deltaY<0?1.1:.91)));RF.dirty=true;},{passive:false});
  (function frame(now){if(RF.auto&&!RF.drag)RF.yaw+=.0012;
    rfDraw(now||0);requestAnimationFrame(frame);})(0);
}
rfBind();

function applyProfile(state){
  const profile='benchmark';
  document.body.dataset.profile=profile;
  document.getElementById('page-title').textContent=
    `${ARMS.length}개 arm 비교 · ${ARMS.map(a=>a.label).join(' / ')}`;
  const hasGraph=Boolean(state.graph)||Object.keys(state.graphs||{}).length>0;
  for(const c of CARDS){
    const el=document.getElementById('card-'+c.id);
    // A card that describes a mechanism this run does not use is hidden
    // rather than drawn empty: an FOV-reward panel on a graph-state run would
    // read as a broken measurement instead of an absent one.
    el.hidden=Boolean(c.view&&c.view!=='common'&&c.view!==profile)
      ||((c.kind==='graph'||c.kind==='rgatflow')&&!hasGraph)
      ||!hasRole(c.requiresRole);
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
let lastRevision=-1,lastAt=0,hashScrolled=false;
async function tick(){
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),15000);
  try{
    const r=await fetch('api/state',{cache:'no-store',signal:controller.signal});
    if(!r.ok)throw new Error(`telemetry HTTP ${r.status}`);
    const state=await r.json();
    if(!state.stage)throw new Error(state.error||'telemetry state is incomplete');
    const upstream=r.headers.get('X-Dashboard-Upstream');
    document.getElementById('stage').textContent=
      upstream==='stale'?'reconnecting':state.stage.name;
    document.getElementById('detail').textContent=upstream==='stale'
      ?`last normal snapshot · ${state.stage.name}${state.stage.detail?' · '+state.stage.detail:''}`
      :(state.stage.detail||'');
    if(state.revision!==lastRevision){
      lastRevision=state.revision;lastAt=Date.now();
      syncArms(state);
      const profile=applyProfile(state);
      if(!hashScrolled&&location.hash){
        const target=document.querySelector(location.hash);
        if(target&&!target.hidden){target.scrollIntoView();hashScrolled=true;}}
      tiles(state);
      phasePanel(state);
      runPipelinePanel(state);
      collectionPanel(state);
      fovStatusPanel(state);
      algoPipelinePanel(state);
      mdpPanel(state);
      if(profile==='benchmark'){pairPanel(state);teacherPanel(state);drawTrajectories(state);}
      for(const c of CARDS){
        if(!hasRole(c.requiresRole))continue;
        if(c.id==='tiles'||c.kind==='graph'||c.kind==='rgatflow'||c.kind==='contract'||c.kind==='pairs'||
           c.kind==='phase'||c.kind==='heading'||c.kind==='runpipe'||
           c.kind==='fovstatus'||c.kind==='teacher'||c.kind==='trajectories'||
           c.kind==='algopipe'||c.kind==='mdp'||c.kind==='collection'||
           c.kind==='pairplots'||
           (c.view&&c.view!=='common'&&c.view!==profile))continue;
        c.kind==='evalbars'?drawEvaluationBars(c,state):draw(c,state);
      }
      for(const c of CARDS.filter(item=>item.kind==='pairplots'))drawPairPlots(c,state);
      benchmarkPanel(state);
      graphSnapshots(state);
      rfSnapshot(state);
    }
  }catch(e){document.getElementById('stage').textContent='disconnected';}
  finally{clearTimeout(timeout);setTimeout(tick,1500);}
  const age=lastAt?Math.round((Date.now()-lastAt)/1000):0;
  document.getElementById('age').textContent=lastAt?`updated ${age}s ago`:'';
}
tick();
window.addEventListener('resize',()=>{lastRevision=-1;G.dirty=true;RF.dirty=true;});
</script></body></html>
"""


class _SerializedState:
    """Keep one encoded state body for all dashboard HTTP workers."""

    def __init__(self, store: LiveStore):
        self.store = store
        self._lock = threading.Lock()
        self._revision = -1
        self._body = b""

    def body(self) -> bytes:
        revision = self.store.revision
        if revision == self._revision:
            return self._body
        with self._lock:
            revision = self.store.revision
            if revision != self._revision:
                snapshot = self.store.snapshot()
                self._body = json.dumps(
                    snapshot, default=_jsonable,
                    separators=(",", ":")).encode("utf-8")
                self._revision = int(snapshot["revision"])
            return self._body


class _Handler(BaseHTTPRequestHandler):
    store: LiveStore = STORE
    encoder: _SerializedState | None = None
    # The browser polls rather than streams. Closing each response prevents a
    # reverse proxy's idle keep-alive from retaining a worker and a reference
    # to a potentially large telemetry body.
    protocol_version = "HTTP/1.0"

    def handle(self) -> None:
        """End a request quietly when the polling browser has gone away.

        Refreshing or closing the dashboard can reset either the response
        write or the next keep-alive read.  Both are normal client lifecycle
        events; letting them escape makes ``socketserver`` print a full
        traceback for every abandoned poll and obscures failures in the run
        itself.
        """
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self) -> None:                          # noqa: N802 - stdlib API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            payload = ((self.encoder or _SerializedState(self.store)).body())
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
        handler = type("BoundHandler", (_Handler,), {
            "store": self.store,
            "encoder": _SerializedState(self.store),
        })
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
