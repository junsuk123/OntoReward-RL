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
from typing import Any

from .live import STORE, LiveStore

__all__ = ["Dashboard"]


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ontology-RGAT training</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#6b7280;
--line:#d8dce3;--accent:#1a59bf;--good:#0d8c4d;--warn:#d96619;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1c1f25;--ink:#e8eaee;
--muted:#9aa2af;--line:#2c313a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;
gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:650;letter-spacing:-.01em}
#stage{font-weight:600;color:var(--accent)}
#age{color:var(--muted);font-variant-numeric:tabular-nums;margin-left:auto}
main{padding:16px 20px;display:grid;gap:14px;
grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:12px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);font-weight:650}
canvas{width:100%;height:150px;display:block}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.tile b{display:block;font-size:19px;font-variant-numeric:tabular-nums;font-weight:650}
.tile span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:10px;height:3px;vertical-align:middle;margin-right:4px}
.wide{grid-column:1/-1}
.g3d{position:relative}
.g3d canvas{height:440px;cursor:grab;touch-action:none}
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
.note{font-size:11px;color:var(--muted);margin-top:8px}
</style></head><body>
<header><h1>Ontology-RGAT &middot; Isaac Sim + PX4</h1>
<span id="stage">connecting</span><span id="detail"></span><span id="age"></span></header>
<main id="root"></main>
<script>
const PALETTE=['#1a59bf','#d96619','#0d8c4d','#7333a6','#c0392b','#0e7c86'];
const CARDS=[
 {id:'tiles',title:null},
 {id:'graph3d',kind:'graph',title:'Learned ontology graph (3D, R-GAT attention)'},
 {id:'ppo_return',title:'PPO episode return',series:['ppo_manual','ppo_proposed'],
  x:'episode',y:'return',smooth:20},
 {id:'ppo_success',title:'PPO moving success rate',series:['ppo_manual','ppo_proposed'],
  x:'episode',y:'success',smooth:40,ymin:0,ymax:1},
 {id:'ppo_steps',title:'PPO episode length (steps)',series:['ppo_manual','ppo_proposed'],
  x:'episode',y:'steps',smooth:20},
 {id:'ppo_std',title:'Exploration std',series:['ppo_manual','ppo_proposed'],
  x:'episode',y:'policy_std'},
 {id:'rgat_loss',title:'R-GAT potential loss',series:['rgat'],x:'epoch',
  y:['train_mse','val_mse'],labels:['train','validation']},
 {id:'dataset',title:'Expert dataset success rate',series:['dataset'],
  x:'episode',y:'success',smooth:8,ymin:0,ymax:1},
 {id:'episode_z',title:'Live episode: altitude above deck (m)',series:['episode'],
  x:'t',y:'z'},
 {id:'episode_speed',title:'Live episode: deck and closing speed (m/s)',
  series:['episode'],x:'t',y:['pad_speed','closing_speed'],
  labels:['deck','closing']},
 {id:'episode_energy',title:'Live episode: hover seconds remaining',
  series:['episode'],x:'t',y:'hover_seconds_left'},
 {id:'episode_nav',title:'Live episode: what knows where the lorry is',
  series:['episode'],x:'t',y:['marker_quality','gnss_quality','nav_confidence'],
  labels:['markers','GNSS','combined'],ymin:0,ymax:1},
 {id:'episode_gnss_error',title:'Live episode: error in the pose being flown on (m)',
  series:['episode'],x:'t',y:['estimate_error_m','gnss_sigma_xy'],
  labels:['actual','reported 1-sigma']},
];
const LABELS={ppo_manual:'Manual',ppo_proposed:'Ontology-RGAT',rgat:'R-GAT',
 dataset:'expert',episode:'episode'};
const root=document.getElementById('root');
for(const c of CARDS){
  const el=document.createElement('section');
  el.className='card'+((c.id==='tiles'||c.kind==='graph')?' wide':'')
    +(c.kind==='graph'?' g3d':'');
  el.id='card-'+c.id;
  if(c.id==='tiles'){el.innerHTML='<div class="tiles" id="tiles"></div>';}
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
  const line=css.getPropertyValue('--line').trim();
  const muted=css.getPropertyValue('--muted').trim();
  const lines=[];const yKeys=Array.isArray(card.y)?card.y:[card.y];
  for(const s of card.series){
    const rows=state.series[s]||[];if(!rows.length)continue;
    for(const key of yKeys){
      const xs=[],ys=[];
      for(const r of rows){const yv=r[key];
        if(yv===null||yv===undefined||Number.isNaN(yv))continue;
        xs.push(r[card.x]);ys.push(yv);}
      if(!xs.length)continue;
      const label=(card.labels&&yKeys.length>1)
        ?`${LABELS[s]||s} ${card.labels[yKeys.indexOf(key)]}`
        :(yKeys.length>1?card.labels[yKeys.indexOf(key)]:(LABELS[s]||s));
      lines.push({xs,ys:smooth(ys,card.smooth||1),label});
    }
  }
  const lg=document.getElementById('lg-'+card.id);
  if(!lines.length){g.fillStyle=muted;g.font='12px sans-serif';
    g.fillText('no data yet',10,h/2);lg.innerHTML='';return;}
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(const l of lines){for(let i=0;i<l.xs.length;i++){
    x0=Math.min(x0,l.xs[i]);x1=Math.max(x1,l.xs[i]);
    y0=Math.min(y0,l.ys[i]);y1=Math.max(y1,l.ys[i]);}}
  if(card.ymin!==undefined)y0=card.ymin;if(card.ymax!==undefined)y1=card.ymax;
  if(x1===x0)x1=x0+1;if(y1===y0){y1=y0+1;y0-=1;}
  const pad={l:44,r:8,t:8,b:20};
  const px=v=>pad.l+(v-x0)/(x1-x0)*(w-pad.l-pad.r);
  const py=v=>h-pad.b-(v-y0)/(y1-y0)*(h-pad.t-pad.b);
  g.strokeStyle=line;g.lineWidth=1;g.fillStyle=muted;g.font='10px sans-serif';
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,y=py(v);
    g.beginPath();g.moveTo(pad.l,y);g.lineTo(w-pad.r,y);g.stroke();
    g.fillText(v.toFixed(Math.abs(v)>=100?0:2),4,y+3);}
  g.fillText(String(Math.round(x0)),pad.l,h-6);
  g.textAlign='right';g.fillText(String(Math.round(x1)),w-pad.r,h-6);g.textAlign='left';
  lines.forEach((l,i)=>{g.strokeStyle=PALETTE[i%PALETTE.length];g.lineWidth=1.8;
    g.beginPath();for(let k=0;k<l.xs.length;k++){
      const X=px(l.xs[k]),Y=py(l.ys[k]);k?g.lineTo(X,Y):g.moveTo(X,Y);}g.stroke();});
  lg.innerHTML=lines.map((l,i)=>
    `<span><i style="background:${PALETTE[i%PALETTE.length]}"></i>${l.label}</span>`).join('');
}
function tiles(state){
  const s=state.scalars||{};const out=[];
  const add=(label,value)=>out.push(
    `<div class="tile"><b>${value}</b><span>${label}</span></div>`);
  const last=(name,key)=>{const r=state.series[name];
    return r&&r.length?r[r.length-1][key]:null;};
  add('stage',state.stage.name);
  if(s.dataset_episode)add('dataset episode',s.dataset_episode);
  if(s.rgat_epoch)add('R-GAT epoch',`${s.rgat_epoch} (${s.rgat_device||'?'})`);
  const rv=last('rgat','val_mse');if(rv!==null)add('R-GAT val MSE',rv.toFixed(4));
  for(const arm of ['manual','proposed']){
    const ep=s['ppo_'+arm+'_episode'];if(ep)add('PPO '+arm+' episode',ep);
    const rows=state.series['ppo_'+arm];
    if(rows&&rows.length){const n=Math.min(50,rows.length);
      const w=rows.slice(-n);
      const sr=w.reduce((a,r)=>a+r.success,0)/n;
      add(arm+' success (last '+n+')',(100*sr).toFixed(1)+'%');}
  }
  const st=last('episode','status');if(st)add('episode status',st);
  const z=last('episode','z');if(z!==null)add('altitude',z.toFixed(2)+' m');
  document.getElementById('tiles').innerHTML=out.join('');
}
// ------------------------------------------------------------ 3D ontology
// Hand-rolled: the page has to work with no network, so there is no three.js
// to reach for. Fourteen nodes and forty edges is well inside what a painter's
// algorithm on a 2D canvas can do at 60 Hz, and doing it by hand is what lets
// the depth cue, the attention width and the hover isolation share one pass.
const REL_COLORS=['#c0392b','#0d8c4d','#1a59bf','#8a8f98'];
const G={yaw:-0.55,pitch:0.30,zoom:1,auto:true,floor:0,hover:-1,drag:null,
         data:null,dirty:true,pts:[]};
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
  const d=y2+DIST, f=Math.min(w,h)*2.0/Math.max(d,0.4)*G.zoom;
  return {x:w/2+x*f, y:h/2-z2*f, d:d, f:f};
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
  const P=data.nodes.map(n=>project(n.pos,w,h));
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
    const p=P[i];
    const r=(n.role==='goal'?0.105:0.050+0.048*clamp01(n.value))*p.f;
    G.pts.push({x:p.x,y:p.y,r:Math.max(r,6),i:i});
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
  // Labels last and near-to-far, so the front of the graph stays readable.
  const order=data.nodes.map((n,i)=>i).sort((a,b)=>P[b].d-P[a].d);
  g.textAlign='center';g.font='600 10.5px ui-sans-serif,system-ui,sans-serif';
  for(const i of order){
    const n=data.nodes[i],p=P[i];
    const al=fade(p.d)*(G.hover<0||G.hover===i?1:0.28);
    const r=(n.role==='goal'?0.105:0.050+0.048*clamp01(n.value))*p.f;
    const text=n.role==='goal'?n.name:`${n.name} ${n.value.toFixed(2)}`;
    const ty=p.y-Math.max(r,6)-5;
    // A halo in the card colour: a label over the far side of the ring is
    // otherwise unreadable against the edges crossing behind it.
    g.lineWidth=3;g.lineJoin='round';
    g.strokeStyle=rgba(hexRGB(card),al*0.85);g.strokeText(text,p.x,ty);
    g.fillStyle=rgba(hexRGB(ink),al);g.fillText(text,p.x,ty);
  }
  g.textAlign='left';
}
function graphLegend(){
  const box=document.getElementById('rel-graph3d');if(!box)return;
  const data=G.data;
  if(!data||!data.relations){box.innerHTML='';return;}
  const peak=Math.max(1e-9,...data.relations.map(r=>r.mean||0));
  box.innerHTML=data.relations.map((r,i)=>{
    const c=REL_COLORS[i%REL_COLORS.length];
    const m=r.mean;
    return `<span style="color:${c}">&#9632; ${r.name}</span>`
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

let lastRevision=-1,lastAt=0;
async function tick(){
  try{
    const r=await fetch('api/state',{cache:'no-store'});
    const state=await r.json();
    document.getElementById('stage').textContent=state.stage.name;
    document.getElementById('detail').textContent=state.stage.detail||'';
    if(state.revision!==lastRevision){
      lastRevision=state.revision;lastAt=Date.now();
      tiles(state);
      for(const c of CARDS)if(c.id!=='tiles'&&c.kind!=='graph')draw(c,state);
      const gs=state.graph||null;
      const stamp=gs?`${gs.source||'graph'}${gs.attention?'':' (schema only, '
        +'the R-GAT has not been trained yet)'}`
        +(gs.phi!==undefined?`  \u03a6=${gs.phi.toFixed(3)}`:'')
        :'waiting for a graph';
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
