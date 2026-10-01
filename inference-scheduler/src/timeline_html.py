"""
The execution timeline as one self-contained HTML file, in the style of a
GPU profiler (NVIDIA Nsight Systems): one row per lane — the CPU and every
kernel — with the simulated schedule of ``src/codegen/timing.simulate`` drawn
on a time axis.

  data      ``timeline_entry()`` turns a simulation into a compact dict:
            lanes (spans as flat [t0, t1, node, kind] lists), a node table
            (name, op, class, lane, µs, price source exact / model /
            unpriced, kernel calls, output shape, predecessors, inputs as
            [shape, tensor, is weight])
  render    ``render_html()`` embeds the entries as JSON in a page that draws
            them on a canvas: it binary-searches the visible window of each
            lane and merges spans narrower than a pixel into one bar, so a
            graph of any size scrolls and zooms at full speed

Viewer: wheel = zoom at the cursor, drag = pan, shift+wheel / trackpad =
pan, W/S zoom, A/D pan, F fit, double-click a span = zoom to it, drag on the
ruler = measure a range (Z zooms to it), click = select (its dependency
arrows, details, predecessors / successors), "/" = search, Esc = clear.
The overview strip above shows the whole run and the visible window.

Written by ``inference_scheduler.py --plan / --plan-report`` (timeline.html
next to report.md) and ``perf_calibrate.py simulate --html FILE``.
"""

from __future__ import annotations

import html
import json
import time
from typing import Dict, Iterable, List, Optional

from .codegen.timing import kernel_duration_fn, simulate

LANE_ORDER = ("CPU", "KERNEL_CONV", "KERNEL_MATMUL", "KERNEL_VECTOROP", "KERNEL_POOL")
LANE_LABELS = {"CPU": "CPU (A53)", "KERNEL_CONV": "ConvKernel", "KERNEL_MATMUL": "MatmulKernel",
               "KERNEL_VECTOROP": "VectorOPKernel", "KERNEL_POOL": "PoolingKernel"}
KINDS = ("kernel", "host", "issue", "wait", "sync")


def _shape(sn) -> str:
    try:
        return "x".join(str(d) for d in sn.output.shape)
    except Exception:                                      # noqa: BLE001
        return ""


def _inputs(sn) -> list:
    """[[shape, tensor name, 1 if a weight], ...] of the node's inputs."""
    out = []
    for t in getattr(sn, "inputs", None) or []:
        if t is None:
            continue
        try:
            shape = "x".join(str(d) for d in t.shape)
        except Exception:                                  # noqa: BLE001
            shape = ""
        out.append([shape, getattr(t, "onnx_name", "") or "", 1 if getattr(t, "is_weight", False) else 0])
    return out


def _price_source(sn, layouts, pm, hm, unpriced: set) -> str:
    """exact (measured on the board) / model (a fitted family or kind) /
    mixed / unpriced / "" (no model given)."""
    if sn.index in unpriced:
        return "unpriced"
    try:
        if getattr(type(sn), "kernel_name", "") == "" and hm is not None:
            p = hm.predict(sn)
            return "unpriced" if p is None else p[1]
        if hasattr(sn, "kernel_calls") and pm is not None:
            srcs = set()
            for c in sn.kernel_calls(layouts):
                p = pm.predict(c)
                srcs.add("unpriced" if p is None else p[1])
            if not srcs:
                return ""
            return srcs.pop() if len(srcs) == 1 else "mixed"
    except Exception:                                      # noqa: BLE001 — informational only
        return ""
    return ""


def _calls(sn, layouts) -> int:
    try:
        return int(sum(c.count for c in sn.kernel_calls(layouts))) if hasattr(sn, "kernel_calls") else 0
    except Exception:                                      # noqa: BLE001
        return 0


def timeline_entry(name: str, cg, tl=None, pm=None, hm=None, measured_us: Optional[float] = None,
                   keys_of=None, note: str = "") -> dict:
    """One entry of the page: the simulation ``tl`` of ``cg`` (run here from
    ``pm`` / ``hm`` when not given), its lanes, node table and dependencies."""
    graph = cg._graph
    if tl is None:
        tl = simulate(cg, kernel_duration_fn(pm, cg._layouts, keys_of=keys_of), hm.us if hm else (lambda sn: None))
    by = {sn.index: sn for sn in graph.nodes}
    used = sorted({s[1] for s in tl.spans})
    ref = {idx: k for k, idx in enumerate(used)}
    unpriced = set(tl.unpriced)

    preds: Dict[int, List[int]] = {}
    try:
        from .schedule import Dag
        dag = Dag.from_graph(graph)

        def drawn_preds(idx, seen):                        # through nodes without spans (views)
            out = []
            for p in dag.predecessors(idx):
                if p in seen:
                    continue
                seen.add(p)
                out += [p] if p in ref else drawn_preds(p, seen)
            return out
        preds = {idx: sorted({ref[p] for p in drawn_preds(idx, set())}) for idx in used}
    except Exception:                                      # noqa: BLE001 — arrows are optional
        preds = {idx: [] for idx in used}

    dur: Dict[int, float] = {}
    for _lane, idx, t0, t1, kind in tl.spans:
        if kind in ("kernel", "host"):
            dur[idx] = dur.get(idx, 0.0) + (t1 - t0)
    nodes = []
    for idx in used:
        sn = by[idx]
        onnx = getattr(sn, "onnx_node", None)
        nodes.append([idx, (onnx.name if onnx is not None and onnx.name else type(sn).__name__),
                      onnx.op_type if onnx is not None else "", type(sn).__name__,
                      cg._kernel_id_of(sn) or "CPU", round(dur.get(idx, 0.0), 3),
                      _price_source(sn, cg._layouts, pm, hm, unpriced), _calls(sn, cg._layouts),
                      _shape(sn), preds.get(idx, []), _inputs(sn)])
    lanes = []
    present = {s[0] for s in tl.spans}
    for lane in [ln for ln in LANE_ORDER if ln in present] + sorted(present - set(LANE_ORDER)):
        spans = sorted((s for s in tl.spans if s[0] == lane), key=lambda s: (s[2], s[3]))
        flat: List[float] = []
        for _, idx, t0, t1, kind in spans:
            flat += [round(t0, 3), round(t1, 3), ref[idx], KINDS.index(kind)]
        busy = sum(s[3] - s[2] for s in spans if s[4] in ("kernel", "host", "issue"))
        lanes.append({"id": lane, "label": LANE_LABELS.get(lane, lane), "s": flat, "busy_us": round(busy, 3)})
    return {"name": name, "total_us": round(tl.total_us, 3), "cpu_us": round(tl.cpu_us, 3),
            "wait_us": round(tl.wait_us, 3), "measured_us": measured_us, "note": note,
            "unpriced": len(unpriced), "lanes": lanes, "nodes": nodes}


def render_html(entries: Iterable[dict], title: str, subtitle: str = "") -> str:
    data = {"title": title, "subtitle": subtitle, "generated": time.strftime("%Y-%m-%d %H:%M"),
            "kinds": list(KINDS), "entries": list(entries)}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", html.escape(title))
                 .replace("__DATA__", blob))


def write_html(path: str, entries: Iterable[dict], title: str, subtitle: str = "") -> str:
    with open(path, "w") as f:
        f.write(render_html(entries, title, subtitle))
    return path


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#16181c;--panel:#1f2227;--panel2:#272b31;--fg:#d9dde4;--dim:#8b929d;--faint:#5c636e;
  --grid:#2a2e35;--border:#33373f;--accent:#76b900;--sel:#ffd54a;--row:#1b1e22;--rowalt:#191b1f;
  --tip:#0f1113ee;--wait:#c0392b;--sync:#d68910;--issue:#4a5868;--cluster:#7f8a99}
:root[data-theme=light]{--bg:#eef0f3;--panel:#ffffff;--panel2:#f3f4f6;--fg:#1c2026;--dim:#5b6470;
  --faint:#9aa1ab;--grid:#dfe3e8;--border:#d3d8de;--accent:#4b8a00;--sel:#d99a00;--row:#ffffff;
  --rowalt:#f7f8fa;--tip:#ffffffee;--wait:#c0392b;--sync:#c27c0e;--issue:#8796a8;--cluster:#8d97a5}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font:12px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;overflow:hidden}
#app{display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto auto;height:100vh}
header{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:8px 12px;background:var(--panel);
  border-bottom:1px solid var(--border)}
header h1{font-size:13px;font-weight:600;margin:0;white-space:nowrap}
header .sub{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40ch}
.stat{color:var(--dim);white-space:nowrap}.stat b{color:var(--fg);font-weight:600}
.spacer{flex:1}
select,input,button{font:inherit;color:var(--fg);background:var(--panel2);border:1px solid var(--border);
  border-radius:4px;padding:3px 7px}
input{width:16ch}button{cursor:pointer}button:hover,select:hover{border-color:var(--faint)}
#legend{display:flex;flex-wrap:wrap;gap:4px 12px;padding:4px 12px;background:var(--panel);
  border-bottom:1px solid var(--border);color:var(--dim)}
.chip{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
#ovwrap{position:relative;height:46px;background:var(--panel);border-bottom:1px solid var(--border)}
#tlwrap{position:relative;min-height:0}
canvas{display:block;width:100%;height:100%}
#tip{position:fixed;pointer-events:none;background:var(--tip);border:1px solid var(--border);border-radius:5px;
  padding:6px 8px;max-width:420px;display:none;z-index:5;box-shadow:0 4px 14px #0006}
#tip .n{font-weight:600;word-break:break-all}#tip .k{color:var(--dim)}
#split{height:7px;cursor:row-resize;background:var(--panel);border-top:1px solid var(--border);
  position:relative;touch-action:none;outline:none}
#split::after{content:"";position:absolute;left:50%;top:2px;width:40px;height:3px;margin-left:-20px;
  border-radius:2px;background:var(--faint)}
#split:hover,#split.drag{background:var(--panel2)}#split:hover::after,#split.drag::after{background:var(--accent)}
#split:focus-visible{box-shadow:inset 0 0 0 2px var(--accent)}
#details{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:0;height:190px;
  background:var(--panel);min-height:0}
#details.collapsed{display:none}
#details>div{overflow:auto;padding:8px 12px;min-width:0}
#details>div+div{border-left:1px solid var(--border)}
#details h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 6px;font-weight:600}
table{border-collapse:collapse;width:100%}td,th{padding:2px 6px;text-align:left;white-space:nowrap}
th{color:var(--dim);font-weight:500}td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
tr.link{cursor:pointer}tr.link:hover td{background:var(--panel2)}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:1px 12px}.kv .k{color:var(--dim)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px}
.pill{display:inline-block;padding:0 6px;border-radius:9px;border:1px solid var(--border);color:var(--dim);cursor:help}
.exact{color:var(--accent);border-color:var(--accent)}.model{color:var(--sync);border-color:var(--sync)}
.unpriced{color:var(--wait);border-color:var(--wait)}
a.nl{color:var(--fg);text-decoration:none;border-bottom:1px dotted var(--faint);cursor:pointer}
#help{position:fixed;inset:0;background:#000a;display:none;align-items:center;justify-content:center;z-index:9}
#help>div{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px 20px;max-width:560px}
#help td{padding:2px 10px 2px 0;white-space:normal}
@media (max-width:700px){#details{grid-template-columns:1fr}#details>div+div{border-left:0;border-top:1px solid var(--border)}
  header .sub{display:none}}
</style>
</head>
<body>
<div id="app">
  <header>
    <h1 id="title"></h1><span class="sub" id="sub"></span>
    <select id="entry" title="entry / phase"></select>
    <span class="stat" id="stats"></span>
    <span class="spacer"></span>
    <input id="search" placeholder="search  ( / )" spellcheck="false">
    <span class="stat" id="matches"></span>
    <button id="zout" title="zoom out (S)" aria-label="zoom out">− Zoom out</button>
    <button id="zin" title="zoom in (W)" aria-label="zoom in">+ Zoom in</button>
    <button id="fit" title="fit the whole run (F)">Fit</button>
    <button id="theme" title="dark / light">Theme</button>
    <button id="helpb" title="keys">?</button>
  </header>
  <div id="legend"></div>
  <div id="ovwrap"><canvas id="ov"></canvas></div>
  <div id="tlwrap"><canvas id="tl"></canvas></div>
  <div id="split" role="separator" aria-orientation="horizontal" aria-label="resize the details panel"
       aria-valuemin="0" tabindex="0" title="drag to resize · double-click to collapse / restore"></div>
  <div id="details"><div id="sel"></div><div id="top"></div></div>
</div>
<div id="tip"></div>
<div id="help"><div>
  <h2 style="margin:0 0 8px;font-size:14px">Navigation</h2>
  <table>
  <tr><td><b>wheel</b></td><td>zoom at the cursor (shift+wheel or a horizontal swipe pans)</td></tr>
  <tr><td><b>drag</b></td><td>pan; on the ruler: measure a range (<b>Z</b> zooms to it)</td></tr>
  <tr><td><b>W / S, A / D</b></td><td>zoom in / out, pan left / right (arrows too); the Zoom in / Zoom out buttons zoom around the selection</td></tr>
  <tr><td><b>F</b></td><td>fit the whole run</td></tr>
  <tr><td><b>click</b></td><td>select a node: its spans, dependency arrows (blue = inputs, gray = consumers) and details</td></tr>
  <tr><td><b>double-click</b></td><td>zoom to the node</td></tr>
  <tr><td><b>/</b>, <b>Enter</b></td><td>search names, ops and classes; Enter jumps to the next match</td></tr>
  <tr><td><b>Esc</b></td><td>clear the selection, the range and the search</td></tr>
  <tr><td><b>overview</b></td><td>the whole run; click or drag to move the window</td></tr>
  <tr><td><b>panel divider</b></td><td>drag to resize the details panel, double-click to collapse / restore it
    (focused: ↑ / ↓ resize, Enter collapses)</td></tr>
  </table>
  <p style="color:var(--dim);margin:10px 0 0">CPU row: <span style="color:var(--wait)">red</span> = blocked on a kernel,
  <span style="color:var(--sync)">amber</span> = a synchronous call, gray-blue = register writes / call issue,
  colored = host ops. Bars narrower than a pixel are merged (hatched gray). Times are the
  performance model's prediction, not a measurement.</p>
</div></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const KIND = {kernel:0, host:1, issue:2, wait:3, sync:4};
const $ = id => document.getElementById(id);
const tlc = $("tl"), ovc = $("ov"), tip = $("tip");
const GUT = 132, RULER = 24;
let ROWH = 30, ROWPAD = 6;                       // rows grow to fill the height (sizeRows)
let E = null, V = {t0: 0, ppu: 1}, W = 0, H = 0, OW = 0, OH = 0, dpr = 1;
let sel = -1, hover = null, range = null, query = "", matches = [], mi = -1, ovImg = null, dirty = true;
let drag = null;

/* ---------- colors ---------- */
const PAL = ["#4e9a06","#3a7bd5","#9b59b6","#e67e22","#16a085","#c0392b","#2e86c1","#b7950b","#8e44ad",
             "#27ae60","#d35400","#1abc9c","#7d6608","#5b2c6f","#1f618d","#a04000","#117a65","#6c3483"];
const FIXED = {ConvNode:"#76b900", MatmulConvNode:"#2fa58a", MatmulNode:"#3a7bd5", LlmAttnConvNode:"#17a2b8",
               PoolNode:"#e67e22", ScheduledNode:"#9b59b6", VectorOPNode:"#9b59b6"};
const colorCache = {};
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function clsColor(c){ if(colorCache[c]) return colorCache[c];
  return colorCache[c] = FIXED[c] || PAL[hash(c) % PAL.length]; }
function cssv(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
let C = {};
function readTheme(){ for (const k of ["bg","panel","panel2","fg","dim","faint","grid","border","accent","sel","row","rowalt","wait","sync","issue","cluster"]) C[k]=cssv("--"+k); }

/* ---------- data prep ---------- */
function prep(e){
  if (e._prepped) return e;
  e.succ = e.nodes.map(() => []);
  e.nodes.forEach((n, i) => n[9].forEach(p => e.succ[p].push(i)));
  e.nodeSpans = e.nodes.map(() => []);
  e.lanes.forEach((L, li) => {
    const n = L.s.length / 4;
    L.t0 = new Float64Array(n); L.t1 = new Float64Array(n); L.nd = new Int32Array(n); L.k = new Uint8Array(n);
    for (let i = 0; i < n; i++) { L.t0[i]=L.s[4*i]; L.t1[i]=L.s[4*i+1]; L.nd[i]=L.s[4*i+2]; L.k[i]=L.s[4*i+3];
      e.nodeSpans[L.nd[i]].push([li, i]); }
    L.n = n;
  });
  e.search = e.nodes.map(n => (n[1] + " " + n[2] + " " + n[3]).toLowerCase());
  e._prepped = true; return e;
}
function spanColor(L, i, nd){
  const k = L.k[i];
  if (k === KIND.wait) return C.wait;
  if (k === KIND.sync) return C.sync;
  if (k === KIND.issue) return C.issue;
  return clsColor(E.nodes[nd][3]);
}

/* ---------- geometry ---------- */
const X = t => GUT + (t - V.t0) * V.ppu;
const T = x => V.t0 + (x - GUT) / V.ppu;
function laneY(li){ return RULER + li * ROWH; }
function sizeRows(){ const n = E ? E.lanes.length : 1;
  ROWH = Math.max(28, Math.min(64, Math.floor((H - RULER - 4) / Math.max(n, 1))));
  ROWPAD = Math.max(6, Math.round(ROWH * 0.22)); }
function fit(){ V.t0 = -E.total_us * 0.01; V.ppu = (W - GUT - 8) / (E.total_us * 1.02 || 1); dirty = true; }
function clampView(){ const minP = (W - GUT) / (E.total_us * 1.5 || 1), maxP = 2000; // up to 2000 px per µs
  V.ppu = Math.min(Math.max(V.ppu, minP), maxP);
  const span = (W - GUT) / V.ppu, lo = -span * 0.5, hi = E.total_us - span * 0.5;
  V.t0 = Math.min(Math.max(V.t0, lo), Math.max(hi, lo)); }
function zoomAt(x, f){ const t = T(x); V.ppu *= f; clampView(); V.t0 = t - (x - GUT) / V.ppu; clampView(); dirty = true; }
function zoomTo(a, b){ const w = Math.max(b - a, 1e-3); V.ppu = (W - GUT) * 0.9 / w; clampView();
  V.t0 = a - (W - GUT) * 0.05 / V.ppu; clampView(); dirty = true; }
function lowerBoundT1(L, t){ let lo = 0, hi = L.n; while (lo < hi){ const m = (lo + hi) >> 1; if (L.t1[m] < t) lo = m + 1; else hi = m; } return lo; }

/* ---------- units ---------- */
function fmt(us){ const a = Math.abs(us); if (a < 1e-9) return "0";
  if (a >= 1e6) return (us/1e6).toFixed(a >= 1e7 ? 2 : 3) + " s";
  if (a >= 1e3) return (us/1e3).toFixed(a >= 1e5 ? 1 : a >= 1e4 ? 2 : 3) + " ms";
  if (a >= 1) return us.toFixed(a >= 100 ? 1 : 2) + " µs";
  return (us*1e3).toFixed(0) + " ns"; }
function niceStep(raw){ const p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }

/* ---------- sizing ---------- */
function resize(){
  dpr = window.devicePixelRatio || 1;
  const r = tlc.parentElement.getBoundingClientRect(), o = ovc.parentElement.getBoundingClientRect();
  const keep = E && W ? {t: T(GUT + (W - GUT) / 2), span: (W - GUT) / V.ppu} : null;
  W = Math.max(200, r.width); H = Math.max(60, r.height); OW = Math.max(200, o.width); OH = o.height;
  for (const [c, w, h] of [[tlc, W, H], [ovc, OW, OH]]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  ovImg = null; sizeRows();
  if (E) { if (keep && isFinite(keep.span) && keep.span > 0) { V.ppu = (W - GUT) / keep.span; V.t0 = keep.t - keep.span / 2; clampView(); } else fit(); }
  dirty = true;
}

/* ---------- drawing ---------- */
const hatch = {};
function hatchPattern(ctx, color){
  if (hatch[color]) return hatch[color];
  const p = document.createElement("canvas"); p.width = p.height = 6;
  const g = p.getContext("2d"); g.strokeStyle = color; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(0, 6); g.lineTo(6, 0); g.moveTo(-3, 3); g.lineTo(3, -3); g.moveTo(3, 9); g.lineTo(9, 3); g.stroke();
  return hatch[color] = ctx.createPattern(p, "repeat");
}
function drawLane(ctx, L, li, y, h, x0lim, x1lim, mapX, labels, outlineNode, isMatch){
  // spans [t0,t1] sorted and non-overlapping: binary search the window, merge sub-pixel spans
  const tA = mapX === X ? T(x0lim) : 0, tB = mapX === X ? T(x1lim) : Infinity;
  let i = mapX === X ? lowerBoundT1(L, tA) : 0;
  let cx0 = -1, cx1 = -1, cn = 0;
  const flush = () => { if (cn) { const w = Math.max(1, cx1 - cx0);
      ctx.fillStyle = C.cluster; ctx.globalAlpha = Math.min(0.9, 0.35 + 0.08 * cn); ctx.fillRect(cx0, y, w, h);
      ctx.globalAlpha = 1; cn = 0; cx0 = -1; } };
  for (; i < L.n; i++) {
    const t0 = L.t0[i], t1 = L.t1[i];
    if (t0 > tB) break;
    let a = mapX(t0), b = mapX(t1);
    if (b < x0lim) continue;
    if (a < x0lim) a = x0lim; if (b > x1lim) b = x1lim;
    const w = b - a, nd = L.nd[i];
    if (w < 1.2 && !(labels && (nd === outlineNode || (isMatch && isMatch[nd])))) {
      if (cn && a - cx1 <= 1) { cx1 = Math.max(cx1, b); cn++; }
      else { flush(); cx0 = a; cx1 = b; cn = 1; }
      continue;
    }
    flush();
    const k = L.k[i], col = spanColor(L, i, nd);
    if (k === KIND.wait || k === KIND.sync) {
      ctx.fillStyle = col; ctx.globalAlpha = 0.25; ctx.fillRect(a, y, Math.max(w, 1), h); ctx.globalAlpha = 1;
      if (w > 3) { ctx.fillStyle = hatchPattern(ctx, col); ctx.fillRect(a, y, w, h); }
    } else { ctx.fillStyle = col; ctx.fillRect(a, y, Math.max(w, 1), h); }
    if (labels) {
      if (w > 4) { ctx.fillStyle = "#0004"; ctx.fillRect(b - 1, y, 1, h); }
      if (w > 36 && k !== KIND.issue) {
        const name = k === KIND.wait ? "wait " + E.nodes[nd][1] : k === KIND.sync ? "sync " + E.nodes[nd][1] : E.nodes[nd][1];
        const max = Math.floor((w - 8) / 6.3);
        ctx.fillStyle = k === KIND.wait || k === KIND.sync ? C.fg : "#fff";
        ctx.fillText(name.length > max ? name.slice(0, Math.max(1, max - 1)) + "…" : name, a + 4, y + h / 2 + 4);
      }
      if (isMatch && isMatch[nd]) { ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.strokeRect(a + 1, y + 1, Math.max(w, 2) - 2, h - 2); }
      if (nd === outlineNode) { ctx.strokeStyle = C.sel; ctx.lineWidth = 2; ctx.strokeRect(a + 1, y - 1, Math.max(w, 3) - 2, h + 2); }
    }
  }
  flush();
}
function drawOverviewImage(){
  const c = document.createElement("canvas"); c.width = Math.round(OW * dpr); c.height = Math.round(OH * dpr);
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  g.fillStyle = C.panel; g.fillRect(0, 0, OW, OH);
  const n = E.lanes.length, lh = Math.max(3, Math.floor((OH - 8) / Math.max(n, 1)) - 1), x0 = 6, x1 = OW - 6;
  const sc = (x1 - x0) / (E.total_us || 1), mapX = t => x0 + t * sc;
  E.lanes.forEach((L, li) => drawLane(g, L, li, 4 + li * (lh + 1), lh, x0, x1, mapX, false, -1, null));
  ovImg = {c, x0, x1, sc};
}
let hashTimer = 0;
function saveHash(){ clearTimeout(hashTimer); hashTimer = setTimeout(() => {
  const p = new URLSearchParams(); p.set("e", E._k); p.set("t", T(GUT).toFixed(2) + "," + T(W).toFixed(2));
  if (sel >= 0) p.set("n", E.nodes[sel][1]);
  try { history.replaceState(null, "", "#" + p.toString()); } catch (e) {} }, 250); }
function loadHash(){
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.has("e") && +p.get("e") < D.entries.length) { $("entry").value = +p.get("e"); setEntry(+p.get("e")); }
  if (p.has("t")) { const [a, b] = p.get("t").split(",").map(Number); if (isFinite(a) && isFinite(b) && b > a) {
    V.ppu = (W - GUT) / (b - a); V.t0 = a; clampView(); dirty = true; } }
  if (p.has("n")) { const i = E.nodes.findIndex(n => n[1] === p.get("n")); if (i >= 0) select(i, !p.has("t")); }
}
function render(){
  if (!dirty || !E) return; dirty = false; saveHash();
  const ctx = tlc.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  ctx.font = "11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
  // rows
  E.lanes.forEach((L, li) => { const y = laneY(li); ctx.fillStyle = li % 2 ? C.rowalt : C.row; ctx.fillRect(0, y, W, ROWH); });
  // grid + ruler
  const step = niceStep(110 / V.ppu), tStart = Math.floor(T(GUT) / step) * step;
  ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, RULER);
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.dim;
  for (let t = tStart; X(t) < W; t += step) { const x = Math.round(X(t)) + 0.5; if (x < GUT) continue;
    ctx.beginPath(); ctx.moveTo(x, RULER - 6); ctx.lineTo(x, H); ctx.stroke(); ctx.fillText(fmt(t), x + 3, 15); }
  // minor ticks
  ctx.strokeStyle = C.border;
  for (let t = tStart; X(t) < W; t += step / 5) { const x = Math.round(X(t)) + 0.5; if (x < GUT) continue;
    ctx.beginPath(); ctx.moveTo(x, RULER - 3); ctx.lineTo(x, RULER); ctx.stroke(); }
  // run end
  const xe = X(E.total_us); if (xe > GUT && xe < W) { ctx.strokeStyle = C.accent; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(xe + .5, RULER); ctx.lineTo(xe + .5, laneY(E.lanes.length)); ctx.stroke(); ctx.setLineDash([]); }
  // spans
  const isMatch = matches.length ? E._matchSet : null;
  ctx.save(); ctx.beginPath(); ctx.rect(GUT, RULER, W - GUT, H - RULER); ctx.clip();
  E.lanes.forEach((L, li) => drawLane(ctx, L, li, laneY(li) + ROWPAD / 2, ROWH - ROWPAD, GUT, W, X, true, sel, isMatch));
  // range
  if (range) { const a = X(Math.min(range[0], range[1])), b = X(Math.max(range[0], range[1]));
    ctx.fillStyle = C.accent; ctx.globalAlpha = 0.12; ctx.fillRect(a, RULER, b - a, H); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.accent; ctx.strokeRect(a + .5, RULER, b - a, H); }
  // dependency arrows of the selection
  if (sel >= 0) drawArrows(ctx);
  ctx.restore();
  // range label on the ruler
  if (range) { const a = X(Math.min(range[0], range[1])), b = X(Math.max(range[0], range[1]));
    const s = fmt(Math.abs(range[1] - range[0])); ctx.fillStyle = C.accent;
    const tw = ctx.measureText(s).width + 10, cx = Math.max(GUT, Math.min(W - tw, (a + b) / 2 - tw / 2));
    ctx.fillRect(cx, 2, tw, RULER - 5); ctx.fillStyle = "#fff"; ctx.fillText(s, cx + 5, 15); }
  // gutter with lane labels
  ctx.fillStyle = C.panel; ctx.fillRect(0, 0, GUT, H);
  ctx.strokeStyle = C.border; ctx.beginPath(); ctx.moveTo(GUT - .5, 0); ctx.lineTo(GUT - .5, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, RULER - .5); ctx.lineTo(W, RULER - .5); ctx.stroke();
  E.lanes.forEach((L, li) => { const y = laneY(li);
    ctx.fillStyle = C.fg; ctx.font = "600 11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
    ctx.fillText(L.label, 10, y + ROWH / 2 - 2);
    ctx.fillStyle = C.dim; ctx.font = "10px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
    ctx.fillText(fmt(L.busy_us) + " · " + (100 * L.busy_us / (E.total_us || 1)).toFixed(0) + " %", 10, y + ROWH / 2 + 11);
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(0, y + ROWH - .5); ctx.lineTo(W, y + ROWH - .5); ctx.stroke(); });
  ctx.fillStyle = C.dim; ctx.font = "10px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
  ctx.fillText("predicted", 10, 15);
  // overview
  const o = ovc.getContext("2d"); o.setTransform(1, 0, 0, 1, 0, 0);
  if (!ovImg) drawOverviewImage();
  o.drawImage(ovImg.c, 0, 0); o.setTransform(dpr, 0, 0, dpr, 0, 0);
  const va = ovImg.x0 + Math.max(0, T(GUT)) * ovImg.sc, vb = ovImg.x0 + Math.min(E.total_us, T(W)) * ovImg.sc;
  o.fillStyle = C.bg; o.globalAlpha = 0.55; o.fillRect(0, 0, Math.max(0, va), OH); o.fillRect(vb, 0, OW - vb, OH); o.globalAlpha = 1;
  o.strokeStyle = C.accent; o.lineWidth = 1.5; o.strokeRect(va, 1, Math.max(2, vb - va), OH - 2);
}
function nodeSpan(nd){ // the node's main span: kernel / host, else the first
  const sp = E.nodeSpans[nd]; let best = sp[0];
  for (const s of sp) { const k = E.lanes[s[0]].k[s[1]]; if (k === KIND.kernel || k === KIND.host) { best = s; break; } }
  return best; }
function drawArrows(ctx){
  const [li, i] = nodeSpan(sel), L = E.lanes[li], ty = laneY(li) + ROWH / 2, tx = X(L.t0[i]);
  const arrow = (fx, fy, tx2, ty2, col) => { ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
    const mx = Math.max(Math.abs(tx2 - fx) * 0.5, 30);
    ctx.beginPath(); ctx.moveTo(fx, fy); ctx.bezierCurveTo(fx + mx, fy, tx2 - mx, ty2, tx2, ty2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx2, ty2); ctx.lineTo(tx2 - 7, ty2 - 4); ctx.lineTo(tx2 - 7, ty2 + 4); ctx.fill(); };
  for (const p of E.nodes[sel][9]) { const [pl, pi] = nodeSpan(p), P = E.lanes[pl];
    arrow(X(P.t1[pi]), laneY(pl) + ROWH / 2, tx, ty, "#4aa3ff"); }
  const L2 = E.lanes[li], ex = X(L2.t1[i]);
  for (const s of E.succ[sel]) { const [sl, si] = nodeSpan(s), S = E.lanes[sl];
    arrow(ex, ty, X(S.t0[si]), laneY(sl) + ROWH / 2, C.faint); }
}

/* ---------- hit testing ---------- */
function hit(x, y){
  if (x < GUT || y < RULER) return null;
  const li = Math.floor((y - RULER) / ROWH); if (li < 0 || li >= E.lanes.length) return null;
  const L = E.lanes[li], t = T(x), tol = 3 / V.ppu;
  let i = lowerBoundT1(L, t - tol), best = null, bd = Infinity;
  for (; i < L.n && L.t0[i] <= t + tol; i++) {
    const d = t < L.t0[i] ? L.t0[i] - t : t > L.t1[i] ? t - L.t1[i] : 0;
    if (d < bd) { bd = d; best = {li, i, nd: L.nd[i]}; }
  }
  return best;
}

/* ---------- panels ---------- */
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
const SRC_HINT = {
  exact: "exact: measured on the board. This exact kernel call (same registers) or host op (same op and "
       + "shapes) was timed when the performance model was calibrated, so the time is a measurement.",
  model: "model: estimated. This exact call was never measured; a model fitted to similar measured calls "
       + "(the kernel family or the host-op kind) predicts it, so expect a larger error.",
  mixed: "mixed: some of the node's kernel calls were measured on the board, the others are estimated by a fitted model.",
  unpriced: "unpriced: outside the performance model (no measurement, no fitted model covers it); counted as 0 in the prediction."};
function srcPill(s){ return s ? `<span class="pill ${esc(s)}" title="${esc(SRC_HINT[s] || s)}">${esc(s)}</span>` : ""; }
function nodeLink(i){ const n = E.nodes[i]; return `<a class="nl" data-n="${i}">${esc(n[1])}</a>`; }
function showSel(){
  const box = $("sel");
  if (sel < 0) { box.innerHTML = `<h2>Selection</h2><div style="color:var(--dim)">Click a span to see the node, its price and its dependencies.
    Hover for a quick look; double-click to zoom to it.</div>`; return; }
  const n = E.nodes[sel], sp = E.nodeSpans[sel].map(([li, i]) => E.lanes[li]);
  const parts = E.nodeSpans[sel].map(([li, i]) => { const L = E.lanes[li];
    return `${esc(L.label)} ${D.kinds[L.k[i]]} ${fmt(L.t0[i])} → ${fmt(L.t1[i])} (${fmt(L.t1[i] - L.t0[i])})`; });
  box.innerHTML = `<h2>Selection</h2><div class="kv">
    <span class="k">node</span><span class="mono">${esc(n[1])}</span>
    <span class="k">op / class</span><span>${esc(n[2] || "—")} · ${esc(n[3])}</span>
    <span class="k">lane</span><span>${esc(n[4])}</span>
    <span class="k">duration</span><span><b>${fmt(n[5])}</b> (${(100 * n[5] / (E.total_us || 1)).toFixed(2)} % of the run) ${srcPill(n[6])}</span>
    ${n[7] ? `<span class="k">kernel calls</span><span>${n[7]}</span>` : ""}
    ${(n[10] || []).length ? `<span class="k">inputs</span><span>${n[10].map(([sh, nm, w]) =>
      `<span class="mono">${esc(sh || "?")}</span> <span class="k">${w ? "weight · " : ""}${esc(nm)}</span>`).join("<br>")}</span>` : ""}
    ${n[8] ? `<span class="k">output</span><span class="mono">${esc(n[8])}</span>` : ""}
    <span class="k">spans</span><span>${parts.map(esc).join("<br>")}</span>
    <span class="k">inputs from</span><span>${n[9].length ? n[9].map(nodeLink).join(", ") : "—"}</span>
    <span class="k">consumed by</span><span>${E.succ[sel].length ? E.succ[sel].map(nodeLink).join(", ") : "—"}</span>
  </div>`;
}
function showTop(){
  const rows = E.nodes.map((n, i) => [n[5], i]).filter(r => r[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 40);
  const waits = {}; E.lanes.forEach(L => { for (let i = 0; i < L.n; i++) if (L.k[i] === KIND.wait) {
      const lane = E.nodes[L.nd[i]][4]; waits[lane] = (waits[lane] || 0) + L.t1[i] - L.t0[i]; } });
  const lanes = E.lanes.map(L => `<tr><td>${esc(L.label)}</td><td class="r">${fmt(L.busy_us)}</td>
    <td class="r">${(100 * L.busy_us / (E.total_us || 1)).toFixed(1)} %</td><td class="r">${waits[L.id] ? fmt(waits[L.id]) : ""}</td></tr>`).join("");
  $("top").innerHTML = `<h2>Lanes</h2><table><tr><th>lane</th><th class="r">busy</th><th class="r">of run</th>
    <th class="r">CPU waits on it</th></tr>${lanes}</table>
    <h2 style="margin-top:10px">Longest nodes</h2><table><tr><th>node</th><th>class</th><th class="r">time</th><th class="r">%</th><th title="exact = measured on the board; model = estimated by a fitted model; hover a label for details">price</th></tr>
    ${rows.map(([d, i]) => { const n = E.nodes[i]; return `<tr class="link" data-n="${i}"><td class="mono">${esc(n[1].length > 46 ? n[1].slice(0, 45) + "…" : n[1])}</td>
      <td>${esc(n[3])}</td><td class="r">${fmt(d)}</td><td class="r">${(100 * d / (E.total_us || 1)).toFixed(1)}</td><td>${srcPill(n[6])}</td></tr>`; }).join("")}</table>`;
}
function select(nd, zoom){ sel = nd; showSel(); if (nd >= 0 && zoom) { const [li, i] = nodeSpan(nd), L = E.lanes[li];
    const d = L.t1[i] - L.t0[i]; zoomTo(L.t0[i] - d * 2, L.t1[i] + d * 2); } dirty = true; }
function legend(){
  const seen = new Map(); E.nodes.forEach(n => seen.set(n[3], (seen.get(n[3]) || 0) + 1));
  const chips = [...seen.entries()].sort((a, b) => b[1] - a[1]).map(([c, k]) =>
    `<span class="chip"><span class="sw" style="background:${clsColor(c)}"></span>${esc(c)} <span style="color:var(--faint)">${k}</span></span>`);
  chips.push(`<span class="chip"><span class="sw" style="background:${C.issue}"></span>issue</span>`,
             `<span class="chip"><span class="sw" style="background:${C.wait}"></span>CPU waits</span>`,
             `<span class="chip"><span class="sw" style="background:${C.sync}"></span>sync call</span>`,
             `<span class="chip"><span class="sw" style="background:${C.cluster}"></span>merged (sub-pixel)</span>`);
  $("legend").innerHTML = chips.join("");
}
function headerStats(){
  const m = E.measured_us, err = m ? ((E.total_us - m) / m * 100) : null;
  $("stats").innerHTML = `predicted <b>${fmt(E.total_us)}</b>` + (m ? ` · measured <b>${fmt(m)}</b> (${err >= 0 ? "+" : ""}${err.toFixed(1)} %)` : "")
    + ` · CPU busy ${fmt(E.cpu_us)}, waiting ${fmt(E.wait_us)} · ${E.nodes.length} nodes`
    + (E.unpriced ? ` · <span style="color:var(--wait)">${E.unpriced} unpriced</span>` : "") + (E.note ? ` · ${esc(E.note)}` : "");
}
function setEntry(k){
  E = prep(D.entries[k]); E._k = k; sel = -1; range = null; ovImg = null; sizeRows(); legend(); headerStats(); showSel(); showTop();
  if (query) doSearch(query); else { matches = []; $("matches").textContent = ""; }
  fit();
}
function doSearch(q){
  query = q.trim().toLowerCase(); matches = []; mi = -1; E._matchSet = null;
  if (query) { E._matchSet = new Uint8Array(E.nodes.length);
    E.search.forEach((s, i) => { if (s.includes(query)) { matches.push(i); E._matchSet[i] = 1; } }); }
  matches.sort((a, b) => E.lanes[nodeSpan(a)[0]].t0[nodeSpan(a)[1]] - E.lanes[nodeSpan(b)[0]].t0[nodeSpan(b)[1]]);
  $("matches").textContent = query ? `${matches.length} match${matches.length === 1 ? "" : "es"}` : "";
  dirty = true;
}

/* ---------- events ---------- */
tlc.addEventListener("wheel", ev => { ev.preventDefault(); if (!E) return;
  const r = tlc.getBoundingClientRect(), x = ev.clientX - r.left;
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) { V.t0 += (ev.shiftKey ? ev.deltaY : ev.deltaX) / V.ppu; clampView(); dirty = true; }
  else zoomAt(Math.max(x, GUT), Math.pow(1.0018, -ev.deltaY * (ev.deltaMode ? 30 : 1))); }, {passive: false});
tlc.addEventListener("mousedown", ev => { if (!E) return; const r = tlc.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  drag = {x, y, t0: V.t0, moved: false, ruler: y < RULER && x > GUT}; if (drag.ruler) range = [T(x), T(x)]; });
window.addEventListener("mousemove", ev => { if (!E) return; const r = tlc.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  if (drag) { if (Math.abs(x - drag.x) > 2) drag.moved = true;
    if (drag.ruler) range[1] = T(x); else { V.t0 = drag.t0 - (x - drag.x) / V.ppu; clampView(); }
    dirty = true; tip.style.display = "none"; return; }
  if (ev.target !== tlc) { tip.style.display = "none"; return; }
  const h = hit(x, y); hover = h;
  if (!h) { tip.style.display = "none"; tlc.style.cursor = y < RULER ? "col-resize" : "grab"; return; }
  tlc.style.cursor = "pointer";
  const L = E.lanes[h.li], n = E.nodes[h.nd], k = D.kinds[L.k[h.i]];
  const what = k === "wait" ? `CPU blocked on ${esc(n[4])}` : k === "sync" ? "CPU in a synchronous call" : k === "issue" ? "register writes + start" : "";
  tip.innerHTML = `<div class="n mono">${esc(n[1])}</div><div class="k">${esc(n[2] || n[3])} · ${esc(L.label)} · ${k}${what ? " — " + what : ""}</div>
    <div>${fmt(L.t0[h.i])} → ${fmt(L.t1[h.i])} · <b>${fmt(L.t1[h.i] - L.t0[h.i])}</b> ${srcPill(n[6])}</div>`
    + (n[8] ? `<div class="k mono">${(n[10] || []).length ? "in " + n[10].map(i => esc(i[0] || "?") + (i[2] ? " (weight)" : "")).join(", ") + " " : ""}→ ${esc(n[8])}${n[7] ? " · " + n[7] + " call" + (n[7] > 1 ? "s" : "") : ""}</div>` : "");
  tip.style.display = "block";
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tw - 6) + "px";
  tip.style.top = (ev.clientY + 16 + th > window.innerHeight ? ev.clientY - th - 10 : ev.clientY + 16) + "px"; });
window.addEventListener("mouseup", ev => { if (!drag) return; const d = drag; drag = null;
  if (d.ruler) { if (!d.moved) range = null; dirty = true; return; }
  if (!d.moved) { const r = tlc.getBoundingClientRect(), h = hit(ev.clientX - r.left, ev.clientY - r.top);
    select(h ? h.nd : -1, false); } });
tlc.addEventListener("dblclick", ev => { const r = tlc.getBoundingClientRect(), h = hit(ev.clientX - r.left, ev.clientY - r.top);
  if (h) select(h.nd, true); });
tlc.addEventListener("mouseleave", () => { tip.style.display = "none"; });
let ovDrag = false;
function ovMove(ev){ if (!E || !ovImg) return; const r = ovc.getBoundingClientRect();
  const t = (ev.clientX - r.left - ovImg.x0) / ovImg.sc, span = (W - GUT) / V.ppu; V.t0 = t - span / 2; clampView(); dirty = true; }
ovc.addEventListener("mousedown", ev => { ovDrag = true; ovMove(ev); });
window.addEventListener("mousemove", ev => { if (ovDrag) ovMove(ev); });
window.addEventListener("mouseup", () => { ovDrag = false; });
ovc.addEventListener("wheel", ev => { ev.preventDefault(); zoomAt(GUT + (W - GUT) / 2, Math.pow(1.0018, -ev.deltaY)); }, {passive: false});
document.addEventListener("click", ev => { const a = ev.target.closest("[data-n]"); if (a && E) select(+a.dataset.n, true); });
window.addEventListener("keydown", ev => {
  if (ev.target === $("search")) { if (ev.key === "Enter" && matches.length) { mi = (mi + 1) % matches.length; select(matches[mi], true); }
    if (ev.key === "Escape") { $("search").value = ""; doSearch(""); $("search").blur(); } return; }
  if (!E) return; const c = GUT + (W - GUT) / 2, k = ev.key.toLowerCase();
  if (k === "w" || k === "+" || k === "=") zoomAt(c, 1.5); else if (k === "s" || k === "-") zoomAt(c, 1 / 1.5);
  else if (k === "a" || ev.key === "ArrowLeft") { V.t0 -= (W - GUT) * 0.2 / V.ppu; clampView(); dirty = true; }
  else if (k === "d" || ev.key === "ArrowRight") { V.t0 += (W - GUT) * 0.2 / V.ppu; clampView(); dirty = true; }
  else if (k === "f" || k === "0") fit();
  else if (k === "z" && range) { zoomTo(Math.min(range[0], range[1]), Math.max(range[0], range[1])); }
  else if (k === "/") { ev.preventDefault(); $("search").focus(); }
  else if (k === "escape") { select(-1); range = null; $("search").value = ""; doSearch(""); }
  else if (k === "?") $("help").style.display = "flex"; });
$("search").addEventListener("input", ev => doSearch(ev.target.value));
$("fit").onclick = () => fit();
function zoomButton(f){ if (!E) return;            // around the selection when it is in view, else the middle
  let x = GUT + (W - GUT) / 2;
  if (sel >= 0) { const [li, i] = nodeSpan(sel), L = E.lanes[li], xs = X((L.t0[i] + L.t1[i]) / 2);
    if (xs >= GUT && xs <= W) x = xs; }
  zoomAt(x, f); }
$("zin").onclick = () => zoomButton(1.5);
$("zout").onclick = () => zoomButton(1 / 1.5);
$("helpb").onclick = () => { $("help").style.display = "flex"; };
$("help").onclick = () => { $("help").style.display = "none"; };
$("theme").onclick = () => { const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "light" ? "dark" : "light";
  try { localStorage.setItem("tl-theme", r.dataset.theme); } catch (e) {}
  readTheme(); for (const k in hatch) delete hatch[k]; ovImg = null; if (E) legend(); dirty = true; };
$("entry").onchange = ev => setEntry(+ev.target.value);

/* ---------- the resizable details panel ---------- */
const split = $("split"), det = $("details");
let detH = 190, detOpen = true, splitDrag = null;
const detMax = () => Math.max(80, window.innerHeight - 160);   // leave the timeline room
function setDetails(h, open, save){
  detH = Math.max(60, Math.min(detMax(), Math.round(h)));
  detOpen = open; det.style.height = detH + "px"; det.classList.toggle("collapsed", !open);
  split.setAttribute("aria-valuenow", open ? detH : 0);
  if (save) { try { localStorage.setItem("tl-details", JSON.stringify({h: detH, open})); } catch (e) {} }
}
split.addEventListener("pointerdown", ev => { ev.preventDefault(); try { split.setPointerCapture(ev.pointerId); } catch (e) {}
  splitDrag = {y: ev.clientY, h: detOpen ? det.getBoundingClientRect().height : 0}; split.classList.add("drag"); });
split.addEventListener("pointermove", ev => { if (!splitDrag) return;
  const h = splitDrag.h - (ev.clientY - splitDrag.y);
  if (h < 40) setDetails(detH, false, false); else setDetails(h, true, false); });
split.addEventListener("pointerup", ev => { if (!splitDrag) return; splitDrag = null; split.classList.remove("drag");
  try { split.releasePointerCapture(ev.pointerId); } catch (e) {} setDetails(detH, detOpen, true); });
split.addEventListener("dblclick", () => setDetails(detH, !detOpen, true));
split.addEventListener("keydown", ev => { ev.stopPropagation();
  if (ev.key === "ArrowUp") { ev.preventDefault(); setDetails((detOpen ? detH : 0) + 24, true, true); }
  else if (ev.key === "ArrowDown") { ev.preventDefault(); if (detOpen && detH - 24 < 60) setDetails(detH, false, true); else setDetails(detH - 24, detOpen, true); }
  else if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setDetails(detH, !detOpen, true); } });
window.addEventListener("resize", () => { if (detOpen && detH > detMax()) setDetails(detMax(), true, false); });
try { const s = JSON.parse(localStorage.getItem("tl-details") || "null"); if (s) setDetails(+s.h || 190, s.open !== false, false); } catch (e) {}
window.addEventListener("resize", () => { resize(); });
new ResizeObserver(() => resize()).observe($("tlwrap"));

/* ---------- start ---------- */
try { const t = localStorage.getItem("tl-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
readTheme();
$("title").textContent = D.title; $("sub").textContent = D.subtitle || ""; $("sub").title = D.subtitle || "";
document.title = D.title;
D.entries.forEach((e, k) => { const o = document.createElement("option"); o.value = k;
  o.textContent = `${e.name} — ${fmt(e.total_us)}`; $("entry").appendChild(o); });
if (D.entries.length < 2) $("entry").style.display = "none";
resize();
if (D.entries.length) { setEntry(0); loadHash(); } else { $("stats").textContent = "no entries"; }
(function loop(){ render(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
"""


__all__ = ("timeline_entry", "render_html", "write_html", "LANE_LABELS", "KINDS")
