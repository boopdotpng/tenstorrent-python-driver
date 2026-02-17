"""Profiler web UI — serves an interactive grid visualization of device profiler data."""
import json, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

_data_json = ""

class _Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/api/data":
      self._json(200, _data_json)
    elif self.path == "/" or self.path == "/index.html":
      self._html(200, _HTML)
    else:
      self._json(404, '{"error":"not found"}')

  def _json(self, code, body):
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(body.encode() if isinstance(body, str) else body)

  def _html(self, code, body):
    self.send_response(code)
    self.send_header("Content-Type", "text/html")
    self.end_headers()
    self.wfile.write(body.encode())

  def log_message(self, *_): pass

def serve(data: dict, port: int = 8884):
  global _data_json
  _data_json = json.dumps(data)
  server = HTTPServer(("", port), _Handler)
  print(f"\n  Profiler UI: http://localhost:{port}\n")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print()
  finally:
    server.server_close()

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>blackhole-py profiler</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;color:#ccc;font:13px/1.4 'SF Mono',Menlo,monospace;display:flex;height:100vh;overflow:hidden}
#sidebar{width:180px;min-width:180px;background:#1a1a1a;border-right:1px solid #333;padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
#sidebar h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
#prog-list{display:flex;flex-direction:column;gap:4px;overflow-y:auto;max-height:40vh;flex-shrink:1}
.note{margin-top:8px;color:#9a9a9a;font-size:10px;line-height:1.35}
.dispatch-badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;margin-top:4px}
.dispatch-badge.fast{background:#2a2a4a;color:#88f}
.dispatch-badge.slow{background:#4a3a1a;color:#fa4}
.prog-btn{width:100%;background:#222;border:1px solid #444;color:#ccc;padding:6px 8px;cursor:pointer;border-radius:3px;text-align:left;font:12px/1.3 inherit}
.prog-btn:hover{background:#2a2a2a}
.prog-btn.active{background:#1a3a2a;border-color:#4a8;color:#8f8}
#center{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#top-bar{background:#1a1a1a;border-bottom:1px solid #333;padding:8px 16px;display:flex;align-items:center;gap:16px;font-size:12px;flex-shrink:0}
.stat{color:#888}.stat b{color:#eee}
#grid-wrap{flex:1;padding:16px;overflow:auto;display:flex;align-items:start;justify-content:center}
.cell rect{transition:stroke .1s,stroke-width .1s}
.cell.active rect:hover{stroke:#fff;stroke-width:2}
.cell.active rect.selected{stroke:#4af;stroke-width:2}
.cell text{pointer-events:none;font-family:'SF Mono',Menlo,monospace;font-size:11px;font-weight:600;fill:rgba(0,0,0,.7)}
#dock{background:#1a1a1a;border-top:1px solid #333;height:0;overflow:hidden;transition:height .15s ease;flex-shrink:0}
#dock.open{height:220px;overflow-y:auto}
#dock-inner{padding:10px 16px;display:flex;gap:16px;align-items:start}
#dock-content{flex:1;min-width:0}
.dock-close{border:1px solid #444;background:#222;color:#aaa;border-radius:3px;padding:1px 7px;cursor:pointer;font:12px/1.2 inherit;flex-shrink:0;align-self:start}
.dock-close:hover{background:#2d2d2d;color:#eee}
#dock-tables{display:flex;gap:24px;flex:1;overflow-x:auto}
#dock-tables table:first-child{width:38%}
#dock-tables table:nth-child(2){width:62%}
table.prof{border-collapse:collapse;margin:0}
table.prof th{text-align:left;color:#666;font-weight:normal;padding:2px 10px 4px 0;border-bottom:1px solid #2a2a2a;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
table.prof td{padding:2px 10px 2px 0;font-size:12px;white-space:nowrap}
table.prof tr:hover td{background:#222}
.zone-dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:middle}
.zone-avg{color:#888;font-size:10px;margin-left:4px}
#dock-warn{padding:6px 0 0;font-size:11px;color:#f88}
#rpanel{width:0;overflow:hidden;background:#161616;border-left:1px solid #333;display:flex;flex-direction:column;transition:width .15s ease}
#rpanel.open{width:380px;min-width:380px}
#rpanel-scroll{flex:1;overflow-y:auto;padding:14px}
#source-panel{display:none}
#source-panel h4{color:#888;font-size:10px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;margin-top:10px}
#source-panel h4:first-child{margin-top:0}
pre.src{background:#1e1e1e;padding:0;border-radius:3px;overflow-x:auto;margin-bottom:0;border:1px solid #222}
pre.src code{font-size:11px;line-height:1.35;padding:8px!important;display:block}
.src-line{display:block;margin:0 -8px;padding:0 8px;border-left:3px solid transparent}
.zone-tag{font-size:9px;padding:1px 5px;border-radius:2px;margin-left:8px;font-family:'SF Mono',Menlo,monospace;vertical-align:middle}
</style></head>
<body>
<div id="sidebar"><h3>Programs</h3><div id="dispatch-line"></div><div id="prog-list"></div><div id="limit-note" class="note"></div></div>
<div id="center">
  <div id="top-bar">
    <span class="stat">Cores: <b id="st-cores">-</b></span>
    <span class="stat">Max: <b id="st-max">-</b></span>
    <span class="stat">Min: <b id="st-min">-</b></span>
    <span class="stat">Avg: <b id="st-avg">-</b></span>
  </div>
  <div id="grid-wrap"><svg id="grid-svg"></svg></div>
  <div id="dock"><div id="dock-inner">
    <div id="dock-content"><div id="dock-tables"></div><div id="dock-warn"></div></div>
    <button class="dock-close" id="dock-close-btn">&times;</button>
  </div></div>
</div>
<div id="rpanel"><div id="rpanel-scroll">
  <div id="source-panel"></div>
</div></div>
<script>
const CELL=40, GAP=4;
const ZONE_COLORS = ['#e6a02f','#5b9bd5','#70ad47','#ed7d31','#a855f7','#06b6d4','#f472b6','#84cc16'];
let D, selProg=0, selCore=null;

fetch('/api/data').then(r=>r.json()).then(d=>{ D=d; init() });

function init(){
  buildSidebar();
  d3.select('#dock-close-btn').on('click', clearSelection);
  selectProgram(0);
}

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }
function fmtUs(cycles){ return cycles!=null ? (cycles/D.freq_mhz).toFixed(2)+'us' : '-' }
function fmtCyc(v){ return v!=null ? v.toLocaleString() : '-' }

// --- sidebar ---
function buildSidebar(){
  d3.select('#dispatch-line').html(`<span class="dispatch-badge ${D.dispatch_mode}">${D.dispatch_mode} dispatch</span>`);
  const btns = d3.select('#prog-list').selectAll('button').data(D.programs);
  btns.enter().append('button').attr('class','prog-btn')
    .on('click', (_,d)=> selectProgram(D.programs.indexOf(d)))
    .merge(btns)
    .html(p => {
      const label = p.name&&p.name.trim() ? esc(p.name) : `Program ${p.index}`;
      return `${label}<br><span style="color:#888;font-size:11px">${p.cores.length} cores</span>`;
    });
  d3.select('#limit-note').text(D.dispatch_mode==='fast' ? 'Fast dispatch DRAM profiling currently keeps only the first 32 programs.' : '');
}

function selectProgram(idx){
  selProg=idx; selCore=null;
  d3.selectAll('.prog-btn').classed('active', (_,i)=> i===idx);
  d3.select('#dock').classed('open', false);
  buildGrid();
  updateStats();
  showSources();
}

// --- grid (SVG + d3) ---
function buildGrid(){
  const svg = d3.select('#grid-svg');
  const prog = D.programs[selProg];
  const coreSet = new Set(prog.cores.map(c=>`${c[0]},${c[1]}`));
  const dispSet = new Set(D.dispatch_cores.map(c=>`${c[0]},${c[1]}`));
  const allX = [...new Set(D.grid_x)].sort((a,b)=>a-b);
  const gy = D.grid_y;

  const vals = Object.values(prog.profiles).map(p=>p.total_cycles).filter(v=>v>0);
  const [mn,mx] = vals.length ? d3.extent(vals) : [0,1];
  const color = d3.scaleSequential(d3.interpolateRdYlGn).domain([mx, mn]);

  const w = allX.length*(CELL+GAP)-GAP, h = gy.length*(CELL+GAP)-GAP;
  svg.attr('width',w).attr('height',h);

  const cells = [];
  for (const [yi,y] of gy.entries())
    for (const [xi,x] of allX.entries())
      cells.push({x, y, xi, yi, key:`${x},${y}`});

  const groups = svg.selectAll('g.cell').data(cells, d=>d.key);
  const enter = groups.enter().append('g').attr('class','cell');
  enter.append('rect').attr('width',CELL).attr('height',CELL).attr('rx',3);
  enter.append('title');
  enter.append('text').attr('x',CELL/2).attr('y',CELL/2).attr('dy','.35em').attr('text-anchor','middle');

  const merged = enter.merge(groups);
  merged.attr('transform', d=>`translate(${d.xi*(CELL+GAP)},${d.yi*(CELL+GAP)})`);

  merged.each(function(d) {
    const g = d3.select(this);
    const isCore = coreSet.has(d.key), isDisp = dispSet.has(d.key);
    const p = isCore ? prog.profiles[d.key] : null;
    const hasCycles = p && p.total_cycles > 0;

    g.classed('active', isCore && !isDisp);
    const rect = g.select('rect');
    rect.classed('selected', false);
    if (isDisp) {
      rect.attr('fill','#282828').attr('stroke','#333').attr('stroke-width',1).attr('stroke-dasharray','4,3');
    } else if (!isCore) {
      rect.attr('fill','#1a1a1a').attr('stroke','none').attr('stroke-dasharray',null);
    } else if (hasCycles) {
      rect.attr('fill', color(p.total_cycles)).attr('stroke','none').attr('stroke-dasharray',null);
    } else {
      rect.attr('fill','#2a2a2a').attr('stroke','none').attr('stroke-dasharray',null);
    }

    const usVal = hasCycles ? (p.total_cycles/D.freq_mhz).toFixed(1) : '';
    g.select('text').text(usVal.length<=4 ? usVal : '');
    g.select('title').text(
      isDisp ? `(${d.x},${d.y}) reserved by fast dispatch` :
      !isCore ? `(${d.x},${d.y}) inactive` :
      hasCycles ? `(${d.x},${d.y}) ${usVal}us` :
      `(${d.x},${d.y}) no data`
    );
    g.on('click', isCore && !isDisp ? ()=>selectCore(d.key) : ()=>clearSelection());
  });
  groups.exit().remove();
  svg.on('click', e=>{ if(e.target===svg.node()) clearSelection() });
}

function selectCore(key){
  selCore = key;
  d3.selectAll('.cell rect').classed('selected', false);
  d3.selectAll('.cell').filter(d=>d.key===key).select('rect').classed('selected', true);
  showDock(key);
}

function clearSelection(){
  selCore = null;
  d3.selectAll('.cell rect').classed('selected', false);
  d3.select('#dock').classed('open', false);
}

// --- bottom dock (per-core RISC + zone tables) ---
function sourceOrder(prog){
  // Build zone-name -> order index based on first appearance in sources (reader, compute, writer)
  const ordered = ['reader','compute','writer'];
  const all = [...ordered.filter(k=>prog.sources&&prog.sources[k]), ...Object.keys(prog.sources||{}).filter(k=>!ordered.includes(k))];
  const order = {};
  let idx = 0;
  for (const label of all) {
    const src = prog.sources[label];
    if (!src) continue;
    const re = /DeviceZoneScopedN\s*\(\s*"([^"]+)"/g;
    let m;
    while ((m = re.exec(src)) !== null) {
      if (!(m[1] in order)) order[m[1]] = idx++;
    }
  }
  return order;
}

function getZoneData(key){
  const prog = D.programs[selProg], p = prog.profiles[key];
  if(!p) return [];
  const zones = [];
  for (const r of p.riscs) {
    const starts={}, ends={}, totals={};
    for (const z of r.custom) {
      if(z.type===0) (starts[z.hash]=starts[z.hash]||[]).push(z.ts);
      else if(z.type===1) (ends[z.hash]=ends[z.hash]||[]).push(z.ts);
      else if(z.type===2) totals[z.hash] = (totals[z.hash]||0) + z.ts;
    }
    // Collect all hashes that have any data
    const allHashes = new Set([...Object.keys(starts), ...Object.keys(totals)]);
    for (const hash of allHashes) {
      let dur = 0;
      const s = starts[hash]||[], e = ends[hash]||[];
      if (s.length && e.length) {
        // Zip all start/end pairs and sum (fix: was only using first pair)
        const n = Math.min(s.length, e.length);
        for (let i = 0; i < n; i++) if (e[i] > s[i]) dur += e[i] - s[i];
      }
      // type=2 TOTAL markers: firmware pre-accumulated cycle count
      if (totals[hash]) dur += totals[hash];
      if (!dur) continue;
      const zn = D.zone_names[hash];
      const name = zn ? zn.name : `0x${(+hash).toString(16)}`;
      const existing = zones.find(z=>z.name===name);
      if (existing) existing.dur += dur;
      else zones.push({name, hash, dur, risc: r.name});
    }
  }
  // Sort by appearance order in source (reader → compute → writer)
  const order = sourceOrder(prog);
  zones.sort((a,b) => (order[a.name]??999) - (order[b.name]??999));
  zones.forEach((z,i) => z.color = ZONE_COLORS[i % ZONE_COLORS.length]);
  return zones;
}

function getZoneAvgs(){
  // Compute avg duration per zone name across all cores in the current program
  const prog = D.programs[selProg], sums={}, counts={};
  for (const [key, p] of Object.entries(prog.profiles)) {
    const zones = getZoneData(key);
    for (const z of zones) {
      sums[z.name] = (sums[z.name]||0) + z.dur;
      counts[z.name] = (counts[z.name]||0) + 1;
    }
  }
  const avgs = {};
  for (const name of Object.keys(sums)) avgs[name] = sums[name] / counts[name];
  return avgs;
}

function showDock(key){
  const prog = D.programs[selProg], p = prog.profiles[key];
  if(!p){ d3.select('#dock').classed('open',false); return }

  // RISC table
  let h = `<table class="prof"><tr><th>RISC</th><th>FW</th><th>Kernel</th></tr>`;
  for (const r of p.riscs) {
    const fw = (r.fw_start!=null && r.fw_end!=null) ? r.fw_end-r.fw_start : null;
    const kern = (r.kern_start!=null && r.kern_end!=null) ? r.kern_end-r.kern_start : null;
    h += `<tr><td>${r.name}</td><td>${fmtUs(fw)}</td><td>${fmtUs(kern)}</td></tr>`;
  }
  h += '</table>';

  // Zone table with cross-core avg
  const zones = getZoneData(key);
  if (zones.length) {
    const avgs = getZoneAvgs();
    h += `<table class="prof"><tr><th>Zone</th><th>Time</th><th>Cycles</th></tr>`;
    for (const z of zones) {
      const avg = avgs[z.name];
      const avgHint = avg ? `<span class="zone-avg">avg ${fmtUs(avg)}</span>` : '';
      h += `<tr><td><span class="zone-dot" style="background:${z.color}"></span>${esc(z.name)}</td><td>${fmtUs(z.dur)}${avgHint}</td><td>${fmtCyc(z.dur)}</td></tr>`;
    }
    h += '</table>';
  }
  d3.select('#dock-tables').html(h);

  // Warnings
  const warns = [];
  if (p.dropped) warns.push('Profiler buffer overflow — some markers were dropped. Zone timings may be incomplete or missing.');
  if (!p.done) warns.push('Kernel did not signal completion — profiler data may be partial.');
  d3.select('#dock-warn').html(warns.map(w=>`<div>${w}</div>`).join(''));

  d3.select('#dock').classed('open', true);
}

// --- right panel (source code with scope highlighting) ---
function openPanel(){ d3.select('#rpanel').classed('open',true) }

function getZoneColors(){
  // Stable color assignment from source order (doesn't depend on core selection)
  const prog = D.programs[selProg];
  const order = sourceOrder(prog);
  const colors = {};
  for (const [name, idx] of Object.entries(order))
    colors[name] = ZONE_COLORS[idx % ZONE_COLORS.length];
  return colors;
}

function findZoneScopes(src){
  // Find the scope (start line → end line) for each DeviceZoneScopedN declaration
  const lines = src.split('\n'), scopes = [], pending = [];
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    // Detect zone declaration before counting braces on this line
    const m = lines[i].match(/DeviceZoneScopedN\s*\(\s*"([^"]+)"/);
    if (m) pending.push({name: m[1], line: i, depth});
    // Count braces (simple — doesn't parse strings/comments, good enough for kernels)
    for (const ch of lines[i]) {
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        for (let j = pending.length - 1; j >= 0; j--) {
          if (depth < pending[j].depth) {
            const z = pending.splice(j, 1)[0];
            scopes.push({name: z.name, start: z.line, end: i});
          }
        }
      }
    }
  }
  return scopes;
}

function showSources(){
  const prog = D.programs[selProg];
  const el = d3.select('#source-panel');
  if(!prog.sources || !Object.keys(prog.sources).length){ el.style('display','none'); d3.select('#rpanel').classed('open',false); return }
  const ordered = ['reader','compute','writer'];
  const all = [...ordered.filter(k=>prog.sources[k]), ...Object.keys(prog.sources).filter(k=>!ordered.includes(k))];
  const colors = getZoneColors();

  let h = '';
  for (const label of all)
    h += `<h4>${label}</h4><pre class="src"><code class="language-cpp">${esc(prog.sources[label])}</code></pre>`;
  el.html(h).style('display','block');

  // Syntax highlight
  el.selectAll('code.language-cpp').each(function(){ hljs.highlightElement(this) });

  // Scope highlighting: wrap each line in a span, color zone scopes
  el.selectAll('pre.src').each(function(_, preIdx) {
    const pre = d3.select(this), code = pre.select('code');
    const rawSrc = prog.sources[all[preIdx]];
    const scopes = findZoneScopes(rawSrc);
    const htmlLines = code.html().split('\n');

    const rawLines = rawSrc.split('\n');
    // Build map: declaration line index → zone name for tag rendering
    const declMap = {};
    for (const s of scopes) declMap[s.start] = s.name;
    code.html(htmlLines.map((line, i) => {
      // Find innermost (narrowest) zone scope covering this line
      let scope = null;
      for (const s of scopes)
        if (i >= s.start && i < s.end && (!scope || (s.end - s.start) < (scope.end - scope.start))) scope = s;
      const c = scope ? colors[scope.name] : null;
      const style = c ? `border-left-color:${c};background:${c}18` : '';
      // Replace DeviceZoneScopedN declaration lines with a clean tag
      if (declMap[i]) {
        const zc = colors[declMap[i]] || '#888';
        const tag = `<span class="zone-tag" style="background:${zc}30;color:${zc}">&#9654; ${declMap[i]}</span>`;
        return `<span class="src-line" style="${style}">${tag}</span>`;
      }
      return `<span class="src-line" style="${style}">${line}</span>`;
    }).join(''));
  });
  openPanel();
}

function updateStats(){
  const prog = D.programs[selProg];
  const vals = Object.values(prog.profiles).map(p=>p.total_cycles).filter(v=>v>0);
  const fmt = v=>(v/D.freq_mhz).toFixed(2)+'us';
  d3.select('#st-cores').text(prog.cores.length);
  d3.select('#st-max').text(vals.length ? fmt(d3.max(vals)) : '-');
  d3.select('#st-min').text(vals.length ? fmt(d3.min(vals)) : '-');
  d3.select('#st-avg').text(vals.length ? fmt(d3.mean(vals)) : '-');
}
</script>
</body></html>"""
