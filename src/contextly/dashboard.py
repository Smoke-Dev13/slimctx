"""Self-contained HTML dashboard for the Contextly proxy.

Served at ``GET /dashboard``. It polls the existing ``/stats`` and ``/quality``
endpoints from the browser and renders live token-savings, cost-savings, and
quality numbers. The markup is fully inline — no CDN, fonts, or build step — to
keep the proxy offline-first and the page screenshot-ready out of the box.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Contextly — live token savings</title>
<style>
:root{
  --bg:#0a0e14;--sidebar:#0d1117;--card:#111820;--card2:#141c25;
  --line:#1e2730;--fg:#e2eaf3;--muted:#7a8799;--muted2:#4d5a6a;
  --green:#3fb950;--blue:#58a6ff;--warn:#e3b341;--red:#f85149;
  --green-dim:rgba(63,185,80,.08);--blue-dim:rgba(88,166,255,.08);
  --r:14px;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--fg);
     font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;
     font-size:14px;display:flex;min-height:100vh;}
.sidebar{width:220px;min-height:100vh;background:var(--sidebar);
  border-right:1px solid var(--line);display:flex;flex-direction:column;flex-shrink:0;}
.brand{padding:20px 18px 16px;display:flex;align-items:center;gap:10px;
       border-bottom:1px solid var(--line);}
.brand-name{font-size:15px;font-weight:700;letter-spacing:-.3px;}
.brand-name em{font-style:normal;color:var(--green);}
.nav{padding:12px 8px;display:flex;flex-direction:column;gap:2px;flex:1;}
.nav-item{display:flex;align-items:center;gap:10px;padding:8px 10px;
           border-radius:8px;color:var(--muted);cursor:pointer;
           transition:background .15s,color .15s;font-size:13px;font-weight:500;}
.nav-item:hover{background:rgba(255,255,255,.04);color:var(--fg);}
.nav-item.active{background:rgba(63,185,80,.1);color:var(--green);}
.nav-item svg{opacity:.7;flex-shrink:0;}
.nav-item.active svg{opacity:1;}
.sidebar-footer{padding:12px 18px;border-top:1px solid var(--line);}
.status-chip{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--green);
      animation:pulse 2s ease-in-out infinite;flex-shrink:0;}
.sdot.err{background:var(--red);animation:none;}
@keyframes pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(63,185,80,.5);}
  50%{box-shadow:0 0 0 4px rgba(63,185,80,0);}
}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.topbar{display:flex;align-items:center;justify-content:space-between;
        padding:14px 28px;border-bottom:1px solid var(--line);
        background:rgba(10,14,20,.95);position:sticky;top:0;z-index:10;}
.page-title{font-size:14px;font-weight:600;color:var(--fg);}
.topbar-right{display:flex;align-items:center;gap:12px;}
.price-row{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);}
.price-row input{background:transparent;color:var(--fg);border:1px solid var(--line);
                  border-radius:6px;padding:3px 7px;width:70px;font-size:12px;}
.price-row input:focus{outline:2px solid var(--blue);outline-offset:1px;}
.uptime-badge{font-size:11px;color:var(--muted2);background:var(--card);
              border:1px solid var(--line);border-radius:20px;padding:3px 10px;}
.content{padding:24px 28px 48px;overflow-y:auto;}
.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px;}
.hero-card{border-radius:var(--r);padding:20px 22px;position:relative;overflow:hidden;
  border:1px solid var(--line);}
.hero-card.green{background:linear-gradient(135deg,var(--green-dim),var(--card));
                  border-color:rgba(63,185,80,.2);}
.hero-card.blue{background:linear-gradient(135deg,var(--blue-dim),var(--card));
                border-color:rgba(88,166,255,.2);}
.hero-card.neutral{background:var(--card);}
.hero-glow{position:absolute;top:-30px;right:-30px;width:90px;height:90px;
           border-radius:50%;opacity:.15;filter:blur(30px);}
.hero-card.green .hero-glow{background:var(--green);}
.hero-card.blue  .hero-glow{background:var(--blue);}
.hero-icon{width:32px;height:32px;border-radius:8px;display:flex;
           align-items:center;justify-content:center;margin-bottom:14px;}
.hero-card.green .hero-icon{background:rgba(63,185,80,.15);}
.hero-card.blue  .hero-icon{background:rgba(88,166,255,.15);}
.hero-card.neutral .hero-icon{background:rgba(255,255,255,.06);}
.hero-label{font-size:11px;text-transform:uppercase;letter-spacing:.8px;
            color:var(--muted);font-weight:500;margin-bottom:6px;}
.hero-value{font-size:32px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;}
.hero-card.green .hero-value{color:var(--green);}
.hero-card.blue  .hero-value{color:var(--blue);}
.hero-card.neutral .hero-value{color:var(--fg);}
.hero-sub{font-size:12px;color:var(--muted);margin-top:6px;}
.hero-trend{font-size:12px;font-weight:600;color:var(--green);margin-top:4px;}
.ring-wrap{display:flex;align-items:center;gap:14px;margin-top:8px;}
.ring{position:relative;width:56px;height:56px;flex-shrink:0;}
.ring svg{transform:rotate(-90deg);}
.ring-bg{fill:none;stroke:var(--line);stroke-width:5;}
.ring-fill{fill:none;stroke:var(--blue);stroke-width:5;stroke-linecap:round;
           stroke-dasharray:138.2;stroke-dashoffset:138.2;transition:stroke-dashoffset .6s ease;}
.ring-label{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
            font-size:11px;font-weight:700;color:var(--blue);}
.spark-wrap{margin-top:10px;}
.spark-wrap svg{width:100%;height:28px;display:block;}
.spark-wrap polyline{fill:none;stroke:var(--green);stroke-width:1.5;
                     stroke-linejoin:round;stroke-linecap:round;opacity:.8;}
.section-head{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.section-head h2{font-size:11px;font-weight:700;text-transform:uppercase;
                 letter-spacing:.8px;color:var(--muted2);white-space:nowrap;}
.section-head::after{content:'';flex:1;height:1px;background:var(--line);}
.q-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:28px;}
.q-card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:20px 22px;}
.q-title{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:500;}
.q-score{font-size:40px;font-weight:800;margin:8px 0 4px;font-variant-numeric:tabular-nums;}
.q-score.hi{color:var(--green);}
.q-score.mid{color:var(--warn);}
.q-score.lo{color:var(--red);}
.q-bar{height:6px;border-radius:3px;background:var(--line);overflow:hidden;margin-top:12px;}
.q-bar-fill{height:100%;border-radius:3px;background:var(--green);transition:width .6s ease;}
.tbl-wrap{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;margin-bottom:28px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
th{background:var(--bg);color:var(--muted2);font-size:11px;font-weight:600;
   text-transform:uppercase;letter-spacing:.5px;padding:10px 16px;
   border-bottom:1px solid var(--line);text-align:left;}
td{padding:10px 16px;border-bottom:1px solid var(--line);color:var(--fg);}
tr:last-child td{border-bottom:none;}
tbody tr{transition:background .12s;}
tbody tr:hover td{background:rgba(255,255,255,.025);}
td.r{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);}
.name-cell{display:flex;align-items:center;gap:8px;}
.tool-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;}
.badge.hi {background:rgba(63,185,80,.12);color:var(--green);}
.badge.mid{background:rgba(227,179,65,.12);color:var(--warn);}
.badge.lo {background:rgba(248,81,73,.12);color:var(--red);}
.pbar-wrap{display:flex;align-items:center;gap:8px;}
.pbar{height:4px;border-radius:2px;transition:width .5s;}
.pbar.g{background:var(--green);}
.pbar.b{background:var(--blue);}
.pbar.w{background:var(--warn);}
.pbar.m{background:var(--muted);}
.empty-state{text-align:center;padding:32px 16px;color:var(--muted);}
.empty-state svg{opacity:.3;margin-bottom:10px;}
.empty-state p{font-size:13px;}
.empty-state code{font-size:12px;background:var(--line);border-radius:4px;padding:1px 5px;}
.bar-wrap{height:4px;background:var(--line);border-radius:2px;margin-top:10px;overflow:hidden;}
.bar{height:100%;border-radius:2px;transition:width .6s ease;}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="brand">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
      <rect width="26" height="26" rx="7" fill="#161b22" stroke="#21262d"/>
      <path d="M6 13 Q13 6 20 13 Q13 20 6 13Z" fill="none" stroke="#3fb950" stroke-width="1.6"/>
      <circle cx="13" cy="13" r="2.2" fill="#58a6ff"/>
    </svg>
    <span class="brand-name">Context<em>ly</em></span>
  </div>
  <nav class="nav">
    <div class="nav-item active">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      Overview
    </div>
    <div class="nav-item">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Proxy
    </div>
    <div class="nav-item">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
      MCP Gateway
    </div>
    <div class="nav-item">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      Quality
    </div>
  </nav>
  <div class="sidebar-footer">
    <div class="status-chip">
      <span class="sdot" id="sdot"></span>
      <span id="status">connecting…</span>
    </div>
    <div id="uptime" style="font-size:11px;color:var(--muted2);margin-top:5px;"></div>
  </div>
</aside>

<div class="main">
  <div class="topbar">
    <span class="page-title">Overview</span>
    <div class="topbar-right">
      <div class="price-row">
        <span>Fallback $/1M:</span>
        <input id="price" type="number" step="0.05" min="0" value="0.50"/>
      </div>
      <span class="uptime-badge" id="uptime2"></span>
    </div>
  </div>

  <div class="content">

    <div class="hero">
      <div class="hero-card green">
        <div class="hero-glow"></div>
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
        </div>
        <div class="hero-label">Tokens saved</div>
        <div class="hero-value" id="tokens">—</div>
        <div id="tokens-trend" class="hero-trend" style="visibility:hidden;"> </div>
        <div class="spark-wrap">
          <svg viewBox="0 0 120 28" preserveAspectRatio="none">
            <polyline id="sl-tokens" points=""/>
          </svg>
        </div>
      </div>

      <div class="hero-card green">
        <div class="hero-glow"></div>
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        </div>
        <div class="hero-label">Cost saved</div>
        <div class="hero-value" id="cost">—</div>
        <div class="hero-sub" id="cost-sub"></div>
      </div>

      <div class="hero-card neutral">
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7a8799" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </div>
        <div class="hero-label">Requests</div>
        <div class="hero-value" id="requests">—</div>
        <div class="hero-sub" id="compressed"></div>
        <div class="bar-wrap"><div class="bar" id="req-bar" style="width:0%;background:var(--blue)"></div></div>
      </div>

      <div class="hero-card blue">
        <div class="hero-glow"></div>
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2"><polyline points="22 8 22 16"/><polyline points="2 8 2 16"/><path d="M5 5h14l2 3-9 4-9-4 2-3Z"/></svg>
        </div>
        <div class="hero-label">Avg compression</div>
        <div class="ring-wrap">
          <div class="ring">
            <svg width="56" height="56" viewBox="0 0 56 56">
              <circle class="ring-bg" cx="28" cy="28" r="22"/>
              <circle class="ring-fill" id="ring-arc" cx="28" cy="28" r="22"/>
            </svg>
            <div class="ring-label" id="ring-pct">0%</div>
          </div>
          <div>
            <div class="hero-value" id="ratio">—</div>
            <div class="hero-sub">smaller</div>
          </div>
        </div>
      </div>
    </div>

    <div class="section-head"><h2>Quality metrics</h2></div>
    <div class="q-grid">
      <div class="q-card">
        <div class="q-title">ROUGE-1 quality (A/B)</div>
        <div class="q-score hi" id="quality">—</div>
        <div class="hero-sub">mean across samples</div>
        <div class="q-bar"><div class="q-bar-fill" id="quality-bar" style="width:0%"></div></div>
      </div>
      <div class="q-card">
        <div class="q-title">Numeric consistency</div>
        <div class="q-score hi" id="numeric">—</div>
        <div class="hero-sub">exact value preservation</div>
        <div class="q-bar"><div class="q-bar-fill" id="numeric-bar" style="width:0%"></div></div>
      </div>
    </div>

    <div class="section-head"><h2>Per-compressor quality</h2></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Compressor</th><th class="r">Samples</th>
          <th class="r">Quality</th><th class="r">Numeric</th><th>Score</th>
        </tr></thead>
        <tbody id="byc-body">
          <tr><td colspan="5"><div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 17H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v5"/><path d="m19 12-7 7-4-4"/></svg>
            <p>No A/B samples yet — run the proxy with<br><code>--ab-sample-rate &gt; 0</code></p>
          </div></td></tr>
        </tbody>
      </table>
    </div>

    <div class="section-head"><h2>MCP gateway · per-tool savings</h2></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Server · tool</th><th class="r">Calls</th>
          <th class="r">Before</th><th class="r">After</th>
          <th class="r">Saved</th><th>Reduction</th>
        </tr></thead>
        <tbody id="bytool-body">
          <tr><td colspan="6"><div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
            <p>No gateway tool calls yet —<br>run a tool through <code>contextly mcp-gateway</code></p>
          </div></td></tr>
        </tbody>
      </table>
    </div>

  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = n => (+n || 0).toLocaleString();
const pct = x => (x * 100).toFixed(1) + '%';
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));

const COLORS = ['#3fb950','#58a6ff','#e3b341','#7a8799','#f85149','#a371f7'];
const history = { tokens: [] };
const HIST = 40;
function pushHistory(key, val) {
  history[key].push(val);
  if (history[key].length > HIST) history[key].shift();
}
function renderSparkline(polyId, data) {
  if (data.length < 2) return;
  const max = Math.max(...data) || 1, min = Math.min(...data);
  const W = 120, H = 28, pad = 2;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - min) / (max - min || 1)) * (H - pad * 2);
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
  $(polyId).setAttribute('points', pts);
}

const _disp = {};
function animateTo(id, target, format) {
  const el = $(id); if (!el) return;
  const start = _disp[id] ?? target; _disp[id] = target;
  if (start === target) { el.textContent = format(target); return; }
  const dur = 400, t0 = performance.now();
  (function step(now) {
    const p = clamp((now - t0) / dur, 0, 1);
    el.textContent = format(Math.round(start + (target - start) * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}

function qClass(v) { return v >= 0.8 ? 'hi' : v >= 0.5 ? 'mid' : 'lo'; }
function badgeClass(p) { return p >= 40 ? 'hi' : p >= 15 ? 'mid' : 'lo'; }

function setRing(pctVal) {
  const circumference = 138.2;
  const offset = circumference * (1 - clamp(pctVal, 0, 1));
  $('ring-arc').style.strokeDashoffset = offset.toFixed(2);
  $('ring-pct').textContent = (pctVal * 100).toFixed(0) + '%';
}

const startedAt = Date.now();
function fmtUptime() {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  const m = Math.floor(s / 60), sec = s % 60, h = Math.floor(m / 60), min = m % 60;
  return h ? 'session ' + h + 'h ' + min + 'm'
    : m ? 'session ' + m + 'm ' + sec + 's' : 'session ' + sec + 's';
}
setInterval(() => {
  const u = fmtUptime();
  $('uptime').textContent = u;
  $('uptime2').textContent = u;
}, 1000);

let prevTokens = null;
async function tick() {
  try {
    const [s, q, g] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/quality').then(r => r.json()),
      fetch('/gateway-stats').then(r => r.json()).catch(() => ({})),
    ]);

    const tokens = s.tokens_saved_estimate_total || 0;
    const price  = parseFloat($('price').value) || 0;
    const total  = s.requests_total || 0, comp = s.requests_compressed || 0;
    const saved  = Math.max(0, 1 - (s.compression_ratio_mean ?? 1));

    pushHistory('tokens', tokens);
    renderSparkline('sl-tokens', history.tokens);
    animateTo('tokens', tokens, fmt);

    const trend = $('tokens-trend');
    if (prevTokens !== null && tokens !== prevTokens) {
      const delta = tokens - prevTokens;
      trend.style.visibility = 'visible';
      trend.style.color = delta > 0 ? 'var(--green)' : 'var(--red)';
      trend.textContent = (delta > 0 ? '\\u2191 +' : '\\u2193 ') + fmt(Math.abs(delta)) + ' this interval';
    }
    prevTokens = tokens;

    // Use server-side dollar savings if available, otherwise fall back to manual price.
    // Also add gateway cost savings if the gateway recorded them (--gateway-model set).
    const serverDollars = (s.dollars_saved_total || 0) + (g.dollars_saved_total || 0);
    const costVal = (serverDollars > 0)
      ? serverDollars.toFixed(4) : (tokens / 1e6 * price).toFixed(4);
    $('cost').textContent = '$' + costVal;
    $('cost-sub').textContent = (serverDollars > 0)
      ? 'server-side pricing' : (price ? 'at $' + price + '/1M tokens' : '');
    $('cost-sub').style.color = (serverDollars != null && serverDollars > 0)
      ? 'var(--green)' : 'var(--muted)';

    animateTo('requests', total, fmt);
    $('compressed').textContent = comp ? fmt(comp) + ' compressed' : '';
    $('req-bar').style.width = total ? pct(clamp(comp / total, 0, 1)) : '0%';

    $('ratio').textContent = pct(saved);
    setRing(saved);

    const qMean = q.quality?.mean, nMean = q.numeric_consistency?.mean;
    const qEl = $('quality'), nEl = $('numeric');
    qEl.textContent = qMean != null ? qMean.toFixed(3) : '\\u2014';
    qEl.className = 'q-score ' + (qMean != null ? qClass(qMean) : 'hi');
    $('quality-bar').style.width = qMean != null ? (qMean * 100).toFixed(1) + '%' : '0%';
    nEl.textContent = nMean != null ? nMean.toFixed(3) : '\\u2014';
    nEl.className = 'q-score ' + (nMean != null ? qClass(nMean) : 'hi');
    $('numeric-bar').style.width = nMean != null ? (nMean * 100).toFixed(1) + '%' : '0%';

    const cRows = Object.entries(q.by_compressor || {});
    if (cRows.length) {
      $('byc-body').innerHTML = cRows
        .sort((a, b) => b[1].samples - a[1].samples)
        .map(([name, c], i) => {
          const qv = c.mean_quality ?? 0, nv = c.mean_numeric_consistency ?? 1;
          const col = COLORS[i % COLORS.length];
          return '<tr>'
            + '<td><div class="name-cell"><span class="tool-dot" style="background:' + col + '"></span>' + name + '</div></td>'
            + '<td class="r">' + fmt(c.samples) + '</td>'
            + '<td class="r"><span style="color:var(--' + (qv>=.8?'green':qv>=.5?'warn':'red') + ');font-weight:600">' + qv.toFixed(3) + '</span></td>'
            + '<td class="r"><span style="color:var(--' + (nv>=.8?'green':nv>=.5?'warn':'red') + ');font-weight:600">' + nv.toFixed(3) + '</span></td>'
            + '<td><div class="pbar-wrap"><div class="pbar ' + (qv>=.8?'g':qv>=.5?'w':'m') + '" style="width:' + clamp(qv*100,0,100) + 'px"></div>'
            + '<span style="font-size:11px;color:var(--muted)">' + (qv*100).toFixed(0) + '%</span></div></td>'
            + '</tr>';
        }).join('');
    }

    const rows = Object.entries(g.by_tool || {});
    if (rows.length) {
      $('bytool-body').innerHTML = rows
        .sort((a, b) => b[1].chars_saved - a[1].chars_saved)
        .map(([name, t], i) => {
          const sp = parseFloat(t.saved_pct) || 0;
          const col = COLORS[i % COLORS.length];
          return '<tr>'
            + '<td><div class="name-cell"><span class="tool-dot" style="background:' + col + '"></span><strong>' + name + '</strong></div></td>'
            + '<td class="r">' + fmt(t.calls) + '</td>'
            + '<td class="r">' + fmt(t.chars_before) + '</td>'
            + '<td class="r">' + fmt(t.chars_after) + '</td>'
            + '<td class="r" style="color:var(--fg)">' + fmt(t.chars_saved) + '</td>'
            + '<td><div class="pbar-wrap"><div class="pbar g" style="width:' + clamp(sp,0,100) + 'px"></div>'
            + '<span class="badge ' + badgeClass(sp) + '">' + sp.toFixed(1) + '%</span></div></td>'
            + '</tr>';
        }).join('');
    }

    $('sdot').className = 'sdot';
    $('status').textContent = 'live \\u00b7 ' + new Date().toLocaleTimeString();
  } catch(e) {
    $('sdot').className = 'sdot err';
    $('status').textContent = 'disconnected';
  }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""
