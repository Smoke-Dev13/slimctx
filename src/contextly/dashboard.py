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
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Contextly — live token savings</title>
<style>
  :root {
    --bg:#0d1117; --card:#161b22; --card-hover:#1c2129; --line:#21262d;
    --fg:#e6edf3; --muted:#8b949e; --muted2:#6e7681;
    --accent:#3fb950; --accent2:#58a6ff; --warn:#d29922; --danger:#f85149;
    --radius:12px; --radius-sm:8px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--fg);
         font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
         font-size:14px; line-height:1.5; min-height:100vh; }
  header { display:flex; align-items:center; gap:14px;
           padding:20px 28px 16px; border-bottom:1px solid var(--line); }
  .logo { display:flex; align-items:center; gap:8px; }
  h1 { font-size:17px; font-weight:600; letter-spacing:-.2px; }
  h1 em { font-style:normal; color:var(--accent); }
  .header-meta { margin-left:auto; display:flex; align-items:center; gap:16px;
                 color:var(--muted); font-size:12px; white-space:nowrap; }
  .price-wrap { display:flex; align-items:center; gap:6px; }
  input[type=number] { background:#0d1117; color:var(--fg);
                        border:1px solid var(--line); border-radius:var(--radius-sm);
                        padding:4px 8px; width:76px; font-size:13px; }
  input[type=number]:focus { outline:2px solid var(--accent2); outline-offset:1px; }
  .statusbar { display:flex; align-items:center; gap:8px;
               padding:6px 28px; background:var(--card); border-bottom:1px solid var(--line);
               font-size:12px; color:var(--muted); }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent);
         flex-shrink:0; transition:background .4s; }
  .dot.live { animation:pulse 2s ease-in-out infinite; }
  @keyframes pulse {
    0%,100% { box-shadow:0 0 0 0 rgba(63,185,80,.5); }
    50%      { box-shadow:0 0 0 5px rgba(63,185,80,0); }
  }
  .dot.err { background:var(--danger); animation:none; }
  .uptime { margin-left:auto; }
  .wrap { padding:20px 28px 48px; }
  .section { margin-top:28px; }
  .section-title { font-size:11px; font-weight:600; text-transform:uppercase;
                   letter-spacing:.8px; color:var(--muted2); margin-bottom:12px;
                   display:flex; align-items:center; gap:8px; }
  .section-title::after { content:''; flex:1; height:1px; background:var(--line); }
  .grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
          padding:16px 18px; transition:border-color .2s,background .2s; overflow:hidden; }
  .card:hover { border-color:var(--muted2); background:var(--card-hover); }
  .card-accent  { border-left:3px solid var(--accent); }
  .card-accent2 { border-left:3px solid var(--accent2); }
  .label { color:var(--muted); font-size:11px; text-transform:uppercase;
           letter-spacing:.6px; font-weight:500; }
  .value { font-size:28px; font-weight:700; margin-top:4px; line-height:1.1;
           font-variant-numeric:tabular-nums; transition:color .3s; }
  .value.green { color:var(--accent); }
  .value.blue  { color:var(--accent2); }
  .unit { font-size:13px; color:var(--muted); font-weight:400; margin-left:3px; }
  .card-sub { font-size:12px; color:var(--muted); margin-top:5px; }
  .trend { font-size:12px; font-weight:600; }
  .trend.up   { color:var(--accent); }
  .trend.down { color:var(--danger); }
  .bar-wrap { height:4px; background:var(--line); border-radius:2px; margin-top:10px; overflow:hidden; }
  .bar { height:100%; border-radius:2px; transition:width .6s ease; background:var(--accent2); }
  .spark { display:block; width:100%; height:32px; margin-top:8px; }
  .spark polyline { fill:none; stroke-width:1.5; stroke-linejoin:round; stroke-linecap:round; }
  .table-wrap { border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; margin-top:4px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { color:var(--muted2); font-weight:500; font-size:11px; text-transform:uppercase;
       letter-spacing:.5px; padding:9px 14px; border-bottom:1px solid var(--line);
       background:var(--bg); text-align:left; }
  td { padding:9px 14px; border-bottom:1px solid var(--line); color:var(--fg); }
  tr:last-child td { border-bottom:none; }
  tbody tr { transition:background .15s; }
  tbody tr:hover td { background:var(--card-hover); }
  td.num { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
  td.saved { text-align:right; font-variant-numeric:tabular-nums; }
  .badge { display:inline-block; padding:1px 7px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.hi  { background:rgba(63,185,80,.15);  color:var(--accent); }
  .badge.mid { background:rgba(210,153,34,.15); color:var(--warn); }
  .badge.lo  { background:rgba(248,81,73,.15);  color:var(--danger); }
  .q-score { font-variant-numeric:tabular-nums; font-weight:600; }
  .q-hi  { color:var(--accent); }
  .q-mid { color:var(--warn); }
  .q-lo  { color:var(--danger); }
  .mini-bar-wrap { display:flex; align-items:center; gap:6px; }
  .mini-bar { height:5px; border-radius:3px; min-width:2px; transition:width .5s; }
  .mini-bar.green { background:var(--accent); }
  .mini-bar.blue  { background:var(--accent2); }
  .empty-state { text-align:center; padding:28px 16px; color:var(--muted); }
  .empty-state svg { opacity:.35; margin-bottom:10px; }
  .empty-state p { font-size:13px; }
  .empty-state code { font-size:12px; background:var(--line); border-radius:4px; padding:1px 5px; }
  [data-tip] { position:relative; cursor:default; }
  [data-tip]:hover::after {
    content:attr(data-tip); position:absolute; bottom:calc(100% + 6px); left:50%;
    transform:translateX(-50%); background:#1f2937; color:#f3f4f6; font-size:11px;
    padding:4px 8px; border-radius:5px; white-space:nowrap; pointer-events:none;
    z-index:10; border:1px solid #374151;
  }
</style>
</head>
<body>
<header>
  <div class="logo">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <rect width="28" height="28" rx="8" fill="#161b22" stroke="#21262d"/>
      <path d="M7 14 Q14 7 21 14 Q14 21 7 14Z" fill="none" stroke="#3fb950" stroke-width="1.8"/>
      <circle cx="14" cy="14" r="2.5" fill="#58a6ff"/>
    </svg>
    <h1>Context<em>ly</em></h1>
  </div>
  <div class="header-meta">
    <div class="price-wrap">
      <label for="price">Fallback $/1M tokens:</label>
      <input id="price" type="number" step="0.05" min="0" value="0.50" />
    </div>
  </div>
</header>
<div class="statusbar">
  <span class="dot live" id="dot"></span>
  <span id="status">connecting…</span>
  <span class="uptime" id="uptime"></span>
</div>
<div class="wrap">
  <div class="section-title">Proxy · token savings</div>
  <div class="grid">
    <div class="card card-accent" data-tip="Total input tokens eliminated from upstream requests">
      <div class="label">Tokens saved</div>
      <div class="value green" id="tokens">—</div>
      <div class="card-sub" id="tokens-sub"></div>
      <svg class="spark" viewBox="0 0 120 32" preserveAspectRatio="none">
        <polyline id="sl-tokens" stroke="#3fb950" points=""/>
      </svg>
    </div>
    <div class="card card-accent" data-tip="Server-side savings when model pricing is known, otherwise uses the fallback price above">
      <div class="label">Cost saved</div>
      <div class="value green" id="cost">—</div>
      <div class="card-sub" id="cost-sub"></div>
    </div>
    <div class="card" data-tip="Total vs compressed requests processed">
      <div class="label">Requests</div>
      <div class="value" id="requests">—</div>
      <div class="card-sub" id="compressed"></div>
      <div class="bar-wrap"><div class="bar" id="req-bar" style="width:0%"></div></div>
    </div>
    <div class="card card-accent2" data-tip="Mean size reduction across all compressed messages">
      <div class="label">Avg compression</div>
      <div class="value blue" id="ratio">—</div>
      <div class="bar-wrap"><div class="bar" id="ratio-bar" style="width:0%;background:var(--accent2)"></div></div>
    </div>
    <div class="card" data-tip="Mean ROUGE-1 F1 — semantic fidelity of compressed vs original">
      <div class="label">Quality (ROUGE-1)</div>
      <div class="value" id="quality">—</div>
      <div class="card-sub">A/B sample score</div>
    </div>
    <div class="card" data-tip="Fraction of numeric values preserved exactly after compression">
      <div class="label">Numeric consistency</div>
      <div class="value" id="numeric">—</div>
      <div class="card-sub">exact preservation</div>
    </div>
    <div class="card" data-tip="Requests flagged for potential prompt injection">
      <div class="label">Injection alerts</div>
      <div class="value" id="injection" style="color:var(--danger)">—</div>
      <div class="card-sub">detected / blocked</div>
    </div>
    <div class="card card-accent" data-tip="Dollars saved via prompt caching">
      <div class="label">Cache savings</div>
      <div class="value green" id="cache">—</div>
      <div class="card-sub" id="cache-sub">prompt cache</div>
    </div>
    <div class="card" data-tip="Closed-loop compression tuning: step-ups / step-downs and verbosity spikes">
      <div class="label">Adaptive control</div>
      <div class="value" id="adaptive">—</div>
      <div class="card-sub" id="adaptive-sub">spikes</div>
    </div>
    <div class="card" data-tip="Image parts compressed (detail downgrade / downscale)">
      <div class="label">Images compressed</div>
      <div class="value blue" id="images">—</div>
      <div class="card-sub">multimodal</div>
    </div>
    <div class="card" data-tip="Secrets / PII redacted before reaching the upstream LLM">
      <div class="label">Secrets redacted</div>
      <div class="value" id="secrets" style="color:var(--danger)">—</div>
      <div class="card-sub">semantic firewall</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Per-compressor quality</div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Compressor</th><th class="num">Samples</th>
          <th class="num">Quality</th><th class="num">Numeric</th><th>Score bar</th>
        </tr></thead>
        <tbody id="byc-body">
          <tr><td colspan="5"><div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 17H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v5"/><path d="m19 12-7 7-4-4"/></svg>
            <p>No A/B samples yet — run the proxy with<br><code>--ab-sample-rate &gt; 0</code></p>
          </div></td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">MCP gateway · tool-output savings</div>
    <div class="grid" style="margin-bottom:14px">
      <div class="card card-accent">
        <div class="label">Gateway tokens saved</div>
        <div class="value green" id="g_tokens">—</div>
        <svg class="spark" viewBox="0 0 120 32" preserveAspectRatio="none">
          <polyline id="sl-gtokens" stroke="#3fb950" points=""/>
        </svg>
      </div>
      <div class="card">
        <div class="label">Gateway tool calls</div>
        <div class="value" id="g_calls">—</div>
        <div class="card-sub" id="g_calls_sub"></div>
        <div class="bar-wrap"><div class="bar" id="gcall-bar" style="width:0%"></div></div>
      </div>
      <div class="card card-accent2">
        <div class="label">Gateway avg compression</div>
        <div class="value blue" id="g_ratio">—</div>
        <div class="bar-wrap"><div class="bar" id="gratio-bar" style="width:0%;background:var(--accent2)"></div></div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Server · tool</th><th class="num">Calls</th>
          <th class="num">Before</th><th class="num">After</th>
          <th class="num">Chars saved</th><th>Reduction</th>
        </tr></thead>
        <tbody id="bytool-body">
          <tr><td colspan="6"><div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
            <p>No gateway tool calls recorded yet —<br>run a tool through <code>contextly mcp-gateway</code></p>
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

const history = { tokens: [], gtokens: [] };
const HIST = 40;
function pushHistory(key, val) {
  history[key].push(val);
  if (history[key].length > HIST) history[key].shift();
}
function renderSparkline(polyId, data) {
  if (data.length < 2) return;
  const max = Math.max(...data) || 1, min = Math.min(...data);
  const W = 120, H = 32, pad = 2;
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

function qClass(v) { return v >= 0.8 ? 'q-hi' : v >= 0.5 ? 'q-mid' : 'q-lo'; }
function badgeClass(p) { return p >= 40 ? 'hi' : p >= 15 ? 'mid' : 'lo'; }

const startedAt = Date.now();
setInterval(() => {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  const m = Math.floor(s / 60), sec = s % 60, h = Math.floor(m / 60), min = m % 60;
  $('uptime').textContent = h ? 'session ' + h + 'h ' + min + 'm'
    : m ? 'session ' + m + 'm ' + sec + 's' : 'session ' + sec + 's';
}, 1000);

function renderGateway(g) {
  const gsaved = Math.max(0, 1 - (g.compression_ratio_mean ?? 1));
  const gtok = g.tokens_saved_estimate_total || 0;
  const gcalls = g.tool_calls_total || 0, gcomp = g.tool_calls_compressed || 0;
  pushHistory('gtokens', gtok);
  renderSparkline('sl-gtokens', history.gtokens);
  animateTo('g_tokens', gtok, fmt);
  animateTo('g_calls', gcalls, fmt);
  $('g_calls_sub').textContent = gcomp ? fmt(gcomp) + ' compressed' : '';
  $('g_ratio').innerHTML = pct(gsaved) + '<span class="unit">smaller</span>';
  $('gratio-bar').style.width = pct(clamp(gsaved, 0, 1));
  $('gcall-bar').style.width = gcalls ? pct(clamp(gcomp / gcalls, 0, 1)) : '0%';
  const rows = Object.entries(g.by_tool || {});
  if (!rows.length) return;
  $('bytool-body').innerHTML = rows
    .sort((a, b) => b[1].chars_saved - a[1].chars_saved)
    .map(([name, t]) => {
      const sp = parseFloat(t.saved_pct) || 0;
      return '<tr><td><strong>' + name + '</strong></td>'
        + '<td class="num">' + fmt(t.calls) + '</td>'
        + '<td class="num">' + fmt(t.chars_before) + '</td>'
        + '<td class="num">' + fmt(t.chars_after) + '</td>'
        + '<td class="saved">' + fmt(t.chars_saved) + '</td>'
        + '<td><div class="mini-bar-wrap">'
        + '<div class="mini-bar green" style="width:' + clamp(sp,0,100) + 'px"></div>'
        + '<span class="badge ' + badgeClass(sp) + '">' + sp.toFixed(1) + '%</span>'
        + '</div></td></tr>';
    }).join('');
}

let prevTokens = null;
async function tick() {
  try {
    const [s, q, g] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/quality').then(r => r.json()),
      fetch('/gateway-stats').then(r => r.json()).catch(() => ({})),
    ]);
    renderGateway(g);
    const tokens = s.tokens_saved_estimate_total || 0;
    const price  = parseFloat($('price').value) || 0;
    const total  = s.requests_total || 0, comp = s.requests_compressed || 0;
    const saved  = Math.max(0, 1 - (s.compression_ratio_mean ?? 1));
    pushHistory('tokens', tokens);
    renderSparkline('sl-tokens', history.tokens);
    animateTo('tokens', tokens, fmt);
    if (prevTokens !== null && tokens !== prevTokens) {
      const delta = tokens - prevTokens;
      $('tokens-sub').innerHTML = delta > 0
        ? '<span class="trend up">↑ +' + fmt(delta) + ' this interval</span>'
        : '<span class="trend down">↓ ' + fmt(delta) + '</span>';
    }
    prevTokens = tokens;
    // Use server-side dollar savings if available (has real model pricing),
    // otherwise fall back to the manual price input for a rough estimate.
    const serverDollars = s.dollars_saved_total;
    const costVal = (serverDollars != null && serverDollars > 0)
      ? serverDollars.toFixed(4) : (tokens / 1e6 * price).toFixed(4);
    $('cost').textContent = '$' + costVal;
    $('cost-sub').textContent = (serverDollars != null && serverDollars > 0)
      ? 'server-side pricing' : (price ? 'at $' + price + '/1M tokens' : '');
    animateTo('requests', total, fmt);
    $('compressed').textContent = comp ? fmt(comp) + ' compressed' : '';
    $('req-bar').style.width = total ? pct(clamp(comp / total, 0, 1)) : '0%';
    $('ratio').innerHTML = pct(saved) + '<span class="unit">smaller</span>';
    $('ratio-bar').style.width = pct(clamp(saved, 0, 1));
    const qMean = q.quality?.mean, nMean = q.numeric_consistency?.mean;
    $('quality').textContent = qMean != null ? qMean.toFixed(3) : '—';
    $('quality').className = 'value' + (qMean != null ? ' ' + qClass(qMean) : '');
    $('numeric').textContent = nMean != null ? nMean.toFixed(3) : '—';
    $('numeric').className = 'value' + (nMean != null ? ' ' + qClass(nMean) : '');
    const detected = s.injections_detected_total || 0, blocked = s.injections_blocked_total || 0;
    $('injection').textContent = detected + ' / ' + blocked;
    $('cache').textContent = '$' + (s.cache_savings_dollars_total || 0).toFixed(4);
    $('cache-sub').textContent = fmt(s.cache_hit_tokens_total || 0) + ' cached tokens';
    const ups = s.adaptive_stepups_total || 0, downs = s.adaptive_stepdowns_total || 0;
    $('adaptive').textContent = '↑' + ups + ' ↓' + downs;
    $('adaptive-sub').textContent = fmt(s.verbosity_spikes_total || 0) + ' verbosity spikes';
    $('images').textContent = fmt(s.image_parts_compressed_total || 0);
    $('secrets').textContent = fmt(s.secrets_redacted_total || 0);
    const cRows = Object.entries(q.by_compressor || {});
    if (cRows.length) {
      $('byc-body').innerHTML = cRows
        .sort((a, b) => b[1].samples - a[1].samples)
        .map(([name, c]) => {
          const qv = c.mean_quality ?? 0, nv = c.mean_numeric_consistency ?? 1;
          return '<tr><td><strong>' + name + '</strong></td>'
            + '<td class="num">' + fmt(c.samples) + '</td>'
            + '<td class="num"><span class="q-score ' + qClass(qv) + '">' + qv.toFixed(3) + '</span></td>'
            + '<td class="num"><span class="q-score ' + qClass(nv) + '">' + nv.toFixed(3) + '</span></td>'
            + '<td><div class="mini-bar-wrap">'
            + '<div class="mini-bar blue" style="width:' + clamp(qv*100,0,100) + 'px"></div>'
            + '<span style="font-size:11px;color:var(--muted)">' + (qv*100).toFixed(0) + '%</span>'
            + '</div></td></tr>';
        }).join('');
    }
    $('dot').className = 'dot live';
    $('status').textContent = 'live · updated ' + new Date().toLocaleTimeString();
  } catch(e) {
    $('dot').className = 'dot err';
    $('status').textContent = 'disconnected — is the proxy running?';
  }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""
