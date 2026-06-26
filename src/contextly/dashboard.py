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
  :root { --bg:#0e1116; --card:#161b22; --line:#272e38; --fg:#e6edf3;
          --muted:#8b949e; --accent:#3fb950; --accent2:#58a6ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding:24px 28px 8px; }
  h1 { margin:0; font-size:20px; letter-spacing:.2px; }
  h1 span { color:var(--accent); }
  .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  .wrap { padding:16px 28px 40px; }
  .grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.6px; }
  .value { font-size:30px; font-weight:650; margin-top:6px; }
  .value.green { color:var(--accent); }
  .value.blue { color:var(--accent2); }
  .unit { font-size:14px; color:var(--muted); font-weight:400; margin-left:4px; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-size:14px; }
  th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; font-size:12px; text-transform:uppercase; }
  td.num { text-align:right; font-variant-numeric: tabular-nums; }
  .section { margin-top:26px; }
  .section h2 { font-size:14px; color:var(--muted); font-weight:600;
                text-transform:uppercase; letter-spacing:.6px; margin:0 0 6px; }
  .row { display:flex; align-items:center; gap:10px; margin-top:10px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--accent); }
  .muted { color:var(--muted); }
  input { background:#0d1117; color:var(--fg); border:1px solid var(--line);
          border-radius:7px; padding:4px 8px; width:84px; }
  .empty { color:var(--muted); font-style:italic; padding:8px 0; }
</style>
</head>
<body>
<header>
  <h1>Context<span>ly</span> — live savings</h1>
  <div class="sub">Polling <code>/stats</code>, <code>/quality</code> and
    <code>/gateway-stats</code> every 2s
    · price <input id="price" type="number" step="0.05" value="0.50" /> $/1M input tokens</div>
</header>
<div class="wrap">
  <div class="grid">
    <div class="card"><div class="label">Tokens saved</div>
      <div class="value green" id="tokens">—</div></div>
    <div class="card"><div class="label">Estimated cost saved</div>
      <div class="value green" id="cost">—</div></div>
    <div class="card"><div class="label">Requests</div>
      <div class="value" id="requests">—</div>
      <div class="muted" id="compressed" style="font-size:13px;margin-top:4px"></div></div>
    <div class="card"><div class="label">Avg compression</div>
      <div class="value blue" id="ratio">—</div></div>
    <div class="card"><div class="label">Quality (ROUGE-1)</div>
      <div class="value" id="quality">—</div></div>
    <div class="card"><div class="label">Numeric consistency</div>
      <div class="value" id="numeric">—</div></div>
  </div>

  <div class="section">
    <h2>Per-compressor quality</h2>
    <table id="byc">
      <thead><tr><th>Compressor</th><th class="num">Samples</th>
        <th class="num">Quality</th><th class="num">Numeric</th></tr></thead>
      <tbody><tr><td colspan="4" class="empty">No A/B samples yet
        (run the proxy with --ab-sample-rate &gt; 0).</td></tr></tbody>
    </table>
  </div>

  <div class="section">
    <h2>MCP gateway — tool-output savings</h2>
    <div class="grid" style="margin-bottom:6px">
      <div class="card"><div class="label">Gateway tokens saved</div>
        <div class="value green" id="g_tokens">—</div></div>
      <div class="card"><div class="label">Gateway tool calls</div>
        <div class="value" id="g_calls">—</div></div>
      <div class="card"><div class="label">Gateway avg compression</div>
        <div class="value blue" id="g_ratio">—</div></div>
    </div>
    <table id="bytool">
      <thead><tr><th>Server · tool</th><th class="num">Calls</th>
        <th class="num">Before</th><th class="num">After</th>
        <th class="num">Saved</th></tr></thead>
      <tbody><tr><td colspan="5" class="empty">No gateway tool calls recorded yet —
        run a tool through <code>contextly mcp-gateway</code>.</td></tr></tbody>
    </table>
  </div>

  <div class="row"><span class="dot" id="live"></span>
    <span class="muted" id="status">connecting…</span></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString();
function pct(x) { return (x*100).toFixed(1) + '%'; }

function renderGateway(g) {
  $('g_tokens').textContent = fmt(g.tokens_saved_estimate_total || 0);
  $('g_calls').textContent = fmt(g.tool_calls_total || 0);
  const gsaved = 1 - (g.compression_ratio_mean ?? 1);
  $('g_ratio').innerHTML = pct(Math.max(0, gsaved)) + '<span class="unit">smaller</span>';
  const tbody = $('bytool').querySelector('tbody');
  const rows = Object.entries(g.by_tool || {});
  if (rows.length) {
    tbody.innerHTML = rows.map(([name, t]) =>
      `<tr><td>${name}</td><td class="num">${fmt(t.calls)}</td>` +
      `<td class="num">${fmt(t.chars_before)}</td>` +
      `<td class="num">${fmt(t.chars_after)}</td>` +
      `<td class="num">${fmt(t.chars_saved)} (${t.saved_pct}%)</td></tr>`).join('');
  }
}

async function tick() {
  try {
    const [s, q, g] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/quality').then(r => r.json()),
      fetch('/gateway-stats').then(r => r.json()).catch(() => ({})),
    ]);
    renderGateway(g);
    const tokens = s.tokens_saved_estimate_total || 0;
    const price = parseFloat($('price').value) || 0;
    $('tokens').textContent = fmt(tokens);
    $('cost').innerHTML = '$' + (tokens / 1e6 * price).toFixed(2);
    $('requests').textContent = fmt(s.requests_total || 0);
    $('compressed').textContent = fmt(s.requests_compressed || 0) + ' compressed';
    const saved = 1 - (s.compression_ratio_mean ?? 1);
    $('ratio').innerHTML = pct(Math.max(0, saved)) + '<span class="unit">smaller</span>';
    $('quality').textContent = q.quality ? q.quality.mean.toFixed(3) : '—';
    $('numeric').textContent = q.numeric_consistency ? q.numeric_consistency.mean.toFixed(3) : '—';

    const tbody = $('byc').querySelector('tbody');
    const rows = Object.entries(q.by_compressor || {});
    if (rows.length) {
      tbody.innerHTML = rows.map(([name, c]) =>
        `<tr><td>${name}</td><td class="num">${c.samples}</td>` +
        `<td class="num">${c.mean_quality.toFixed(3)}</td>` +
        `<td class="num">${(c.mean_numeric_consistency ?? 1).toFixed(3)}</td></tr>`).join('');
    }
    $('live').style.background = '#3fb950';
    $('status').textContent = 'live · updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    $('live').style.background = '#f85149';
    $('status').textContent = 'disconnected — is the proxy running?';
  }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""
