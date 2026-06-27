"""Optional live dashboard for the MCP gateway.

The gateway runs as a stdio process whose stdout is reserved for JSON-RPC, so it
cannot reuse the proxy's FastAPI ``/dashboard``. Instead this module spins up a
tiny stdlib HTTP server on a background daemon thread that serves a self-updating
page polling the gateway's :class:`~contextly.gateway_stats.StatsRecorder`.

Open it in a browser while Claude Desktop (or any MCP client) is connected to see
per-tool compression savings update live. It is intentionally built on
``http.server`` rather than uvicorn so it shares no event loop with the stdio
transport and can never interfere with the JSON-RPC stream.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import structlog

from contextly.gateway_stats import StatsRecorder

logger = structlog.get_logger(__name__)

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Contextly gateway — live token savings</title>
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
  .badge-mcp { margin-left:8px; background:rgba(88,166,255,.15); color:var(--accent2);
               font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px;
               letter-spacing:.4px; vertical-align:middle; }
  .statusbar { display:flex; align-items:center; gap:8px;
               padding:6px 28px; background:var(--card); border-bottom:1px solid var(--line);
               font-size:12px; color:var(--muted); }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); flex-shrink:0; }
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
          padding:16px 18px; transition:border-color .2s,background .2s; }
  .card:hover { border-color:var(--muted2); background:var(--card-hover); }
  .card-accent  { border-left:3px solid var(--accent); }
  .card-accent2 { border-left:3px solid var(--accent2); }
  .label { color:var(--muted); font-size:11px; text-transform:uppercase;
           letter-spacing:.6px; font-weight:500; }
  .value { font-size:28px; font-weight:700; margin-top:4px; line-height:1.1;
           font-variant-numeric:tabular-nums; }
  .value.green { color:var(--accent); }
  .value.blue  { color:var(--accent2); }
  .unit { font-size:13px; color:var(--muted); font-weight:400; margin-left:3px; }
  .card-sub { font-size:12px; color:var(--muted); margin-top:5px; }
  .bar-wrap { height:4px; background:var(--line); border-radius:2px; margin-top:10px; overflow:hidden; }
  .bar { height:100%; border-radius:2px; transition:width .6s ease; }
  .bar.green { background:var(--accent); }
  .bar.blue  { background:var(--accent2); }
  .spark { display:block; width:100%; height:32px; margin-top:8px; }
  .spark polyline { fill:none; stroke-width:1.5; stroke-linejoin:round; stroke-linecap:round; }
  .table-wrap { border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; margin-top:4px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { color:var(--muted2); font-weight:500; font-size:11px; text-transform:uppercase;
       letter-spacing:.5px; padding:9px 14px; border-bottom:1px solid var(--line);
       background:var(--bg); text-align:left; }
  td { padding:9px 14px; border-bottom:1px solid var(--line); }
  tr:last-child td { border-bottom:none; }
  tbody tr { transition:background .15s; }
  tbody tr:hover td { background:var(--card-hover); }
  td.num { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
  .badge { display:inline-block; padding:1px 7px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.hi  { background:rgba(63,185,80,.15);  color:var(--accent); }
  .badge.mid { background:rgba(210,153,34,.15); color:var(--warn); }
  .badge.lo  { background:rgba(248,81,73,.15);  color:var(--danger); }
  .mini-bar-wrap { display:flex; align-items:center; gap:6px; }
  .mini-bar { height:5px; border-radius:3px; min-width:2px; transition:width .5s; background:var(--accent); }
  .empty-state { text-align:center; padding:28px 16px; color:var(--muted); }
  .empty-state svg { opacity:.35; margin-bottom:10px; }
  .empty-state p { font-size:13px; }
  .empty-state code { font-size:12px; background:var(--line); border-radius:4px; padding:1px 5px; }
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
    <h1>Context<em>ly</em> <span class="badge-mcp">MCP Gateway</span></h1>
  </div>
</header>
<div class="statusbar">
  <span class="dot live" id="dot"></span>
  <span id="status">connecting…</span>
  <span class="uptime" id="uptime"></span>
</div>
<div class="wrap">
  <div class="section-title">Tool-output compression · live metrics</div>
  <div class="grid">
    <div class="card card-accent">
      <div class="label">Tokens saved (est.)</div>
      <div class="value green" id="tokens">—</div>
      <svg class="spark" viewBox="0 0 120 32" preserveAspectRatio="none">
        <polyline id="sl-tokens" stroke="#3fb950" points=""/>
      </svg>
    </div>
    <div class="card card-accent">
      <div class="label">Characters saved</div>
      <div class="value green" id="chars">—</div>
      <div class="card-sub" id="chars-sub"></div>
    </div>
    <div class="card">
      <div class="label">Tool calls</div>
      <div class="value" id="calls">—</div>
      <div class="card-sub" id="compressed"></div>
      <div class="bar-wrap"><div class="bar green" id="calls-bar" style="width:0%"></div></div>
    </div>
    <div class="card card-accent2">
      <div class="label">Avg compression</div>
      <div class="value blue" id="ratio">—</div>
      <div class="bar-wrap"><div class="bar blue" id="ratio-bar" style="width:0%"></div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Per-tool breakdown</div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Tool</th><th class="num">Calls</th>
          <th class="num">Before</th><th class="num">After</th>
          <th class="num">Chars saved</th><th>Reduction</th>
        </tr></thead>
        <tbody id="bytool-body">
          <tr><td colspan="6"><div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
            <p>No tool calls yet —<br>run a tool in your MCP client</p>
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

const sparkData = [];
const HIST = 40;
function pushSpark(val) { sparkData.push(val); if (sparkData.length > HIST) sparkData.shift(); }
function renderSpark(polyId, data) {
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

const _prev = {};
function animateTo(id, target, fmt2) {
  const el = $(id); if (!el) return;
  const start = _prev[id] ?? target; _prev[id] = target;
  const dur = 400, t0 = performance.now();
  (function step(now) {
    const p = clamp((now - t0) / dur, 0, 1);
    el.textContent = fmt2(Math.round(start + (target - start) * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}

const t0 = Date.now();
setInterval(() => {
  const s = Math.floor((Date.now() - t0) / 1000);
  const m = Math.floor(s / 60), sec = s % 60, h = Math.floor(m / 60), min = m % 60;
  $('uptime').textContent = h ? 'session ' + h + 'h ' + min + 'm'
    : m ? 'session ' + m + 'm ' + sec + 's' : 'session ' + sec + 's';
}, 1000);

function badgeClass(p) { return p >= 40 ? 'hi' : p >= 15 ? 'mid' : 'lo'; }

async function tick() {
  try {
    const s = await fetch('/stats').then(r => r.json());
    const tokens = s.tokens_saved_estimate_total || 0;
    const chars  = s.chars_saved_total || 0;
    const total  = s.tool_calls_total || 0, comp = s.tool_calls_compressed || 0;
    const saved  = Math.max(0, 1 - (s.compression_ratio_mean ?? 1));
    pushSpark(tokens);
    renderSpark('sl-tokens', sparkData);
    animateTo('tokens', tokens, fmt);
    animateTo('chars',  chars,  fmt);
    animateTo('calls',  total,  fmt);
    $('chars-sub').textContent = chars ? '≈ ' + Math.round(chars / 4).toLocaleString() + ' tokens' : '';
    $('compressed').textContent = comp ? fmt(comp) + ' compressed' : '';
    $('calls-bar').style.width  = total ? pct(clamp(comp / total, 0, 1)) : '0%';
    $('ratio').innerHTML = pct(saved) + '<span class="unit">smaller</span>';
    $('ratio-bar').style.width = pct(clamp(saved, 0, 1));
    const rows = Object.entries(s.by_tool || {});
    if (rows.length) {
      $('bytool-body').innerHTML = rows
        .sort((a, b) => b[1].chars_saved - a[1].chars_saved)
        .map(([name, t]) => {
          const sp = parseFloat(t.saved_pct) || 0;
          return '<tr><td><strong>' + name + '</strong></td>'
            + '<td class="num">' + fmt(t.calls) + '</td>'
            + '<td class="num">' + fmt(t.chars_before) + '</td>'
            + '<td class="num">' + fmt(t.chars_after) + '</td>'
            + '<td class="num">' + fmt(t.chars_saved) + '</td>'
            + '<td><div class="mini-bar-wrap">'
            + '<div class="mini-bar" style="width:' + clamp(sp,0,100) + 'px"></div>'
            + '<span class="badge ' + badgeClass(sp) + '">' + sp.toFixed(1) + '%</span>'
            + '</div></td></tr>';
        }).join('');
    }
    $('dot').className = 'dot live';
    $('status').textContent = 'live · updated ' + new Date().toLocaleTimeString();
  } catch(e) {
    $('dot').className = 'dot err';
    $('status').textContent = 'disconnected — is the gateway running?';
  }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard page and a JSON stats snapshot.

    The owning stats recorder is attached to the server instance so each
    request can read the live snapshot without a module global.
    """

    # Silence the default stderr request logging — it would clutter the gateway's
    # own structured logs (which already share stderr).
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/dashboard"):
            self._send(200, _DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/stats":
            stats: StatsRecorder = self.server.gateway_stats  # type: ignore[attr-defined]
            body = json.dumps(stats.snapshot()).encode("utf-8")
            self._send(200, body, "application/json")
        elif path == "/favicon.ico":
            self._send(204, b"", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def start_dashboard(stats: StatsRecorder, host: str, port: int) -> ThreadingHTTPServer | None:
    """Start the dashboard HTTP server on a background daemon thread.

    Returns the server (so the caller may close it) or ``None`` if the port could
    not be bound — a busy port must never take the gateway down, since the
    dashboard is a convenience, not part of the MCP transport.
    """
    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        logger.warning("gateway_dashboard_bind_failed", host=host, port=port, error=str(exc))
        return None
    server.gateway_stats = stats  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="contextly-dashboard", daemon=True)
    thread.start()
    logger.info("gateway_dashboard_started", url=f"http://{host}:{port}/dashboard")
    return server
