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
  .empty { color:var(--muted); font-style:italic; padding:8px 0; }
</style>
</head>
<body>
<header>
  <h1>Context<span>ly</span> gateway — live savings</h1>
  <div class="sub">Compressed MCP tool outputs across all wrapped servers ·
    polling <code>/stats</code> every 2s</div>
</header>
<div class="wrap">
  <div class="grid">
    <div class="card"><div class="label">Tokens saved (est.)</div>
      <div class="value green" id="tokens">—</div></div>
    <div class="card"><div class="label">Characters saved</div>
      <div class="value green" id="chars">—</div></div>
    <div class="card"><div class="label">Tool calls</div>
      <div class="value" id="calls">—</div>
      <div class="muted" id="compressed" style="font-size:13px;margin-top:4px"></div></div>
    <div class="card"><div class="label">Avg compression</div>
      <div class="value blue" id="ratio">—</div></div>
  </div>

  <div class="section">
    <h2>Per-tool savings</h2>
    <table id="bytool">
      <thead><tr><th>Tool</th><th class="num">Calls</th>
        <th class="num">Before</th><th class="num">After</th>
        <th class="num">Saved</th></tr></thead>
      <tbody><tr><td colspan="5" class="empty">No tool calls yet —
        run a tool in your MCP client.</td></tr></tbody>
    </table>
  </div>

  <div class="row"><span class="dot" id="live"></span>
    <span class="muted" id="status">connecting…</span></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString();
function pct(x) { return (x*100).toFixed(1) + '%'; }

async function tick() {
  try {
    const s = await fetch('/stats').then(r => r.json());
    $('tokens').textContent = fmt(s.tokens_saved_estimate_total || 0);
    $('chars').textContent = fmt(s.chars_saved_total || 0);
    $('calls').textContent = fmt(s.tool_calls_total || 0);
    $('compressed').textContent = fmt(s.tool_calls_compressed || 0) + ' compressed';
    const saved = 1 - (s.compression_ratio_mean ?? 1);
    $('ratio').innerHTML = pct(Math.max(0, saved)) + '<span class="unit">smaller</span>';

    const tbody = $('bytool').querySelector('tbody');
    const rows = Object.entries(s.by_tool || {});
    if (rows.length) {
      tbody.innerHTML = rows.map(([name, t]) =>
        `<tr><td>${name}</td><td class="num">${fmt(t.calls)}</td>` +
        `<td class="num">${fmt(t.chars_before)}</td>` +
        `<td class="num">${fmt(t.chars_after)}</td>` +
        `<td class="num">${fmt(t.chars_saved)} (${t.saved_pct}%)</td></tr>`).join('');
    }
    $('live').style.background = '#3fb950';
    $('status').textContent = 'live · updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    $('live').style.background = '#f85149';
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
