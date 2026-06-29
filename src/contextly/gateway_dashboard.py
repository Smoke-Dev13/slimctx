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
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Contextly gateway — live savings</title>
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
.badge-mcp{font-size:10px;font-weight:700;background:rgba(88,166,255,.15);
           color:var(--blue);padding:2px 7px;border-radius:20px;letter-spacing:.4px;}
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
.bar-wrap{height:4px;background:var(--line);border-radius:2px;margin-top:10px;overflow:hidden;}
.bar{height:100%;border-radius:2px;transition:width .6s ease;}
.empty-state{text-align:center;padding:32px 16px;color:var(--muted);}
.empty-state svg{opacity:.3;margin-bottom:10px;}
.empty-state p{font-size:13px;}
.empty-state code{font-size:12px;background:var(--line);border-radius:4px;padding:1px 5px;}
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
    <div>
      <span class="brand-name">Context<em>ly</em></span>
      <div class="badge-mcp" style="margin-top:3px;display:inline-block;">MCP</div>
    </div>
  </div>
  <nav class="nav">
    <div class="nav-item active">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
      Gateway
    </div>
    <div class="nav-item">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      Per-tool
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
    <span class="page-title">MCP Gateway — live tool savings</span>
    <span class="uptime-badge" id="uptime2"></span>
  </div>

  <div class="content">

    <div class="hero">
      <div class="hero-card green">
        <div class="hero-glow"></div>
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
        </div>
        <div class="hero-label">Tokens saved (est.)</div>
        <div class="hero-value" id="tokens">—</div>
        <div class="spark-wrap">
          <svg viewBox="0 0 120 28" preserveAspectRatio="none">
            <polyline id="sl-tokens" points=""/>
          </svg>
        </div>
      </div>

      <div class="hero-card green">
        <div class="hero-glow"></div>
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2"><path d="M21 15V6"/><path d="M18.5 18a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/><path d="M12 12H3"/><path d="M16 6H3"/><path d="M12 18H3"/></svg>
        </div>
        <div class="hero-label">Characters saved</div>
        <div class="hero-value" id="chars">—</div>
        <div class="hero-sub" id="chars-sub"></div>
      </div>

      <div class="hero-card neutral">
        <div class="hero-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7a8799" stroke-width="2"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>
        </div>
        <div class="hero-label">Tool calls</div>
        <div class="hero-value" id="calls">—</div>
        <div class="hero-sub" id="compressed"></div>
        <div class="bar-wrap"><div class="bar" id="calls-bar" style="width:0%;background:var(--blue)"></div></div>
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

    <div class="section-head"><h2>Per-tool breakdown</h2></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Tool</th><th class="r">Calls</th>
          <th class="r">Before</th><th class="r">After</th>
          <th class="r">Saved</th><th>Reduction</th>
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

const COLORS = ['#3fb950','#58a6ff','#e3b341','#7a8799','#f85149','#a371f7'];
const sparkData = [];
const HIST = 40;
function pushSpark(val) { sparkData.push(val); if (sparkData.length > HIST) sparkData.shift(); }
function renderSpark(polyId, data) {
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

function setRing(pctVal) {
  const circumference = 138.2;
  const offset = circumference * (1 - clamp(pctVal, 0, 1));
  $('ring-arc').style.strokeDashoffset = offset.toFixed(2);
  $('ring-pct').textContent = (pctVal * 100).toFixed(0) + '%';
}

function badgeClass(p) { return p >= 40 ? 'hi' : p >= 15 ? 'mid' : 'lo'; }

const t0 = Date.now();
function fmtUptime() {
  const s = Math.floor((Date.now() - t0) / 1000);
  const m = Math.floor(s / 60), sec = s % 60, h = Math.floor(m / 60), min = m % 60;
  return h ? 'session ' + h + 'h ' + min + 'm'
    : m ? 'session ' + m + 'm ' + sec + 's' : 'session ' + sec + 's';
}
setInterval(() => {
  const u = fmtUptime();
  $('uptime').textContent = u;
  $('uptime2').textContent = u;
}, 1000);

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

    $('chars-sub').textContent = chars ? '\\u2248 ' + Math.round(chars / 4).toLocaleString() + ' tokens' : '';
    $('compressed').textContent = comp ? fmt(comp) + ' compressed' : '';
    $('calls-bar').style.width = total ? pct(clamp(comp / total, 0, 1)) : '0%';
    $('ratio').textContent = pct(saved);
    setRing(saved);

    const rows = Object.entries(s.by_tool || {});
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
