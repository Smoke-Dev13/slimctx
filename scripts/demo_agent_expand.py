#!/usr/bin/env python3
"""Demo: compress a tool output, then expand it back on demand.

Simulates the agent/MCP-gateway flow with no network and no API key. A noisy
"tool output" (repetitive logs + a JSON record dump) is compressed through the
proxy's /v1/compress endpoint, then fully recovered through /v1/expand — showing
that aggressive token savings never become permanent data loss.

    python scripts/demo_agent_expand.py
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app
from contextly.tokenizer.registry import get_tokenizer

_LOGS = "\n".join(
    f"2026-06-22T10:{i // 60:02d}:{i % 60:02d} INFO GET /api/orders/{1000 + i} -> 200 in {i % 80}ms"
    for i in range(400)
)
_JSON = json.dumps(
    [{"id": 5000 + i, "sku": f"SKU-{i:04d}", "qty": i % 7, "warehouse": "EU-1"} for i in range(150)]
)


def _show(client: TestClient, label: str, content: str, tok: object) -> None:
    r = client.post("/v1/compress", json={"content": content}).json()
    before, after = tok.count(content), tok.count(r["content"])  # type: ignore[attr-defined]
    saved = 100 * (1 - after / before) if before else 0
    print(f"\n=== {label} ===")
    print(f"  compressor : {r['compressor']}")
    print(f"  tokens     : {before} -> {after}  (-{saved:.0f}%)")
    print(f"  expand_ref : {r['ccr_key']}")
    if r["ccr_key"]:
        expanded = client.get(f"/v1/expand/{r['ccr_key']}").json()
        ok = expanded["found"] and expanded["content"] == content
        print(
            f"  expand()   : recovered original verbatim = {ok} ({len(expanded['content'])} chars)"
        )


def main() -> None:
    client = TestClient(create_app(Config(upstream_api_key="unused", compression_enabled=True)))
    tok = get_tokenizer("gpt-4o")
    print("Contextly — compress tool output, expand on demand (offline demo)")
    _show(client, "Server logs (400 lines)", _LOGS, tok)
    _show(client, "JSON record dump (150 records)", _JSON, tok)
    print("\nTakeaway: the agent sends a fraction of the tokens, and can call")
    print("expand(expand_ref) to get the full original back whenever it needs a detail.")


if __name__ == "__main__":
    main()
