"""Safe mode keeps lossy compressors out of the routing chain.

In safe mode the proxy must never drop prose sentences, so prose routes to
passthrough. JSON still uses the lossless json_table compressor (every record
preserved) in both modes, and code compression (structure-preserving) stays
enabled throughout.
"""

from __future__ import annotations

import json

from contextly.compressors.registry import ContentRouter
from contextly.config import Config
from contextly.server import create_app

_JSON = json.dumps([{"id": i, "name": f"user_{i}"} for i in range(50)])

_PROSE = (
    "Machine learning models require careful tuning. The learning rate matters most. "
    "Regularization prevents overfitting on the training data. Transfer learning helps too. "
    "Ensembles reduce variance across multiple models in production systems."
)

_CODE = (
    "import os\n\n\ndef process(items):\n    # tally totals\n    return sum(i for i in items)\n\n\n"
    "def validate(items):\n    return all(isinstance(i, int) for i in items)\n"
)


def _router(safe_mode: bool) -> ContentRouter:
    config = Config(upstream_api_key="test-key", compression_enabled=True, safe_mode=safe_mode)
    app = create_app(config)
    router: ContentRouter = app.state.content_router
    return router


def test_safe_mode_json_uses_lossless_table() -> None:
    # JSON compression is lossless (json_table), so it is allowed in safe mode.
    assert _router(safe_mode=True).select(_JSON).name == "json_table"


def test_safe_mode_prose_passes_through() -> None:
    assert _router(safe_mode=True).select(_PROSE).name == "passthrough"


def test_safe_mode_code_still_compresses() -> None:
    assert _router(safe_mode=True).select(_CODE).name == "code"


def test_default_mode_json_uses_json_table() -> None:
    assert _router(safe_mode=False).select(_JSON).name == "json_table"


def test_default_mode_prose_uses_prose() -> None:
    assert _router(safe_mode=False).select(_PROSE).name == "prose"
