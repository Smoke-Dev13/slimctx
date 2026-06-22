"""Safe mode keeps lossy compressors out of the routing chain.

In safe mode the proxy must never drop JSON records or prose sentences, so
those content types route to passthrough. Code compression (comment/whitespace
stripping, structure-preserving) stays enabled.
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


def test_safe_mode_json_passes_through() -> None:
    assert _router(safe_mode=True).select(_JSON).name == "passthrough"


def test_safe_mode_prose_passes_through() -> None:
    assert _router(safe_mode=True).select(_PROSE).name == "passthrough"


def test_safe_mode_code_still_compresses() -> None:
    assert _router(safe_mode=True).select(_CODE).name == "code"


def test_default_mode_json_uses_json_smart() -> None:
    assert _router(safe_mode=False).select(_JSON).name == "json_smart"


def test_default_mode_prose_uses_prose() -> None:
    assert _router(safe_mode=False).select(_PROSE).name == "prose"
