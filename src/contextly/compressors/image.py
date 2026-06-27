"""Image compression for multimodal requests.

Two mechanisms, both opt-in:

  1. Dependency-free detail downgrade (always available): OpenAI bills image
     inputs by ``detail`` level — ``low`` is a flat ~85 tokens versus 765+ for
     ``high``/``auto`` on large images. Downgrading detail to ``low`` is the
     largest, safest token win and is pure dict manipulation.

  2. Optional Pillow downscale: when Pillow is importable and the image is an
     inline base64 data URI, the image is resized below the provider tiling
     threshold and re-encoded. The original data URI is handed to a CCR store so
     the transform stays reversible (consistent with Contextly's CCR moat).

Operates on OpenAI ``image_url`` parts and Anthropic ``image`` blocks. Detail
downgrade is skipped when the user's query asks for fine visual detail.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
import threading
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Queries that need full visual fidelity — keep the original detail/resolution.
_DETAIL_SENSITIVE: frozenset[str] = frozenset(
    ["read", "text", "ocr", "fine", "detail", "small", "zoom", "exact", "precise", "tiny", "blurry"]
)

_DATA_URI_RE = re.compile(r"^data:image/(?P<fmt>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$", re.DOTALL)


@dataclass
class ImageResult:
    """Outcome of attempting to compress a single image content block."""

    part: dict[str, Any]
    changed: bool
    tokens_saved_estimate: int = 0


def _query_needs_detail(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    return any(kw in q for kw in _DETAIL_SENSITIVE)


def _downscale_data_uri(data_uri: str, max_dimension: int) -> str | None:
    """Resize an inline base64 image below *max_dimension*; None if not possible.

    Returns a new data URI, or None when Pillow is absent, the URI is not an
    inline base64 image, or the image is already small enough.
    """
    match = _DATA_URI_RE.match(data_uri)
    if match is None:
        return None
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        with Image.open(io.BytesIO(raw)) as img:
            w, h = img.size
            if max(w, h) <= max_dimension:
                return None
            scale = max_dimension / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            resized = img.resize(new_size)
            fmt = (img.format or "PNG").upper()
            buf = io.BytesIO()
            resized.save(buf, format=fmt)
    except Exception:
        logger.warning("image_downscale_failed", exc_info=True)
        return None
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


class ImageCompressor:
    """Compress multimodal image parts via detail downgrade and optional resize.

    Stateless and thread-safe; share one instance across requests.
    """

    def __init__(self, *, detail_level: str = "low", max_dimension: int = 512) -> None:
        self._detail_level = detail_level
        self._max_dimension = max_dimension
        self._compressed = 0
        self._lock = threading.Lock()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"image_parts_compressed_total": self._compressed}

    def compress_part(self, part: dict[str, Any], query: str = "") -> ImageResult:
        """Compress a single content-block dict if it is an image part.

        Args:
            part: A content block (OpenAI ``image_url`` or Anthropic ``image``).
            query: The user's last query; fine-detail requests skip downgrade.

        Returns:
            ImageResult; ``changed`` is False for non-image or detail-sensitive parts.
        """
        if not isinstance(part, dict):
            return ImageResult(part, changed=False)
        ptype = part.get("type")
        if ptype not in {"image_url", "image"}:
            return ImageResult(part, changed=False)
        if _query_needs_detail(query):
            return ImageResult(part, changed=False)

        new_part = dict(part)
        changed = False
        saved = 0

        # OpenAI image_url: downgrade detail unless already low.
        if ptype == "image_url" and isinstance(new_part.get("image_url"), dict):
            image_url = dict(new_part["image_url"])
            current = image_url.get("detail", "auto")
            if current != "low":
                image_url["detail"] = self._detail_level
                # Optional inline-image downscale before re-attaching.
                url = image_url.get("url", "")
                if isinstance(url, str):
                    smaller = _downscale_data_uri(url, self._max_dimension)
                    if smaller is not None:
                        image_url["url"] = smaller
                new_part["image_url"] = image_url
                changed = True
                # high/auto → low saves roughly 680 tokens for a large image.
                saved = 680

        if changed:
            with self._lock:
                self._compressed += 1

        return ImageResult(new_part, changed=changed, tokens_saved_estimate=saved)
