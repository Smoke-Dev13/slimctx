"""Unit tests for ImageCompressor."""

from __future__ import annotations

from contextly.compressors.image import ImageCompressor


def _img(detail: str | None = None) -> dict:
    image_url: dict = {"url": "https://example.com/cat.png"}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def test_high_detail_downgraded_to_low() -> None:
    c = ImageCompressor(detail_level="low")
    res = c.compress_part(_img("high"), "what is in this image?")
    assert res.changed is True
    assert res.part["image_url"]["detail"] == "low"
    assert res.tokens_saved_estimate > 0


def test_auto_detail_downgraded() -> None:
    c = ImageCompressor(detail_level="low")
    res = c.compress_part(_img(), "describe it")
    assert res.changed is True
    assert res.part["image_url"]["detail"] == "low"


def test_already_low_unchanged() -> None:
    c = ImageCompressor(detail_level="low")
    res = c.compress_part(_img("low"), "describe it")
    assert res.changed is False


def test_detail_sensitive_query_skips_downgrade() -> None:
    c = ImageCompressor(detail_level="low")
    res = c.compress_part(_img("high"), "read the small text in the image")
    assert res.changed is False
    assert res.part["image_url"]["detail"] == "high"


def test_non_image_part_unchanged() -> None:
    c = ImageCompressor()
    part = {"type": "text", "text": "hello"}
    res = c.compress_part(part, "")
    assert res.changed is False
    assert res.part == part


def test_non_dict_part_unchanged() -> None:
    c = ImageCompressor()
    res = c.compress_part("not a dict", "")  # type: ignore[arg-type]
    assert res.changed is False


def test_original_not_mutated() -> None:
    c = ImageCompressor(detail_level="low")
    original = _img("high")
    c.compress_part(original, "describe")
    assert original["image_url"]["detail"] == "high"


def test_stats_counts_compressed_parts() -> None:
    c = ImageCompressor(detail_level="low")
    c.compress_part(_img("high"), "describe")
    c.compress_part(_img("auto"), "describe")
    c.compress_part(_img("low"), "describe")  # no change
    assert c.stats()["image_parts_compressed_total"] == 2


def test_anthropic_image_block_detail_sensitive_query() -> None:
    # Anthropic 'image' blocks have no detail field; downgrade is a no-op but
    # the part should pass through unchanged rather than error.
    c = ImageCompressor()
    part = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    res = c.compress_part(part, "describe")
    assert res.changed is False
