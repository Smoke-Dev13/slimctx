"""Unit tests for ContentRouter."""

from __future__ import annotations

import pytest

from contextly.compressors.base import Compressor, CompressResult
from contextly.compressors.passthrough import PassthroughCompressor
from contextly.compressors.registry import ContentRouter


class _AlwaysApply(Compressor):
    """Stub that always applies."""

    @property
    def name(self) -> str:
        return "always"

    def should_apply(self, content: str, query: str = "") -> bool:
        return True

    def compress(self, content: str, query: str = "") -> CompressResult:
        compressed = content[:10]
        return CompressResult(
            content=compressed,
            original_length=len(content),
            compressed_length=len(compressed),
            compressor_name=self.name,
        )


class _NeverApply(Compressor):
    """Stub that never applies."""

    @property
    def name(self) -> str:
        return "never"

    def should_apply(self, content: str, query: str = "") -> bool:
        return False

    def compress(self, content: str, query: str = "") -> CompressResult:
        raise AssertionError("should never be called")


class _Raises(Compressor):
    """Stub whose should_apply raises."""

    @property
    def name(self) -> str:
        return "raises"

    def should_apply(self, content: str, query: str = "") -> bool:
        raise RuntimeError("boom")

    def compress(self, content: str, query: str = "") -> CompressResult:
        raise AssertionError("should never be called")


@pytest.fixture
def router() -> ContentRouter:
    return ContentRouter()


def test_empty_router_falls_back_to_passthrough(router: ContentRouter) -> None:
    result = router.select("any content")
    assert isinstance(result, PassthroughCompressor)


def test_registered_compressor_selected_when_applicable(router: ContentRouter) -> None:
    router.register(_AlwaysApply())
    selected = router.select("hello world")
    assert selected.name == "always"


def test_never_apply_skipped(router: ContentRouter) -> None:
    router.register(_NeverApply())
    selected = router.select("hello world")
    assert isinstance(selected, PassthroughCompressor)


def test_first_match_wins(router: ContentRouter) -> None:
    router.register(_AlwaysApply())
    router.register(_NeverApply())
    selected = router.select("content")
    assert selected.name == "always"


def test_error_in_should_apply_falls_through(router: ContentRouter) -> None:
    router.register(_Raises())
    router.register(_AlwaysApply())
    selected = router.select("content")
    assert selected.name == "always"


def test_error_only_compressor_falls_back_to_passthrough(router: ContentRouter) -> None:
    router.register(_Raises())
    selected = router.select("content")
    assert isinstance(selected, PassthroughCompressor)
