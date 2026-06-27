"""Context importance scoring and message reordering.

Scores each message on three axes (stdlib only, zero new deps):
  1. Recency — exponential decay by distance from the end
  2. Query relevance — TF-IDF cosine similarity against the last user query
  3. Role weight — system > user > assistant > tool

``reorder()`` stable-sorts by score descending while always keeping the
system prompt first and the latest user message last, preserving coherence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_ROLE_WEIGHT: dict[str, float] = {
    "system": 1.0,
    "user": 0.9,
    "assistant": 0.7,
    "tool": 0.5,
}

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should "
    "may might shall can need dare ought used to of in on at by for with about against between "
    "into through during before after above below to from up down out off over under again "
    "further then once i you he she it we they what which who whom this that these those am "
    "and or but if as because as until while nor not no so yet both either neither just than "
    "too very s t don didn won can't couldn't shouldn't wouldn't mustn't isn't aren't wasn't "
    "weren't hasn't haven't hadn't doesn't don't didn't".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def _tfidf_cosine(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q_freq: dict[str, int] = {}
    for t in query_tokens:
        q_freq[t] = q_freq.get(t, 0) + 1
    d_freq: dict[str, int] = {}
    for t in doc_tokens:
        d_freq[t] = d_freq.get(t, 0) + 1
    dot = sum(q_freq[t] * d_freq.get(t, 0) for t in q_freq)
    norm_q = math.sqrt(sum(v * v for v in q_freq.values()))
    norm_d = math.sqrt(sum(v * v for v in d_freq.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


def _extract_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


@dataclass
class ScoredMessage:
    message: dict[str, Any]
    score: float
    original_index: int


class MessageScorer:
    """Score and reorder messages by relevance, recency, and role.

    Thread-safe: stateless after construction.
    """

    def __init__(self, decay: float = 0.85) -> None:
        self._decay = decay

    def score_messages(self, messages: list[dict[str, Any]], query: str) -> list[ScoredMessage]:
        """Score each message on recency, relevance, and role weight.

        Args:
            messages: List of chat message dicts (role + content).
            query: The last user query text, used for relevance scoring.

        Returns:
            List of ScoredMessage, one per input message, in original order.
        """
        n = len(messages)
        if n == 0:
            return []
        query_tokens = _tokenize(query)
        scored: list[ScoredMessage] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            # System messages always score 1.0 so they sort to the top.
            if role == "system":
                scored.append(ScoredMessage(msg, 1.0, i))
                continue
            # Recency: distance from end; position 0 = oldest.
            recency = self._decay ** (n - 1 - i)
            # Query relevance
            doc_tokens = _tokenize(_extract_text(msg))
            relevance = _tfidf_cosine(query_tokens, doc_tokens)
            role_w = _ROLE_WEIGHT.get(role, 0.5)
            score = 0.4 * recency + 0.4 * relevance + 0.2 * role_w
            scored.append(ScoredMessage(msg, score, i))
        return scored

    def reorder(
        self,
        messages: list[dict[str, Any]],
        query: str,
        *,
        min_messages: int = 5,
        cache_stable: bool = False,
        recent_window: int = 2,
    ) -> list[dict[str, Any]]:
        """Reorder messages by importance score.

        System prompt is always first; the latest user message is always last.
        No reordering is applied when the message count is below min_messages.

        When ``cache_stable`` is True, the older history is kept in its original
        chronological order (so the prompt-cache prefix stays byte-identical
        across turns) and only the last ``recent_window`` messages — the fresh,
        non-cacheable tail — are relevance-sorted. This reconciles relevance
        reordering with prompt caching, which both OpenAI and Anthropic key off a
        stable prefix.

        Args:
            messages: Original message list.
            query: Last user query for relevance scoring.
            min_messages: Minimum list length before reordering is attempted.
            cache_stable: Keep the older prefix chronological; sort only the tail.
            recent_window: Size of the fresh tail that may be reordered.

        Returns:
            Reordered message list, or the original list if unchanged.
        """
        if len(messages) < min_messages:
            return messages

        if cache_stable:
            return self._reorder_cache_stable(messages, query, recent_window)

        scored = self.score_messages(messages, query)

        # Identify pinned positions: system messages + the last user message.
        system_msgs = [s for s in scored if s.message.get("role") == "system"]
        last_user_idx = max(
            (s.original_index for s in scored if s.message.get("role") == "user"),
            default=None,
        )

        middle = [
            s
            for s in scored
            if s.message.get("role") != "system" and s.original_index != last_user_idx
        ]
        middle.sort(key=lambda s: s.score, reverse=True)

        result: list[dict[str, Any]] = [s.message for s in system_msgs]
        result.extend(s.message for s in middle)
        if last_user_idx is not None:
            result.append(messages[last_user_idx])

        return result

    def _reorder_cache_stable(
        self,
        messages: list[dict[str, Any]],
        query: str,
        recent_window: int,
    ) -> list[dict[str, Any]]:
        """Sort only the trailing ``recent_window`` messages; keep prefix as-is.

        The stable prefix (everything before the tail) preserves chronological
        order so the prompt-cache prefix is byte-identical between turns. Within
        the tail, the latest user message is still pinned last.
        """
        if recent_window <= 1:
            return messages

        boundary = len(messages) - recent_window
        if boundary <= 0:
            return messages

        prefix = messages[:boundary]
        tail = messages[boundary:]

        last_user_offset = max(
            (i for i, m in enumerate(tail) if m.get("role") == "user"),
            default=None,
        )

        scored_tail = self.score_messages(tail, query)
        middle = [s for s in scored_tail if s.original_index != last_user_offset]
        middle.sort(key=lambda s: s.score, reverse=True)

        result: list[dict[str, Any]] = [*prefix]
        result.extend(s.message for s in middle)
        if last_user_offset is not None:
            result.append(tail[last_user_offset])

        return result
