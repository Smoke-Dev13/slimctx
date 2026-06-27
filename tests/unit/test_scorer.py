"""Unit tests for MessageScorer."""

from __future__ import annotations

from contextly.scorer import MessageScorer, _tfidf_cosine, _tokenize


def _scorer() -> MessageScorer:
    return MessageScorer()


# ── tokenize / cosine helpers ──────────────────────────────────────────────────


def test_tokenize_lowercases_and_strips_stopwords() -> None:
    tokens = _tokenize("The quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens


def test_tfidf_cosine_identical() -> None:
    t = _tokenize("database query results")
    assert abs(_tfidf_cosine(t, t) - 1.0) < 1e-9


def test_tfidf_cosine_disjoint() -> None:
    assert _tfidf_cosine(_tokenize("hello world"), _tokenize("foo bar baz")) == 0.0


def test_tfidf_cosine_empty() -> None:
    assert _tfidf_cosine([], ["foo"]) == 0.0
    assert _tfidf_cosine(["foo"], []) == 0.0


# ── score_messages ─────────────────────────────────────────────────────────────


def test_system_message_scores_one() -> None:
    s = _scorer()
    msgs = [{"role": "system", "content": "You are helpful."}]
    scored = s.score_messages(msgs, "anything")
    assert scored[0].score == 1.0


def test_recency_later_message_scores_higher() -> None:
    s = _scorer()
    msgs = [
        {"role": "user", "content": "old message"},
        {"role": "user", "content": "new message"},
    ]
    scored = s.score_messages(msgs, "")
    assert scored[1].score > scored[0].score


def test_relevance_boosts_score() -> None:
    s = _scorer()
    msgs = [
        {"role": "assistant", "content": "The database query returned results."},
        {"role": "assistant", "content": "The weather is nice today."},
    ]
    scored = s.score_messages(msgs, "database query")
    assert scored[0].score > scored[1].score


def test_role_weight_user_over_tool() -> None:
    s = _scorer()
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "x"},
    ]
    scored = s.score_messages(msgs, "")
    # same recency position, same relevance — role weight decides
    assert scored[0].score > scored[1].score


def test_empty_messages() -> None:
    s = _scorer()
    assert s.score_messages([], "query") == []


# ── reorder ────────────────────────────────────────────────────────────────────


def test_reorder_system_always_first() -> None:
    s = _scorer()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "latest question"},
        {"role": "tool", "content": "tool output"},
    ]
    result = s.reorder(msgs, "latest question")
    assert result[0]["role"] == "system"


def test_reorder_latest_user_always_last() -> None:
    s = _scorer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "resp"},
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool"},
    ]
    result = s.reorder(msgs, "latest")
    assert result[-1]["content"] == "latest"


def test_reorder_below_min_messages_unchanged() -> None:
    s = _scorer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    result = s.reorder(msgs, "q", min_messages=5)
    assert result is msgs


def test_reorder_preserves_all_messages() -> None:
    s = _scorer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "tool", "content": "d"},
    ]
    result = s.reorder(msgs, "c")
    assert len(result) == len(msgs)
    original_contents = {m["content"] for m in msgs}
    result_contents = {m["content"] for m in result}
    assert original_contents == result_contents


def test_reorder_no_user_message() -> None:
    s = _scorer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "tool", "content": "d"},
    ]
    result = s.reorder(msgs, "query")
    assert result[0]["role"] == "system"
    assert len(result) == 5


def test_reorder_list_content_messages() -> None:
    s = _scorer()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "image question"}]},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool output"},
    ]
    result = s.reorder(msgs, "latest")
    assert result[-1]["content"] == "latest"
    assert len(result) == 5
