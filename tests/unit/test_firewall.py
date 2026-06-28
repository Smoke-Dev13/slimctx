"""Unit tests for SecretRedactor (semantic firewall)."""

from __future__ import annotations

from contextly.ccr import CCRStore
from contextly.firewall import SecretRedactor, _luhn_valid


def _r() -> SecretRedactor:
    return SecretRedactor()


# ── Luhn ─────────────────────────────────────────────────────────────────────────


def test_luhn_valid_visa() -> None:
    assert _luhn_valid("4242424242424242") is True


def test_luhn_invalid() -> None:
    assert _luhn_valid("4242424242424241") is False


def test_luhn_too_short() -> None:
    assert _luhn_valid("123") is False


# ── individual secret types ──────────────────────────────────────────────────────


def test_openai_key_redacted() -> None:
    res = _r().redact("my key is sk-abcdefghij1234567890ABCD here")
    assert "sk-abcdefghij" not in res.redacted_text
    assert res.count == 1
    assert res.findings[0].type == "openai_key"


def test_anthropic_key_redacted_with_specific_type() -> None:
    res = _r().redact("key sk-ant-abcdefghij1234567890ABCDEFG done")
    assert res.count == 1
    assert res.findings[0].type == "anthropic_key"
    assert "sk-ant-" not in res.redacted_text


def test_aws_access_key_redacted() -> None:
    res = _r().redact("AKIAIOSFODNN7EXAMPLE in config")
    assert res.count == 1
    assert res.findings[0].type == "aws_access_key"


def test_private_key_block_redacted() -> None:
    res = _r().redact("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n")
    assert res.count == 1
    assert res.findings[0].type == "private_key_block"


def test_email_redacted() -> None:
    res = _r().redact("contact alice@example.com please")
    assert "alice@example.com" not in res.redacted_text
    assert res.findings[0].type == "email"


def test_ssn_redacted() -> None:
    res = _r().redact("SSN 123-45-6789 on file")
    assert res.count == 1
    assert res.findings[0].type == "ssn"


def test_valid_credit_card_redacted() -> None:
    res = _r().redact("card 4242 4242 4242 4242 expiry")
    assert res.count == 1
    assert res.findings[0].type == "credit_card"


def test_invalid_credit_card_not_redacted() -> None:
    # 16 digits failing Luhn → left untouched
    res = _r().redact("number 1234 5678 9012 3456 here")
    assert res.count == 0
    assert "1234 5678 9012 3456" in res.redacted_text


def test_plain_prose_no_false_positive() -> None:
    res = _r().redact("The quick brown fox jumps over the lazy dog twice.")
    assert res.count == 0


# ── placeholder behaviour ────────────────────────────────────────────────────────


def test_placeholder_stable_across_calls() -> None:
    r = _r()
    a = r.redact("sk-abcdefghij1234567890ABCD")
    b = r.redact("sk-abcdefghij1234567890ABCD")
    assert a.redacted_text == b.redacted_text


def test_placeholder_format() -> None:
    res = _r().redact("sk-abcdefghij1234567890ABCD")
    assert res.redacted_text.startswith("[REDACTED:openai_key:")


def test_reversible_stores_in_ccr() -> None:
    store = CCRStore()
    r = _r()
    secret = "sk-abcdefghij1234567890ABCD"
    res = r.redact(secret, ccr_store=store)
    assert "contextly_expand" in res.redacted_text
    # extract the key and confirm round-trip
    import re

    key = re.search(r'contextly_expand\("([^"]+)"\)', res.redacted_text).group(1)  # type: ignore[union-attr]
    assert store.retrieve(key) == secret


# ── redact_messages + stats ──────────────────────────────────────────────────────


def test_redact_messages_string_and_list() -> None:
    r = _r()
    messages = [
        {"role": "system", "content": "no secrets here"},
        {"role": "user", "content": "my key sk-abcdefghij1234567890ABCD"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "email me at bob@example.com"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        },
    ]
    out, total = r.redact_messages(messages)
    assert total == 2
    assert "sk-abcdefghij" not in out[1]["content"]
    assert "bob@example.com" not in out[2]["content"][0]["text"]
    # non-text part untouched
    assert out[2]["content"][1]["type"] == "image_url"


def test_stats_counters() -> None:
    r = _r()
    msg = {"role": "user", "content": "sk-abcdefghij1234567890ABCD x bob@example.com"}
    r.redact_messages([msg])
    st = r.stats()
    assert st["secrets_redacted_total"] == 2
    assert st["requests_with_secrets_total"] == 1


def test_stats_initial() -> None:
    assert _r().stats() == {
        "secrets_redacted_total": 0,
        "requests_with_secrets_total": 0,
        "response_secrets_redacted_total": 0,
        "responses_with_secrets_total": 0,
    }


def test_response_redaction_counters() -> None:
    r = _r()
    # The model echoed a secret back in its reply — caught on the outbound side.
    leaked = r.redact("here is the key: sk-abcdefghij1234567890ABCD")
    assert leaked.count == 1
    r.record_response_redaction(leaked.count)
    st = r.stats()
    assert st["response_secrets_redacted_total"] == 1
    assert st["responses_with_secrets_total"] == 1
    # Inbound counters untouched.
    assert st["secrets_redacted_total"] == 0


def test_record_response_redaction_ignores_zero() -> None:
    r = _r()
    r.record_response_redaction(0)
    assert r.stats()["responses_with_secrets_total"] == 0
