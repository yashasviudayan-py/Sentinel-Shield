"""Tests for the RedactionEngine (Phase 2)."""

import pytest
from redactor import RedactionEngine, Finding

engine = RedactionEngine()


# ---------------------------------------------------------------------------
# Regex pattern coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_type,expected_placeholder", [
    ("email me at user@example.com please",       "EMAIL",          "[EMAIL_REDACTED]"),
    ("my SSN is 123-45-6789",                     "SSN",            "[SSN_REDACTED]"),
    ("card number 4111 1111 1111 1111 thanks",     "CREDIT_CARD",    "[CREDIT_CARD_REDACTED]"),
    ("call me at (555) 867-5309",                  "PHONE",          "[PHONE_REDACTED]"),
    ("server is at 192.168.1.100",                 "IP_ADDRESS",     "[IP_REDACTED]"),
    # JWT — 3-segment base64url token
    (
        "token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "JWT",
        "[JWT_REDACTED]",
    ),
    ("key: sk-abcdefghijklmnopqrstuv",             "API_KEY_OPENAI", "[API_KEY_REDACTED]"),
    ("token ghp_" + "a" * 36,                      "API_KEY_GITHUB", "[API_KEY_REDACTED]"),
    ("access key AKIAIOSFODNN7EXAMPLE",             "API_KEY_AWS",    "[API_KEY_REDACTED]"),
    ("bot token xoxb-123-456-abc",                  "API_KEY_SLACK",  "[API_KEY_REDACTED]"),
    ("password: MyS3cr3tPass!",                     "PASSWORD_INLINE","[PASSWORD_REDACTED]"),
    ("pwd=hunter2",                                 "PASSWORD_INLINE","[PASSWORD_REDACTED]"),
])
def test_regex_pattern_detected_and_replaced(text, expected_type, expected_placeholder):
    result = engine.redact(text)
    found_types = [f.entity_type for f in result.findings]
    assert expected_type in found_types, f"{expected_type} not found in {found_types}"
    assert expected_placeholder in result.text, f"Placeholder not substituted in: {result.text!r}"


def test_no_pii_returns_original_text():
    text = "What is the capital of France?"
    result = engine.redact(text)
    assert result.text == text
    assert result.findings == []


def test_multiple_patterns_in_one_string():
    text = "Email user@example.com and SSN 123-45-6789"
    result = engine.redact(text)
    types = {f.entity_type for f in result.findings}
    assert "EMAIL" in types
    assert "SSN" in types
    assert "user@example.com" not in result.text
    assert "123-45-6789" not in result.text


# ---------------------------------------------------------------------------
# redact_messages
# ---------------------------------------------------------------------------

def test_redact_messages_aggregates_counts():
    messages = [
        {"role": "user", "content": "my email is a@b.com"},
        {"role": "user", "content": "also c@d.com and e@f.com"},
    ]
    _, findings = engine.redact_messages(messages)
    email_finding = next(f for f in findings if f.entity_type == "EMAIL")
    assert email_finding.count == 3


def test_redact_messages_preserves_non_content_fields():
    messages = [
        {"role": "user", "content": "my email is a@b.com", "name": "alice"},
        {"role": "assistant", "content": "noted"},
    ]
    sanitised, _ = engine.redact_messages(messages)
    assert sanitised[0]["role"] == "user"
    assert sanitised[0]["name"] == "alice"
    assert sanitised[1]["content"] == "noted"


def test_redact_messages_empty_list():
    sanitised, findings = engine.redact_messages([])
    assert sanitised == []
    assert findings == []


def test_finding_count_reflects_occurrences():
    result = engine.redact("a@b.com and c@d.com")
    email = next(f for f in result.findings if f.entity_type == "EMAIL")
    assert email.count == 2
