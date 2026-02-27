"""
Sentinel-Shield — Redaction Engine (Phase 2)

Regex-based PII/secret redaction + optional spaCy NER for named entities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("sentinel-shield.redactor")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    entity_type: str
    original: str
    placeholder: str
    count: int


@dataclass
class RedactionResult:
    text: str
    findings: list[Finding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns  (entity_type, compiled_pattern, placeholder)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "EMAIL",
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "[EMAIL_REDACTED]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "[CREDIT_CARD_REDACTED]",
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP_REDACTED]",
    ),
    (
        "JWT",
        re.compile(
            r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"
        ),
        "[JWT_REDACTED]",
    ),
    (
        "API_KEY_OPENAI",
        re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
        "[API_KEY_REDACTED]",
    ),
    (
        "API_KEY_GITHUB",
        re.compile(r"\b(?:ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{82})\b"),
        "[API_KEY_REDACTED]",
    ),
    (
        "API_KEY_AWS",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[API_KEY_REDACTED]",
    ),
    (
        "API_KEY_SLACK",
        re.compile(r"\bxox[baprs]-[a-zA-Z0-9\-]+\b"),
        "[API_KEY_REDACTED]",
    ),
    (
        "PASSWORD_INLINE",
        re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S+"),
        "[PASSWORD_REDACTED]",
    ),
]


# ---------------------------------------------------------------------------
# spaCy NER (optional — graceful fallback if model is unavailable)
# ---------------------------------------------------------------------------

_NER_LABEL_MAP: dict[str, str] = {
    "PERSON": "[PERSON_REDACTED]",
    "ORG":    "[ORG_REDACTED]",
    "GPE":    "[LOCATION_REDACTED]",
}

_nlp = None
_ner_available = False

try:
    import spacy  # noqa: PLC0415

    _nlp = spacy.load("en_core_web_sm")
    _ner_available = True
    logger.info("spaCy NER loaded: en_core_web_sm")
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "spaCy NER unavailable — skipping named-entity redaction: %s", exc
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RedactionEngine:
    """Apply regex and (optionally) NER-based redaction to text."""

    def redact(self, text: str) -> RedactionResult:
        """Redact PII and secrets from *text*, returning a RedactionResult."""
        findings: list[Finding] = []
        current = text

        # 1. Regex pass
        for entity_type, pattern, placeholder in _PATTERNS:
            matches = pattern.findall(current)
            if matches:
                current, n = pattern.subn(placeholder, current)
                findings.append(
                    Finding(
                        entity_type=entity_type,
                        original=matches[0],
                        placeholder=placeholder,
                        count=n,
                    )
                )

        # 2. spaCy NER pass — runs on already-redacted text so substitution
        #    placeholders are not double-processed as entities.
        if _ner_available and _nlp is not None:
            doc = _nlp(current)

            spans_by_label: dict[str, list] = {}
            for ent in doc.ents:
                if ent.label_ in _NER_LABEL_MAP:
                    spans_by_label.setdefault(ent.label_, []).append(ent)

            for label, spans in spans_by_label.items():
                placeholder = _NER_LABEL_MAP[label]
                # Replace from end → start to keep character offsets valid
                for span in sorted(spans, key=lambda s: s.start_char, reverse=True):
                    current = (
                        current[: span.start_char]
                        + placeholder
                        + current[span.end_char :]
                    )
                findings.append(
                    Finding(
                        entity_type=label,
                        original=spans[0].text,
                        placeholder=placeholder,
                        count=len(spans),
                    )
                )

        return RedactionResult(text=current, findings=findings)

    def redact_messages(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[Finding]]:
        """Redact all content fields in *messages*.

        Returns the sanitised messages list and aggregated findings
        (counts accumulated across all messages).
        """
        accumulated: dict[str, Finding] = {}
        sanitised: list[dict] = []

        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str):
                result = self.redact(content)
                new_msg["content"] = result.text
                for f in result.findings:
                    if f.entity_type in accumulated:
                        existing = accumulated[f.entity_type]
                        accumulated[f.entity_type] = Finding(
                            entity_type=existing.entity_type,
                            original=existing.original,
                            placeholder=existing.placeholder,
                            count=existing.count + f.count,
                        )
                    else:
                        accumulated[f.entity_type] = f
            sanitised.append(new_msg)

        return sanitised, list(accumulated.values())
