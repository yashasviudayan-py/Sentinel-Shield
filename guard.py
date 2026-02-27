"""
Sentinel-Shield — Prompt Guard (Phase 3)

Detects jailbreak attempts and prompt-injection attacks in sanitised messages.
Runs after redaction on already-cleaned text (PII/secrets are already removed).
Uses stdlib only (re, dataclasses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Threat:
    category: str          # PROMPT_INJECTION | JAILBREAK_ROLEPLAY | CONTEXT_HIJACK | SYSTEM_OVERRIDE | OBFUSCATION
    rule_name: str         # snake_case rule identifier
    matched_text: str      # first regex match — audit log only, never echoed in HTTP response
    severity: Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class GuardResult:
    blocked: bool
    threats: list[Threat] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Rule table — compiled once at import time
# ---------------------------------------------------------------------------
# Each entry: (category, rule_name, compiled_pattern, severity)

_RULES: list[tuple[str, str, re.Pattern, str]] = [
    # --- PROMPT_INJECTION ---
    (
        "PROMPT_INJECTION",
        "ignore_instructions",
        re.compile(r"ignore (previous|prior|earlier) instructions?", re.IGNORECASE),
        "HIGH",
    ),
    (
        "PROMPT_INJECTION",
        "new_instructions",
        re.compile(r"new instructions?:|your new task is", re.IGNORECASE),
        "HIGH",
    ),
    (
        "PROMPT_INJECTION",
        "override_system",
        re.compile(r"override (your )?system prompt", re.IGNORECASE),
        "HIGH",
    ),
    # --- JAILBREAK_ROLEPLAY ---
    (
        "JAILBREAK_ROLEPLAY",
        "dan_mode",
        re.compile(r"DAN mode|do anything now", re.IGNORECASE),
        "HIGH",
    ),
    (
        "JAILBREAK_ROLEPLAY",
        "developer_mode",
        re.compile(r"developer mode enabled", re.IGNORECASE),
        "HIGH",
    ),
    (
        "JAILBREAK_ROLEPLAY",
        "persona_switch",
        re.compile(r"you are now|pretend to be|act as if you have no", re.IGNORECASE),
        "MEDIUM",
    ),
    (
        "JAILBREAK_ROLEPLAY",
        "hypothetical_bypass",
        re.compile(r"hypothetically|in a fictional world|as a thought experiment", re.IGNORECASE),
        "LOW",
    ),
    # --- CONTEXT_HIJACK ---
    (
        "CONTEXT_HIJACK",
        "separator_injection",
        re.compile(r"^(#{3,}|---+|===+|<<<+)", re.MULTILINE),
        "MEDIUM",
    ),
    (
        "CONTEXT_HIJACK",
        "token_smuggling",
        re.compile(r"<\|im_start\|>|<\|endoftext\|>|\[INST\]|\[/INST\]"),
        "HIGH",
    ),
    # --- SYSTEM_OVERRIDE ---
    (
        "SYSTEM_OVERRIDE",
        "disregard_training",
        re.compile(r"disregard your training|forget your guidelines|ignore your rules", re.IGNORECASE),
        "HIGH",
    ),
    (
        "SYSTEM_OVERRIDE",
        "jailbreak_keyword",
        re.compile(r"jailbreak|bypass the filter|bypass restrictions", re.IGNORECASE),
        "MEDIUM",
    ),
    # --- OBFUSCATION ---
    (
        "OBFUSCATION",
        "base64_blob",
        re.compile(r"[A-Za-z0-9+/=]{40,}"),
        "MEDIUM",
    ),
    (
        "OBFUSCATION",
        "hex_encoding",
        re.compile(r"(\\x[0-9a-fA-F]{2}){2,}"),
        "MEDIUM",
    ),
]


# ---------------------------------------------------------------------------
# Blocking policy
# ---------------------------------------------------------------------------

def _blocking_policy(threats: list[Threat]) -> tuple[bool, str]:
    """Decide whether to block based on the collected threats.

    Rules:
    - Any HIGH severity threat  → block
    - 2+ distinct MEDIUM rules  → block
    - Otherwise                 → pass (threats still recorded in metadata)
    """
    high_threats = [t for t in threats if t.severity == "HIGH"]
    if high_threats:
        rule_names = ", ".join(t.rule_name for t in high_threats)
        return True, f"HIGH-severity threat(s) detected: {rule_names}"

    medium_rules = {t.rule_name for t in threats if t.severity == "MEDIUM"}
    if len(medium_rules) >= 2:
        return True, f"Multiple MEDIUM-severity threats detected: {', '.join(sorted(medium_rules))}"

    return False, ""


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class PromptGuard:
    """Inspect text (or message lists) for jailbreak / prompt-injection patterns."""

    def inspect(self, text: str) -> GuardResult:
        """Run all rules against *text* and apply blocking policy."""
        threats: list[Threat] = []

        for category, rule_name, pattern, severity in _RULES:
            match = pattern.search(text)
            if match:
                threats.append(
                    Threat(
                        category=category,
                        rule_name=rule_name,
                        matched_text=match.group(0),
                        severity=severity,  # type: ignore[arg-type]
                    )
                )

        blocked, reason = _blocking_policy(threats)
        return GuardResult(blocked=blocked, threats=threats, reason=reason)

    def inspect_messages(self, messages: list[dict]) -> GuardResult:
        """Inspect all messages; deduplicate threats by (category, rule_name).

        Deduplication prevents a single repeated phrase from inflating the
        MEDIUM count across multiple messages and triggering a spurious block.
        """
        seen: set[tuple[str, str]] = set()
        all_threats: list[Threat] = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            result = self.inspect(content)
            for threat in result.threats:
                key = (threat.category, threat.rule_name)
                if key not in seen:
                    seen.add(key)
                    all_threats.append(threat)

        blocked, reason = _blocking_policy(all_threats)
        return GuardResult(blocked=blocked, threats=all_threats, reason=reason)
