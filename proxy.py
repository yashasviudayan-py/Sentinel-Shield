"""
Sentinel-Shield — AI Security Proxy (Phase 1: Gateway Core)

A lightweight FastAPI proxy that intercepts OpenAI-format chat completion
requests, redacts sensitive data (emails), logs the event, and returns
a simulated response.
"""

import re
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sentinel-shield")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentinel-Shield",
    version="0.1.0",
    description="AI Security Proxy — intercept, redact, and audit LLM traffic.",
)

# ---------------------------------------------------------------------------
# Redaction engine (Phase 1: email only)
# ---------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
)

REDACTION_PLACEHOLDER = "[EMAIL_REDACTED]"


def redact_emails(text: str) -> tuple[str, int]:
    """Replace all email addresses in *text* with a placeholder.

    Returns the sanitised text and the count of redactions made.
    """
    sanitised, count = EMAIL_PATTERN.subn(REDACTION_PLACEHOLDER, text)
    return sanitised, count


def redact_message_content(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Walk the messages array and redact emails from every content field.

    Returns the cleaned messages list and total redaction count.
    """
    total_redactions = 0
    sanitised_messages: list[dict[str, Any]] = []

    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content", "")
        if isinstance(content, str):
            new_msg["content"], count = redact_emails(content)
            total_redactions += count
        sanitised_messages.append(new_msg)

    return sanitised_messages, total_redactions


# ---------------------------------------------------------------------------
# Audit log (in-memory for Phase 1; SQLite in Phase 4)
# ---------------------------------------------------------------------------
audit_log: list[dict[str, Any]] = []


def record_audit_event(
    request_id: str,
    redactions: int,
    blocked: bool,
    detail: str,
) -> None:
    entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "redactions": redactions,
        "blocked": blocked,
        "detail": detail,
    }
    audit_log.append(entry)
    logger.info("AUDIT | %s", entry)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-shield"}


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> JSONResponse:
    """Intercept an OpenAI-format chat completion request.

    1. Parse and log the incoming body.
    2. Redact emails from all message content.
    3. Return a simulated completion response with redaction metadata.
    """
    request_id = str(uuid.uuid4())
    body: dict[str, Any] = await request.json()

    # --- log raw request (excluding sensitive headers) ---
    logger.info(
        "INCOMING | id=%s model=%s messages=%d",
        request_id,
        body.get("model", "unknown"),
        len(body.get("messages", [])),
    )

    # --- redact ---
    messages = body.get("messages", [])
    sanitised_messages, redaction_count = redact_message_content(messages)

    if redaction_count > 0:
        logger.warning(
            "REDACTED | id=%s count=%d", request_id, redaction_count
        )
        record_audit_event(
            request_id=request_id,
            redactions=redaction_count,
            blocked=False,
            detail=f"Redacted {redaction_count} email(s) from prompt",
        )

    # --- simulate LLM response ---
    # In Phase 2+ this will forward the sanitised payload to the real LLM.
    simulated_response = {
        "id": f"chatcmpl-{request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-3.5-turbo"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "[Sentinel-Shield] Request intercepted. "
                        f"{redaction_count} email(s) redacted. "
                        "Forwarding to upstream LLM is not yet enabled."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_sentinel": {
            "request_id": request_id,
            "redactions": redaction_count,
            "blocked": False,
        },
    }

    return JSONResponse(content=simulated_response)


@app.get("/v1/audit")
async def get_audit_log() -> list[dict[str, Any]]:
    """Return the in-memory audit trail (Phase 1 convenience endpoint)."""
    return audit_log
