"""
Sentinel-Shield — AI Security Proxy (Phase 2: Data Sanitizer)

A lightweight FastAPI proxy that intercepts OpenAI-format chat completion
requests, redacts PII and secrets via a multi-pattern redaction engine
(regex + optional spaCy NER), logs the event, and returns a simulated
response with redaction metadata.
"""

import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from redactor import RedactionEngine, Finding

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
    version="0.2.0",
    description="AI Security Proxy — intercept, redact, and audit LLM traffic.",
)

# ---------------------------------------------------------------------------
# Redaction engine (Phase 2: full PII + NER)
# ---------------------------------------------------------------------------
engine = RedactionEngine()


# ---------------------------------------------------------------------------
# Audit log (in-memory for Phase 1–3; SQLite in Phase 4)
# ---------------------------------------------------------------------------
audit_log: list[dict[str, Any]] = []


def record_audit_event(
    request_id: str,
    redactions: int,
    blocked: bool,
    detail: str,
    redaction_summary: dict[str, int] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "redactions": redactions,
        "blocked": blocked,
        "detail": detail,
    }
    if redaction_summary:
        entry["redaction_summary"] = redaction_summary
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
    2. Redact PII/secrets from all message content.
    3. Return a simulated completion response with redaction metadata.
    """
    request_id = str(uuid.uuid4())
    body: dict[str, Any] = await request.json()

    logger.info(
        "INCOMING | id=%s model=%s messages=%d",
        request_id,
        body.get("model", "unknown"),
        len(body.get("messages", [])),
    )

    # --- redact ---
    messages = body.get("messages", [])
    sanitised_messages, findings = engine.redact_messages(messages)

    redaction_count = sum(f.count for f in findings)
    redaction_summary: dict[str, int] = {f.entity_type: f.count for f in findings}

    if redaction_count > 0:
        logger.warning(
            "REDACTED | id=%s count=%d types=%s",
            request_id,
            redaction_count,
            list(redaction_summary.keys()),
        )
        record_audit_event(
            request_id=request_id,
            redactions=redaction_count,
            blocked=False,
            detail=f"Redacted {redaction_count} item(s) from prompt",
            redaction_summary=redaction_summary,
        )

    # --- build a human-readable summary for the simulated assistant message ---
    if findings:
        types_found = ", ".join(sorted(redaction_summary.keys()))
        assistant_note = (
            f"[Sentinel-Shield] Request intercepted. "
            f"{redaction_count} item(s) redacted ({types_found}). "
            "Forwarding to upstream LLM is not yet enabled."
        )
    else:
        assistant_note = (
            "[Sentinel-Shield] Request intercepted. "
            "No sensitive data detected. "
            "Forwarding to upstream LLM is not yet enabled."
        )

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
                    "content": assistant_note,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_sentinel": {
            "request_id": request_id,
            "redactions": redaction_count,
            "redaction_summary": redaction_summary,
            "blocked": False,
        },
    }

    return JSONResponse(content=simulated_response)


@app.get("/v1/audit")
async def get_audit_log() -> list[dict[str, Any]]:
    """Return the in-memory audit trail."""
    return audit_log
