"""
Sentinel-Shield — AI Security Proxy (Phase 4: SQLite Audit Persistence)

A lightweight FastAPI proxy that intercepts OpenAI-format chat completion
requests, redacts PII and secrets, runs jailbreak/prompt-injection detection,
and persists every audit event to a SQLite database.

Environment variables:
  SENTINEL_DB_PATH   Path to the SQLite database file (default: sentinel_audit.db)
"""

import os
import uuid
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from redactor import RedactionEngine, Finding
from guard import PromptGuard, GuardResult, Threat
from db import AuditDB

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
# Lifespan (open / close DB around app lifetime)
# ---------------------------------------------------------------------------
_db: AuditDB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    db_path = os.getenv("SENTINEL_DB_PATH", "sentinel_audit.db")
    _db = AuditDB(path=db_path)
    yield
    _db.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentinel-Shield",
    version="0.4.0",
    description="AI Security Proxy — intercept, redact, guard, and persist LLM traffic.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
engine = RedactionEngine()
guard = PromptGuard()


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def record_audit_event(
    request_id: str,
    redactions: int,
    blocked: bool,
    detail: str,
    redaction_summary: dict[str, int] | None = None,
    threats: list[dict] | None = None,
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
    if threats is not None:
        entry["threats"] = threats

    assert _db is not None, "AuditDB not initialised"
    _db.insert(entry)
    logger.info("AUDIT | %s", entry)


def _serialise_threats(threats: list[Threat], *, include_matched_text: bool) -> list[dict]:
    """Serialise Threat dataclasses to dicts.

    matched_text is included only when writing to the audit log (never echoed
    in HTTP responses to avoid reflecting adversarial content back to callers).
    """
    result = []
    for t in threats:
        item: dict[str, Any] = {
            "category": t.category,
            "rule_name": t.rule_name,
            "severity": t.severity,
        }
        if include_matched_text:
            item["matched_text"] = t.matched_text
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-shield"}


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> JSONResponse:
    """Intercept an OpenAI-format chat completion request.

    Pipeline: Parse → Redact → Guard → [HTTP 403 if blocked] → Audit → HTTP 200
    """
    request_id = str(uuid.uuid4())
    body: dict[str, Any] = await request.json()

    logger.info(
        "INCOMING | id=%s model=%s messages=%d",
        request_id,
        body.get("model", "unknown"),
        len(body.get("messages", [])),
    )

    # --- 1. Redact ---
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

    # --- 2. Guard (runs on sanitised messages) ---
    guard_result: GuardResult = guard.inspect_messages(sanitised_messages)

    threats_for_audit = _serialise_threats(guard_result.threats, include_matched_text=True)
    threats_for_response = _serialise_threats(guard_result.threats, include_matched_text=False)

    if guard_result.blocked:
        logger.warning(
            "BLOCKED | id=%s reason=%r threats=%d",
            request_id,
            guard_result.reason,
            len(guard_result.threats),
        )
        record_audit_event(
            request_id=request_id,
            redactions=redaction_count,
            blocked=True,
            detail=guard_result.reason,
            redaction_summary=redaction_summary or None,
            threats=threats_for_audit,
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "type": "request_blocked",
                    "message": guard_result.reason,
                    "code": "policy_violation",
                },
                "_sentinel": {
                    "request_id": request_id,
                    "blocked": True,
                    "guard": {
                        "blocked": True,
                        "threats": threats_for_response,
                        "reason": guard_result.reason,
                    },
                },
            },
        )

    # --- 3. Audit (pass-through) ---
    record_audit_event(
        request_id=request_id,
        redactions=redaction_count,
        blocked=False,
        detail=(
            f"Redacted {redaction_count} item(s) from prompt"
            if redaction_count > 0
            else "No sensitive data detected"
        ),
        redaction_summary=redaction_summary or None,
        threats=threats_for_audit if threats_for_audit else None,
    )

    # --- 4. Build simulated response ---
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
            "guard": {
                "blocked": False,
                "threats": threats_for_response,
                "reason": guard_result.reason,
            },
        },
    }

    return JSONResponse(content=simulated_response)


@app.get("/v1/audit")
async def get_audit_log() -> list[dict[str, Any]]:
    """Return the full audit trail from SQLite."""
    assert _db is not None, "AuditDB not initialised"
    return _db.get_all()
