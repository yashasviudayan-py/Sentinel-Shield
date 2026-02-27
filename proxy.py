"""
Sentinel-Shield — AI Security Proxy (Phase 5: Upstream LLM Forwarding)

Pipeline per request:
  Parse → Redact → Guard → [HTTP 403 if blocked] → Forward to upstream LLM
  (or simulated response if SENTINEL_UPSTREAM_URL is unset) → Audit → HTTP 200

Environment variables:
  SENTINEL_DB_PATH       SQLite file path (default: sentinel_audit.db)
  SENTINEL_API_KEY       Bearer token required on /v1/* routes (auth disabled if unset)
  SENTINEL_UPSTREAM_URL  Base URL of the OpenAI-compatible upstream LLM
                         e.g. http://localhost:11434/v1  (Ollama)
                              http://localhost:1234/v1   (LM Studio)
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from db import AuditDB
from guard import GuardResult, PromptGuard, Threat
from redactor import Finding, RedactionEngine

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
# Lifespan — open / close DB around app lifetime
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
    version="0.5.0",
    description="AI Security Proxy — redact, guard, and forward LLM traffic.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
engine = RedactionEngine()
guard = PromptGuard()

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


async def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """FastAPI dependency: validates bearer token when SENTINEL_API_KEY is set.

    Auth is disabled (open) when the env var is absent or empty.
    Uses secrets.compare_digest to prevent timing-based attacks.
    """
    required = os.getenv("SENTINEL_API_KEY", "")
    if not required:
        return  # Auth disabled
    if credentials is None or not secrets.compare_digest(credentials.credentials, required):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

async def record_audit_event(
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
    await asyncio.to_thread(_db.insert, entry)
    logger.info("AUDIT | %s", entry)


def _serialise_threats(threats: list[Threat], *, include_matched_text: bool) -> list[dict]:
    """Serialise Threat dataclasses to dicts.

    matched_text is included only for the audit log — never echoed in HTTP
    responses to avoid reflecting adversarial content back to callers.
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
# Upstream forwarding
# ---------------------------------------------------------------------------

async def _forward_to_upstream(body: dict[str, Any], upstream_url: str) -> dict[str, Any]:
    """POST *body* to the upstream LLM and return its JSON response."""
    url = upstream_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.post(url, json=body)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-shield"}


@app.post("/v1/chat/completions", dependencies=[Depends(_verify_api_key)])
async def proxy_chat_completions(request: Request) -> JSONResponse:
    """Intercept an OpenAI-format chat completion request.

    Pipeline: Parse → Redact → Guard → [HTTP 403 if blocked] → Upstream/Simulated → Audit → HTTP 200
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
        await record_audit_event(
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

    # --- 3. Forward to upstream LLM (or fall back to simulated response) ---
    sentinel_meta: dict[str, Any] = {
        "request_id": request_id,
        "redactions": redaction_count,
        "redaction_summary": redaction_summary,
        "blocked": False,
        "guard": {
            "blocked": False,
            "threats": threats_for_response,
            "reason": guard_result.reason,
        },
    }

    upstream_url = os.getenv("SENTINEL_UPSTREAM_URL", "")
    if upstream_url:
        try:
            upstream_body = {**body, "messages": sanitised_messages}
            upstream_data = await _forward_to_upstream(upstream_body, upstream_url)
            upstream_data["_sentinel"] = sentinel_meta

            await record_audit_event(
                request_id=request_id,
                redactions=redaction_count,
                blocked=False,
                detail=f"Forwarded to upstream: {upstream_url}",
                redaction_summary=redaction_summary or None,
                threats=threats_for_audit if threats_for_audit else None,
            )
            return JSONResponse(content=upstream_data)

        except httpx.HTTPError as exc:
            logger.error("UPSTREAM_ERROR | id=%s %s", request_id, exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": "upstream_error",
                        "message": "Upstream LLM request failed",
                    },
                    "_sentinel": {"request_id": request_id},
                },
            )

    # --- 4. Simulated response (no upstream configured) ---
    await record_audit_event(
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

    if findings:
        types_found = ", ".join(sorted(redaction_summary.keys()))
        assistant_note = (
            f"[Sentinel-Shield] Request intercepted. "
            f"{redaction_count} item(s) redacted ({types_found}). "
            "Set SENTINEL_UPSTREAM_URL to forward to a real LLM."
        )
    else:
        assistant_note = (
            "[Sentinel-Shield] Request intercepted. "
            "No sensitive data detected. "
            "Set SENTINEL_UPSTREAM_URL to forward to a real LLM."
        )

    return JSONResponse(content={
        "id": f"chatcmpl-{request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-3.5-turbo"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_note},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_sentinel": sentinel_meta,
    })


@app.get("/v1/audit", dependencies=[Depends(_verify_api_key)])
async def get_audit_log() -> list[dict[str, Any]]:
    """Return the full audit trail from SQLite."""
    assert _db is not None, "AuditDB not initialised"
    return await asyncio.to_thread(_db.get_all)
