"""
Sentinel-Shield — AI Security Proxy (v0.8.0)

Pipeline per request:
  Parse → [413 if too large] → Redact → Guard → [403 if blocked]
      → Forward to upstream LLM (or simulated)
      → Scan response (redact PII + guard)
      → Audit → HTTP 200

Streaming pipeline (stream=true):
  Parse → [413 if too large] → Redact → Guard → [403 if blocked]
      → Collect upstream SSE → Scan assembled response
      → Re-emit sanitized SSE → Audit

Environment variables:
  SENTINEL_DB_PATH         SQLite file path           (default: sentinel_audit.db)
  SENTINEL_API_KEY         Bearer token for /v1/*     (auth disabled if unset)
  SENTINEL_UPSTREAM_URL    OpenAI-compatible LLM URL  (e.g. http://localhost:11434/v1)
  SENTINEL_RATE_LIMIT      slowapi limit string        (default: 60/minute)
  SENTINEL_WEBHOOK_URL     Alert webhook URL           (disabled if unset)
  SENTINEL_MAX_BODY_BYTES  Max request body in bytes   (default: 1048576 = 1 MB)
  SENTINEL_TRUSTED_ROLES   Comma-separated roles that  (default: system)
                           skip the guard (still redacted)
"""

from __future__ import annotations

import asyncio
import json
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
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from db import AuditDB
from guard import GuardResult, PromptGuard, Threat
from redactor import Finding, RedactionEngine, _ner_available

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
# Prometheus metrics
# ---------------------------------------------------------------------------
_requests_total = Counter(
    "sentinel_requests_total",
    "Total proxy requests processed",
    ["blocked", "streamed"],
)
_redactions_total = Counter(
    "sentinel_redactions_total",
    "Total entity redactions by type",
    ["entity_type"],
)
_threats_total = Counter(
    "sentinel_threats_total",
    "Total threats detected by category and severity",
    ["category", "severity"],
)
_request_duration = Histogram(
    "sentinel_request_duration_seconds",
    "End-to-end request processing time in seconds",
)
_upstream_errors_total = Counter(
    "sentinel_upstream_errors_total",
    "Total upstream LLM errors",
)

# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------
_MAX_BODY_DEFAULT = 1_048_576  # 1 MB

# ---------------------------------------------------------------------------
# Rate limiter (must be created before app)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


def _rate_limit() -> str:
    """Read rate limit from env at request time so tests can override it."""
    return os.getenv("SENTINEL_RATE_LIMIT", "60/minute")


async def _handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "type": "rate_limit_exceeded",
                "message": str(exc),
                "code": "rate_limit_exceeded",
            }
        },
    )


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
    version="0.8.0",
    description="AI Security Proxy — redact, guard, rate-limit, and forward LLM traffic.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)  # type: ignore[arg-type]

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
    """Validate bearer token when SENTINEL_API_KEY is set.

    Auth is disabled (open) when the env var is absent or empty.
    Uses secrets.compare_digest to prevent timing-based attacks.
    """
    required = os.getenv("SENTINEL_API_KEY", "")
    if not required:
        return
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
    response: dict[str, Any] | None = None,
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
    if response is not None:
        entry["response"] = response

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
# Webhook
# ---------------------------------------------------------------------------

# Strong references to in-flight webhook tasks so they aren't garbage-collected
# before completion.  The done callback removes each task from the set.
_webhook_tasks: set[asyncio.Task] = set()


def _schedule_webhook(payload: dict[str, Any]) -> None:
    """Schedule _fire_webhook as a fire-and-forget background task."""
    task = asyncio.create_task(_fire_webhook(payload))
    _webhook_tasks.add(task)
    task.add_done_callback(_webhook_tasks.discard)


async def _fire_webhook(payload: dict[str, Any]) -> None:
    """POST a JSON alert payload to SENTINEL_WEBHOOK_URL.

    Silently swallows all errors so a broken webhook never blocks a response.
    matched_text is never included in webhook payloads.
    """
    url = os.getenv("SENTINEL_WEBHOOK_URL", "")
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            await http_client.post(url, json=payload)
        logger.info(
            "WEBHOOK_SENT | event=%s request_id=%s",
            payload.get("event"),
            payload.get("request_id"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("WEBHOOK_ERROR | %s", exc)


# ---------------------------------------------------------------------------
# Response scanning
# ---------------------------------------------------------------------------

def _scan_upstream_response(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Redact PII and check for threats in the upstream LLM response.

    Modifies *data* in-place (response content fields).
    Returns (data, response_meta) where response_meta is suitable for
    both the audit log and the _sentinel.response field.

    If the response content contains a HIGH-severity threat (the model is
    attempting prompt injection downstream), the content is replaced with a
    sanitisation notice. matched_text is never included in response_meta.
    """
    total_redactions = 0
    resp_redaction_summary: dict[str, int] = {}
    seen_threats: dict[tuple[str, str], Threat] = {}
    response_blocked = False
    block_reason = ""

    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue

        # 1. Redact PII in the response content
        redact_result = engine.redact(content)
        msg["content"] = redact_result.text
        for f in redact_result.findings:
            total_redactions += f.count
            resp_redaction_summary[f.entity_type] = (
                resp_redaction_summary.get(f.entity_type, 0) + f.count
            )

        # 2. Guard on the original (pre-redaction) content so we catch raw patterns
        guard_result = guard.inspect(content)
        for threat in guard_result.threats:
            key = (threat.category, threat.rule_name)
            if key not in seen_threats:
                seen_threats[key] = threat

        if guard_result.blocked:
            response_blocked = True
            block_reason = guard_result.reason
            logger.warning("RESPONSE_BLOCKED | reason=%r", block_reason)
            msg["content"] = (
                "[Sentinel-Shield] Response sanitised: "
                "potential prompt injection detected in model output."
            )

    unique_threats = list(seen_threats.values())
    response_meta: dict[str, Any] = {
        "redactions": total_redactions,
        "redaction_summary": resp_redaction_summary,
        "blocked": response_blocked,
        "guard": {
            "blocked": response_blocked,
            "threats": _serialise_threats(unique_threats, include_matched_text=False),
            "reason": block_reason,
        },
    }
    return data, response_meta


# ---------------------------------------------------------------------------
# Upstream forwarding — non-streaming
# ---------------------------------------------------------------------------

async def _forward_to_upstream(body: dict[str, Any], upstream_url: str) -> dict[str, Any]:
    """POST *body* to the upstream LLM and return its JSON response."""
    url = upstream_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.post(url, json=body)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Upstream forwarding — streaming
# ---------------------------------------------------------------------------

async def _collect_and_scan_stream(
    body: dict[str, Any],
    upstream_url: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Stream from upstream LLM, buffer all SSE chunks, scan, return sanitized content.

    Buffers the entire response before scanning so the guard and redaction
    engines always operate on complete text. This adds latency proportional
    to the upstream response time but is necessary for correct security checks.

    Returns:
        (sanitized_content, first_chunk_meta, response_meta)

    Raises:
        httpx.HTTPError on upstream failure.
    """
    url = upstream_url.rstrip("/") + "/chat/completions"
    full_content = ""
    first_chunk: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        async with http_client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not first_chunk:
                    first_chunk = chunk
                delta_content = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content") or ""
                )
                full_content += delta_content

    # Build a complete response object so _scan_upstream_response can operate normally
    full_resp: dict[str, Any] = {
        "id": first_chunk.get("id", ""),
        "object": "chat.completion",
        "created": first_chunk.get("created", int(time.time())),
        "model": first_chunk.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
    }
    full_resp, response_meta = _scan_upstream_response(full_resp)
    sanitized_content = full_resp["choices"][0]["message"]["content"]
    return sanitized_content, first_chunk, response_meta


# ---------------------------------------------------------------------------
# SSE response builder
# ---------------------------------------------------------------------------

def _build_sse_response(
    content: str,
    sentinel_meta: dict[str, Any],
    comp_id: str,
    created: int,
    model: str,
) -> StreamingResponse:
    """Return a StreamingResponse emitting OpenAI-compatible SSE chunks.

    Emits three chunks:
      1. role-announcing delta (role: "assistant", content: "")
      2. content delta with the full sanitized text
      3. finish chunk (finish_reason: "stop") carrying _sentinel metadata
    Followed by data: [DONE].
    """
    def _generate():
        role_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        content_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        finish_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "_sentinel": sentinel_meta,
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Sentinel-Request-ID": sentinel_meta["request_id"],
        },
    )


# ---------------------------------------------------------------------------
# Health-check helpers
# ---------------------------------------------------------------------------

async def _check_db() -> dict[str, str]:
    """Verify the AuditDB is open and responding."""
    if _db is None:
        return {"status": "error", "detail": "not initialized"}
    try:
        await asyncio.to_thread(_db.get_all, limit=1)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


async def _check_upstream(url: str) -> dict[str, str]:
    """Ping the upstream base URL; any HTTP response counts as reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            await http_client.get(url)
        return {"status": "ok", "url": url}
    except Exception:  # noqa: BLE001
        return {"status": "unreachable", "url": url}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness + readiness check.

    Runs DB connectivity, NER model, and optional upstream reachability
    checks in parallel. Returns 200 when all critical checks pass, 503 if
    the database is unavailable.
    """
    upstream_url = os.getenv("SENTINEL_UPSTREAM_URL", "")

    coros: list = [_check_db()]
    if upstream_url:
        coros.append(_check_upstream(upstream_url))

    results = await asyncio.gather(*coros)
    db_status = results[0]
    upstream_status = results[1] if upstream_url else {"status": "unconfigured"}

    ner_status = (
        {"status": "ok", "model": "en_core_web_sm"}
        if _ner_available
        else {"status": "unavailable"}
    )

    overall = "ok" if db_status["status"] == "ok" else "degraded"
    return JSONResponse(
        status_code=200 if overall == "ok" else 503,
        content={
            "status": overall,
            "service": "sentinel-shield",
            "version": "0.8.0",
            "checks": {
                "database": db_status,
                "ner_model": ner_status,
                "upstream": upstream_status,
            },
        },
    )


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", dependencies=[Depends(_verify_api_key)])
@limiter.limit(_rate_limit)
async def proxy_chat_completions(request: Request) -> Response:
    """Intercept an OpenAI-format chat completion request.

    Pipeline: Parse → Redact → Guard → [403 if blocked]
              → Upstream/Simulated → Scan response → Audit → 200/SSE
    """
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

    # --- 0. Size guard ---
    max_body = int(os.getenv("SENTINEL_MAX_BODY_BYTES", _MAX_BODY_DEFAULT))
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_body:
        return JSONResponse(
            status_code=413,
            content={"error": {"type": "request_too_large",
                               "message": f"Request body exceeds {max_body} bytes",
                               "code": "request_too_large"}},
        )
    raw_body = await request.body()
    if len(raw_body) > max_body:
        return JSONResponse(
            status_code=413,
            content={"error": {"type": "request_too_large",
                               "message": f"Request body exceeds {max_body} bytes",
                               "code": "request_too_large"}},
        )

    body: dict[str, Any] = await request.json()
    is_streaming = bool(body.get("stream"))

    logger.info(
        "INCOMING | id=%s model=%s messages=%d stream=%s",
        request_id,
        body.get("model", "unknown"),
        len(body.get("messages", [])),
        is_streaming,
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
        for entity_type, count in redaction_summary.items():
            _redactions_total.labels(entity_type=entity_type).inc(count)

    # --- 2. Guard (runs on sanitised messages, skipping trusted roles) ---
    trusted_roles = {
        r.strip().lower()
        for r in os.getenv("SENTINEL_TRUSTED_ROLES", "system").split(",")
        if r.strip()
    }
    messages_for_guard = [
        m for m in sanitised_messages
        if m.get("role", "").lower() not in trusted_roles
    ]
    guard_result: GuardResult = guard.inspect_messages(messages_for_guard)

    threats_for_audit = _serialise_threats(guard_result.threats, include_matched_text=True)
    threats_for_response = _serialise_threats(guard_result.threats, include_matched_text=False)

    for t in guard_result.threats:
        _threats_total.labels(category=t.category, severity=t.severity).inc()

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
        _schedule_webhook({
            "event": "request_blocked",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": guard_result.reason,
            "threats": threats_for_response,
            "redaction_summary": redaction_summary,
        })
        _requests_total.labels(blocked="true", streamed=str(is_streaming).lower()).inc()
        _request_duration.observe(time.monotonic() - t_start)
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

    # --- 3. Build base _sentinel metadata (request-level) ---
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

    # --- 4. Forward to upstream LLM (or fall back to simulated response) ---
    upstream_url = os.getenv("SENTINEL_UPSTREAM_URL", "")
    if upstream_url:
        try:
            upstream_body = {**body, "messages": sanitised_messages}

            if is_streaming:
                # --- Streaming path: buffer, scan, re-emit ---
                sanitized_content, first_chunk, response_meta = (
                    await _collect_and_scan_stream(upstream_body, upstream_url)
                )
                sentinel_meta["response"] = response_meta

                if response_meta.get("blocked"):
                    _schedule_webhook({
                        "event": "response_blocked",
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reason": response_meta["guard"]["reason"],
                        "threats": response_meta["guard"]["threats"],
                    })

                await record_audit_event(
                    request_id=request_id,
                    redactions=redaction_count,
                    blocked=False,
                    detail=f"Streamed from upstream: {upstream_url}",
                    redaction_summary=redaction_summary or None,
                    threats=threats_for_audit if threats_for_audit else None,
                    response=response_meta,
                )
                _requests_total.labels(blocked="false", streamed="true").inc()
                _request_duration.observe(time.monotonic() - t_start)
                return _build_sse_response(
                    sanitized_content,
                    sentinel_meta,
                    comp_id=first_chunk.get("id", f"chatcmpl-{request_id[:8]}"),
                    created=first_chunk.get("created", int(time.time())),
                    model=first_chunk.get("model", body.get("model", "unknown")),
                )

            else:
                # --- Non-streaming path ---
                upstream_data = await _forward_to_upstream(upstream_body, upstream_url)

                # --- 5. Scan the upstream response ---
                upstream_data, response_meta = _scan_upstream_response(upstream_data)
                sentinel_meta["response"] = response_meta

                if response_meta.get("blocked"):
                    _schedule_webhook({
                        "event": "response_blocked",
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reason": response_meta["guard"]["reason"],
                        "threats": response_meta["guard"]["threats"],
                    })

                upstream_data["_sentinel"] = sentinel_meta

                await record_audit_event(
                    request_id=request_id,
                    redactions=redaction_count,
                    blocked=False,
                    detail=f"Forwarded to upstream: {upstream_url}",
                    redaction_summary=redaction_summary or None,
                    threats=threats_for_audit if threats_for_audit else None,
                    response=response_meta,
                )
                _requests_total.labels(blocked="false", streamed="false").inc()
                _request_duration.observe(time.monotonic() - t_start)
                return JSONResponse(content=upstream_data)

        except httpx.HTTPError as exc:
            logger.error("UPSTREAM_ERROR | id=%s %s", request_id, exc)
            _upstream_errors_total.inc()
            _requests_total.labels(blocked="false", streamed=str(is_streaming).lower()).inc()
            _request_duration.observe(time.monotonic() - t_start)
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

    # --- 6. Simulated response (no upstream configured) ---
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

    comp_id = f"chatcmpl-{request_id[:8]}"
    _requests_total.labels(blocked="false", streamed=str(is_streaming).lower()).inc()
    _request_duration.observe(time.monotonic() - t_start)

    if is_streaming:
        return _build_sse_response(
            assistant_note,
            sentinel_meta,
            comp_id=comp_id,
            created=int(time.time()),
            model=body.get("model", "gpt-3.5-turbo"),
        )

    return JSONResponse(content={
        "id": comp_id,
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
async def get_audit_log(
    limit: int | None = None,
    offset: int = 0,
    blocked: bool | None = None,
) -> list[dict[str, Any]]:
    """Return audit events with optional pagination and filtering.

    Query params:
      limit   — max rows to return
      offset  — rows to skip
      blocked — true/false to filter by blocked status
    """
    assert _db is not None, "AuditDB not initialised"
    return await asyncio.to_thread(_db.get_all, limit=limit, offset=offset, blocked=blocked)
