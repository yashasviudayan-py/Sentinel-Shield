<p align="center">
  <img src="https://img.shields.io/badge/Sentinel--Shield-v1.0.0-2563EB?style=for-the-badge&logo=shield&logoColor=white" alt="version"/>
</p>

<p align="center">
  <a href="https://github.com/yashasviudayan-py/Sentinel-Shield/actions/workflows/ci.yml">
    <img src="https://github.com/yashasviudayan-py/Sentinel-Shield/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="fastapi"/>
  <img src="https://img.shields.io/badge/tests-153%20passed-22c55e?logo=pytest&logoColor=white" alt="tests"/>
  <img src="https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white" alt="openai-compatible"/>
  <img src="https://img.shields.io/badge/Anthropic-compatible-D97706?logo=anthropic&logoColor=white" alt="anthropic-compatible"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="docker"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license"/>
</p>

<p align="center">
  <b>An AI Security Proxy that sits between your app and any LLM.</b><br/>
  Redacts PII, blocks jailbreaks, scans responses, logs everything — zero changes to your existing code.
</p>

---

## What it does

```
Your App  ──►  Sentinel-Shield  ──►  OpenAI / Anthropic / Ollama
                     │
                     ├─ Strip PII & secrets from prompts
                     ├─ Block jailbreak / prompt-injection attempts
                     ├─ Scan LLM responses for leaked data
                     ├─ Enforce rate limits
                     ├─ Persist full audit trail (SQLite)
                     └─ Emit Prometheus metrics + webhook alerts
```

---

## Features

| Category | Capability |
|---|---|
| **PII Redaction** | Email, SSN, credit card, phone, IP, JWT, AWS/OpenAI/GitHub/Slack API keys, passwords |
| **NER Redaction** | PERSON, ORG, GPE via spaCy `en_core_web_sm` (graceful fallback if not installed) |
| **Jailbreak Guard** | 13 built-in rules across PROMPT_INJECTION, JAILBREAK_ROLEPLAY, CONTEXT_HIJACK, SYSTEM_OVERRIDE, OBFUSCATION |
| **Custom Rules** | Load additional guard rules from a JSON file at startup |
| **Response Scanning** | Redacts PII and guards against model-injected threats in LLM output |
| **Streaming** | Full SSE pass-through — buffer, scan, re-emit sanitised chunks |
| **Multi-provider** | OpenAI-compatible upstream **or** Anthropic Messages API (auto-converts format) |
| **Auth** | Bearer token (`SENTINEL_API_KEY`) with constant-time comparison |
| **Rate limiting** | Per-IP via slowapi, configurable string (e.g. `60/minute`) |
| **Audit log** | SQLite, WAL mode, pagination, filter by blocked status |
| **Audit export** | Bulk download as NDJSON or CSV |
| **Token tracking** | prompt / completion / total tokens per request, aggregated at `/v1/usage` |
| **Health check** | DB + NER model + upstream reachability in parallel |
| **Metrics** | Prometheus counters/histograms at `/metrics` |
| **Webhooks** | Fire-and-forget POST on block events |
| **Size limits** | 413 on oversized bodies (configurable) |
| **Trusted roles** | Skip guard for trusted message roles (e.g. `system`) |
| **Docker** | `Dockerfile` + `docker-compose.yml` included |

---

## Quick Start

### Run locally (no LLM needed)

```bash
git clone https://github.com/yashasviudayan-py/Sentinel-Shield.git
cd Sentinel-Shield
pip install -r requirements.txt
uvicorn proxy:app --reload --port 8000
```

```bash
# PII gets stripped, simulated response returned
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"My email is bob@secret.com"}]}' \
  | python3 -m json.tool
```

### Run with Ollama (local LLM)

```bash
ollama serve
ollama pull llama3.2

SENTINEL_UPSTREAM_URL=http://localhost:11434/v1 uvicorn proxy:app --port 8000
```

### Run with Docker (full stack)

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.2
# Proxy live at http://localhost:8000
```

---

## Integration

It's a **drop-in replacement** — change one line in your app.

**Python (OpenAI SDK)**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-sentinel-key",          # or any string when auth is disabled
)
response = client.chat.completions.create(model="gpt-4", messages=[...])
```

**LangChain**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="your-sentinel-key")
```

**Node.js / TypeScript**
```typescript
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "your-sentinel-key" });
```

**Environment variable (zero code changes)**
```bash
OPENAI_BASE_URL=http://sentinel-shield:8000/v1
OPENAI_API_KEY=your-sentinel-key
```

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | optional | Proxy endpoint — OpenAI-compatible |
| `GET` | `/v1/audit` | optional | Paginated audit log |
| `GET` | `/v1/audit/export` | optional | Bulk export as NDJSON or CSV |
| `GET` | `/v1/usage` | optional | Aggregated token usage totals |
| `GET` | `/health` | none | Liveness + readiness check |
| `GET` | `/metrics` | none | Prometheus metrics scrape |

Auth is enforced on all `/v1/*` routes when `SENTINEL_API_KEY` is set.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `SENTINEL_UPSTREAM_URL` | _(none)_ | Upstream LLM base URL — e.g. `http://localhost:11434/v1` or `https://api.openai.com/v1` |
| `SENTINEL_UPSTREAM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `SENTINEL_API_KEY` | _(none)_ | Bearer token protecting `/v1/*`; auth disabled when unset |
| `SENTINEL_DB_PATH` | `sentinel_audit.db` | SQLite file path (`:memory:` for tests) |
| `SENTINEL_RATE_LIMIT` | `60/minute` | slowapi limit string per IP |
| `SENTINEL_WEBHOOK_URL` | _(none)_ | POST alert payload here on every block event |
| `SENTINEL_MAX_BODY_BYTES` | `1048576` | Request body size limit (413 if exceeded) |
| `SENTINEL_TRUSTED_ROLES` | `system` | Comma-separated roles that skip the guard |
| `SENTINEL_GUARD_RULES_FILE` | _(none)_ | Path to JSON file with custom guard rules |
| `ANTHROPIC_API_KEY` | _(none)_ | Required when `SENTINEL_UPSTREAM_PROVIDER=anthropic` |

---

## Custom Guard Rules

Add your own detection rules without touching the source code:

```json
{
  "rules": [
    {
      "category": "CUSTOM",
      "rule_name": "block_competitor",
      "pattern": "\\bcompetitor_name\\b",
      "flags": ["IGNORECASE"],
      "severity": "HIGH"
    }
  ]
}
```

```bash
SENTINEL_GUARD_RULES_FILE=./my_rules.json uvicorn proxy:app --port 8000
```

---

## Response shape

Every response carries a `_sentinel` metadata block:

```json
{
  "choices": [...],
  "usage": {...},
  "_sentinel": {
    "request_id": "3f2a1b...",
    "blocked": false,
    "redactions": 2,
    "redaction_summary": { "EMAIL": 1, "SSN": 1 },
    "guard": { "blocked": false, "threats": [], "reason": "" },
    "usage": { "model": "gpt-4", "prompt_tokens": 28, "completion_tokens": 42, "total_tokens": 70 },
    "response": { "blocked": false, "redactions": 0, "guard": { "blocked": false } }
  }
}
```

Blocked requests return HTTP `403`:

```json
{
  "error": { "type": "request_blocked", "message": "HIGH-severity threat(s) detected: ignore_instructions", "code": "policy_violation" },
  "_sentinel": { "blocked": true, "guard": { "threats": [{ "category": "PROMPT_INJECTION", "rule_name": "ignore_instructions", "severity": "HIGH" }] } }
}
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
# 153 tests across test_proxy.py, test_guard.py, test_redactor.py
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
