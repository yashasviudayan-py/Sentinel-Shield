# Sentinel-Shield

A standalone AI Security Proxy that acts as a gateway between users and LLM providers (OpenAI/Anthropic). It prevents sensitive data leakage and enforces security protocols at the proxy level.

## Features

- **PII/Secret Redaction** — Scans incoming prompts for emails, SSNs, API keys, and passwords, replacing them with placeholders before forwarding to the LLM.
- **Jailbreak Detection** — Detects and blocks prompt injection attempts.
- **Response Sanitization** — Inspects LLM output to prevent accidental leakage of internal schemas or sensitive data.
- **Audit Logging** — Records every redacted event and blocked request for compliance review.

## Architecture

```
Client ──► Sentinel-Shield Proxy ──► LLM Provider
               │
               ├─ Redaction Engine
               ├─ Jailbreak Detector
               ├─ Response Sanitizer
               └─ Audit Logger
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/Sentinel-Shield.git
cd Sentinel-Shield
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Proxy

```bash
uvicorn proxy:app --host 0.0.0.0 --port 8080 --reload
```

### Test It

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "My email is john.doe@example.com, please help me."}
    ]
  }' | python3 -m json.tool
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Proxy endpoint (OpenAI-compatible) |
| `GET` | `/v1/audit` | View the audit trail |
| `GET` | `/health` | Liveness check |

## Project Roadmap

| Phase | Title | Objective |
|-------|-------|-----------|
| 1 | The Gateway Core | FastAPI proxy, request logging, basic pass-through |
| 2 | The Data Sanitizer | Regex-based PII/secret redaction, entity recognition |
| 3 | The Security Guard | Jailbreak detection, prompt safety filters |
| 4 | The Auditor | SQLite audit trail, usage tracking, metrics |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
