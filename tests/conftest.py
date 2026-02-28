"""
Pytest configuration — sets env vars before any test module imports proxy.py
so the lifespan always gets an in-memory SQLite DB and auth is disabled by default.
"""

import os

# Must be set before proxy.py is imported so the lifespan picks it up.
os.environ.setdefault("SENTINEL_DB_PATH", ":memory:")
# Use a very high rate limit in tests so the shared slowapi counter never
# interferes with functional tests.  Individual rate-limit tests override
# this via monkeypatch.
os.environ.setdefault("SENTINEL_RATE_LIMIT", "100000/minute")
os.environ.pop("SENTINEL_API_KEY", None)
os.environ.pop("SENTINEL_UPSTREAM_URL", None)
os.environ.pop("SENTINEL_UPSTREAM_PROVIDER", None)
