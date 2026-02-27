"""
Pytest configuration — sets env vars before any test module imports proxy.py
so the lifespan always gets an in-memory SQLite DB and auth is disabled by default.
"""

import os

# Must be set before proxy.py is imported so the lifespan picks it up.
os.environ.setdefault("SENTINEL_DB_PATH", ":memory:")
os.environ.pop("SENTINEL_API_KEY", None)
os.environ.pop("SENTINEL_UPSTREAM_URL", None)
