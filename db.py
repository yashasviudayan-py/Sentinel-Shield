"""
Sentinel-Shield — Audit Database (Phase 4)

Thread-safe SQLite persistence for audit events.
Uses stdlib only: sqlite3, json, threading, os.

Configure the database path via the SENTINEL_DB_PATH environment variable
(defaults to "sentinel_audit.db" in the working directory).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any

logger = logging.getLogger("sentinel-shield.db")

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    redactions        INTEGER NOT NULL DEFAULT 0,
    blocked           INTEGER NOT NULL DEFAULT 0,
    detail            TEXT    NOT NULL DEFAULT '',
    redaction_summary TEXT,
    threats           TEXT
);
"""

_INSERT = """
INSERT INTO audit_events
    (request_id, timestamp, redactions, blocked, detail, redaction_summary, threats)
VALUES
    (:request_id, :timestamp, :redactions, :blocked, :detail, :redaction_summary, :threats);
"""

_SELECT_ALL = """
SELECT id, request_id, timestamp, redactions, blocked, detail, redaction_summary, threats
FROM   audit_events
ORDER  BY id ASC;
"""


# ---------------------------------------------------------------------------
# AuditDB
# ---------------------------------------------------------------------------

class AuditDB:
    """Thread-safe SQLite-backed audit log.

    A single connection is reused across threads; a Lock serialises writes.
    WAL journal mode allows concurrent reads without blocking writers.
    """

    def __init__(self, path: str = "sentinel_audit.db") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.info("AuditDB ready: %s", os.path.abspath(path))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, entry: dict[str, Any]) -> None:
        """Persist one audit event to the database."""
        row = {
            "request_id": entry["request_id"],
            "timestamp": entry["timestamp"],
            "redactions": entry["redactions"],
            "blocked": int(entry["blocked"]),
            "detail": entry.get("detail", ""),
            "redaction_summary": (
                json.dumps(entry["redaction_summary"])
                if entry.get("redaction_summary")
                else None
            ),
            "threats": (
                json.dumps(entry["threats"])
                if entry.get("threats")
                else None
            ),
        }
        with self._lock:
            self._conn.execute(_INSERT, row)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict[str, Any]]:
        """Return all audit events in insertion order."""
        with self._lock:
            rows = self._conn.execute(_SELECT_ALL).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            event: dict[str, Any] = {
                "id": row["id"],
                "request_id": row["request_id"],
                "timestamp": row["timestamp"],
                "redactions": row["redactions"],
                "blocked": bool(row["blocked"]),
                "detail": row["detail"],
            }
            if row["redaction_summary"]:
                event["redaction_summary"] = json.loads(row["redaction_summary"])
            if row["threats"]:
                event["threats"] = json.loads(row["threats"])
            result.append(event)
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection (called on app shutdown)."""
        with self._lock:
            self._conn.close()
        logger.info("AuditDB closed: %s", self._path)
