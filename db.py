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
    threats           TEXT,
    response          TEXT
);
"""

_INSERT = """
INSERT INTO audit_events
    (request_id, timestamp, redactions, blocked, detail, redaction_summary, threats, response)
VALUES
    (:request_id, :timestamp, :redactions, :blocked, :detail, :redaction_summary, :threats, :response);
"""

_SELECT_BASE = """
SELECT id, request_id, timestamp, redactions, blocked, detail, redaction_summary, threats, response
FROM   audit_events
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
        self._migrate()
        display = path if path == ":memory:" else os.path.abspath(path)
        logger.info("AuditDB ready: %s", display)

    def _migrate(self) -> None:
        """Apply additive schema migrations for existing databases."""
        for col, definition in [("response", "TEXT")]:
            try:
                self._conn.execute(f"ALTER TABLE audit_events ADD COLUMN {col} {definition};")
                self._conn.commit()
                logger.info("DB migration: added column '%s'", col)
            except sqlite3.OperationalError:
                pass  # Column already exists

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
            "response": (
                json.dumps(entry["response"])
                if entry.get("response")
                else None
            ),
        }
        with self._lock:
            self._conn.execute(_INSERT, row)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(
        self,
        limit: int | None = None,
        offset: int = 0,
        blocked: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return audit events with optional filtering and pagination.

        Args:
            limit:   Maximum number of rows to return (None = unlimited).
            offset:  Number of rows to skip from the start.
            blocked: If True/False, filter by blocked status; None returns all.
        """
        query = _SELECT_BASE
        params: list[Any] = []

        if blocked is not None:
            query += " WHERE blocked = ?"
            params.append(int(blocked))

        query += " ORDER BY id ASC"

        # SQLite requires LIMIT when OFFSET is used; -1 means unlimited.
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        elif offset:
            query += " LIMIT -1"

        if offset:
            query += " OFFSET ?"
            params.append(offset)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

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
            if row["response"]:
                event["response"] = json.loads(row["response"])
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
