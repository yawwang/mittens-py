"""SQLite database — secondary index for run history.

The markdown ledger remains the source of truth. This DB provides
fast querying for the web UI: list runs, filter events, etc.
Can be rebuilt at any time by replaying ledger files.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from mittens.ledger import utc_now

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    mission TEXT NOT NULL,
    tier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_phases INTEGER,
    total_iterations INTEGER,
    cost_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    phase_id TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    produced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);
"""


class Database:
    """Async SQLite database for run history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init_db(self) -> None:
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        return self._db

    # -- Runs --

    async def create_run(
        self,
        workflow_id: str,
        mission: str,
        tier: str,
        run_id: str | None = None,
    ) -> str:
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        now = utc_now()
        await self.db.execute(
            "INSERT INTO runs (id, workflow_id, mission, tier, status, started_at) "
            "VALUES (?, ?, ?, ?, 'IN_PROGRESS', ?)",
            (run_id, workflow_id, mission, tier, now),
        )
        await self.db.commit()
        return run_id

    async def update_run(
        self,
        run_id: str,
        status: str | None = None,
        total_phases: int | None = None,
        total_iterations: int | None = None,
        cost_json: str | None = None,
    ) -> None:
        updates = []
        params: list[Any] = []
        if status:
            updates.append("status = ?")
            params.append(status)
            if status in ("COMPLETED", "FAILED"):
                updates.append("completed_at = ?")
                params.append(utc_now())
        if total_phases is not None:
            updates.append("total_phases = ?")
            params.append(total_phases)
        if total_iterations is not None:
            updates.append("total_iterations = ?")
            params.append(total_iterations)
        if cost_json is not None:
            updates.append("cost_json = ?")
            params.append(cost_json)

        if updates:
            params.append(run_id)
            await self.db.execute(
                f"UPDATE runs SET {', '.join(updates)} WHERE id = ?", params
            )
            await self.db.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # -- Events --

    async def add_event(
        self,
        run_id: str,
        event_type: str,
        timestamp: str,
        fields: dict[str, Any],
        phase_id: str | None = None,
    ) -> int:
        cursor = await self.db.execute(
            "INSERT INTO events (run_id, event_type, timestamp, fields_json, phase_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, event_type, timestamp, json.dumps(fields), phase_id),
        )
        await self.db.commit()
        return cursor.lastrowid or 0

    async def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if event_type:
            cursor = await self.db.execute(
                "SELECT * FROM events WHERE run_id = ? AND event_type = ? "
                "ORDER BY id LIMIT ?",
                (run_id, event_type, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY id LIMIT ?",
                (run_id, limit),
            )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["fields"] = json.loads(d.pop("fields_json"))
            result.append(d)
        return result

    # -- Artifacts --

    async def add_artifact(
        self, run_id: str, name: str, path: str
    ) -> int:
        now = utc_now()
        cursor = await self.db.execute(
            "INSERT INTO artifacts (run_id, name, path, produced_at) VALUES (?, ?, ?, ?)",
            (run_id, name, path, now),
        )
        await self.db.commit()
        return cursor.lastrowid or 0

    async def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY produced_at", (run_id,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
