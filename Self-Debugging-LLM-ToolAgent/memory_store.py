"""SQLite persistence for AgentShield execution history and memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    supplied_text TEXT NOT NULL DEFAULT '',
                    use_tool INTEGER NOT NULL,
                    selected_tools TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    tool_calls TEXT NOT NULL,
                    errors TEXT NOT NULL,
                    critical_error TEXT,
                    answer TEXT NOT NULL,
                    recovered INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_executions_created_at
                    ON executions(created_at DESC);

                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id INTEGER,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(execution_id) REFERENCES executions(id)
                        ON DELETE CASCADE
                );
                """
            )

    def save_execution(self, report) -> int:
        provider = getattr(report, "provider", {})
        values = (
            report.query,
            "",  # supplied_text is attached separately by save_execution_details
            int(report.use_tool),
            json.dumps(report.selected_tools),
            json.dumps(report.plan),
            json.dumps(report.tool_calls),
            json.dumps(report.errors),
            json.dumps(report.critical_error) if report.critical_error else None,
            report.answer,
            int(report.recovered),
            json.dumps(provider),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO executions (
                    query, supplied_text, use_tool, selected_tools, plan,
                    tool_calls, errors, critical_error, answer, recovered, provider
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return int(cursor.lastrowid)

    def save_execution_details(self, execution_id: int, supplied_text: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE executions SET supplied_text = ? WHERE id = ?",
                (supplied_text[:200_000], execution_id),
            )

    def save_memory(self, execution_id: int, key: str, value: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_items (execution_id, key, value)
                VALUES (?, ?, ?)
                """,
                (execution_id, key[:200], value[:200_000]),
            )
            return int(cursor.lastrowid)

    def list_executions(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(limit, 200))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, query, use_tool, selected_tools, answer,
                       recovered, provider, created_at
                FROM executions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._execution_summary(row) for row in rows]

    def get_execution(self, execution_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            return None
        return self._execution_detail(row)

    def list_memory(self, limit: int = 100) -> list[dict]:
        limit = max(1, min(limit, 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, execution_id, key, value, created_at
                FROM memory_items
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_items")
            connection.execute("DELETE FROM executions")

    @staticmethod
    def _json(value: str | None, fallback):
        if value is None:
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback

    def _execution_summary(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "query": row["query"],
            "use_tool": bool(row["use_tool"]),
            "selected_tools": self._json(row["selected_tools"], []),
            "answer": row["answer"],
            "recovered": bool(row["recovered"]),
            "provider": self._json(row["provider"], {}),
            "created_at": row["created_at"],
        }

    def _execution_detail(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "query": row["query"],
            "supplied_text": row["supplied_text"],
            "use_tool": bool(row["use_tool"]),
            "selected_tools": self._json(row["selected_tools"], []),
            "plan": self._json(row["plan"], []),
            "tool_calls": self._json(row["tool_calls"], []),
            "errors": self._json(row["errors"], []),
            "critical_error": self._json(row["critical_error"], None),
            "answer": row["answer"],
            "recovered": bool(row["recovered"]),
            "provider": self._json(row["provider"], {}),
            "created_at": row["created_at"],
        }
