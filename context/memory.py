from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from context.models import SharedContext

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "token",
    "password",
    "secret",
    "credential",
    "credentials",
}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|secret|credential)\b\s*[:=]\s*(['\"]?)[^\s,;'\"]+"
)
_COMMON_API_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,})\b")


def sanitize(value: Any) -> Any:
    """Remove credential fields and redact common secret forms recursively."""
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _SECRET_KEYS
            and not str(key).lower().endswith("_api_key")
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = _SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", value)
        return _COMMON_API_KEY.sub("[REDACTED]", value)
    return value


class SQLiteMemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else Path.home() / ".orbit" / "memory.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS project_memory ("
            "project_id TEXT PRIMARY KEY, context_json TEXT NOT NULL)"
        )
        return connection

    def save_memory(self, project_id: str, context: SharedContext | dict[str, Any]) -> None:
        payload = context.model_dump(mode="json") if isinstance(context, SharedContext) else context
        encoded = json.dumps(sanitize(payload), ensure_ascii=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO project_memory(project_id, context_json) VALUES (?, ?) "
                "ON CONFLICT(project_id) DO UPDATE SET context_json = excluded.context_json",
                (project_id, encoded),
            )

    def load_memory(self, project_id: str) -> SharedContext:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT context_json FROM project_memory WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            return SharedContext(project_id=project_id)
        payload = json.loads(row[0])
        payload["project_id"] = project_id
        return SharedContext.model_validate(payload)

    def clear_memory(self, project_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM project_memory WHERE project_id = ?", (project_id,))