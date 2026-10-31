# ruff: noqa: UP007, UP045
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel

from context.models import SharedContext

_CURRENT_SCHEMA_VERSION = 1
_PROJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SECRET_KEY_SUFFIXES = (
    "apikey",
    "apikeys",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "tokens",
    "token",
    "clientsecret",
    "secretkey",
    "privatekey",
    "signingkey",
    "password",
    "credential",
    "credentials",
    "secret",
    "secrets",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[\s_-]?key|access[\s_-]?token|refresh[\s_-]?token|auth[\s_-]?token|token|client[\s_-]?secret|password|secret|credential)\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_COMMON_API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,})\b"
)


def validate_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError("project_id must be 1-128 ASCII letters, digits, '.', '_', ':' or '-' and start with a letter or digit")
    return project_id


def _is_credential_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return (
        normalized in {"token", "tokens", "authorization", "bearertoken"}
        or any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)
    )


def sanitize(value: Any) -> Any:
    """Remove credential fields and redact common secret forms recursively."""
    if isinstance(value, BaseModel):
        return sanitize(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if not _is_credential_key(key)
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        value = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
        return _COMMON_API_KEY.sub("[REDACTED]", value)
    return value


class SQLiteMemoryStore:
    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path).expanduser() if path is not None else Path.home() / ".orbit" / "memory.sqlite3"

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is newer than supported version {_CURRENT_SCHEMA_VERSION}"
            )
        if version < 1:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS project_memory ("
                "project_id TEXT PRIMARY KEY, context_json TEXT NOT NULL)"
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            self._migrate(connection)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_memory(self, project_id: str, context: Union[SharedContext, dict[str, Any]]) -> None:
        project_id = validate_project_id(project_id)
        payload = context.model_dump(mode="json") if isinstance(context, SharedContext) else dict(context)
        context_project_id = payload.get("project_id")
        if context_project_id is not None and validate_project_id(context_project_id) != project_id:
            raise ValueError("Cannot save context under a project_id that differs from its own")
        payload["project_id"] = project_id
        encoded = json.dumps(sanitize(payload), ensure_ascii=True)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO project_memory(project_id, context_json) VALUES (?, ?) "
                "ON CONFLICT(project_id) DO UPDATE SET context_json = excluded.context_json",
                (project_id, encoded),
            )

    def load_memory(self, project_id: str) -> SharedContext:
        project_id = validate_project_id(project_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT context_json FROM project_memory WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            return SharedContext(project_id=project_id)
        payload = sanitize(json.loads(row[0]))
        payload["project_id"] = project_id
        return SharedContext.model_validate(payload)

    def clear_memory(self, project_id: str) -> None:
        project_id = validate_project_id(project_id)
        with self._connection() as connection:
            connection.execute("DELETE FROM project_memory WHERE project_id = ?", (project_id,))