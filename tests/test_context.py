import json
import sqlite3

import pytest

from context.manager import ContextManager, build_context
from context.memory import SQLiteMemoryStore, sanitize
from context.models import Decision, ProjectContext, SharedContext, SubTask, TaskResult


def test_context_defaults_to_minimal_project_data_and_excludes_history():
    shared = SharedContext(
        project_id="project-a",
        objective="Design a student collaboration platform.",
        conversation_history=[
            {"role": "user", "content": "Earlier: Flutter with Firebase."},
            {"role": "assistant", "content": "A separate private conversation."},
        ],
        project_context={"framework": "Flutter", "backend": "Firebase"},
        constraints=["Keep student data private"],
        decisions=[
            Decision(text="Use Firebase", status="confirmed"),
            Decision(text="Maybe use a hosted search service", status="proposed"),
        ],
    )

    package = build_context(
        {"task_id": "security", "provider": "anthropic", "description": "Review security"},
        shared,
    )

    assert package.objective == "Design a student collaboration platform."
    assert package.project_context == {}
    assert package.constraints == []
    assert package.decisions == []
    assert "conversation_history" not in str(package.model_dump())
    assert "private conversation" not in str(package.model_dump())


def test_context_includes_only_explicitly_requested_fields_and_confirmed_decisions():
    shared = SharedContext(
        project_id="project-a",
        objective="Build the platform.",
        project_context={"framework": "Flutter", "billing": "unused"},
        constraints=["Keep student data private"],
        decisions=[
            Decision(text="Use Firebase", status="confirmed"),
            Decision(text="Maybe use a hosted search service", status="proposed"),
        ],
    )

    package = build_context(
        SubTask(
            task_id="coding",
            task_type="coding",
            provider="anthropic",
            model="test-model",
            description="Implement authentication",
            context_requirements=["framework", "constraints", "decisions"],
        ),
        shared,
    )

    assert package.project_context == {"framework": "Flutter"}
    assert package.constraints == ["Keep student data private"]
    assert [decision.text for decision in package.decisions] == ["Use Firebase"]
    assert all(decision.status == "confirmed" for decision in package.decisions)


def test_canonical_subtask_and_task_result_cross_context_boundary(tmp_path):
    task = SubTask(
        task_id="research-01",
        task_type="research",
        description="Inspect project structure",
        provider="fake-provider",
        model="fake-model",
        context_requirements=["project_name"],
    )
    shared = SharedContext(
        project_id="project-a",
        project_context=ProjectContext(project_name="ORBIT", description="Local project"),
    )

    package = build_context(task, shared)
    manager = ContextManager("project-a", str(tmp_path / "memory.sqlite3"), initial_context=shared)
    result = TaskResult(
        task_id=task.task_id,
        provider=task.provider,
        model=task.model,
        status="success",
        result="Structure inspected",
        execution_time=0.25,
    )

    returned = manager.update_context(result)

    assert package.task.task_type == "research"
    assert package.project_context == {"project_name": "ORBIT"}
    assert isinstance(returned, TaskResult)
    assert manager.context.task_results[0].execution_time == 0.25


def test_project_memory_isolation_cross_project_write_and_clear(tmp_path):
    path = str(tmp_path / "memory.sqlite3")
    project_a = ContextManager(
        "project-a",
        path,
        initial_context=SharedContext(project_id="project-a", project_context={"marker": "A"}),
    )
    project_b = ContextManager(
        "project-b",
        path,
        initial_context=SharedContext(project_id="project-b", project_context={"marker": "B"}),
    )
    project_a.save_memory("project-a")
    project_b.save_memory("project-b")

    with pytest.raises(ValueError, match="different project_id"):
        project_a.save_memory("project-b")
    project_a.context.project_id = "project-b"
    with pytest.raises(ValueError, match="different project_id"):
        project_a.save_memory("project-a")
    project_a.context.project_id = "project-a"

    reopened_a = ContextManager("project-a", path)
    reopened_b = ContextManager("project-b", path)
    assert reopened_a.context.project_context == {"marker": "A"}
    assert reopened_b.context.project_context == {"marker": "B"}

    reopened_a.clear_memory("project-a")
    assert ContextManager("project-a", path).context.project_context == {}
    assert ContextManager("project-b", path).context.project_context == {"marker": "B"}


@pytest.mark.parametrize("project_id", ["", "../outside", "contains space", "line\nbreak"])
def test_invalid_project_ids_are_rejected(tmp_path, project_id):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError, match="project_id"):
        store.load_memory(project_id)


def test_schema_v0_migrates_and_database_can_be_reopened_and_renamed(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE project_memory (project_id TEXT PRIMARY KEY, context_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO project_memory VALUES (?, ?)",
            ("legacy", json.dumps({"project_id": "legacy", "objective": "retained"})),
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteMemoryStore(path)
    assert store.load_memory("legacy").objective == "retained"
    store.save_memory("new", SharedContext(project_id="new", objective="new data"))
    assert SQLiteMemoryStore(path).load_memory("new").objective == "new data"

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        connection.close()

    renamed_path = tmp_path / "renamed.sqlite3"
    path.rename(renamed_path)
    assert renamed_path.exists()


def test_sqlite_connection_closes_when_migration_raises(tmp_path, monkeypatch):
    store = SQLiteMemoryStore(tmp_path / "broken.sqlite3")
    connections = []
    real_connect = sqlite3.connect

    def capture_connection(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail_migration(connection):
        raise RuntimeError("synthetic migration failure")

    monkeypatch.setattr(sqlite3, "connect", capture_connection)
    monkeypatch.setattr(store, "_migrate", fail_migration)

    with pytest.raises(RuntimeError, match="synthetic migration failure"):
        store.load_memory("project-a")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")


def test_credentials_are_redacted_before_distribution_and_persistence(tmp_path):
    manager = ContextManager(
        "project-a",
        str(tmp_path / "memory.sqlite3"),
        initial_context=SharedContext(
            project_id="project-a",
            objective="api_key=sk-12345678901234567890 should not leak",
            project_context={"provider": "anthropic", "api_key": "top-secret"},
        ),
    )
    package = manager.build_context(
        {"task_id": "review", "provider": "gemini", "description": "Review"}
    )
    manager.save_memory("project-a")

    assert "api_key" not in package.project_context
    assert "[REDACTED]" in package.objective
    persisted = ContextManager("project-a", str(tmp_path / "memory.sqlite3")).context
    assert "api_key" not in persisted.project_context
    assert "top-secret" not in str(persisted.model_dump())


def test_nested_and_inline_credential_formats_are_sanitized():
    sanitized = sanitize(
        {
            "nested": [
                {
                    "api_key": "SYNTHETIC-NESTED-KEY",
                    "api_keys": ["SYNTHETIC-PLURAL-API-KEY"],
                    "refresh_token": "SYNTHETIC-REFRESH-TOKEN",
                    "session_token": "SYNTHETIC-SESSION-TOKEN",
                    "credentials": {"provider": "SYNTHETIC-CREDENTIAL"},
                    "tokens": ["SYNTHETIC-PLURAL-TOKEN"],
                    "authorization": "Bearer SYNTHETIC-AUTH-TOKEN",
                    "message": (
                        "token=SYNTHETIC-TOKEN api_key='SYNTHETIC-KEY' "
                        "Bearer SYNTHETIC-BEARER sk-12345678901234567890"
                    ),
                }
            ]
        }
    )
    rendered = str(sanitized)

    for secret in (
        "SYNTHETIC-NESTED-KEY",
        "SYNTHETIC-PLURAL-API-KEY",
        "SYNTHETIC-REFRESH-TOKEN",
        "SYNTHETIC-SESSION-TOKEN",
        "SYNTHETIC-CREDENTIAL",
        "SYNTHETIC-PLURAL-TOKEN",
        "SYNTHETIC-AUTH-TOKEN",
        "SYNTHETIC-TOKEN",
        "SYNTHETIC-KEY",
        "SYNTHETIC-BEARER",
        "sk-12345678901234567890",
    ):
        assert secret not in rendered


def test_explicit_empty_context_updates_override_payload_metadata(tmp_path):
    manager = ContextManager("project-a", str(tmp_path / "memory.sqlite3"))
    manager.update_context(
        {
            "task_id": "empty-updates",
            "provider": "fake-provider",
            "model": "fake-model",
            "status": "success",
            "result": "done",
            "proposed_decisions": ["do not record"],
            "project_updates": {"do_not_record": "value"},
        },
        proposed_decisions=[],
        project_updates={},
    )

    assert manager.context.decisions == []
    assert manager.context.project_findings == []


def test_user_objective_is_sanitized_in_memory_and_before_persistence(tmp_path):
    manager = ContextManager("project-a", str(tmp_path / "memory.sqlite3"))
    manager.add_conversation_message("user", "Use token=SYNTHETIC-USER-TOKEN")

    assert "SYNTHETIC-USER-TOKEN" not in manager.context.objective
    reloaded = ContextManager("project-a", str(tmp_path / "memory.sqlite3"))
    assert "SYNTHETIC-USER-TOKEN" not in reloaded.context.objective