from context.manager import ContextManager, build_context
from context.models import SharedContext


def test_context_uses_prior_project_details_without_unrelated_history():
    shared = SharedContext(
        project_id="project-a",
        objective="Design a student collaboration platform.",
        conversation_history=[
            {"role": "user", "content": "Earlier: Flutter with Firebase."},
            {"role": "assistant", "content": "A separate private conversation."},
        ],
        project_context={"framework": "Flutter", "backend": "Firebase"},
    )

    package = build_context(
        {"task_id": "security", "provider": "anthropic", "description": "Review security"},
        shared,
    )

    assert package["objective"] == "Design a student collaboration platform."
    assert package["project_context"] == {"framework": "Flutter", "backend": "Firebase"}
    assert "conversation_history" not in package
    assert "private conversation" not in str(package)


def test_memory_survives_reload_and_clear_is_project_scoped(tmp_path):
    path = str(tmp_path / "memory.sqlite3")
    first = ContextManager("project-a", path)
    first.add_conversation_message("user", "Use Flutter and Firebase.")
    first.set_project_context({"framework": "Flutter"})
    first.save_memory("project-b")
    first.context.project_id = "project-a"
    first.save_memory("project-a")

    reloaded = ContextManager("project-a", path)
    assert reloaded.context.project_context["framework"] == "Flutter"
    assert reloaded.context.objective == "Use Flutter and Firebase."

    reloaded.clear_memory("project-a")
    assert ContextManager("project-a", path).context.project_context == {}
    assert ContextManager("project-b", path).context.project_id == "project-b"


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

    assert "api_key" not in package["project_context"]
    assert "[REDACTED]" in package["objective"]
    persisted = ContextManager("project-a", str(tmp_path / "memory.sqlite3")).context
    assert "api_key" not in persisted.project_context
    assert "top-secret" not in str(persisted.model_dump())