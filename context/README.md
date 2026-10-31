# Context and Execution Contracts

`ContextManager` owns the complete local project context and persists it in SQLite at `~/.orbit/memory.sqlite3` by default. Pass `memory_path` to use another database. Its `load_memory`, `save_memory`, and `clear_memory` methods are scoped by `project_id`.

Use the Pydantic `Task`, `TaskResult`, `Decision`, and `SharedContext` schemas from `context.models`. `build_context(task, shared_context)` creates a provider-specific package containing the objective, selected project fields (`Task.context_keys`), constraints, applicable decisions, and successful results from explicitly listed dependencies. It does not forward conversation history wholesale.

Provider adapters are not implemented here. The executor receives a provider manager implementing:

```python
async def execute_task(task: Task, context_package: dict[str, Any]) -> Any:
    ...
```

Call `execute_tasks(tasks, context_manager, provider_manager, max_concurrency=4, timeout_seconds=120)`. Independent tasks run concurrently; tasks with `depends_on` run after their prerequisites and receive successful prerequisite findings. Each outcome is normalized as a `TaskResult` and synchronized to context. Model decisions and project updates are recorded as proposed, not confirmed.

Credential-named fields are removed and common inline key formats are redacted before distribution and persistence. Callers should still keep credentials out of user prompts and provider results.