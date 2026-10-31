# Context and Execution Integration Contract

## Canonical Schema Snapshot

The fetched `origin/feature/orchestrator` Stage 1 commit `5621b55` defines canonical models in `config/models.py`. `context.models` imports and re-exports `SubTask`, `TaskResult`, and `ProjectContext`; it does not define duplicate task/result classes. `Task` is a compatibility alias to `SubTask`.

| Model | Canonical fields/role | Context disposition |
| --- | --- | --- |
| `SubTask` | `task_id` (generated UUID by default), required `description` and `task_type`, optional `provider` and `model`, `dependencies`, `context_requirements` | Canonical execution input; context accepts it directly |
| `TaskPlan` | Original request, summary, ordered subtasks | Orchestrator plan; context/execution consume its subtasks, not the plan |
| `TaskResult` | Task/provider/model IDs, status, optional text result/error/execution time | Canonical execution and persistence result |
| `ProjectContext` | Project name/description, files, string decisions, additional context | Canonical project metadata converted to a context-owned mapping |
| `TaskDescriptor` | Reduced task identity, including task type, embedded in a package | Context-specific; not a replacement for `SubTask` |
| `ConversationMessage` | Persisted role/content pair | Context-specific |
| `Decision` | Proposed/confirmed decision and source attribution | Context-specific provenance; distinct from `ProjectContext.decisions: list[str]` |
| `ProjectFinding` | Project update and source attribution/status | Context synchronization detail |
| `SharedContext` | Local conversation, project data, decisions, findings, and canonical task results | Context-specific aggregate |
| `ContextPackage` | Typed, filtered per-task context passed to a provider manager | Context-specific provider package |

Canonical validation details matter: `SubTask.provider` and `SubTask.model` default to `None`; `TaskResult.provider` and `TaskResult.model` are required non-empty strings. `TaskResult.status` allows `success`, `error`, `timeout`, and `skipped`. `TaskResult.execution_time` is optional and non-negative. Crucially, `TaskResult.result` is `Optional[str]`, not arbitrary structured data. The executor rejects non-text results instead of stringifying them; structured-output support requires a canonical schema change and is a blocker for full integration.

## Public APIs

```python
build_context(
    task: Union[SubTask, Mapping[str, Any]],
    shared_context: Union[SharedContext, dict[str, Any]],
) -> ContextPackage

ContextManager.__init__(
    self, project_id: str = "default", memory_path: Optional[str] = None, *,
    memory_store: Optional[SQLiteMemoryStore] = None,
    initial_context: Optional[Union[SharedContext, dict[str, Any]]] = None,
) -> None

ContextManager.build_context(self, task: Union[SubTask, Mapping[str, Any]]) -> ContextPackage
ContextManager.add_conversation_message(self, role: str, content: str) -> None
ContextManager.set_project_context(self, values: dict[str, Any]) -> None
ContextManager.update_context(
    self, task_result: Union[TaskResult, dict[str, Any]], *,
    proposed_decisions: Optional[list[str]] = None,
    project_updates: Optional[dict[str, Any]] = None,
) -> TaskResult
ContextManager.load_memory(self, project_id: str) -> SharedContext
ContextManager.save_memory(self, project_id: str) -> None
ContextManager.clear_memory(self, project_id: str) -> None

SQLiteMemoryStore.__init__(self, path: Optional[Union[str, Path]] = None) -> None
SQLiteMemoryStore.save_memory(
    self, project_id: str, context: Union[SharedContext, dict[str, Any]]
) -> None
SQLiteMemoryStore.load_memory(self, project_id: str) -> SharedContext
SQLiteMemoryStore.clear_memory(self, project_id: str) -> None

ProviderManager.execute_task(
    self, task: SubTask, context_package: ContextPackage
) -> Any  # async protocol method

execute_tasks(
    tasks: list[Union[SubTask, Mapping[str, Any]]],
    context: ContextManager,
    provider_manager: ProviderManager,
    *, max_concurrency: int = 4, timeout_seconds: float = 120,
) -> list[TaskResult]
```

`sanitize(value: Any) -> Any` and `validate_project_id(project_id: str) -> str` are public helpers. Memory load/save/clear are methods, not module-level functions. Provider/model assignment is required before execution so every canonical result can carry truthful provider/model attribution. SubTask has no per-task timeout; use the executor's `timeout_seconds` argument. Legacy mapping inputs may still provide `timeout_seconds`.

## Conversion and Filtering

- `coerce_subtask` accepts canonical `SubTask` directly. For legacy mappings only, it maps `depends_on` to `dependencies`, `context_keys` to `context_requirements`, old include flags to the `constraints`/`decisions` requirements, and defaults missing `task_type` to `general`.
- Context requirement strings are matched exactly against project metadata keys. The special values `constraints` and `decisions` opt in those sections. Unsupported prose requirements do not cause unrelated fields to be sent.
- Canonical `ProjectContext` is flattened with `model_dump(exclude_none=True)` at the context boundary; legacy dictionaries are preserved. Conversation history remains local. The current objective is included.
- Previous findings are included only for declared dependency IDs with successful canonical results. Failed, timed-out, and skipped results are not exposed as successful findings. Decision provenance remains local and proposed decisions remain proposed.
- Provider results and exception messages are sanitized before return or persistence. Project IDs are validated; cross-project saves are rejected. SQLite uses `PRAGMA user_version`, migrates an unversioned database to v1, and closes connections after each operation.
- Missing dependency IDs and cycles produce distinct `error` results. Failed prerequisites cause dependent tasks to be `skipped` without provider calls. Independent tasks run concurrently up to `max_concurrency`; results return in input order. `execution_time` is measured for provider calls and set to zero for tasks not run.

## Integration Requirements and Blockers

The local `main` branch does not contain Stage 1 config files. `main` and `origin/feature/orchestrator` have diverged by one commit each from `ca4dd7f`; do not merge or rebase with this dirty worktree. The safest next step is a deliberate integration branch/merge after preserving the eight existing dirty files, then resolve only owned context/execution files against the canonical commit. The tests for this preparation were run by loading the fetched canonical `config.models` source in memory; no shared config files were copied or changed.

The provider manager remains an owned protocol, not a verified Dhinesh interface. Agree whether adapters accept `SubTask` and `ContextPackage` directly or require serialization. Also request that Magesh widen canonical `TaskResult.result` to a JSON-compatible structured type before structured provider results can be supported without loss. No canonical schemas were modified here.