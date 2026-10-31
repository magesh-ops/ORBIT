from __future__ import annotations

from typing import Any

from context.memory import SQLiteMemoryStore, sanitize
from context.models import Decision, SharedContext, Task, TaskResult


def _shared_context(value: SharedContext | dict[str, Any]) -> SharedContext:
	if isinstance(value, SharedContext):
		return value
	return SharedContext.model_validate(value)


def build_context(task: Task | dict[str, Any], shared_context: SharedContext | dict[str, Any]) -> dict[str, Any]:
	task_model = task if isinstance(task, Task) else Task.model_validate(task)
	shared = _shared_context(shared_context)
	objective = shared.objective
	if not objective:
		objective = next(
			(
				str(message.get("content", ""))
				for message in reversed(shared.conversation_history)
				if message.get("role") == "user" and message.get("content")
			),
			"",
		)

	if task_model.context_keys:
		project_context = {
			key: shared.project_context[key]
			for key in task_model.context_keys
			if key in shared.project_context
		}
	else:
		project_context = shared.project_context

	dependencies = set(task_model.depends_on)
	previous_findings = [
		result.model_dump(mode="json")
		for result in shared.task_results
		if result.task_id in dependencies and result.status == "success"
	]
	relevant_decisions = [
		decision.model_dump(mode="json")
		for decision in shared.decisions
		if decision.status == "confirmed" or decision.source_task_id in dependencies
	]

	package = {
		"project_id": shared.project_id,
		"objective": objective,
		"project_context": project_context,
		"task": {
			"task_id": task_model.task_id,
			"provider": task_model.provider,
			"model": task_model.model,
			"description": task_model.description,
		},
		"constraints": shared.constraints,
		"decisions": relevant_decisions,
		"previous_findings": previous_findings,
	}
	return sanitize(package)


class ContextManager:
	def __init__(
		self,
		project_id: str = "default",
		memory_path: str | None = None,
		*,
		memory_store: SQLiteMemoryStore | None = None,
		initial_context: SharedContext | dict[str, Any] | None = None,
	) -> None:
		self.memory = memory_store or SQLiteMemoryStore(memory_path)
		self.context = (
			_shared_context(initial_context)
			if initial_context is not None
			else self.memory.load_memory(project_id)
		)
		self.context.project_id = project_id

	def build_context(self, task: Task | dict[str, Any]) -> dict[str, Any]:
		return build_context(task, self.context)

	def add_conversation_message(self, role: str, content: str) -> None:
		self.context.conversation_history.append(
			sanitize({"role": role, "content": content})
		)
		if role == "user":
			self.context.objective = str(content)
		self.save_memory(self.context.project_id)

	def set_project_context(self, values: dict[str, Any]) -> None:
		self.context.project_context.update(sanitize(values))
		self.save_memory(self.context.project_id)

	def update_context(self, task_result: TaskResult | dict[str, Any]) -> TaskResult:
		result = task_result if isinstance(task_result, TaskResult) else TaskResult.model_validate(task_result)
		result = TaskResult.model_validate(sanitize(result.model_dump(mode="json")))
		self.context.task_results = [
			existing for existing in self.context.task_results if existing.task_id != result.task_id
		]
		self.context.task_results.append(result)

		for text in result.proposed_decisions:
			self.context.decisions.append(
				Decision(
					text=text,
					status="proposed",
					source_task_id=result.task_id,
					source_provider=result.provider,
					source_model=result.model,
				)
			)
		for key, value in result.project_updates.items():
			self.context.project_findings.append(
				{
					"key": key,
					"value": sanitize(value),
					"status": "proposed",
					"source_task_id": result.task_id,
					"source_provider": result.provider,
					"source_model": result.model,
				}
			)
		self.save_memory(self.context.project_id)
		return result

	def load_memory(self, project_id: str) -> SharedContext:
		self.context = self.memory.load_memory(project_id)
		return self.context

	def save_memory(self, project_id: str) -> None:
		self.context.project_id = project_id
		self.memory.save_memory(project_id, self.context)

	def clear_memory(self, project_id: str) -> None:
		self.memory.clear_memory(project_id)
		if self.context.project_id == project_id:
			self.context = SharedContext(project_id=project_id)
