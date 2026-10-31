# ruff: noqa: UP007, UP045
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union

from config.models import SubTask, TaskResult
from context.memory import SQLiteMemoryStore, sanitize, validate_project_id
from context.models import (
	ContextPackage,
	ConversationMessage,
	Decision,
	ProjectFinding,
	SharedContext,
	TaskDescriptor,
	coerce_subtask,
)


def _shared_context(value: Union[SharedContext, dict[str, Any]]) -> SharedContext:
	if isinstance(value, SharedContext):
		return value
	return SharedContext.model_validate(value)


def build_context(
	task: Union[SubTask, Mapping[str, Any]],
	shared_context: Union[SharedContext, dict[str, Any]],
) -> ContextPackage:
	task_model = coerce_subtask(task)
	shared = _shared_context(shared_context)
	objective = shared.objective
	if not objective:
		objective = next(
			(
				message.content
				for message in reversed(shared.conversation_history)
				if message.role == "user" and message.content
			),
			"",
		)

	project_values = shared.project_context
	requested = set(task_model.context_requirements)
	project_context = {key: project_values[key] for key in requested if key in project_values}

	dependencies = set(task_model.dependencies)
	previous_findings: list[TaskResult] = []
	for stored_result in shared.task_results:
		if isinstance(stored_result, dict):
			try:
				stored_result = TaskResult.model_validate(stored_result)
			except (TypeError, ValueError):
				continue
		if stored_result.task_id in dependencies and stored_result.status == "success":
			previous_findings.append(stored_result)
	relevant_decisions = [
		decision
		for decision in shared.decisions
		if ("decisions" in requested and decision.status == "confirmed")
		or decision.source_task_id in dependencies
	]

	package = ContextPackage(
		project_id=shared.project_id,
		objective=objective,
		project_context=project_context,
		task=TaskDescriptor(
			task_id=task_model.task_id,
			task_type=task_model.task_type,
			provider=task_model.provider,
			model=task_model.model,
			description=task_model.description,
		),
		constraints=shared.constraints if "constraints" in requested else [],
		decisions=relevant_decisions,
		previous_findings=previous_findings,
	)
	return ContextPackage.model_validate(sanitize(package))


class ContextManager:
	def __init__(
		self,
		project_id: str = "default",
		memory_path: Optional[str] = None,
		*,
		memory_store: Optional[SQLiteMemoryStore] = None,
		initial_context: Optional[Union[SharedContext, dict[str, Any]]] = None,
	) -> None:
		project_id = validate_project_id(project_id)
		self.memory = memory_store or SQLiteMemoryStore(memory_path)
		if initial_context is None:
			context = self.memory.load_memory(project_id)
		else:
			initial_data = (
				initial_context.model_dump(mode="json")
				if isinstance(initial_context, SharedContext)
				else dict(initial_context)
			)
			initial_project_id = initial_data.get("project_id", project_id)
			if validate_project_id(initial_project_id) != project_id:
				raise ValueError("initial context project_id does not match ContextManager project_id")
			initial_data["project_id"] = project_id
			context = _shared_context(sanitize(initial_data))
		self.project_id = project_id
		self.context = context

	def build_context(self, task: Union[SubTask, Mapping[str, Any]]) -> ContextPackage:
		return build_context(task, self.context)

	def add_conversation_message(self, role: str, content: str) -> None:
		safe_message = sanitize({"role": role, "content": content})
		message = ConversationMessage.model_validate(safe_message)
		self.context.conversation_history.append(message)
		if role == "user":
			self.context.objective = message.content
		self.save_memory(self.context.project_id)

	def set_project_context(self, values: dict[str, Any]) -> None:
		updates = sanitize(values)
		self.context.project_context.update(updates)
		self.save_memory(self.context.project_id)

	def update_context(
		self,
		task_result: Union[TaskResult, dict[str, Any]],
		*,
		proposed_decisions: Optional[list[str]] = None,
		project_updates: Optional[dict[str, Any]] = None,
	) -> TaskResult:
		if isinstance(task_result, TaskResult):
			payload = task_result.model_dump(mode="json")
		else:
			payload = dict(task_result)
			if proposed_decisions is None:
				proposed_decisions = payload.pop("proposed_decisions", None)
			if project_updates is None:
				project_updates = payload.pop("project_updates", None)
		result_fields = TaskResult.model_fields.keys()
		result = TaskResult.model_validate(sanitize({key: value for key, value in payload.items() if key in result_fields}))
		self.context.task_results = [
			existing
			for existing in self.context.task_results
			if (existing.task_id if isinstance(existing, TaskResult) else existing.get("task_id")) != result.task_id
		]
		self.context.task_results.append(result)

		if result.status == "success":
			for text in proposed_decisions or []:
				self.context.decisions.append(
					Decision(
						text=text,
						status="proposed",
						source_task_id=result.task_id,
						source_provider=result.provider,
						source_model=result.model,
					)
				)
			for key, value in (project_updates or {}).items():
				self.context.project_findings.append(
					ProjectFinding(
						key=key,
						value=sanitize(value),
						status="proposed",
						source_task_id=result.task_id,
						source_provider=result.provider,
						source_model=result.model,
					)
				)
		self.save_memory(self.context.project_id)
		return result

	def load_memory(self, project_id: str) -> SharedContext:
		project_id = validate_project_id(project_id)
		self.context = self.memory.load_memory(project_id)
		self.project_id = project_id
		return self.context

	def save_memory(self, project_id: str) -> None:
		project_id = validate_project_id(project_id)
		if project_id != self.project_id or self.context.project_id != self.project_id:
			raise ValueError("Cannot save context under a different project_id")
		self.memory.save_memory(project_id, self.context)

	def clear_memory(self, project_id: str) -> None:
		project_id = validate_project_id(project_id)
		self.memory.clear_memory(project_id)
		if self.context.project_id == project_id:
			self.context = SharedContext(project_id=project_id)
