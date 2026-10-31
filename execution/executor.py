from __future__ import annotations

import asyncio
from typing import Any, Protocol

from context.manager import ContextManager
from context.models import Task, TaskResult


class ProviderManager(Protocol):
	async def execute_task(self, task: Task, context_package: dict[str, Any]) -> Any:
		"""Execute a task using the selected provider and return its output."""


async def execute_tasks(
	tasks: list[Task | dict[str, Any]],
	context: ContextManager,
	provider_manager: ProviderManager,
	*,
	max_concurrency: int = 4,
	timeout_seconds: float = 120,
) -> list[TaskResult]:
	if max_concurrency < 1:
		raise ValueError("max_concurrency must be at least 1")
	if timeout_seconds <= 0:
		raise ValueError("timeout_seconds must be greater than 0")

	task_models = [task if isinstance(task, Task) else Task.model_validate(task) for task in tasks]
	if len({task.task_id for task in task_models}) != len(task_models):
		raise ValueError("task_id values must be unique")

	semaphore = asyncio.Semaphore(max_concurrency)
	remaining = {task.task_id: task for task in task_models}
	completed: set[str] = set()
	results: list[TaskResult] = []

	async def run_task(task: Task) -> TaskResult:
		package = context.build_context(task)
		timeout = task.timeout_seconds or timeout_seconds
		async with semaphore:
			try:
				response = await asyncio.wait_for(
					provider_manager.execute_task(task, package), timeout=timeout
				)
				if isinstance(response, dict) and "result" in response:
					return TaskResult(
						task_id=task.task_id,
						provider=str(response.get("provider", task.provider)),
						model=response.get("model", task.model),
						status="success",
						result=response["result"],
						proposed_decisions=response.get("proposed_decisions", []),
						project_updates=response.get("project_updates", {}),
					)
				return TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="success",
					result=response,
				)
			except TimeoutError:
				return TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="timeout",
					error=f"Task exceeded its {timeout:g}s timeout",
				)
			except Exception as error:  # noqa: BLE001 - isolate provider failures per task
				return TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="error",
					error=f"{type(error).__name__}: {error}",
				)

	while remaining:
		ready = [
			task for task in remaining.values()
			if set(task.depends_on).issubset(completed)
		]
		if not ready:
			for task in remaining.values():
				result = TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="error",
					error="Unresolved or cyclic task dependencies",
				)
				context.update_context(result)
				results.append(result)
			break

		batch_results = await asyncio.gather(*(run_task(task) for task in ready))
		for task, result in zip(ready, batch_results):
			context.update_context(result)
			completed.add(task.task_id)
			remaining.pop(task.task_id)
			results.append(result)

	return results
