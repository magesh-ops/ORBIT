# ruff: noqa: UP007
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol, Union

from config.models import SubTask, TaskResult
from context.manager import ContextManager
from context.memory import sanitize
from context.models import ContextPackage, coerce_subtask


class ProviderManager(Protocol):
	async def execute_task(self, task: SubTask, context_package: ContextPackage) -> Any:
		"""Execute a task using the selected provider and return its output."""


async def execute_tasks(
	tasks: list[Union[SubTask, Mapping[str, Any]]],
	context: ContextManager,
	provider_manager: ProviderManager,
	*,
	max_concurrency: int = 4,
	timeout_seconds: float = 120,
) -> list[TaskResult]:
	if (
		isinstance(max_concurrency, bool)
		or not isinstance(max_concurrency, int)
		or max_concurrency < 1
	):
		raise ValueError("max_concurrency must be at least 1")
	if timeout_seconds <= 0:
		raise ValueError("timeout_seconds must be greater than 0")

	legacy_timeouts: dict[str, float] = {}
	for task in tasks:
		if isinstance(task, Mapping) and task.get("timeout_seconds") is not None:
			task_timeout = float(task["timeout_seconds"])
			if task_timeout <= 0:
				raise ValueError("task timeout_seconds must be greater than 0")
			legacy_timeouts[str(task.get("task_id", ""))] = task_timeout
	task_models = [coerce_subtask(task) for task in tasks]
	if len({task.task_id for task in task_models}) != len(task_models):
		raise ValueError("task_id values must be unique")
	for task in task_models:
		if not task.task_id.strip():
			raise ValueError("SubTask.task_id must not be empty")
		if not task.provider or not task.provider.strip():
			raise ValueError(f"SubTask {task.task_id!r} must have a provider assigned before execution")
		if not task.model or not task.model.strip():
			raise ValueError(f"SubTask {task.task_id!r} must have a model assigned before execution")

	semaphore = asyncio.Semaphore(max_concurrency)
	tasks_by_id = {task.task_id: task for task in task_models}
	results_by_id: dict[str, TaskResult] = {}
	cycle_paths = _dependency_cycles(tasks_by_id)

	for task in task_models:
		missing_ids = sorted(set(task.dependencies) - tasks_by_id.keys())
		if missing_ids:
			results_by_id[task.task_id] = TaskResult(
				task_id=task.task_id,
				provider=task.provider,
				model=task.model,
				status="error",
				error=f"Missing dependency ID(s): {', '.join(missing_ids)}",
				execution_time=0.0,
			)
		elif task.task_id in cycle_paths:
			path = " -> ".join(cycle_paths[task.task_id])
			results_by_id[task.task_id] = TaskResult(
				task_id=task.task_id,
				provider=task.provider,
				model=task.model,
				status="error",
				error=f"Dependency cycle detected: {path}",
				execution_time=0.0,
			)

	for result in results_by_id.values():
		context.update_context(result)

	remaining = {
		task_id: task
		for task_id, task in tasks_by_id.items()
		if task_id not in results_by_id
	}

	async def run_task(
		task: SubTask,
	) -> tuple[TaskResult, list[str], dict[str, Any]]:
		package = context.build_context(task)
		timeout = legacy_timeouts.get(task.task_id) or timeout_seconds
		async with semaphore:
			started = perf_counter()
			try:
				response = await asyncio.wait_for(
					provider_manager.execute_task(task, package), timeout=timeout
				)
				elapsed = perf_counter() - started
				if isinstance(response, TaskResult):
					result = TaskResult(
						task_id=task.task_id,
						provider=task.provider,
						model=task.model,
						status=response.status,
						result=sanitize(response.result),
						error=(
							str(sanitize(response.error))[:5000]
							if response.error is not None
							else None
						),
						execution_time=elapsed,
					)
					return TaskResult.model_validate(sanitize(result.model_dump(mode="json"))), [], {}
				elif isinstance(response, Mapping):
					response_data = response
					result_value = response.get("result", response)
				else:
					response_data = {}
					result_value = response
				if result_value is not None and not isinstance(result_value, str):
					return TaskResult(
						task_id=task.task_id,
						provider=task.provider,
						model=task.model,
						status="error",
						error=(
							"Canonical TaskResult.result is text-only; structured provider "
							"output was rejected rather than stringified"
						),
						execution_time=elapsed,
					), [], {}
				result = TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="success",
					result=sanitize(result_value),
					execution_time=elapsed,
				)
				result = TaskResult.model_validate(sanitize(result.model_dump(mode="json")))
				return (
					result,
					list(response_data.get("proposed_decisions", [])),
					dict(response_data.get("project_updates", {})),
				)
			except asyncio.TimeoutError:  # noqa: UP041 - required before Python 3.11
				elapsed = perf_counter() - started
				return TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="timeout",
					error=f"Task exceeded its {timeout:g}s timeout",
					execution_time=elapsed,
				), [], {}
			except Exception as error:  # noqa: BLE001 - isolate provider failures per task
				safe_error = str(sanitize(f"{type(error).__name__}: {error}"))[:5000]
				elapsed = perf_counter() - started
				return TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="error",
					error=safe_error,
					execution_time=elapsed,
				), [], {}

	while remaining:
		blocked = [
			task
			for task in remaining.values()
			if any(
				dependency in results_by_id
				and results_by_id[dependency].status in {"error", "timeout", "skipped"}
				for dependency in task.dependencies
			)
		]
		for task in blocked:
			failed_dependencies = [
				dependency
				for dependency in task.dependencies
				if dependency in results_by_id
				and results_by_id[dependency].status in {"error", "timeout", "skipped"}
			]
			result = TaskResult(
				task_id=task.task_id,
				provider=task.provider,
				model=task.model,
				status="skipped",
				error=f"Skipped because prerequisite(s) did not succeed: {', '.join(failed_dependencies)}",
				execution_time=0.0,
			)
			results_by_id[task.task_id] = context.update_context(result)
			remaining.pop(task.task_id)

		ready = [
			task
			for task in remaining.values()
			if all(
				dependency in results_by_id
				and results_by_id[dependency].status == "success"
				for dependency in task.dependencies
			)
		]
		if not ready:
			for task in remaining.values():
				result = TaskResult(
					task_id=task.task_id,
					provider=task.provider,
					model=task.model,
					status="error",
					error="Dependency cycle detected; task could not be scheduled",
					execution_time=0.0,
				)
				results_by_id[task.task_id] = context.update_context(result)
			remaining.clear()
			break

		batch_results = await asyncio.gather(*(run_task(task) for task in ready))
		for task, (result, proposed_decisions, project_updates) in zip(ready, batch_results):
			results_by_id[task.task_id] = context.update_context(
				result,
				proposed_decisions=proposed_decisions,
				project_updates=project_updates,
			)
			remaining.pop(task.task_id)

	return [results_by_id[task.task_id] for task in task_models]


def _dependency_cycles(tasks_by_id: dict[str, SubTask]) -> dict[str, list[str]]:
	state: dict[str, int] = {}
	stack: list[str] = []
	cycles: dict[str, list[str]] = {}

	def visit(task_id: str) -> None:
		state[task_id] = 1
		stack.append(task_id)
		for dependency in tasks_by_id[task_id].dependencies:
			if dependency not in tasks_by_id:
				continue
			dependency_state = state.get(dependency, 0)
			if dependency_state == 0:
				visit(dependency)
			elif dependency_state == 1:
				cycle = stack[stack.index(dependency):] + [dependency]
				for cycle_task in cycle[:-1]:
					cycles[cycle_task] = cycle
		stack.pop()
		state[task_id] = 2

	for task_id in tasks_by_id:
		if state.get(task_id, 0) == 0:
			visit(task_id)
	return cycles
