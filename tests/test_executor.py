import asyncio

import pytest

from context.manager import ContextManager
from context.models import SharedContext
from execution.executor import execute_tasks


class FakeProviderManager:
	def __init__(self):
		self.packages = {}
		self.active = 0
		self.max_active = 0

	async def execute_task(self, task, context_package):
		self.packages[task.task_id] = context_package
		self.active += 1
		self.max_active = max(self.max_active, self.active)
		try:
			await asyncio.sleep(0.02)
			if task.task_id == "bad":
				raise RuntimeError("provider failed")
			if task.task_id == "source":
				return {
					"result": "Use Firebase Authentication.",
					"proposed_decisions": ["Use Firebase Authentication"],
					"project_updates": {"authentication": "Firebase"},
				}
			return f"completed {task.task_id}"
		finally:
			self.active -= 1


def make_context(tmp_path):
	return ContextManager(
		"student-platform",
		str(tmp_path / "memory.sqlite3"),
		initial_context=SharedContext(
			project_id="student-platform",
			objective="Design a student collaboration platform.",
			conversation_history=[
				{"role": "user", "content": "Earlier details: use Flutter and Firebase."},
				{"role": "assistant", "content": "A private unrelated exchange."},
			],
			project_context={"framework": "Flutter", "database": "Cloud Firestore", "billing": "not applicable"},
			constraints=["Support offline drafts"],
		),
	)


@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently_with_isolated_context(tmp_path):
	context = make_context(tmp_path)
	provider = FakeProviderManager()
	tasks = [
		{"task_id": "auth", "provider": "anthropic", "description": "Design authentication", "context_keys": ["framework"]},
		{"task_id": "scale", "provider": "gemini", "description": "Evaluate scalability", "context_keys": ["database"]},
	]

	results = await execute_tasks(tasks, context, provider)

	assert [result.status for result in results] == ["success", "success"]
	assert provider.max_active == 2
	assert provider.packages["auth"]["objective"] == "Design a student collaboration platform."
	assert provider.packages["auth"]["project_context"] == {"framework": "Flutter"}
	assert provider.packages["scale"]["project_context"] == {"database": "Cloud Firestore"}
	assert "conversation_history" not in provider.packages["auth"]
	assert "unrelated" not in str(provider.packages["auth"])


@pytest.mark.asyncio
async def test_failure_is_recorded_without_cancelling_other_tasks(tmp_path):
	context = make_context(tmp_path)
	results = await execute_tasks(
		[
			{"task_id": "bad", "provider": "anthropic", "description": "Fail"},
			{"task_id": "ok", "provider": "gemini", "description": "Succeed"},
		],
		context,
		FakeProviderManager(),
	)

	assert [result.status for result in results] == ["error", "success"]
	assert "provider failed" in results[0].error
	assert len(context.context.task_results) == 2


@pytest.mark.asyncio
async def test_dependency_receives_prior_result_and_syncs_proposals(tmp_path):
	context = make_context(tmp_path)
	provider = FakeProviderManager()
	results = await execute_tasks(
		[
			{"task_id": "source", "provider": "anthropic", "description": "Choose auth"},
			{"task_id": "followup", "provider": "gemini", "description": "Review auth", "depends_on": ["source"]},
		],
		context,
		provider,
	)

	assert all(result.status == "success" for result in results)
	assert provider.packages["followup"]["previous_findings"][0]["result"] == "Use Firebase Authentication."
	assert context.context.decisions[0].status == "proposed"
	assert context.context.decisions[0].source_provider == "anthropic"
	assert context.context.project_findings[0]["status"] == "proposed"


@pytest.mark.asyncio
async def test_task_timeout_is_normalized(tmp_path):
	class SlowProvider:
		async def execute_task(self, task, context_package):
			await asyncio.sleep(0.1)
			return "late"

	results = await execute_tasks(
		[{"task_id": "slow", "provider": "local", "description": "Slow", "timeout_seconds": 0.001}],
		make_context(tmp_path),
		SlowProvider(),
	)

	assert results[0].status == "timeout"
