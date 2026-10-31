import asyncio

import pytest

from context.manager import ContextManager
from context.models import ContextPackage, SharedContext, SubTask, TaskResult
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
		{"task_id": "auth", "provider": "anthropic", "model": "auth-model", "description": "Design authentication", "context_keys": ["framework"]},
		{"task_id": "scale", "provider": "gemini", "model": "scale-model", "description": "Evaluate scalability", "context_keys": ["database"]},
	]

	results = await execute_tasks(tasks, context, provider)

	assert [result.status for result in results] == ["success", "success"]
	assert all(isinstance(result, TaskResult) for result in results)
	assert all(result.execution_time is not None and result.execution_time >= 0 for result in results)
	assert provider.max_active == 2
	assert isinstance(provider.packages["auth"], ContextPackage)
	assert provider.packages["auth"].objective == "Design a student collaboration platform."
	assert provider.packages["auth"].project_context == {"framework": "Flutter"}
	assert provider.packages["scale"].project_context == {"database": "Cloud Firestore"}
	assert "conversation_history" not in str(provider.packages["auth"].model_dump())
	assert "unrelated" not in str(provider.packages["auth"].model_dump())


@pytest.mark.asyncio
async def test_failure_is_recorded_without_cancelling_other_tasks(tmp_path):
	context = make_context(tmp_path)
	results = await execute_tasks(
		[
			{"task_id": "bad", "provider": "anthropic", "model": "bad-model", "description": "Fail"},
			{"task_id": "ok", "provider": "gemini", "model": "ok-model", "description": "Succeed"},
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
			{"task_id": "source", "provider": "anthropic", "model": "source-model", "description": "Choose auth"},
			{"task_id": "followup", "provider": "gemini", "model": "followup-model", "description": "Review auth", "depends_on": ["source"]},
		],
		context,
		provider,
	)

	assert all(result.status == "success" for result in results)
	assert provider.packages["followup"].previous_findings[0].result == "Use Firebase Authentication."
	assert provider.packages["followup"].decisions[0].status == "proposed"
	assert context.context.decisions[0].status == "proposed"
	assert context.context.decisions[0].source_provider == "anthropic"
	assert context.context.project_findings[0].status == "proposed"


@pytest.mark.asyncio
async def test_task_timeout_is_normalized(tmp_path):
	class SlowProvider:
		async def execute_task(self, task, context_package):
			await asyncio.sleep(0.1)
			return "late"

	results = await execute_tasks(
		[{"task_id": "slow", "provider": "local", "model": "slow-model", "description": "Slow", "timeout_seconds": 0.001}],
		make_context(tmp_path),
		SlowProvider(),
	)

	assert results[0].status == "timeout"


@pytest.mark.asyncio
async def test_failed_prerequisite_skips_dependents_but_not_independent_tasks(tmp_path):
	context = make_context(tmp_path)
	provider = FakeProviderManager()
	tasks = [
		{"task_id": "bad", "provider": "anthropic", "model": "bad-model", "description": "Fail"},
		{"task_id": "blocked", "provider": "gemini", "model": "blocked-model", "description": "Requires bad", "depends_on": ["bad"]},
		{"task_id": "independent", "provider": "local", "model": "independent-model", "description": "Still run"},
	]

	results = await execute_tasks(tasks, context, provider)
	by_id = {result.task_id: result for result in results}

	assert by_id["bad"].status == "error"
	assert by_id["blocked"].status == "skipped"
	assert "bad" in by_id["blocked"].error
	assert by_id["independent"].status == "success"
	assert "blocked" not in provider.packages


@pytest.mark.asyncio
async def test_missing_dependency_and_cycle_have_distinct_errors_without_provider_calls(tmp_path):
	context = make_context(tmp_path)
	provider = FakeProviderManager()
	tasks = [
		{"task_id": "missing", "provider": "fake", "model": "missing-model", "description": "Missing", "depends_on": ["absent"]},
		{"task_id": "cycle-a", "provider": "fake", "model": "cycle-model", "description": "Cycle A", "depends_on": ["cycle-b"]},
		{"task_id": "cycle-b", "provider": "fake", "model": "cycle-model", "description": "Cycle B", "depends_on": ["cycle-a"]},
	]

	results = await execute_tasks(tasks, context, provider)
	by_id = {result.task_id: result for result in results}

	assert by_id["missing"].status == "error"
	assert "Missing dependency ID" in by_id["missing"].error
	assert "Dependency cycle detected" in by_id["cycle-a"].error
	assert "Dependency cycle detected" in by_id["cycle-b"].error
	assert provider.packages == {}


@pytest.mark.asyncio
async def test_chain_runs_after_prerequisites_and_results_keep_input_order(tmp_path):
	class ChainProvider:
		def __init__(self):
			self.packages = {}

		async def execute_task(self, task, context_package):
			self.packages[task.task_id] = context_package
			return {"result": f"finding from {task.task_id}"}

	context = make_context(tmp_path)
	provider = ChainProvider()
	tasks = [
		SubTask(task_id="leaf", task_type="coding", provider="fake", model="leaf-model", description="Leaf", dependencies=["middle"]),
		SubTask(task_id="free-a", task_type="research", provider="fake", model="free-model", description="Independent A"),
		SubTask(task_id="root", task_type="research", provider="fake", model="root-model", description="Root"),
		SubTask(task_id="middle", task_type="reasoning", provider="fake", model="middle-model", description="Middle", dependencies=["root"]),
		SubTask(task_id="free-b", task_type="general", provider="fake", model="free-model", description="Independent B"),
	]

	results = await execute_tasks(tasks, context, provider)

	assert [result.task_id for result in results] == [task.task_id for task in tasks]
	assert provider.packages["middle"].previous_findings[0].task_id == "root"
	assert provider.packages["leaf"].previous_findings[0].task_id == "middle"


@pytest.mark.asyncio
async def test_exception_and_provider_output_credentials_are_scrubbed_everywhere(tmp_path):
	class CredentialProvider:
		def __init__(self):
			self.packages = {}

		async def execute_task(self, task, context_package):
			self.packages[task.task_id] = context_package
			if task.task_id == "exception":
				raise RuntimeError("Authorization: Bearer SYNTHETIC-EXCEPTION-TOKEN")
			if task.task_id == "source":
				return {"result": "token=SYNTHETIC-RESULT-TOKEN"}
			return {"result": "dependent done"}

	context = make_context(tmp_path)
	provider = CredentialProvider()
	results = await execute_tasks(
		[
			{"task_id": "exception", "provider": "fake", "model": "exception-model", "description": "Exception"},
			{"task_id": "exception-dependent", "provider": "fake", "model": "dependent-model", "description": "Must not run", "depends_on": ["exception"]},
			{"task_id": "source", "provider": "fake", "model": "source-model", "description": "Source"},
			{"task_id": "dependent", "provider": "fake", "model": "dependent-model", "description": "Dependent", "depends_on": ["source"]},
		],
		context,
		provider,
	)
	by_id = {result.task_id: result for result in results}

	assert "SYNTHETIC-EXCEPTION-TOKEN" not in (by_id["exception"].error or "")
	assert by_id["exception-dependent"].status == "skipped"
	assert "exception-dependent" not in provider.packages
	assert "SYNTHETIC-EXCEPTION-TOKEN" not in str(context.context.model_dump())
	assert "SYNTHETIC-RESULT-TOKEN" not in str(by_id["source"].model_dump())
	assert "SYNTHETIC-RESULT-TOKEN" not in str(context.context.model_dump())
	assert "SYNTHETIC-RESULT-TOKEN" not in str(provider.packages["dependent"].model_dump())
	assert "[REDACTED]" in str(provider.packages["dependent"].previous_findings[0].result)


@pytest.mark.asyncio
async def test_structured_provider_result_is_not_stringified_for_canonical_schema(tmp_path):
	class StructuredProvider:
		async def execute_task(self, task, context_package):
			return {"ok": True, "count": 2}

	result = (await execute_tasks(
		[SubTask(task_id="structured", task_type="research", description="Return data", provider="fake", model="fake-model")],
		make_context(tmp_path),
		StructuredProvider(),
	))[0]

	assert result.status == "error"
	assert result.result is None
	assert "text-only" in result.error


@pytest.mark.asyncio
async def test_oversized_provider_exception_is_truncated_to_canonical_limit(tmp_path):
	class LongErrorProvider:
		async def execute_task(self, task, context_package):
			raise RuntimeError("x" * 6000)

	result = (await execute_tasks(
		[SubTask(task_id="long-error", task_type="general", description="Fail", provider="fake", model="fake-model")],
		make_context(tmp_path),
		LongErrorProvider(),
	))[0]

	assert result.status == "error"
	assert len(result.error) <= 5000


@pytest.mark.asyncio
@pytest.mark.parametrize(
	("status", "error"),
	[("error", "provider rejected task"), ("timeout", "provider timed out")],
)
async def test_canonical_provider_failures_keep_status_error_and_assigned_attribution(
	tmp_path, status, error
):
	class CanonicalResultProvider:
		def __init__(self):
			self.called = []

		async def execute_task(self, task, context_package):
			self.called.append(task.task_id)
			return TaskResult(
				task_id="provider-supplied-id",
				provider="untrusted-provider",
				model="untrusted-model",
				status=status,
				error=error,
				execution_time=999,
			)

	provider = CanonicalResultProvider()
	results = await execute_tasks(
		[
			SubTask(
				task_id="assigned-task",
				task_type="general",
				description="Must fail",
				provider="assigned-provider",
				model="assigned-model",
			),
			SubTask(
				task_id="dependent-task",
				task_type="general",
				description="Must be skipped",
				provider="assigned-provider",
				model="assigned-model",
				dependencies=["assigned-task"],
			),
		],
		make_context(tmp_path),
		provider,
	)

	assert results[0].status == status
	assert results[0].error == error
	assert results[0].task_id == "assigned-task"
	assert results[0].provider == "assigned-provider"
	assert results[0].model == "assigned-model"
	assert 0 <= results[0].execution_time < 999
	assert results[1].status == "skipped"
	assert provider.called == ["assigned-task"]


@pytest.mark.asyncio
async def test_canonical_success_uses_assigned_provider_and_model(tmp_path):
	class MismatchedAttributionProvider:
		async def execute_task(self, task, context_package):
			return TaskResult(
				task_id="different-id",
				provider="different-provider",
				model="different-model",
				status="success",
				result="completed",
			)

	result = (await execute_tasks(
		[
			SubTask(
				task_id="assigned-task",
				task_type="general",
				description="Run task",
				provider="assigned-provider",
				model="assigned-model",
			)
		],
		make_context(tmp_path),
		MismatchedAttributionProvider(),
	))[0]

	assert result.task_id == "assigned-task"
	assert result.provider == "assigned-provider"
	assert result.model == "assigned-model"
	assert result.status == "success"
	assert result.result == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
async def test_invalid_max_concurrency_is_rejected(tmp_path, value):
	class UnusedProvider:
		async def execute_task(self, task, context_package):
			raise AssertionError("provider must not be called")

	with pytest.raises(ValueError, match="max_concurrency"):
		await execute_tasks([], make_context(tmp_path), UnusedProvider(), max_concurrency=value)
