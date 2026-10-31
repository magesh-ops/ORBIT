import pytest

from config.models import SubTask, TaskPlan
from config.settings import settings
from orchestrator import Router, RouterError


def make_plan() -> TaskPlan:
    return TaskPlan(
        request="Build a feature",
        summary="Implement a feature",
        subtasks=[
            SubTask(
                description="Implement the backend",
                task_type="coding",
                dependencies=[],
            ),
            SubTask(
                description="Review the implementation",
                task_type="reasoning",
                dependencies=[],
            ),
        ],
    )


def test_routes_all_subtasks_to_configured_ollama_model():
    plan = make_plan()

    routed_plan = Router().route(plan)

    assert isinstance(routed_plan, TaskPlan)
    assert all(task.provider == "ollama" for task in routed_plan.subtasks)
    assert all(task.model == settings.ollama_model for task in routed_plan.subtasks)
    assert all(task.provider is None and task.model is None for task in plan.subtasks)


def test_router_accepts_explicit_provider_and_model():
    routed_plan = Router(provider="custom", model="model-v1").route(make_plan())

    assert all(task.provider == "custom" for task in routed_plan.subtasks)
    assert all(task.model == "model-v1" for task in routed_plan.subtasks)


@pytest.mark.parametrize(
    ("provider", "model", "message"),
    [
        (" ", "model-v1", "Provider name cannot be empty"),
        ("ollama", " ", "Model name cannot be empty"),
    ],
)
def test_router_rejects_empty_attribution(provider, model, message):
    with pytest.raises(RouterError, match=message):
        Router(provider=provider, model=model)
