"""Assign configured provider and model information to planned tasks."""

from __future__ import annotations

from config.models import TaskPlan
from config.settings import settings


class RouterError(Exception):
    """Raised when a task plan cannot be routed."""


class Router:
    """Route all subtasks to the configured local Ollama model."""

    def __init__(self, provider: str = "ollama", model: str | None = None):
        self.provider = provider.strip()
        self.model = (settings.ollama_model if model is None else model).strip()
        if not self.provider:
            raise RouterError("Provider name cannot be empty")
        if not self.model:
            raise RouterError("Model name cannot be empty")

    def route(self, plan: TaskPlan) -> TaskPlan:
        """Return a validated copy of the plan with provider attribution."""
        plan_data = plan.model_dump()
        for subtask in plan_data["subtasks"]:
            subtask["provider"] = self.provider
            subtask["model"] = self.model
        return TaskPlan.model_validate(plan_data)
