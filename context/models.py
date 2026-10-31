from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    provider: str
    description: str
    model: str | None = None
    context_keys: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = Field(default=None, gt=0)


class Decision(BaseModel):
    text: str
    status: Literal["proposed", "confirmed"] = "proposed"
    source_task_id: str | None = None
    source_provider: str | None = None
    source_model: str | None = None


class TaskResult(BaseModel):
    task_id: str
    provider: str
    model: str | None = None
    status: Literal["success", "error", "timeout"]
    result: Any = None
    error: str | None = None
    proposed_decisions: list[str] = Field(default_factory=list)
    project_updates: dict[str, Any] = Field(default_factory=dict)


class SharedContext(BaseModel):
    project_id: str = "default"
    objective: str = ""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    project_context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    project_findings: list[dict[str, Any]] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)