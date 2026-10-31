# ruff: noqa: UP007, UP045
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.models import ProjectContext, SubTask, TaskResult

Task = SubTask


def coerce_subtask(task: Union[SubTask, Mapping[str, Any]]) -> SubTask:
    if isinstance(task, SubTask):
        return task
    values = dict(task)
    if "dependencies" not in values and "depends_on" in values:
        values["dependencies"] = values.pop("depends_on")
    if "context_requirements" not in values and "context_keys" in values:
        values["context_requirements"] = values.pop("context_keys")
    requirements = list(values.get("context_requirements", []))
    if values.pop("include_constraints", False):
        requirements.append("constraints")
    if values.pop("include_confirmed_decisions", False):
        requirements.append("decisions")
    values["context_requirements"] = requirements
    values.setdefault("task_type", "general")
    return SubTask.model_validate(values)


class TaskDescriptor(BaseModel):
    task_id: str
    task_type: Literal["coding", "research", "reasoning", "general"]
    provider: Optional[str] = None
    model: Optional[str] = None
    description: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class Decision(BaseModel):
    text: str
    status: Literal["proposed", "confirmed"] = "proposed"
    source_task_id: Optional[str] = None
    source_provider: Optional[str] = None
    source_model: Optional[str] = None


class ProjectFinding(BaseModel):
    key: str
    value: Any
    status: Literal["proposed", "confirmed"] = "proposed"
    source_task_id: str
    source_provider: str
    source_model: Optional[str] = None


class SharedContext(BaseModel):
    project_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    objective: str = ""
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    project_context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    project_findings: list[ProjectFinding] = Field(default_factory=list)
    task_results: list[Union[TaskResult, dict[str, Any]]] = Field(default_factory=list)

    @field_validator("project_context", mode="before")
    @classmethod
    def normalize_project_context(
        cls, value: Union[ProjectContext, Mapping[str, Any]]
    ) -> dict[str, Any]:
        if isinstance(value, ProjectContext):
            return value.model_dump(exclude_none=True)
        if isinstance(value, Mapping):
            return dict(value)
        return value


class ContextPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    objective: str
    project_context: dict[str, Any] = Field(default_factory=dict)
    task: TaskDescriptor
    constraints: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    previous_findings: list[TaskResult] = Field(default_factory=list)