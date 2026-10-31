"""
ORBIT Data Models

Pydantic v2 models for tasks, plans, results, and project context.
These models provide validation, serialization, and type safety across the orchestrator.
"""

from typing import Optional, Literal, List
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class SubTask(BaseModel):
    """
    Represents a single subtask within a task plan.

    The planner creates subtasks with descriptions and dependencies.
    The router populates provider and model fields.
    The executor runs the task and records results.
    """

    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this subtask"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Clear description of the work to be performed"
    )
    task_type: Literal["coding", "research", "reasoning", "general"] = Field(
        ...,
        description="Category of task determining provider selection strategy"
    )
    provider: Optional[str] = Field(
        default=None,
        description="Provider name assigned by router (e.g., 'openai', 'anthropic')"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier assigned by router (e.g., 'gpt-4', 'claude-sonnet-4')"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_id values that must complete before this task can start"
    )
    context_requirements: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="List of context categories or instructions needed for this task"
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure description is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Description cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("context_requirements")
    @classmethod
    def validate_context_requirements(cls, v: List[str]) -> List[str]:
        """Ensure context requirements are non-empty strings."""
        return [item.strip() for item in v if item.strip()]


class TaskPlan(BaseModel):
    """
    Represents a complete plan for handling a user request.

    Created by the planner after analyzing the user's request.
    Contains the original request, a summary, and a list of subtasks.
    """

    request: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Original user request text"
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Brief description of the overall plan"
    )
    subtasks: List[SubTask] = Field(
        default_factory=list,
        max_length=100,
        description="Ordered list of subtasks to complete the request. "
                    "This 100-subtask limit is an absolute safety maximum. "
                    "The planner should enforce the configurable limit from settings "
                    "(default 20 subtasks per request)."
    )

    @field_validator("request", "summary")
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Ensure text fields are not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return v.strip()


class TaskResult(BaseModel):
    """
    Represents the result of executing a single subtask.

    Created by the executor after a task completes, errors, or times out.
    Contains the outcome, any output or error, and execution metadata.
    """

    task_id: str = Field(
        ...,
        description="ID of the corresponding SubTask"
    )
    provider: str = Field(
        ...,
        min_length=1,
        description="Provider that executed this task"
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model that was used"
    )
    status: Literal["success", "error", "timeout", "skipped"] = Field(
        ...,
        description="Execution outcome status"
    )
    result: Optional[str] = Field(
        default=None,
        max_length=100000,
        description="Text output from successful execution"
    )
    error: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Error description if status is error or timeout"
    )
    execution_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Task execution duration in seconds"
    )

    @field_validator("provider", "model")
    @classmethod
    def validate_provider_model(cls, v: str) -> str:
        """Ensure provider and model are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class ProjectContext(BaseModel):
    """
    Represents contextual information about a project or codebase.

    Used by the context manager to provide relevant background information
    to the planner and subtasks. Can include project metadata, file references,
    architectural decisions, and other relevant context.
    """

    project_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Name of the project"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="High-level description of the project"
    )
    files: List[str] = Field(
        default_factory=list,
        max_length=500,
        description="List of relevant file paths or references"
    )
    decisions: List[str] = Field(
        default_factory=list,
        max_length=100,
        description="List of important architectural or technical decisions"
    )
    additional_context: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Any additional relevant context or notes"
    )

    @field_validator("files", "decisions")
    @classmethod
    def validate_list_items(cls, v: List[str]) -> List[str]:
        """Remove empty strings from lists."""
        return [item.strip() for item in v if item.strip()]
