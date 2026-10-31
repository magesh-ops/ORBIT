"""
ORBIT Application Settings

Central configuration for the ORBIT orchestrator.
Uses environment variables for non-secret configuration.

API keys and secrets are NEVER stored here - they are managed separately
through the security/credentials module using Python keyring.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application configuration for ORBIT.

    All configuration is loaded from environment variables with sensible defaults.
    Secrets such as API keys are NOT stored here - use the credentials module instead.
    """

    model_config = SettingsConfigDict(
        env_prefix="ORBIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application metadata
    app_name: str = Field(
        default="ORBIT",
        description="Application name"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance"
    )
    ollama_model: str = Field(
        default="gemma4:12b",
        description="Default local model name for orchestration"
    )

    # Execution limits
    max_concurrent_tasks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of subtasks that can run concurrently"
    )
    default_task_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Default timeout for subtask execution in seconds"
    )
    max_subtasks_per_request: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of subtasks the planner should create per user request. "
                    "This is enforced by the planner. The TaskPlan model has an absolute "
                    "safety limit of 100 subtasks."
    )

    # Data directory
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".orbit" / "data",
        description="Application data directory for storing context and memory"
    )

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Ensure Ollama URL is well-formed."""
        v = v.strip()
        if not v:
            raise ValueError("Ollama base URL cannot be empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Ollama base URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, v: str) -> str:
        """Ensure model name is not empty."""
        v = v.strip()
        if not v:
            raise ValueError("Ollama model name cannot be empty")
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def ensure_data_dir_exists(cls, v: Path) -> Path:
        """Create data directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# Global settings instance
settings = Settings()
