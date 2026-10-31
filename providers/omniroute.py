"""OmniRoute — intelligent provider router.

Selects the best provider and model for a given query based on task type,
cost constraints, and provider availability.
"""

from __future__ import annotations

from typing import Any

from config.model_catalog import PROVIDER_MODELS, ModelConfig, default_model_for
from config.settings import Settings, get_settings
from providers.base import BaseProvider, Message, ProviderResponse


# Task-type hints that influence provider selection
TASK_HINTS: dict[str, list[str]] = {
    "code": ["anthropic", "openai"],
    "creative": ["anthropic", "openai"],
    "analysis": ["anthropic", "gemini"],
    "search": ["perplexity"],
    "chat": ["anthropic", "openai", "ollama"],
    "summarize": ["gemini", "anthropic"],
    "local": ["ollama"],
}


class OmniRoute:
    """Picks the best provider for a query and delegates to it."""

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        settings: Settings | None = None,
    ) -> None:
        self._providers = providers
        self._settings = settings or get_settings()

    def select_provider(
        self,
        task_type: str | None = None,
        prefer: str | None = None,
    ) -> BaseProvider:
        """Choose a provider based on task type and preference.

        Priority order:
        1. Explicit *prefer* (if available)
        2. Task-type recommendation
        3. Default provider from settings
        """
        if prefer and prefer in self._providers:
            return self._providers[prefer]

        if task_type and task_type in TASK_HINTS:
            for candidate in TASK_HINTS[task_type]:
                if candidate in self._providers:
                    return self._providers[candidate]

        default = self._settings.default_provider
        if default in self._providers:
            return self._providers[default]

        # Fallback: first available
        return next(iter(self._providers.values()))

    async def route(
        self,
        messages: list[Message],
        *,
        task_type: str | None = None,
        prefer: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Route a request to the best available provider."""
        provider = self.select_provider(task_type=task_type, prefer=prefer)
        return await provider.generate(messages, model=model, **kwargs)
