"""Model configuration and provider-model mappings.

This module contains provider model metadata including context windows,
pricing, and capabilities. It is separate from config/models.py which
contains Magesh's canonical task orchestration schemas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Describes a single model offered by a provider."""

    id: str
    name: str
    provider: str
    context_window: int = 128_000
    max_output_tokens: int = 4_096
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0


# ── Provider → model catalogue ──────────────────────────────────────────

PROVIDER_MODELS: dict[str, list[ModelConfig]] = {
    "anthropic": [
        ModelConfig(
            id="claude-sonnet-4-20250514",
            name="Claude Sonnet 4",
            provider="anthropic",
            context_window=200_000,
            max_output_tokens=16_384,
            supports_vision=True,
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
        ),
        ModelConfig(
            id="claude-haiku-3-5-20241022",
            name="Claude 3.5 Haiku",
            provider="anthropic",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_vision=True,
            cost_per_million_input=0.80,
            cost_per_million_output=4.0,
        ),
    ],
    "openai": [
        ModelConfig(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_vision=True,
            cost_per_million_input=2.50,
            cost_per_million_output=10.0,
        ),
        ModelConfig(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider="openai",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_vision=True,
            cost_per_million_input=0.15,
            cost_per_million_output=0.60,
        ),
    ],
    "gemini": [
        ModelConfig(
            id="gemini-2.0-flash",
            name="Gemini 2.0 Flash",
            provider="gemini",
            context_window=1_000_000,
            max_output_tokens=8_192,
            supports_vision=True,
            cost_per_million_input=0.075,
            cost_per_million_output=0.30,
        ),
        ModelConfig(
            id="gemini-1.5-pro",
            name="Gemini 1.5 Pro",
            provider="gemini",
            context_window=2_000_000,
            max_output_tokens=8_192,
            supports_vision=True,
            cost_per_million_input=1.25,
            cost_per_million_output=5.0,
        ),
    ],
    "ollama": [
        ModelConfig(
            id="llama3.1",
            name="Llama 3.1 (local)",
            provider="ollama",
            context_window=128_000,
            max_output_tokens=4_096,
            supports_vision=False,
        ),
        ModelConfig(
            id="mistral",
            name="Mistral (local)",
            provider="ollama",
            context_window=32_000,
            max_output_tokens=4_096,
            supports_vision=False,
        ),
    ],
    "perplexity": [
        ModelConfig(
            id="sonar-pro",
            name="Sonar Pro",
            provider="perplexity",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_tools=False,
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
        ),
        ModelConfig(
            id="sonar",
            name="Sonar",
            provider="perplexity",
            context_window=128_000,
            max_output_tokens=8_192,
            supports_tools=False,
            cost_per_million_input=1.0,
            cost_per_million_output=1.0,
        ),
    ],
}


def default_model_for(provider: str) -> ModelConfig | None:
    """Return the first (preferred) model for a provider."""
    models = PROVIDER_MODELS.get(provider, [])
    return models[0] if models else None


def find_model(model_id: str) -> ModelConfig | None:
    """Look up a model across all providers by its id."""
    for models in PROVIDER_MODELS.values():
        for m in models:
            if m.id == model_id:
                return m
    return None
