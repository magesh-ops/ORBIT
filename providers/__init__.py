"""Provider registry — import and instantiate providers by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.base import BaseProvider, Message, ProviderResponse

if TYPE_CHECKING:
    from providers.omniroute import OmniRoute

# Lazy imports to avoid pulling in SDK dependencies until needed
_PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
    "anthropic": ("providers.anthropic_provider", "AnthropicProvider"),
    "openai": ("providers.openai_provider", "OpenAIProvider"),
    "gemini": ("providers.gemini_provider", "GeminiProvider"),
    "ollama": ("providers.ollama", "OllamaProvider"),
    "perplexity": ("providers.perplexity_provider", "PerplexityProvider"),
}


def create_provider(name: str, **kwargs) -> BaseProvider:
    """Instantiate a provider by name. Raises KeyError for unknown names."""
    if name not in _PROVIDER_CLASSES:
        raise KeyError(
            f"Unknown provider '{name}'. "
            f"Available: {', '.join(sorted(_PROVIDER_CLASSES))}"
        )
    module_path, class_name = _PROVIDER_CLASSES[name]
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def create_available_providers() -> dict[str, BaseProvider]:
    """Create provider instances for all configured providers.

    Skips providers whose credentials are missing (except ollama).
    """
    from config.settings import get_settings

    settings = get_settings()
    providers: dict[str, BaseProvider] = {}

    for name in settings.available_providers():
        try:
            providers[name] = create_provider(name)
        except Exception:
            continue  # skip providers that fail to initialise

    return providers


__all__ = [
    "BaseProvider",
    "Message",
    "ProviderResponse",
    "create_available_providers",
    "create_provider",
]
