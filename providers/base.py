"""Abstract base for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Standardised response from any provider."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Every provider implements this interface."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send messages and return a full response."""

    async def stream(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield response chunks. Default falls back to generate()."""
        resp = await self.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )
        yield resp.content

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the key is valid."""

    def _timer(self) -> "_Timer":
        return _Timer()


class _Timer:
    """Tiny context-manager to measure elapsed milliseconds."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
