"""Perplexity provider — web-connected search + LLM."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import openai

from config.model_catalog import default_model_for
from providers.base import BaseProvider, Message, ProviderResponse
from security.credentials import require_credential


class PerplexityProvider(BaseProvider):
    """Perplexity uses an OpenAI-compatible API."""

    name = "perplexity"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or require_credential("perplexity")
        self._client = openai.AsyncOpenAI(
            api_key=key,
            base_url="https://api.perplexity.ai",
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        model = model or default_model_for("perplexity").id

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages
        )

        with self._timer() as t:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        choice = resp.choices[0]
        usage = resp.usage
        return ProviderResponse(
            content=choice.message.content or "",
            model=resp.model,
            provider=self.name,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            latency_ms=t.elapsed_ms,
            metadata={"citations": getattr(resp, "citations", [])},
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        model = model or default_model_for("perplexity").id

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages
        )

        stream = await self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def health_check(self) -> bool:
        try:
            resp = await self._client.chat.completions.create(
                model="sonar",
                max_tokens=16,
                messages=[{"role": "user", "content": "ping"}],
            )
            return bool(resp.choices)
        except Exception:
            return False
