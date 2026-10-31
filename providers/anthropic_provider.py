"""Anthropic (Claude) provider."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import anthropic

from config.model_catalog import default_model_for
from providers.base import BaseProvider, Message, ProviderResponse
from security.credentials import require_credential


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or require_credential("anthropic")
        self._client = anthropic.AsyncAnthropic(api_key=key)

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
        model = model or default_model_for("anthropic").id

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_prompt:
            params["system"] = system_prompt

        with self._timer() as t:
            resp = await self._client.messages.create(**params)

        return ProviderResponse(
            content=resp.content[0].text,
            model=resp.model,
            provider=self.name,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            latency_ms=t.elapsed_ms,
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
        model = model or default_model_for("anthropic").id
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_prompt:
            params["system"] = system_prompt

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        try:
            resp = await self._client.messages.create(
                model="claude-haiku-3-5-20241022",
                max_tokens=16,
                messages=[{"role": "user", "content": "ping"}],
            )
            return bool(resp.content)
        except Exception:
            return False
