"""Ollama provider — local models, no API key required."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import httpx

from config.model_catalog import default_model_for
from config.settings import get_settings
from providers.base import BaseProvider, Message, ProviderResponse


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or get_settings().ollama_base_url).rstrip("/")

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
        model = model or default_model_for("ollama").id

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        with self._timer() as t:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()

        msg = data.get("message", {})
        return ProviderResponse(
            content=msg.get("content", ""),
            model=data.get("model", model),
            provider=self.name,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
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
        model = model or default_model_for("ollama").id

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(
            {"role": m.role, "content": m.content} for m in messages
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                import json

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
