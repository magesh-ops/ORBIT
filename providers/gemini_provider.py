"""Google Gemini provider."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import google.generativeai as genai

from config.model_catalog import default_model_for
from providers.base import BaseProvider, Message, ProviderResponse
from security.credentials import require_credential


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or require_credential("gemini")
        genai.configure(api_key=key)
        self._api_key = key

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
        model_id = model or default_model_for("gemini").id

        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gm = genai.GenerativeModel(
            model_name=model_id,
            generation_config=gen_config,
            system_instruction=system_prompt if system_prompt else None,
        )

        # Convert messages to Gemini's content format
        contents = self._to_gemini_contents(messages)

        with self._timer() as t:
            resp = await gm.generate_content_async(contents)

        text = resp.text or ""
        usage_meta = resp.usage_metadata
        return ProviderResponse(
            content=text,
            model=model_id,
            provider=self.name,
            usage={
                "input_tokens": getattr(usage_meta, "prompt_token_count", 0),
                "output_tokens": getattr(usage_meta, "candidates_token_count", 0),
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
        model_id = model or default_model_for("gemini").id

        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gm = genai.GenerativeModel(
            model_name=model_id,
            generation_config=gen_config,
            system_instruction=system_prompt if system_prompt else None,
        )

        contents = self._to_gemini_contents(messages)
        resp = await gm.generate_content_async(contents, stream=True)
        async for chunk in resp:
            if chunk.text:
                yield chunk.text

    async def health_check(self) -> bool:
        try:
            gm = genai.GenerativeModel("gemini-2.0-flash")
            resp = await gm.generate_content_async("ping")
            return bool(resp.text)
        except Exception:
            return False

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message list to Gemini's content format."""
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [m.content]})
        return contents
