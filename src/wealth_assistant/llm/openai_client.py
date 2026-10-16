"""OpenAI implementation of LLMClient (lazy SDK import)."""
from __future__ import annotations

import time
from typing import Any

from .. import config
from .base import LLMClient, LLMError, LLMResponse


class OpenAIClient(LLMClient):
    """LLMClient backed by the official ``openai`` SDK.

    The SDK is imported lazily inside ``__init__`` so the project runs
    fully even when the SDK is not installed; its absence raises LLMError.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """Create the client.

        Args:
            api_key: OpenAI API key (never logged).
            model: Model name, e.g. from the LLM_MODEL env var.
        """
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                "openai SDK is not installed; install requirements-llm.txt "
                "or unset LLM_PROVIDER"
            ) from exc
        self.provider = "openai"
        self.model = model
        try:
            self._client: Any = openai.OpenAI(
                api_key=api_key, timeout=config.LLM_TIMEOUT_S
            )
        except Exception as exc:
            raise LLMError(f"openai client init failed: {exc}") from exc

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Generate a completion with temperature=0. Wraps SDK errors."""
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"openai completion failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def ping(self) -> bool:
        """A minimal completion must succeed. Never raises."""
        try:
            self.complete("Reply with the single word: ok.", "ok", max_tokens=8)
            return True
        except LLMError:
            return False
