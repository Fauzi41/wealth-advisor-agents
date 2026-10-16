"""Anthropic implementation of LLMClient (lazy SDK import)."""
from __future__ import annotations

import time
from typing import Any

from .. import config
from .base import LLMClient, LLMError, LLMResponse


class AnthropicClient(LLMClient):
    """LLMClient backed by the official ``anthropic`` SDK.

    The SDK is imported lazily inside ``__init__`` so the project runs
    fully even when the SDK is not installed; its absence raises LLMError.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """Create the client.

        Args:
            api_key: Anthropic API key (never logged).
            model: Model name, e.g. from the LLM_MODEL env var.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "anthropic SDK is not installed; install requirements-llm.txt "
                "or unset LLM_PROVIDER"
            ) from exc
        self.provider = "anthropic"
        self.model = model
        try:
            self._client: Any = anthropic.Anthropic(
                api_key=api_key, timeout=config.LLM_TIMEOUT_S
            )
        except Exception as exc:
            raise LLMError(f"anthropic client init failed: {exc}") from exc

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Generate a completion with temperature=0. Wraps SDK errors."""
        start = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"anthropic completion failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = "".join(
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def ping(self) -> bool:
        """A minimal completion must succeed. Never raises."""
        try:
            self.complete("Reply with the single word: ok.", "ok", max_tokens=8)
            return True
        except LLMError:
            return False
