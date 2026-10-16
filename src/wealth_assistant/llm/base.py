"""LLM client abstraction: provider-agnostic interface for narrative generation.

The LLM is strictly optional. It is used only to write prose from structured
findings; all numbers, anomalies, risk scores and recommendations stay
deterministic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(Exception):
    """Raised for any LLM provider failure (auth, network, bad response, ...).

    All SDK-specific exceptions are wrapped into this type so callers only
    ever handle one error class.
    """


@dataclass
class LLMResponse:
    """Result of one LLM completion."""

    text: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    """Abstract provider-agnostic LLM client."""

    provider: str = "unknown"
    model: str = "unknown"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Generate a completion.

        Raises:
            LLMError: on any provider/SDK failure.
        """
        ...

    @abstractmethod
    def ping(self) -> bool:
        """Return True when the provider is reachable and usable.

        Must never raise; failures are reported as False.
        """
        ...
