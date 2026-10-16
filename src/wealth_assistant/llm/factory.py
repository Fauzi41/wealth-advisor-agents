"""Build the configured LLM client, or None when the LLM is unavailable.

The LLM is optional: any missing piece (provider not selected, key absent,
SDK absent, ping failed) logs a WARNING with the reason -- never the key --
and returns None, in which case the template narrative is used.
The result is cached per process; use reset_llm_client_cache() in tests.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple, Type

from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMError
from .openai_client import OpenAIClient

_logger = logging.getLogger(__name__)

#: provider -> (api key env var, client class)
_CLIENTS: Dict[str, Tuple[str, Type[LLMClient]]] = {
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicClient),
    "openai": ("OPENAI_API_KEY", OpenAIClient),
}

_UNSET: Any = object()
_cached: Any = _UNSET


def reset_llm_client_cache() -> None:
    """Clear the per-process client cache (used by tests)."""
    global _cached
    _cached = _UNSET


def _load_dotenv() -> None:
    """Load a .env file when python-dotenv is installed; no-op otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build() -> Optional[LLMClient]:
    _load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "none").strip().lower()
    if provider == "none":
        _logger.warning(
            "LLM disabled (LLM_PROVIDER=%r); using template narratives", provider
        )
        return None

    entry = _CLIENTS.get(provider)
    if entry is None:
        _logger.warning(
            "Unknown LLM_PROVIDER=%r; expected one of %s; "
            "using template narratives",
            provider,
            sorted(_CLIENTS),
        )
        return None

    key_name, client_cls = entry
    api_key = os.getenv(key_name)
    if not api_key:
        _logger.warning(
            "LLM provider %r selected but %s is not set; "
            "using template narratives",
            provider,
            key_name,
        )
        return None

    model = (os.getenv("LLM_MODEL") or "").strip()
    if not model:
        _logger.warning(
            "LLM provider %r selected but LLM_MODEL is not set; "
            "using template narratives",
            provider,
        )
        return None

    try:
        client = client_cls(api_key=api_key, model=model)
    except LLMError as exc:
        _logger.warning(
            "LLM client init failed for provider %r: %s; "
            "using template narratives",
            provider,
            exc,
        )
        return None

    try:
        alive = client.ping()
    except Exception as exc:  # ping must not raise, but stay safe
        _logger.warning(
            "LLM ping raised for provider %r: %s; using template narratives",
            provider,
            exc,
        )
        return None
    if not alive:
        _logger.warning(
            "LLM ping failed for provider %r model %r; "
            "using template narratives",
            provider,
            model,
        )
        return None
    return client


def build_llm_client() -> Optional[LLMClient]:
    """Return the configured LLM client, or None when unavailable.

    The result is cached per process: repeated calls return the same
    instance (or the same None).
    """
    global _cached
    if _cached is _UNSET:
        _cached = _build()
    return _cached
