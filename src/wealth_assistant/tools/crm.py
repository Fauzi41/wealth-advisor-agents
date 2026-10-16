"""Mock CRM client with failure injection, wrapped by CRMTool with retries."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BACKOFF_BASE_S, RETRY_COUNT
from ..models import CRMRecord
from .base import Tool, ToolResult

#: Supported values for the CRM_FAILURE_MODE environment variable.
FAILURE_MODES = ("none", "timeout", "server_error", "not_found")

DEFAULT_PATH = "data/crm_mock.json"


class CRMError(Exception):
    """Base class for simulated CRM failures."""


class CRMTimeoutError(CRMError):
    """Simulated CRM timeout."""


class CRMServerError(CRMError):
    """Simulated CRM 5xx error."""


class MockCRMClient:
    """File-backed mock CRM with env-var failure injection.

    Set ``CRM_FAILURE_MODE`` to one of ``none|timeout|server_error|not_found``
    (default ``none``) to simulate outages in tests and demos. An explicit
    ``failure_mode`` argument overrides the environment variable.
    """

    def __init__(
        self, path: str = DEFAULT_PATH, failure_mode: Optional[str] = None
    ) -> None:
        """Create a client.

        Args:
            path: JSON file holding the CRM records.
            failure_mode: Override for the CRM_FAILURE_MODE env var.
        """
        self.path = path
        self._failure_mode_override = failure_mode

    @property
    def failure_mode(self) -> str:
        """Active failure mode, validated against FAILURE_MODES."""
        mode = self._failure_mode_override or os.getenv("CRM_FAILURE_MODE", "none")
        if mode not in FAILURE_MODES:
            raise ValueError(
                f"invalid CRM_FAILURE_MODE={mode!r}; expected one of {FAILURE_MODES}"
            )
        return mode

    def get_record(self, client_id: str) -> Optional[CRMRecord]:
        """Return the CRM record for ``client_id``, or None when not found.

        Raises:
            CRMTimeoutError: when failure mode is ``timeout``.
            CRMServerError: when failure mode is ``server_error``.
        """
        mode = self.failure_mode
        if mode == "timeout":
            raise CRMTimeoutError("simulated CRM timeout")
        if mode == "server_error":
            raise CRMServerError("simulated CRM 500 error")
        if mode == "not_found":
            return None
        for record in self._load():
            if record.get("client_id") == client_id:
                return CRMRecord(**record)
        return None

    def _load(self) -> List[Dict[str, Any]]:
        text = Path(self.path).read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise CRMError(f"invalid CRM data in {self.path}: expected a list")
        return payload


class CRMTool(Tool):
    """Look up a CRM record with retries on transient failures.

    - ``timeout`` / ``server_error``: retried RETRY_COUNT times with
      exponential backoff, then ``ToolResult(ok=False)``.
    - ``not_found``: NOT retried; returns ``ok=True`` with ``data=None`` so
      the orchestrator can continue in degraded mode.
    Never raises.
    """

    name = "crm_lookup"

    def _execute(self, client_id: str, path: str = DEFAULT_PATH) -> ToolResult:
        client = MockCRMClient(path=path)
        attempts = 0
        last_error: Optional[str] = None
        while attempts < RETRY_COUNT:
            attempts += 1
            try:
                record = client.get_record(client_id)
            except (CRMTimeoutError, CRMServerError) as exc:
                last_error = str(exc)
                if attempts < RETRY_COUNT:
                    time.sleep(BACKOFF_BASE_S * (2 ** (attempts - 1)))
                continue
            # Success, or a definitive not_found (None): do not retry.
            return ToolResult(
                ok=True,
                data=record,
                meta={
                    "client_id": client_id,
                    "found": record is not None,
                    "attempts": attempts,
                },
            )
        return ToolResult(
            ok=False,
            error=f"CRM lookup failed after {RETRY_COUNT} attempts: {last_error}",
            meta={"client_id": client_id, "attempts": attempts},
        )
