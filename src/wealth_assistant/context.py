"""Per-run mutable context shared by the agents in a pipeline run."""
from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from .models import TraceEntry


class RunContext:
    """Holds everything agents need to share during one client run.

    Attributes:
        run_id: Unique id for the run (uuid4 by default).
        client_id: The client being processed.
        trace: Ordered list of TraceEntry records, one per pipeline step.
        warnings: Data-quality / degradation warnings surfaced in the report.
    """

    def __init__(self, client_id: str, run_id: Optional[str] = None) -> None:
        """Create a context for one client run.

        Args:
            client_id: The client being processed.
            run_id: Optional explicit run id; a uuid4 string is generated
                when omitted.
        """
        self.run_id: str = run_id or str(uuid4())
        self.client_id: str = client_id
        self._store: dict[str, Any] = {}
        self.trace: list[TraceEntry] = []
        self.warnings: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        """Return a stored value, or ``default`` when the key is absent."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value for later steps / agents in this run."""
        self._store[key] = value

    def add_trace(
        self, step: str, agent: str, status: str, duration_ms: int
    ) -> TraceEntry:
        """Append a pipeline step record and return it."""
        entry = TraceEntry(
            step=step, agent=agent, status=status, duration_ms=duration_ms
        )
        self.trace.append(entry)
        return entry

    def add_warning(self, message: str) -> None:
        """Record a warning surfaced in the final report."""
        self.warnings.append(message)
