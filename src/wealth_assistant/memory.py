"""Long-term memory: per-client run summaries persisted as JSON.

Each summary holds: timestamp, risk_score, anomaly type+id+severity list,
recommendation actions and total balance. Used for balance-drop detection
and for ``changes_since_last_review`` in the report.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

DEFAULT_PATH = "memory_store/memory.json"

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


class LongTermMemory:
    """JSON-file-backed per-client memory with atomic writes."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        """Create the memory, loading existing summaries if present.

        A missing or corrupt file starts empty and logs a WARNING.
        """
        self.path = Path(path)
        self._store: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            _logger.warning(
                "memory_read_failed: %s (%s); starting empty", self.path, exc
            )
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            _logger.warning(
                "memory_corrupt: %s (%s); starting empty", self.path, exc
            )
            return {}
        if not isinstance(data, dict):
            _logger.warning(
                "memory_corrupt: %s (expected a JSON object); starting empty",
                self.path,
            )
            return {}
        return data

    def get_last_summary(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Return the last stored summary for a client, or None."""
        return self._store.get(client_id)

    def save_summary(self, client_id: str, summary: Dict[str, Any]) -> None:
        """Store a run summary (timestamp is added) with an atomic write."""
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **summary}
        self._store[client_id] = record
        self._write_atomic()

    def _write_atomic(self) -> None:
        """Write via temp file + rename so a crash never leaves half a file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".memory-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._store, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        except OSError as exc:
            _logger.warning("memory_write_failed: %s (%s)", self.path, exc)
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def diff_with_last(
        self, client_id: str, current: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Diff current anomalies against the last summary.

        Args:
            client_id: Client to compare against.
            current: {"anomalies": [{"type", "id", "severity"}]}.

        Returns:
            {"new": [ids], "resolved": [ids], "worsened": [ids]}, where
            "worsened" means present in both runs with a higher severity.
            On a first run every current anomaly counts as new.
        """
        current_map = {
            a["id"]: a for a in current.get("anomalies", []) if "id" in a
        }
        last = self.get_last_summary(client_id)
        if last is None:
            return {
                "new": sorted(current_map),
                "resolved": [],
                "worsened": [],
            }
        last_map = {
            a["id"]: a for a in last.get("anomalies", []) if "id" in a
        }
        new = sorted(i for i in current_map if i not in last_map)
        resolved = sorted(i for i in last_map if i not in current_map)
        worsened = sorted(
            i
            for i in current_map
            if i in last_map
            and _SEVERITY_RANK.get(str(current_map[i].get("severity")), 0)
            > _SEVERITY_RANK.get(str(last_map[i].get("severity")), 0)
        )
        return {"new": new, "resolved": resolved, "worsened": worsened}
