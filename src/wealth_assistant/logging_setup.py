"""Structured JSON-lines logging.

Every log line is one JSON object with the fields:
ts, level, run_id, agent, event, message, extra.

Lines go to the console (stdout) and are appended to outputs/run.log.

Usage:
    logger = get_logger(__name__, run_id=ctx.run_id)
    agent_log = logger.bind(agent="analyzer")
    agent_log.info("analyze_complete", "analysis finished", extra={"anomalies": 3})
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path("outputs")
LOG_FILE = LOG_DIR / "run.log"


class _JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "run_id": getattr(record, "run_id", None),
            "agent": getattr(record, "agent", None),
            "event": getattr(record, "event", ""),
            "message": record.getMessage(),
            "extra": getattr(record, "fields", None) or {},
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def _ensure_handlers(logger: logging.Logger) -> None:
    """Attach console + file handlers exactly once per logger name."""
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_JsonFormatter())
    logger.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)


class RunLogger:
    """Thin wrapper binding run_id / agent context to a standard logger."""

    def __init__(
        self,
        logger: logging.Logger,
        run_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        self._logger = logger
        self._run_id = run_id
        self._agent = agent

    def bind(
        self, agent: Optional[str] = None, run_id: Optional[str] = None
    ) -> "RunLogger":
        """Return a new logger with overridden agent / run_id bindings."""
        return RunLogger(
            self._logger,
            run_id=run_id if run_id is not None else self._run_id,
            agent=agent if agent is not None else self._agent,
        )

    def _emit(
        self,
        level: int,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        self._logger.log(
            level,
            message,
            extra={
                "run_id": self._run_id,
                "agent": agent or self._agent,
                "event": event,
                "fields": dict(extra or {}),
            },
            exc_info=exc_info,
        )

    def debug(
        self,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a DEBUG line (suppressed unless the level is lowered)."""
        self._emit(logging.DEBUG, event, message, agent, extra)

    def info(
        self,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an INFO line."""
        self._emit(logging.INFO, event, message, agent, extra)

    def warning(
        self,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a WARNING line."""
        self._emit(logging.WARNING, event, message, agent, extra)

    def error(
        self,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an ERROR line."""
        self._emit(logging.ERROR, event, message, agent, extra)

    def exception(
        self,
        event: str,
        message: str = "",
        agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an ERROR line with the current exception traceback.

        Must be called from inside an ``except`` block.
        """
        self._emit(logging.ERROR, event, message, agent, extra, exc_info=True)


def get_logger(
    name: str,
    run_id: Optional[str] = None,
    agent: Optional[str] = None,
) -> RunLogger:
    """Create (or reuse) a JSON-lines logger bound to a run_id.

    Args:
        name: Logger name, usually ``__name__``.
        run_id: Run identifier stamped on every emitted line.
        agent: Default agent name; can be overridden per call or via bind().

    Returns:
        A RunLogger writing JSON lines to console and outputs/run.log.
    """
    logger = logging.getLogger(name)
    _ensure_handlers(logger)
    return RunLogger(logger, run_id=run_id, agent=agent)
