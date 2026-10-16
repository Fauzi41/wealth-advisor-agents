"""ClientLoaderTool: load and per-client-validate data/clients.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from ..models import Client
from .base import Tool, ToolResult

#: Optional client fields: absent ones get safe defaults plus a warning.
OPTIONAL_FIELDS = (
    "age",
    "monthly_income",
    "target_allocation",
    "holdings",
    "transactions",
)

DEFAULT_PATH = "data/clients.json"


class ClientLoaderTool(Tool):
    """Load clients from a JSON file, validating each record separately.

    A malformed client is reported in ``data["errors"]`` and never fails
    the whole file. Missing optional fields produce data-quality warnings
    in ``data["warnings"]``. A missing file or invalid top-level JSON
    returns ``ToolResult(ok=False)``.
    """

    name = "client_loader"

    def _execute(self, path: str = DEFAULT_PATH) -> ToolResult:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ToolResult(
                ok=False,
                error=f"clients file not found: {path}",
                meta={"path": path},
            )
        except OSError as exc:
            return ToolResult(
                ok=False,
                error=f"cannot read clients file {path}: {exc}",
                meta={"path": path},
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return ToolResult(
                ok=False,
                error=f"invalid JSON in {path}: {exc}",
                meta={"path": path},
            )
        if not isinstance(payload, list):
            return ToolResult(
                ok=False,
                error=f"invalid top-level JSON in {path}: expected a list of clients",
                meta={"path": path},
            )

        clients: List[Client] = []
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append(
                    {
                        "index": index,
                        "client_id": None,
                        "error": f"expected object, got {type(item).__name__}",
                    }
                )
                continue
            try:
                client = Client(**item)
            except ValidationError as exc:
                errors.append(
                    {
                        "index": index,
                        "client_id": item.get("client_id"),
                        "error": str(exc),
                    }
                )
                continue
            clients.append(client)
            missing = [f for f in OPTIONAL_FIELDS if f not in item]
            if missing:
                warnings.append(
                    {"client_id": client.client_id, "missing_fields": missing}
                )

        return ToolResult(
            ok=True,
            data={"clients": clients, "errors": errors, "warnings": warnings},
            meta={
                "path": path,
                "total": len(payload),
                "valid": len(clients),
                "invalid": len(errors),
            },
        )
