"""DataFetcherAgent: INGEST step -- load the client and its CRM record."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import RunContext
from ..models import Client, CRMRecord
from .base import AgentResult, BaseAgent


class DataFetcherAgent(BaseAgent):
    """Load the client record and CRM context for this run.

    Uses the ``client_loader`` tool once per run (the full client list is
    cached in the context) and the ``crm_lookup`` tool. Writes ``client``
    and ``crm`` to the context and records data-quality warnings for
    missing optional fields.
    """

    name = "data_fetcher"

    def run(self, context: RunContext) -> AgentResult:
        """Fetch client + CRM data into the context."""
        log = self._log(context)
        log.info("fetch_start", f"loading data for client {context.client_id}")

        client = self._load_client(context, log)
        if client is None:
            return AgentResult(
                status="failed",
                message=f"client {context.client_id} could not be loaded",
            )
        context.set("client", client)

        degraded, crm = self._load_crm(context, log, client)
        context.set("crm", crm)

        log.info(
            "fetch_end",
            f"client={client.client_id} crm_found={crm is not None}",
        )
        if degraded:
            return AgentResult(
                status="degraded", message="client loaded; CRM degraded"
            )
        return AgentResult(status="ok", message="client and CRM loaded")

    def _load_client(self, context: RunContext, log: Any) -> Optional[Client]:
        """Return the client, loading (once) and caching the client list."""
        cached = context.get("client")
        if cached is not None:
            return cached

        cache: Optional[Dict[str, Client]] = context.get("clients_cache")
        warnings: List[Dict[str, Any]] = []
        if cache is None:
            result = self.registry.get("client_loader").run()
            if not result.ok:
                log.error("client_load_failed", result.error or "unknown error")
                return None
            data = result.data or {}
            clients: List[Client] = data.get("clients", [])
            cache = {c.client_id: c for c in clients}
            context.set("clients_cache", cache)
            warnings = data.get("warnings", [])
            log.info(
                "clients_loaded",
                f"{len(clients)} valid, "
                f"{len(data.get('errors', []))} invalid",
            )

        client = cache.get(context.client_id)
        if client is None:
            log.error("client_not_found", f"client {context.client_id} not found")
            return None
        for warning in warnings:
            if warning.get("client_id") == client.client_id:
                message = (
                    "missing optional fields: "
                    + ", ".join(warning.get("missing_fields", []))
                )
                context.add_warning(message)
                log.warning("data_quality", message)
        return client

    def _load_crm(
        self, context: RunContext, log: Any, client: Client
    ) -> tuple:
        """Return (degraded, crm_record). CRM problems degrade, never fail."""
        result = self.registry.get("crm_lookup").run(client_id=client.client_id)
        if not result.ok:
            message = f"CRM lookup failed: {result.error}"
            context.add_warning(message)
            log.warning("crm_failed", message)
            return True, None
        crm: Optional[CRMRecord] = result.data
        if crm is None:
            message = "no CRM record"
            context.add_warning(message)
            log.warning("crm_not_found", message)
            return True, None
        log.info("crm_loaded", f"advisor={crm.advisor_name}")
        return False, crm
