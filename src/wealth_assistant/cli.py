"""Command-line interface: python -m wealth_assistant run ..."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .orchestrator import Orchestrator, resolve_client_ids


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m wealth_assistant",
        description="Multi-agent Wealth Advisor Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the advisor pipeline")
    run.add_argument("--data", default="data/clients.json",
                     help="Path to clients.json (default: data/clients.json)")
    target = run.add_mutually_exclusive_group()
    target.add_argument("--client", metavar="CLIENT_ID",
                        help="Process a single client")
    target.add_argument("--all", action="store_true",
                        help="Process all clients (default)")
    run.add_argument("--auto-approve", action="store_true",
                     help="Auto-approve the HITL checkpoint")
    run.add_argument("--no-llm", action="store_true",
                     help="Disable the LLM; always use template narratives")
    run.add_argument("--llm-provider", choices=["anthropic", "openai"],
                     help="Override the LLM_PROVIDER env var")
    run.add_argument("--crm-failure",
                     choices=["none", "timeout", "server_error", "not_found"],
                     help="Inject a CRM failure mode")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    return 2


def _run_command(args: argparse.Namespace) -> int:
    if args.crm_failure:
        os.environ["CRM_FAILURE_MODE"] = args.crm_failure
    if args.llm_provider and not args.no_llm:
        os.environ["LLM_PROVIDER"] = args.llm_provider

    which = args.client if args.client else "all"
    try:
        client_ids = resolve_client_ids(args.data, which)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not client_ids:
        print("error: no clients to process", file=sys.stderr)
        return 1

    orchestrator = Orchestrator(data_path=args.data, no_llm=args.no_llm)
    print(orchestrator.llm_mode)
    results = orchestrator.run(client_ids, auto_approve=args.auto_approve)

    print(f"\n{'client_id':<10}{'status':<10}{'risk':<6}"
          f"{'anomalies':<10}report")
    for result in results:
        print(f"{result['client_id']:<10}{result['status']:<10}"
              f"{result['risk_score']:<6}{result['anomaly_count']:<10}"
              f"{result['report_path']}")

    if all(r["status"] == "failed" for r in results):
        return 1
    return 0
