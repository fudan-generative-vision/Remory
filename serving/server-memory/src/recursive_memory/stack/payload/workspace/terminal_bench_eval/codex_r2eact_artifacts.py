"""Shared paths for host-owned Codex r^2eAct controller artifacts."""

from __future__ import annotations


CONTROLLER_ARTIFACT_DIRNAME = ".r2eact-controller"
CONTROLLER_ARTIFACT_GUEST_DIR = f"/logs/agent/{CONTROLLER_ARTIFACT_DIRNAME}"

GATEWAY_EVENTS_FILENAME = "codex-r2eact-gateway-events.jsonl"
# New sessions persist one durable receipt per line.  Keep the historical JSON
# array name as a read-only migration source for in-flight and archived runs.
GATEWAY_RECEIPTS_JSONL_FILENAME = "codex-r2eact-gateway-receipts.jsonl"
GATEWAY_RECEIPTS_FILENAME = "codex-r2eact-gateway-receipts.json"
GATEWAY_RECEIPTS_SUMMARY_FILENAME = "codex-r2eact-gateway-receipts-summary.json"
GATEWAY_RECEIPT_RECORD_SCHEMA = "codex_r2eact_gateway_receipt_record_v1"
GATEWAY_RECEIPT_SUMMARY_SCHEMA = "codex_r2eact_gateway_receipt_summary_v1"
GATEWAY_RECEIPT_FILENAMES = (
    GATEWAY_RECEIPTS_JSONL_FILENAME,
    GATEWAY_RECEIPTS_FILENAME,
)
GATEWAY_ROUNDS_FILENAME = "codex-r2eact-rounds.jsonl"
GATEWAY_TOOL_RESPONSES_FILENAME = "codex-r2eact-tool-responses.jsonl"
GATEWAY_IMAGE_RECEIPTS_FILENAME = "codex-r2eact-image-receipts.jsonl"
GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME = "codex-r2eact-web-search.jsonl"
MEMORY_FILENAME = "memory.jsonl"

CONTROLLER_ARTIFACT_FILENAMES = frozenset(
    {
        GATEWAY_EVENTS_FILENAME,
        GATEWAY_RECEIPTS_JSONL_FILENAME,
        GATEWAY_RECEIPTS_FILENAME,
        GATEWAY_RECEIPTS_SUMMARY_FILENAME,
        GATEWAY_ROUNDS_FILENAME,
        GATEWAY_TOOL_RESPONSES_FILENAME,
        GATEWAY_IMAGE_RECEIPTS_FILENAME,
        GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
        MEMORY_FILENAME,
    }
)


__all__ = [
    "CONTROLLER_ARTIFACT_DIRNAME",
    "CONTROLLER_ARTIFACT_FILENAMES",
    "CONTROLLER_ARTIFACT_GUEST_DIR",
    "GATEWAY_EVENTS_FILENAME",
    "GATEWAY_IMAGE_RECEIPTS_FILENAME",
    "GATEWAY_RECEIPT_FILENAMES",
    "GATEWAY_RECEIPT_RECORD_SCHEMA",
    "GATEWAY_RECEIPT_SUMMARY_SCHEMA",
    "GATEWAY_RECEIPTS_FILENAME",
    "GATEWAY_RECEIPTS_JSONL_FILENAME",
    "GATEWAY_RECEIPTS_SUMMARY_FILENAME",
    "GATEWAY_ROUNDS_FILENAME",
    "GATEWAY_TOOL_RESPONSES_FILENAME",
    "GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME",
    "MEMORY_FILENAME",
]
