"""Use JobBench's exact document/runtime tools alongside official business APIs."""
from __future__ import annotations

from pathlib import Path

JOBBENCH_TOOLS = "jobbench_plus_automationbench_v1"
LEGACY_TOOLS = "automationbench_api_shell_plan_v1"


def tool_profile(args):
    # Requests already written by pre-upgrade workers intentionally retain
    # their old tool ABI. New workers explicitly stamp JOBBENCH_TOOLS.
    profile = getattr(args, "tool_profile", LEGACY_TOOLS)
    if profile not in {JOBBENCH_TOOLS, LEGACY_TOOLS}:
        raise ValueError(f"Unknown AutomationBench tool profile: {profile}")
    return profile


def prepare_tools(profile):
    if profile == LEGACY_TOOLS:
        from .runner import local_codex
        return local_codex(), None
    if profile != JOBBENCH_TOOLS:
        raise ValueError(f"Unknown AutomationBench tool profile: {profile}")
    from shared_runtime.runner import preflight
    # Same pinned Codex, node-local CPU document venv, credential checks and
    # unprivileged Landlock boundary as JobBench. This does NOT claim its pool.
    return preflight()


def mcp_servers(official_url, document_venv):
    servers = {"automationbench": {"url": official_url,
        "bearer_token_env_var": "OPENAI_API_KEY", "required": True,
        "startup_timeout_sec": 30, "tool_timeout_sec": 120}}
    if document_venv is not None:
        from shared_runtime.runner import document_servers
        servers.update(document_servers(Path(document_venv)))
    return servers


def instructions(document_venv):
    if document_venv is None:
        return ""
    return (
        "\nFor external references use the search MCP (Serper) and shell HTTP tools. "
        "This is distinct from AutomationBench api_search, which searches simulated API documentation. "
        "External searches do not read or modify the simulated business world. "
        "Use Docling MCP to convert documents, then export_docling_document_to_markdown to read them. "
        "If Docling fails or extraction is unusable, use MinerU MCP as a fallback; "
        "MinerU uploads the selected documents to its parsing service. "
        "For images, render documents/pages to local files and call view_image with the file path. "
        "You may install necessary Python packages locally under your workspace, not system directories. "
        f"The document-tools Python interpreter is {document_venv / 'bin/python'}; "
        "it includes pandas, openpyxl, python-docx, python-pptx, pdfplumber and matplotlib."
    )
