"""Interactive Codex, sharing the installed TB model/policy/tool contract."""

from __future__ import annotations

import contextlib
import secrets
import signal
import subprocess
import tempfile
from pathlib import Path

from .deploy import atomic
from .runtime import tb_profile


def gateway_config(state, memory_path):
    from terminal_bench_eval.codex_r2eact_gateway import R2EActGatewayConfig

    # A single source of truth. Do not let the interactive launcher drift from TB.
    kwargs = tb_profile(state, "interactive")["agents"][0]["kwargs"]
    names = (
        "specforge_root",
        "checkpoint",
        "corpus_dir",
        "sglang_url",
        "working_memory_turns",
        "serving_eviction_policy",
        "residual_compaction",
        "native_compaction",
        "native_compaction_text_summary",
        "qwen_reasoning_effort",
        "vision_enabled",
        "web_search_enabled",
        "update_plan_enabled",
    )
    values = {key: kwargs[key] for key in names}
    for key in ("specforge_root", "checkpoint", "corpus_dir"):
        values[key] = Path(values[key])
    return R2EActGatewayConfig(**values, memory_guest_path=str(memory_path))


def codex_command(binary, workspace, *, unsafe=False, prompt=None):
    from terminal_bench_eval.codex_r2eact_agent import CODEX_R2EACT_MODEL_SLUG

    argv = [
        str(binary),
        "-C",
        str(workspace),
        "-m",
        CODEX_R2EACT_MODEL_SLUG,
        "--no-alt-screen",
        "--enable",
        "unified_exec",
    ]
    if unsafe:
        argv.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        argv.extend(["-s", "workspace-write", "-a", "on-request"])
    if prompt:
        argv.append(prompt)
    return argv


def run_terminal(argv, workspace, env):
    """Codex owns Ctrl+C cancellation; TERM/HUP still tear down our child."""
    child = None
    previous = {}

    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt

    try:
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, lambda *_: None)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, terminate)
        # Same foreground process group/TTY: Ctrl+C reaches Codex, not an orphan PTY.
        child = subprocess.Popen(argv, cwd=workspace, env=env)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def interactive(state, args):
    from shared_runtime.residual_backend import ResidualBackend
    from shared_runtime.residual_preflight import get_server_info
    from shared_runtime.runner import (
        document_servers,
        local_runtime,
        safe_environment,
        toml_value,
    )
    from terminal_bench_eval.codex_r2eact_agent import (
        _build_codex_runtime_config,
        _build_merged_model_catalog,
    )
    from terminal_bench_eval.codex_r2eact_gateway import (
        CodexGatewayServer,
        CodexResponsesGateway,
    )
    from terminal_bench_eval.document_tools import INSTRUCTIONS

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("--workspace must be an existing directory")
    prefix = Path(state["prefix"])
    if workspace == prefix or workspace.is_relative_to(prefix):
        raise ValueError("choose a working directory outside the installed runtime")
    sessions = prefix / "runs/chat"
    sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = Path(tempfile.mkdtemp(prefix="session-", dir=sessions))
    for name in (".codex", "tmp", "gateway"):
        (session / name).mkdir(mode=0o700)
    # Only the memory archive is exposed in the working directory; gateway
    # receipts and the isolated Codex home stay in the private session directory.
    memory_root = workspace / ".r2eact"
    if memory_root.is_symlink():
        raise ValueError("workspace .r2eact must not be a symlink")
    memory_root.mkdir(exist_ok=True, mode=0o700)
    memory = Path(tempfile.mkdtemp(prefix="session-", dir=memory_root)) / "memory.jsonl"
    with memory.open("x"):
        pass
    memory.chmod(0o600)
    config = gateway_config(state, memory)
    live_server_info = get_server_info(config.sglang_url)
    binary, documents = local_runtime()
    backend = ResidualBackend(config, session / "gateway", memory_mirror_path=memory)
    gateway = CodexResponsesGateway(
        backend,
        session / "gateway",
        native_compaction=config.native_compaction,
        native_compaction_text_summary=config.native_compaction_text_summary,
        working_memory_turns=config.working_memory_turns,
        vision_enabled=config.vision_enabled,
        web_search_enabled=config.web_search_enabled,
        update_plan_enabled=config.update_plan_enabled,
        trusted_mcp_aliases={"docling": "docling", "mineru": "mineru"},
    )
    token = "sk-local-" + secrets.token_hex(24)
    server = CodexGatewayServer(gateway, bearer_token=token)
    # shutdown() requires serve_forever() to be running, including error paths.
    server.start()
    try:
        runtime = _build_codex_runtime_config(
            server.base_url,
            native_compaction=config.native_compaction,
            web_search_enabled=config.web_search_enabled,
            mcp_server_url=server.mcp_url,
            trusted_mcp_servers=document_servers(documents),
            server_info=live_server_info,
        )
        catalog = session / ".codex/model-catalog.json"
        catalog.write_bytes(_build_merged_model_catalog(binary))
        runtime["model_catalog_json"] = str(catalog)
        # Tool guidance is shared with TB; adapt only its guest-filesystem paths
        # because interactive MCPs see the selected host working directory.
        runtime["developer_instructions"] = INSTRUCTIONS.replace(
            "inside this task environment", "inside the working directory"
        ).replace(
            "Saved document outputs are available under /tmp/r2eact-documents.",
            "Read saved documents at the paths returned by the document tools.",
        )
        (session / ".codex/config.toml").write_text(
            "\n".join(f"{key} = {toml_value(value)}" for key, value in runtime.items())
            + "\n"
        )
        atomic(
            session / "session.json",
            {
                "workspace": str(workspace),
                "memory": str(memory),
                "checkpoint": str(config.checkpoint),
                "harness_mode": "native_compact",
                "reasoning_effort": config.qwen_reasoning_effort,
                "working_memory_turns": None,
                "serving_eviction_policy": config.serving_eviction_policy,
                "profile": tb_profile(state, "interactive")["agents"][0],
            },
        )
        env = safe_environment(session, token)
        env["CACHE_DIR"] = str(session / "documents/docling")
        env["OUTPUT_DIR"] = str(session / "documents/mineru")
        print(f"[stack] interactive session: {session}", flush=True)
        print(f"[stack] memory archive: {memory}", flush=True)
        return run_terminal(
            codex_command(binary, workspace, unsafe=args.unsafe, prompt=args.prompt),
            workspace,
            env,
        )
    finally:
        # Closing the gateway releases the model lock; never stop another SGLang.
        with contextlib.suppress(Exception):
            server.close()
        with contextlib.suppress(Exception):
            backend.close()
