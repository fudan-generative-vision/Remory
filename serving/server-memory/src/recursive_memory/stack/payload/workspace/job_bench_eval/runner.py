"""One JobBench execution using the unchanged TB Codex/Compressor gateway."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import JobBenchConfig
from .dataset import JobBenchTask, prepare_private_workspace
from shared_runtime.storage import atomic_json, read_json, utcnow, lock


from shared_runtime.runner import document_servers, safe_environment, sandbox_command, start_codex, local_runtime, preflight, toml_value
from shared_runtime.residual_preflight import get_server_info

DEFAULT_SPECFORGE = Path(__file__).resolve().parent.parent

def prepare_scratch_root(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o711)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise RuntimeError("workspace root must be an owned directory without group/other write access")
    # mkdir's mode is masked by a private controller umask (often 077).
    # Delegated task UIDs need traversal, but cannot list sibling workspaces.
    path.chmod(0o711)
    return path

def collect_output(workspace, execution):
    """Copy regular deliverables only; reject links/special files, hash each file."""
    source = workspace / "output"
    target = execution / "model_output"
    target.mkdir(exist_ok=True)
    manifest = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"unsafe deliverable: {path.relative_to(source)}")
        if path.is_file():
            relative = path.relative_to(source)
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            manifest.append({"path": relative.as_posix(), "bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    atomic_json(execution / "deliverables.json", {"files": manifest})
    return manifest

def run_jobbench_task(config, task, execution, *, sglang_url="http://127.0.0.1:30000", prompt_builder=None,
                     system_instructions=None):
    from terminal_bench_eval.codex_r2eact_agent import CODEX_R2EACT_MODEL_SLUG, _build_codex_runtime_config, _build_merged_model_catalog
    from terminal_bench_eval.codex_r2eact_gateway import R2EActGatewayConfig, CodexResponsesGateway, CodexGatewayServer

    config.validate()
    binary, mcp = preflight()
    live_server_info = get_server_info(sglang_url)
    execution = Path(execution).resolve()
    execution.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Work locally (Lustre is incompatible with this host's Landlock ABI).
    # Controller trajectory/memory and final deliverables are shared-persistent.
    scratch_root = Path(os.environ.get("JOBBENCH_WORKSPACE_ROOT", f"/tmp/jobbench-workspaces-{os.getuid()}"))
    prepare_scratch_root(scratch_root)
    workspace = Path(tempfile.mkdtemp(prefix="task-", dir=scratch_root))
    prepare_private_workspace(task, workspace / "task_folder")
    for name in ("output", "tmp", ".codex"):
        (workspace / name).mkdir()
    native = config.harness_mode == "native_compact"
    specforge = Path(os.environ.get("SPECFORGE_ROOT", str(DEFAULT_SPECFORGE)))
    gateway_config = R2EActGatewayConfig(
        specforge_root=specforge,
        checkpoint=Path(config.residual_checkpoint),
        compressor_enabled=config.compressor_enabled,
        residual_compaction=config.residual_compaction,
        corpus_dir=Path(os.environ.get("R2EACT_CORPUS_DIR", str(specforge / "data/wiki-18-corpus"))),
        sglang_url=sglang_url, working_memory_turns=config.working_memory_turns,
        serving_eviction_policy=config.serving_eviction_policy,
        native_compaction=native, native_compaction_text_summary=config.native_compaction_text_summary,
        vision_enabled=True, web_search_enabled=True, update_plan_enabled=True, qwen_reasoning_effort="xhigh",
        memory_guest_path=str(workspace / "memory.jsonl"),
    )
    trace = execution / "gateway"
    trace.mkdir(mode=0o700)
    # The paired backend shares the exact summary contract and switches to a
    # transport that rejects encoder/soft-memory calls when Compressor is off.
    from shared_runtime.residual_backend import ResidualBackend
    backend = ResidualBackend(gateway_config, trace, memory_mirror_path=workspace / "memory.jsonl")
    from shared_runtime.subagents import gateway_class
    gateway = gateway_class()(backend, trace, native_compaction=native,
        native_compaction_text_summary=config.native_compaction_text_summary,
        working_memory_turns=config.working_memory_turns, vision_enabled=True,
        web_search_enabled=True, update_plan_enabled=True, trusted_mcp_aliases={"docling": "docling", "mineru": "mineru"})
    token = "sk-jobbench-" + secrets.token_hex(24)
    server = CodexGatewayServer(gateway, bearer_token=token)
    runtime = _build_codex_runtime_config(server.base_url, native_compaction=native,
        web_search_enabled=True, mcp_server_url=server.mcp_url, trusted_mcp_servers=document_servers(mcp), server_info=live_server_info)
    if system_instructions:
        # Codex sends developer instructions separately; the gateway folds them
        # into the persistent Qwen system message, outside compressed history.
        runtime["developer_instructions"] = "\n\n".join(filter(None, (
            runtime.get("developer_instructions"), system_instructions)))
        atomic_json(execution / "system-instructions.json", {
            "text": system_instructions,
            "sha256": hashlib.sha256(system_instructions.encode()).hexdigest(),
            "transport": "codex_developer_to_persistent_qwen_system_v1"})
    catalog = workspace / ".codex/model-catalog.json"
    catalog.write_bytes(_build_merged_model_catalog(binary))
    runtime["model_catalog_json"] = str(catalog)
    (workspace / ".codex/config.toml").write_text("\n".join(f"{k} = {toml_value(v)}" for k, v in runtime.items()) + "\n")
    prompt = f"""=== TASK FOLDER ===
{workspace / 'task_folder'}
Read TASK_INSTRUCTIONS.txt and the reference files in this task folder, then complete the task.
Only save final deliverables (not intermediate files) to {workspace / 'output'}.
Use only files in {workspace}, installed software, or online references. Explain conflicting information and justify your choices.
Use Docling MCP for document conversion and export_docling_document_to_markdown to read the converted document. If Docling fails or the extraction is unusable, use MinerU MCP as a fallback. MinerU uploads documents to its parsing service.
For online references use the search MCP and shell HTTP tools. For images, render documents/pages and call view_image. update_plan can help manage multi-step work.
You may install necessary Python packages locally under this workspace. Do not install into system directories. Keep outputs in the specified output directory.
The document-tools Python interpreter is {mcp / 'bin/python'} and includes pandas, openpyxl, python-docx, python-pptx, pdfplumber and matplotlib.
"""
    if prompt_builder is not None:
        prompt = prompt_builder(workspace, mcp)
    uid = 60000 + secrets.randbelow(5000) if os.geteuid() == 0 else os.getuid()
    for path in [workspace, *workspace.rglob("*")]:
        os.chown(path, uid, uid)
    env = safe_environment(workspace, token)
    command = [str(binary), "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
        "--enable", "unified_exec", "-C", str(workspace), "-m", CODEX_R2EACT_MODEL_SLUG,
        "--output-last-message", str(workspace / "final_answer.txt"), "-"]
    argv = sandbox_command(workspace, uid, binary, mcp, command)
    atomic_json(execution / "metadata.json", {"started_at": utcnow(), "workspace": str(workspace), "task_id": task.task_id,
        "config": config.to_json(), "compressor_checkpoint": str(gateway_config.checkpoint) if config.compressor_enabled else None,
        "source_sha256": task.source_sha256, "sandbox": os.environ.get("R2EACT_DOCUMENT_SANDBOX", "landlock"), "reasoning_effort": "xhigh", "document_tools": ["docling", "mineru"],
        "prompt_transport": "stdin_utf8_v1", "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "controller_binary": str(binary), "document_runtime": str(mcp),
        "controller_path_policy": "separate_from_document_generation_v1"})
    started = time.monotonic()
    process = None
    status = "errored"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    server.start()
    try:
        with (execution / "trajectory.jsonl").open("wb") as stdout, (execution / "codex.stderr.log").open("wb") as stderr:
            process = start_codex(argv, prompt, env=env, cwd=workspace, stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=config.timeout_seconds)
                status = "completed" if code == 0 else "errored"
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
    except KeyboardInterrupt:
        status, code = "interrupted", None
    finally:
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        server.close()
        for s, handler in previous.items():
            signal.signal(s, handler)
    files = collect_output(workspace, execution)
    final = workspace / "final_answer.txt"
    if final.is_file() and not final.is_symlink():
        shutil.copyfile(final, execution / "final_answer.txt")
    result = {"task_id": task.task_id, "status": status, "exit_code": code,
        "elapsed_seconds": time.monotonic() - started, "finished_at": utcnow(),
        "deliverables": len(files), "workspace": str(workspace), "judge_status": "pending", "reward": None}
    if isinstance(code, int) and code < 0:
        result["termination_signal"] = -code
        result["error"] = f"Codex process terminated by signal {-code}; see the last tool call and controller logs"
    atomic_json(execution / "result.json", result)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    if a.preflight:
        preflight()
        print("JobBench runner preflight passed", flush=True)
        return
    from .maintenance import prepare_request, prepare_memory_request
    prepare_request(a.request)
    request = prepare_memory_request(a.request)
    task = dict(request["task"])
    task["task_folder"] = Path(task["task_folder"])
    run_jobbench_task(JobBenchConfig.from_json(request["config"]), JobBenchTask(**task), Path(request["execution"]), sglang_url=request["sglang_url"])

if __name__ == "__main__":
    main()
