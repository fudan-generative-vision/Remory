"""A bounded, resumable AutomationBench pilot using the existing Codex gateway.

No pool claims, VPN changes, or SGLang startup/restart. A busy endpoint is an
explicit preflight failure; it is never commandeered from a running benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from shared_runtime.storage import atomic_json, read_json, utcnow, lock
from shared_runtime.discrete_concurrency import summary_concurrency, assert_discrete_concurrency
from .bridge import OfficialSession, MCPServer
from .runtime import OFFICIAL_REVISION, ensure_runtime, official_python, interpreter_environment
from .tool_runtime import JOBBENCH_TOOLS, LEGACY_TOOLS, tool_profile, prepare_tools, mcp_servers, instructions

PROJECT = Path(__file__).resolve().parents[1]
REVISION = OFFICIAL_REVISION
DEFAULT_RUNTIME = PROJECT / "eval/automation_bench"
DEFAULT_RESIDUAL = "/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/outputs/qwen38-compact-residual-rkl-c4500-stream-20260908-v3-step2320-serving"


class EndpointBusyError(RuntimeError):
    """Pre-start contention is not a scored attempt or a model error."""


def toml_value(value):
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(k) + " = " + toml_value(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if value is None:
        raise TypeError("TOML has no null value")
    return json.dumps(value, ensure_ascii=False)


def check_source(root):
    revision = subprocess.check_output(["git", "-C", str(root / "source"), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root / "source"), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise RuntimeError("AutomationBench source must match the clean pinned revision; run setup_runtime.sh")
    ensure_runtime(root)


def task_names(root, domain):
    env = interpreter_environment()
    output = subprocess.check_output([str(official_python(root)),
        str(Path(__file__).with_name("official_session.py")), "--list", "--domain", domain],
        env=env, cwd="/tmp", timeout=180, text=True)
    return json.loads(output)


def local_codex():
    from terminal_bench_eval.codex_r2eact_agent import PINNED_CODEX_BINARY, PINNED_CODEX_SHA256
    binary = Path(PINNED_CODEX_BINARY)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != PINNED_CODEX_SHA256:
        raise RuntimeError("Codex binary does not match the pinned TB/JobBench version")
    # Landlock execution on this cluster requires a node-local binary.
    cache = Path(f"/tmp/automationbench-runtime-{os.getuid()}-{PINNED_CODEX_SHA256[:16]}")
    cache.mkdir(mode=0o755, exist_ok=True)
    with lock(cache / "setup.lock"):
        dest = cache / "codex"
        if not dest.exists():
            staging = cache / f"codex-{os.getpid()}.tmp"
            shutil.copyfile(binary, staging)
            staging.chmod(0o755)
            staging.replace(dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != PINNED_CODEX_SHA256:
            raise RuntimeError("Node-local Codex checksum mismatch")
    return dest


def ensure_endpoint_available(args):
    from shared_runtime.residual_preflight import get_server_info
    from terminal_bench_eval.codex_r2eact_gateway import DEFAULT_PROCESS_LOCK
    concurrency = summary_concurrency(args)
    # Explicit discrete/residual slots share the lease only after ABI admission.
    with DEFAULT_PROCESS_LOCK.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EndpointBusyError("SGLang is owned by another evaluation (JobBench/TB). No AutomationBench task started. Release that endpoint or use a free worker.") from None
    info = get_server_info(args.sglang_url)
    if concurrency > 1:
        if args.compressor == "residual":
            from shared_runtime.residual_preflight import assert_residual_concurrency
            assert_residual_concurrency(info, concurrency)
        else:
            assert_discrete_concurrency(info, concurrency)
    if args.compressor in {"residual", "none"}:
        from shared_runtime.config import HarnessConfig
        from shared_runtime.residual_preflight import assert_paired_server, assert_residual_concurrency
        native = args.mode == "native_compact"
        receipt = assert_paired_server(HarnessConfig(residual_compaction=args.compressor == "residual",
            compressor_enabled=args.compressor == "residual", residual_checkpoint=args.checkpoint,
            harness_mode=args.mode, native_compaction_text_summary=native,
            working_memory_turns=12, serving_eviction_policy=("checkpoint_1024_v1" if native
                else "qwen38_serving_8192_accumulation_canary_v1")), args.sglang_url)
        if concurrency > 1:
            # assert_residual_server returns an attestation receipt, not the
            # raw /get_server_info payload consumed by the task runner.
            assert_residual_concurrency({
                "request_capabilities": [receipt.get("concurrency_capability")],
                "max_running_requests": receipt.get("max_running_requests"),
            }, concurrency)
    return info


def limited_gateway_class():
    from terminal_bench_eval.codex_r2eact_gateway import CodexResponsesGateway, GatewayRequestError

    class PilotGateway(CodexResponsesGateway):
        def __init__(self, *args, max_steps=0, **kwargs):
            super().__init__(*args, **kwargs)
            if type(max_steps) is not int or max_steps < 0:
                raise ValueError("max_steps must be a nonnegative integer (0 means unlimited)")
            self.max_steps = max_steps
            self.seen_requests = set()
            self.step_limit_reached = False

        def handle(self, request):
            if self.max_steps == 0:
                # The base gateway owns synchronization and replay handling.
                # Unlimited runs do not need an ever-growing budget digest set.
                return super().handle(request)
            compact = self._is_native_compaction_request(request)
            digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            with self._lock:
                if not compact and digest not in self.seen_requests:
                    if len(self.seen_requests) >= self.max_steps:
                        self.step_limit_reached = True
                        raise GatewayRequestError(f"AutomationBench {self.max_steps}-step model response budget exhausted")
                    self.seen_requests.add(digest)
                return super().handle(request)
    return PilotGateway


def run_task(args, task_id, execution):
    from .reasoning_policy import task_identity
    official_task_id, identity = task_identity(args, task_id)
    from shared_runtime.runner import start_codex, DEFAULT_SPECFORGE
    from terminal_bench_eval.codex_r2eact_agent import (
        CODEX_R2EACT_MODEL_SLUG, _build_codex_runtime_config, _build_merged_model_catalog)
    from terminal_bench_eval.codex_r2eact_gateway import (
        R2EActGatewayConfig, LiveR2EActBackend, CodexGatewayServer)

    profile = tool_profile(args)
    binary, document_venv = prepare_tools(profile)
    full_tools = profile == JOBBENCH_TOOLS
    native = args.mode == "native_compact"
    info = ensure_endpoint_available(args)
    # The pool reserves a fresh execution and writes its request before spawning
    # us. Never reuse any directory containing an earlier task's artifacts.
    execution.mkdir(parents=True, mode=0o700, exist_ok=True)
    if any(p.name not in {"request.json", "runner.log"} for p in execution.iterdir()):
        raise RuntimeError(f"Execution already contains task artifacts: {execution}")
    with contextlib.ExitStack() as cleanup:
        session = OfficialSession(args.runtime_root, args.domain, official_task_id, execution)
        cleanup.callback(session.close)
        expected = getattr(args, "task_contract_sha256", None)
        if expected and session.public["task_contract_sha256"] != expected:
            raise RuntimeError("Official task differs from the pool's pinned contract")
        workspace = Path(tempfile.mkdtemp(prefix="automationbench-task-"))
        for name in ("tmp", ".codex"):
            (workspace / name).mkdir()
        bearer = "sk-automationbench-" + secrets.token_hex(24)
        mcp = MCPServer(session, bearer)
        mcp.start()
        cleanup.callback(mcp.close)
        config = R2EActGatewayConfig(
            specforge_root=Path(os.environ.get("SPECFORGE_ROOT", str(DEFAULT_SPECFORGE))),
            corpus_dir=Path(os.environ.get("R2EACT_CORPUS_DIR", str(DEFAULT_SPECFORGE / "data/wiki-18-corpus"))),
            checkpoint=Path(args.checkpoint),
            compressor_enabled=args.compressor != "none", residual_compaction=args.compressor == "residual",
            sglang_url=args.sglang_url, working_memory_turns=12,
            serving_eviction_policy=("checkpoint_1024_v1" if native else "qwen38_serving_8192_accumulation_canary_v1"),
            native_compaction=native, native_compaction_text_summary=native,
            qwen_reasoning_effort=identity.get("reasoning_effort"),
            memory_guest_path=str(workspace / "memory.jsonl"),
            vision_enabled=full_tools, web_search_enabled=full_tools, update_plan_enabled=True)
        from shared_runtime.residual_backend import ResidualBackend
        trace = execution / "gateway"
        trace.mkdir(mode=0o700)
        (trace / "memory.jsonl").touch(mode=0o600)
        backend = ResidualBackend(config, trace, memory_mirror_path=workspace / "memory.jsonl")
        if args.compressor in {"residual", "none"}:
            backend.serving_concurrency = summary_concurrency(args)
        if args.compressor in {"residual", "none"}:
            from .prompt_view import initial_prompt_observer
            backend.initial_prompt_observer = initial_prompt_observer(execution,
                model_path=info.get("model_path"), reasoning_effort=identity.get("reasoning_effort"))
        gateway = limited_gateway_class()(backend, trace, native_compaction=native,
            native_compaction_text_summary=native, working_memory_turns=12,
            vision_enabled=full_tools, web_search_enabled=full_tools, update_plan_enabled=True,
            trusted_mcp_aliases=({"automationbench": "automationbench", "docling": "docling", "mineru": "mineru"}
                if full_tools else {"automationbench": "automationbench"}), max_steps=args.max_steps)
        server = CodexGatewayServer(gateway, bearer_token=bearer)
        server.start()
        gateway_closed = False
        def close_gateway():
            nonlocal gateway_closed
            if not gateway_closed:
                server.close()
                gateway_closed = True
        cleanup.callback(close_gateway)
        runtime = _build_codex_runtime_config(server.base_url, native_compaction=native,
            web_search_enabled=full_tools, mcp_server_url=server.mcp_url if full_tools else None,
            trusted_mcp_servers=mcp_servers(mcp.url, document_venv), server_info=info)
        official_prompt = session.public["prompt"]
        runtime["developer_instructions"] = "\n\n".join(
            m["content"] for m in official_prompt if m["role"] == "system") + (
            "\nThe AutomationBench MCP api_search/api_fetch/base64_encode tools operate a local simulated business environment. "
            "Use those tools for all business data and actions; shell/curl requests to real SaaS URLs do not affect this environment. "
            "api_search searches simulated API documentation, not the public web. Do not access real customer services. "
            "Shell is available for local scratch calculations and memory.jsonl retrieval; update_plan is available for planning."
            + instructions(document_venv))
        catalog = workspace / ".codex/model-catalog.json"
        catalog.write_bytes(_build_merged_model_catalog(binary))
        runtime["model_catalog_json"] = str(catalog)
        (workspace / ".codex/config.toml").write_text("\n".join(
            f"{k} = {toml_value(v)}" for k, v in runtime.items()) + "\n")
        prompt = "\n\n".join(m["content"] for m in official_prompt if m["role"] == "user")
        uid = 60000 + secrets.randbelow(5000) if os.geteuid() == 0 else os.getuid()
        for path in (workspace, *workspace.rglob("*")):
            os.chown(path, uid, uid)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(workspace),
               "CODEX_HOME": str(workspace / ".codex"), "TMPDIR": str(workspace / "tmp"),
               "OPENAI_API_KEY": bearer, "NO_PROXY": "127.0.0.1,localhost,::1",
               "CUDA_VISIBLE_DEVICES": "", "GIT_TERMINAL_PROMPT": "0"}
        if full_tools:
            from shared_runtime.runner import safe_environment
            env = safe_environment(workspace, bearer)
        command = [str(binary), "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
                   "--enable", "unified_exec", "-C", str(workspace), "-m", CODEX_R2EACT_MODEL_SLUG,
                   "--output-last-message", str(workspace / "final_answer.txt"), "-"]
        argv = ["/usr/bin/python3", str(PROJECT / "shared_runtime/sandbox.py"),
                "--workspace", str(workspace), "--uid", str(uid)]
        for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", str(binary)):
            argv += ["--read", path]
        argv += ["--", *command]
        if full_tools:
            from shared_runtime.runner import sandbox_command
            argv = sandbox_command(workspace, uid, binary, document_venv, command)
        atomic_json(execution / "metadata.json", {"task_id": task_id, "domain": args.domain, **identity,
            "official_commit": REVISION, "task_contract_sha256": session.public["task_contract_sha256"],
            "harness": ("Codex CLI + native compact (summary only)" if args.compressor == "none"
                else "Codex CLI + native compact (summary + residual soft memory)" if native and args.compressor == "residual"
                else "Codex CLI + r²eAct"),
            "mode": args.mode, "compressor": args.compressor,
            "concurrency": summary_concurrency(args),
            "checkpoint": str(config.checkpoint) if config.checkpoint is not None else None,
            "max_steps": args.max_steps, "timeout_seconds": args.timeout_seconds,
            "timeout_policy": "local_wallclock_safety_guard_not_official",
            "working_memory_turns": None if native else 12,
            "serving_eviction_policy": config.serving_eviction_policy, "toolset": "api",
            "tool_profile": profile,
            **({"initial_prompt_capture": "first_decode_system_user_input_v1"}
               if args.compressor in {"residual", "none"} else {}),
            "api_compatibility": session.public.get("api_compatibility", "unmodified_official_api"),
            "extra_tools": (["exec_command", "write_stdin", "update_plan", "view_image", "web_search",
                "docling", "mineru"] if full_tools else ["exec_command", "write_stdin", "update_plan"]),
            "vision_enabled": full_tools, "web_search_enabled": full_tools,
            "document_tools": ["docling", "mineru"] if full_tools else [],
            "workspace": str(workspace),
            "started_at": utcnow(), "model_path": info.get("model_path"),
            "max_total_num_tokens": info.get("max_total_num_tokens")})
        started = time.monotonic()
        status, code, error = "errored", None, None
        process = None
        def interrupted(signum, _frame):
            raise InterruptedError(f"Worker interrupted by signal {signum}")
        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (execution / "codex-events.jsonl").open("w") as stdout, (execution / "codex-stderr.log").open("w") as stderr:
                process = start_codex(argv, prompt, env=env, cwd=workspace, stdout=stdout, stderr=stderr)
                code = process.wait(timeout=args.timeout_seconds)
            status = "step_limit" if gateway.step_limit_reached else "completed" if code == 0 else "errored"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except (InterruptedError, KeyboardInterrupt) as exc:
            status, error = "interrupted", str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
            if process is not None and process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # Join in-flight model/tool work before snapshotting and judging.
            close_gateway()
            mcp.close()
        final = workspace / "final_answer.txt"
        if final.is_file() and not final.is_symlink():
            shutil.copyfile(final, execution / "final_answer.txt")
        try:
            verification = session.request("evaluate")
        except Exception as exc:
            verification = {}
            status, error = "errored", f"Official verifier failed: {type(exc).__name__}: {exc}"
        reward = verification.get("reward") if status in {"completed", "step_limit", "timeout"} else None
        from terminal_bench_eval.codex_r2eact_artifacts import GATEWAY_RECEIPTS_SUMMARY_FILENAME, GATEWAY_EVENTS_FILENAME
        usage_path = trace / GATEWAY_RECEIPTS_SUMMARY_FILENAME
        usage = read_json(usage_path) if usage_path.exists() else {}
        if status == "errored" and not error:
            error = f"Codex exited with status {code}; see codex-events.jsonl and gateway diagnostics"
            events_path = trace / GATEWAY_EVENTS_FILENAME
            if events_path.exists():
                for line in events_path.open(encoding="utf-8"):
                    with contextlib.suppress(ValueError):
                        event = json.loads(line)
                        if event.get("event") == "gateway_request_failed":
                            error = f"{event.get('error_type')}: {event.get('error_message')}"
        result = {"task_id": task_id, "domain": args.domain, **identity, "status": status, "exit_code": code,
            "reward": reward, "partial_credit": verification.get("partial_credit"),
            "included_in_benchmark_score": args.domain != "simple" and reward is not None,
            "error": error, "model_responses": len(gateway.seen_requests),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "tool_calls": verification.get("tool_calls"), "elapsed_seconds": time.monotonic() - started,
            "finished_at": utcnow(), "workspace": str(workspace)}
        atomic_json(execution / "result.json", result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    p.add_argument("--job-name", default="qwen38-automationbench-public-pilot-v1")
    p.add_argument("--domain", choices=["sales", "marketing", "operations", "support", "finance", "hr", "simple"], default="sales")
    p.add_argument("--task-id", action="append")
    p.add_argument("--num-examples", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="Model-response limit; 0 means unlimited (default)")
    p.add_argument("--timeout-seconds", type=float, default=3600)
    p.add_argument("--mode", choices=["native_compact"], default="native_compact")
    p.add_argument("--compressor", choices=["residual", "none"], default="residual")
    p.add_argument("--reasoning-effort", choices=["xhigh"], default="xhigh")
    p.add_argument("--tool-profile", choices=[JOBBENCH_TOOLS], default=JOBBENCH_TOOLS)
    p.add_argument("--checkpoint", help="Explicit serving export; otherwise pin the current experiment's default")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    args.serving_eviction_policy = "checkpoint_1024_v1"
    if args.num_examples < 1 or args.max_steps < 0 or not 0 < args.timeout_seconds < float("inf"):
        p.error("example count and timeout must be positive and finite; max-steps must be >= 0")
    if args.compressor == "none" and args.mode != "native_compact":
        p.error("--compressor none requires --mode native_compact")
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_RESIDUAL  # text-only still pins summary metadata
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).resolve())
    if not args.job_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.job_name):
        p.error("job-name may only contain letters, digits, - and _")
    args.runtime_root = args.runtime_root.resolve()
    check_source(args.runtime_root)
    names = task_names(args.runtime_root, args.domain)
    if args.list:
        print(json.dumps(names, ensure_ascii=False, indent=2))
        return
    selected = args.task_id or names[:args.num_examples]
    if len(set(selected)) != len(selected) or any(t not in names for t in selected):
        p.error("task-id must be unique exact names from --list")
    job = args.runtime_root / "jobs" / args.job_name
    job.mkdir(parents=True, exist_ok=True, mode=0o700)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"prepare_only", "list", "sglang_url", "num_examples", "task_id"}}
    contract.update({"task_ids": selected, "official_commit": REVISION})
    with lock(job / "run.lock"):
        if (job / "config.json").exists() and read_json(job / "config.json") != contract:
            raise RuntimeError("Pilot contract changed; use a new --job-name (old results preserved)")
        atomic_json(job / "config.json", contract)
        print(json.dumps({"job": str(job), "tasks": selected, "prepared": True}, ensure_ascii=False), flush=True)
        if args.prepare_only:
            return
        for task_id in selected:
            execution = job / task_id
            if (execution / "result.json").exists():
                print(json.dumps(read_json(execution / "result.json"), ensure_ascii=False), flush=True)
                continue
            if execution.exists():
                raise RuntimeError(f"Incomplete prior execution preserved at {execution}; use a new --job-name")
            result = run_task(args, task_id, execution)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] in {"errored", "interrupted"}:
                break


if __name__ == "__main__":
    main()
