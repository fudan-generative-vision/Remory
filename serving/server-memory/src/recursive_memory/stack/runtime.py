"""Deployment controller. Run under the installed Python 3.12 environment."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import importlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from . import gpu
from .deploy import (
    DATASET,
    DATASET_REF,
    POLICY,
    TOOLS,
    atomic,
    canonical,
    environment,
    read,
    run,
    sha,
)


def endpoint(state):
    return state.get("sglang_url") or f"http://127.0.0.1:{state['server']['port']}"


def external_serving_state(state, url):
    """Select an explicit service without changing the installed model recipe."""
    url = url.rstrip("/")
    parsed = urlsplit(url)
    route = r"(?:/groups/[A-Za-z0-9_.:-]{1,128})?(?:/backends/[A-Za-z0-9_.:-]{1,128})?"
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not re.fullmatch(route, parsed.path)):
        raise ValueError("--sglang-url requires a credential-free HTTP(S) origin or shared gateway route")
    return {**state, "sglang_url": url}


def residual_config(state):
    from shared_runtime.config import HarnessConfig

    return HarnessConfig(
        harness_mode="native_compact",
        native_compaction_text_summary=True,
        residual_compaction=True,
        residual_checkpoint=state["checkpoint"],
        working_memory_turns=12,
        serving_eviction_policy=POLICY,
    )


def admit_server(state):
    from shared_runtime.residual_preflight import (
        assert_residual_server,
        get_server_info,
    )

    receipt = assert_residual_server(residual_config(state), endpoint(state))
    info = get_server_info(endpoint(state))
    if info.get("language_only") is True or info.get("enable_multimodal") is False:
        raise RuntimeError("the target must retain its vision module")
    return receipt


def tool_preflight():
    from shared_runtime.runner import preflight
    from terminal_bench_eval.codex_r2eact_gateway import (
        CODEX_R2EACT_WEB_SEARCH_ENDPOINT,
        _fetch_serper_search,
    )

    binary, documents = preflight()
    # Same transport as model-side searches; an IP-only proxy probe is not
    # sufficient. This consumes one Serper search credit, before any task claim.
    reply = _fetch_serper_search(
        api_key=os.environ["SERPER_API_KEY"],
        endpoint=CODEX_R2EACT_WEB_SEARCH_ENDPOINT,
        query="IANA example domains",
        num_results=1,
        timeout_seconds=30,
    )
    if not reply.get("organic"):
        raise RuntimeError(
            "Serper search preflight returned no results; no task claimed"
        )
    return {"codex": str(binary), "documents": str(documents), "search": "ok"}


def doctor(state):
    from automation_bench_eval.runtime import runtime_error
    from context_compression.qwen38_residual_serving_contract import (
        load_residual_serving_contract,
        verify_residual_target,
    )

    prefix = Path(state["prefix"])
    contract = load_residual_serving_contract(state["checkpoint"])
    verify_residual_target(contract, state["target"])
    for name, (_, _, checksum) in TOOLS.items():
        if sha(prefix / "bin" / name) != checksum:
            raise RuntimeError(f"{name} differs from the pinned executable")
    imports = [
        "terminal_bench_eval.codex_r2eact_agent",
        "automation_bench_eval.pool",
        "eval.terminal_bench.proot_environment",
        "shared_runtime.runner",
        "job_bench_eval.runner",
        "job_bench_eval.codex_judge",
        "browsecomp_bench_eval.codex_pool",
        "browsecomp_bench_eval.codex100_review",
        "context_compression.qwen38_residual_sglang_client",
    ]
    for module in imports:
        importlib.import_module(module)
    problem = runtime_error(prefix / "runs/automationbench")
    if problem:
        raise RuntimeError(problem)
    from .sandbox import check
    boundary = check()
    if not (prefix / "empty-corpus/wiki_corpus.jsonl").is_file():
        raise RuntimeError("required empty corpus placeholder is missing")
    report = {
        "status": "ok",
        "imports": imports,
        "source_sha256": state["source_sha256"],
        "harbor": importlib.metadata.version("harbor"),
        "checkpoint": state["checkpoint"],
        "harness_mode": "native_compact",
        "reasoning_effort": "xhigh",
        "working_memory_rounds": None,
        "accumulation_tokens": None,
        "physical_leaf_tokens": 1024,
        "credentials_present": {
            key: bool(os.environ.get(key))
            for key in ("SERPER_API_KEY", "MINERU_API_KEY")
        },
        "gpu_generation_tested": False,
        "claims_tasks": False,
        "document_sandbox": boundary,
    }
    atomic(prefix / "receipts/doctor.json", report)
    print(json.dumps(report, indent=2))


def server_command(state):
    from context_compression.sglang_compressor_hook import (
        GENERATION_DEGENERATION_GUARD_SERVER_ABI,
    )

    state = {
        **state,
        "server": gpu.resolve(
            state["server"], Path(state["prefix"]) / "server/bin/python"
        ),
    }
    hooks = [
        {
            "name": "qwen38_summary_residual_compressor",
            "target_modules": ["logits_processor"],
            "hook_factory": "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook",
            "config": {
                "checkpoint": state["checkpoint"],
                "residual_serving_abi": "qwen38_summary_residual_sglang_v1",
                "sync_token_ids_across_tp": True,
                "generation_degeneration_guard_server_abi": GENERATION_DEGENERATION_GUARD_SERVER_ABI,
            },
        }
    ]
    if os.environ.get("REMORY_SERVER_MEMORY") == "1":
        from context_compression.sglang_compressor_hook import GPU_MEMORY_HANDLE_POLICY
        hooks[0]["config"].update(gpu_memory_handle_policy=GPU_MEMORY_HANDLE_POLICY,
            gpu_memory_handle_ttl_seconds=3600,
            gpu_memory_handle_max_entries_per_tp_rank=256,
            gpu_memory_handle_max_bytes_per_tp_rank=int(os.environ.get("REMORY_MEMORY_BYTES", 2 << 30)))
    prefix = Path(state["prefix"])
    argv = [
        str(prefix / "server/bin/python"),
        "-m",
        "sglang.launch_server",
        "--model-path",
        state["target"],
        "--served-model-name",
        state.get("served_model_name", "r2eact-qwen38-27b-residual-compressor-step2320"),
        "--host",
        "127.0.0.1",
        "--port",
        str(state["server"]["port"]),
        "--tp-size",
        str(state["server"]["tp"]),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "flashinfer",
        "--context-length",
        str(state["server"].get("context_length", 262144)),
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mem-fraction-static",
        str(state["server"]["memory_fraction"]),
        "--max-running-requests",
        str(state["server"].get("max_running_requests", 1)),
        "--max-total-tokens",
        str(state["server"].get("max_total_tokens", 262144)),
        "--chunked-prefill-size",
        str(state["server"].get("chunked_prefill_size", -1)),
        "--random-seed",
        "42",
        "--enable-return-hidden-states",
        "--enable-custom-logit-processor",
        "--forward-hooks",
        json.dumps(hooks),
        "--disable-custom-all-reduce",
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--cuda-graph-max-bs",
        str(state["server"].get("max_running_requests", 1)),
        "--log-level",
        "info",
    ]
    if not state["server"].get("overlap_schedule", False):
        argv.append("--disable-overlap-schedule")
    if not state["server"].get("piecewise_cuda_graph", False):
        argv.append("--disable-piecewise-cuda-graph")
    env = environment(state)
    env.update(
        {
            "SYNC_TOKEN_IDS_ACROSS_TP": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            # A 179K-token unchunked prefill can exceed the generated health
            # probe's 20s window. Keep /health as readiness; SGLang's real warmup,
            # the watchdog and our stochastic smoke still check generation.
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
        }
    )
    if os.environ.get("REMORY_SERVER_MEMORY") == "1":
        import uuid
        env["SGLANG_SPECFORGE_MEMORY_SERVER_EPOCH"] = uuid.uuid4().hex
        env["REMORY_MEMORY_DIR"] = os.environ.get("REMORY_MEMORY_DIR", str(prefix / "server-memory"))
    devices = state["server"]["devices"]
    env.pop("CUDA_VISIBLE_DEVICE", None)
    if devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = devices
    if state["server"]["nccl_p2p_disable"] is not None:
        env["NCCL_P2P_DISABLE"] = state["server"]["nccl_p2p_disable"]
    if not state["server"]["cuda_graph"]:
        i = argv.index("--cuda-graph-max-bs")
        del argv[i : i + 2]
        argv.append("--disable-cuda-graph")
    print("[stack] resolved GPU profile: " + json.dumps(state["server"]), flush=True)
    env["PYTHONPATH"] = str(prefix / "sglang/python") + os.pathsep + env["PYTHONPATH"]
    return argv, env


def occupied(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def stop_owned(child):
    """Only a child created in a new session by this process; never read a PID file."""
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


@contextlib.contextmanager
def serving(state, *, start=False):
    if state.get("sglang_url"):
        if start:
            raise ValueError("--sglang-url cannot be combined with --start-server")
        admit_server(state)
        print(f"[stack] using external SGLang {endpoint(state)} (not owned)", flush=True)
        yield None
        return
    port = state["server"]["port"]
    if occupied(port):
        admit_server(state)  # Healthy but wrong checkpoint is an error, not a restart.
        print(
            f"[stack] reusing attested server {endpoint(state)} (not owned)", flush=True
        )
        yield None
        return
    if not start:
        raise RuntimeError(
            "SGLang is not listening; start `server` or pass --start-server"
        )
    lock_path = Path(f"/tmp/recursive-memory-stack-server-{os.getuid()}-{port}.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another launcher is starting this port; wait for it"
            ) from exc
        if occupied(port):
            raise RuntimeError("port became occupied; refusing to replace its owner")
        argv, env = server_command(state)
        log_path = (
            Path(state["prefix"])
            / "logs"
            / f"sglang-{int(time.time())}-{os.getpid()}.log"
        )
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(
                argv,
                env=env,
                cwd=state["workspace"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"[stack] starting owned SGLang; log: {log_path}", flush=True)
        try:
            deadline, next_log = time.monotonic() + 1800, 0.0
            while True:
                if child.poll() is not None:
                    raise RuntimeError(
                        f"SGLang exited with {child.returncode}; inspect {log_path}"
                    )
                if occupied(port):
                    try:
                        admit_server(state)
                        break
                    except Exception as exc:
                        last = f"{type(exc).__name__}: {exc}"
                else:
                    last = "waiting for listener"
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError(
                        f"SGLang startup timed out: {last}; log: {log_path}"
                    )
                if now >= next_log:
                    print(f"[stack] SGLang startup: {last}", flush=True)
                    next_log = now + 30
                time.sleep(2)
            print(f"[stack] SGLang admitted at {endpoint(state)}", flush=True)
            yield child
        finally:
            stop_owned(child)


def smoke(state):
    """Real encode -> recursive encode -> recovery; no benchmark task or score."""
    from context_compression.qwen38_residual_serving_contract import (
        load_residual_serving_contract,
    )
    from context_compression.qwen38_residual_sglang_client import (
        SummaryResidualSGLangClient,
    )
    from terminal_bench_eval.codex_r2eact_agent import _build_merged_model_catalog
    from transformers import AutoTokenizer

    prefix = Path(state["prefix"])
    # Runs the real Codex metadata handshake without account/API credentials.
    _build_merged_model_catalog(prefix / "bin/codex")
    admit_server(state)
    tokenizer = AutoTokenizer.from_pretrained(state["target"], local_files_only=True)
    contract = load_residual_serving_contract(state["checkpoint"])
    client = SummaryResidualSGLangClient(
        endpoint(state), request_timeout=180, eos_token_id=tokenizer.eos_token_id
    )
    client.attest_residual_server(state["checkpoint"])

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": "Remember that the package color is blue."},
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    history = encode("The package color is blue. " * 160)
    summary = encode("The package color is blue.")
    first = client.build_residual_memory(
        ids, history, summary, contract=contract["summary_contract"]
    )
    memory = first["memory_embeddings"]
    previous = (
        summary
        + [tokenizer.eos_token_id] * len(memory)
        + encode("The package is ready.")
    )
    second = client.build_residual_memory(
        ids,
        previous,
        summary,
        contract=contract["summary_contract"],
        source_memory=memory,
        source_memory_positions=list(
            range(len(ids) + len(summary), len(ids) + len(summary) + len(memory))
        ),
        source_memory_depths=first["receipt"]["slot_depths"],
    )
    summary_contract = contract["summary_contract"]
    base = (
        ids
        + summary_contract["student_summary_before_ids"]
        + summary
        + summary_contract["student_summary_after_ids"]
        + summary_contract["student_memory_before_ids"]
    )
    memory = second["memory_embeddings"]
    positions = list(range(len(base), len(base) + len(memory)))
    recovery = (
        base
        + [tokenizer.eos_token_id] * len(memory)
        + summary_contract["student_memory_after_ids"]
        + encode("What color is the package?")
    )
    generated = client.generate_with_memory(
        recovery,
        memory_positions=positions,
        memory_embeddings=memory,
        max_new_tokens=64,
        expected_hidden_size=5120,
        sampling_overrides={"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    )
    output = tokenizer.decode(list(generated.token_ids), skip_special_tokens=False)
    report = {
        "status": "ok",
        "first_encode": first["receipt"],
        "recursive_encode": second["receipt"],
        "output": output,
        "benchmark_score": None,
        "note": "synthetic transport/recursion smoke, not a quality or vision benchmark",
    }
    atomic(prefix / "receipts/smoke.json", report)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": output,
                "receipt": str(prefix / "receipts/smoke.json"),
            },
            indent=2,
        )
    )


def tb_profile(state, job, *, memory="residual", serving_concurrency=1):
    if memory not in {"residual", "none"}:
        raise ValueError("memory must be residual or none")
    profile = _tb_residual_profile(state, job)
    if memory == "none":
        profile["agents"][0]["kwargs"].update(
            residual_compaction=False, compressor_enabled=False,
            serving_concurrency=serving_concurrency,
        )
    return profile


def _tb_residual_profile(state, job):
    prefix = Path(state["prefix"])
    return {
        "job_name": job,
        "jobs_dir": str(prefix / "runs/tb21/jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "datasets": [{"name": DATASET, "ref": DATASET_REF}],
        "agents": [
            {
                "import_path": "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent",
                "model_name": "r2eact-qwen38-27b-compressor-r16-step1000",
                "override_timeout_sec": 10800,
                "include_logs": [
                    "codex.txt",
                    "sessions/**",
                    "memory.jsonl",
                    "codex-r2eact-*.json*",
                ],
                "kwargs": {
                    "specforge_root": state.get("tb_specforge_root", state["workspace"]),
                    "checkpoint": state["checkpoint"],
                    "corpus_dir": str(prefix / "empty-corpus"),
                    "sglang_url": endpoint(state),
                    "codex_binary": str(prefix / "bin/codex"),
                    "residual_compaction": True,
                    "native_compaction": True,
                    "native_compaction_text_summary": True,
                    "qwen_reasoning_effort": "xhigh",
                    "document_tools_enabled": True,
                    "working_memory_turns": 12,
                    "serving_eviction_policy": POLICY,
                    "vision_enabled": True,
                    "web_search_enabled": True,
                    "update_plan_enabled": True,
                },
            }
        ],
        "environment": {
            "import_path": "eval.terminal_bench.proot_environment:ProotEnvironment",
            "override_cpus": 32,
            "override_memory_mb": 49152,
            "kwargs": {"guest_utility_profile": "tb20-file-xxd-rg-v1"},
        },
    }


def tb_preflight(state, config_path, output, *, memory="residual", serving_concurrency=1):
    import yaml
    from eval.terminal_bench.proot_environment import ProotEnvironment
    from harbor.models.job.config import JobConfig
    from terminal_bench_eval.tb21_v050_preflight import _dataset_report

    config = JobConfig.model_validate(yaml.safe_load(config_path.read_text()))
    expected = JobConfig.model_validate(tb_profile(state, config.job_name,
        memory=memory, serving_concurrency=serving_concurrency))
    expected.jobs_dir = config.jobs_dir
    if config.model_dump(mode="python") != expected.model_dump(mode="python"):
        raise RuntimeError(
            "pool configuration differs from this installation; no task claimed"
        )
    try:
        ProotEnvironment.preflight()
    except SystemExit as exc:
        raise RuntimeError(f"PRoot preflight failed: {exc}") from exc
    from terminal_bench_eval.terminal_bench_pool import load_pool_spec

    if (config_path.parent / "pool.json").is_file():
        frozen = load_pool_spec(
            job_name=config_path.parent.name, jobs_root=config_path.parent.parent
        )
        dataset = {
            "task_count": len({c.task_name for c in frozen.cells}),
            "registry_queried": False,
            "source": DATASET,
        }
    else:
        dataset = asyncio.run(_dataset_report())
    report = {
        "status": "ok",
        "dataset": dataset,
        "server": admit_server(state),
        "tools": tool_preflight(),
    }
    if output:
        atomic(output, report)
    print(json.dumps(report, indent=2))


def evaluation_commands(state, args):
    prefix = Path(state["prefix"])
    memory = getattr(args, "memory", "residual")
    if memory not in {"residual", "none"}:
        raise ValueError("memory must be residual or none")
    variant = "residual-c2320" if memory == "residual" else "text-only"
    default_name = f"portable-{args.benchmark}-native-compact-{variant}-xhigh-{state['source_sha256'][:8]}"
    if args.benchmark == "browsecomp100":
        default_name += f"-seed{getattr(args, 'seed', 20260912)}"
    if args.task:
        default_name += "-subset-" + canonical(sorted(args.task))[:8]
    job = args.job or default_name
    if not job or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in job
    ):
        raise ValueError("job name must contain only letters, numbers, _, . and -")
    if args.benchmark in {"jobbench", "browsecomp100"}:
        from .research_benchmarks import commands

        prepare, worker = commands(state, args, job)
    elif args.benchmark == "automationbench":
        root = str(prefix / "runs/automationbench")
        prepare = [
            sys.executable,
            "-m",
            "automation_bench_eval.launch",
            "--runtime-root",
            root,
            "--job-name",
            job,
            "--mode",
            "native_compact",
            "--reasoning-effort",
            "xhigh",
            "--compressor",
            memory,
            "--checkpoint",
            state["checkpoint"],
            "--sglang-url",
            endpoint(state),
            "--prepare-only",
        ]
        for task in args.task:
            prepare.extend(["--task-id", task])
        worker = [
            sys.executable,
            "-m",
            "automation_bench_eval.pool",
            "worker",
            "--runtime-root",
            root,
            "--job-name",
            job,
            "--sglang-url",
            endpoint(state),
            "--idle-exit-seconds",
            "120",
        ]
        if args.once:
            worker.extend(["--max-tasks", "1"])
    else:
        path = prefix / "configs" / f"{job}.json"
        profile = tb_profile(state, job, memory=getattr(args, "memory", "residual"),
                             serving_concurrency=getattr(args, "serving_concurrency", getattr(args, "concurrency", 1)))
        if path.exists() and read(path) != profile:
            raise RuntimeError("immutable profile changed; choose a new job name")
        atomic(path, profile)
        root = str(prefix / "runs/tb21/jobs")
        prepare = [
            sys.executable,
            "-m",
            "terminal_bench_eval.terminal_bench_pool",
            "init",
            "--config",
            str(path),
            "--jobs-root",
            root,
            "--job-name",
            job,
            "--attempts",
            "1",
        ]
        for task in args.task:
            prepare.extend(["--task-name", task])
        if not args.task:
            prepare.append("--all")
        worker = [
            sys.executable,
            "-m",
            "terminal_bench_eval.terminal_bench_pool",
            "worker",
            "--jobs-root",
            root,
            "--job-name",
            job,
            "--sglang-url",
            endpoint(state),
            "--harbor-bin",
            str(prefix / "controller/bin/harbor"),
            "--preflight-module",
            "recursive_memory.stack.preflight",
        ]
        if args.once:
            worker.append("--once")
    if args.worker_id:
        worker.extend(["--worker-id", args.worker_id])
    return prepare, worker


def worker_commands(command, count, *, worker_id=None):
    if not 1 <= count <= 32:
        raise ValueError("concurrency must be 1..32")
    if count == 1:
        return [list(command)]
    base = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
    commands = []
    for slot in range(count):
        argv = list(command)
        label = f"{base}-slot-{slot}"
        if "--worker-id" in argv:
            argv[argv.index("--worker-id") + 1] = label
        else:
            argv.extend(["--worker-id", label])
        if "terminal_bench_eval.terminal_bench_pool" in argv:
            argv.extend(["--device-label", label])
        elif "automation_bench_eval.pool" in argv:
            argv.extend(["--concurrency", str(count)])
        commands.append(argv)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("server")
    sub.add_parser("check-sandbox")
    sub.add_parser("check-kernels")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--start-server", action="store_true")
    runner = sub.add_parser("run")
    runner.add_argument("benchmark", choices=("automationbench", "tb21", "jobbench", "browsecomp100"))
    runner.add_argument("--offline-dataset", action="store_true")
    runner.add_argument("--seed", type=int, default=20260912, help="BrowseComp100 sampling seed")
    runner.add_argument("--job")
    runner.add_argument("--task", action="append", default=[])
    runner.add_argument("--worker-id")
    runner.add_argument("--once", action="store_true")
    runner.add_argument("--prepare-only", action="store_true")
    runner.add_argument("--start-server", action="store_true")
    runner.add_argument("--sglang-url", help="existing shared service; workers never start or stop its models")
    runner.add_argument("--memory", choices=("residual", "none"), default="residual",
                        help="native compact with residual soft memory, or paired text-summary-only ablation")
    runner.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="task slots sharing the selected service (default: 2)",
    )
    monitor = sub.add_parser("monitor")
    monitor.add_argument("benchmark", choices=("all", "automationbench", "tb21", "jobbench", "browsecomp100"))
    monitor.add_argument("--port", type=int, default=8507)
    judge = sub.add_parser("judge", help="Controller-only post-hoc judging; never starts a generation worker")
    judge.add_argument("benchmark", choices=("jobbench", "browsecomp100"))
    judge.add_argument("--job", required=True)
    judge.add_argument("--codex-binary", type=Path, required=True,
                       help="separately installed/logged-in CLI with --ignore-user-config")
    judge.add_argument("--model", default="gpt-5.6-sol")
    judge.add_argument("--effort", default="medium")
    judge.add_argument("--watch", action="store_true")
    judge.add_argument("--sglang-url", help="shared service for BrowseComp's local Qwen judging")
    verify = sub.add_parser(
        "verify", help="Resume pending TB verification without calling the model"
    )
    verify.add_argument("benchmark", choices=("tb21",))
    verify.add_argument("--job", required=True)
    verify.add_argument("--once", action="store_true")
    chat = sub.add_parser(
        "chat", help="Interactive Codex using the TB checkpoint/policy/tools"
    )
    chat.add_argument("--workspace", type=Path, required=True)
    chat.add_argument(
        "--prompt",
        help="Optional first user message; subsequent turns use the terminal",
    )
    chat.add_argument("--start-server", action="store_true")
    chat.add_argument(
        "--unsafe",
        action="store_true",
        help="Disable host command approvals/sandbox (trusted workspaces only)",
    )
    args = parser.parse_args()
    if args.command == "run" and not 1 <= args.concurrency <= 32:
        parser.error("--concurrency must be 1..32")
    if getattr(args, "sglang_url", None) and getattr(args, "start_server", False):
        parser.error("--sglang-url cannot be combined with --start-server")

    # Ensure owned server cleanup also runs for scheduler TERM/HUP, not only
    # interactive Ctrl+C. Workers receive a graceful signal below.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    state = read(os.environ["R2EACT_STACK_STATE"])
    if getattr(args, "sglang_url", None):
        try:
            state = external_serving_state(state, args.sglang_url)
        except ValueError as exc:
            parser.error(str(exc))
        os.environ["R2EACT_SHARED_SGLANG_URL"] = endpoint(state)
    else:
        os.environ.pop("R2EACT_SHARED_SGLANG_URL", None)
    if args.command == "doctor":
        doctor(state)
    elif args.command == "server":
        with serving(state, start=True) as child:
            if child is not None:
                raise SystemExit(child.wait())
    elif args.command == "smoke":
        from .validation import documents, kernels
        print("[stack] checking stochastic GPU kernels", flush=True)
        kernels(state)
        print("[stack] checking document sandbox and MCP conversion", flush=True)
        documents(state)
        with serving(state, start=args.start_server):
            smoke(state)
    elif args.command == "check-sandbox":
        from .sandbox import check
        print(json.dumps(check(), indent=2))
    elif args.command == "check-kernels":
        from .validation import kernels
        print(json.dumps(kernels(state), indent=2))
    elif args.command == "chat":
        from .chat import interactive

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            parser.error("chat requires an interactive terminal (tmux is supported)")
        tool_preflight()
        with serving(state, start=args.start_server):
            raise SystemExit(interactive(state, args))
    elif args.command == "run":
        os.environ["R2EACT_TASK_MEMORY"] = args.memory
        os.environ["R2EACT_TASK_CONCURRENCY"] = str(args.concurrency)
        prepare, worker = evaluation_commands(state, args)
        run(prepare)
        if not args.prepare_only:
            tool_preflight()
            # Runtime topology/capacity does not alter task selection or pool
            # contracts. A foreign server is attested, never reconfigured.
            state = {
                **state,
                "server": {
                    **state["server"],
                    "max_running_requests": max(
                        state["server"].get("max_running_requests", 1), args.concurrency
                    ),
                },
            }
            with serving(state, start=args.start_server):
                from terminal_bench_eval.parallel_workers import attest, supervise

                if args.concurrency > 1:
                    attest(endpoint(state), args.concurrency,
                           mode="discrete" if args.memory == "none" else "residual")
                commands = worker_commands(
                    worker, args.concurrency, worker_id=args.worker_id
                )
                print(
                    f"[stack] {len(commands)} task slot(s) sharing {endpoint(state)}",
                    flush=True,
                )
                code = supervise(commands)
                if code:
                    raise SystemExit(code)
    elif args.command == "verify":
        command = [
            sys.executable,
            "-m",
            "terminal_bench_eval.verifier_queue",
            "--jobs-root",
            str(Path(state["prefix"]) / "runs/tb21/jobs"),
            "--job-name",
            args.job,
        ]
        if args.once:
            command.append("--once")
        run(command)
    elif args.command == "judge":
        from .research_benchmarks import judge_command

        command, env = judge_command(state, args)
        run(command, env=env)
    else:
        if not 1024 <= args.port <= 65535:
            parser.error("monitor port must be in [1024, 65535]")
        file = {
            "all": "shared_runtime/benchmark_monitor.py",
            "automationbench": "automation_bench_eval/monitor_streamlit.py",
            "tb21": "terminal_bench_eval/terminal_bench_monitor_streamlit.py",
            "jobbench": "job_bench_eval/monitor_streamlit.py",
            "browsecomp100": "browsecomp_bench_eval/codex_monitor_streamlit.py",
        }[args.benchmark]
        os.environ["R2EACT_MONITOR_PREFIX"] = state["prefix"]
        argv = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(state["workspace"]) / file),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
        os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
