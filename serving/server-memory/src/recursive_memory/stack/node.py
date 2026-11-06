"""Independent TP1 replicas: persistent model services or a pinned TB2.1 queue.

plan/check do not start servers, initialize queues, claim tasks, or consume API
credits. serve starts only models; start also starts task workers.
No installed source is edited.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

RECORD = Path(__file__).resolve().parent
STATE_FILE = Path(os.environ.get("R2EACT_STACK_STATE", "installed.json"))
TOOL_KEYS = ("SERPER_API_KEY", "MINERU_API_KEY")


def load_tool_env(path):
    from .deploy import load_tool_env as load

    load(path)


def credentials_report():
    present = {key: bool(os.environ.get(key, "").strip()) for key in TOOL_KEYS}
    return {
        "status": "ok" if all(present.values()) else "blocked",
        "present": present,
        "missing": [key for key, exists in present.items() if not exists],
        "api_validity_tested": False,
        "starts_services": False,
        "claims_tasks": False,
    }


def require_credentials():
    missing = credentials_report()["missing"]
    if missing:
        raise SystemExit(
            "Missing tool credentials: "
            + ", ".join(missing)
            + ". Export them in this job, or pass --env-file /absolute/path/tools.env. "
            "Verify with: stack.sh node --prefix /path/to/stack check-env --env-file /absolute/path/tools.env"
        )


def sandbox_report(state):
    from .sandbox import check

    try:
        return {
            "status": "ok",
            "boundary": check(),
            "starts_services": False,
            "claims_tasks": False,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {
            "status": "blocked",
            "errors": [str(exc)],
            "starts_services": False,
            "claims_tasks": False,
        }


def node_environment(state):
    from .deploy import environment

    return environment(state)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(state):
    from . import deploy, gpu, runtime

    return deploy, gpu, runtime


def gpu_indices(value):
    if value == "auto":
        return "auto"
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise argparse.ArgumentTypeError("use GPU ordinals such as 0,1,2,3,4,5,6,7")
    values = [int(x) for x in value.split(",")]
    if len(values) != len(set(values)) or any(x > 7 for x in values):
        raise argparse.ArgumentTypeError(
            "select distinct local GPU ordinals from 0 through 7"
        )
    return values


def build_plan(
    state,
    *,
    node_id,
    gpus,
    slots=2,
    context=262144,
    token_pool=524288,
    port_base=30000,
    job=None,
    resource_policy="shared",
    deployment=None,
    cuda_graph=True,
    memory_fraction=0.70,
    memory="residual",
    serve_only=False,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", node_id):
        raise ValueError("node-id may contain letters, digits, _, . and -")
    if type(slots) is not int or slots not in (1, 2, 4) or context not in (65536, 131072, 262144):
        raise ValueError("supported slots=1/2/4; context=65536/131072/262144")
    if not context <= token_pool <= 1048576:
        raise ValueError("token pool must be between context length and 1048576")
    if not 1024 <= port_base <= 65528:
        raise ValueError("port-base must be between 1024 and 65528")
    if (
        not gpus
        or any(type(x) is not int or not 0 <= x <= 7 for x in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError(
            "select one to eight distinct visible GPU ordinals from 0 through 7"
        )
    if resource_policy not in ("shared", "budgeted"):
        raise ValueError("resource-policy must be shared or budgeted")
    if type(cuda_graph) is not bool or not 0.1 <= memory_fraction <= 0.9:
        raise ValueError(
            "cuda_graph must be boolean and memory_fraction must be between 0.1 and 0.9"
        )
    if memory not in {"residual", "none"}:
        raise ValueError("memory must be residual or none")
    job = job or ("shared-sglang" if serve_only else
                  "tb21-native-residual-c2320-xhigh" if memory == "residual"
                  else "tb21-native-text-only-xhigh")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
        raise ValueError("invalid job name")
    deployment = (
        deployment
        or f"{job}-c{context}-p{token_pool}-s{slots}-cg{int(cuda_graph)}-m{memory_fraction:g}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", deployment):
        raise ValueError("invalid deployment name")
    deploy, _, runtime = bootstrap(state)
    profile = runtime.tb_profile(state, job, memory=memory, serving_concurrency=slots)
    task_cpus = profile["environment"]["override_cpus"]
    task_ram_gib = profile["environment"]["override_memory_mb"] / 1024
    contract = {
        "schema": "recursive_memory_node_pool_v1",
        "job": job,
        "deployment": deployment,
        "prefix": state["prefix"],
        "resource_policy": resource_policy,
        "source_sha256": state["source_sha256"],
        "base_state_sha256": digest(state),
        "profile_sha256": digest(profile),
        "memory": memory,
        "slots_per_replica": slots,
        "server": {
            "tp": 1,
            "context_length": context,
            "max_total_tokens": token_pool,
            "memory_fraction": memory_fraction,
            "max_running_requests": slots,
            "cuda_graph": cuda_graph,
        },
        "launcher_sha256": file_digest(__file__),
        "preflight_sha256": file_digest(RECORD / "replica_preflight.py"),
        "sandbox_backend": state.get("document_sandbox", {"backend": "landlock"})[
            "backend"
        ],
    }
    if serve_only:
        contract["serve_only"] = True
    replicas = []
    for index in gpus:
        port = port_base + index
        replicas.append(
            {
                "gpu_index": index,
                "port": port,
                "endpoint": f"http://127.0.0.1:{port}",
                "worker_ids": [
                    f"{node_id}-g{index}-slot-{slot}" for slot in range(slots)
                ],
            }
        )
    return {
        "contract": contract,
        "node_id": node_id,
        "replicas": replicas,
        "queue": None if serve_only else str(Path(state["prefix"]) / "runs/tb21/jobs" / job),
        "canonical_profile_endpoint": profile["agents"][0]["kwargs"]["sglang_url"],
        "task_slots_on_node": 0 if serve_only else len(gpus) * slots,
        "request_capacity_on_node": len(gpus) * slots,
        "task_cpu_budget": 0 if serve_only else len(gpus) * slots * task_cpus,
        "nominal_task_ram_gib": 0 if serve_only else len(gpus) * slots * task_ram_gib,
        "suggested_ram_gib": len(gpus) * (48 if serve_only else task_ram_gib * slots + 48),
        "resource_budget_note": "PRoot does not reserve per-task CPU/RAM; all processes share the allocation. "
        "RAM includes an estimated 48 GiB per replica for service/loading/tools, not a measured minimum.",
        "starts_services": False,
        "claims_tasks": False,
    }


def local_resources():
    cpu = float(len(os.sched_getaffinity(0)))
    cgroup = Path("/sys/fs/cgroup")
    quota = (
        (cgroup / "cpu.max").read_text().split()
        if (cgroup / "cpu.max").is_file()
        else ["max", "100000"]
    )
    if quota[0] != "max":
        cpu = min(cpu, int(quota[0]) / int(quota[1]))
    mem = (
        (cgroup / "memory.max").read_text().strip()
        if (cgroup / "memory.max").is_file()
        else "max"
    )
    if mem == "max":
        mem = (
            int(
                next(
                    x.split()[1]
                    for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemTotal:")
                )
            )
            * 1024
        )
    return {"cpu_limit": cpu, "memory_limit_gib": int(mem) / 2**30}


def discover_inventory(state, discover=None):
    deploy, gpu, _ = bootstrap(state)
    env = deploy.environment(state)
    mask = gpu.selected_devices(state["server"], env)
    return (discover or gpu.discover)(
        Path(state["prefix"]) / "server/bin/python", mask, env
    )


def inventory_failure(exc, resources):
    return {
        "status": "blocked",
        "errors": [
            f"CUDA inventory probe failed ({type(exc).__name__}); run inside the GPU allocation "
            "and check CUDA_VISIBLE_DEVICES and the serving Python environment"
        ],
        "warnings": [],
        "resources": resources,
        "gpu_inventory": [],
        "context_load_tested": False,
        "starts_services": False,
        "claims_tasks": False,
    }


def check_plan(
    state, plan, *, discover=None, resources=None, occupied=None, busy_uuids=None
):
    deploy, _, runtime = bootstrap(state)
    env = deploy.environment(state)
    resources = resources or local_resources()
    try:
        rows = discover_inventory(state, discover)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        return inventory_failure(exc, resources)
    errors, warnings = [], []
    binaries = ("gcc", "numactl") if plan["contract"].get("serve_only") else (
        "git", "gcc", "skopeo", "umoci", "curl", "flock", "numactl")
    for binary in binaries:
        if not shutil.which(binary, path=env["PATH"]):
            errors.append(
                f"host dependency missing: {binary}; provision the full node before starting"
            )
    occupied = occupied or runtime.occupied
    if busy_uuids is None:
        try:
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=30,
            )
            busy_uuids = set(processes.splitlines())
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"cannot check active GPU processes: {type(exc).__name__}")
            busy_uuids = set()
    for replica in plan["replicas"]:
        index = replica["gpu_index"]
        if index >= len(rows):
            errors.append(
                f"GPU {index} unavailable: this allocation exposes {len(rows)} GPU(s)"
            )
        elif rows[index]["capability"][0] < 8:
            errors.append(f"GPU {index} cannot serve BF16 on this stack")
        elif rows[index]["uuid"] in busy_uuids:
            errors.append(f"GPU {index} is already used by a compute process")
        if occupied(replica["port"]):
            errors.append(
                f"port {replica['port']} is occupied; no existing service will be reused"
            )
    policy = plan["contract"]["resource_policy"]
    budget_messages = errors if policy == "budgeted" else warnings
    if resources["cpu_limit"] < plan["task_cpu_budget"]:
        budget_messages.append(
            f"nominal task CPU budget {plan['task_cpu_budget']} exceeds allocation limit "
            f"{resources['cpu_limit']}; tasks, inference and tools share this quota; "
            "CPU contention can slow tasks and affect verifier timeouts"
        )
    if resources["memory_limit_gib"] < plan["suggested_ram_gib"]:
        budget_messages.append(
            f"estimated RAM budget {plan['suggested_ram_gib']:g} GiB exceeds allocation limit "
            f"{resources['memory_limit_gib']:.1f} GiB; reduce replicas with --gpus or request more RAM; "
            "this is an estimate, not a measured minimum"
        )
    count, tasks = len(plan["replicas"]), plan["task_slots_on_node"]
    assessment = {
        "policy": policy,
        "per_task_resources_reserved": False,
        "nominal_task_cpu_budget": plan["task_cpu_budget"],
        "nominal_task_ram_gib": plan["nominal_task_ram_gib"],
        "estimated_ram_budget_gib": plan["suggested_ram_gib"],
        "cpu_per_task_before_service_overhead": resources["cpu_limit"] / tasks if tasks else None,
        "ram_per_replica_before_task_and_service_overhead_gib": resources[
            "memory_limit_gib"
        ]
        / count,
        "replicas_within_estimated_ram_budget": min(
            count,
            int(resources["memory_limit_gib"] // (plan["suggested_ram_gib"] / count)),
        ),
    }
    receipt = read(Path(state["prefix"]) / "receipts/smoke.json")
    if receipt.get("status") != "ok":
        errors.append("base installation smoke has not passed")
    return {
        "status": "blocked" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "resources": resources,
        "gpu_inventory": rows,
        "credentials": credentials_report(),
        "selected_gpu_indices": [r["gpu_index"] for r in plan["replicas"]],
        "task_slots_on_node": tasks,
        "resource_assessment": assessment,
        "context_load_tested": False,
        "starts_services": False,
        "claims_tasks": False,
    }


@contextlib.contextmanager
def exclusive(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def worker_commands(state, plan, replica):
    prefix = Path(state["prefix"])
    return [
        [
            str(prefix / "controller/bin/python"),
            "-m",
            "terminal_bench_eval.terminal_bench_pool",
            "worker",
            "--jobs-root",
            str(prefix / "runs/tb21/jobs"),
            "--job-name",
            plan["contract"]["job"],
            "--worker-id",
            worker,
            "--device-label",
            f"{plan['node_id']}-gpu{replica['gpu_index']}-slot{slot}",
            "--sglang-url",
            replica["endpoint"],
            "--harbor-bin",
            str(prefix / "controller/bin/harbor"),
            "--preflight-module",
            "recursive_memory.stack.replica_preflight",
        ]
        for slot, worker in enumerate(replica["worker_ids"])
    ]


def deployment_root(state, contract):
    return (
        Path(state["prefix"])
        / "runs/tb21/node-launchers"
        / contract.get("deployment", contract["job"])
    )


def load_spec(path):
    spec = read(path)
    state = read(STATE_FILE)
    contract = spec["plan"]["contract"]
    if digest(state) != contract["base_state_sha256"]:
        raise RuntimeError(
            "installed state changed since replica specification was created"
        )
    shared = deployment_root(state, contract) / "contract.json"
    if read(shared) != contract or spec["replica"] not in spec["plan"]["replicas"]:
        raise RuntimeError(
            "replica specification differs from the shared launch contract"
        )
    if spec["replica"]["endpoint"] != f"http://127.0.0.1:{spec['replica']['port']}":
        raise RuntimeError("replica endpoint and port disagree")
    if (
        file_digest(__file__) != contract["launcher_sha256"]
        or file_digest(RECORD / "replica_preflight.py") != contract["preflight_sha256"]
    ):
        raise RuntimeError("launcher or preflight changed; use a new deployment")
    live = copy.deepcopy(state)
    live["server"].update(
        contract["server"], port=spec["replica"]["port"], devices=spec["gpu_uuid"]
    )
    return state, live, spec


def attest_capacity(info, spec):
    expected = spec["plan"]["contract"]["server"]
    if info.get("disable_cuda_graph") is not (not expected["cuda_graph"]):
        raise RuntimeError("replica CUDA Graph mode differs from the deployment")
    if (
        expected["cuda_graph"]
        and info.get("cuda_graph_max_bs", 0) < expected["max_running_requests"]
    ):
        raise RuntimeError(
            "replica CUDA Graph batch capacity is below requested concurrency"
        )
    for key, wanted in (("tp_size", 1), ("context_length", expected["context_length"])):
        if info.get(key) != wanted:
            raise RuntimeError(
                f"replica {key} differs: expected {wanted}, got {info.get(key)}"
            )
    for key, wanted in (
        ("max_total_num_tokens", expected["max_total_tokens"]),
        ("max_running_requests", expected["max_running_requests"]),
    ):
        if type(info.get(key)) is not int or info[key] < wanted:
            raise RuntimeError(f"replica {key} is below requested capacity {wanted}")
    input_limit = info.get("max_req_input_len")
    if type(input_limit) is not int or input_limit <= 16384:
        raise RuntimeError("replica does not advertise a usable input capacity")
    compact_limit = min(
        179200,
        min(info["context_length"], info["max_total_num_tokens"], input_limit) - 16384,
    )
    if expected["context_length"] == 262144 and compact_limit != 179200:
        raise RuntimeError(
            "256K replica cannot retain the intended 179200-token native compact threshold"
        )
    return {**info, "node_pool_native_compact_token_limit": compact_limit}


@contextlib.contextmanager
def protect_owned_cleanup(runtime):
    """Let the existing bounded TERM/wait/KILL cleanup finish despite repeated signals."""
    original = runtime.stop_owned

    def stop(child):
        handlers = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            original(child)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    runtime.stop_owned = stop
    try:
        yield
    finally:
        runtime.stop_owned = original


def run_replica(path):
    state, live, spec = load_spec(path)
    deploy, _, runtime = bootstrap(state)
    env = node_environment(state)
    env["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
    env.pop("CUDA_VISIBLE_DEVICE", None)
    env["R2EACT_NODE_POOL_SPEC"] = str(Path(path).resolve())
    os.environ.clear()
    os.environ.update(env)
    from shared_runtime.residual_preflight import get_server_info
    from terminal_bench_eval.parallel_workers import attest, supervise

    replica = spec["replica"]
    key = hashlib.sha256(spec["gpu_uuid"].encode()).hexdigest()[:16]
    with (
        exclusive(f"/tmp/rm-node-pool-{os.getuid()}-gpu-{key}.lock"),
        protect_owned_cleanup(runtime),
    ):
        if runtime.occupied(replica["port"]):
            raise RuntimeError(
                "replica port became occupied; refusing to reuse a foreign server"
            )
        with runtime.serving(live, start=True) as child:
            if child is None:
                raise RuntimeError("replica must own its GPU server")
            info = attest_capacity(get_server_info(replica["endpoint"]), spec)
            deploy.atomic(Path(path).with_suffix(".server.json"), info)
            attest(replica["endpoint"], len(replica["worker_ids"]))
            # Each worker runs rm_replica_preflight before its first task claim.
            if spec["plan"]["contract"].get("serve_only"):
                # Model ownership is independent of any queue or client. Only
                # this service supervisor's shutdown stops its owned server.
                code = child.wait()
            else:
                code = supervise(worker_commands(state, spec["plan"], replica))
            if code:
                raise SystemExit(code)


def start_node(state, plan):
    serve_only = plan["contract"].get("serve_only", False)
    if not serve_only:
        require_credentials()
    boundary = {"status": "ok", "scope": "server only; no task tools"} if serve_only else sandbox_report(state)
    if boundary["status"] != "ok":
        print(json.dumps(boundary, indent=2), flush=True)
        raise SystemExit(2)
    deploy, _, runtime = bootstrap(state)
    report = check_plan(state, plan)
    if report["status"] != "ok":
        raise RuntimeError("; ".join(report["errors"]))
    for warning in report["warnings"]:
        print(f"[node-pool] WARNING: {warning}", file=sys.stderr, flush=True)
    from terminal_bench_eval.parallel_workers import supervise

    root = deployment_root(state, plan["contract"])
    with exclusive(root / f"node-{plan['node_id']}.lock"):
        # Version the serving deployment independently from the immutable task queue.
        with exclusive(root / "prepare.lock", blocking=True):
            contract_path = root / "contract.json"
            if contract_path.exists() and read(contract_path) != plan["contract"]:
                raise RuntimeError(
                    "launch settings changed; choose a new deployment name"
                )
            deploy.atomic(contract_path, plan["contract"])
            env = node_environment(state)
            old_env = dict(os.environ)
            os.environ.clear()
            os.environ.update(env)
            try:
                if not serve_only:
                    args = argparse.Namespace(
                        benchmark="tb21", task=[], job=plan["contract"]["job"],
                        once=False, worker_id=None,
                        memory=plan["contract"].get("memory", "residual"),
                        serving_concurrency=plan["contract"]["slots_per_replica"],
                    )
                    prepare, _ = runtime.evaluation_commands(state, args)
                    subprocess.run(prepare, env=env, check=True)
            finally:
                os.environ.clear()
                os.environ.update(old_env)
        directory = root / plan["node_id"]
        directory.mkdir(exist_ok=True)
        deploy.atomic(
            directory / "allocation.json",
            {
                **report,
                "sandbox": boundary,
                "hostname": socket.gethostname(),
                "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
        commands = []
        for replica in plan["replicas"]:
            path = directory / f"gpu-{replica['gpu_index']}.json"
            spec = {
                "plan": plan,
                "replica": replica,
                "gpu_uuid": report["gpu_inventory"][replica["gpu_index"]]["uuid"],
            }
            deploy.atomic(path, spec)
            commands.append(
                [
                    str(Path(state["prefix"]) / "controller/bin/python"),
                    "-m",
                    "recursive_memory.stack.node",
                    "replica",
                    "--spec",
                    str(path),
                ]
            )
        # Inner workers have 60s grace, then the owned SGLang has up to 30s.
        # Leave the outer supervisor enough time for both cleanup stages.
        code = supervise(commands, grace=120)
        if code:
            raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "check", "check-env", "check-sandbox", "start", "serve", "replica"),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="load SERPER_API_KEY/MINERU_API_KEY as literal dotenv values; file values override inherited values",
    )
    parser.add_argument("--node-id")
    parser.add_argument(
        "--gpus",
        type=gpu_indices,
        default="auto",
        help="auto (default): use all GPUs visible to this allocation; or explicit ordinals, e.g. 0,1",
    )
    parser.add_argument(
        "--resource-policy",
        choices=("shared", "budgeted"),
        default="shared",
        help="shared: warn on nominal CPU/estimated RAM budget excess; budgeted: block on budget excess",
    )
    parser.add_argument("--slots", "--concurrency", type=int, default=2)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--token-pool", type=int, default=524288)
    parser.add_argument("--port-base", type=int, default=30000)
    parser.add_argument("--job")
    parser.add_argument("--memory", choices=("residual", "none"), default="residual",
                        help="none: paired native text-summary ablation; no residual encoding or soft tokens")
    parser.add_argument(
        "--deployment",
        help="separate serving revision; --job still selects the frozen task queue",
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--serve-only", action="store_true",
                        help="plan/check a persistent model service without creating or running a task pool")
    args = parser.parse_args()
    if args.env_file is not None:
        try:
            load_tool_env(args.env_file)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "check-env":
        report = credentials_report()
        print(json.dumps(report, indent=2))
        if report["status"] != "ok":
            raise SystemExit(2) from None
        return
    if args.command == "check-sandbox":
        report = sandbox_report(read(STATE_FILE))
        print(json.dumps(report, indent=2))
        if report["status"] != "ok":
            raise SystemExit(2)
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if args.command == "replica":
        if args.spec is None:
            parser.error("replica requires --spec")
        run_replica(args.spec)
        return
    if not args.node_id:
        parser.error("--node-id is required; use different IDs on the two nodes")
    state = read(STATE_FILE)
    rows = None
    if args.gpus == "auto":
        try:
            rows = discover_inventory(state)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(json.dumps(inventory_failure(exc, local_resources()), indent=2))
            raise SystemExit(2) from exc
        args.gpus = list(range(len(rows)))
    plan = build_plan(
        state,
        node_id=args.node_id,
        gpus=args.gpus,
        slots=args.slots,
        context=args.context,
        token_pool=args.token_pool,
        port_base=args.port_base,
        job=args.job,
        resource_policy=args.resource_policy,
        deployment=args.deployment,
        cuda_graph=args.cuda_graph,
        memory_fraction=args.memory_fraction,
        memory=args.memory,
        serve_only=args.serve_only or args.command == "serve",
    )
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
    elif args.command == "check":
        report = check_plan(
            state, plan, discover=(lambda *_: rows) if rows is not None else None
        )
        print(json.dumps(report, indent=2))
        if report["status"] != "ok":
            raise SystemExit(2)
    else:
        start_node(state, plan)


if __name__ == "__main__":
    main()
