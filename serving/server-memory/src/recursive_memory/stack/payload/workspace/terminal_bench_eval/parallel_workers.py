"""Run task slots against one owned SGLang; never initialize or rewrite pools."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


CAPABILITY = "qwen38_residual_cpu_concurrent_v1"


def slot_args(args, slot):
    args = list(args)
    for name in ("--worker-id", "--device-label"):
        for i, value in enumerate(args):
            if value == name:
                if i + 1 == len(args):
                    raise ValueError(f"{name} needs a value")
                args[i + 1] = f"{args[i + 1]}-slot-{slot}"
                break
            if value.startswith(name + "="):
                args[i] = value + f"-slot-{slot}"
                break
        else:
            raise ValueError(f"supervisor must supply {name}")
    return args


def attest(url, count, *, mode="residual"):
    from shared_runtime.residual_preflight import get_server_info

    info = get_server_info(url)
    limit = info.get("max_running_requests")
    if mode == "discrete":
        from shared_runtime.discrete_concurrency import assert_discrete_concurrency
        assert_discrete_concurrency(info, count)
    elif mode != "residual":
        raise ValueError("parallel mode must be residual or discrete")
    elif (
        CAPABILITY not in (info.get("request_capabilities") or [])
        or type(limit) is not int
        or limit < count
    ):
        raise RuntimeError(
            f"{count} parallel slots require patched residual SGLang with "
            f"MAX_RUNNING_REQUESTS>={count}; no task claimed. Restart the server, not only the worker."
        )
    expected_tp = os.environ.get("TERMINAL_BENCH_POOL_TP_SIZE")
    if expected_tp and info.get("tp_size") != int(expected_tp):
        raise RuntimeError(
            f"existing SGLang TP={info.get('tp_size')} does not match requested TP={expected_tp}; no task claimed"
        )


def supervise(commands, *, grace=60.0):
    children = []
    received = []
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[sig] = signal.signal(
            sig, lambda signum, frame: received.append(signum)
        )
    result = 0
    try:
        for command in commands:
            if received:
                break
            children.append(subprocess.Popen(command, start_new_session=True))
        while any(child.poll() is None for child in children) and not received:
            failed = next(
                (
                    child.returncode
                    for child in children
                    if child.poll() not in (None, 0)
                ),
                None,
            )
            if failed is not None:
                if failed == 75 and os.environ.get("TERMINAL_BENCH_POOL_RECOVERY_SIGNAL_FILE"):
                    from pathlib import Path
                    Path(os.environ["TERMINAL_BENCH_POOL_RECOVERY_SIGNAL_FILE"]).write_text("sglang_unavailable\n")
                result = failed if failed > 0 else 128 - failed
                break
            time.sleep(0.2)
        if received:
            result = 128 + received[0]
        if not result:
            result = next(
                (child.returncode for child in children if child.returncode), 0
            )
    finally:
        alive = [child for child in children if child.poll() is None]
        for child in alive:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        for child in alive:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if not 2 <= options.count <= 32 or not options.args or options.args[0] != "worker":
        parser.error("requires 2..32 slots and the worker subcommand")
    commands = [
        ["/bin/bash", options.launcher, *slot_args(options.args, slot)]
        for slot in range(options.count)
    ]
    try:
        attest(os.environ["TERMINAL_BENCH_POOL_SGLANG_URL"], options.count,
               mode=os.environ.get("TERMINAL_BENCH_POOL_PARALLEL_MODE", "residual"))
    except Exception as exc:
        from shared_runtime.serving_recovery import transient_transport
        if not transient_transport(exc):
            raise
        print(f"[terminal-bench] serving unavailable before slot startup: {exc}", flush=True)
        return 75
    print(
        f"[terminal-bench] {options.count} task slots sharing one SGLang endpoint",
        flush=True,
    )
    return supervise(commands)


if __name__ == "__main__":
    raise SystemExit(main())
