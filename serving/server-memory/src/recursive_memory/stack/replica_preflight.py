"""Pinned TB preflight with canonical queue identity and a per-replica endpoint.

Adapted from the installed runtime.tb_preflight. Profile comparison, PRoot
validation, frozen task catalog, server attestation and tool preflight remain.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .node import attest_capacity, bootstrap, digest, load_spec


def validate_profile(config, expected):
    # Harbor may normalize jobs_dir when freezing the pool, as in upstream.
    expected.jobs_dir = config.jobs_dir
    if config.model_dump(mode="python") != expected.model_dump(mode="python"):
        raise RuntimeError(
            "pool configuration differs from the canonical installation; no task claimed"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state, live, spec = load_spec(os.environ["R2EACT_NODE_POOL_SPEC"])
    deploy, _, runtime = bootstrap(state)
    if args.sglang_url.rstrip("/") != spec["replica"]["endpoint"]:
        parser.error("worker endpoint differs from its assigned replica")
    import yaml
    from eval.terminal_bench.proot_environment import ProotEnvironment
    from harbor.models.job.config import JobConfig
    from shared_runtime.residual_preflight import get_server_info
    from terminal_bench_eval.terminal_bench_pool import load_pool_spec

    config = JobConfig.model_validate(yaml.safe_load(args.config.read_text()))
    expected_profile = runtime.tb_profile(
        state, spec["plan"]["contract"]["job"],
        memory=spec["plan"]["contract"].get("memory", "residual"),
        serving_concurrency=spec["plan"]["contract"]["slots_per_replica"],
    )
    if digest(expected_profile) != spec["plan"]["contract"]["profile_sha256"]:
        raise RuntimeError("canonical profile changed")
    validate_profile(config, JobConfig.model_validate(expected_profile))
    try:
        ProotEnvironment.preflight()
    except SystemExit as exc:
        raise RuntimeError(f"PRoot preflight failed: {exc}") from exc
    frozen = load_pool_spec(
        job_name=args.config.parent.name, jobs_root=args.config.parent.parent
    )
    tasks = {cell.task_name for cell in frozen.cells}
    if len(tasks) != 89 or len(frozen.cells) != 89:
        raise RuntimeError(
            "this launcher requires the full 89-task, one-attempt TB2.1 pool"
        )
    attestation = runtime.admit_server(live)
    info = attest_capacity(get_server_info(args.sglang_url), spec)
    try:
        tool_report = runtime.tool_preflight()
    except subprocess.CalledProcessError as exc:
        # The installed runner captures this stderr, which previously hid the cause.
        if isinstance(exc.cmd, list | tuple) and any(
            Path(str(arg)).name in {"sandbox.py", "bubblewrap.py"} for arg in exc.cmd
        ):
            detail = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
            print(
                "[node-pool] document sandbox probe stderr:\n"
                + (detail or "<empty>")[-4000:],
                file=sys.stderr,
            )
        raise
    tool_report["sandbox_backend"] = spec["plan"]["contract"]["sandbox_backend"]
    report = {
        "status": "ok",
        "dataset": {
            "task_count": len(tasks),
            "registry_queried": False,
            "source": deploy.DATASET,
        },
        "server": attestation,
        "tools": tool_report,
        "replica": {
            "node": spec["plan"]["node_id"],
            "gpu_uuid": spec["gpu_uuid"],
            "deployment": spec["plan"]["contract"].get(
                "deployment", spec["plan"]["contract"]["job"]
            ),
            "endpoint": args.sglang_url,
            "tp_size": info["tp_size"],
            "cuda_graph": not info["disable_cuda_graph"],
            "cuda_graph_max_bs": info.get("cuda_graph_max_bs"),
            "max_running_requests": info["max_running_requests"],
            "context_length": info["context_length"],
            "max_total_num_tokens": info["max_total_num_tokens"],
            "native_compact_token_limit": info["node_pool_native_compact_token_limit"],
            "max_req_input_len": info.get("max_req_input_len"),
        },
    }
    if args.output:
        deploy.atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
