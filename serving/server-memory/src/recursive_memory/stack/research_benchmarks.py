"""Portable JobBench/BrowseComp entrypoints; no site paths or live pool imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .deploy import POLICY, read

BENCHMARKS = ("jobbench", "browsecomp100")
DEFAULT_SEED = 20260912


def root_for(state, benchmark):
    if benchmark not in BENCHMARKS:
        raise ValueError("unsupported research benchmark")
    return Path(state["prefix"]) / "runs" / benchmark


def config_for(state, benchmark, *, memory="residual"):
    from browsecomp_bench_eval.codex_config import BrowseCompCodexConfig
    from job_bench_eval.config import JobBenchConfig

    root = root_for(state, benchmark)
    if memory not in {"residual", "none"}:
        raise ValueError("memory must be residual or none")
    cls = JobBenchConfig if benchmark == "jobbench" else BrowseCompCodexConfig
    return cls(
        runtime_root=str(root),
        dataset_root=str(root / "source/dataset"),
        harness_name=("Codex CLI / native compact + residual / xhigh" if memory == "residual"
                      else "Codex CLI / native compact + text summary only / xhigh"),
        harness_mode="native_compact",
        native_compaction_text_summary=True,
        compressor_enabled=memory == "residual",
        residual_compaction=memory == "residual",
        residual_checkpoint=state["checkpoint"],
        working_memory_turns=12,  # compatibility field; not a native trigger
        serving_eviction_policy=POLICY,
    ).validate()


def prepare(state, benchmark, job, *, task_ids=(), seed=DEFAULT_SEED, offline=False, memory="residual"):
    from shared_runtime.storage import safe_name

    root = root_for(state, benchmark)
    safe_name(job)
    config = config_for(state, benchmark, memory=memory)
    offline = offline or (root / "jobs" / job / "pool.json").is_file()
    if benchmark == "jobbench":
        from job_bench_eval.bootstrap import prepare as bootstrap
        from job_bench_eval.pool import initialize

        bootstrap(root, config.split, offline)
        path = initialize(root, job, config, list(task_ids) or None)
    else:
        from browsecomp_bench_eval.codex_pool import initialize

        path = initialize(
            root, job, config, cache_dir=root / "cache", seed=seed, offline=offline,
            task_ids=list(task_ids) or None,
        )
    print(
        json.dumps(
            {
                "job": str(path),
                "benchmark": benchmark,
                "mode": "native_compact",
                "memory": memory,
                "reasoning_effort": "xhigh",
                "existing_results_preserved": True,
            },
            indent=2,
        )
    )
    return path


def commands(state, args, job):
    from .runtime import endpoint

    root = root_for(state, args.benchmark)
    prepare_command = [
        sys.executable,
        "-m",
        __name__,
        "prepare",
        args.benchmark,
        "--job-name",
        job,
        "--seed",
        str(getattr(args, "seed", DEFAULT_SEED)),
        "--memory", getattr(args, "memory", "residual"),
    ]
    if getattr(args, "offline_dataset", False):
        prepare_command.append("--offline-dataset")
    for task in args.task:
        prepare_command.extend(["--task-id", task])
    module = (
        "job_bench_eval.pool"
        if args.benchmark == "jobbench"
        else "browsecomp_bench_eval.codex_pool"
    )
    worker = [
        sys.executable,
        "-m",
        module,
        "worker",
        "--runtime-root",
        str(root),
        "--job-name",
        job,
        "--sglang-url",
        endpoint(state),
        "--idle-exit-seconds",
        "120",
    ]
    if args.once:
        worker.extend(["--max-tasks", "1"])
    return prepare_command, worker


def judge_command(state, args):
    from shared_runtime.storage import safe_name

    from .runtime import endpoint

    job = root_for(state, args.benchmark) / "jobs" / safe_name(args.job)
    if not (job / "pool.json").is_file():
        raise RuntimeError(f"prepare the pool before starting its judge: {job}")
    # Generation uses pinned Codex 0.149 against SGLang; judges use a separately
    # installed/authenticated CLI supporting --ignore-user-config. Never select
    # the generation binary merely because the installed PATH puts it first.
    binary = args.codex_binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(
            "--codex-binary must identify an executable logged-in judge CLI"
        )
    if args.benchmark == "jobbench":
        command = [
            str(Path(state["prefix"]) / "documents/bin/python"),
            "-m",
            "job_bench_eval.codex_judge",
            "--job",
            str(job),
            "--model",
            args.model,
            "--reasoning-effort",
            args.effort,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "browsecomp_bench_eval.codex100_review",
            "--runtime-root",
            str(job.parent.parent),
            "--job-name",
            job.name,
            "--model",
            args.model,
            "--effort",
            args.effort,
            "--sglang-url",
            endpoint(state),
        ]
    if args.watch:
        command.append("--watch")
    return command, {**os.environ, "JOBBENCH_JUDGE_CODEX": str(binary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--offline-dataset", action="store_true")
    parser.add_argument("--memory", choices=("residual", "none"), default="residual")
    args = parser.parse_args()
    prepare(
        read(os.environ["R2EACT_STACK_STATE"]),
        args.benchmark,
        args.job_name,
        task_ids=args.task_id,
        seed=args.seed,
        offline=args.offline_dataset,
        memory=args.memory,
    )


if __name__ == "__main__":
    main()
