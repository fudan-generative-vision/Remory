"""Prepare paired native-compact + xhigh pools, with or without residual memory.

Server ownership and concurrent worker supervision belong to stack.runtime.
This module only prepares the immutable pool; it never starts a second server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import pool, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get(
        "AUTOMATIONBENCH_RUNTIME_ROOT", runner.DEFAULT_RUNTIME)))
    parser.add_argument("--job-name", default=pool.DEFAULT_JOB)
    parser.add_argument("--domain", choices=pool.DOMAINS)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=pool.DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--mode", choices=["native_compact"], default="native_compact")
    parser.add_argument("--compressor", choices=["residual", "none"], default="residual")
    parser.add_argument("--reasoning-effort", choices=["xhigh"], default="xhigh")
    parser.add_argument("--checkpoint", default=os.environ.get("AUTOMATIONBENCH_CHECKPOINT", runner.DEFAULT_RESIDUAL))
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not (args.prepare_only or args.list):
        parser.error("Use stack.sh run automationbench to supervise evaluation workers")
    if (args.max_steps < 0 or not 0 < args.timeout_seconds < float("inf")
            or args.num_examples is not None and args.num_examples < 1
            or not 0 <= args.shuffle_seed < 2**64):
        parser.error("Invalid task count, seed or budget")
    root = args.runtime_root.resolve()
    runner.check_source(root)
    tasks = pool.catalog(root)
    if args.domain:
        tasks = [task for task in tasks if task["domain"] == args.domain]
    if args.task_id:
        if len(set(args.task_id)) != len(args.task_id) or set(args.task_id) - {t["task_id"] for t in tasks}:
            parser.error("task-id must be unique exact public task names")
        tasks = [task for task in tasks if task["task_id"] in args.task_id]
    if args.num_examples is not None:
        tasks = tasks[:args.num_examples]
    if args.list:
        print(json.dumps([t["task_id"] for t in tasks], indent=2))
        return
    if len(tasks) != 600 and args.job_name == pool.DEFAULT_JOB:
        parser.error("A subset requires a separate job name")
    config = {**pool.default_config(str(Path(args.checkpoint).resolve())),
              "compressor": args.compressor,
              "max_steps": args.max_steps, "timeout_seconds": args.timeout_seconds,
              "shuffle_seed": args.shuffle_seed}
    job = pool.initialize(root, args.job_name, config, pool.shuffled_tasks(tasks, args.shuffle_seed))
    print(json.dumps(pool.status(job), indent=2), flush=True)


if __name__ == "__main__":
    main()
