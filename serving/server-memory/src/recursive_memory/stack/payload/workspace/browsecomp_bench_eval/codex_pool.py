"""Fixed random BrowseComp-100 pool with the TB/JobBench Codex residual runner.

Only question files cross the Landlock boundary. Gold answers are read after
the agent has exited; judge failures stay pending, never become reward=0.
Legacy pool services are not used. Each fresh execution pins its BrowseComp
system policy; local Qwen rejections are reviewed by an isolated Codex controller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket

from browsecomp_bench_eval.dataset import (
    EncryptedBrowseCompDataset, ensure_dataset_cached, generation_sample, grading_reference,
)
from browsecomp_bench_eval.grader import OpenAICompatibleGrader, OPENAI_SIMPLE_EVALS_GRADER_TEMPLATE_SHA256
from .codex_config import BrowseCompCodexConfig
from .codex_policy import SYSTEM_INSTRUCTIONS, POLICY_ID, confidence_receipt
from job_bench_eval.dataset import JobBenchTask, task_folder_digest
from job_bench_eval.pool import status, worker
from job_bench_eval.runner import run_jobbench_task
from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT / "eval/browsecomp100_codex"
DEFAULT_CACHE = PROJECT / "eval/browsecomp_bench/cache"
DEFAULT_SEED = 20260912
JOB_NAME = "browsecomp100-native-compact-residual-xhigh-seed20260912"
SCHEMA = "browsecomp100_codex_pool_v1"


def select_rows(dataset, seed, count=100):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if len(dataset.rows) < count:
        raise ValueError(f"need at least {count} source questions")
    return random.Random(seed).sample(sorted(dataset.rows, key=lambda r: r.row_index), count)


def selected_rows(dataset, seed, task_ids=None):
    rows = select_rows(dataset, seed)
    if task_ids is None:
        return rows
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("BrowseComp subset must contain unique task ids")
    wanted = set(task_ids)
    if not wanted <= {row.sample_id for row in rows}:
        raise ValueError("BrowseComp subset contains tasks outside the fixed seeded 100")
    return [row for row in rows if row.sample_id in wanted]


def initialize(root, name, config, *, cache_dir=DEFAULT_CACHE, seed=DEFAULT_SEED, offline=False,
               paired_job=None, task_ids=None):
    root = Path(root).resolve()
    job = root / "jobs" / safe_name(name)
    config.validate()
    cache = ensure_dataset_cached(cache_dir, offline=offline)
    dataset = EncryptedBrowseCompDataset.from_csv(cache)
    selected = selected_rows(dataset, seed, task_ids)
    selection = {"policy": ("python_random_sample_without_replacement_v1" if task_ids is None
                            else "explicit_subset_of_seeded_100_v1"), "seed": seed,
        "count": len(selected), "population": len(dataset.rows), "dataset_sha256": dataset.sha256,
        "row_indices": [r.row_index for r in selected], "sample_ids": [r.sample_id for r in selected]}
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    paired = None
    if paired_job is not None:
        if config.harness_mode != "native_compact" or paired_job == name:
            raise ValueError("paired BrowseComp native compact requires a different reference pool")
        source_path = root / "jobs" / safe_name(paired_job)
        if not (source_path / "pool.json").is_file():
            raise ValueError(f"paired BrowseComp reference pool is unavailable: {source_path}")
        source = validate_pool(source_path)
        # Only the compact trigger/representation differs, never the selected
        # questions, checkpoint, task budget or generation/judging protocol.
        ignored = {"harness_mode", "harness_name", "native_compaction_text_summary", "serving_eviction_policy"}
        source_config = {k: v for k, v in source["config"].items() if k not in ignored}
        target_config = {k: v for k, v in config.to_json().items() if k not in ignored}
        if (source["config"]["harness_mode"] != "r2eact" or source_config != target_config
            or source["selection"] != selection or source["dataset_sha256"] != dataset.sha256):
            raise ValueError("paired BrowseComp questions, checkpoint or budget differ from the reference")
        paired = {"source_job": paired_job,
            "source_manifest_sha256": hashlib.sha256((source_path / "pool.json").read_bytes()).hexdigest(),
            "selection_sha256": selection_hash, "import_results": False}
    with lock(job / "init.lock"):
        job.chmod(0o700)
        if (job / "pool.json").exists():
            manifest = validate_pool(job)
            if (manifest["config"] != config.to_json() or manifest["selection"] != selection or
                manifest["dataset_path"] != str(dataset.source_path.resolve()) or
                manifest.get("paired_pool") != paired):
                raise RuntimeError("existing BrowseComp-100 contract/selection differs; use a new pool name")
            return job
        tasks = []
        for row in selected:
            folder = job / "inputs" / row.sample_id
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            instruction = folder / "TASK_INSTRUCTIONS.txt"
            question = generation_sample(row).question
            # Only deterministic question data, never an answer/canary/rubric.
            if instruction.exists() and instruction.read_text() != question:
                raise RuntimeError("partial BrowseComp question projection differs")
            instruction.write_text(question, encoding="utf-8")
            instruction.chmod(0o600)
            task = JobBenchTask(row.sample_id, "browsecomp", row.sample_id, folder,
                row.sample_id, task_folder_digest(folder))
            record = asdict(task)
            record["task_folder"] = str(folder)
            tasks.append(record)
            cell = job / "cells" / row.sample_id
            if not (cell / "state.json").exists():
                atomic_json(cell / "state.json", {"status": "pending", "execution": 0})
        manifest = {"schema": SCHEMA, "benchmark": "BrowseComp", "job_name": name,
            "config": config.to_json(), "contract_hash": config.contract_hash,
            "dataset_path": str(dataset.source_path.resolve()), "dataset_sha256": dataset.sha256,
            "selection": selection, "selection_sha256": selection_hash, "tasks": tasks,
            "judge": {"provider": "local_sglang", "prompt_sha256": OPENAI_SIMPLE_EVALS_GRADER_TEMPLATE_SHA256},
            "created_at": utcnow()}
        if paired is not None:
            manifest["paired_pool"] = paired
        atomic_json(job / "selection.json", selection)
        atomic_json(job / "pool.json", manifest)  # commit last
    return job


def validate_pool(job):
    job = Path(job).resolve()
    manifest = read_json(job / "pool.json")
    config = BrowseCompCodexConfig.from_json(manifest["config"])
    if manifest["schema"] != SCHEMA or config.contract_hash != manifest["contract_hash"]:
        raise ValueError("BrowseComp-100 pool contract mismatch")
    selection = manifest["selection"]
    if hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest() != manifest["selection_sha256"]:
        raise ValueError("BrowseComp-100 selection hash mismatch")
    dataset = EncryptedBrowseCompDataset.from_csv(manifest["dataset_path"], expected_sha256=manifest["dataset_sha256"])
    policy = selection.get("policy")
    if policy not in {"python_random_sample_without_replacement_v1", "explicit_subset_of_seeded_100_v1"}:
        raise ValueError("Unknown BrowseComp selection policy")
    rows = selected_rows(dataset, selection["seed"], selection["sample_ids"]
        if policy == "explicit_subset_of_seeded_100_v1" else None)
    ids = [r.sample_id for r in rows]
    if (ids != selection["sample_ids"] or len(manifest["tasks"]) != len(rows) or
        selection["count"] != len(rows) or
        [t["task_id"] for t in manifest["tasks"]] != ids or
        selection["row_indices"] != [r.row_index for r in rows]):
        raise ValueError("BrowseComp-100 selection differs from its seed/source")
    for task, row in zip(manifest["tasks"], rows):
        folder = job / "inputs" / row.sample_id
        if (Path(task["task_folder"]).resolve() != folder or
            task_folder_digest(folder) != task["source_sha256"] or
            (folder / "TASK_INSTRUCTIONS.txt").read_text() != generation_sample(row).question):
            raise ValueError("BrowseComp question projection changed")
    return manifest


def prompt(workspace, mcp):
    question = (workspace / "task_folder/TASK_INSTRUCTIONS.txt").read_text(encoding="utf-8")
    return f"""{question}

Research with the available web search, shell, document and image tools as needed.
Finish with: Explanation: <brief evidence and sources>\nExact Answer: <answer>\nConfidence: <0-100%>.
Put the answer in your final assistant message. Do not stop at a plan or a saved file.
Previous compressed history can be retrieved using shell commands on {workspace / 'memory.jsonl'}.
Use Docling for PDF/document parsing and MinerU as a fallback. Use view_image for images;
update_plan is available. You may install packages locally within your workspace.
The document-tools Python interpreter is {mcp / 'bin/python'}.
"""


def grade_execution(job, execution, *, sglang_url):
    """Post-hoc judging is independent of generation and can be retried alone."""
    execution = Path(execution).resolve()
    result = read_json(execution / "result.json")
    if result.get("status") != "completed" or result.get("judge_status") in {"completed", "review_pending"}:
        return result
    with lock(execution / "judge.lock"):
        result = read_json(execution / "result.json")
        if result.get("judge_status") in {"completed", "review_pending"}:
            return result
        try:
            manifest = read_json(Path(job) / "pool.json")
            dataset = EncryptedBrowseCompDataset.from_csv(manifest["dataset_path"], expected_sha256=manifest["dataset_sha256"])
            row = next(r for r in dataset.rows if r.sample_id == result["task_id"])
            answer = (execution / "final_answer.txt").read_text()
            # Shares the request-isolated ABI; blocks behind legacy exclusive clients.
            from shared_runtime.residual_preflight import get_server_info, RESIDUAL_CONCURRENCY_CAPABILITY
            info = get_server_info(sglang_url)
            served_model = info.get("served_model_name")
            if not isinstance(served_model, str) or not served_model.strip():
                raise RuntimeError("Qwen judge server does not advertise its served_model_name")
            concurrent = RESIDUAL_CONCURRENCY_CAPABILITY in (info.get("request_capabilities") or [])
            with lock(Path("/tmp/r2eact-terminal-agent-sglang.lock"), shared=concurrent):
                grade = OpenAICompatibleGrader(base_url=sglang_url,
                    model=served_model,
                    disable_thinking=True, strict=True, bypass_proxy=True).grade(
                    question=generation_sample(row).question,
                    correct_answer=grading_reference(row).answer, response=answer)
            verdicts = re.findall(r"(?im)^\s*correct\s*:\s*(yes|no)\b", grade.response_text)
            if len(verdicts) != 1:
                raise ValueError("judge output has no unique correct: yes/no verdict")
            atomic_json(execution / "judge.json", {**grade.to_private_json(),
                "prompt_sha256": OPENAI_SIMPLE_EVALS_GRADER_TEMPLATE_SHA256,
                "ground_truth": grading_reference(row).answer})
            result.update(qwen_reward=int(grade.correct), **confidence_receipt(answer))
            if grade.correct:
                result.update(reward=1, judge_status="completed", reward_source="Qwen",
                              codex_review_status="not_required")
            else:
                result.update(reward=None, judge_status="review_pending",
                              reward_source=None, codex_review_status="pending")
            result.pop("judge_error", None)
        except Exception as exc:
            # Network/format failures must not count as an incorrect model answer.
            result.update(judge_status="pending", reward=None, judge_error=f"{type(exc).__name__}: {exc}")
        atomic_json(execution / "result.json", result)
    return result


def run_request(path):
    request = read_json(path)
    execution = Path(request["execution"]).resolve()
    job = execution.parents[3]
    manifest = validate_pool(job)
    if (request["task"] not in manifest["tasks"] or request["config"] != manifest["config"] or
        execution.parent.parent.name != request["task"]["task_id"]):
        raise ValueError("BrowseComp execution does not match its immutable pool")
    task = dict(request["task"])
    task["task_folder"] = Path(task["task_folder"])
    atomic_json(execution / "browsecomp-policy.json", {"policy_id": POLICY_ID,
        "system_instructions": SYSTEM_INSTRUCTIONS, "applies_to": "this_execution_only",
        "judge_policy": "qwen_then_codex_review_all_rejections_v1"})
    run_jobbench_task(BrowseCompCodexConfig.from_json(request["config"]), JobBenchTask(**task), execution,
        sglang_url=request["sglang_url"], prompt_builder=prompt, system_instructions=SYSTEM_INSTRUCTIONS)
    grade_execution(job, execution, sglang_url=request["sglang_url"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", choices=["worker", "status", "grade"])
    p.add_argument("--request", type=Path)
    p.add_argument("--runtime-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--job-name", default=JOB_NAME)
    p.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    p.add_argument("--device-label", default="platform-assigned")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--idle-exit-seconds", type=float, default=120)
    p.add_argument("--offline-dataset", action="store_true")
    a = p.parse_args()
    if a.request:
        run_request(a.request)
        return
    job = a.runtime_root.resolve() / "jobs" / safe_name(a.job_name)
    manifest = validate_pool(job)
    worker_code = 0
    if a.action == "worker":
        worker_code = worker(job, worker_id=a.worker_id, device_label=a.device_label, sglang_url=a.sglang_url,
            idle_exit_seconds=a.idle_exit_seconds, max_tasks=a.max_tasks, runner_module=__spec__.name,
            config_class=BrowseCompCodexConfig)
    elif a.action == "grade":
        for task in manifest["tasks"]:
            state = read_json(job / "cells" / task["task_id"] / "state.json")
            if state["status"] == "completed" and state.get("result"):
                grade_execution(job, Path(state["result"]).parent, sglang_url=a.sglang_url)
    print(json.dumps(status(job), indent=2))
    if worker_code:
        raise SystemExit(worker_code)


if __name__ == "__main__":
    main()
