"""Controller-only, resumable official JobBench rubrics through Codex login."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

from .codex_judge_client import CodexJudgeClient, digest, resolve_profile
from .config import OFFICIAL_EVAL_COMMIT
from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow


def load_official(runtime_root):
    repo = Path(runtime_root) / "source/official-eval"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    source = repo / "eval/judge.py"
    committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{OFFICIAL_EVAL_COMMIT}:eval/judge.py"])
    if revision != OFFICIAL_EVAL_COMMIT or source.read_bytes() != committed:
        raise RuntimeError("official judge source differs from the pinned revision")
    spec = importlib.util.spec_from_file_location("jobbench_pinned_official_judge", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_manifest(directory):
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("committed model_output directory is missing or a symlink")
    files = []
    for path in sorted(directory.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"refusing non-regular judge input: {path}")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        files.append({"path": path.relative_to(directory).as_posix(), "sha256": checksum.hexdigest()})
    return files


def score_execution(official, profile, task, execution, *, timeout=300, retries=2):
    rubric_path = Path(task["task_folder"]).parent / "RUBRICS.json"
    payload = read_json(rubric_path)
    rubrics = payload.get("rubrics") or payload.get("evaluation_rubrics") or []
    if not rubrics or any(not isinstance(r.get("weight"), (int, float)) or isinstance(r["weight"], bool)
        or not math.isfinite(r["weight"]) or r["weight"] < 0 for r in rubrics) or sum(r["weight"] for r in rubrics) <= 0:
        raise RuntimeError("invalid or empty official rubrics")
    output = execution / "model_output"
    snapshot = {"profile": profile, "official_eval_commit": OFFICIAL_EVAL_COMMIT,
        "rubrics": rubrics, "files": file_manifest(output)}
    identity = digest(snapshot)
    artifacts = execution / "judge_codex" / identity
    if (artifacts / "judge.json").exists():
        return read_json(artifacts / "judge.json")
    atomic_json(artifacts / "input.json", snapshot)
    contents = official.extract_all_file_contents(output) if snapshot["files"] else ""
    # Converter failures are infrastructure failures, not a zero-score answer.
    if "[ERROR:" in contents:
        raise RuntimeError("official document conversion reported an error; inspect deliverables and judge dependencies")
    images = official.collect_image_attachments(output) if contents.strip() else []
    results = []
    for index, rubric in enumerate(rubrics):
        rubric_dir = artifacts / f"rubric-{index:03d}"
        cached = rubric_dir / "result.json"
        if cached.exists():
            results.append(read_json(cached))
            continue
        if not contents.strip():
            reason = "No output files found in the model output directory." if not snapshot["files"] else "Output files were unreadable or empty after conversion."
            result = official.build_failed_rubric_result(index, rubric, reason)
        else:
            client = CodexJudgeClient(profile, rubric_dir, len(official.normalize_criteria(rubric)))
            previous = official.get_openai_client
            try:
                # Only this transport is replaced. The original function builds
                # system/user prompts, selects vision, enriches criteria and scores.
                official.get_openai_client = lambda *args, **kwargs: client
                result, debug = official.judge_rubric(index, rubric, contents, profile["model"],
                    "codex://saved-chatgpt-login", "", timeout, retries, images)
            finally:
                official.get_openai_client = previous
            atomic_json(rubric_dir / "debug.json", debug)
            if debug.get("api_exit_code") != 0:
                # Upstream emits an all-false fallback here; deliberately never
                # commit/cache that fallback as a judgment.
                raise RuntimeError(f"rubric {index} is pending: {debug.get('error', 'judge call failed')}")
        atomic_json(cached, result)
        results.append(result)
    if snapshot["files"] != file_manifest(output) or payload != read_json(rubric_path):
        raise RuntimeError("judge inputs changed during scoring; result not committed")
    report = official.build_details_report(task["task_id"], profile["model"], results, total_count=len(rubrics))
    atomic_json(artifacts / "report.json", report)
    scorecard = official.build_scorecard(results)
    result = {"task_id": task["task_id"], "status": "completed", "backend": "codex_chatgpt",
        "judge_model": profile["model"], "judge_profile": profile, "input_fingerprint": identity,
        "score": report["total_score"], "max_score": report["max_score"],
        "reward": report["total_score"] / report["max_score"],
        "official_rounded_reward": official.build_reward(scorecard)["reward"],
        "official_eval_commit": OFFICIAL_EVAL_COMMIT, "report": str(artifacts / "report.json"),
        "judged_at": utcnow()}
    atomic_json(artifacts / "judge.json", result)
    return result


def judge_pass(job, official, profile, *, timeout=300, retries=2, retry_seconds=300):
    manifest = read_json(job / "pool.json")
    reports, pending = [], []
    for task in manifest["tasks"]:
        cell = job / "cells" / safe_name(task["task_id"])
        with lock(cell / "owner.lock", blocking=False) as owner:
            if owner is None:
                continue
            state = read_json(cell / "state.json")
            if state.get("status") not in {"completed", "timeout"}:
                continue
            execution = Path(state["result"]).resolve().parent
            if not execution.is_relative_to((cell / "executions").resolve()):
                raise RuntimeError("result is outside its cell executions")
            existing = read_json(execution / "judge.json") if (execution / "judge.json").exists() else {}
            if existing.get("status") == "pending" and existing.get("judge_profile") == profile:
                if time.time() < existing.get("retry_after_epoch", 0):
                    pending.append({"task_id": task["task_id"], "error": existing.get("error")})
                    continue
            try:
                judged = score_execution(official, profile, task, execution, timeout=timeout, retries=retries)
                atomic_json(execution / "judge.json", judged)
                # Preserve the generation outcome. Reward is a separate commit
                # and can be repaired here after a controller crash.
                result = read_json(execution / "result.json")
                atomic_json(execution / "result.json", {**result, "judge_status": "completed",
                    "reward": judged["reward"], "judge_file": str(execution / "judge.json")})
                reports.append(judged)
                if existing.get("input_fingerprint") != judged["input_fingerprint"]:
                    print(f"[jobbench-judge] {task['task_id']}: reward={judged['reward']:.4f}", flush=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                atomic_json(execution / "judge.json", {"task_id": task["task_id"], "status": "pending",
                    "reward": None, "judge_profile": profile, "error": error, "updated_at": utcnow(),
                    "retry_after_epoch": time.time() + retry_seconds})
                result = read_json(execution / "result.json")
                atomic_json(execution / "result.json", {**result, "judge_status": "pending", "reward": None})
                pending.append({"task_id": task["task_id"], "error": error})
                print(f"[jobbench-judge] {task['task_id']}: pending ({error})", flush=True)
    summary = {"updated_at": utcnow(), "backend": "codex_chatgpt", "judge_profile": profile,
        "scored_tasks": len(reports), "pending_tasks": len(pending), "pending": pending,
        "mean_weighted_score": sum(r["reward"] for r in reports) / len(reports) if reports else None,
        "reports": reports}
    atomic_json(job / "judge_summary.json", summary)
    return summary


def run(job, *, model=None, effort=None, watch=False, poll_seconds=60, timeout=300, retries=2):
    pinned = job / "codex_judge_profile.json"
    if pinned.exists():
        saved = read_json(pinned)
        model = model or os.environ.get("JOBBENCH_JUDGE_MODEL") or saved["model"]
        effort = effort or os.environ.get("JOBBENCH_JUDGE_REASONING_EFFORT") or saved["reasoning_effort"]
    profile = resolve_profile(model, effort)
    official = load_official(read_json(job / "pool.json")["config"]["runtime_root"])
    with lock(job / "judge.lock", blocking=False) as owner:
        if owner is None:
            raise RuntimeError("this JobBench job already has a judge running")
        if pinned.exists() and read_json(pinned) != profile:
            raise RuntimeError("judge profile differs from this job's pinned profile; keep the original model/CLI settings")
        atomic_json(pinned, profile)
        print(f"[jobbench-judge] local ChatGPT login · {profile['model']} · {profile['reasoning_effort']} · watch={watch}", flush=True)
        stop = threading.Event()
        def interrupt(signum, frame):
            stop.set()
            raise KeyboardInterrupt
        previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not stop.is_set():
                summary = judge_pass(job, official, profile, timeout=timeout, retries=retries)
                if not watch:
                    if summary["pending_tasks"]:
                        raise RuntimeError("some tasks remain pending; inspect judge_summary.json and retry later")
                    return summary
                stop.wait(poll_seconds)
        except KeyboardInterrupt:
            print("[jobbench-judge] stopped; committed rubric results are resumable", flush=True)
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--reasoning-effort")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--timeout-per-rubric", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=2)
    a = p.parse_args()
    if a.poll_seconds < 1 or a.timeout_per_rubric < 1 or a.max_retries < 1:
        p.error("poll, timeout, and retries must be positive")
    run(a.job.resolve(), model=a.model, effort=a.reasoning_effort, watch=a.watch,
        poll_seconds=a.poll_seconds, timeout=a.timeout_per_rubric, retries=a.max_retries)


if __name__ == "__main__":
    main()
