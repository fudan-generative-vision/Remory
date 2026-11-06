"""Controller-only Qwen -> Codex appeals for the two BrowseComp100 pools.

Failed judging stays pending. A completed appeal is authoritative; inconclusive
appeals retain the Qwen rejection, matching the legacy monitor's policy.
Generation workers never receive Codex credentials or gold/review artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .codex_appeal import build_codex_appeal_prompt, parse_codex_appeal_output
from .codex_appeal_contract import CODEX_APPEAL_OUTPUT_SCHEMA_PATH, CODEX_APPEAL_PROMPT_SHA256
from .codex_policy import confidence_receipt
from .dataset import EncryptedBrowseCompDataset, generation_sample, grading_reference
from job_bench_eval.codex_judge_client import resolve_profile, login_environment, digest
from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow

POLICY = "qwen_then_codex_review_all_rejections_v1"
SCHEMA = "browsecomp100_codex_review_v1"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "eval/browsecomp100_codex"


def review_profile(model, effort):
    profile = resolve_profile(model, effort)
    # Unlike a tools-free rubric judge, an appeal must actually retrieve public
    # evidence. The bundled V8 tool dispatcher is not a shell/filesystem tool.
    dispatch = {"code_mode", "code_mode_host"}
    profile["enabled_features"] = sorted(dispatch.intersection(profile["disabled_features"]))
    profile["disabled_features"] = [f for f in profile["disabled_features"] if f not in dispatch]
    profile.update(adapter="browsecomp100_codex_web_review_v2", web_search="live")
    return profile


def validate_review_events(events, verdict):
    completed = [e for e in events if e.get("type") == "turn.completed"]
    errors = []
    for e in events:
        item = e.get("item", {})
        if e.get("type") == "turn.failed":
            errors.append(e)
        elif e.get("type") == "error" and not str(e.get("message", "")).startswith("Reconnecting..."):
            errors.append(e)
        elif item.get("type") == "error" and not str(item.get("message", "")).startswith(
                "Falling back from WebSockets to HTTPS transport."):
            errors.append(e)
    # CLI reconnect/fallback diagnostics are recoverable IF a successful final
    # turn and real retrieval follow. Do not throw away a completed paid review.
    if not completed or errors:
        raise RuntimeError("Codex review had an execution/tool error; verdict not accepted")
    if any(e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "file_change"} for e in events):
        raise RuntimeError("Codex review used an unauthorized local tool")
    if verdict["decision"] != "inconclusive" and not any(
        e.get("type") == "item.completed" and e.get("item", {}).get("type") == "web_search"
        for e in events):
        raise RuntimeError("conclusive Codex appeal has no recorded web retrieval")
    return completed


def completed_call(call):
    events = [json.loads(line) for line in (call / "events.jsonl").read_text().splitlines() if line.strip()]
    verdict = asdict(parse_codex_appeal_output((call / "answer.json").read_bytes()))
    completed = validate_review_events(events, verdict)
    return verdict, [e.get("usage") for e in completed]


def qwen_reward(result, judge):
    value = result.get("qwen_reward")
    if type(value) is int and value in (0, 1):
        return value
    # Only recognize the old, committed local judgment, not an execution error.
    if type(judge.get("correct")) is bool:
        return int(judge["correct"])
    if result.get("judge_status") == "completed" and type(result.get("reward")) is int:
        if result["reward"] in (0, 1) and not result.get("codex_review_status"):
            return result["reward"]
    return None


def prepare_result(execution):
    """Caller holds cell ownership and judge lock. Migrate old Qwen receipts."""
    result = read_json(execution / "result.json")
    if result.get("status") != "completed":
        return result
    judge_path = execution / "judge.json"
    judge = read_json(judge_path) if judge_path.is_file() else {}
    value = qwen_reward(result, judge)
    if value is None:
        return result
    updated = {**result, "qwen_reward": value, "review_policy": POLICY,
               **confidence_receipt((execution / "final_answer.txt").read_text())}
    if value == 1:
        updated.update(reward=1, judge_status="completed", reward_source="Qwen",
                       codex_review_status="not_required")
    elif result.get("codex_review_status") != "completed":
        updated.update(reward=None, judge_status="review_pending", reward_source=None,
                       codex_review_status="pending")
    if updated != result:
        # Record the unmodified historical score exactly once, before migration.
        original = execution / "pre-codex-review-result.json"
        if not original.exists():
            atomic_json(original, result)
        atomic_json(execution / "result.json", updated)
    return updated


def review_reward(decision):
    if decision not in {"request_overturn", "uphold", "inconclusive"}:
        raise ValueError("unknown Codex review decision")
    return int(decision == "request_overturn")


def invoke_review(profile, evidence, artifacts, timeout):
    """Isolated logged-in Codex with web search only, never local task tools."""
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    for previous in sorted(artifacts.glob("call-*"), reverse=True):
        try:
            request = read_json(previous / "request.json")
            if (request.get("evidence") != evidence or request.get("profile") != profile
                    or request.get("prompt_sha256") != CODEX_APPEAL_PROMPT_SHA256):
                continue
            verdict, usage = completed_call(previous)
            atomic_json(previous / "metadata.json", {**read_json(previous / "metadata.json"),
                "status": "completed", "usage": usage, "revalidated_at": utcnow()})
            return verdict
        except (OSError, ValueError, RuntimeError):
            continue
    call = Path(tempfile.mkdtemp(prefix="call-", dir=artifacts))
    prompt = build_codex_appeal_prompt(**evidence)
    atomic_json(call / "request.json", {"evidence": evidence, "profile": profile,
        "prompt_sha256": CODEX_APPEAL_PROMPT_SHA256, "web_search": "live"})
    command = [profile["binary"], "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--color", "never",
        "-m", profile["model"], "--output-schema", str(CODEX_APPEAL_OUTPUT_SCHEMA_PATH.resolve()),
        "-o", str(call / "answer.json"), "-c", 'model_provider="openai"',
        "-c", 'forced_login_method="chatgpt"', "-c", 'approval_policy="never"',
        "-c", "project_doc_max_bytes=0", "-c", 'web_search="live"',
        "-c", "suppress_unstable_features_warning=true",
        "-c", "features.skip_host_skill_discovery=true",
        "-c", "model_reasoning_effort=" + json.dumps(profile["reasoning_effort"])]
    for feature in profile["disabled_features"]:
        command.extend(["--disable", feature])
    for feature in profile.get("enabled_features", []):
        command.extend(["--enable", feature])
    command.append("-")
    process = None
    started = time.monotonic()
    meta = {"status": "running", "started_at": utcnow(), "profile": profile}
    try:
        with tempfile.TemporaryDirectory(prefix="browsecomp100-review-") as cwd, \
            (call / "events.jsonl").open("w") as stdout, (call / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=cwd, env=login_environment(), stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, text=True, start_new_session=True)
            process.communicate(prompt, timeout=timeout)
        if process.returncode:
            raise RuntimeError(f"Codex review exited {process.returncode}; see {call / 'stderr.log'}")
        verdict, usage = completed_call(call)
        meta.update(status="completed", usage=usage)
        return verdict
    except BaseException as exc:
        meta.update(status="pending", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        meta["elapsed_seconds"] = time.monotonic() - started
        atomic_json(call / "metadata.json", meta)


def process_cell(job, task_id, manifest, rows, profile, *, prepare_only=False,
                 timeout=900, retry_seconds=300, invoke=invoke_review, sglang_url=None):
    cell = job / "cells" / safe_name(task_id)
    # Retaining ownership during judging prevents requeue from moving the
    # execution beneath a live reviewer. It does not hold the GPU/model lock.
    with lock(cell / "owner.lock", blocking=False) as owner:
        if owner is None:
            return "busy"
        state = read_json(cell / "state.json")
        if state.get("status") != "completed":
            return "skip"
        execution = cell / "executions" / f"{state['execution']:04d}"
        if sglang_url and not prepare_only:
            from .codex_pool import grade_execution
            grade_execution(job, execution, sglang_url=sglang_url)
        with lock(execution / "judge.lock", blocking=False) as judge_lock:
            if judge_lock is None:
                return "busy"
            result = prepare_result(execution)
            if result.get("qwen_reward") != 0:
                return "skip"
            if prepare_only:
                return "prepared"
            row = rows[task_id]
            evidence = dict(question=generation_sample(row).question,
                prediction=(execution / "final_answer.txt").read_text(),
                reference_answer=grading_reference(row).answer,
                qwen_reason=read_json(execution / "judge.json").get("response_text", ""))
            identity = digest({"evidence": evidence, "execution": str(execution),
                "dataset_sha256": manifest["dataset_sha256"], "profile": profile,
                "prompt_sha256": CODEX_APPEAL_PROMPT_SHA256})
            receipt_path = execution / "codex-review.json"
            receipt = read_json(receipt_path) if receipt_path.is_file() else {}
            if receipt.get("source_sha256") == identity and receipt.get("status") == "completed":
                verdict = receipt
            else:
                # Never keep an obsolete accepted result while a new review is pending.
                result.update(reward=None, judge_status="review_pending", reward_source=None,
                              codex_review_status="pending")
                atomic_json(execution / "result.json", result)
                if receipt.get("source_sha256") == identity and receipt.get("retry_after_epoch", 0) > time.time():
                    return "cooldown"
                base = dict(schema=SCHEMA, policy=POLICY, source_sha256=identity,
                    profile=profile, task_id=task_id, execution=state["execution"],
                    started_at=utcnow(), prompt_sha256=CODEX_APPEAL_PROMPT_SHA256)
                try:
                    verdict = invoke(profile, evidence, execution / "codex-review-calls", timeout)
                    # Validate injected/test clients identically to the CLI output.
                    verdict = asdict(parse_codex_appeal_output(json.dumps(verdict)))
                    atomic_json(receipt_path, {**base, **verdict, "status": "completed", "finished_at": utcnow()})
                except Exception as exc:
                    atomic_json(receipt_path, {**base, "status": "pending", "error": f"{type(exc).__name__}: {exc}",
                        "retry_after_epoch": time.time() + retry_seconds})
                    return "pending"
            reward = review_reward(verdict["decision"])
            result.update(reward=reward, judge_status="completed", codex_review_status="completed",
                codex_review_decision=verdict["decision"], codex_review_source_sha256=identity,
                reward_source="Qwen (Codex inconclusive)" if verdict["decision"] == "inconclusive" else "Codex")
            result.pop("judge_error", None)
            atomic_json(execution / "result.json", result)
            return f"{verdict['decision']} reward={reward}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--job-name", action="append", required=True)
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--effort", default="medium")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()
    if args.timeout <= 0 or args.poll_seconds <= 0:
        p.error("timeouts must be positive")
    from .codex_pool import validate_pool
    jobs = []
    for name in dict.fromkeys(args.job_name):
        job = args.runtime_root.resolve() / "jobs" / safe_name(name)
        manifest = validate_pool(job)
        ds = EncryptedBrowseCompDataset.from_csv(manifest["dataset_path"], expected_sha256=manifest["dataset_sha256"])
        jobs.append((job, manifest, {r.sample_id: r for r in ds.rows}))
    profile = review_profile(args.model, args.effort)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        # One controller can review both arms. Each cell also has its own lock.
        while True:
            # Mark all old Qwen rejections pending before making lengthy calls.
            for prepare in (True, False):
                for job, manifest, rows in jobs:
                    for task in manifest["tasks"]:
                        try:
                            outcome = process_cell(job, task["task_id"], manifest, rows, profile,
                                prepare_only=prepare, timeout=args.timeout, sglang_url=args.sglang_url)
                            if outcome not in {"skip", "busy", "prepared", "cooldown"}:
                                print(f"[browsecomp-review] {job.name} {task['task_id']}: {outcome}", flush=True)
                        except (OSError, ValueError) as exc:
                            print(f"[browsecomp-review] {task['task_id']}: pending ({type(exc).__name__}: {exc})", flush=True)
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("[browsecomp-review] stopped; unfinished reviews remain pending", flush=True)


if __name__ == "__main__":
    main()
