"""Read-only BrowseComp100 Codex monitor; no claims, judging or serving calls."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
from pathlib import Path

from automation_bench_eval.monitor_streamlit import latest_summary
from browsecomp_bench_eval.dataset import DatasetError
from job_bench_eval.monitor_streamlit import (
    duration, elapsed, image_previews, read_bytes, read_object, read_text, safe_child, trajectory,
)
from shared_runtime.storage import safe_name
from terminal_bench_eval import terminal_bench_monitor_streamlit as tb
from terminal_bench_eval.codex_r2eact_artifacts import GATEWAY_RECEIPTS_JSONL_FILENAME, MEMORY_FILENAME

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = PROJECT / "eval/browsecomp100_codex/jobs"
SCHEMA = "browsecomp100_codex_pool_v1"
TERMINAL = {"completed", "timeout", "errored"}


def discover_jobs(root):
    jobs = []
    for marker in root.glob("*/pool.json"):
        if tb._has_symlink_component(marker):
            continue
        try:
            if read_object(marker).get("schema") == SCHEMA:
                jobs.append((marker.stat().st_mtime, marker.parent.name))
        except OSError:
            continue
    return [name for _, name in sorted(jobs, reverse=True)]


def binary_reward(result):
    value = result.get("reward")
    if (result.get("status") == "completed" and result.get("judge_status") == "completed"
        and type(value) in (int, float) and value in (0, 1)):
        return int(value)
    return None


def sample_snapshot(job, task):
    task_id = safe_name(task["task_id"])
    cell = safe_child(job, "cells/" + task_id)
    state = read_object(cell / "state.json")
    number = state.get("execution", 0)
    # A requeued sample must not resurrect its previous answer/reward.
    execution = (safe_child(cell, f"executions/{number:04d}") if state.get("status") != "pending"
        and type(number) is int and number > 0 else None)
    metadata = read_object(execution / "metadata.json") if execution else {}
    result = read_object(execution / "result.json") if execution else {}
    terminal = result.get("status") in TERMINAL
    status = result["status"] if terminal else state.get("status", "pending")
    # Never even open judge artifacts before an agent terminal result commit.
    judge = read_object(execution / "judge.json") if terminal else {}
    telemetry = tb._receipt_summary(execution / "gateway" / GATEWAY_RECEIPTS_JSONL_FILENAME) if execution else {}
    return dict(task_id=task_id, state=state, execution=execution, metadata=metadata,
        result=result, terminal=terminal, status=status, judge=judge,
        judge_status=result.get("judge_status", "pending") if terminal and status == "completed" else "—",
        telemetry=telemetry, reward=binary_reward(result),
        elapsed=elapsed({**state, **metadata, "status": status}, result))


def job_snapshot(job):
    manifest = read_object(job / "pool.json")
    tasks = manifest.get("tasks")
    if (manifest.get("schema") != SCHEMA or not isinstance(tasks, list) or not tasks
        or len(tasks) > 100 or any(not isinstance(t, dict) or not isinstance(t.get("task_id"), str) for t in tasks)):
        raise ValueError("No readable committed BrowseComp100 pool manifest")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task identities in BrowseComp100 manifest")
    return manifest, [sample_snapshot(job, task) for task in tasks]


def table_rows(samples, *, manifest=None, include_answers=False):
    rows = [dict(task=s["task_id"], status=s["status"], reward=s["reward"], judge=s["judge_status"],
        execution=s["state"].get("execution", 0), rounds=s["telemetry"].get("latest_round") or None,
        input_tokens=s["telemetry"].get("input_tokens"), output_tokens=s["telemetry"].get("output_tokens"),
        tool_calls=s["telemetry"].get("tool_calls"), elapsed=duration(s["elapsed"]),
        worker=s["state"].get("worker", "—")) for s in samples]
    if include_answers:
        for i, (row, sample) in enumerate(zip(rows, samples)):
            answers = answer_details(manifest or {}, sample)
            row["final answer"] = answers["final_answer"]
            from browsecomp_bench_eval.codex_policy import confidence_receipt
            row["confidence %"] = (confidence_receipt(answers["final_answer"])["confidence_percent"]
                if answers["final_answer"] else None)
            row["reward source"] = sample["result"].get("reward_source", "Qwen" if sample["reward"] is not None else "—")
            row["Codex review"] = sample["result"].get("codex_review_status", "—")
            row["ground truth"] = answers["ground_truth"] or (
                "Reference unavailable" if answers["reference_error"] else None)
            front = ("task", "status", "reward", "final answer", "ground truth")
            rows[i] = {**{key: row[key] for key in front}, **row}
    return rows


def overview_values(samples):
    counts = Counter(s["status"] for s in samples)
    rewards = [s["reward"] for s in samples if s["reward"] is not None]
    return [len(samples), sum(s["terminal"] for s in samples), counts["running"] + counts["starting"],
        counts["pending"], len(rewards), f"{sum(rewards) / len(rewards):.1%}" if rewards else "—"]


@lru_cache(maxsize=128)
def _reference(path, sha256, sample_id):
    from browsecomp_bench_eval.dataset import EncryptedBrowseCompDataset, grading_reference
    if tb._has_symlink_component(Path(path)):
        raise ValueError("Dataset reference cannot be a symlink")
    dataset = EncryptedBrowseCompDataset.from_csv(path, expected_sha256=sha256)
    row = next((row for row in dataset.rows if row.sample_id == sample_id), None)
    if row is None:
        raise ValueError("Sample not found in the frozen dataset")
    return grading_reference(row).answer


def ground_truth(manifest, sample):
    if not sample["terminal"]:
        return None
    value = sample["judge"].get("ground_truth")
    if isinstance(value, str):
        return value
    # A completed answer is visible even while its independent judge is pending.
    # Only terminal samples (detail or completed table) may decrypt a reference.
    path, digest = manifest.get("dataset_path"), manifest.get("dataset_sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    return _reference(path, digest, sample["task_id"])


def answer_details(manifest, sample):
    """Both page sections share the terminal-commit gate, including after requeue."""
    values = dict(final_answer=None, ground_truth=None, reference_error=None)
    if not sample["terminal"] or sample["execution"] is None:
        return values
    values["final_answer"] = read_text(sample["execution"] / "final_answer.txt") or None
    try:
        values["ground_truth"] = ground_truth(manifest, sample)
    except (OSError, ValueError, DatasetError) as exc:
        values["reference_error"] = str(exc)
    return values


def render_answers(st, manifest, sample):
    # Match the original BrowseComp layout: visible above trajectory, not
    # hidden in a tab. Judging may still be pending after generation completes.
    if not sample["terminal"]:
        return
    values = answer_details(manifest, sample)
    st.subheader("Final answer")
    st.code(values["final_answer"] or "No final answer was saved.", language=None, wrap_lines=True)
    st.subheader("Ground truth")
    if values["reference_error"]:
        st.warning(f"Frozen reference unavailable: {values['reference_error']}")
    else:
        st.code(values["ground_truth"] if values["ground_truth"] is not None
            else "Reference not available yet.", language=None, wrap_lines=True)


def render_overview(st, job):
    manifest, samples = job_snapshot(job)
    for column, label, value in zip(st.columns(6),
        ["Total", "Finished", "Running", "Pending", "Scored", "Accuracy"], overview_values(samples)):
        column.metric(label, value)
    st.caption("Accuracy = sum(reward) / 已判分样本数；待判分和执行异常不记作 0 分。")
    for title, selected in [("Completed samples", [s for s in samples if s["terminal"]]),
        ("Running samples", [s for s in samples if s["status"] in {"running", "starting"}])]:
        st.subheader(title)
        if selected:
            st.dataframe(table_rows(selected, manifest=manifest,
                include_answers=title == "Completed samples"), hide_index=True, width="stretch")
        else:
            st.info("No completed sample yet." if title == "Completed samples" else "No running sample.")
    pending = [s for s in samples if s["status"] == "pending"]
    if pending:
        with st.expander("Pending samples"):
            st.dataframe(table_rows(pending), hide_index=True, width="stretch")


def render_sample(st, job):
    manifest, samples = job_snapshot(job)
    priority = {"running": 0, "starting": 0, "completed": 1, "timeout": 2, "errored": 2, "pending": 3}
    lookup = {s["task_id"]: s for s in sorted(samples, key=lambda s: (priority.get(s["status"], 4), s["task_id"]))}
    selected = st.selectbox("Active sample", list(lookup),
        format_func=lambda key: f"{key} · {lookup[key]['status']}", key=f"bc100-sample-{job}")
    paused = st.checkbox("暂停轨迹刷新", key=f"bc100-pause-{job}")
    sample = lookup[selected]
    identity = (str(job), selected, str(sample["execution"]))
    cache_key = "browsecomp100-detail-snapshot"
    cached = st.session_state.get(cache_key)
    if not paused or cached is None or cached[0] != identity:
        steps = trajectory(sample["execution"]) if sample["execution"] else []
        st.session_state[cache_key] = (identity, sample, steps)
    _, sample, steps = st.session_state[cache_key]
    telemetry = sample["telemetry"]
    values = [sample["status"], telemetry.get("latest_round") or "—", telemetry.get("input_tokens"),
        telemetry.get("output_tokens"), telemetry.get("tool_calls"), duration(sample["elapsed"])]
    for column, label, value in zip(st.columns(6),
        ["Status", "Round", "Input tokens", "Output tokens", "Tool calls", "Elapsed"], values):
        column.metric(label, "—" if value is None else value)
    with st.expander("Question", expanded=True):
        question = read_text(safe_child(job, f"inputs/{selected}/TASK_INSTRUCTIONS.txt"))
        st.code(question or "No committed question file.", language=None, wrap_lines=True)
    if not sample["execution"]:
        st.info("This sample has not started.")
        return
    execution = sample["execution"]
    render_answers(st, manifest, sample)
    tabs = st.tabs(["Trajectory", "Judge", "Memory / logs"])
    with tabs[0]:
        if steps:
            page = st.selectbox("Rounds", list(range((len(steps) + 9) // 10)), key=f"bc100-page-{identity}",
                format_func=lambda p: f"{steps[p*10]['round_index']} → {steps[min(p*10+9, len(steps)-1)]['round_index']}")
            visible = steps[page * 10:page * 10 + 10]
            previews = image_previews(execution, {s["round_index"] for s in visible})
            for index, step in enumerate(visible):
                tb._render_step(st, {**step, "display_label": f"Round {step['round_index']}",
                    "image_previews": previews.get(step["round_index"], [])}, expanded=index == 0)
        else:
            st.info("No committed model round yet; startup or generation may still be in flight.")
        with st.expander("Codex live events"):
            st.code(read_text(execution / "codex-events.jsonl", tail=True) or "No CLI events yet.", language=None)
    with tabs[1]:
        if not sample["terminal"]:
            st.info("Judging starts after generation finishes.")
        else:
            cols = st.columns(2)
            cols[0].metric("Reward", "—" if sample["reward"] is None else sample["reward"])
            cols[1].metric("Judge status", sample["judge_status"])
            st.code(sample["judge"].get("response_text") or "No committed judge response.", language=None, wrap_lines=True)
            review = read_object(execution / "codex-review.json")
            if review:
                st.markdown("**Codex review**")
                st.write(review.get("decision") or review.get("status", "pending"))
                st.code(review.get("reason") or review.get("error") or "Review pending.",
                    language=None, wrap_lines=True)
                if review.get("primary_sources"):
                    st.json(review["primary_sources"], expanded=False)
            for key in ("error", "judge_error"):
                if sample["result"].get(key):
                    st.warning(sample["result"][key])
    with tabs[2]:
        summary = latest_summary(execution)
        if summary:
            with st.expander("Latest compact summary"):
                st.code(summary, language=None, wrap_lines=True)
        with st.expander("Context / compression"):
            st.json(telemetry.get("latest_compression", {}), expanded=False)
        with st.expander("Experiment settings"):
            config = manifest.get("config", {})
            st.json({k: config.get(k) for k in ("harness_name", "harness_mode", "native_compaction_text_summary",
                "residual_checkpoint", "working_memory_turns", "serving_eviction_policy", "timeout_seconds")}, expanded=False)
        for label, path in [("memory.jsonl", execution / "gateway" / MEMORY_FILENAME),
            ("codex-stderr.log", execution / "codex-stderr.log"), ("runner.log", execution / "runner.log")]:
            with st.expander(label):
                st.code(read_text(path, tail=True) or "No records yet.", language=None, wrap_lines=True)
        for name, path in [("memory.jsonl", execution / "gateway" / MEMORY_FILENAME),
            ("trajectory.jsonl", execution / "trajectory.jsonl")]:
            data = read_bytes(path, limit=32 * 1024 * 1024, exact=True)
            if data:
                st.download_button(f"Download {name}", data, file_name=name, key=f"bc100-download-{identity}-{name}")


def main():
    import streamlit as st
    st.set_page_config(page_title="BrowseComp100 · Codex live monitor", layout="wide")
    st.title("BrowseComp100 · Codex live monitor")
    root = Path(os.environ.get("BROWSECOMP_CODEX_MONITOR_JOBS_ROOT", str(DEFAULT_JOBS))).expanduser()
    names = discover_jobs(root)
    if not names:
        st.info("No BrowseComp100 Codex pool found.")
        return
    selected = st.selectbox("Job", names)
    job = safe_child(root, selected)
    def overview():
        try:
            render_overview(st, job)
        except (OSError, ValueError) as exc:
            st.warning(f"Cannot read committed pool: {exc}")
    def detail():
        try:
            render_sample(st, job)
        except (OSError, ValueError) as exc:
            st.warning(f"Cannot read committed sample: {exc}")
    st.fragment(run_every=30)(overview)()
    st.fragment(run_every=60)(detail)()


if __name__ == "__main__":
    main()
