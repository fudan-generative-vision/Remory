"""Read-only AutomationBench pool/pilot monitor; no claims or judging."""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path

from automation_bench_eval.prompt_view import (
    CAPTURE_POLICY, FILENAME as PROMPT_FILENAME, digest, reconstruct_public_prompt,
)
from shared_runtime.monitor_streamlit import (
    duration, elapsed, read_bytes, read_object, read_text, safe_child, trajectory, image_previews,
)
from terminal_bench_eval import terminal_bench_monitor_streamlit as tb
from terminal_bench_eval.codex_r2eact_artifacts import (
    GATEWAY_RECEIPTS_JSONL_FILENAME, GATEWAY_ROUNDS_FILENAME,
    GATEWAY_TOOL_RESPONSES_FILENAME, MEMORY_FILENAME,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = PROJECT / "eval/automation_bench/jobs"
TERMINAL = {"completed", "step_limit", "timeout", "errored", "interrupted"}
SCOREABLE = {"completed", "step_limit", "timeout"}


def binary_reward(result):
    value = result.get("reward")
    return int(value) if result.get("status") in SCOREABLE and type(value) in (int, float) and value in (0, 1) else None


def fraction(value):
    return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else None


def official_tool_tail(execution):
    raw = read_bytes(execution / "tools.jsonl", limit=4 * 1024 * 1024, tail=True)
    lines = raw.splitlines()
    if raw and not raw.endswith(b"\n"):
        lines = lines[:-1]  # An in-flight write is not a committed observation.
    records = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict) and type(row.get("call")) is int and row["call"] > 0:
                records.append(row)
        except (ValueError, UnicodeDecodeError):
            pass
    return records


def sample_snapshot(job, config, task_id, *, cell_state=None):
    execution = safe_child(job, task_id)
    if cell_state is not None:
        cell = safe_child(job / "cells", task_id)
        number = cell_state.get("execution", 0)
        # An interrupted historical execution is not the current pending task.
        if cell_state.get("status") == "pending" or type(number) is not int or number < 1:
            execution = cell / "unstarted"
        else:
            execution = safe_child(cell / "executions", f"{number:04d}")
    metadata = read_object(execution / "metadata.json")
    result = read_object(execution / "result.json")
    terminal = result.get("status") in TERMINAL
    status = result["status"] if terminal else "running" if metadata else "starting" if execution.is_dir() else "pending"
    if cell_state is not None:
        status = (result["status"] if terminal and result["status"] != "interrupted"
                  else cell_state.get("status", "pending"))
        terminal = status in TERMINAL and bool(result)
    telemetry = tb._receipt_summary(execution / "gateway" / GATEWAY_RECEIPTS_JSONL_FILENAME) if execution.is_dir() else {}
    # Assertion specs and verification results must not be loaded at all until
    # a terminal result commit, even if the private verifier has already written.
    verification = read_object(execution / "verification.json") if terminal else {}
    tools = official_tool_tail(execution) if execution.is_dir() else []
    domain = (cell_state or {}).get("domain", config.get("domain", metadata.get("domain", "")))
    reward = binary_reward(result)
    return dict(task_id=task_id, domain=domain, execution=execution if execution.is_dir() else None,
        source_task_id=metadata.get("source_task_id", (cell_state or {}).get("source_task_id", task_id)),
        reasoning_effort=metadata.get("reasoning_effort", (cell_state or {}).get("reasoning_effort")),
        metadata=metadata, result=result, terminal=terminal, status=status,
        verification=verification, telemetry=telemetry, reward=reward,
        partial_credit=fraction(result.get("partial_credit")) if terminal else None,
        scored=reward is not None and domain != "simple" and result.get("included_in_benchmark_score", True) is not False,
        tools=tools, tool_calls=result.get("tool_calls") if terminal else tools[-1]["call"] if tools else None,
        worker_id=(cell_state or {}).get("worker_id"),
        elapsed=elapsed({**(cell_state or {}), **metadata, "status": status}, result),
        public=read_object(execution / "public-task.json") if execution.is_dir() else {})


def job_snapshot(job):
    config = read_object(job / "config.json")
    manifest = read_object(job / "pool.json")
    if manifest.get("schema") == "automationbench_pool_v1":
        config = manifest["config"]
        return config, [sample_snapshot(job, config, task["task_id"], cell_state={
            **{k: task[k] for k in ("source_task_id", "reasoning_effort") if k in task},
            **read_object(safe_child(job / "cells", task["task_id"]) / "state.json")}) for task in manifest["tasks"]]
    tasks = config.get("task_ids")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or not t for t in tasks):
        raise ValueError("No readable AutomationBench configuration")
    return config, [sample_snapshot(job, config, task_id) for task_id in tasks]


def discover_jobs(root):
    candidates = []
    for path in root.glob("*/config.json"):
        if tb._has_symlink_component(path):
            continue
        config = read_object(path)
        tasks = config.get("task_ids")
        if isinstance(tasks, list) and tasks and all(isinstance(t, str) and t for t in tasks):
            try:
                for task in tasks:
                    safe_child(path.parent, task)
                candidates.append((path.stat().st_mtime, path.parent.name))
            except (OSError, ValueError):
                pass
    return [name for _, name in sorted(candidates, reverse=True)]


def table_rows(samples):
    sweep = any(s.get("reasoning_effort") for s in samples)
    return [dict(task=s.get("source_task_id", s["task_id"]),
        **({"reasoning_effort": s.get("reasoning_effort")} if sweep else {}),
        status=s["status"], reward=s["reward"], partial_credit=s["partial_credit"],
        rounds=s["telemetry"].get("latest_round") or s["result"].get("model_responses"),
        input_tokens=s["telemetry"].get("input_tokens", s["result"].get("input_tokens")),
        output_tokens=s["telemetry"].get("output_tokens", s["result"].get("output_tokens")),
        tool_calls=s["tool_calls"], elapsed=duration(s["elapsed"]), worker=s.get("worker_id")) for s in samples]


def effort_rows(samples):
    rows = []
    for effort in ("low", "medium", "xhigh"):
        group = [s for s in samples if s.get("reasoning_effort") == effort]
        if not group:
            continue
        scores = [s["reward"] for s in group if s["scored"]]
        rows.append(dict(reasoning_effort=effort, total=len(group),
            finished=sum(s["terminal"] for s in group), scored=len(scores),
            correct=sum(scores), accuracy=sum(scores) / len(scores) if scores else None))
    return rows


def render_overview(st, job):
    config, samples = job_snapshot(job)
    counts = Counter(s["status"] for s in samples)
    scored = [s["reward"] for s in samples if s["scored"]]
    values = [len(samples), sum(s["terminal"] for s in samples), counts["running"] + counts["starting"],
              counts["pending"], len(scored), f"{sum(scored) / len(scored):.1%}" if scored else "—"]
    for column, label, value in zip(st.columns(6), ["Total", "Finished", "Running", "Pending", "Scored", "Accuracy"], values):
        column.metric(label, value)
    st.caption("Accuracy = sum(reward) / scored samples；执行异常与 simple 基础题不计入。")
    comparison = effort_rows(samples) if config.get("reasoning_sweep") else []
    if comparison:
        st.dataframe(comparison, hide_index=True, width="stretch")
    for title, selected in [("Completed samples", [s for s in samples if s["terminal"]]),
                            ("Running samples", [s for s in samples if s["status"] in {"running", "starting"}])]:
        st.subheader(title)
        if selected:
            st.dataframe(table_rows(selected), hide_index=True, width="stretch")
        else:
            st.info("No completed sample yet." if title == "Completed samples" else "No running sample.")
    pending = [s for s in samples if s["status"] == "pending"]
    if pending:
        with st.expander("Pending samples"):
            st.dataframe(table_rows(pending), hide_index=True, width="stretch")


def assertion_rows(verification):
    return [dict(assertion=row.get("type", ""), passed=row.get("passed"), excluded=row.get("excluded", False),
                 parameters=json.dumps(row.get("params", {}), ensure_ascii=False))
            for row in verification.get("assertions", []) if isinstance(row, dict)]


def latest_summary(execution):
    gateway = execution / "gateway"
    for name in ("codex-residual-wm-summary.json", "codex-native-compact-summary.json"):
        record = read_object(gateway / name)
        summary = record.get("summary", record)
        if isinstance(summary, dict) and isinstance(summary.get("text"), str):
            return summary["text"]
    return None


def task_instructions_view(sample):
    execution = sample.get("execution")
    record = read_object(execution / "gateway" / PROMPT_FILENAME) if execution else {}
    if record:
        text = record.get("text")
        if (record.get("schema") != CAPTURE_POLICY or not isinstance(text, str)
            or record.get("text_sha256") != digest(text)):
            return {"error": "Saved template input is invalid; not displaying a substitute."}
        return {"text": text, "caption":
            "Exact initial model input · system + user + tool definitions · includes assistant generation prefix"}
    if sample["metadata"].get("initial_prompt_capture") == CAPTURE_POLICY:
        return {"error": ("No initial model input was captured for this execution." if sample["terminal"]
                          else "Waiting for the initial templated model input to be committed.")}
    if not sample["public"].get("prompt"):
        return {"error": "Task instructions will appear when the sample starts."}
    try:
        policy = read_object(execution / "gateway" / "qwen-reasoning-policy.json") if execution else {}
        rendered = reconstruct_public_prompt(sample["metadata"], sample["public"], policy)
    except (OSError, ValueError) as exc:
        return {"error": "Cannot reconstruct legacy template input: " + str(exc)}
    return {"text": rendered["text"], "caption":
        "Reconstructed official system + user only; full Codex instructions/tool schemas were not saved. "
        + ("Recorded template hash verified." if rendered["template_verified"]
           else "Uses the current local template; this execution did not record its template hash.")}


def render_sample(st, job):
    config, samples = job_snapshot(job)
    priority = {"running": 0, "starting": 0, "completed": 1, "step_limit": 1, "timeout": 1,
                "errored": 2, "interrupted": 2, "pending": 3}
    lookup = {s["task_id"]: s for s in sorted(samples, key=lambda s: (priority[s["status"]], s["task_id"]))}
    selected = st.selectbox("Active sample", list(lookup),
        format_func=lambda key: f"{key} · {lookup[key]['status']}", key=f"ab-sample-{job}")
    paused = st.checkbox("暂停轨迹刷新", key=f"ab-pause-{job}")
    sample = lookup[selected]
    identity = (str(job), selected, str(sample["execution"]))
    cache_key = "automationbench-detail-snapshot"
    cached = st.session_state.get(cache_key)
    if not paused or cached is None or cached[0] != identity:
        steps = trajectory(sample["execution"]) if sample["execution"] else []
        st.session_state[cache_key] = (identity, sample, steps)
    _, sample, steps = st.session_state[cache_key]
    telemetry = sample["telemetry"]
    values = [sample["status"], telemetry.get("latest_round") or "—", telemetry.get("input_tokens"),
              telemetry.get("output_tokens"), sample["tool_calls"], duration(sample["elapsed"])]
    for column, label, value in zip(st.columns(6), ["Status", "Round", "Input tokens", "Output tokens", "Tool calls", "Elapsed"], values):
        column.metric(label, "—" if value is None else value)
    with st.expander("Task instructions"):
        if sample["result"].get("imported_from"):
            st.caption("Reused completed xhigh result; no new model execution for this cell.")
        if sample["metadata"].get("tool_profile"):
            st.caption("Tools: " + sample["metadata"]["tool_profile"])
        view = task_instructions_view(sample)
        if "text" in view:
            st.caption(view["caption"])
            st.code(view["text"], language=None, wrap_lines=True)
        else:
            st.info(view["error"])
    if not sample["execution"]:
        return
    execution = sample["execution"]
    tabs = st.tabs(["Trajectory", "Final answer", "Verification", "Memory / logs"])
    with tabs[0]:
        if steps:
            pages = list(range((len(steps) + 9) // 10))
            page = st.selectbox("Rounds", pages, key=f"ab-round-page-{identity}",
                format_func=lambda p: f"{steps[p*10]['round_index']} → {steps[min(p*10+9, len(steps)-1)]['round_index']}")
            visible = steps[page * 10:page * 10 + 10]
            previews = image_previews(execution, {s["round_index"] for s in visible})
            for index, step in enumerate(visible):
                tb._render_step(st, {**step, "display_label": f"Round {step['round_index']}",
                    "image_previews": previews.get(step["round_index"], [])}, expanded=index == 0)
        else:
            st.info("No committed model round yet; startup or generation may still be in flight.")
        with st.expander("Official tool activity (before context admission)"):
            for tool in reversed(sample["tools"][-10:]):
                st.caption(f"Call {tool['call']} · {tool.get('name', '')} · {tool.get('elapsed_seconds', 0):.3f}s")
                st.json(tool.get("arguments", {}), expanded=False)
                st.code(str(tool.get("output", "")), language=None, wrap_lines=True)
        with st.expander("Codex live events"):
            st.code(read_text(execution / "codex-events.jsonl", tail=True) or "No CLI events yet.", language=None)
    with tabs[1]:
        if sample["terminal"]:
            st.code(read_text(execution / "final_answer.txt") or "No final text answer was saved.", language=None, wrap_lines=True)
        else:
            st.info("Final answer appears after the sample finishes.")
    with tabs[2]:
        if not sample["terminal"]:
            st.info("Official assertions and results appear only after the sample finishes.")
        else:
            cols = st.columns(2)
            cols[0].metric("Reward", "—" if sample["reward"] is None else sample["reward"])
            cols[1].metric("Partial credit", "—" if sample["partial_credit"] is None else f"{sample['partial_credit']:.3f}")
            if sample["reward"] is None:
                st.warning("Execution is unscoreable; it is not counted as reward 0.")
            if sample["result"].get("error"):
                st.code(sample["result"]["error"], language=None, wrap_lines=True)
            rows = assertion_rows(sample["verification"])
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("No committed official assertion report.")
            for name in ("verification.json", "final-state.json"):
                data = read_bytes(safe_child(execution, name), limit=32 * 1024 * 1024, exact=True)
                if data:
                    st.download_button(f"Download {name}", data, file_name=name, key=f"ab-download-{identity}-{name}")
    with tabs[3]:
        summary = latest_summary(execution)
        if summary:
            with st.expander("Latest compact summary"):
                st.code(summary, language=None, wrap_lines=True)
        with st.expander("Context / compression"):
            st.json(telemetry.get("latest_compression", {}), expanded=False)
        with st.expander("Experiment settings"):
            st.json({k: config.get(k) for k in ("mode", "compressor", "checkpoint", "max_steps", "timeout_seconds", "official_commit")}, expanded=False)
        for label, path in [("memory.jsonl", execution / "gateway" / MEMORY_FILENAME),
                            ("codex-stderr.log", execution / "codex-stderr.log")]:
            with st.expander(label):
                st.code(read_text(path, tail=True) or "No records yet.", language=None, wrap_lines=True)
        # Assertion exceptions may include private expected values. Gate the
        # official simulator's stderr as strictly as the verification report.
        if sample["terminal"]:
            with st.expander("official-stderr.log"):
                st.code(read_text(execution / "official-stderr.log", tail=True) or "No records.", language=None, wrap_lines=True)


def main():
    import streamlit as st
    st.set_page_config(page_title="AutomationBench · Codex live monitor", layout="wide")
    st.title("AutomationBench · Codex live monitor")
    root = Path(os.environ.get("AUTOMATIONBENCH_MONITOR_JOBS_ROOT", str(DEFAULT_JOBS))).expanduser()
    names = discover_jobs(root)
    if not names:
        st.info("No AutomationBench pool found.")
        return
    selected = st.selectbox("Job", names)
    job = safe_child(root, selected)
    # Same cadence as JobBench; only the overview fragment refreshes every 30s.
    # Detail refreshes every 60s and can be frozen while reading long outputs.
    st.fragment(run_every=30)(lambda: render_overview(st, job))()
    st.fragment(run_every=60)(lambda: render_sample(st, job))()


if __name__ == "__main__":
    main()
