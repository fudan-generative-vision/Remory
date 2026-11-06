"""Read-only JobBench dashboard. Never claim tasks, acquire pool locks or judge."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from shared_runtime.storage import safe_name
from terminal_bench_eval import terminal_bench_monitor_streamlit as tb
from terminal_bench_eval.codex_r2eact_artifacts import (
    GATEWAY_RECEIPTS_JSONL_FILENAME, GATEWAY_ROUNDS_FILENAME,
    GATEWAY_TOOL_RESPONSES_FILENAME, GATEWAY_IMAGE_RECEIPTS_FILENAME, MEMORY_FILENAME,
)
from terminal_bench_eval.codex_image_store import IMAGE_STORE_DIRNAME, MAX_IMAGE_BYTES

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = PROJECT / "eval/job_bench/jobs"
TERMINAL = {"completed", "timeout", "errored"}


def read_object(path: Path) -> dict:
    return dict(tb._read_json(path, max_bytes=4 * 1024 * 1024,
                              retries=1, use_last_good=False) or {})


def safe_child(root: Path, name: str) -> Path:
    path = root / name
    if tb._has_symlink_component(path) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path leaves its private sample directory")
    return path


def visible_jobs(jobs_root: Path) -> list[Path]:
    """Monitor-only allowlist; never mutate pool manifests or task states."""
    config_path = jobs_root.parent / "monitor.json"
    allowed = None
    if config_path.exists() or config_path.is_symlink():
        if tb._has_symlink_component(config_path):
            raise ValueError("JobBench monitor visibility config cannot be a symlink")
        config = read_object(config_path)
        names = config.get("visible_jobs")
        if (config.get("schema") != "jobbench_monitor_visibility_v1"
                or not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)):
            raise ValueError("Invalid JobBench monitor visibility config")
        allowed = set(names)
    jobs = []
    for marker in jobs_root.glob("*/pool.json"):
        if allowed is not None and marker.parent.name not in allowed:
            continue
        if not tb._has_symlink_component(marker) and marker.is_file():
            jobs.append(marker)
    return sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True)


def read_bytes(path: Path, limit=256 * 1024, *, tail=False, exact=False) -> bytes:
    if tb._has_symlink_component(path):
        return b""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or (exact and info.st_size > limit):
                return b""
            if tail and info.st_size > limit:
                stream.seek(info.st_size - limit)
            return stream.read(limit)
    except OSError:
        return b""


def read_text(path: Path, *, tail=False) -> str:
    return read_bytes(path, tail=tail).decode("utf-8", errors="replace")


def elapsed(state, result) -> float | None:
    value = result.get("elapsed_seconds")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return max(0.0, value)
    try:
        start = datetime.fromisoformat(state["started_at"].replace("Z", "+00:00"))
        end_text = result.get("finished_at")
        end = datetime.fromisoformat(end_text.replace("Z", "+00:00")) if end_text else datetime.now(timezone.utc)
        if not result and state.get("status") != "running":
            return None
        return max(0.0, (end - start).total_seconds())
    except (KeyError, ValueError, TypeError):
        return None


def duration(value) -> str:
    if value is None:
        return "—"
    seconds = int(value)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def score(judge, *, terminal: bool):
    value = judge.get("reward")
    if terminal and judge.get("status") == "completed" and type(value) in (int, float):
        if math.isfinite(value) and 0 <= value <= 1:
            return float(value)
    return None


def sample_snapshot(job: Path, task: dict) -> dict:
    cell = safe_child(job, "cells/" + safe_name(task["task_id"]))
    state = read_object(cell / "state.json")
    index = state.get("execution", 0)
    if type(index) is not int or index < 0:
        index = 0
    execution = safe_child(cell, f"executions/{index:04d}") if index else None
    result = read_object(execution / "result.json") if execution else {}
    terminal = result.get("status") in TERMINAL
    judge = read_object(execution / "judge.json") if terminal else {}
    summary = tb._receipt_summary(execution / "gateway" / GATEWAY_RECEIPTS_JSONL_FILENAME) if execution else {}
    return dict(task=task, state=state, result=result, judge=judge, terminal=terminal,
                execution=execution, telemetry=summary, task_id=task["task_id"],
                status=result.get("status") if terminal else state.get("status", "pending"),
                reward=score(judge, terminal=terminal), elapsed=elapsed(state, result))


def job_snapshot(job: Path):
    manifest = read_object(job / "pool.json")
    if manifest.get("schema") != "jobbench_pool_v1":
        raise ValueError("No readable, committed JobBench pool manifest")
    samples = [sample_snapshot(job, task) for task in manifest.get("tasks", [])]
    return manifest, samples


def trajectory(execution: Path) -> list[dict]:
    gateway = execution / "gateway"
    steps = {step["round_index"]: step for step in tb._round_trace_steps(gateway / GATEWAY_ROUNDS_FILENAME)}
    for response in tb._tool_response_trace_steps(gateway / GATEWAY_TOOL_RESPONSES_FILENAME):
        index = response["round_index"]
        if index not in steps:
            steps[index] = response
        else:
            steps[index].update({key: response[key] for key in (
                "tool_responses", "tool_response_source", "output") if key in response})
    # Latest-first pagination; exact counters come from the receipt journal,
    # not the bounded 16 MiB display tail, so long runs do not reset to zero.
    return [steps[key] for key in sorted(steps, reverse=True)]


def table_rows(samples):
    return [dict(task=s["task_id"], status=s["status"],
                 reward=s["reward"], judge=(s["judge"].get("status", "pending")
                     if s["terminal"] else "—"),
                 execution=s["state"].get("execution", 0),
                 rounds=s["telemetry"].get("latest_round") or None,
                 input_tokens=s["telemetry"].get("input_tokens"),
                 output_tokens=s["telemetry"].get("output_tokens"),
                 elapsed=duration(s["elapsed"]), worker=s["state"].get("worker", "—"))
            for s in samples]


def image_previews(execution: Path, rounds: set[int]) -> dict[int, list[dict]]:
    """Render only hash-matching pixels; never fetch URLs or trust a mutable path."""
    from PIL import Image

    gateway = execution / "gateway"
    workspace = read_object(execution / "metadata.json").get("workspace")
    result: dict[int, list[dict]] = {}
    seen = set()
    for receipt in tb._complete_jsonl(gateway / GATEWAY_IMAGE_RECEIPTS_FILENAME,
                                     tail_bytes=tb.MAX_JSONL_BYTES):
        index, digest = receipt.get("round_index"), receipt.get("sha256")
        if (receipt.get("schema") != "codex_r2eact_image_receipt_v1"
                or receipt.get("function_name") != "view_image"
                or not isinstance(receipt.get("call_id"), str)
                or type(index) is not int or index not in rounds
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            continue
        identity = (index, receipt.get("call_id"), digest)
        if identity in seen:
            continue
        seen.add(identity)
        preview = {"path": str(receipt.get("path") or ""), "sha256": digest}
        candidates = []
        try:
            candidates.append(safe_child(gateway, f"{IMAGE_STORE_DIRNAME}/{digest}"))
        except ValueError:
            pass
        if isinstance(workspace, str) and Path(workspace).is_absolute() and preview["path"]:
            try:
                candidates.append(safe_child(Path(workspace), preview["path"]))
            except ValueError:
                pass
        for path in candidates:
            data = read_bytes(path, MAX_IMAGE_BYTES, exact=True)
            if not data or hashlib.sha256(data).hexdigest() != digest:
                continue
            try:
                with Image.open(io.BytesIO(data)) as img:
                    if img.format not in {"PNG", "JPEG", "WEBP", "GIF"} or img.width * img.height > 25_000_000:
                        continue
                    width = min(img.width, 900)
                    img.verify()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                continue
            preview.update(data=data, width=width)
            break
        result.setdefault(index, []).append(preview)
    return result


def render_overview(st, job):
    _, samples = job_snapshot(job)
    counts = Counter(sample["status"] for sample in samples)
    rewards = [s["reward"] for s in samples if s["reward"] is not None]
    columns = st.columns(6)
    for column, label, value in zip(columns,
            ["Completed", "Running", "Pending", "Error / timeout", "Judged", "Mean reward"],
            [counts["completed"], counts["running"], counts["pending"],
             counts["errored"] + counts["timeout"], len(rewards),
             f"{sum(rewards) / len(rewards):.3f}" if rewards else "—"]):
        column.metric(label, value)
    st.caption("Completed = generation finished. Mean reward uses judged samples only; pending judgments are not zero.")
    finished = [s for s in samples if s["status"] in TERMINAL]
    st.subheader("Completed samples")
    if finished:
        st.dataframe(table_rows(finished), hide_index=True, width="stretch")
    else:
        st.info("No completed sample yet.")
    running = [s for s in samples if s["status"] == "running"]
    if running:
        st.subheader("Running samples")
        st.dataframe(table_rows(running), hide_index=True, width="stretch")


def render_downloads(st, sample):
    execution = sample["execution"]
    if not sample["terminal"]:
        st.info("Final deliverables will appear after the sample finishes.")
        return
    files = read_object(execution / "deliverables.json").get("files", [])
    if not files:
        st.info("No committed deliverables.")
        return
    st.dataframe(files, hide_index=True, width="stretch")
    names = [row["path"] for row in files if isinstance(row.get("path"), str)]
    selected = st.selectbox("Download deliverable", names, key=f"download-{execution}")
    row = next(row for row in files if row.get("path") == selected)
    try:
        path = safe_child(execution / "model_output", selected)
        data = read_bytes(path, 32 * 1024 * 1024, exact=True)
        if hashlib.sha256(data).hexdigest() != row.get("sha256"):
            st.warning("File unavailable, over 32 MiB, or hash differs from its committed manifest.")
        else:
            st.download_button("Download", data, file_name=Path(selected).name, key=f"file-{execution}-{selected}")
    except ValueError as exc:
        st.error(str(exc))


def render_sample(st, job):
    manifest, samples = job_snapshot(job)
    priority = {"running": 0, "completed": 1, "timeout": 2, "errored": 2, "pending": 3}
    samples.sort(key=lambda s: (priority.get(s["status"], 4), s["task_id"]))
    lookup = {s["task_id"]: s for s in samples}
    if not lookup:
        return
    selected = st.selectbox("Active sample", list(lookup),
        format_func=lambda key: f"{key} · {lookup[key]['status']}", key=f"sample-{job.name}")
    paused = st.checkbox("暂停轨迹刷新", key=f"paused-{job.name}")
    sample = lookup[selected]
    cache_key = "jobbench-detail-snapshot"
    identity = (str(job), selected, str(sample["execution"]))
    cached = st.session_state.get(cache_key)
    if not paused or cached is None or cached[0] != identity:
        steps = trajectory(sample["execution"]) if sample["execution"] else []
        st.session_state[cache_key] = (identity, sample, steps)
    _, sample, steps = st.session_state[cache_key]
    telemetry = sample["telemetry"]
    columns = st.columns(5)
    for column, label, value in zip(columns,
            ["Status", "Round", "Input tokens", "Output tokens", "Elapsed"],
            [sample["status"], telemetry.get("latest_round") or "—",
             telemetry.get("input_tokens"), telemetry.get("output_tokens"), duration(sample["elapsed"])]):
        column.metric(label, "—" if value is None else value)
    with st.expander("Task instructions"):
        config = manifest.get("config", {})
        try:
            source = safe_child(Path(config["dataset_root"]), sample["task"]["task_folder"])
            st.text(read_text(source / "TASK_INSTRUCTIONS.txt") or "Task instructions unavailable.")
        except (KeyError, ValueError):
            st.warning("Task source path is unavailable or outside the dataset.")
    if not sample["execution"]:
        st.info("This sample has not started.")
        return
    tabs = st.tabs(["Trajectory", "Final answer / files", "Judge", "Memory / logs"])
    with tabs[0]:
        try:
            if any((sample["execution"] / "gateway" / name).stat().st_size > tb.MAX_JSONL_BYTES
                   for name in (GATEWAY_ROUNDS_FILENAME, GATEWAY_TOOL_RESPONSES_FILENAME)):
                st.caption("Trajectory shows the recent 16 MiB of each trace. Token totals still cover the complete run.")
        except OSError:
            pass
        if steps:
            pages = list(range((len(steps) + 9) // 10))
            page = st.selectbox("Rounds", pages,
                format_func=lambda p: f"{steps[p*10]['round_index']} → {steps[min(p*10+9, len(steps)-1)]['round_index']}",
                key=f"round-page-{identity}")
            visible = steps[page*10:page*10+10]
            previews = image_previews(sample["execution"], {s["round_index"] for s in visible})
            for index, step in enumerate(visible):
                step = {**step, "display_label": f"Round {step['round_index']}",
                        "image_previews": previews.get(step["round_index"], [])}
                tb._render_step(st, step, expanded=index == 0)
        else:
            st.info("No committed model round yet; startup or generation may still be in flight.")
        with st.expander("Codex live events"):
            st.code(read_text(sample["execution"] / "trajectory.jsonl", tail=True) or "No CLI events yet.", language=None)
    with tabs[1]:
        if sample["terminal"]:
            st.subheader("Final answer")
            st.code(read_text(sample["execution"] / "final_answer.txt") or "No final text answer was saved.", language=None, wrap_lines=True)
        render_downloads(st, sample)
    with tabs[2]:
        if not sample["terminal"]:
            st.info("Judge results appear only after the sample finishes.")
        elif sample["reward"] is None:
            st.info("Pending judgment — generation completion is not a correctness verdict.")
            if sample["judge"].get("error"):
                st.code(sample["judge"]["error"], language=None, wrap_lines=True)
        else:
            judge = sample["judge"]
            st.metric("Reward", f"{sample['reward']:.4f}")
            st.caption(f"Judge: {judge.get('judge_model', '—')} · score {judge.get('score', '—')} / {judge.get('max_score', '—')}")
            if judge.get("report"):
                try:
                    st.json(read_object(safe_child(sample["execution"], judge["report"])), expanded=False)
                except ValueError:
                    st.warning("Judge report path is outside this execution.")
    with tabs[3]:
        gateway = sample["execution"] / "gateway"
        summary = read_object(gateway / "codex-native-compact-summary.json")
        if summary.get("text"):
            st.subheader("Latest compact checkpoint")
            st.code(summary["text"], language=None, wrap_lines=True)
        with st.expander("Context / compression"):
            st.json(telemetry.get("latest_compression", {}), expanded=False)
        for label, path in [("memory.jsonl", gateway / MEMORY_FILENAME),
                            ("runner.log", sample["execution"] / "runner.log"),
                            ("codex.stderr.log", sample["execution"] / "codex.stderr.log")]:
            with st.expander(label):
                st.code(read_text(path, tail=True) or "No records yet.", language=None, wrap_lines=True)


def main():
    import streamlit as st
    st.set_page_config(page_title="JobBench · Codex live monitor", layout="wide")
    st.title("JobBench · Codex live monitor")
    jobs_root = Path(os.environ.get("JOBBENCH_MONITOR_JOBS_ROOT", str(DEFAULT_JOBS))).expanduser()
    try:
        jobs = visible_jobs(jobs_root)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not jobs:
        st.info("No committed JobBench pool found.")
        return
    selected = st.selectbox("Job", [p.parent.name for p in jobs])
    job = safe_child(jobs_root, selected)
    st.fragment(run_every=30)(lambda: render_overview(st, job))()
    st.fragment(run_every=60)(lambda: render_sample(st, job))()


if __name__ == "__main__":
    main()
