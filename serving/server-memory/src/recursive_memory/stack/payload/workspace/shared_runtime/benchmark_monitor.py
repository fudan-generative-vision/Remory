"""Read-only overview and existing task detail views for all benchmark pools."""
from __future__ import annotations

from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.request

from terminal_bench_eval import terminal_bench_monitor_streamlit as tb
from shared_runtime.monitor_streamlit import read_object, safe_child

LABELS = {"tb21": "Terminal-Bench 2.1", "jobbench": "JobBench",
          "browsecomp100": "BrowseComp100", "automationbench": "AutomationBench"}
SCHEMAS = {"tb21": tb.POOL_SCHEMA, "jobbench": "jobbench_pool_v1",
           "browsecomp100": "browsecomp100_codex_pool_v1", "automationbench": "automationbench_pool_v1"}


def pool_entries(prefix, config_path=None):
    """An optional operator allowlist keeps paused historical pools out of view."""
    prefix = Path(prefix).expanduser().resolve()
    if config_path:
        config = read_object(Path(config_path))
        if config.get("schema") != "shared_benchmark_monitor_v1" or not isinstance(config.get("pools"), list):
            raise ValueError("Invalid joint monitor configuration")
        candidates = config["pools"]
    else:
        candidates = []
        for benchmark in LABELS:
            root = prefix / "runs" / benchmark / "jobs"
            jobs = (tb._visible_jobs(root, tb._discover_jobs(root)) if benchmark == "tb21"
                    else [p.parent for p in sorted(root.glob("*/pool.json"))])
            candidates.extend({"benchmark": benchmark, "job": p.name} for p in jobs)
    entries, seen = [], set()
    for item in candidates:
        benchmark, name = item.get("benchmark"), item.get("job")
        if benchmark not in LABELS or not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", name):
            raise ValueError("Invalid benchmark or pool name")
        if (benchmark, name) in seen:
            raise ValueError("Duplicate pool in joint monitor")
        seen.add((benchmark, name))
        job = safe_child(prefix / "runs" / benchmark / "jobs", name)
        marker = read_object(job / "pool.json")
        if marker.get("schema") != SCHEMAS[benchmark]:
            if config_path:
                raise ValueError(f"Pool manifest unavailable: {name}")
            continue
        config = read_object(job / "config.json") if benchmark == "tb21" else marker.get("config", {})
        if benchmark == "tb21":
            residual = config.get("agents", [{}])[0].get("kwargs", {}).get("residual_compaction", True)
        elif benchmark == "automationbench":
            residual = config.get("compressor", "residual") == "residual"
        else:
            residual = config.get("residual_compaction", True)
        mode = "residual" if residual else "no residual"
        entries.append({"benchmark": benchmark, "name": name, "job": job,
                        "memory": mode, "label": item.get("label") or f"{LABELS[benchmark]} · {mode}"})
    return entries


def snapshot(entry):
    benchmark, job = entry["benchmark"], entry["job"]
    inherited = 0
    if benchmark == "tb21":
        marker = read_object(job / "pool.json")
        rows = [tb._pool_cell_projection(job, cell, expected_contract_hash=marker.get("config_contract_hash"))
                for cell in tb._pool_cells(marker)]
        tb._apply_pool_attempt_selection(rows, tb._pool_attempt_selection(job, marker))
        tb._apply_pool_task_selection(rows, tb._pool_task_selection(job, marker))
        rows = [r for r in rows if r.get("score_state") != "excluded"]
        samples = []
        for row in rows:
            if row["terminal"]:
                inherited += bool(read_object(job / "cells" / row["cell_id"] / "result.json").get("inherited_result"))
            samples.append({"task_id": row["task"], "status": row["status"], "terminal": row["terminal"],
                            "reward": row["reward"] if row.get("score_state") == "scored" else None,
                            "worker": row.get("worker", ""), "in_error": row.get("score_state") == "unscoreable"
                            or row["status"] in {"artifact error", "pool_error", "stale"},
                            "awaiting_judge": row["status"] == "verification_pending"})
    else:
        if benchmark == "jobbench":
            from job_bench_eval.monitor_streamlit import job_snapshot
        elif benchmark == "browsecomp100":
            from browsecomp_bench_eval.codex_monitor_streamlit import job_snapshot
        else:
            from automation_bench_eval.monitor_streamlit import job_snapshot
        _, samples = job_snapshot(job)
        samples = [{**s, "worker": s.get("state", {}).get("worker", s.get("worker_id", "")),
                    "reward": s["reward"] if s.get("scored", True) else None,
                    "in_error": s["status"] in {"errored", "interrupted"}
                    or (s["status"] == "timeout" and s["reward"] is None),
                    "awaiting_judge": s["status"] == "completed" and s["reward"] is None} for s in samples]
    counts = Counter(s["status"] for s in samples)
    rewards = [s["reward"] for s in samples if s["terminal"] and type(s["reward"]) in (float, int)
               and math.isfinite(s["reward"])]
    done = sum(s["terminal"] for s in samples)
    row = {"实验池": entry["name"], "基准": LABELS[benchmark], "记忆": entry["memory"],
           "进度": done / len(samples) if samples else 0, "总数": len(samples), "已结束": done,
           "运行中": counts["running"] + counts["starting"], "排队": counts["pending"] + counts["queued"],
           "待评分": sum(s["awaiting_judge"] for s in samples), "异常 / 失联": sum(s["in_error"] for s in samples),
           "已继承": inherited, "已评分": len(rewards), "评分均值": sum(rewards) / len(rewards) if rewards else None}
    active = [{"实验": entry["label"], "任务": s["task_id"], "状态": s["status"], "worker": s["worker"]}
              for s in samples if s["status"] in {"running", "starting", "stale", "verification_pending"}]
    return row, active


def gateway_snapshot(base_url):
    request = urllib.request.Request(base_url.rstrip("/") + "/queue/status", method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=2) as response:
        return json.loads(response.read(256 * 1024))


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True)
    rows = []
    for index, name, used, total, utilization, power in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        rows.append({"GPU": index, "型号": name, "显存 GiB": round(float(used) / 1024, 1),
                     "总显存 GiB": round(float(total) / 1024, 1), "利用率 %": float(utilization),
                     "功耗 W": None if power in {"[N/A]", "N/A"} else float(power)})
    return rows


def render_overview(st, entries, gateway_url):
    rows, active = [], []
    for entry in entries:
        try:
            row, running = snapshot(entry)
            rows.append(row)
            active.extend(running)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(f"{entry['label']}：暂时无法读取状态：{exc}")
    labels = ["实验池", "任务总数", "已结束", "运行中", "排队", "待评分"]
    values = [len(rows), *[sum(row[key] for row in rows) for key in ["总数", "已结束", "运行中", "排队", "待评分"]]]
    for col, label, value in zip(st.columns(6), labels, values):
        col.metric(label, value)
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "进度": st.column_config.ProgressColumn("进度", min_value=0, max_value=1, format="percent"),
        "评分均值": st.column_config.NumberColumn(format="%.3f")})
    left, right = st.columns(2)
    with left:
        st.subheader("共享推理队列")
        try:
            queue = gateway_snapshot(gateway_url)
            backends = queue.get("backends", [])
            st.dataframe([{"副本": b.get("name"), "状态": "就绪" if b.get("healthy") else "不可用",
                           "在途请求": b.get("active"), "观测负载": b.get("observed_load"), "容量": b.get("capacity")}
                          for b in backends], hide_index=True, width="stretch")
            if queue.get("groups"):
                st.caption("待调度分组")
                st.json(queue["groups"], expanded=False)
        except (OSError, ValueError) as exc:
            st.warning(f"共享队列状态不可用：{exc}")
    with right:
        st.subheader("本机 GPU")
        try:
            st.dataframe(gpu_snapshot(), hide_index=True, width="stretch")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            st.warning(f"GPU 状态不可用：{exc}")
    with st.expander(f"正在运行 / 等待 verifier 的任务（{len(active)}）", expanded=True):
        if active:
            st.dataframe(active, hide_index=True, width="stretch")
        else:
            st.info("暂无正在运行的任务。")


def render_detail(st, entry):
    job, benchmark = entry["job"], entry["benchmark"]
    try:
        if benchmark == "tb21":
            tb._render_job(st, jobs_root=job.parent, job_dir=job)
        else:
            if benchmark == "jobbench":
                from job_bench_eval import monitor_streamlit as monitor
            elif benchmark == "browsecomp100":
                from browsecomp_bench_eval import codex_monitor_streamlit as monitor
            else:
                from automation_bench_eval import monitor_streamlit as monitor
            monitor.render_overview(st, job)
            monitor.render_sample(st, job)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f"任务详情暂时不可用：{exc}")


def main():
    import streamlit as st
    st.set_page_config(page_title="Benchmark · 联合实时监控", layout="wide")
    st.title("Benchmark · 联合实时监控")
    st.caption("Qwen3.8-27B · Codex CLI · xhigh · residual / no residual")
    prefix = os.environ.get("R2EACT_MONITOR_PREFIX")
    if not prefix:
        st.error("R2EACT_MONITOR_PREFIX is required")
        return
    try:
        entries = pool_entries(prefix, os.environ.get("R2EACT_MONITOR_CONFIG"))
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    if not entries:
        st.info("暂无实验池。")
        return
    gateway_url = os.environ.get("R2EACT_MONITOR_GATEWAY", "http://127.0.0.1:30080")
    paused = st.toggle("暂停页面自动刷新", value=False, help="仅暂停页面刷新，实验继续运行。")
    st.fragment(run_every=None if paused else 10)(lambda: render_overview(st, entries, gateway_url))()
    st.divider()
    st.subheader("实验池详情")
    selected = st.selectbox("选择实验池", range(len(entries)), format_func=lambda i: entries[i]["label"], key="joint-selected-pool")
    st.caption(entries[selected]["name"])
    st.fragment(run_every=None if paused else 30)(lambda: render_detail(st, entries[selected]))()


if __name__ == "__main__":
    main()
