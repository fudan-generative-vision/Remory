"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


import hashlib


import io


import math


import os


import re


import stat


from datetime import datetime, timezone


from pathlib import Path


from terminal_bench_eval import terminal_bench_monitor_streamlit as tb


from terminal_bench_eval.codex_r2eact_artifacts import (
    GATEWAY_RECEIPTS_JSONL_FILENAME,
    GATEWAY_ROUNDS_FILENAME,
    GATEWAY_TOOL_RESPONSES_FILENAME,
    GATEWAY_IMAGE_RECEIPTS_FILENAME,
    MEMORY_FILENAME,
)


from terminal_bench_eval.codex_image_store import IMAGE_STORE_DIRNAME, MAX_IMAGE_BYTES


def read_object(path: Path) -> dict:
    return dict(
        tb._read_json(path, max_bytes=4 * 1024 * 1024, retries=1, use_last_good=False)
        or {}
    )


def safe_child(root: Path, name: str) -> Path:
    path = root / name
    if tb._has_symlink_component(path) or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("artifact path leaves its private sample directory")
    return path


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
        end = (
            datetime.fromisoformat(end_text.replace("Z", "+00:00"))
            if end_text
            else datetime.now(timezone.utc)
        )
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


def trajectory(execution: Path) -> list[dict]:
    gateway = execution / "gateway"
    steps = {
        step["round_index"]: step
        for step in tb._round_trace_steps(gateway / GATEWAY_ROUNDS_FILENAME)
    }
    for response in tb._tool_response_trace_steps(
        gateway / GATEWAY_TOOL_RESPONSES_FILENAME
    ):
        index = response["round_index"]
        if index not in steps:
            steps[index] = response
        else:
            steps[index].update(
                {
                    key: response[key]
                    for key in ("tool_responses", "tool_response_source", "output")
                    if key in response
                }
            )
    # Latest-first pagination; exact counters come from the receipt journal,
    # not the bounded 16 MiB display tail, so long runs do not reset to zero.
    return [steps[key] for key in sorted(steps, reverse=True)]


def image_previews(execution: Path, rounds: set[int]) -> dict[int, list[dict]]:
    """Render only hash-matching pixels; never fetch URLs or trust a mutable path."""
    from PIL import Image

    gateway = execution / "gateway"
    workspace = read_object(execution / "metadata.json").get("workspace")
    result: dict[int, list[dict]] = {}
    seen = set()
    for receipt in tb._complete_jsonl(
        gateway / GATEWAY_IMAGE_RECEIPTS_FILENAME, tail_bytes=tb.MAX_JSONL_BYTES
    ):
        index, digest = receipt.get("round_index"), receipt.get("sha256")
        if (
            receipt.get("schema") != "codex_r2eact_image_receipt_v1"
            or receipt.get("function_name") != "view_image"
            or not isinstance(receipt.get("call_id"), str)
            or type(index) is not int
            or index not in rounds
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
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
        if (
            isinstance(workspace, str)
            and Path(workspace).is_absolute()
            and preview["path"]
        ):
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
                    if (
                        img.format not in {"PNG", "JPEG", "WEBP", "GIF"}
                        or img.width * img.height > 25_000_000
                    ):
                        continue
                    width = min(img.width, 900)
                    img.verify()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                continue
            preview.update(data=data, width=width)
            break
        result.setdefault(index, []).append(preview)
    return result
