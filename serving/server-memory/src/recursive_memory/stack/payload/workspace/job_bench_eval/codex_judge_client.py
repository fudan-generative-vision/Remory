"""Narrow chat-completions-shaped adapter to a locally authenticated Codex CLI.

No auth files are read/copied. Only the controller launches this process; pool
workers and the model being evaluated never receive the judge's login.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

from shared_runtime.storage import atomic_json, read_json, utcnow


ADAPTER_VERSION = "jobbench_codex_login_judge_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def login_environment():
    # In particular, never inherit API keys, local gateway URLs, or task creds.
    allowed = {"PATH", "HOME", "CODEX_HOME", "USER", "LOGNAME", "SHELL", "TMPDIR",
        "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k in allowed}


def resolve_profile(model=None, effort=None):
    binary = shutil.which(os.environ.get("JOBBENCH_JUDGE_CODEX", "codex"))
    if not binary:
        raise RuntimeError("local Codex CLI is missing")
    env = login_environment()
    help_text = subprocess.check_output([binary, "exec", "--help"], env=env, text=True, timeout=15)
    if "--ignore-user-config" not in help_text:
        raise RuntimeError("judge needs a Codex CLI with --ignore-user-config (local 0.153.4 supports it)")
    auth = subprocess.run([binary, "login", "status"], env=env, capture_output=True, text=True, timeout=15)
    if auth.returncode or "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("run codex login on the judge host using ChatGPT; API-key login is not accepted here")
    path = Path(env.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    config = tomllib.loads(path.read_text()) if path.exists() else {}
    model = model or os.environ.get("JOBBENCH_JUDGE_MODEL") or config.get("model")
    effort = effort or os.environ.get("JOBBENCH_JUDGE_REASONING_EFFORT") or config.get("model_reasoning_effort", "medium")
    if not model:
        raise RuntimeError("set --model or JOBBENCH_JUDGE_MODEL to pin a judge model")
    available = subprocess.check_output([binary, "features", "list"], env=env, text=True, timeout=15)
    available = {line.split()[0] for line in available.splitlines() if line.split()}
    disabled = sorted(available.intersection({"apps", "plugins", "remote_plugin", "hooks",
        "shell_tool", "unified_exec", "view_image", "browser_use", "browser_use_external",
        "computer_use", "image_generation", "multi_agent", "multi_agent_v2", "memories",
        "goals", "code_mode", "code_mode_host", "skill_search", "workspace_dependencies",
        "tool_suggest", "shell_snapshot", "sleep_tool", "unbounded_connection_retries"}))
    return {"adapter": ADAPTER_VERSION, "backend": "codex_chatgpt", "model": model,
        "reasoning_effort": effort, "binary": str(Path(binary).resolve()),
        "codex_version": subprocess.check_output([binary, "--version"], env=env, text=True, timeout=15).strip(),
        "disabled_features": disabled, "auth": "saved_chatgpt_login",
        "temperature": "CLI-managed; upstream temperature=0 is not exposed by codex exec"}


def output_schema(count):
    return {"type": "object", "additionalProperties": False,
        "required": ["criteria_results", "rubric_passed", "overall_reasoning"],
        "properties": {
            "criteria_results": {"type": "array", "minItems": count, "maxItems": count,
                "items": {"type": "object", "additionalProperties": False,
                    "required": ["index", "passed", "reasoning", "evidence"],
                    "properties": {"index": {"type": "integer"}, "passed": {"type": "boolean"},
                        "reasoning": {"type": "string"}, "evidence": {"type": "string"}}}},
            "rubric_passed": {"type": "boolean"}, "overall_reasoning": {"type": "string"}}}


def validate_answer(value, count):
    criteria = value.get("criteria_results", [])
    if len(criteria) != count or any(type(c.get("index")) is not int or c["index"] != i
        or type(c.get("passed")) is not bool
        or not all(isinstance(c.get(k), str) for k in ("reasoning", "evidence"))
        for i, c in enumerate(criteria)):
        raise ValueError("judge response has missing, reordered, or malformed criteria")
    if type(value.get("rubric_passed")) is not bool or value["rubric_passed"] != all(c["passed"] for c in criteria):
        raise ValueError("rubric_passed disagrees with its criteria")
    if not isinstance(value.get("overall_reasoning"), str):
        raise ValueError("judge response is missing overall_reasoning")
    return value


def split_messages(messages, directory):
    """Preserve the upstream system/user text and attach its exact image bytes."""
    system, text, images = [], [], []
    for message in messages:
        if message["role"] == "system":
            system.append(message["content"])
            continue
        if message["role"] != "user":
            raise ValueError("unexpected official judge message role")
        content = message["content"]
        if isinstance(content, str):
            text.append(content)
            continue
        for part in content:
            if part["type"] == "text":
                text.append(part["text"])
            elif part["type"] == "image_url":
                header, encoded = part["image_url"]["url"].split(",", 1)
                mime = header.removeprefix("data:").removesuffix(";base64")
                extension = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}.get(mime)
                if not extension or header != f"data:{mime};base64":
                    raise ValueError("judge only accepts upstream inline image attachments")
                path = directory / f"image-{len(images)}.{extension}"
                path.write_bytes(base64.b64decode(encoded, validate=True))
                images.append(path)
            else:
                raise ValueError("unsupported official judge content part")
    return "\n\n".join(system), "\n".join(text), images


class CodexJudgeClient:
    def __init__(self, profile, artifacts, criterion_count):
        self.profile, self.artifacts, self.count = profile, Path(artifacts), criterion_count
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, messages, timeout, **ignored):
        if model != self.profile["model"]:
            raise ValueError("judge model must match the pinned profile")
        self.artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
        call = Path(tempfile.mkdtemp(prefix="call-", dir=self.artifacts))
        system, prompt, images = split_messages(messages, call)
        atomic_json(call / "request.json", {"model": model, "messages": messages,
            "upstream_parameters": ignored, "profile": self.profile})
        atomic_json(call / "schema.json", output_schema(self.count))
        command = [self.profile["binary"], "exec", "--ignore-user-config", "--ignore-rules",
            "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--color", "never",
            "-m", model, "--output-schema", str(call / "schema.json"),
            "-o", str(call / "answer.json"), "-c", "model_provider=\"openai\"",
            "-c", "forced_login_method=\"chatgpt\"", "-c", "approval_policy=\"never\"",
            "-c", "project_doc_max_bytes=0", "-c", "web_search=\"disabled\"",
            "-c", "features.skip_host_skill_discovery=true",
            "-c", "base_instructions=" + json.dumps(system),
            "-c", "model_reasoning_effort=" + json.dumps(self.profile["reasoning_effort"])]
        for feature in self.profile["disabled_features"]:
            command.extend(["--disable", feature])
        for path in images:
            command.extend(["--image", str(path)])
        command.append("-")
        started = time.monotonic()
        metadata = {"status": "running", "started_at": utcnow(), "profile": self.profile, "request_sha256": digest(messages)}
        process = None
        try:
            # A clean local cwd avoids importing benchmark or project instructions.
            with tempfile.TemporaryDirectory(prefix="jobbench-judge-cwd-") as cwd, \
                (call / "events.jsonl").open("w") as stdout, (call / "stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=cwd, env=login_environment(), stdin=subprocess.PIPE,
                    stdout=stdout, stderr=stderr, text=True, start_new_session=True)
                process.communicate(prompt, timeout=timeout)
            if process.returncode:
                raise RuntimeError(f"codex judge exited {process.returncode}; see {call / 'stderr.log'}")
            events = [json.loads(line) for line in (call / "events.jsonl").read_text().splitlines() if line.strip()]
            completed = [e for e in events if e.get("type") == "turn.completed"]
            if not completed or any(e.get("type") in {"turn.failed", "error"} for e in events):
                raise RuntimeError("Codex did not commit a successful judge turn")
            if any(e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"} for e in events):
                raise RuntimeError("judge unexpectedly used a tool; score not accepted")
            value = validate_answer(read_json(call / "answer.json"), self.count)
            metadata.update(status="completed", usage=[e.get("usage") for e in completed])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])
        except BaseException as exc:
            metadata.update(status="pending", error=f"{type(exc).__name__}: {exc}")
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
            metadata["elapsed_seconds"] = time.monotonic() - started
            atomic_json(call / "metadata.json", metadata)
