"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


import hashlib


import json


import os


import secrets


import shutil


import subprocess


import tempfile


from pathlib import Path


from .storage import atomic_json, read_json, utcnow, lock


PROJECT = Path(__file__).resolve().parent.parent


DEFAULT_MCP_VENV = PROJECT / "eval/document_venv"


DEFAULT_SPECFORGE = PROJECT


def toml_value(value):
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def document_servers(venv):
    return {
        "docling": {
            "command": str(venv / "bin/python"),
            "args": [
                str(venv / "bin/docling-mcp-server"),
                "--transport",
                "stdio",
                "conversion",
                "generation",
                "manipulation",
            ],
            "env_vars": [
                "DOCLING_MCP_CONVERSION_MODE",
                "DOCLING_DEVICE",
                "OMP_NUM_THREADS",
                "HF_HOME",
                "TMPDIR",
            ],
            "startup_timeout_sec": 180,
            "tool_timeout_sec": 900,
            "required": True,
            "enabled_tools": [
                "convert_document_into_docling_document",
                "export_docling_document_to_markdown",
                "save_docling_document",
                "get_overview_of_document_anchors",
                "search_for_text_in_document_anchors",
                "get_text_of_document_item_at_anchor",
            ],
        },
        "mineru": {
            "command": str(venv / "bin/python"),
            "args": [str(venv / "bin/mineru-open-mcp")],
            "env_vars": ["MINERU_API_TOKEN", "TMPDIR"],
            "startup_timeout_sec": 60,
            "tool_timeout_sec": 900,
            "required": True,
        },
    }


def safe_environment(workspace, bearer):
    # Do not inherit personal Codex login, SSH agent, cloud credentials, or HF_TOKEN.
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "PATH",
            "LANG",
            "LC_ALL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        }
    }
    env.update(
        {
            "HOME": str(workspace),
            "CODEX_HOME": str(workspace / ".codex"),
            "TMPDIR": str(workspace / "tmp"),
            "HF_HOME": str(workspace / ".cache/huggingface"),
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "OPENAI_API_KEY": bearer,
            "CODEX_API_KEY": bearer,
            "NO_PROXY": "127.0.0.1,localhost,::1,hf-mirror.com,.hf-mirror.com,huggingface.co,.huggingface.co,hf.co,.hf.co,xethub.hf.co,.xethub.hf.co",
            "no_proxy": "127.0.0.1,localhost,::1,hf-mirror.com,.hf-mirror.com,huggingface.co,.huggingface.co,hf.co,.hf.co,xethub.hf.co,.xethub.hf.co",
            "DOCLING_MCP_CONVERSION_MODE": "local",
            "DOCLING_DEVICE": "cpu",
            "OMP_NUM_THREADS": "4",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "MINERU_API_TOKEN": os.environ.get("MINERU_API_KEY")
            or os.environ.get("MINERU_API_TOKEN", ""),
        }
    )
    return env


def sandbox_command(workspace, uid, codex, mcp_venv, command):
    if os.environ.get("R2EACT_DOCUMENT_SANDBOX") == "bubblewrap":
        from .bubblewrap import sandbox_command as namespaced_command
        return namespaced_command(workspace, uid, codex, mcp_venv, command)
    readable = [
        Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc")
    ]
    readable += [codex, mcp_venv]
    argv = [
        "/usr/bin/python3",
        str(Path(__file__).with_name("sandbox.py")),
        "--workspace",
        str(workspace),
        "--uid",
        str(uid),
    ]
    for path in readable:
        argv += ["--read", str(path)]
    return argv + ["--", *command]


def start_codex(argv, prompt, *, env, cwd, stdout, stderr):
    """Deliver the prompt on stdin, never in the visible process command line.

    A tool's ``pgrep -f <python path>`` otherwise also matches Codex because the
    task prompt names the Python interpreter. Use an anonymous seekable file:
    it has immediate EOF, cannot consume the worker's terminal input, and does
    not block before the task timeout when the prompt exceeds a pipe buffer.
    Popen duplicates this descriptor; closing our copy does not close stdin in
    the child, including across the sandbox's exec.
    """
    with tempfile.TemporaryFile(mode="w+b") as stdin:
        stdin.write(prompt.encode("utf-8"))
        stdin.seek(0)
        return subprocess.Popen(
            argv,
            stdin=stdin,
            env=env,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def controller_binary(source: Path, cache: Path, digest: str) -> Path:
    """Keep the controller argv outside the document interpreter's generation.

    A model may clean up a stuck Python process with ``pkill -f staging-...``.
    Sharing that generation directory with Codex made the same command kill
    the agent host. Publish a separate, immutable, hash-verified executable;
    never move or overwrite a binary used by an existing execution.
    """
    destination = cache / "controllers" / digest / "codex"
    with lock(cache / "controller-setup.lock"):
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        (cache / "controllers").chmod(0o755)
        destination.parent.chmod(0o755)
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix="publish-", dir=destination.parent) as name:
                staged = Path(name) / "codex"
                shutil.copyfile(source, staged)
                if hashlib.sha256(staged.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("Codex controller source checksum mismatch")
                staged.chmod(0o755)
                staged.replace(destination)
        if destination.is_symlink() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError("node-local Codex controller checksum mismatch")
    return destination


def local_runtime():
    from terminal_bench_eval.codex_r2eact_agent import (
        PINNED_CODEX_BINARY,
        PINNED_CODEX_SHA256,
        _build_merged_model_catalog,
    )

    binary = Path(os.environ.get("R2EACT_STACK_CODEX_BINARY", str(PINNED_CODEX_BINARY)))
    if hashlib.sha256(binary.read_bytes()).hexdigest() != PINNED_CODEX_SHA256:
        raise RuntimeError("Codex binary must match the pinned TB CLI")
    mcp = Path(os.environ.get("R2EACT_DOCUMENT_VENV", str(DEFAULT_MCP_VENV)))
    for name in ("docling-mcp-server", "mineru-open-mcp"):
        if not (mcp / "bin" / name).is_file():
            raise RuntimeError(
                f"MCP executable missing: {name}; run shared_runtime/setup_runtime.sh"
            )
    # Landlock ABI 1 cannot grant execution on this cluster's Lustre filesystem.
    # A node-local cache also keeps document imports off the shared metadata server.
    cache = Path(
        os.environ.get("R2EACT_LOCAL_RUNTIME", f"/tmp/r2eact-runtime-{os.getuid()}-v2")
    )
    cache.mkdir(parents=True, exist_ok=True, mode=0o755)
    cache.chmod(0o755)
    with lock(cache / "setup.lock"):
        if not (cache / "ready.json").is_file():
            staging = Path(tempfile.mkdtemp(prefix="staging-", dir=cache))
            shutil.copytree(mcp, staging / "mcp", symlinks=True)
            # The runtime is public executable code. A private caller umask
            # must not make its node-local copy unreadable to delegated UIDs.
            for root, directories, files in os.walk(staging / "mcp", followlinks=False):
                Path(root).chmod(0o755)
                for name in files:
                    path = Path(root) / name
                    if path.is_file() and not path.is_symlink():
                        path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
            shutil.copyfile(binary, staging / "codex")
            (staging / "codex").chmod(0o755)
            # Keep immutable generations; never overwrite a cache in use.
            staging.chmod(0o755)
            atomic_json(
                cache / "ready.json",
                {"root": str(staging), "codex_sha256": PINNED_CODEX_SHA256},
            )
        ready = read_json(cache / "ready.json")
    local = Path(ready["root"])
    if (
        hashlib.sha256((local / "codex").read_bytes()).hexdigest()
        != PINNED_CODEX_SHA256
    ):
        raise RuntimeError("node-local Codex checksum mismatch")
    return controller_binary(binary, cache, PINNED_CODEX_SHA256), local / "mcp"


def preflight():
    from terminal_bench_eval.codex_r2eact_agent import _build_merged_model_catalog

    binary, mcp = local_runtime()
    for key in ("SERPER_API_KEY", "MINERU_API_KEY"):
        if not os.environ.get(key) and not (
            key == "MINERU_API_KEY" and os.environ.get("MINERU_API_TOKEN")
        ):
            raise RuntimeError(f"export {key} before launching the harness")
    # Verify the exact final process boundary before claiming any task.
    with tempfile.TemporaryDirectory(prefix="r2eact-preflight-") as name:
        workspace = Path(name)
        uid = 60000 + secrets.randbelow(5000) if os.geteuid() == 0 else os.getuid()
        os.chown(workspace, uid, uid)
        probe = sandbox_command(
            workspace,
            uid,
            binary,
            mcp,
            [
                "/bin/sh",
                "-c",
                '! cat /root/.codex/auth.json >/dev/null 2>&1 && ! cat "$1" >/dev/null 2>&1 && touch probe && echo sandbox-ok',
                "probe",
                str(PROJECT / "SPECFORGE_LICENSE"),
            ],
        )
        subprocess.run(
            probe, check=True, capture_output=True, timeout=15, cwd=workspace
        )
    _build_merged_model_catalog(binary)
    return binary, mcp
