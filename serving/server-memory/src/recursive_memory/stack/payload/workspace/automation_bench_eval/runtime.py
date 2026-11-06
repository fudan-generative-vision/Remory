"""Portable, versioned official runtime shared by cluster task instances.

The legacy venv points into the first machine's /root/.local/share/uv. Never
rewrite it: an existing execution may still have that interpreter running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OFFICIAL_REVISION = "4a8e1061254004d9dac807054eed33fad7d1ff14"
RUNTIME_SCHEMA = "automationbench_shared_python_runtime_v1"
PYTHON_VERSION = "3.13.7"
RUNTIME_DIRECTORY = "runtimes/shared-cpython-3.13.7-v1"
PACKAGES = {"automation-bench": "1.0.6", "verifiers": "0.2.1", "datasets": "4.4.2"}


def official_python(root):
    return Path(root) / RUNTIME_DIRECTORY / "venv/bin/python"


def ready_path(root):
    return Path(root) / RUNTIME_DIRECTORY / "ready.json"


def interpreter_environment():
    return {k: v for k, v in os.environ.items() if k not in {
        "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV"}}


def _identity(root):
    root = Path(root).resolve()
    interpreter = official_python(root)
    resolved = interpreter.resolve(strict=True)
    managed = (root / "python-installations").resolve(strict=True)
    if not resolved.is_relative_to(managed):
        raise RuntimeError(f"Official interpreter must live in shared {managed}, not {resolved}")
    # -I excludes cwd/PYTHONPATH; the project's local verifiers package must not
    # shadow the locked official verifiers distribution in this subprocess.
    code = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({'python_version': '.'.join(map(str, sys.version_info[:3])), "
        "'base_prefix': sys.base_prefix, 'prefix': sys.prefix, "
        f"'packages': {{n: m.version(n) for n in {list(PACKAGES)!r}}}}}))"
    )
    result = subprocess.run([str(interpreter), "-I", "-c", code],
        env=interpreter_environment(), cwd="/tmp", text=True, capture_output=True,
        check=True, timeout=30)
    identity = json.loads(result.stdout)
    if identity["python_version"] != PYTHON_VERSION or identity["packages"] != PACKAGES:
        raise RuntimeError("Official runtime Python/dependency versions do not match the pinned environment")
    if not Path(identity["base_prefix"]).resolve().is_relative_to(managed):
        raise RuntimeError("Python base_prefix still points outside the shared interpreter installation")
    if Path(identity["prefix"]).resolve() != interpreter.parent.parent.resolve():
        raise RuntimeError("Official interpreter is not using its isolated virtual environment")
    return {"schema": RUNTIME_SCHEMA, "official_commit": OFFICIAL_REVISION,
            "lock_sha256": hashlib.sha256((root / "source/uv.lock").read_bytes()).hexdigest(),
            "interpreter": str(resolved), **identity}


def runtime_error(root):
    """Return a short diagnosis; never mutate a published environment."""
    try:
        recorded = json.loads(ready_path(root).read_text())
        if recorded != _identity(root):
            return "Official shared runtime identity differs from its ready marker"
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def publish_runtime(root):
    """Called under setup.lock only after uv sync completed successfully."""
    identity = _identity(root)
    marker = ready_path(root)
    temporary = marker.with_name(f"ready-{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(identity, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(marker)


def ensure_runtime(root):
    root = Path(root).resolve()
    problem = runtime_error(root)
    if problem is None:
        return official_python(root)
    print("[automationbench] preparing the shared official Python runtime before any task starts", flush=True)
    env = interpreter_environment()
    env["AUTOMATIONBENCH_RUNTIME_ROOT"] = str(root)
    subprocess.run(["bash", str(Path(__file__).with_name("setup_runtime.sh"))], env=env, check=True)
    problem = runtime_error(root)
    if problem:
        raise RuntimeError(f"Official shared Python runtime is still unavailable: {problem}. No task started.")
    return official_python(root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    if args.publish:
        publish_runtime(args.runtime_root)
    error = runtime_error(args.runtime_root)
    if error:
        raise SystemExit(error)
    print(f"[automationbench] shared runtime verified: {official_python(args.runtime_root)}")


if __name__ == "__main__":
    main()
