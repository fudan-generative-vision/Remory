#!/usr/bin/env python3
"""Dispatch existing r-codex sessions to their saved memory transport."""
import argparse
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
BASE = SOURCE.parent
LEGACY = BASE / "acl2027-flow-research"


def memory_endpoint(manifest, environment):
    saved = manifest.get("server_memory")
    if saved:
        if saved.get("schema") != 1 or not isinstance(saved.get("endpoint"), str):
            raise ValueError("Unsupported saved server-memory configuration")
        return saved["endpoint"]
    return environment.get("REMORY_SGLANG_URL", "").rstrip("/")


def client_settings(manifest, environment, default, *, resuming=False):
    """Keep existing sessions on their saved model; defaults apply to new ones."""
    endpoint = memory_endpoint(manifest, environment)
    saved = manifest.get("server_memory") or {}
    runtime = saved.get("runtime_file") or environment.get("REMORY_RUNTIME_FILE")
    if not resuming:
        endpoint = endpoint or default.get("endpoint", "")
        runtime = runtime or default.get("runtime_file")
    return endpoint, runtime


def main():
    args = sys.argv[1:]
    manifest = {}
    sys.path.append(str(LEGACY / "harness"))
    if args[:1] == ["resume"] and not any(a in ("--help", "-h") for a in args):
        parser = argparse.ArgumentParser(prog="r-codex resume")
        parser.add_argument("session", nargs="?")
        parser.add_argument("--last", action="store_true")
        parser.add_argument("--unsafe", action="store_true")
        parser.add_argument("--prompt")
        options = parser.parse_args(args[1:])
        if options.session and options.last:
            parser.error("Provide either a session name or --last, not both")
        sessions = BASE / "r-codex-sessions"
        name = options.session
        if options.last:
            from client_node import last_path
            import hashlib
            import socket
            # Both legacy and remote clients record their most recent session.
            # Pick the newest record belonging to this execution node.
            key = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]
            candidates = [sessions / "last.json", sessions / ("last-" + key + ".json")]
            candidates = [p for p in candidates if p.is_file()]
            path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else last_path(sessions)
            name = json.loads(path.read_text())["session"]
        elif not name and sys.stdin.isatty() and sys.stdout.isatty():
            from session_picker import choose_session
            name = choose_session(sessions)
            if name is None:
                return 0
        if name:
            from session_registry import resolve_session
            kind, name, folder = resolve_session(sessions, name)
            if kind == "interactive":
                manifest = json.loads((folder / "session.json").read_text())
            args = ["resume", name]
            if options.unsafe:
                args.append("--unsafe")
            if options.prompt is not None:
                args.extend(["--prompt", options.prompt])
    defaults = SOURCE / "deployment.json"
    default = json.loads(defaults.read_text()) if defaults.is_file() else {}
    endpoint, runtime = client_settings(manifest, os.environ, default,
                                        resuming=args[:1] == ["resume"])
    script = LEGACY / "harness/interactive_cli.py"
    if endpoint:
        os.environ.update(REMORY_SGLANG_URL=endpoint, REMORY_SERVER_MEMORY="1")
        if runtime:
            os.environ["REMORY_RUNTIME_FILE"] = str(Path(runtime).expanduser().resolve(strict=True))
        script = SOURCE / "client_harness/interactive_cli.py"
    entry = BASE / "rm-qwen-eval/20261012-subagent-folding/code/entry.py"
    python = BASE / "rm-stack/controller/bin/python"
    os.execv(str(python), [str(python), str(entry), "--script", str(script), *args])


if __name__ == "__main__":
    raise SystemExit(main())
