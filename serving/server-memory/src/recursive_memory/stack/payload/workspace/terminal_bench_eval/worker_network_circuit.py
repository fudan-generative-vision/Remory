"""Host/egress-local claim gate. Never mutate VPNs, scores, or active trials."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def image_target(failure):
    # Called only after the pool's host-receipt/exception classifier accepted
    # this infrastructure failure; never match arbitrary model/tool output.
    if failure.get("cause") != "proot_image_pull_transport_failure":
        return None
    first_line = str(failure.get("message", "")).partition("\n")[0]
    match = re.search(r"\bdocker://([A-Za-z0-9_./:@-]+)\s+oci:", first_line)
    if not match:
        raise RuntimeError("classified image outage lacks a safe recovery target")
    return {"kind": "image", "image": match.group(1)}


def transport_key(environ=None):
    env = os.environ if environ is None else environ
    # Share between slots using the same egress, not across nodes/proxies.
    # Only the digest is persisted; proxy credentials never enter the journal.
    values = {k: env.get(k, "") for k in (
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    )}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@contextmanager
def _lock(path, *, blocking=True):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


@contextmanager
def _cancellable_signals():
    # The supervisor sends TERM on Ctrl+C. Python's default TERM handler would
    # skip finally and orphan the separately sessioned download probe.
    previous = {}
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class NetworkCircuit:
    def __init__(self, root=None, *, clock=time.time, sleep=time.sleep,
                 probe=None, cooldown=30.0, stable_interval=10.0):
        self.root = Path(root) if root is not None else Path(
            f"/tmp/terminal-bench-egress-{os.getuid()}"
        ) / transport_key()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise RuntimeError("unsafe worker network circuit directory")
        self.path = self.root / "state.json"
        self.clock, self.sleep = clock, sleep
        self.probe = probe or probe_target
        self.cooldown, self.stable_interval = cooldown, stable_interval

    def _read(self):
        if not self.path.exists():
            return {"targets": {}}
        if self.path.is_symlink() or self.path.stat().st_size > 1024 * 1024:
            raise RuntimeError("unsafe worker network circuit state")
        value = json.loads(self.path.read_text())
        if not isinstance(value.get("targets"), dict):
            raise RuntimeError("invalid worker network circuit state")
        return value

    def _write(self, value):
        fd, name = tempfile.mkstemp(dir=self.root, prefix=".state-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def trip(self, target):
        key = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
        with _lock(self.root / "state.lock"):
            state = self._read()
            state["targets"][key] = target
            state.update(revision=uuid.uuid4().hex, successes=0,
                         next_probe=self.clock() + self.cooldown, failures=0)
            self._write(state)
        print("[terminal-bench] network circuit OPEN: pausing new claims on this egress", flush=True)

    def wait(self, heartbeat):
        """No cell lock is held here. Other slots finish their current trials."""
        with _cancellable_signals():
            self._wait(heartbeat)

    def _wait(self, heartbeat):
        while True:
            with _lock(self.root / "state.lock"):
                state = self._read()
            if not state["targets"]:
                return
            detail = {"circuit": "open", "target_count": len(state["targets"]),
                      "next_probe_at": state.get("next_probe"),
                      "probe_failures": state.get("failures", 0)}
            heartbeat(detail)
            if self.clock() < state.get("next_probe", 0):
                self.sleep(min(1.0, state["next_probe"] - self.clock()))
                continue
            # One recovery probe per egress, not one per slot. A concurrent
            # trip changes revision and cannot be cleared by this older probe.
            with _lock(self.root / "probe.lock", blocking=False) as acquired:
                if acquired:
                    with _lock(self.root / "state.lock"):
                        snapshot = self._read()
                    if not snapshot["targets"] or self.clock() < snapshot.get("next_probe", 0):
                        continue
                    healthy = all(self.probe(target, lambda: heartbeat(detail))
                                  for target in snapshot["targets"].values())
                    with _lock(self.root / "state.lock"):
                        current = self._read()
                        if current.get("revision") != snapshot.get("revision"):
                            continue
                        successes = snapshot.get("successes", 0) + 1 if healthy else 0
                        failures = 0 if healthy else snapshot.get("failures", 0) + 1
                        if successes >= 2:
                            self._write({"targets": {}})
                            print("[terminal-bench] network circuit CLOSED: recovery probes passed twice; claims resume", flush=True)
                            return
                        delay = self.stable_interval if healthy else min(300.0, self.cooldown * 2 ** min(failures - 1, 4))
                        current.update(successes=successes, failures=failures,
                                       next_probe=self.clock() + delay)
                        self._write(current)
            self.sleep(1.0)


def probe_target(target, heartbeat, *, timeout=90.0):
    if target.get("kind") == "image":
        image = target.get("image", "")
        if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", image) or image.startswith("-"):
            raise RuntimeError("invalid image recovery target")
        # Skopeo checks the actual registry + auth + manifest route using the
        # same inherited proxy/TLS policy as the failed image pull. /v2/ alone
        # or a generic IP/Serper check is not sufficient. No TLS bypass.
        command = ["skopeo", "--command-timeout", "75s", "inspect", "--raw", "docker://" + image]
    elif target.get("kind") == "task":
        command = [sys.executable, "-m", __name__, "--probe-task", json.dumps(target["config"])]
    else:
        raise RuntimeError("unknown network recovery target")
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return False  # A missing/broken probe is never evidence of recovery.
    deadline = time.monotonic() + timeout
    try:
        while child.poll() is None:
            heartbeat()
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)
        return child.returncode == 0
    finally:
        # Cancel just this read/download probe, not the worker or proxy.
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe-task":
        raise SystemExit("requires --probe-task JSON")
    from terminal_bench_eval.terminal_bench_pool import _prewarm_task_cache
    # Actual pinned-package fetch/verification under its normal local cache
    # lock, not a registry metadata refresh or an unrelated connectivity test.
    _prewarm_task_cache(json.loads(sys.argv[2]))
