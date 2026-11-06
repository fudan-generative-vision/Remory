"""Linux Landlock + unprivileged UID boundary for Codex and its MCP children.

Fail closed on hosts without Landlock. Network access is intentionally allowed.
Unlike the public JobBench prompt-only restriction, the dataset/controller tree
is not readable and filesystem writes are restricted to the sample workspace.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
from pathlib import Path


def restrict(workspace: Path, readable: list[Path], uid: int):
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("unsupported Landlock syscall architecture")
    libc = ctypes.CDLL(None, use_errno=True)

    def check(result):
        if result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return result

    abi = check(libc.syscall(444, 0, 0, 1))
    if abi < 1:
        raise RuntimeError("Landlock is required; refusing an unrestricted agent")
    # ABI 1: bits 0..12; ABI 2 adds REFER; ABI 3 adds TRUNCATE.
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    ruleset = ctypes.c_uint64(handled)
    fd = check(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    try:
        read_bits = (1 << 0) | (1 << 2) | (1 << 3)
        rules = [(p, read_bits) for p in readable]
        rules.append((workspace, handled))
        for name in ("null", "zero", "random", "urandom", "tty", "ptmx", "pts"):
            rules.append((Path("/dev") / name, read_bits | (1 << 1)))
        for path, access in rules:
            if not path.exists():
                continue
            if not path.is_dir():
                access &= (1 << 0) | (1 << 1) | (1 << 2) | ((1 << 14) if abi >= 3 else 0)
            pathfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, pathfd)
                check(libc.syscall(445, fd, 1, ctypes.byref(rule), 0))
            finally:
                os.close(pathfd)
        check(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        if os.geteuid() == 0:
            if uid <= 0:
                raise ValueError("agent uid must not be root")
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
        elif os.geteuid() != uid:
            raise RuntimeError("cannot delegate the workspace UID")
        check(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--read", type=Path, action="append", default=[])
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    # Codex must not outlive a killed runner and keep using its model endpoint.
    expected_parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, 9, 0, 0, 0) < 0:  # PR_SET_PDEATHSIG, SIGKILL
        raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
    if os.getppid() != expected_parent:
        raise SystemExit("runner already exited")
    restrict(a.workspace.resolve(), a.read, a.uid)
    # setuid clears PDEATHSIG, so set it again after dropping privileges.
    if libc.prctl(1, 9, 0, 0, 0) < 0 or os.getppid() != expected_parent:
        raise SystemExit("runner exited while establishing the sandbox")
    os.chdir(a.workspace)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
