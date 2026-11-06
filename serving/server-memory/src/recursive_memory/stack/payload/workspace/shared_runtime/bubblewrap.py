"""Local document sandbox using user/mount/PID namespaces on hosts without Landlock.

The task still runs as its delegated host UID. Only the system runtime and
explicitly staged files are visible; only the workspace is a writable host bind.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

BWRAP = Path(os.environ.get("R2EACT_BWRAP_BINARY", "/usr/bin/bwrap"))
BWRAP_SHA256 = "52231e1caf55bcbc667b269f49c63599a6f7db4767ae6a039580d0ff853db712"
BACKEND = "bubblewrap-0.9.0-uid-mount-pid-v1"


def verify_binary():
    if hashlib.sha256(BWRAP.read_bytes()).hexdigest() != BWRAP_SHA256:
        raise RuntimeError("bubblewrap differs from the pinned Ubuntu package")


def local_binary():
    verify_binary()
    cache = Path(f"/tmp/rm-bubblewrap-{os.getuid()}-{BWRAP_SHA256[:16]}")
    cache.mkdir(mode=0o755, exist_ok=True)
    info = cache.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError("bubblewrap cache must be a private-owner directory")
    cache.chmod(0o755)
    target = cache / "bwrap"
    if target.exists():
        info = target.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError("bubblewrap cache binary ownership or mode is invalid")
        if hashlib.sha256(target.read_bytes()).hexdigest() != BWRAP_SHA256:
            raise RuntimeError("cached bubblewrap checksum mismatch")
    else:
        fd, temporary = tempfile.mkstemp(dir=cache)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(BWRAP.read_bytes())
            os.chmod(temporary, 0o755)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return target


def sandbox_command(workspace, uid, codex, mcp_venv, command):
    # Inputs are controller-owned and must stay unwritable by document children.
    # A caller's umask=077 otherwise also makes them unreadable to the delegated UID.
    inputs = Path(workspace) / "inputs"
    if inputs.is_dir() and not inputs.is_symlink() and inputs.stat().st_uid == os.geteuid():
        inputs.chmod(0o755)
    return ["/usr/bin/python3", str(Path(__file__).resolve()), "--workspace", str(workspace),
            "--binary", str(BWRAP), "--uid", str(uid), "--read", str(codex), "--read", str(mcp_venv), "--", *command]


def seccomp_fd():
    """Deny namespace creation in descendants; keep normal threads/fork working."""
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp is required for the namespace sandbox")
    lib = ctypes.CDLL(library, use_errno=True)

    class Comparison(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Comparison)]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("cannot create seccomp policy")
    fd = os.memfd_create("rm-document-seccomp", os.MFD_CLOEXEC)
    try:
        def deny(name, error=1, comparison=None):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                raise RuntimeError("seccomp does not recognize syscall " + name)
            result = lib.seccomp_rule_add_array(context, 0x00050000 | error, number,
                1 if comparison is not None else 0,
                ctypes.byref(comparison) if comparison is not None else None)
            if result < 0:
                raise RuntimeError(f"cannot add seccomp rule for {name}: {result}")
        for name in ("unshare", "setns", "mount", "umount2", "pivot_root", "chroot"):
            deny(name)
        # clone3's flags are behind a pointer, so return ENOSYS for the standard
        # libc fallback to clone. Filter namespace flags on clone's first arg.
        deny("clone3", error=38)
        for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000,
                     0x10000000, 0x20000000, 0x40000000):
            deny("clone", comparison=Comparison(0, 7, flag, flag))  # SCMP_CMP_MASKED_EQ
        if lib.seccomp_export_bpf(context, fd) < 0:
            raise RuntimeError("cannot export seccomp filter")
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise
    finally:
        lib.seccomp_release(context)


def bwrap_command(workspace, readable, command, filter_fd):
    argv = [str(local_binary()), "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--seccomp", str(filter_fd), "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
    for name in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(name)
        if path.is_symlink():
            argv += ["--symlink", os.readlink(path), name]
        elif path.exists():
            argv += ["--ro-bind", name, name]
    # This cluster forbids mounting a fresh procfs. Expose only CPU/memory
    # information, never host PID directories, /proc/self/fd or /proc/*/root.
    argv += ["--dir", "/proc", "--dev", "/dev", "--dir", "/tmp"]
    for name in ("cpuinfo", "meminfo", "stat", "uptime"):
        argv += ["--ro-bind", "/proc/" + name, "/proc/" + name]
    # Codex/Rust resolves its executable during config loading. This host
    # forbids a fresh procfs mount, so expose only that executable link; never
    # bind the host's /proc/self (which would expose controller descriptors).
    # This is a compatibility link for the initial sandbox command, not a
    # full procfs implementation for arbitrary descendant executables.
    executable = Path(command[0])
    if not executable.is_absolute():
        raise ValueError("sandbox command executable must be absolute")
    argv += ["--dir", "/proc/self", "--symlink", str(executable), "/proc/self/exe"]
    for path in dict.fromkeys(str(Path(p).absolute()) for p in readable):
        if not Path(path).exists():
            raise FileNotFoundError("sandbox runtime path is missing: " + path)
        argv += ["--ro-bind", path, path]
    argv += ["--bind", str(workspace), str(workspace), "--chdir", str(workspace),
             "--remount-ro", "/", "--", *command]
    return argv


def boundary_probe():
    with tempfile.TemporaryDirectory(prefix="rm-bwrap-probe-", dir="/tmp") as name:
        root = Path(name)
        root.chmod(0o755)
        workspace, runtime = root / "workspace", root / "runtime"
        workspace.mkdir(mode=0o700)
        runtime.mkdir(mode=0o755)
        runtime.chmod(0o755)
        allowed, outside = runtime / "allowed", root / "outside"
        for p in (allowed, outside):
            p.write_text("public synthetic canary\n")
            p.chmod(0o644)
        uid = 60000 + os.getpid() % 5000 if os.geteuid() == 0 else os.getuid()
        if os.geteuid() == 0:
            os.chown(workspace, uid, uid)
        script = '''import ctypes,json,os,subprocess,sys
from pathlib import Path
allowed,outside,expected_uid,parent_pid=sys.argv[1:]
assert os.getuid()==int(expected_uid) and os.getuid()!=0
assert Path(allowed).read_text()=="public synthetic canary\\n"
assert not Path(outside).exists()
assert not Path("/proc/"+parent_pid+"/root").exists()
try:
    Path(allowed).write_text("must fail")
except OSError:
    pass
else:
    raise AssertionError("runtime was writable")
Path("probe").write_text("workspace writable")
class Header(ctypes.Structure):
    _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class Caps(ctypes.Structure):
    _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
header=Header(0x20080522,0)
caps=(Caps*2)()
assert ctypes.CDLL(None).capget(ctypes.byref(header),ctypes.byref(caps))==0
assert not any(x.effective or x.permitted for x in caps)
assert subprocess.run(["/usr/bin/unshare","--user","/bin/true"],capture_output=True).returncode != 0
print(json.dumps({"status":"ok","uid":os.getuid(),"outside_hidden":True,"runtime_readonly":True,"host_processes_hidden":True,"nested_userns_disabled":True}))
'''
        command = sandbox_command(workspace, uid, Path("/bin/true"), runtime,
            ["/usr/bin/python3", "-c", script, str(allowed), str(outside), str(uid), str(os.getpid())])
        result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, timeout=30,
                                env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"})
        if result.returncode:
            raise RuntimeError("bubblewrap canary failed: " + result.stderr[-4000:])
        assert (workspace / "probe").is_file()
        return json.loads(result.stdout)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--read", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    global BWRAP
    BWRAP = args.binary
    workspace = args.workspace.resolve(strict=True)
    verify_binary()
    argv = bwrap_command(workspace, args.read, command, seccomp_fd())
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
    if os.geteuid() == 0:
        if args.uid <= 0:
            raise ValueError("sandbox uid must not be root")
        os.setgroups([])
        os.setgid(args.uid)
        os.setuid(args.uid)
    elif os.geteuid() != args.uid:
        raise RuntimeError("cannot delegate the workspace UID")
    if libc.prctl(1, 9, 0, 0, 0) < 0 or os.getppid() != parent:
        raise RuntimeError("sandbox parent exited during setup")
    os.chdir(workspace)
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
