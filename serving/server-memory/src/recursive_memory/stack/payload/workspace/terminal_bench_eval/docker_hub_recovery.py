"""Opt-in Docker Hub recovery using isolated proxies, never an existing listener.

The operator runs this on the affected worker. A successful prefetch verifies
every OCI blob before publishing the PRoot cache marker. No task is claimed,
no result is changed, and no existing proxy or worker is signalled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time


PROTOCOLS = {"hysteria2", "trojan", "vmess", "vless", "shadowsocks", "tuic", "anytls"}
IMAGE_RE = re.compile(r"(?:docker\.io/)?[a-z0-9_-]+/[a-z0-9_.-]+(?::[A-Za-z0-9_.-]+|@sha256:[0-9a-f]{64})")
SHARED_CACHE = Path(os.environ.get("HARBOR_SHARED_OCI_CACHE", str(
    Path(__file__).resolve().parents[1] / "eval/terminal_bench/shared-dockerhub-cache")))


def candidates(config):
    # Interleave protocols instead of spending the entire budget on one family.
    groups = defaultdict(deque)
    for node in config.get("outbounds", []):
        if node.get("type") in PROTOCOLS and not node.get("detour"):
            groups[node["type"]].append(node)
    while any(groups.values()):
        for group in groups.values():
            if group:
                yield group.popleft()


def isolated_config(source, node, port):
    outbound = copy.deepcopy(node)
    outbound["tag"] = "dockerhub"
    value = {
        "log": {"level": "warn", "timestamp": True},
        "inbounds": [{"type": "http", "tag": "dockerhub-in", "listen": "127.0.0.1", "listen_port": port}],
        "outbounds": [outbound, {"type": "direct", "tag": "direct"}],
        "route": {"auto_detect_interface": True, "final": "dockerhub"},
    }
    if source.get("dns"):
        value["dns"] = copy.deepcopy(source["dns"])
    return value


def proxy_env(port):
    env = dict(os.environ)
    for key in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        env.pop(key, None)
    url = f"http://127.0.0.1:{port}"
    env.update(http_proxy=url, HTTP_PROXY=url, https_proxy=url, HTTPS_PROXY=url,
               no_proxy="127.0.0.1,localhost,::1", NO_PROXY="127.0.0.1,localhost,::1")
    return env


def stop_owned(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run(command, *, env=None, timeout=45):
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = child.communicate(timeout=timeout)
        if child.returncode:
            # Never expose node config/credentials or registry auth tokens.
            message = err.decode(errors="replace").lower()
            category = next((kind for kind in ("tls handshake timeout", "connection reset", "i/o timeout",
                "unauthorized", "toomanyrequests", "manifest unknown", "context deadline exceeded") if kind in message),
                f"exit {child.returncode}")
            raise RuntimeError(category)
        return out
    finally:
        stop_owned(child)


@contextmanager
def isolated_proxy(source, node, root, binary):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = root / "sing-box.json"
    config.write_text(json.dumps(isolated_config(source, node, port)))
    config.chmod(0o600)
    run([binary, "check", "-c", str(config)], timeout=10)
    with (root / "sing-box.log").open("ab") as log:
        child = subprocess.Popen([binary, "run", "-c", str(config)], stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and child.poll() is None:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("isolated proxy did not bind its new port")
            time.sleep(.1)
            if child.poll() is not None:
                raise RuntimeError("isolated proxy exited during startup")
            yield proxy_env(port)
        finally:
            stop_owned(child)


def manifest_digest(payload):
    manifest = json.loads(payload)
    if manifest.get("schemaVersion") != 2 or not isinstance(manifest.get("layers"), list):
        raise ValueError("expected a single-platform OCI/Docker image manifest")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def verify_oci(path, expected):
    for component in (path, path / "index.json", path / "blobs", path / "blobs/sha256"):
        if component.is_symlink():
            raise ValueError("symlinked OCI layout")
    index = json.loads((path / "index.json").read_text())
    manifests = index.get("manifests", [])
    if len(manifests) != 1 or manifests[0].get("digest") != expected:
        raise ValueError("OCI cache manifest does not match the verified registry manifest")
    def blob(digest):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("unsupported OCI blob digest")
        file = path / "blobs" / "sha256" / digest[7:]
        if file.is_symlink() or not file.is_file():
            raise ValueError("missing/unsafe OCI blob")
        with file.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest[7:]:
            raise ValueError("OCI blob SHA-256 mismatch")
        return file
    manifest = json.loads(blob(expected).read_text())
    for item in [manifest["config"], *manifest["layers"]]:
        file = blob(item["digest"])
        if file.stat().st_size != item["size"]:
            raise ValueError("OCI blob size mismatch")
    return len(manifest["layers"])


def shared_oci_source(image, *, cache=None):
    """Only explicitly published, fully verified images replace registry reads."""
    cache = SHARED_CACHE if cache is None else Path(cache)
    key = hashlib.sha256(image.encode()).hexdigest()[:20]
    images = cache / "images"
    marker, source = images / f"{key}.complete", images / key
    if not marker.exists():
        return None
    for path in (cache, images, source, marker, source / "index.json"):
        if path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
            raise ValueError("shared OCI cache must be owned, non-symlinked and not writable by others")
    if marker.read_text() != image + "\n":
        raise ValueError("shared OCI cache marker names a different image")
    manifests = json.loads((source / "index.json").read_text()).get("manifests", [])
    if len(manifests) != 1:
        raise ValueError("invalid shared OCI index")
    digest = manifests[0]["digest"]
    verify_oci(source, digest)
    return source, digest


def prefetch(image, digest, env, root, cache, *, expected_oci_digest=None):
    staged = root / "oci"
    repository = image.split("@", 1)[0].split(":", 1)[0]
    # Match PRoot's normal Docker-to-OCI conversion. Registry and OCI manifest
    # digests can differ solely because their mediaType fields differ.
    run(["skopeo", "--command-timeout", "900s", "copy", "--retry-times", "2",
         f"docker://{repository}@{digest}", f"oci:{staged}:image"], env=env, timeout=910)
    oci_digest = json.loads((staged / "index.json").read_text())["manifests"][0]["digest"]
    if expected_oci_digest and oci_digest != expected_oci_digest:
        raise ValueError("converted OCI image differs from the historical benchmark image")
    layers = verify_oci(staged, oci_digest)
    print(f"[dockerhub-recovery] OCI manifest verified: {oci_digest}", flush=True)
    publish_oci(image, staged, oci_digest, cache)
    return layers


def publish_oci(image, staged, oci_digest, cache):
    # Do not replace/delete an existing cache directory. Skopeo copies verified
    # content-addressed blobs and publishes the index; the ready marker is last.
    layers = verify_oci(staged, oci_digest)
    images = cache / "images"
    images.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(image.encode()).hexdigest()[:20]
    destination, marker = images / key, images / f"{key}.complete"
    for path in (images, destination, marker):
        if path.is_symlink():
            raise ValueError("refusing a symlinked PRoot image cache")
    with (images / f".{key}.recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            if marker.read_text() != image + "\n":
                raise ValueError("existing cache marker names a different image")
            verify_oci(destination, oci_digest)
            return layers
        run(["skopeo", "copy", "--preserve-digests", f"oci:{staged}:image", f"oci:{destination}:image"], timeout=300)
        verify_oci(destination, oci_digest)
        # root and cache may be on different mounts, so create marker locally.
        fd, name = tempfile.mkstemp(prefix=f".{key}.", dir=images)
        with os.fdopen(fd, "w") as stream:
            stream.write(image + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, marker)
        descriptor = os.open(images, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--image", default="alexgshaw/mailman:20251031")
    parser.add_argument("--expected-registry-digest")
    parser.add_argument("--expected-oci-digest")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("HARBOR_PROOT_CACHE", "/tmp/harbor-proot-cache")))
    parser.add_argument("--sing-box", default="sing-box")
    args = parser.parse_args()
    if not IMAGE_RE.fullmatch(args.image) or not 1 <= args.max_candidates <= 64:
        parser.error("expected a public Docker Hub repository:tag and 1-64 candidates")
    for digest in (args.expected_registry_digest, args.expected_oci_digest):
        if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            parser.error("invalid expected digest")
    if args.source_config.is_symlink() or args.source_config.stat().st_uid != os.getuid() or args.source_config.stat().st_mode & 0o022:
        parser.error("source config must be owned and not writable by other users")
    source = json.loads(args.source_config.read_text())
    os.umask(0o077)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix="tb-dockerhub-recovery-") as directory:
        root = Path(directory)
        for number, node in enumerate(candidates(source), 1):
            if number > args.max_candidates:
                break
            print(f"[dockerhub-recovery] isolated candidate {number} ({node['type']})", flush=True)
            candidate_root = root / str(number)
            candidate_root.mkdir()
            try:
                with isolated_proxy(source, node, candidate_root, args.sing_box) as env:
                    payload = run(["skopeo", "--command-timeout", "35s", "inspect", "--raw", f"docker://{args.image}"], env=env)
                    digest = manifest_digest(payload)
                    if args.expected_registry_digest and digest != args.expected_registry_digest:
                        raise ValueError("registry manifest differs from the requested registry digest")
                    # Require a second authenticated lookup, not just a generic IP check.
                    second = run(["skopeo", "--command-timeout", "35s", "inspect", "--raw", f"docker://{args.image}"], env=env)
                    if manifest_digest(second) != digest:
                        raise ValueError("registry manifest changed between probes")
                    print(f"[dockerhub-recovery] registry + auth + manifest passed twice: {digest}", flush=True)
                    if args.prefetch:
                        print("[dockerhub-recovery] downloading and verifying all image layers", flush=True)
                        layers = prefetch(args.image, digest, env, root, args.cache_root,
                            expected_oci_digest=args.expected_oci_digest)
                        print(f"[dockerhub-recovery] READY: {layers} verified layers cached on {socket.gethostname()}; existing proxies/workers unchanged", flush=True)
                    else:
                        print("[dockerhub-recovery] probe only; no cache or running proxy was changed", flush=True)
                    return 0
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                print(f"[dockerhub-recovery] candidate unavailable: {type(exc).__name__}: {exc if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__}", flush=True)
    print("[dockerhub-recovery] no candidate completed the real Docker Hub checks; no ready marker published", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
