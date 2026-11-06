"""Bootstrap-only direct transport for reachable official APT origins.

Do not change sources, suites, signing keys, or the agent's proxy environment.
An unavailable direct route leaves the existing proxy configuration in force.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import urllib.request
from urllib.parse import urlsplit
import uuid

from .proot_apt_sources import _LINE, _field

POLICY = "official_apt_direct_probe_bootstrap_only_v1"
OFFICIAL_HOSTS = frozenset({"deb.debian.org", "security.debian.org",
                           "archive.ubuntu.com", "security.ubuntu.com"})


def _probe(url):
    # No inherited proxy, credentials, redirects or third-party endpoints.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(url, timeout=4) as response:
            return response.status == 200 and response.read(64).startswith(b"-----BEGIN PGP SIGNED MESSAGE-----")
    except Exception:
        return False


def prepare_apt_direct_transport(rootfs):
    root = Path(rootfs).resolve()
    if root == Path("/"):
        raise RuntimeError("Refuse to modify host APT configuration")
    # A tiny direct probe proves reachability, not enough throughput for APT's
    # bounded bootstrap deadline. Respect an explicitly configured egress proxy
    # instead of overriding it with DIRECT after reading only 64 bytes.
    if any(os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")):
        return None, dict(policy="respect_configured_proxy_bootstrap_v1", probes=[],
                          direct_hosts=[], signature_verification_unchanged=True,
                          agent_proxy_unchanged=True)
    apt = root / "etc/apt"
    if not apt.resolve().is_relative_to(root):
        raise RuntimeError("Unsafe guest APT directory")
    files = [apt / "sources.list"]
    parts = apt / "sources.list.d"
    if parts.exists():
        if not parts.resolve().is_relative_to(root):
            raise RuntimeError("Unsafe guest APT sources directory")
        files += sorted(p for p in parts.iterdir() if p.suffix in {".list", ".sources"})
    candidates = {}
    for path in files:
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError("Unsafe guest APT source")
        text = path.read_text()
        pairs = []
        if path.suffix == ".sources":
            for stanza in re.split(r"\n\s*\n", text):
                uris, suites, enabled = (_field(k).search(stanza) for k in ("URIs", "Suites", "Enabled"))
                if not uris or not suites or (enabled and enabled[0].split(":", 1)[1].strip().lower() == "no"):
                    continue
                pairs.extend((u, s) for u in uris[0].split(":", 1)[1].split()
                             for s in suites[0].split(":", 1)[1].split())
        else:
            pairs.extend((m["url"], m["suite"]) for m in _LINE.finditer(text))
        for origin, suite in pairs:
            url = urlsplit(origin)
            if (url.scheme not in {"http", "https"} or url.hostname not in OFFICIAL_HOSTS
                or url.netloc != url.hostname or url.query or url.fragment
                or not re.fullmatch(r"[a-z][a-z0-9-]*", suite)):
                continue
            candidates.setdefault(url.hostname, origin.rstrip("/") + "/dists/" + suite + "/InRelease")
    probes = []
    if candidates:
        with ThreadPoolExecutor(max_workers=len(OFFICIAL_HOSTS)) as executor:
            results = executor.map(_probe, candidates.values())
            probes = [dict(host=host, direct_reachable=ok) for host, ok in zip(candidates, results)]
    hosts = sorted(item["host"] for item in probes if item["direct_reachable"])
    receipt = dict(policy=POLICY, probes=probes, direct_hosts=hosts,
                   signature_verification_unchanged=True, agent_proxy_unchanged=True)
    if not hosts:
        return None, receipt
    config_dir = apt / "apt.conf.d"
    if not config_dir.resolve().is_relative_to(root):
        raise RuntimeError("Unsafe guest APT config directory")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / ("99r2eact-bootstrap-transport-" + uuid.uuid4().hex)
    with config.open("x") as handle:
        os.fchmod(handle.fileno(), 0o644)
        for host in hosts:
            for scheme in ("http", "https"):
                handle.write(f'Acquire::{scheme}::Proxy::{host} "DIRECT";\n')
        handle.flush()
        os.fsync(handle.fileno())
    receipt["temporary_config"] = str(config.relative_to(root))
    return config, receipt
