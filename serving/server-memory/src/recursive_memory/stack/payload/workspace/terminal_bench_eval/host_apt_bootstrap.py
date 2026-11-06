"""Install worker host dependencies without changing its proxy or APT sources.

Try HTTPS official sources, with a probed per-host direct route where available,
then the inherited proxy, then the original sources. APT still verifies Release
signatures/package hashes on every route; a probe is only a reachability check.
This is host bootstrap, never part of a benchmark task's environment.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
import urllib.request


OFFICIAL_HOSTS = frozenset(
    {
        "archive.ubuntu.com",
        "security.ubuntu.com",
        "deb.debian.org",
        "security.debian.org",
    }
)
SOURCE_LINE = re.compile(
    r"^\s*deb(?:-src)?\s+(?:\[[^\]\n]*\]\s+)?(?P<url>\S+)\s+(?P<suite>\S+)",
    re.M,
)
APT_OPTIONS = [
    "-o",
    "Acquire::Retries=1",
    "-o",
    "Acquire::http::Timeout=15",
    "-o",
    "Acquire::https::Timeout=20",
    "-o",
    "DPkg::Lock::Timeout=60",
    "-o",
    "APT::Update::Error-Mode=any",
    "-o",
    "Acquire::AllowInsecureRepositories=false",
    "-o",
    "Acquire::AllowDowngradeToInsecureRepositories=false",
    "-o",
    "APT::Get::AllowUnauthenticated=false",
    "-o",
    "Acquire::https::Verify-Peer=true",
    "-o",
    "Acquire::https::Verify-Host=true",
]


def field(name):
    return re.compile(r"^" + name + r":[^\n]*(?:\n[ \t]+[^\n]*)*", re.M | re.I)


def official_host(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in OFFICIAL_HOSTS
        and parsed.netloc == parsed.hostname
        and not parsed.query
        and not parsed.fragment
    ):
        return parsed.hostname
    return None


def https_url(url):
    return "https:" + url[5:] if url.startswith("http:") and official_host(url) else url


def https_sources(text, suffix):
    """Only change official URI schemes, preserving all trust/suite/options fields."""
    if suffix == ".sources":
        return field("URIs").sub(
            lambda match: re.sub(r"\S+", lambda token: https_url(token[0]), match[0]),
            text,
        )
    return SOURCE_LINE.sub(
        lambda match: match[0][: match.start("url") - match.start()]
        + https_url(match["url"])
        + match[0][match.end("url") - match.start() :],
        text,
    )


def probe_urls(text, suffix):
    pairs = []
    if suffix == ".sources":
        for stanza in re.split(r"\n[ \t]*\n", text):
            uris, suites, enabled = (
                field(k).search(stanza) for k in ("URIs", "Suites", "Enabled")
            )
            if (
                not uris
                or not suites
                or (enabled and enabled[0].split(":", 1)[1].strip().lower() == "no")
            ):
                continue
            pairs.extend(
                (u, s)
                for u in uris[0].split(":", 1)[1].split()
                for s in suites[0].split(":", 1)[1].split()
            )
    else:
        pairs = [(m["url"], m["suite"]) for m in SOURCE_LINE.finditer(text)]
    for url, suite in pairs:
        host = official_host(url)
        if host and re.fullmatch(r"[a-z][a-z0-9-]*", suite):
            yield host, url.rstrip("/") + "/dists/" + suite + "/InRelease"


def direct_probe(url):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )
        with opener.open(url, timeout=4) as response:
            return response.status == 200 and response.read(64).startswith(
                b"-----BEGIN PGP SIGNED MESSAGE-----"
            )
    except Exception:
        return False


def source_routes(apt_dir, staging, *, probe=direct_probe):
    parts = staging / "sources.list.d"
    parts.mkdir()
    # APT reads source definitions as root; keep the temporary copy private.
    main = staging / "sources.list"
    main.write_text("")
    paths = [apt_dir / "sources.list"]
    paths += sorted(
        p
        for p in (apt_dir / "sources.list.d").glob("*")
        if p.suffix in {".sources", ".list"}
    )
    changed = False
    candidates = {}
    for path in paths:
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = https_sources(original, path.suffix)
        changed |= updated != original
        destination = main if path == apt_dir / "sources.list" else parts / path.name
        destination.write_text(updated, encoding="utf-8")
        for host, url in probe_urls(updated, path.suffix):
            candidates.setdefault(host, url)
    direct = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for host, reachable in zip(
            candidates, executor.map(probe, candidates.values())
        ):
            if reachable:
                for scheme in ("http", "https"):
                    direct += ["-o", f"Acquire::{scheme}::Proxy::{host}=DIRECT"]
    source_options = [
        "-o",
        f"Dir::Etc::sourcelist={main}",
        "-o",
        f"Dir::Etc::sourceparts={parts}",
    ]
    routes = []
    if direct:
        routes.append(("official HTTPS / probed direct hosts", source_options + direct))
    if changed or direct:
        routes.append(("official HTTPS / inherited proxy", source_options))
    routes.append(("original sources / inherited proxy", []))
    return routes


def install(
    packages, *, apt_dir=Path("/etc/apt"), run=subprocess.run, probe=direct_probe
):
    if not packages or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", p) for p in packages
    ):
        raise ValueError("Expected one or more APT package names")
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    with tempfile.TemporaryDirectory(prefix="tb-host-apt-") as temporary:
        routes = source_routes(apt_dir, Path(temporary), probe=probe)
        for label, options in routes:
            print(f"[terminal-bench-host-apt] {label}", file=sys.stderr, flush=True)
            base = ["apt-get", *APT_OPTIONS, *options]
            # Error-Mode=any prevents a partial/failed update from looking healthy.
            if run([*base, "update"], env=env).returncode:
                continue
            if not run(
                [*base, "install", "-y", "--no-install-recommends", *packages], env=env
            ).returncode:
                return
    raise RuntimeError(
        "PRoot host package installation failed on all APT routes; signature checks were retained. "
        "No SGLang was started or benchmark task claimed by this bootstrap."
    )


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[terminal-bench-host-apt] {exc}", file=sys.stderr)
        raise SystemExit(1)
