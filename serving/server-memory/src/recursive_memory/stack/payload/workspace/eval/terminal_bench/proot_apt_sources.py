"""Keep EOL Debian task images installable without disabling APT signatures.

Debian 11 LTS ended 2026-08-31. Its last security Release expires on
2026-09-07. Pin that source to the official last-day snapshot and exempt only
that immutable source from Valid-Until, as documented by snapshot.debian.org.
Never change host sources, supported releases, third-party repositories, or
APT's global trust/date settings.
"""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import uuid


POLICY = "debian_bullseye_security_eol_snapshot_20260831_v1"
SNAPSHOT = "snapshot.debian.org/archive/debian-security/20260831T235959Z/"
SUITE = "bullseye-security"
_ORIGINS = {
    f"{scheme}://{host}{path}"
    for scheme in ("http", "https")
    for host, path in (("deb.debian.org", "/debian-security"),
                       ("security.debian.org", "/debian-security"),
                       ("security.debian.org", ""))
}
_LINE = re.compile(r"^(?P<kind>\s*deb(?:-src)?\s+)(?:\[(?P<options>[^\]\n]*)\]\s+)?"
                   r"(?P<url>\S+)(?P<gap>\s+)(?P<suite>\S+)(?P<rest>[^\n]*)$", re.M)


def _list_sources(text, snapshot):
    def replace(match):
        if match["url"].rstrip("/") not in _ORIGINS or match["suite"] != SUITE:
            return match[0]
        options = [v for v in (match["options"] or "").split()
                   if not v.lower().startswith("check-valid-until=")]
        options.append("check-valid-until=no")
        return (match["kind"] + "[" + " ".join(options) + "] " + snapshot
                + match["gap"] + SUITE + match["rest"])
    return _LINE.sub(replace, text)


def _field(name):
    return re.compile(r"^" + name + r":[^\n]*(?:\n[ \t]+[^\n]*)*", re.M | re.I)


def _deb822_sources(text, snapshot):
    parts = re.split(r"(\n[ \t]*\n)", text)
    for index in range(0, len(parts), 2):
        stanza = parts[index]
        uris, suites = _field("URIs").search(stanza), _field("Suites").search(stanza)
        enabled = _field("Enabled").search(stanza)
        if not uris or not suites or (enabled and enabled[0].split(":", 1)[1].strip().lower() == "no"):
            continue
        urls, names = uris[0].split(":", 1)[1].split(), suites[0].split(":", 1)[1].split()
        if SUITE not in names or not any(url.rstrip("/") in _ORIGINS for url in urls):
            continue
        if names != [SUITE] or not all(url.rstrip("/") in _ORIGINS for url in urls):
            raise RuntimeError("Mixed Debian security deb822 stanza needs explicit source separation")
        stanza = _field("URIs").sub("URIs: " + snapshot, stanza)
        if _field("Check-Valid-Until").search(stanza):
            stanza = _field("Check-Valid-Until").sub("Check-Valid-Until: no", stanza)
        else:
            stanza = stanza.rstrip("\n") + "\nCheck-Valid-Until: no\n"
        parts[index] = stanza
    return "".join(parts)


def prepare_bullseye_security_snapshot(rootfs, distro, *, now=None):
    if distro.get("ID", "").lower() != "debian" or distro.get("VERSION_ID") != "11":
        return None
    now = now or datetime.now(timezone.utc)
    if now < datetime(2026, 9, 1, tzinfo=timezone.utc):
        return None
    root = Path(rootfs).resolve()
    if root == Path("/"):
        raise RuntimeError("Refuse to modify host APT sources")
    # Minimal task images may lack ca-certificates; HTTP still verifies Debian
    # Release signatures and every package hash. Never set Verify-Peer=false.
    scheme = "https" if (root / "etc/ssl/certs/ca-certificates.crt").is_file() else "http"
    snapshot = scheme + "://" + SNAPSHOT
    apt = root / "etc/apt"
    if not apt.resolve().is_relative_to(root):
        raise RuntimeError("Unsafe guest APT source directory")
    candidates = [apt / "sources.list"]
    parts = apt / "sources.list.d"
    if parts.is_dir():
        if not parts.resolve().is_relative_to(root):
            raise RuntimeError("Unsafe guest APT source directory")
        candidates += sorted(p for p in parts.iterdir() if p.suffix in {".list", ".sources"})
    changes = []
    for path in candidates:
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError("Unsafe guest APT source path")
        before = path.read_text()
        after = (_deb822_sources if path.suffix == ".sources" else _list_sources)(before, snapshot)
        if after != before:
            changes.append((path, before, after))
    # Validate all source files before changing any; retain original modes and
    # record before/after hashes in the host-owned provisioning receipt.
    records = []
    for path, before, after in changes:
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x") as handle:
                handle.write(after)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        records.append({"path": "/" + str(path.relative_to(root)),
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode()).hexdigest()})
    return {"policy": POLICY, "snapshot": snapshot, "suite": SUITE,
            "source_scoped_valid_until_exception": True,
            "signature_verification_unchanged": True, "changed_files": records}
