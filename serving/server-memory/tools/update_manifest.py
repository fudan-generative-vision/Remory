#!/usr/bin/env python3
"""Rehash the reviewed inference payload; never import or capture live runs."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src/recursive_memory/stack/payload"
manifest_file = root / "manifest.json"
manifest = json.loads(manifest_file.read_text())
entries = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise ValueError(f"payload symlink: {path}")
    if not path.is_file() or path == manifest_file or "__pycache__" in path.parts:
        continue
    if path.suffix not in {
        ".py",
        ".json",
        ".toml",
        ".sh",
        ".c",
        ".h",
        ".txt",
        ".j2",
        ".jinja",
        ".jinja2",
    } and path.name not in {"LICENSE", "SPECFORGE_LICENSE"}:
        raise ValueError(f"unexpected payload file: {path}")
    name = path.relative_to(root).as_posix()
    if any(
        part in {"history", "tests", "jobs", "source", "dataset", "cache"}
        for part in path.parts
    ):
        raise ValueError(f"out-of-scope payload: {name}")
    entries[name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "executable": path.suffix == ".sh",
    }
manifest["files"] = entries
manifest["portability"] = (
    "TB/AutomationBench/JobBench/BrowseComp100 native-compact inference closure; public upstream SGLang overlay; configurable pinned binaries"
)
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Recorded {len(entries)} runtime source files.")
