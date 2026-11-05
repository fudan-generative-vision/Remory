"""Content-free benchmark fingerprints for training-data decontamination.

The index intentionally stores only normalized hashes, hashed token shingles,
repository identifiers, and benchmark source markers.  It never stores an
evaluation prompt, answer, patch, test, verifier, or trajectory.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\W\s]", re.IGNORECASE | re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def normalize_text(value: Any) -> str:
    """Canonicalize text for exact and near-duplicate matching."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def normalize_repository(value: Any) -> str:
    """Return a stable ``owner/repo``-style identifier where possible."""

    repo = normalize_text(value)
    repo = re.sub(r"^(?:https?://)?(?:www\.)?github\.com/", "", repo)
    repo = re.sub(r"^(?:git@github\.com:|ssh://git@github\.com/)", "", repo)
    repo = repo.split("?", 1)[0].split("#", 1)[0].strip(" /")
    if repo.endswith(".git"):
        repo = repo[:-4]
    # Several SWE datasets encode owner/repo as owner__repo.
    if "/" not in repo and "__" in repo:
        repo = repo.replace("__", "/", 1)
    return repo


def text_hash(value: Any) -> str:
    return stable_digest(normalize_text(value))


def source_id_hash(source: Any, source_id: Any) -> str:
    return stable_digest(f"{normalize_text(source)}\x1f{normalize_text(source_id)}")


def token_shingle_hashes(value: Any, *, width: int = 8) -> frozenset[str]:
    """Hash token shingles so the index contains no recoverable prompt text."""

    tokens = TOKEN_RE.findall(normalize_text(value))
    if len(tokens) < width:
        return frozenset()
    return frozenset(
        stable_digest("\x1f".join(tokens[index : index + width]))[:16]
        for index in range(len(tokens) - width + 1)
    )


@dataclass(frozen=True)
class Match:
    reason: str
    benchmark: str = ""
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"reason": self.reason}
        if self.benchmark:
            result["benchmark"] = self.benchmark
        if self.similarity is not None:
            result["similarity"] = round(self.similarity, 6)
        return result


@dataclass
class FingerprintDocument:
    benchmark: str
    text_hash: str
    shingles: frozenset[str]


@dataclass
class DecontaminationIndex:
    """In-memory index used by builders and validation scans."""

    blocked_repositories: dict[str, str] = field(default_factory=dict)
    blocked_source_markers: tuple[str, ...] = ()
    blocked_text_markers: tuple[str, ...] = ()
    source_hashes: dict[str, str] = field(default_factory=dict)
    exact_text_hashes: dict[str, str] = field(default_factory=dict)
    documents: list[FingerprintDocument] = field(default_factory=list)
    shingle_width: int = 8
    near_threshold: float = 0.80
    min_shared_shingles: int = 8

    def __post_init__(self) -> None:
        postings: dict[str, list[int]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            for shingle in document.shingles:
                postings[shingle].append(index)
        self._postings = dict(postings)

    @classmethod
    def from_path(cls, path: str | Path) -> "DecontaminationIndex":
        payload = json.loads(Path(path).read_text())
        policy = payload.get("policy", {})
        blocked_repositories: dict[str, str] = {}
        source_hashes: dict[str, str] = {}
        exact_text_hashes: dict[str, str] = {}
        documents: list[FingerprintDocument] = []
        for benchmark in payload.get("benchmarks", []):
            name = str(benchmark.get("name") or "unknown")
            for repo in benchmark.get("repositories", []):
                normalized = normalize_repository(repo)
                if normalized:
                    blocked_repositories[normalized] = name
            for digest in benchmark.get("source_id_hashes", []):
                source_hashes[str(digest)] = name
            for item in benchmark.get("documents", []):
                digest = str(item.get("text_hash") or "")
                if digest:
                    exact_text_hashes[digest] = name
                shingles = frozenset(str(value) for value in item.get("shingles", []))
                if digest and shingles:
                    documents.append(FingerprintDocument(name, digest, shingles))
        return cls(
            blocked_repositories=blocked_repositories,
            blocked_source_markers=tuple(
                normalize_text(value)
                for value in payload.get("blocked_source_markers", [])
                if normalize_text(value)
            ),
            blocked_text_markers=tuple(
                normalize_text(value)
                for value in payload.get("blocked_text_markers", [])
                if normalize_text(value)
            ),
            source_hashes=source_hashes,
            exact_text_hashes=exact_text_hashes,
            documents=documents,
            shingle_width=int(policy.get("shingle_width", 8)),
            near_threshold=float(policy.get("near_threshold", 0.80)),
            min_shared_shingles=int(policy.get("min_shared_shingles", 8)),
        )

    def screen(self, record: Mapping[str, Any]) -> list[Match]:
        """Return every benchmark-contamination reason for one training row."""

        matches: list[Match] = []
        source = normalize_text(record.get("source"))
        prompt = normalize_text(record.get("prompt"))
        repo = normalize_repository(record.get("repo"))

        if repo and repo in self.blocked_repositories:
            matches.append(Match("blocked_repository", self.blocked_repositories[repo]))
        for marker in self.blocked_source_markers:
            if marker in source:
                matches.append(Match("benchmark_source_marker", marker))
        for marker in self.blocked_text_markers:
            if marker in prompt:
                matches.append(Match("benchmark_name_in_prompt", marker))

        digest = source_id_hash(source, record.get("source_id"))
        if digest in self.source_hashes:
            matches.append(Match("exact_source_id", self.source_hashes[digest]))

        prompt_digest = stable_digest(prompt)
        if prompt_digest in self.exact_text_hashes:
            matches.append(Match("exact_prompt", self.exact_text_hashes[prompt_digest], 1.0))
            return _unique_matches(matches)

        shingles = token_shingle_hashes(prompt, width=self.shingle_width)
        if len(shingles) < self.min_shared_shingles or not self.documents:
            return _unique_matches(matches)
        candidate_counts: Counter[int] = Counter()
        for shingle in shingles:
            candidate_counts.update(self._postings.get(shingle, ()))
        for document_index, shared in candidate_counts.items():
            if shared < self.min_shared_shingles:
                continue
            document = self.documents[document_index]
            union = len(shingles) + len(document.shingles) - shared
            jaccard = shared / union if union else 0.0
            containment = shared / min(len(shingles), len(document.shingles))
            # Containment catches a copied evaluation prompt wrapped in generic
            # agent instructions; Jaccard catches similarly sized rewrites.
            similarity = max(jaccard, containment)
            if similarity >= self.near_threshold:
                matches.append(Match("near_prompt", document.benchmark, similarity))
        return _unique_matches(matches)


def _unique_matches(matches: Iterable[Match]) -> list[Match]:
    result: list[Match] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        key = (match.reason, match.benchmark)
        if key not in seen:
            seen.add(key)
            result.append(match)
    return result


def fingerprint_document(
    benchmark: str,
    text: Any,
    *,
    shingle_width: int = 8,
) -> dict[str, Any]:
    """Create a content-free serialized fingerprint for one eval prompt."""

    normalized = normalize_text(text)
    return {
        "benchmark": benchmark,
        "text_hash": stable_digest(normalized),
        "shingles": sorted(token_shingle_hashes(normalized, width=shingle_width)),
    }


def combine_text_fields(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return "\n\n".join(str(record.get(field) or "") for field in fields).strip()
