"""Official BrowseComp encrypted dataset cache and phase-separated access.

The official CSV stores ``problem`` and ``answer`` as base64-encoded XOR
ciphertext.  The password is the row's ``canary`` and the keystream is the
SHA-256 digest of that password repeated to the ciphertext length, matching
OpenAI's ``simple-evals/browsecomp_eval.py`` contract.

Generation code should consume :func:`generation_sample`, whose return type has
no answer field.  Reference answers are decrypted only through the explicitly
named :func:`grading_reference` / :func:`decrypt_answer` grade-phase APIs.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .artifacts import (
    ArtifactCorruptionError,
    PRIVATE_FILE_MODE,
    atomic_write_bytes,
    atomic_write_json,
    ensure_private_dir,
    read_json_object,
    sha256_bytes,
    sha256_file,
    stable_opaque_sample_id,
)


OFFICIAL_BROWSECOMP_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "browse_comp_test_set.csv"
)
DEFAULT_DATASET_FILENAME = "browse_comp_test_set.csv"
DATASET_CACHE_SCHEMA = "browsecomp_dataset_cache_v1"
REQUIRED_COLUMNS = frozenset({"problem", "answer", "problem_topic", "canary"})
DEFAULT_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class DatasetError(RuntimeError):
    """Base class for BrowseComp dataset failures."""


class DatasetCacheError(DatasetError):
    """The private encrypted dataset cache is missing or inconsistent."""


class DatasetFormatError(DatasetError):
    """The encrypted CSV does not implement the official row schema."""


class DatasetDecryptionError(DatasetError):
    """One encrypted field could not be decrypted under its row canary."""


@dataclass(frozen=True, slots=True)
class DatasetCacheRecord:
    cache_dir: Path
    csv_path: Path
    metadata_path: Path
    source_url: str
    sha256: str
    size_bytes: int
    row_count: int
    etag: str | None = None
    last_modified: str | None = None
    from_cache: bool = False
    not_modified: bool = False


# Define the ciphertext-bearing dataclass explicitly so repr never logs release
# contents or the decryption canary.
@dataclass(frozen=True, slots=True)
class EncryptedBrowseCompRow:
    row_index: int
    sample_id: str
    encrypted_problem: str = field(repr=False)
    encrypted_answer: str = field(repr=False)
    canary: str = field(repr=False)
    problem_topic: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class GenerationSample:
    """The only dataset projection that should enter an agent episode."""

    row_index: int
    sample_id: str
    question: str = field(repr=False)
    problem_topic: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class GradingReference:
    """An answer-bearing value produced only in the explicit grade phase."""

    row_index: int
    sample_id: str
    answer: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class EncryptedBrowseCompDataset:
    rows: tuple[EncryptedBrowseCompRow, ...]
    sha256: str
    source_path: Path

    @classmethod
    def from_csv(
        cls,
        source: str | Path | DatasetCacheRecord,
        *,
        expected_sha256: str | None = None,
    ) -> "EncryptedBrowseCompDataset":
        path = (
            source.csv_path if isinstance(source, DatasetCacheRecord) else Path(source)
        )
        rows = load_encrypted_rows(path, expected_sha256=expected_sha256)
        return cls(rows=rows, sha256=sha256_file(path), source_path=path)


def _cache_paths(cache_dir: str | Path) -> tuple[Path, Path, Path]:
    root = ensure_private_dir(cache_dir)
    csv_path = root / DEFAULT_DATASET_FILENAME
    metadata_path = root / f"{DEFAULT_DATASET_FILENAME}.metadata.json"
    return root, csv_path, metadata_path


def _validate_csv_payload(payload: bytes) -> int:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError("BrowseComp CSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = set(reader.fieldnames or ())
    missing = sorted(REQUIRED_COLUMNS - fields)
    if missing:
        raise DatasetFormatError(
            "BrowseComp CSV is missing required columns: " + ", ".join(missing)
        )
    row_count = 0
    for row_count, row in enumerate(reader, start=1):
        if None in row:
            raise DatasetFormatError(f"malformed encrypted CSV row {row_count}")
        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if not isinstance(value, str) or not value:
                raise DatasetFormatError(
                    f"encrypted CSV row {row_count} has an empty {column!r} field"
                )
    if row_count == 0:
        raise DatasetFormatError("BrowseComp CSV contains no rows")
    return row_count


def _load_cached_record(
    cache_dir: str | Path,
    *,
    expected_sha256: str | None,
) -> DatasetCacheRecord | None:
    root, csv_path, metadata_path = _cache_paths(cache_dir)
    csv_exists = csv_path.is_file()
    metadata_exists = metadata_path.is_file()
    if not csv_exists and not metadata_exists:
        return None
    if csv_exists != metadata_exists:
        raise DatasetCacheError("encrypted dataset cache is only partially committed")

    try:
        metadata = read_json_object(metadata_path)
    except ArtifactCorruptionError as exc:
        raise DatasetCacheError(
            "encrypted dataset cache metadata is unreadable"
        ) from exc
    if metadata.get("schema") != DATASET_CACHE_SCHEMA:
        raise DatasetCacheError(
            "encrypted dataset cache metadata has an unknown schema"
        )
    actual_sha256 = sha256_file(csv_path)
    recorded_sha256 = metadata.get("sha256")
    if not isinstance(recorded_sha256, str) or actual_sha256 != recorded_sha256:
        raise DatasetCacheError(
            "encrypted dataset cache SHA-256 does not match metadata"
        )
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise DatasetCacheError(
            "encrypted dataset cache does not match the requested SHA-256"
        )
    actual_size = csv_path.stat().st_size
    if metadata.get("size_bytes") != actual_size:
        raise DatasetCacheError("encrypted dataset cache size does not match metadata")
    actual_rows = _validate_csv_payload(csv_path.read_bytes())
    if metadata.get("row_count") != actual_rows:
        raise DatasetCacheError(
            "encrypted dataset cache row count does not match metadata"
        )
    source_url = metadata.get("source_url")
    if not isinstance(source_url, str) or not source_url:
        raise DatasetCacheError("encrypted dataset cache has no source URL")

    # Repair an accidentally permissive mode without ever exposing plaintext.
    os.chmod(csv_path, PRIVATE_FILE_MODE)
    os.chmod(metadata_path, PRIVATE_FILE_MODE)
    return DatasetCacheRecord(
        cache_dir=root,
        csv_path=csv_path,
        metadata_path=metadata_path,
        source_url=source_url,
        sha256=actual_sha256,
        size_bytes=actual_size,
        row_count=actual_rows,
        etag=metadata.get("etag") if isinstance(metadata.get("etag"), str) else None,
        last_modified=(
            metadata.get("last_modified")
            if isinstance(metadata.get("last_modified"), str)
            else None
        ),
        from_cache=True,
    )


def _response_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def ensure_dataset_cached(
    cache_dir: str | Path,
    *,
    url: str = OFFICIAL_BROWSECOMP_URL,
    expected_sha256: str | None = None,
    refresh: bool = False,
    offline: bool = False,
    timeout_seconds: float = 120.0,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    opener: Callable[..., Any] | None = None,
) -> DatasetCacheRecord:
    """Return a verified private cache, optionally refreshing it with ETag.

    A normal repeated invocation is fully local.  ``refresh=True`` sends
    ``If-None-Match``/``If-Modified-Since`` and handles HTTP 304 without
    rewriting the encrypted CSV.
    """

    if expected_sha256 is not None:
        normalized = expected_sha256.lower()
        if len(normalized) != 64 or any(
            ch not in "0123456789abcdef" for ch in normalized
        ):
            raise ValueError(
                "expected_sha256 must be a 64-character hexadecimal digest"
            )
        expected_sha256 = normalized
    if max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")

    try:
        cached = _load_cached_record(cache_dir, expected_sha256=expected_sha256)
    except DatasetCacheError:
        if offline:
            raise
        cached = None

    if cached is not None and cached.source_url != url:
        if offline:
            raise DatasetCacheError("cached dataset source URL differs in offline mode")
        cached = None
    if cached is not None and (offline or not refresh):
        return cached
    if offline:
        raise DatasetCacheError("no verified encrypted BrowseComp cache is available")

    root, csv_path, metadata_path = _cache_paths(cache_dir)
    headers = {"Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.1"}
    if cached is not None and cached.etag:
        headers["If-None-Match"] = cached.etag
    if cached is not None and cached.last_modified:
        headers["If-Modified-Since"] = cached.last_modified
    request = urllib.request.Request(url, headers=headers, method="GET")
    open_request = opener or urllib.request.urlopen
    try:
        response = open_request(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and cached is not None:
            exc.close()
            return replace(cached, not_modified=True)
        raise DatasetCacheError(
            f"unable to download encrypted BrowseComp CSV: HTTP {exc.code}"
        ) from exc
    except OSError as exc:
        raise DatasetCacheError("unable to download encrypted BrowseComp CSV") from exc

    with response:
        status = getattr(response, "status", None)
        if status is None and callable(getattr(response, "getcode", None)):
            status = response.getcode()
        if status == 304:
            if cached is None:
                raise DatasetCacheError(
                    "server returned HTTP 304 without a local cache"
                )
            return replace(cached, not_modified=True)
        if status not in (None, 200):
            raise DatasetCacheError(
                f"unable to download encrypted BrowseComp CSV: HTTP {status}"
            )
        content_length = _response_header(response, "Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise DatasetCacheError(
                    "dataset response has an invalid Content-Length"
                ) from exc
            if declared_size > max_download_bytes:
                raise DatasetCacheError(
                    "encrypted BrowseComp CSV exceeds the download limit"
                )
        payload = response.read(max_download_bytes + 1)
        if len(payload) > max_download_bytes:
            raise DatasetCacheError(
                "encrypted BrowseComp CSV exceeds the download limit"
            )
        etag = _response_header(response, "ETag")
        last_modified = _response_header(response, "Last-Modified")

    dataset_sha256 = sha256_bytes(payload)
    if expected_sha256 is not None and dataset_sha256 != expected_sha256:
        raise DatasetCacheError("downloaded BrowseComp CSV failed its SHA-256 pin")
    row_count = _validate_csv_payload(payload)
    metadata = {
        "schema": DATASET_CACHE_SCHEMA,
        "source_url": url,
        "sha256": dataset_sha256,
        "size_bytes": len(payload),
        "row_count": row_count,
        "etag": etag,
        "last_modified": last_modified,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }
    # Data first, metadata second: a crash between them is detected as a
    # partial/mismatched cache and never consumed silently.
    atomic_write_bytes(csv_path, payload)
    atomic_write_json(metadata_path, metadata)
    return DatasetCacheRecord(
        cache_dir=root,
        csv_path=csv_path,
        metadata_path=metadata_path,
        source_url=url,
        sha256=dataset_sha256,
        size_bytes=len(payload),
        row_count=row_count,
        etag=etag,
        last_modified=last_modified,
        from_cache=False,
    )


def derive_key(password: str, length: int) -> bytes:
    """Match OpenAI simple-evals' repeated SHA-256 XOR keystream."""

    if length < 0:
        raise ValueError("length must not be negative")
    key = hashlib.sha256(password.encode("utf-8")).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt_field(
    ciphertext_b64: str, password: str, *, field_name: str, sample_id: str
) -> str:
    try:
        encrypted = base64.b64decode(ciphertext_b64, validate=True)
        key = derive_key(password, len(encrypted))
        plaintext = bytes(left ^ right for left, right in zip(encrypted, key))
        return plaintext.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise DatasetDecryptionError(
            f"unable to decrypt {field_name} for opaque sample {sample_id}"
        ) from exc


def decrypt_problem(row: EncryptedBrowseCompRow) -> str:
    return _decrypt_field(
        row.encrypted_problem,
        row.canary,
        field_name="problem",
        sample_id=row.sample_id,
    )


def decrypt_answer(row: EncryptedBrowseCompRow) -> str:
    """Decrypt a reference answer; call only from the explicit grade phase."""

    return _decrypt_field(
        row.encrypted_answer,
        row.canary,
        field_name="answer",
        sample_id=row.sample_id,
    )


def generation_sample(row: EncryptedBrowseCompRow) -> GenerationSample:
    """Project one encrypted row for generation without touching its answer."""

    return GenerationSample(
        row_index=row.row_index,
        sample_id=row.sample_id,
        question=decrypt_problem(row),
        problem_topic=row.problem_topic,
    )


def grading_reference(row: EncryptedBrowseCompRow) -> GradingReference:
    """Enter the explicit grade phase for exactly one sample."""

    return GradingReference(
        row_index=row.row_index,
        sample_id=row.sample_id,
        answer=decrypt_answer(row),
    )


def load_encrypted_rows(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[EncryptedBrowseCompRow, ...]:
    """Parse encrypted rows without decrypting either protected field."""

    source = Path(path)
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise DatasetCacheError("encrypted BrowseComp CSV failed its SHA-256 pin")
    rows: list[EncryptedBrowseCompRow] = []
    seen_ids: set[str] = set()
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetCacheError("unable to open encrypted BrowseComp CSV") from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise DatasetFormatError(
                "BrowseComp CSV is missing required columns: " + ", ".join(missing)
            )
        for row_index, raw in enumerate(reader):
            if None in raw:
                raise DatasetFormatError(f"malformed encrypted CSV row {row_index + 1}")
            values: dict[str, str] = {}
            for column in REQUIRED_COLUMNS:
                value = raw.get(column)
                if not isinstance(value, str) or not value:
                    raise DatasetFormatError(
                        f"encrypted CSV row {row_index + 1} has an empty {column!r} field"
                    )
                values[column] = value
            sample_id = stable_opaque_sample_id(values["problem"])
            if sample_id in seen_ids:
                raise DatasetFormatError(
                    f"encrypted CSV contains duplicate opaque sample id {sample_id}"
                )
            seen_ids.add(sample_id)
            rows.append(
                EncryptedBrowseCompRow(
                    row_index=row_index,
                    sample_id=sample_id,
                    encrypted_problem=values["problem"],
                    encrypted_answer=values["answer"],
                    canary=values["canary"],
                    problem_topic=values["problem_topic"],
                )
            )
    if not rows:
        raise DatasetFormatError("BrowseComp CSV contains no rows")
    return tuple(rows)
