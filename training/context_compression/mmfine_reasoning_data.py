"""Strict, CPU-only data contracts for MMFineReasoning training.

The module deliberately has no dependency on the compressor or trainer.  It
does three things:

* validates immutable, finalized parquet shards and snapshots content-bound
  receipts for them;
* adapts rows into one VQA-completion event plus three text-reconstruction
  candidates; and
* exposes both the historical 1:1 completion/repeat queue and a repeat-only
  projected reader.  The latter never materializes the parquet image column.

Only :func:`write_validated_shard_manifest` writes manifests.  It writes a
same-directory temporary file, fsyncs it, and publishes it with ``os.replace``.
Source shards are always opened with ``O_NOFOLLOW`` and checked before and
after hashing/validation so a downloader can continue safely in parallel.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _datetime
import hashlib
import io
import json
import math
import os
import stat
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "mmfine_reasoning_validated_shards_v1"
REPEAT_MANIFEST_SCHEMA = "mmfine_reasoning_repeat_validated_shards_v1"
MANIFEST_VERSION = 1
REQUIRED_FIELDS = (
    "image",
    "question",
    "original_answer",
    "qwen3vl_235b_thinking_response",
    "caption",
    "source",
    "id",
    "is_consistent",
    "pass_rate",
)
REQUIRED_IMAGE_FIELDS = ("bytes", "path")
REPEAT_FIELDS = ("question", "original_answer", "caption")
_REQUIRED_LEAF_PATHS = (
    "image.bytes",
    "image.path",
    "question",
    "original_answer",
    "qwen3vl_235b_thinking_response",
    "caption",
    "source",
    "id",
    "is_consistent",
    "pass_rate",
)
_TEXT_FIELDS = (
    "question",
    "original_answer",
    "qwen3vl_235b_thinking_response",
    "caption",
    "source",
)
_ARTIFACT_MARKERS = (".incomplete", ".lock", ".partial", ".tmp", ".part")


class MMFineReasoningDataError(ValueError):
    """Base class for a violated data contract."""


class RowValidationError(MMFineReasoningDataError):
    """A parquet row cannot produce the required training events."""


class ShardValidationError(MMFineReasoningDataError):
    """A finalized parquet file violates the shard contract."""


class UnstableShardError(ShardValidationError):
    """A candidate is incomplete or changed while it was inspected."""


@dataclass(frozen=True)
class MMFineImage:
    """Content-addressed image payload from the parquet ``image`` struct."""

    bytes: bytes | None
    path: str | None
    sha256: str


@dataclass(frozen=True)
class MMFineReasoningRecord:
    """Strict canonical row used by both objectives."""

    sample_id: str
    source: str
    source_id: str | int
    image: MMFineImage
    question: str
    original_answer: str
    qwen3vl_235b_thinking_response: str
    caption: str
    # The regeneration corpus uses null for a small number of unadjudicated
    # rows.  Presence/type of the column is mandatory, but null is a legitimate
    # quality-metadata state rather than corrupt task content.
    is_consistent: bool | None
    pass_rate: float


@dataclass(frozen=True)
class MMFineRepeatRecord:
    """The text-only projection required by repeat pretraining."""

    sample_id: str
    source: str
    source_id: str | int
    question: str
    original_answer: str
    caption: str


@dataclass(frozen=True)
class VQACompletionEvent:
    """Compress image+question and complete the stored 235B response."""

    event_id: str
    sample_id: str
    source: str
    source_id: str | int
    image: MMFineImage
    question: str
    target: str
    objective: str = "vqa_completion"


@dataclass(frozen=True)
class TextRepeatEvent:
    """Compress and exactly reconstruct one self-contained text field."""

    event_id: str
    sample_id: str
    source: str
    source_id: str | int
    field: str
    text: str
    target: str
    objective: str = "text_repeat"


@dataclass(frozen=True)
class JointEventBundle:
    """All trainer-facing events derived from one source row."""

    completion: VQACompletionEvent
    repeats: tuple[TextRepeatEvent, ...]

    def repeat_for_rotation(self, rotation: int) -> TextRepeatEvent:
        if not self.repeats:  # pragma: no cover - strict rows always have repeats.
            raise RowValidationError("record has no non-empty repeat fields")
        seed = int(self.completion.sample_id[-16:], 16)
        return self.repeats[(seed + int(rotation)) % len(self.repeats)]


@dataclass(frozen=True)
class JointEventPair:
    """One completion and one rotated repeat event (objective ratio 1:1)."""

    completion: VQACompletionEvent
    repeat: TextRepeatEvent


@dataclass(frozen=True)
class ShardReceipt:
    """Immutable content and parquet-metadata receipt for one ready shard."""

    path: str
    size_bytes: int
    sha256: str
    rows: int
    schema: tuple[dict[str, Any], ...]
    schema_sha256: str
    null_counts: dict[str, int]
    image_decode_samples: tuple[dict[str, Any], ...]
    device: int
    inode: int
    mtime_ns: int

    @property
    def queue_key(self) -> str:
        return _canonical_sha256(
            {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}
        )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ShardReceipt":
        return cls(
            path=str(value["path"]),
            size_bytes=int(value["size_bytes"]),
            sha256=str(value["sha256"]),
            rows=int(value["rows"]),
            schema=tuple(dict(item) for item in value["schema"]),
            schema_sha256=str(value["schema_sha256"]),
            null_counts={str(k): int(v) for k, v in value["null_counts"].items()},
            image_decode_samples=tuple(
                dict(item) for item in value.get("image_decode_samples", ())
            ),
            device=int(value["device"]),
            inode=int(value["inode"]),
            mtime_ns=int(value["mtime_ns"]),
        )


@dataclass(frozen=True)
class ReadyShardQueue:
    """An in-memory queue view over an immutable validated manifest."""

    root: Path
    shards: tuple[ShardReceipt, ...]

    def __len__(self) -> int:
        return len(self.shards)

    def __iter__(self) -> Iterator[ShardReceipt]:
        return iter(self.shards)

    def paths(self) -> tuple[Path, ...]:
        return tuple(self.root / receipt.path for receipt in self.shards)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        consumed_queue_keys: Iterable[str] = (),
    ) -> "ReadyShardQueue":
        validate_manifest(manifest)
        consumed = {str(value) for value in consumed_queue_keys}
        receipts = tuple(
            receipt
            for receipt in (
                ShardReceipt.from_mapping(value) for value in manifest["shards"]
            )
            if receipt.queue_key not in consumed
        )
        return cls(root=Path(str(manifest["data_root"])), shards=receipts)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    material = dict(manifest)
    material.pop("manifest_sha256", None)
    return _canonical_sha256(material)


def _typed_source_id(value: str | int) -> dict[str, Any]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise RowValidationError("id must be a string or integer (not bool)")
    if isinstance(value, str) and not value:
        raise RowValidationError("id must not be empty")
    return {"type": "str" if isinstance(value, str) else "int", "value": value}


def stable_sample_id(*, source: str, source_id: str | int, image_sha256: str) -> str:
    """Bind sample identity to source/id and the actual image content."""

    if not isinstance(source, str) or not source:
        raise RowValidationError("source must be a non-empty string")
    if len(image_sha256) != 64:
        raise RowValidationError("image_sha256 must be a full SHA256 hex digest")
    try:
        int(image_sha256, 16)
    except ValueError as error:
        raise RowValidationError("image_sha256 is not hexadecimal") from error
    digest = _canonical_sha256(
        {
            "source": source,
            "id": _typed_source_id(source_id),
            "image_sha256": image_sha256.lower(),
        }
    )
    return f"mmfr-{digest}"


def stable_repeat_sample_id(
    *,
    source: str,
    source_id: str | int,
    question: str,
    original_answer: str,
    caption: str,
    allow_empty_fields: bool = False,
) -> str:
    """Bind repeat identity only to fields consumed by repeat training."""

    if not isinstance(source, str) or not source:
        raise RowValidationError("source must be a non-empty string")
    values = {
        "question": question,
        "original_answer": original_answer,
        "caption": caption,
    }
    for name, value in values.items():
        if not isinstance(value, str) or (not allow_empty_fields and not value.strip()):
            raise RowValidationError(f"{name} must be a non-empty string")
    digest = _canonical_sha256(
        {
            "source": source,
            "id": _typed_source_id(source_id),
            "repeat_fields": values,
        }
    )
    return f"mmfr-repeat-{digest}"


def normalize_repeat_question(question: str) -> str:
    """Remove only a leading image marker for the question repeat target."""

    if not isinstance(question, str):
        raise RowValidationError("question must be a string")
    if question.startswith("<image>"):
        return question[len("<image>") :].lstrip()
    return question


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise UnstableShardError(f"candidate disappeared: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise UnstableShardError(f"candidate is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnstableShardError(
            f"cannot safely open candidate: {path}: {error}"
        ) from error
    opened = os.fstat(descriptor)
    if not _same_file_identity(before, opened):
        os.close(descriptor)
        raise UnstableShardError(f"candidate identity changed while opened: {path}")
    return descriptor, opened


def _read_fd(descriptor: int, size: int) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise UnstableShardError("file became shorter while read")
        chunks.append(chunk)
        digest.update(chunk)
        offset += len(chunk)
    return b"".join(chunks), digest.hexdigest()


def _sha256_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise UnstableShardError("file became shorter while hashed")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _check_unchanged(path: Path, descriptor: int, before: os.stat_result) -> None:
    opened_after = os.fstat(descriptor)
    try:
        path_after = os.lstat(path)
    except FileNotFoundError as error:
        raise UnstableShardError(
            f"candidate disappeared while inspected: {path}"
        ) from error
    if not _same_file_identity(before, opened_after) or not _same_file_identity(
        before, path_after
    ):
        raise UnstableShardError(f"candidate changed while inspected: {path}")


def _safe_image_path(path: str, image_root: Path | None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        if image_root is None:
            raise RowValidationError("relative image.path requires image_root")
        candidate = image_root / candidate
    absolute = Path(os.path.abspath(candidate))
    try:
        # O_NOFOLLOW protects the leaf; resolving the parent also prevents a
        # relative image path from escaping through an intermediate symlink.
        absolute = absolute.parent.resolve(strict=True) / absolute.name
    except FileNotFoundError as error:
        raise RowValidationError(
            f"image.path parent does not exist: {path!r}"
        ) from error
    if image_root is not None:
        root = image_root.resolve()
        try:
            absolute.relative_to(root)
        except ValueError as error:
            raise RowValidationError(
                f"image.path escapes image_root: {path!r}"
            ) from error
    return absolute


def _load_path_bytes(path: Path) -> bytes:
    try:
        descriptor, before = _open_regular_nofollow(path)
    except UnstableShardError as error:
        raise RowValidationError(str(error)) from error
    try:
        payload, _ = _read_fd(descriptor, before.st_size)
        try:
            _check_unchanged(path, descriptor, before)
        except UnstableShardError as error:
            raise RowValidationError(str(error)) from error
        return payload
    finally:
        os.close(descriptor)


def _adapt_image(value: Any, *, image_root: Path | None) -> MMFineImage:
    if not isinstance(value, Mapping):
        raise RowValidationError("image must be a {bytes, path} mapping")
    missing = [field for field in REQUIRED_IMAGE_FIELDS if field not in value]
    if missing:
        raise RowValidationError(f"image is missing fields: {missing}")
    raw_bytes = value.get("bytes")
    raw_path = value.get("path")
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    elif isinstance(raw_bytes, bytearray):
        raw_bytes = bytes(raw_bytes)
    if raw_bytes is not None and not isinstance(raw_bytes, bytes):
        raise RowValidationError("image.bytes must be binary or null")
    if raw_path is not None and (not isinstance(raw_path, str) or not raw_path):
        raise RowValidationError("image.path must be a non-empty string or null")
    if not raw_bytes and raw_path is None:
        raise RowValidationError("image must contain non-empty bytes or a path")
    payload = raw_bytes
    if not payload:
        payload = _load_path_bytes(_safe_image_path(raw_path, image_root))  # type: ignore[arg-type]
    assert payload is not None
    return MMFineImage(
        bytes=raw_bytes if raw_bytes else None,
        path=raw_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def adapt_row(
    row: Mapping[str, Any], *, image_root: str | os.PathLike[str] | None = None
) -> MMFineReasoningRecord:
    """Validate and canonicalize one MMFineReasoning parquet row."""

    if not isinstance(row, Mapping):
        raise RowValidationError("row must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise RowValidationError(f"row is missing required fields: {missing}")
    for field in _TEXT_FIELDS:
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            raise RowValidationError(f"{field} must be a non-empty string")
    source_id = row["id"]
    _typed_source_id(source_id)
    if row["is_consistent"] is not None and type(row["is_consistent"]) is not bool:
        raise RowValidationError("is_consistent must be bool or null")
    pass_rate = row["pass_rate"]
    if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
        raise RowValidationError("pass_rate must be numeric")
    pass_rate = float(pass_rate)
    if not math.isfinite(pass_rate) or not 0.0 <= pass_rate <= 1.0:
        raise RowValidationError("pass_rate must be finite and within [0, 1]")
    root = Path(image_root).resolve() if image_root is not None else None
    image = _adapt_image(row["image"], image_root=root)
    source = row["source"]
    return MMFineReasoningRecord(
        sample_id=stable_sample_id(
            source=source, source_id=source_id, image_sha256=image.sha256
        ),
        source=source,
        source_id=source_id,
        image=image,
        question=row["question"],
        original_answer=row["original_answer"],
        qwen3vl_235b_thinking_response=row["qwen3vl_235b_thinking_response"],
        caption=row["caption"],
        is_consistent=row["is_consistent"],
        pass_rate=pass_rate,
    )


def adapt_repeat_row(
    row: Mapping[str, Any], *, allow_empty_fields: bool = False
) -> MMFineRepeatRecord:
    """Project a parquet row to repeat fields without touching image bytes."""

    if not isinstance(row, Mapping):
        raise RowValidationError("row must be a mapping")
    required = ("source", "id", *REPEAT_FIELDS)
    missing = [field for field in required if field not in row]
    if missing:
        raise RowValidationError(f"repeat row is missing required fields: {missing}")
    source = row["source"]
    if not isinstance(source, str) or not source:
        raise RowValidationError("source must be a non-empty string")
    source_id = row["id"]
    _typed_source_id(source_id)
    question = row["question"]
    original_answer = row["original_answer"]
    caption = row["caption"]
    sample_id = stable_repeat_sample_id(
        source=source,
        source_id=source_id,
        question=question,
        original_answer=original_answer,
        caption=caption,
        allow_empty_fields=allow_empty_fields,
    )
    return MMFineRepeatRecord(
        sample_id=sample_id,
        source=source,
        source_id=source_id,
        question=question,
        original_answer=original_answer,
        caption=caption,
    )


def make_completion_event(record: MMFineReasoningRecord) -> VQACompletionEvent:
    return VQACompletionEvent(
        event_id=f"{record.sample_id}:completion",
        sample_id=record.sample_id,
        source=record.source,
        source_id=record.source_id,
        image=record.image,
        # Keep the source row verbatim.  The multimodal collator owns image-token
        # rendering; only the repeat objective normalizes the textual marker.
        question=record.question,
        target=record.qwen3vl_235b_thinking_response,
    )


def make_repeat_bundle(
    record: MMFineReasoningRecord | MMFineRepeatRecord,
    *,
    skip_empty_fields: bool = False,
) -> tuple[TextRepeatEvent, ...]:
    values = {
        "question": normalize_repeat_question(record.question),
        "original_answer": record.original_answer,
        "caption": record.caption,
    }
    events: list[TextRepeatEvent] = []
    for field in REPEAT_FIELDS:
        text = values[field]
        if not text.strip():
            if skip_empty_fields:
                continue
            raise RowValidationError(f"normalized repeat field {field} is empty")
        events.append(
            TextRepeatEvent(
                event_id=f"{record.sample_id}:repeat:{field}",
                sample_id=record.sample_id,
                source=record.source,
                source_id=record.source_id,
                field=field,
                text=text,
                target=text,
            )
        )
    return tuple(events)


def make_joint_event_bundle(record: MMFineReasoningRecord) -> JointEventBundle:
    return JointEventBundle(
        completion=make_completion_event(record), repeats=make_repeat_bundle(record)
    )


def iter_joint_event_pairs(
    records: Iterable[MMFineReasoningRecord], *, repeat_rotation: int
) -> Iterator[JointEventPair]:
    """Yield an exact 1:1 objective stream with deterministic field rotation."""

    for record in records:
        bundle = make_joint_event_bundle(record)
        yield JointEventPair(
            completion=bundle.completion,
            repeat=bundle.repeat_for_rotation(repeat_rotation),
        )


def iter_repeat_events(
    records: Iterable[MMFineRepeatRecord],
    *,
    repeat_rotation: int,
    skip_empty_fields: bool = False,
) -> Iterator[TextRepeatEvent]:
    """Yield deterministic candidate-slot order, optionally omitting empty slots."""

    for record in records:
        repeats = make_repeat_bundle(
            record,
            skip_empty_fields=skip_empty_fields,
        )
        if not repeats:
            raise RowValidationError("record has no non-empty normalized repeat fields")
        seed = int(record.sample_id[-16:], 16)
        start = (seed + int(repeat_rotation)) % len(REPEAT_FIELDS)
        by_field = {event.field: event for event in repeats}
        for offset in range(len(REPEAT_FIELDS)):
            field = REPEAT_FIELDS[(start + offset) % len(REPEAT_FIELDS)]
            event = by_field.get(field)
            if event is not None:
                yield event


def _schema_field_descriptor(field: Any) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    if hasattr(field.type, "num_fields"):
        children = [
            _schema_field_descriptor(field.type[index])
            for index in range(field.type.num_fields)
        ]
    result: dict[str, Any] = {
        "name": field.name,
        "type": str(field.type),
        "nullable": bool(field.nullable),
    }
    if children:
        result["children"] = children
    return result


def _validate_arrow_schema(schema: Any) -> tuple[dict[str, Any], ...]:
    try:
        import pyarrow as pa
    except ImportError as error:  # pragma: no cover - production dependency.
        raise RuntimeError("pyarrow is required for parquet snapshots") from error

    missing = [field for field in REQUIRED_FIELDS if schema.get_field_index(field) < 0]
    if missing:
        raise ShardValidationError(f"parquet schema is missing fields: {missing}")

    for name in _TEXT_FIELDS:
        field = schema.field(name)
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ShardValidationError(
                f"{name} must be an Arrow string, got {field.type}"
            )
    identifier = schema.field("id").type
    if not (
        pa.types.is_integer(identifier)
        or pa.types.is_string(identifier)
        or pa.types.is_large_string(identifier)
    ):
        raise ShardValidationError(f"id must be integer/string, got {identifier}")
    if not pa.types.is_boolean(schema.field("is_consistent").type):
        raise ShardValidationError("is_consistent must be Arrow bool")
    if not pa.types.is_floating(schema.field("pass_rate").type):
        raise ShardValidationError("pass_rate must be Arrow float")

    image = schema.field("image")
    if not pa.types.is_struct(image.type):
        raise ShardValidationError("image must be an Arrow struct")
    for name in REQUIRED_IMAGE_FIELDS:
        index = image.type.get_field_index(name)
        if index < 0:
            raise ShardValidationError(f"image struct is missing {name}")
    image_bytes = image.type.field("bytes").type
    if not (pa.types.is_binary(image_bytes) or pa.types.is_large_binary(image_bytes)):
        raise ShardValidationError(f"image.bytes must be binary, got {image_bytes}")
    image_path = image.type.field("path").type
    if not (pa.types.is_string(image_path) or pa.types.is_large_string(image_path)):
        raise ShardValidationError(f"image.path must be string, got {image_path}")
    return tuple(_schema_field_descriptor(field) for field in schema)


def _nested_value(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if current is None or not isinstance(current, Mapping):
            return None
        current = current.get(component)
    return current


def _scan_null_count(parquet: Any, path: str) -> int:
    top = path.split(".", 1)[0]
    count = 0
    for batch in parquet.iter_batches(
        batch_size=1024, columns=[top], use_threads=False
    ):
        values = batch.column(0).to_pylist()
        if "." not in path:
            count += sum(value is None for value in values)
        else:
            child = path.split(".", 1)[1]
            count += sum(_nested_value(value, child) is None for value in values)
    return count


def _parquet_null_counts(parquet: Any) -> dict[str, int]:
    metadata = parquet.metadata
    counts = {path: 0 for path in _REQUIRED_LEAF_PATHS}
    incomplete: set[str] = set()
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        columns = {
            row_group.column(index).path_in_schema: row_group.column(index)
            for index in range(row_group.num_columns)
        }
        for path in _REQUIRED_LEAF_PATHS:
            column = columns.get(path)
            statistics = None if column is None else column.statistics
            if statistics is None or not getattr(statistics, "has_null_count", False):
                incomplete.add(path)
            else:
                counts[path] += int(statistics.null_count)
    for path in incomplete:
        counts[path] = _scan_null_count(parquet, path)
    return counts


def _validate_null_counts(
    null_counts: Mapping[str, int], *, repeat_only: bool = False
) -> None:
    if repeat_only:
        required = {"question", "original_answer", "caption", "source", "id"}
        nonzero = {path: null_counts[path] for path in required if null_counts[path]}
        if nonzero:
            raise ShardValidationError(
                f"repeat-required parquet fields contain nulls: {nonzero}"
            )
        return
    always_nonnull = set(_REQUIRED_LEAF_PATHS) - {
        "image.bytes",
        "image.path",
        # Null means the consistency judge did not return a verdict.  It is a
        # recorded/filterable quality state; it does not prevent construction
        # of either training objective.
        "is_consistent",
    }
    nonzero = {path: null_counts[path] for path in always_nonnull if null_counts[path]}
    if nonzero:
        raise ShardValidationError(f"required parquet fields contain nulls: {nonzero}")
    # A conservative columnar proof that every row has usable image content:
    # one of the alternatives must be populated for every row.  Row adaptation
    # still enforces the OR contract independently.
    if min(null_counts["image.bytes"], null_counts["image.path"]) != 0:
        raise ShardValidationError(
            "cannot prove image bytes/path coverage from parquet null statistics"
        )


def _decode_image(payload: bytes) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - torchvision installs Pillow.
        raise RuntimeError("Pillow is required for image decode validation") from error
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            image_format = probe.format
            probe.verify()
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
            width, height = decoded.size
            mode = decoded.mode
    except Exception as error:
        raise ShardValidationError(
            f"sample image cannot be decoded: {error}"
        ) from error
    if width <= 0 or height <= 0:
        raise ShardValidationError("sample image has an empty dimension")
    return {"format": image_format, "width": width, "height": height, "mode": mode}


def _sample_images(
    parquet: Any,
    *,
    image_root: Path,
    sample_count: int,
) -> tuple[dict[str, Any], ...]:
    if sample_count < 0:
        raise ValueError("image sample count must be non-negative")
    if sample_count == 0:
        return ()
    samples: list[dict[str, Any]] = []
    row_offset = 0
    for batch in parquet.iter_batches(
        batch_size=32, columns=["image"], use_threads=False
    ):
        for local_index, value in enumerate(batch.column(0).to_pylist()):
            try:
                image = _adapt_image(value, image_root=image_root)
            except RowValidationError as error:
                raise ShardValidationError(
                    f"sample image at row {row_offset + local_index} is invalid: {error}"
                ) from error
            payload = image.bytes
            if payload is None:
                assert image.path is not None
                payload = _load_path_bytes(_safe_image_path(image.path, image_root))
            decoded = _decode_image(payload)
            samples.append(
                {
                    "row": row_offset + local_index,
                    "sha256": image.sha256,
                    **decoded,
                }
            )
            if len(samples) >= sample_count:
                return tuple(samples)
        row_offset += batch.num_rows
    raise ShardValidationError(
        f"requested {sample_count} image samples but shard contains only {len(samples)} rows"
    )


def _relative_shard_path(path: Path, root: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ShardValidationError(f"shard is outside data root: {path}") from error
    return relative.as_posix()


def _has_sidecar_marker(path: Path) -> bool:
    candidates = [path.with_name(path.name + marker) for marker in _ARTIFACT_MARKERS]
    candidates.extend(path.with_suffix(marker) for marker in _ARTIFACT_MARKERS)
    return any(candidate.exists() for candidate in candidates)


def validate_parquet_shard(
    path: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str],
    minimum_stable_age_seconds: float = 0.0,
    image_sample_count: int = 3,
    repeat_only: bool = False,
) -> ShardReceipt:
    """Validate one finalized shard through a stable, no-follow descriptor."""

    if minimum_stable_age_seconds < 0:
        raise ValueError("minimum stable age must be non-negative")
    if repeat_only and image_sample_count:
        raise ValueError("repeat-only shard validation cannot sample images")
    root = Path(data_root).resolve()
    candidate = Path(os.path.abspath(path))
    relative_path = _relative_shard_path(candidate, root)
    if candidate.suffix.lower() != ".parquet":
        raise ShardValidationError(f"not a parquet file: {candidate}")
    if _path_is_excluded(Path(relative_path)):
        raise UnstableShardError(f"temporary/download artifact path: {relative_path}")
    if _has_sidecar_marker(candidate):
        raise UnstableShardError(f"download sidecar still exists: {relative_path}")

    descriptor, before = _open_regular_nofollow(candidate)
    try:
        age_ns = time.time_ns() - before.st_mtime_ns
        if age_ns < int(minimum_stable_age_seconds * 1_000_000_000):
            raise UnstableShardError(
                f"candidate is younger than {minimum_stable_age_seconds:g}s: {relative_path}"
            )
        if before.st_size < 8:
            raise UnstableShardError(f"parquet file is incomplete: {relative_path}")
        if (
            os.pread(descriptor, 4, 0) != b"PAR1"
            or os.pread(descriptor, 4, before.st_size - 4) != b"PAR1"
        ):
            raise UnstableShardError(
                f"parquet footer is not finalized: {relative_path}"
            )

        file_sha256 = _sha256_fd(descriptor, before.st_size)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - production dependency.
            raise RuntimeError("pyarrow is required for parquet snapshots") from error

        file_object = os.fdopen(os.dup(descriptor), "rb", closefd=True)
        arrow_source = pa.PythonFile(file_object, mode="r")
        try:
            parquet = pq.ParquetFile(arrow_source)
            rows = int(parquet.metadata.num_rows)
            if rows <= 0:
                raise ShardValidationError(f"empty parquet shard: {relative_path}")
            schema = _validate_arrow_schema(parquet.schema_arrow)
            null_counts = _parquet_null_counts(parquet)
            _validate_null_counts(null_counts, repeat_only=repeat_only)
            image_samples = _sample_images(
                parquet,
                image_root=root,
                sample_count=min(image_sample_count, rows),
            )
        except (ShardValidationError, UnstableShardError):
            raise
        except Exception as error:
            raise ShardValidationError(
                f"cannot read parquet footer/schema for {relative_path}: {error}"
            ) from error
        finally:
            arrow_source.close()
        _check_unchanged(candidate, descriptor, before)
    finally:
        os.close(descriptor)

    return ShardReceipt(
        path=relative_path,
        size_bytes=before.st_size,
        sha256=file_sha256,
        rows=rows,
        schema=schema,
        schema_sha256=_canonical_sha256(schema),
        null_counts=null_counts,
        image_decode_samples=image_samples,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )


def _path_is_excluded(relative: Path) -> bool:
    for part in relative.parts:
        lowered = part.lower()
        if lowered == ".cache" or lowered.startswith("."):
            return True
        if any(marker in lowered for marker in _ARTIFACT_MARKERS):
            return True
    return False


def discover_finalized_parquet_candidates(
    data_root: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Discover lexical final paths without following directory/file symlinks."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MMFineReasoning data root does not exist: {root}")
    candidates: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _path_is_excluded((base / name).relative_to(root))
            and not (base / name).is_symlink()
        )
        for name in sorted(file_names):
            path = base / name
            relative = path.relative_to(root)
            if path.suffix.lower() != ".parquet" or _path_is_excluded(relative):
                continue
            if path.is_symlink() or _has_sidecar_marker(path):
                continue
            candidates.append(path)
    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def build_validated_shard_manifest(
    data_root: str | os.PathLike[str],
    *,
    minimum_stable_age_seconds: float = 0.0,
    image_sample_count: int = 3,
    strict_invalid_shards: bool = True,
    created_at_utc: str | None = None,
    included_relative_paths: Iterable[str] | None = None,
    repeat_only: bool = False,
) -> dict[str, Any]:
    """Build a frozen snapshot; unstable downloader artifacts are deferred."""

    root = Path(data_root).resolve()
    if repeat_only and image_sample_count:
        raise ValueError("repeat-only manifest cannot sample images")
    receipts: list[ShardReceipt] = []
    deferred: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    discovered = discover_finalized_parquet_candidates(root)
    if included_relative_paths is None:
        candidates = discovered
        selection: dict[str, Any] = {"mode": "all_discovered_finalized_parquet"}
    else:
        included = tuple(sorted({str(value) for value in included_relative_paths}))
        if not included:
            raise ValueError("included_relative_paths must not be empty")
        for value in included:
            relative = Path(value)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix.lower() != ".parquet"
                or _path_is_excluded(relative)
            ):
                raise ValueError(f"unsafe included parquet path: {value!r}")
        discovered_by_name = {
            path.relative_to(root).as_posix(): path for path in discovered
        }
        missing = sorted(set(included) - set(discovered_by_name))
        if missing:
            raise UnstableShardError(
                f"explicit snapshot paths are not finalized/discoverable: {missing}"
            )
        candidates = tuple(discovered_by_name[value] for value in included)
        selection = {
            "mode": "explicit_relative_paths",
            "paths": list(included),
            "paths_sha256": _canonical_sha256(list(included)),
        }
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        try:
            receipt = validate_parquet_shard(
                path,
                data_root=root,
                minimum_stable_age_seconds=minimum_stable_age_seconds,
                image_sample_count=image_sample_count,
                repeat_only=repeat_only,
            )
        except UnstableShardError as error:
            deferred.append({"path": relative, "reason": str(error)})
            continue
        except ShardValidationError as error:
            if strict_invalid_shards:
                raise
            rejected.append({"path": relative, "reason": str(error)})
            continue
        receipts.append(receipt)

    timestamp = (
        created_at_utc or _datetime.datetime.now(_datetime.timezone.utc).isoformat()
    )
    manifest: dict[str, Any] = {
        "schema": REPEAT_MANIFEST_SCHEMA if repeat_only else MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at_utc": timestamp,
        "data_root": str(root),
        "policy": {
            "required_fields": (
                ["source", "id", *REPEAT_FIELDS]
                if repeat_only
                else list(REQUIRED_FIELDS)
            ),
            "required_image_fields": (
                [] if repeat_only else list(REQUIRED_IMAGE_FIELDS)
            ),
            "repeat_fields": list(REPEAT_FIELDS),
            "repeat_question_normalization": "remove_one_leading_<image>_and_following_whitespace",
            "objective": "text_repeat_only" if repeat_only else "joint",
            "joint_objective_ratio": (
                None if repeat_only else {"vqa_completion": 1, "text_repeat": 1}
            ),
            "projection": "repeat_only_text" if repeat_only else "joint_multimodal",
            "minimum_stable_age_seconds": float(minimum_stable_age_seconds),
            "image_sample_count": int(image_sample_count),
            "candidate_order": "relative_path_lexicographic",
            "selection": selection,
        },
        "shards": [receipt.as_dict() for receipt in receipts],
        "deferred": deferred,
        "rejected": rejected,
        "counts": {
            "discovered_candidates": len(candidates),
            "ready_shards": len(receipts),
            "ready_rows": sum(receipt.rows for receipt in receipts),
            "deferred_shards": len(deferred),
            "rejected_shards": len(rejected),
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") not in {MANIFEST_SCHEMA, REPEAT_MANIFEST_SCHEMA}:
        raise ShardValidationError("unexpected manifest schema")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ShardValidationError("unexpected manifest version")
    if manifest.get("schema") == REPEAT_MANIFEST_SCHEMA:
        policy = manifest.get("policy")
        if (
            not isinstance(policy, Mapping)
            or policy.get("projection") != "repeat_only_text"
            or policy.get("objective") != "text_repeat_only"
            or policy.get("required_fields") != ["source", "id", *REPEAT_FIELDS]
            or policy.get("required_image_fields") != []
            or policy.get("image_sample_count") != 0
        ):
            raise ShardValidationError("repeat-only manifest policy is invalid")
    expected = _manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != expected:
        raise ShardValidationError("manifest SHA256 does not match its content")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ShardValidationError("manifest.shards must be a list")
    paths: set[str] = set()
    keys: set[str] = set()
    for value in shards:
        if not isinstance(value, Mapping):
            raise ShardValidationError("manifest shard receipt must be an object")
        receipt = ShardReceipt.from_mapping(value)
        if Path(receipt.path).is_absolute() or ".." in Path(receipt.path).parts:
            raise ShardValidationError(f"unsafe manifest shard path: {receipt.path}")
        if receipt.path in paths:
            raise ShardValidationError(f"duplicate shard path: {receipt.path}")
        if receipt.queue_key in keys:
            raise ShardValidationError(f"duplicate shard queue key: {receipt.path}")
        if len(receipt.sha256) != 64 or receipt.size_bytes <= 0 or receipt.rows <= 0:
            raise ShardValidationError(f"invalid shard receipt: {receipt.path}")
        if receipt.schema_sha256 != _canonical_sha256(receipt.schema):
            raise ShardValidationError(f"schema SHA256 mismatch: {receipt.path}")
        paths.add(receipt.path)
        keys.add(receipt.queue_key)
    if list(paths) and [value["path"] for value in shards] != sorted(paths):
        raise ShardValidationError("manifest shards are not path-sorted")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ShardValidationError("manifest.counts must be an object")
    if int(counts.get("ready_shards", -1)) != len(shards):
        raise ShardValidationError("manifest ready_shards count differs")
    if int(counts.get("ready_rows", -1)) != sum(int(row["rows"]) for row in shards):
        raise ShardValidationError("manifest ready_rows count differs")


def _read_stable_file(path: Path) -> bytes:
    descriptor, before = _open_regular_nofollow(path)
    try:
        payload, _ = _read_fd(descriptor, before.st_size)
        _check_unchanged(path, descriptor, before)
        return payload
    finally:
        os.close(descriptor)


def write_validated_shard_manifest(
    output: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> Path:
    """Atomically publish a manifest without replacing different content."""

    validate_manifest(manifest)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        if _read_stable_file(path) == payload:
            return path
        raise FileExistsError(f"refusing to replace immutable manifest: {path}")

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace immutable manifest: {path}")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_validated_shard_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(_read_stable_file(Path(path)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardValidationError(
            f"manifest is not valid UTF-8 JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ShardValidationError("manifest root must be an object")
    validate_manifest(value)
    return value


def ready_queue_delta(
    previous_manifest: Mapping[str, Any], current_manifest: Mapping[str, Any]
) -> ReadyShardQueue:
    """Return newly ready shards while rejecting mutation/removal of old paths."""

    validate_manifest(previous_manifest)
    validate_manifest(current_manifest)
    old = {
        receipt.path: receipt
        for receipt in (
            ShardReceipt.from_mapping(value) for value in previous_manifest["shards"]
        )
    }
    current = {
        receipt.path: receipt
        for receipt in (
            ShardReceipt.from_mapping(value) for value in current_manifest["shards"]
        )
    }
    for path, receipt in old.items():
        replacement = current.get(path)
        if replacement is None:
            raise ShardValidationError(
                f"ready shard disappeared from later snapshot: {path}"
            )
        if replacement != receipt:
            raise ShardValidationError(f"ready shard changed between snapshots: {path}")
    additions = tuple(current[path] for path in sorted(set(current) - set(old)))
    return ReadyShardQueue(
        root=Path(str(current_manifest["data_root"])), shards=additions
    )


def verify_shard_receipt(
    receipt: ShardReceipt,
    *,
    data_root: str | os.PathLike[str],
    image_sample_count: int = 0,
    repeat_only: bool = False,
) -> None:
    observed = validate_parquet_shard(
        Path(data_root) / receipt.path,
        data_root=data_root,
        minimum_stable_age_seconds=0,
        image_sample_count=image_sample_count,
        repeat_only=repeat_only,
    )
    ignored = {"image_decode_samples"}
    expected_value = {
        key: value for key, value in receipt.as_dict().items() if key not in ignored
    }
    observed_value = {
        key: value for key, value in observed.as_dict().items() if key not in ignored
    }
    if observed_value != expected_value:
        raise ShardValidationError(f"shard differs from frozen receipt: {receipt.path}")


def iter_shard_records(
    receipt: ShardReceipt,
    *,
    data_root: str | os.PathLike[str],
    batch_size: int = 64,
    verify_receipt: bool = True,
) -> Iterator[MMFineReasoningRecord]:
    """Iterate strict rows from a content-bound ready-shard receipt."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(data_root).resolve()
    if verify_receipt:
        verify_shard_receipt(receipt, data_root=root, image_sample_count=0)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - production dependency.
        raise RuntimeError("pyarrow is required for parquet iteration") from error
    path = root / receipt.path
    descriptor, before = _open_regular_nofollow(path)
    try:
        if (
            before.st_dev != receipt.device
            or before.st_ino != receipt.inode
            or before.st_size != receipt.size_bytes
            or before.st_mtime_ns != receipt.mtime_ns
        ):
            raise ShardValidationError(f"shard identity differs: {receipt.path}")
        file_object = os.fdopen(os.dup(descriptor), "rb", closefd=True)
        try:
            parquet = pq.ParquetFile(file_object)
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                for row in batch.to_pylist():
                    yield adapt_row(row, image_root=root)
        finally:
            file_object.close()
        _check_unchanged(path, descriptor, before)
    finally:
        os.close(descriptor)


def iter_shard_repeat_events(
    receipt: ShardReceipt,
    *,
    data_root: str | os.PathLike[str],
    repeat_rotation: int,
    batch_size: int = 256,
    verify_receipt: bool = True,
    skip_empty_fields: bool = False,
) -> Iterator[TextRepeatEvent]:
    """Read only text columns and yield the selected repeat field slots."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(data_root).resolve()
    if verify_receipt:
        verify_shard_receipt(
            receipt,
            data_root=root,
            image_sample_count=0,
            repeat_only=True,
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - production dependency.
        raise RuntimeError("pyarrow is required for parquet iteration") from error
    path = root / receipt.path
    descriptor, before = _open_regular_nofollow(path)
    try:
        if (
            before.st_dev != receipt.device
            or before.st_ino != receipt.inode
            or before.st_size != receipt.size_bytes
            or before.st_mtime_ns != receipt.mtime_ns
        ):
            raise ShardValidationError(f"shard identity differs: {receipt.path}")
        file_object = os.fdopen(os.dup(descriptor), "rb", closefd=True)
        try:
            parquet = pq.ParquetFile(file_object)
            records = (
                adapt_repeat_row(
                    row,
                    allow_empty_fields=skip_empty_fields,
                )
                for batch in parquet.iter_batches(
                    batch_size=batch_size,
                    columns=["source", "id", *REPEAT_FIELDS],
                    use_threads=False,
                )
                for row in batch.to_pylist()
            )
            yield from iter_repeat_events(
                records,
                repeat_rotation=repeat_rotation,
                skip_empty_fields=skip_empty_fields,
            )
        finally:
            file_object.close()
        _check_unchanged(path, descriptor, before)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser(
        "snapshot", help="validate finalized parquet shards and freeze a manifest"
    )
    snapshot.add_argument("--data-root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--minimum-stable-age-seconds", type=float, default=30.0)
    snapshot.add_argument("--image-sample-count", type=int, default=3)
    snapshot.add_argument(
        "--include-path",
        action="append",
        default=None,
        help="freeze only this data-root-relative parquet path (repeatable)",
    )
    snapshot.add_argument(
        "--record-invalid-shards",
        action="store_true",
        help="record invalid finalized shards instead of failing the snapshot",
    )
    snapshot.add_argument(
        "--repeat-only",
        action="store_true",
        help="validate only the source/id/question/original_answer/caption projection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "snapshot":  # pragma: no cover - argparse is exhaustive.
        raise AssertionError(args.command)
    manifest = build_validated_shard_manifest(
        args.data_root,
        minimum_stable_age_seconds=args.minimum_stable_age_seconds,
        image_sample_count=args.image_sample_count,
        strict_invalid_shards=not args.record_invalid_shards,
        included_relative_paths=args.include_path,
        repeat_only=args.repeat_only,
    )
    write_validated_shard_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                **manifest["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
