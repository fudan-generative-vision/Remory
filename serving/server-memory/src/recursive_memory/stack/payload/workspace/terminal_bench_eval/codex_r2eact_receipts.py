"""Crash-safe incremental persistence for Codex r²eAct round receipts.

The v1 gateway rewrote one ever-growing JSON array after every model round.
This module makes an append-only, hash-chained JSONL journal canonical and
keeps a small atomic telemetry projection beside it.  A pre-v2 JSON array is
accepted as an immutable migration source, but is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping

from terminal_bench_eval.codex_r2eact_artifacts import (
    GATEWAY_RECEIPT_RECORD_SCHEMA,
    GATEWAY_RECEIPT_SUMMARY_SCHEMA,
    GATEWAY_RECEIPTS_FILENAME,
    GATEWAY_RECEIPTS_JSONL_FILENAME,
    GATEWAY_RECEIPTS_SUMMARY_FILENAME,
)


MAX_RECEIPT_RECORD_BYTES = 16 * 1024 * 1024
MAX_LEGACY_RECEIPTS_BYTES = 512 * 1024 * 1024


class GatewayReceiptPersistenceError(RuntimeError):
    """The private receipt journal violated its append-only contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GatewayReceiptPersistenceError(
            "gateway receipt is not canonical JSON"
        ) from exc


def _record_payload(
    receipt: Mapping[str, Any],
    *,
    record_index: int,
    previous_record_sha256: str | None,
) -> tuple[dict[str, Any], bytes]:
    unsigned = {
        "schema": GATEWAY_RECEIPT_RECORD_SCHEMA,
        "record_index": record_index,
        "previous_record_sha256": previous_record_sha256,
        "receipt": dict(receipt),
    }
    record = {
        **unsigned,
        "record_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }
    payload = _canonical_bytes(record) + b"\n"
    if len(payload) > MAX_RECEIPT_RECORD_BYTES:
        raise GatewayReceiptPersistenceError("gateway receipt record exceeds limit")
    return record, payload


def decode_receipt_record(
    value: Mapping[str, Any],
    *,
    expected_index: int | None = None,
    expected_previous_sha256: str | None | object = Ellipsis,
) -> tuple[dict[str, Any], str]:
    """Validate and unwrap one canonical JSONL record."""

    if set(value) != {
        "schema",
        "record_index",
        "previous_record_sha256",
        "receipt",
        "record_sha256",
    }:
        raise GatewayReceiptPersistenceError("gateway receipt record fields mismatch")
    index = value.get("record_index")
    previous = value.get("previous_record_sha256")
    receipt = value.get("receipt")
    digest = value.get("record_sha256")
    if (
        value.get("schema") != GATEWAY_RECEIPT_RECORD_SCHEMA
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 1
        or (previous is not None and not _is_sha256(previous))
        or not isinstance(receipt, Mapping)
        or not _is_sha256(digest)
    ):
        raise GatewayReceiptPersistenceError("gateway receipt record is invalid")
    if expected_index is not None and index != expected_index:
        raise GatewayReceiptPersistenceError("gateway receipt sequence has a gap")
    if (
        expected_previous_sha256 is not Ellipsis
        and previous != expected_previous_sha256
    ):
        raise GatewayReceiptPersistenceError("gateway receipt hash chain diverges")
    unsigned = {key: value[key] for key in value if key != "record_sha256"}
    if hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != digest:
        raise GatewayReceiptPersistenceError("gateway receipt digest mismatch")
    return dict(receipt), str(digest)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _secure_open(path: Path, flags: int, mode: int = 0o600) -> int:
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise GatewayReceiptPersistenceError(
            f"unable to open private receipt artifact: {path.name}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise GatewayReceiptPersistenceError(
            "private receipt artifact must be one regular, unlinked file"
        )
    return descriptor


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(dict(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = _secure_open(temporary, flags)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("gateway receipt summary write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_regular_bytes(path: Path, *, limit: int) -> bytes:
    descriptor = _secure_open(path, os.O_RDONLY)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > limit:
            raise GatewayReceiptPersistenceError(
                f"private receipt artifact exceeds limit: {path.name}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise GatewayReceiptPersistenceError(
                    f"private receipt artifact exceeds limit: {path.name}"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _legacy_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() and not path.is_symlink():
        return []
    try:
        value = json.loads(
            _read_regular_bytes(path, limit=MAX_LEGACY_RECEIPTS_BYTES).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayReceiptPersistenceError(
            "legacy gateway receipt array is invalid"
        ) from exc
    if not isinstance(value, list) or any(
        not isinstance(receipt, Mapping) for receipt in value
    ):
        raise GatewayReceiptPersistenceError(
            "legacy gateway receipt artifact is not an object array"
        )
    return [dict(receipt) for receipt in value]


class GatewayReceiptJournal:
    """Single-writer append-only receipt journal with an atomic projection."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        if self.artifact_dir.is_symlink():
            raise GatewayReceiptPersistenceError(
                "gateway receipt directory may not be a symlink"
            )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not self.artifact_dir.is_dir():
            raise GatewayReceiptPersistenceError(
                "gateway receipt directory is not a directory"
            )
        self.path = self.artifact_dir / GATEWAY_RECEIPTS_JSONL_FILENAME
        self.legacy_path = self.artifact_dir / GATEWAY_RECEIPTS_FILENAME
        self.summary_path = self.artifact_dir / GATEWAY_RECEIPTS_SUMMARY_FILENAME
        self._record_count = 0
        self._last_record_sha256: str | None = None
        self._source_size_bytes = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._has_input_tokens = False
        self._has_output_tokens = False
        self._latest_round = 0
        self._latest_receipt: dict[str, Any] = {}
        self._initialize()

    @property
    def record_count(self) -> int:
        return self._record_count

    def _initialize(self) -> None:
        self._ensure_journal_file()
        receipts, tip, size = self._scan_and_repair()
        self._reset(receipts, tip=tip, source_size_bytes=size)
        legacy = _legacy_receipts(self.legacy_path)
        shared = min(len(legacy), len(receipts))
        if any(legacy[index] != receipts[index] for index in range(shared)):
            raise GatewayReceiptPersistenceError(
                "legacy and canonical gateway receipts disagree"
            )
        if len(receipts) < len(legacy):
            for receipt in legacy[len(receipts) :]:
                self.append(receipt)
        self._write_summary()

    def _ensure_journal_file(self) -> None:
        existed = self.path.exists() or self.path.is_symlink()
        descriptor = _secure_open(
            self.path,
            os.O_RDWR | os.O_APPEND | os.O_CREAT,
        )
        try:
            os.fchmod(descriptor, 0o600)
            if not existed:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not existed:
            _fsync_directory(self.artifact_dir)

    def _scan_and_repair(self) -> tuple[list[dict[str, Any]], str | None, int]:
        descriptor = _secure_open(self.path, os.O_RDWR)
        receipts: list[dict[str, Any]] = []
        tip: str | None = None
        committed_bytes = 0
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                while True:
                    start = handle.tell()
                    raw = handle.readline(MAX_RECEIPT_RECORD_BYTES + 1)
                    if not raw:
                        committed_bytes = start
                        break
                    if len(raw) > MAX_RECEIPT_RECORD_BYTES:
                        raise GatewayReceiptPersistenceError(
                            "gateway receipt record exceeds limit"
                        )
                    if not raw.endswith(b"\n"):
                        os.ftruncate(descriptor, start)
                        os.fsync(descriptor)
                        committed_bytes = start
                        break
                    try:
                        value = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise GatewayReceiptPersistenceError(
                            "gateway receipt journal contains malformed JSON"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise GatewayReceiptPersistenceError(
                            "gateway receipt journal record is not an object"
                        )
                    receipt, tip = decode_receipt_record(
                        value,
                        expected_index=len(receipts) + 1,
                        expected_previous_sha256=tip,
                    )
                    receipts.append(receipt)
                    committed_bytes = handle.tell()
            return receipts, tip, committed_bytes
        finally:
            os.close(descriptor)

    def _reset(
        self,
        receipts: list[dict[str, Any]],
        *,
        tip: str | None,
        source_size_bytes: int,
    ) -> None:
        self._record_count = 0
        self._last_record_sha256 = None
        self._source_size_bytes = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._has_input_tokens = False
        self._has_output_tokens = False
        self._latest_round = 0
        self._latest_receipt = {}
        for receipt in receipts:
            self._accumulate(receipt)
        self._last_record_sha256 = tip
        self._source_size_bytes = source_size_bytes

    def _accumulate(self, receipt: Mapping[str, Any]) -> None:
        copied = dict(receipt)
        self._record_count += 1
        round_index = copied.get("round_index")
        if isinstance(round_index, int) and not isinstance(round_index, bool):
            if round_index >= self._latest_round:
                self._latest_round = round_index
                self._latest_receipt = copied
        input_tokens = copied.get("input_tokens")
        if isinstance(input_tokens, (int, float)) and not isinstance(input_tokens, bool):
            self._input_tokens += int(input_tokens)
            self._has_input_tokens = True
        output_tokens = copied.get("output_tokens", copied.get("token_count"))
        if isinstance(output_tokens, (int, float)) and not isinstance(
            output_tokens, bool
        ):
            self._output_tokens += int(output_tokens)
            self._has_output_tokens = True

    def _summary(self) -> dict[str, Any]:
        latest = self._latest_receipt
        latest_output = latest.get("output_tokens", latest.get("token_count"))
        compression = latest.get("compression")
        return {
            "schema": GATEWAY_RECEIPT_SUMMARY_SCHEMA,
            "source_filename": GATEWAY_RECEIPTS_JSONL_FILENAME,
            "source_size_bytes": self._source_size_bytes,
            "records": self._record_count,
            "last_record_sha256": self._last_record_sha256,
            "latest_round": self._latest_round,
            "input_tokens": self._input_tokens if self._has_input_tokens else None,
            "output_tokens": self._output_tokens if self._has_output_tokens else None,
            "latest_generation": (
                {
                    "round_index": latest.get("round_index"),
                    "elapsed_seconds": latest.get("elapsed_seconds"),
                    "input_tokens": latest.get("input_tokens"),
                    "token_count": latest_output,
                    "finished_on_eos": latest.get("finished_on_eos"),
                    "hit_generation_cap": latest.get("hit_generation_cap"),
                    "complete_valid": latest.get("complete_valid"),
                    "finish_reason": latest.get("finish_reason"),
                    "tool_calls": latest.get("tool_calls"),
                }
                if latest
                else {}
            ),
            "latest_compression": (
                dict(compression) if isinstance(compression, Mapping) else {}
            ),
        }

    def _write_summary(self) -> None:
        _atomic_private_json(self.summary_path, self._summary())

    def append(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        record, payload = _record_payload(
            receipt,
            record_index=self._record_count + 1,
            previous_record_sha256=self._last_record_sha256,
        )
        descriptor = _secure_open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            before = os.fstat(descriptor)
            if before.st_size != self._source_size_bytes:
                raise GatewayReceiptPersistenceError(
                    "gateway receipt journal changed outside its writer"
                )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("gateway receipt append made no progress")
                view = view[written:]
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            expected_size = before.st_size + len(payload)
            if after.st_size != expected_size:
                raise GatewayReceiptPersistenceError(
                    "gateway receipt append size mismatch"
                )
        finally:
            os.close(descriptor)
        self._source_size_bytes = expected_size
        self._last_record_sha256 = str(record["record_sha256"])
        self._accumulate(receipt)
        self._write_summary()
        return self._summary()


__all__ = [
    "GatewayReceiptJournal",
    "GatewayReceiptPersistenceError",
    "MAX_LEGACY_RECEIPTS_BYTES",
    "MAX_RECEIPT_RECORD_BYTES",
    "decode_receipt_record",
]
