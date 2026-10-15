"""Causal all-response views and immutable, history-bound compact-summary cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from context_compression.qwen38_allturn_data import digest, load_example

SUMMARY_SCHEMA = "qwen38_compact_residual_fullprefix_summary_v1"
# Match the task-checkpoint instruction used by our native compact harness.
SUMMARY_PROMPT = (
    "Create a concise context checkpoint for another instance continuing this task. "
    "Do not perform the task or call tools. Preserve the user goal and constraints, "
    "completed work, current files/artifacts, exact paths/identifiers, important evidence, "
    "failed approaches, unresolved issues and concrete next steps. Distinguish observations "
    "from guesses. Merge the previous checkpoint if any; do not repeat the whole transcript. "
    "Return only the checkpoint summary."
)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class ResidualEvent:
    key: str
    turn: int
    prefix_ids: tuple[int, ...]
    history_ids: tuple[int, ...]
    opening_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    # Intra-reasoning cuts keep exact raw continuation tokens for the teacher,
    # but restore the assistant/thinking frame in the compressed student.
    student_opening_ids: tuple[int, ...] | None = None
    summary_closing_ids: tuple[int, ...] = ()
    target_offset: int = 0

    @property
    def history_sha256(self):
        # Intentionally does NOT include the answer or any future events.
        return identity({"prefix": self.prefix_ids, "history": self.history_ids})


def residual_events(example):
    history = []
    for turn, segment in enumerate(example["segments"]):
        history.extend(segment["context_ids"])
        if not history or not segment["opening_ids"] or not segment["response_ids"]:
            raise ValueError("residual response requires nonempty history/opening/answer")
        yield ResidualEvent(example["key"], turn, tuple(example["prefix_ids"]), tuple(history),
                            tuple(segment["opening_ids"]), tuple(segment["response_ids"]),
                            (tuple(segment["student_opening_ids"])
                             if "student_opening_ids" in segment else None),
                            tuple(segment.get("summary_closing_ids", ())),
                            segment.get("target_offset", 0))
        history.extend(segment["opening_ids"])
        history.extend(segment["response_ids"])


def summary_contract(tokenizer, *, database_sha256, tokenizer_files, generator_model,
                     max_summary_tokens=512, target_summary_tokens=None):
    if max_summary_tokens <= 0:
        raise ValueError("summary budget must be positive")
    target_summary_tokens = min(512, max_summary_tokens) if target_summary_tokens is None else target_summary_tokens
    if not 0 < target_summary_tokens <= max_summary_tokens:
        raise ValueError("summary target must fit within the hard summary cap")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    end_ids = encode("<|im_end|>")
    if len(end_ids) != 1:
        raise ValueError("Qwen im_end must be one token")
    return {
        "schema": SUMMARY_SCHEMA, "database_sha256": database_sha256,
        "tokenizer_files": tokenizer_files, "generator_model": str(generator_model),
        "mode": "full_causal_prefix_rebuild_no_previous_summary",
        "max_summary_tokens": max_summary_tokens, "target_summary_tokens": target_summary_tokens,
        "temperature": 0.0,
        "prompt": SUMMARY_PROMPT, "end_token_id": end_ids[0],
        "generation_suffix_ids": encode("\n<|im_start|>user\n" + SUMMARY_PROMPT +
            f" Keep the checkpoint within {target_summary_tokens} tokens."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "student_summary_before_ids": encode("<|im_start|>user\nContext checkpoint:\n"),
        "student_summary_after_ids": encode("<|im_end|>\n"),
        "student_memory_before_ids": encode("<|im_start|>user\nResidual context memory:\n"),
        "student_memory_after_ids": encode("<|im_end|>\n"),
    }


def summary_request(event, contract):
    # Synthetic role closure is for the summarizer ONLY, never the raw teacher
    # or Compressor features. No token from the held-out suffix is consulted.
    ids = list(event.prefix_ids + event.history_ids + event.summary_closing_ids) + contract["generation_suffix_ids"]
    body = {"key": event.key, "turn": event.turn, "history_sha256": event.history_sha256,
            "contract_sha256": identity(contract), "input_ids": ids}
    return {**body, "request_sha256": identity(body)}


def summary_envelope(summary_ids, contract):
    return (contract["student_summary_before_ids"] + list(summary_ids) +
            contract["student_summary_after_ids"])


class SummaryCache:
    def __init__(self, path, contract, *, readonly=False):
        self.path, self.contract, self.readonly = Path(path), contract, readonly
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS summaries (key TEXT, turn INTEGER, "
                            "payload TEXT NOT NULL, PRIMARY KEY (key, turn))")
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)",
                                (json.dumps(contract, sort_keys=True),))
        value = self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if value is None or json.loads(value[0]) != contract:
            self.db.close()
            raise ValueError("summary cache contract changed; use a new cache")

    def get(self, event, *, missing_ok=False):
        row = self.db.execute("SELECT payload FROM summaries WHERE key=? AND turn=?",
                              (event.key, event.turn)).fetchone()
        if row is None:
            if missing_ok:
                return None
            raise ValueError(f"missing summary: {event.key} turn {event.turn}")
        item = json.loads(row[0])
        request = summary_request(event, self.contract)
        if item["request_sha256"] != request["request_sha256"]:
            raise ValueError("summary belongs to a different causal history/request")
        expected = identity({"request_sha256": item["request_sha256"], "ids": item["ids"]})
        if item["summary_sha256"] != expected:
            raise ValueError("summary content checksum mismatch")
        self._validate_ids(item["ids"])
        return item

    def _validate_ids(self, ids):
        if (not isinstance(ids, list) or not 0 < len(ids) <= self.contract["max_summary_tokens"]
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ids)
                or self.contract["end_token_id"] in ids):
            raise ValueError("invalid/empty/oversized summary token IDs")

    def put(self, event, ids, *, finish_reason):
        if self.readonly or self.db.execute("SELECT 1 FROM metadata WHERE key='sealed'").fetchone():
            raise ValueError("summary cache is read-only/sealed")
        if finish_reason != "stop":
            raise ValueError("summary was not completed naturally; length-truncated output is rejected")
        ids = list(ids)
        if ids and ids[-1] == self.contract["end_token_id"]:
            ids.pop()
        self._validate_ids(ids)
        request = summary_request(event, self.contract)
        item = {"request_sha256": request["request_sha256"], "ids": ids}
        item["summary_sha256"] = identity(item)
        with self.db:
            self.db.execute("INSERT INTO summaries VALUES (?, ?, ?)",
                            (event.key, event.turn, json.dumps(item)))

    def seal(self, database, *, ordinals=None):
        if self.readonly:
            raise ValueError("cannot seal read-only cache")
        count = 0
        selected = (list(ordinals) if ordinals is not None else
                    [r[0] for r in database.execute("SELECT ordinal FROM examples ORDER BY ordinal")])
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("invalid/empty summary coverage")
        for ordinal in selected:
            for event in residual_events(load_example(database, ordinal)):
                self.get(event)
                count += 1
        if self.db.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] != count:
            raise ValueError("summary cache has extra rows outside requested coverage")
        coverage = {"responses": count, "trajectories": len(selected),
                    "ordinals": list(ordinals) if ordinals is not None else None}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('sealed', ?)",
                            (json.dumps(coverage),))
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("summary database integrity check failed")
        return {"schema": SUMMARY_SCHEMA, "cache": str(self.path.resolve()),
                "cache_sha256": digest(self.path), "contract": self.contract,
                "coverage": coverage}

    def close(self):
        self.db.close()


def open_summary_manifest(path, *, dataset_manifest, canary_ordinals=None, verify_digest=True):
    manifest = json.loads(Path(path).read_text())
    contract = manifest["contract"]
    if (manifest["schema"] != SUMMARY_SCHEMA or contract["schema"] != SUMMARY_SCHEMA
            or contract["database_sha256"] != dataset_manifest["database_sha256"]
            or contract["tokenizer_files"] != dataset_manifest["tokenizer_files"]):
        raise ValueError("summary/data identity mismatch")
    coverage = manifest["coverage"]
    if coverage["ordinals"] is not None and list(canary_ordinals or []) != coverage["ordinals"]:
        raise ValueError("partial summary cache is permitted only for the exact canary ordinals")
    if verify_digest and digest(Path(manifest["cache"])) != manifest["cache_sha256"]:
        raise ValueError("summary cache file checksum mismatch")
    cache = SummaryCache(manifest["cache"], contract, readonly=True)
    sealed = cache.db.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone()
    if not sealed or json.loads(sealed[0]) != coverage:
        cache.close()
        raise ValueError("summary cache is not sealed with this coverage")
    return cache, manifest
