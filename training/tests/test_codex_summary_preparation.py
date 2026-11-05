import hashlib
import sqlite3
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from context_compression import prepare_qwen38_codex_summaries as prepare
from context_compression.codex_compact_v0149 import PROMPT, PROMPT_SHA256, summary_contract
from context_compression.qwen38_residual_data import ResidualEvent, SummaryCache, summary_request


class Tokenizer:
    def encode(self, text, **kwargs):
        return [777] if text == "<|im_end|>" else list(text.encode())


def contract():
    return summary_contract(Tokenizer(), database_sha256="db", tokenizer_files={},
                            generator_model="/model", max_summary_tokens=4096)


def event(i):
    return ResidualEvent(f"coding:{i}", 0, (9,), (10+i,), (19,), (20, 21))


def test_native_prompt_and_request_are_exact_and_answer_free():
    c = contract()
    assert hashlib.sha256(PROMPT.encode()).hexdigest() == PROMPT_SHA256
    expected = "\n<|im_start|>user\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert bytes(c["generation_suffix_ids"]).decode() == expected
    assert "512" not in expected and "Keep the checkpoint within" not in expected
    assert c["prompt_source"]["version"] == "0.149.0"
    assert summary_request(event(0), c) == summary_request(replace(event(0), target_ids=(9999,)), c)
    assert summary_request(event(0), c) != summary_request(replace(event(0), history_ids=(9999,)), c)


def test_legacy_or_changed_prompt_cache_is_rejected(tmp_path):
    path = tmp_path / "cache.sqlite"
    old = {**contract(), "prompt": "old hand-written prompt"}
    SummaryCache(path, old).close()
    with pytest.raises(ValueError, match="contract changed"):
        SummaryCache(path, contract())


def test_binary_verification_allows_repeated_literals_but_rejects_changed_bytes(tmp_path, monkeypatch):
    from context_compression import codex_compact_v0149 as source
    data = PROMPT.encode() * 2
    path = tmp_path / "binary"
    path.write_bytes(data)
    monkeypatch.setattr(source, "CODEX_SHA256", hashlib.sha256(data).hexdigest())
    source.verify_binary(path)
    path.write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="binary differs"):
        source.verify_binary(path)


def test_two_endpoints_are_bounded_and_cache_has_one_writer(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    main_thread = threading.get_ident()
    original_put = cache.put
    writes = []
    barrier = threading.Barrier(2)
    calls = []

    def put(*args, **kwargs):
        writes.append(threading.get_ident())
        return original_put(*args, **kwargs)

    def generate(url, request, c, *, context_limit):
        calls.append((url, request["key"]))
        barrier.wait(timeout=5)
        return [30, 777], "stop"

    cache.put = put
    try:
        result = prepare.fill_cache([event(i) for i in range(6)], cache, contract(),
                                    [("a", 262144), ("b", 262144)], workers=1, generate=generate)
        assert result == {"created": 6, "reused": 0, "limited": False}
        assert {url for url, _ in calls} == {"a", "b"}
        assert writes == [main_thread] * 6
        assert len({key for _, key in calls}) == 6
    finally:
        cache.close()


def test_budget_resume_skips_committed_rows(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    calls = []

    def generate(url, request, c, **kwargs):
        calls.append(request["key"])
        return [31], "stop"

    try:
        kwargs = dict(workers=2, generate=generate)
        first = prepare.fill_cache([event(i) for i in range(5)], cache, contract(), [("a", 262144)], limit=2, **kwargs)
        assert first == {"created": 2, "reused": 0, "limited": True}
        second = prepare.fill_cache([event(i) for i in range(5)], cache, contract(), [("a", 262144)], **kwargs)
        assert second == {"created": 3, "reused": 2, "limited": False}
        assert sorted(calls) == [f"coding:{i}" for i in range(5)]
    finally:
        cache.close()


def test_failure_keeps_other_inflight_success_for_resume(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    barrier = threading.Barrier(2)

    def generate(url, request, c, **kwargs):
        barrier.wait(timeout=5)
        if request["key"] == "coding:0":
            raise ValueError("incomplete response")
        return [30], "stop"

    try:
        with pytest.raises(RuntimeError, match="coding:0"):
            prepare.fill_cache([event(0), event(1)], cache, contract(), [("a", 262144)], workers=2, generate=generate)
        assert cache.get(event(0), missing_ok=True) is None
        assert cache.get(event(1))["ids"] == [30]
    finally:
        cache.close()


def test_length_truncated_output_never_enters_cache(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    try:
        with pytest.raises(RuntimeError, match="naturally"):
            prepare.fill_cache([event(0)], cache, contract(), [("a", 262144)],
                               generate=lambda *a, **k: ([30], "length"))
        assert cache.get(event(0), missing_ok=True) is None
    finally:
        cache.close()


def test_keep_going_records_holes_and_later_resume_fills_them(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    failures, calls = [], []
    def generate(url, request, c, **kwargs):
        calls.append(request["key"])
        if request["key"] == "coding:1":
            raise ValueError("invalid handoff field")
        return [31], "stop"
    try:
        result = prepare.fill_cache([event(i) for i in range(5)], cache, contract(), [("a", 262144)],
                                    workers=2, generate=generate, keep_going=True,
                                    max_attempts=2, on_failure=failures.append)
        assert result == {"created": 4, "reused": 0, "failed": 1, "limited": False}
        assert calls.count("coding:1") == 2
        assert len(failures) == 1 and failures[0]["key"] == "coding:1"
        assert failures[0]["request_sha256"] == summary_request(event(1), contract())["request_sha256"]
        assert cache.get(event(1), missing_ok=True) is None
        assert cache.get(event(4))["ids"] == [31]
        resumed = prepare.fill_cache([event(i) for i in range(5)], cache, contract(), [("a", 262144)],
                                     generate=lambda *a, **k: ([32], "stop"),
                                     keep_going=True, on_failure=failures.append)
        assert resumed == {"created": 1, "reused": 4, "failed": 0, "limited": False}
        assert cache.get(event(1))["ids"] == [32]
    finally:
        cache.close()


def test_bounded_retry_keeps_same_request_and_does_not_log_recovered_failure(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    calls, failures = [], []
    def generate(url, request, c, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise ValueError("temporary invalid output")
        return [31], "stop"
    try:
        result = prepare.fill_cache([event(0)], cache, contract(), [("a", 262144)],
                                    generate=generate, max_attempts=2,
                                    keep_going=True, on_failure=failures.append)
        assert calls[0] == calls[1] == summary_request(event(0), contract())
        assert result["created"] == 1 and result["failed"] == 0 and not failures
    finally:
        cache.close()


def test_keep_going_never_accepts_truncation_and_stops_systemic_failures(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    failures = []
    try:
        with pytest.raises(RuntimeError, match="naturally"):
            prepare.fill_cache([event(i) for i in range(20)], cache, contract(), [("a", 262144)],
                               workers=1, generate=lambda *a, **k: ([31], "length"),
                               keep_going=True, max_attempts=2, max_consecutive_failures=3,
                               on_failure=failures.append)
        assert len(failures) == 3
        assert cache.db.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
    finally:
        cache.close()


def test_keep_going_does_not_swallow_storage_errors(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    def broken_writer(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")
    cache.put = broken_writer
    failures = []
    try:
        with pytest.raises(RuntimeError, match="cache write failed.*disk full"):
            prepare.fill_cache([event(i) for i in range(3)], cache, contract(), [("a", 262144)],
                               workers=1, generate=lambda *a, **k: ([31], "stop"),
                               keep_going=True, on_failure=failures.append)
        assert not failures
    finally:
        cache.close()


def test_keep_going_requires_failure_recording(tmp_path):
    cache = SummaryCache(tmp_path / "cache.sqlite", contract())
    try:
        with pytest.raises(ValueError, match="failure recorder"):
            prepare.fill_cache([], cache, contract(), [("a", 262144)], keep_going=True)
    finally:
        cache.close()


def test_cli_keep_going_does_not_seal_missing_summaries(tmp_path, monkeypatch):
    import json
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE examples (ordinal INTEGER)")
    db.executemany("INSERT INTO examples VALUES (?)", [(0,), (1,), (2,)])
    manifest = {"database_sha256": "db", "tokenizer_files": {}, "stats": {}}
    monkeypatch.setattr(prepare, "validate_dataset", lambda *a: (manifest, db))
    monkeypatch.setattr(prepare, "load_example", lambda db, n: n)
    monkeypatch.setattr(prepare, "residual_events", lambda n: [event(n)])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    monkeypatch.setattr(prepare, "request_json", lambda *a: {
        "model_path": str(tmp_path.resolve()), "context_length": 262144})
    def generate(url, request, c, **kwargs):
        if request["key"] == "coding:1":
            raise ValueError("invalid handoff field")
        return [31], "stop"
    monkeypatch.setattr(prepare, "generate_handoff", generate)
    with pytest.raises(SystemExit) as stop:
        prepare.main(["--manifest", "unused", "--target-model", str(tmp_path),
                      "--cache", str(tmp_path / "cache.sqlite"),
                      "--output-manifest", str(tmp_path / "summary.json"),
                      "--base-url", "http://unused", "--workers", "1",
                      "--allow-unfiltered-data", "--keep-going", "--max-attempts", "2"])
    assert stop.value.code == 2 and not (tmp_path / "summary.json").exists()
    with sqlite3.connect(tmp_path / "cache.sqlite") as cache:
        assert cache.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 2
        assert cache.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone() is None
    failures = (tmp_path / "cache.failures.jsonl").read_text().splitlines()
    assert len(failures) == 1 and json.loads(failures[0])["key"] == "coding:1"


def test_check_only_does_not_contact_servers_or_create_cache(tmp_path, monkeypatch, capsys):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE examples (ordinal INTEGER)")
    db.execute("INSERT INTO examples VALUES (0)")
    manifest = {"database_sha256": "db", "tokenizer_files": {}, "stats": {}}
    monkeypatch.setattr(prepare, "validate_dataset", lambda *a: (manifest, db))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))

    def forbidden(*args, **kwargs):
        pytest.fail("check-only contacted a server")

    monkeypatch.setattr(prepare, "request_json", forbidden)
    prepare.main(["--manifest", "unused", "--target-model", str(tmp_path),
                  "--cache", str(tmp_path / "cache.sqlite"),
                  "--output-manifest", str(tmp_path / "summary.json"),
                  "--base-url", "http://unused", "--check-only", "--allow-unfiltered-data"])
    assert '"validated": true' in capsys.readouterr().out
    assert not (tmp_path / "cache.sqlite").exists()
    assert not (tmp_path / "summary.json").exists()


def test_structured_handoff_preserves_prompt_and_input_but_has_distinct_cache_identity(tmp_path):
    from context_compression.codex_compact_v0149 import structured_summary_contract
    c = structured_summary_contract(Tokenizer(), database_sha256='db', tokenizer_files={},
                                    generator_model='/model', max_summary_tokens=4096)
    ordinary = contract()
    assert c['prompt'] == ordinary['prompt'] == PROMPT
    assert c['generation_suffix_ids'] == ordinary['generation_suffix_ids']
    a, b = summary_request(event(0), c), summary_request(event(0), ordinary)
    assert a['input_ids'] == b['input_ids']
    assert a['request_sha256'] != b['request_sha256']
    p = tmp_path / 'cache.sqlite'
    SummaryCache(p, ordinary).close()
    with pytest.raises(ValueError, match='contract changed'):
        SummaryCache(p, c)


def test_structured_generation_validates_exact_token_output(monkeypatch):
    import json
    from context_compression.codex_compact_v0149 import HANDOFF_FIELDS, structured_summary_contract
    c = structured_summary_contract(Tokenizer(), database_sha256='db', tokenizer_files={},
                                    generator_model='/model', max_summary_tokens=4096)
    value = {k: ['Preserve this observation.'] for k in HANDOFF_FIELDS}
    ids = list(json.dumps(value).encode())
    calls = []
    def request(url, payload):
        calls.append(payload)
        return {'output_ids': ids + [777], 'meta_info': {'finish_reason': {'type': 'stop'}}}
    monkeypatch.setattr(prepare, 'request_json', request)
    tokenizer = SimpleNamespace(decode=lambda values, **kwargs: bytes(values).decode())
    result = prepare.generate_handoff('local', summary_request(event(0), c), c,
                                      context_limit=262144, tokenizer=tokenizer)
    assert result == (ids, 'stop')
    assert json.loads(calls[0]['sampling_params']['json_schema']) == c['output_schema']
    assert calls[0]['input_ids'] == summary_request(event(0), c)['input_ids']


@pytest.mark.parametrize('payload,finish', [
    ('not JSON', 'stop'),
    ('{}', 'stop'),
    ('{}', 'length'),
])
def test_structured_generation_rejects_invalid_or_truncated_output(monkeypatch, payload, finish):
    from context_compression.codex_compact_v0149 import structured_summary_contract
    c = structured_summary_contract(Tokenizer(), database_sha256='db', tokenizer_files={},
                                    generator_model='/model', max_summary_tokens=4096)
    monkeypatch.setattr(prepare, 'request_json', lambda *args, **kwargs: {
        'output_ids': list(payload.encode()) + [777],
        'meta_info': {'finish_reason': {'type': finish}}})
    tokenizer = SimpleNamespace(decode=lambda values, **kwargs: bytes(values).decode())
    with pytest.raises(ValueError):
        prepare.generate_handoff('local', summary_request(event(0), c), c,
                                 context_limit=262144, tokenizer=tokenizer)
