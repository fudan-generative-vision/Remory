import contextlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("shared_runtime.sglang_streaming")

from shared_runtime.sglang_streaming import (
    GenerationStreamingMixin,
    StreamProtocolError,
    _load_snapshot,
    _queued_peer_progress,
)


class BaseClient:
    from context_compression.grepseek_sglang_client import GrepSeekSGLangClient
    _open_request = GrepSeekSGLangClient._open_request

    def __init__(self, base_url, timeout=0.4):
        self.base_url, self.request_timeout = base_url, timeout
        self.aborts, self.nonstream = [], []
        self._residual_stopping = threading.Event()

    def _post(self, route, payload, *, request_timeout=None):
        self.nonstream.append((route, payload))
        return {"nonstream": True}

    def _abort_request(self, rid):
        self.aborts.append(rid)


class Client(GenerationStreamingMixin, BaseClient):
    pass


def frame(ids, *, finish=None, **metadata):
    return dict(
        output_ids=ids,
        text="".join(str(t) for t in ids),
        meta_info=dict(finish_reason=finish, **metadata),
    )


@contextlib.contextmanager
def serve(
    sequence, *, content_type="text/event-stream", status=200, scheduler_load=None
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/get_load" or scheduler_load is None:
                self.send_error(404)
                return
            payload = json.dumps(scheduler_load()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if self.path == "/abort_request":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"success":true}')
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            try:
                for item in sequence(body) if callable(sequence) else sequence:
                    if callable(item):
                        item()
                    elif isinstance(item, float):
                        time.sleep(item)
                    else:
                        wire = (
                            item
                            if isinstance(item, bytes)
                            else b"data: " + json.dumps(item).encode() + b"\n\n"
                        )
                        self.wfile.write(wire)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def client(tmp_path, url, *, incremental=False, timeout=0.4, cancelled=None):
    instance = Client(url, timeout)
    instance.events = []
    instance.configure_generation_streaming(
        artifact_dir=tmp_path,
        incremental=incremental,
        event_sink=instance.events.append,
        cancelled=cancelled,
    )
    instance.set_generation_stream_context(phase="generation", round_index=2)
    return instance


def generate(instance, **extra):
    return instance._post(
        "/generate",
        dict(
            rid="private-rid",
            input_ids=[11, 12],
            sampling_params={"max_new_tokens": 4096, "temperature": 1.0},
            **extra,
        ),
    )


def records(tmp_path):
    path = next((tmp_path / "generation-streams").glob("*.jsonl"))
    assert path.stat().st_mode & 0o777 == 0o600
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("incremental", [False, True])
def test_exact_tokens_metadata_and_payload_preserved(tmp_path, incremental):
    sequence = [
        frame([1, 2], prompt_tokens=17, cache_policy={"isolated": True}),
        frame(
            [3] if incremental else [1, 2, 3],
            finish={"type": "stop", "matched": 3},
            completion_tokens=3,
        ),
    ]
    with serve(sequence) as (url, requests):
        instance = client(tmp_path, url, incremental=incremental)
        result = generate(
            instance,
            image_data=["private-image"],
            input_embeds=[[0.2]],
            bypass_cache=True,
        )
    assert result["output_ids"] == [1, 2, 3]
    assert result["text"] == "123"
    assert result["meta_info"]["cache_policy"] == {"isolated": True}
    assert result["meta_info"]["finish_reason"] == {"type": "stop", "matched": 3}
    assert result["meta_info"]["generation_stream"]["status"] == "completed"
    assert result["meta_info"]["generation_stream"]["round_index"] == 2
    assert instance.aborts == []
    assert requests[0] == dict(
        rid="private-rid",
        input_ids=[11, 12],
        stream=True,
        sampling_params={"max_new_tokens": 4096, "temperature": 1.0},
        image_data=["private-image"],
        input_embeds=[[0.2]],
        bypass_cache=True,
    )
    rows = records(tmp_path)
    assert [r["token_ids"] for r in rows if r["event"] == "chunk"] == [[1, 2], [3]]
    assert "input_embeds" not in json.dumps(rows)


def test_healthy_long_decode_exceeds_timeout_in_total(tmp_path):
    seq = []
    for index in range(10):
        seq += [
            0.06,
            frame(
                list(range(index + 1)),
                finish={"type": "length"} if index == 9 else None,
            ),
        ]
    with serve(seq) as (url, _):
        instance = client(tmp_path, url, timeout=0.4)
        result = generate(instance)
    receipt = result["meta_info"]["generation_stream"]
    assert receipt["seconds"] > 0.4
    assert receipt["status"] == "completed"
    assert result["output_ids"] == list(range(10))


@pytest.mark.parametrize("ending", [[], [b"data: [DONE]\n\n"], [0.7]])
def test_disconnect_done_or_idle_timeout_retains_partial_and_aborts_only_own_rid(
    tmp_path, ending
):
    ids = list(range(1500))
    with serve([frame(ids, cache_policy={"isolated": True})] + ending) as (url, _):
        instance = client(tmp_path, url)
        result = generate(instance)
    assert result["output_ids"] == ids
    assert result["meta_info"]["finish_reason"]["type"] == "abort"
    receipt = result["meta_info"]["generation_stream"]
    assert receipt["status"] == "interrupted" and receipt["abort_acknowledged"]
    assert instance.aborts == ["private-rid"]
    assert records(tmp_path)[-1]["status"] == "interrupted"


@pytest.mark.parametrize(
    "sequence",
    [
        [frame([1]), frame([2])],
        [frame([1]), b"data: {invalid JSON}\n\n"],
        [frame([1]), {"error": {"message": "invalid arguments"}}],
        [frame([1]), frame([1], finish={"type": "abort", "status_code": 400})],
    ],
)
def test_protocol_or_admission_errors_fail_closed(tmp_path, sequence):
    with serve(sequence) as (url, _):
        instance = client(tmp_path, url)
        with pytest.raises(StreamProtocolError):
            generate(instance)
    assert records(tmp_path)[-1]["status"] == "failed"


def test_empty_interruption_cannot_fabricate_a_round(tmp_path):
    with serve([]) as (url, _):
        instance = client(tmp_path, url)
        with pytest.raises(RuntimeError, match="before any output"):
            generate(instance)


def load_row(tokens=20, *, waiting=1, running=1, tic=1):
    return dict(
        dp_rank=0,
        num_reqs=waiting + running,
        num_waiting_reqs=waiting,
        num_tokens=tokens,
        ts_tic=tic,
    )


def test_queued_generation_keeps_original_stream_while_other_slot_progresses(tmp_path):
    counter = [0]

    def load():
        counter[0] += 1
        return [load_row(tokens=20 + counter[0], tic=counter[0])]

    # Use a realistic local-I/O allowance on busy shared CI/cluster hosts;
    # the stream still exceeds two idle windows and requires peer progress.
    with serve([4.1, frame([7, 8], finish={"type": "stop"})], scheduler_load=load) as (
        url,
        requests,
    ):
        instance = client(tmp_path, url, timeout=1.5)
        result = generate(instance)
    assert result["output_ids"] == [7, 8]
    assert result["meta_info"]["generation_stream"]["seconds"] > 3.0
    assert len(requests) == 1 and instance.aborts == []
    assert any(r["event"] == "queued" for r in records(tmp_path))


def test_queue_progress_that_stops_eventually_times_out(tmp_path):
    counter = [0]

    def load():
        counter[0] += 1
        return [load_row(tokens=20 + min(counter[0], 3), tic=counter[0])]

    with serve([5.0], scheduler_load=load) as (url, requests):
        instance = client(tmp_path, url, timeout=1.5)
        with pytest.raises(RuntimeError, match="before any output"):
            generate(instance)
    rows = records(tmp_path)
    assert any(r["event"] == "queued" for r in rows)
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["seconds"] < 4.5
    assert len(requests) == 1


@pytest.mark.parametrize(
    "mode", ["static", "timestamp_only", "no_waiting", "no_running", "unavailable"]
)
def test_queue_wait_needs_real_peer_progress_not_just_http_health(tmp_path, mode):
    with serve([0.85, frame([7], finish={"type": "stop"})]) as (url, requests):
        instance = client(tmp_path, url, timeout=0.25)
        counter = [0]

        def load(timeout):
            counter[0] += 1
            if mode == "unavailable":
                raise TimeoutError("scheduler is unresponsive")
            return _load_snapshot(
                [
                    load_row(
                        tokens=20
                        if mode in ("static", "timestamp_only")
                        else 20 + counter[0],
                        tic=1 if mode == "static" else counter[0],
                        waiting=0 if mode == "no_waiting" else 1,
                        running=0 if mode == "no_running" else 1,
                    )
                ]
            )

        instance._stream_scheduler_load = load
        with pytest.raises(RuntimeError, match="before any output"):
            generate(instance)
    assert len(requests) == 1
    assert not any(r["event"] == "queued" for r in records(tmp_path))


def test_peer_progress_cannot_mask_stall_after_first_output(tmp_path):
    with serve([frame([1]), 0.85]) as (url, _):
        instance = client(tmp_path, url, timeout=0.25)

        def unexpected_load(timeout):
            pytest.fail("peer load must not extend a started decode's idle deadline")

        instance._stream_scheduler_load = unexpected_load
        result = generate(instance)
    assert result["output_ids"] == [1]
    assert result["meta_info"]["finish_reason"]["type"] == "abort"
    assert instance.aborts == ["private-rid"]


def test_queued_cancellation_remains_prompt_without_a_new_sse_frame(tmp_path):
    stop = threading.Event()
    with serve([0.85]) as (url, _):
        instance = client(tmp_path, url, timeout=10, cancelled=stop.is_set)
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="cancelled"):
            generate(instance)
        elapsed = time.monotonic() - started
        timer.join()
    assert elapsed < 1.2
    assert records(tmp_path)[-1]["status"] == "failed"


def test_scheduler_load_validation():
    baseline = _load_snapshot([load_row()])
    assert _queued_peer_progress(baseline, _load_snapshot([load_row(21, tic=2)]))
    assert not _queued_peer_progress(baseline, _load_snapshot([load_row(21, tic=0)]))
    for value in (
        {},
        [],
        [load_row(), load_row()],
        [load_row(tic=float("nan"))],
        [load_row(tokens=True)],
        [dict(load_row(), num_reqs=0)],
    ):
        assert _load_snapshot(value) == {}


def test_shutdown_does_not_turn_into_recovery(tmp_path):
    stop = threading.Event()
    with serve([frame([1, 2]), stop.set, 0.1]) as (url, _):
        instance = client(tmp_path, url, cancelled=stop.is_set)
        with pytest.raises(RuntimeError, match="cancelled"):
            generate(instance)
    assert records(tmp_path)[-1]["status"] == "failed"


def test_abort_failure_prevents_successor_generation(tmp_path):
    with serve([frame([1, 2])]) as (url, _):
        instance = client(tmp_path, url)

        def fail_abort(rid):
            raise RuntimeError("abort unreachable")

        instance._abort_request = fail_abort
        with pytest.raises(RuntimeError, match="abort unreachable"):
            generate(instance)


def test_progress_is_persisted_before_decode_completes(tmp_path):
    consumed = threading.Event()

    def observe():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            paths = list((tmp_path / "generation-streams").glob("*.jsonl"))
            if paths and '"event":"chunk"' in paths[0].read_text():
                consumed.set()
                return
            time.sleep(0.01)

    with serve([frame([1, 2]), observe, frame([1, 2, 3], finish={"type": "stop"})]) as (
        url,
        _,
    ):
        generate(client(tmp_path, url, timeout=3))
    assert consumed.is_set()


def test_hidden_state_encoding_and_other_routes_remain_nonstream(tmp_path):
    instance = client(tmp_path, "http://unused")
    payload = dict(sampling_params={"max_new_tokens": 0}, return_hidden_states=True)
    assert instance._post("/generate", payload) == {"nonstream": True}
    assert instance._post("/other", {"key": 1}) == {"nonstream": True}
    assert instance.nonstream == [("/generate", payload), ("/other", {"key": 1})]
    assert list((tmp_path / "generation-streams").iterdir()) == []


def test_partial_summary_is_rejected_even_if_body_contains_eos(tmp_path):
    from terminal_bench_eval.summary_policy import (
        require_summary_completion,
        summary_completion,
    )

    with serve([frame([1, 2, 0])]) as (url, _):
        response = generate(client(tmp_path, url))
    result = SimpleNamespace(
        token_ids=response["output_ids"],
        finished_on_eos=True,
        finish_reason=response["meta_info"]["finish_reason"],
        meta_info=response["meta_info"],
    )
    with pytest.raises(RuntimeError, match="did not complete safely"):
        require_summary_completion(summary_completion(result))


def test_real_residual_client_keeps_concurrent_rids_cache_receipts_and_no_regeneration(
    tmp_path,
):
    from context_compression.qwen38_residual_sglang_client import (
        SummaryResidualSGLangClient,
    )

    class RealClient(GenerationStreamingMixin, SummaryResidualSGLangClient):
        pass

    ready = threading.Barrier(2)

    def sequence(body):
        ready.wait(timeout=5)
        if "-left-" in body["rid"]:
            return [frame([1, 2], cache_policy="bypass")]
        return [
            frame([3, 4]),
            0.1,
            frame([3, 4, 99], finish={"type": "stop", "matched": 99}),
        ]

    with serve(sequence) as (url, requests):
        instances = []
        for name in ("left", "right"):
            instance = RealClient(url, request_timeout=3, eos_token_id=99)
            instance._request_id_namespace = name
            instance.attest_sglang_cache_policy(
                {
                    "max_running_requests": 2,
                    "request_capabilities": [
                        "request_cache_bypass_v1",
                        "qwen38_summary_residual_sglang_v1",
                        "qwen38_residual_cpu_concurrent_v1",
                    ],
                }
            )
            instance.configure_generation_streaming(artifact_dir=tmp_path / name)
            instance.configure_generation_degeneration_guard(
                True, return_detected_generation=True
            )
            instances.append(instance)
        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(
                instances[0].generate_discrete, [10], max_new_tokens=1024
            )
            right = executor.submit(
                instances[1].generate_discrete, [20], max_new_tokens=1024
            )
            partial, complete = left.result(timeout=5), right.result(timeout=5)
    assert partial.token_ids == (1, 2) and partial.finish_reason["type"] == "abort"
    assert not partial.finished_on_eos
    assert partial.meta_info["cache_policy"] == "bypass"
    assert (
        partial.meta_info["generation_degeneration_guard"][
            "retry_suppressed_finish_reason"
        ]
        == "abort"
    )
    assert complete.token_ids == (3, 4, 99) and complete.finished_on_eos
    generates = [body for body in requests if "input_ids" in body]
    aborts = [body["rid"] for body in requests if set(body) == {"rid"}]
    assert len(generates) == 2  # no silent same-prompt regeneration
    assert len(aborts) == 1 and "-left-" in aborts[0]
    assert all(body["stream"] is True for body in generates)
    assert all(not instance._active_request_ids for instance in instances)
