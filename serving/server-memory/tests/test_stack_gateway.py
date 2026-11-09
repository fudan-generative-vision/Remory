import asyncio
import contextlib
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from recursive_memory.stack.gateway import APP_KEY, create_app, load_config, request_units


@contextlib.asynccontextmanager
async def listen(app):
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


class FakeServer:
    def __init__(self, *, capacity=1, model="/model", capabilities=()):
        self.capacity, self.model = capacity, model
        self.capabilities = list(capabilities)
        self.active, self.external = 0, 0
        self.calls, self.aborted = [], []
        self.gate = asyncio.Event()

    async def info(self, request):
        return web.json_response({"model_path": self.model, "served_model_name": "native",
                                  "request_capabilities": self.capabilities,
                                  "max_running_requests": self.capacity, "api_key": "secret",
                                  "internal_states": [{"admin_api_key": "secret"}]})

    async def load(self, request):
        return web.json_response([{"num_reqs": self.active + self.external, "num_waiting_reqs": 0}])

    async def loads_v1(self, request):
        return web.json_response({"loads": [{"num_running_reqs": self.active + self.external,
                                             "num_waiting_reqs": 0}]})

    async def abort(self, request):
        self.aborted.append(await request.json())
        return web.json_response({"ok": True})

    async def generate(self, request):
        data = await request.json()
        self.calls.append((request.path, data, dict(request.headers)))
        self.active += 1
        try:
            if data.get("stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(b'data: {"text":"first"}\n\n')
                await self.gate.wait()
                await response.write(b"data: [DONE]\n\n")
                return response
            if data.get("wait"):
                await self.gate.wait()
            return web.json_response({"body": data}, status=data.get("status", 200))
        finally:
            self.active -= 1

    def app(self, *, modern_load=False):
        app = web.Application()
        app.router.add_get("/get_server_info", self.info)
        app.router.add_get("/get_load", self.load)
        if modern_load:
            app.router.add_get("/v1/loads", self.loads_v1)
        app.router.add_post("/abort_request", self.abort)
        for path in ["/generate", "/v1/chat/completions", "/v1/responses", "/close_session"]:
            app.router.add_post(path, self.generate)
        return app


def config(tmp_path, urls, **overrides):
    payload = {"max_pending": 8, "probe_interval": 0.02, "queue_timeout": 2,
               "request_timeout": 3, "backends": [
                   {"name": f"gpu{i}", "url": url, "model": "public", "max_inflight": 1}
                   for i, url in enumerate(urls)], **overrides}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(payload))
    return load_config(path)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


def test_openai_alias_auth_metadata_and_blocked_admin_routes(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url]), api_key="gateway-secret")
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                r = await client.get(proxy + "/v1/models")
                assert r.status == 401
                headers = {"Authorization": "Bearer gateway-secret"}
                r = await client.post(proxy + "/v1/chat/completions", headers=headers,
                                      json={"model": "public", "messages": [{"role": "user", "content": "hello"}]})
                assert r.status == 200 and r.headers["X-SGLang-Backend"] == "gpu0"
                assert (await r.json())["body"]["model"] == "native"
                assert "Authorization" not in backend.calls[0][2]
                r = await client.get(proxy + "/backends/gpu0/get_server_info", headers=headers)
                info = await r.json()
                assert "api_key" not in info and "admin_api_key" not in info["internal_states"][0]
                assert info["served_model_name"] == "public"
                r = await client.post(proxy + "/flush_cache", headers=headers, json={})
                assert r.status == 404 and len(backend.calls) == 1
                r = await client.post(proxy + "/abort_request", headers=headers, json={"abort_all": True})
                assert r.status == 400 and not backend.aborted
    asyncio.run(check())


def test_sse_passes_immediately_holds_capacity_and_disconnect_aborts_only_its_rid(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url]))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                response = await client.post(proxy + "/generate", json={"text": "a", "stream": True, "rid": "owned"})
                first = await asyncio.wait_for(response.content.readline(), 1)
                assert b"first" in first and app[APP_KEY].backends[0].active == 1
                response.close()
                await until(lambda: bool(backend.aborted))
                assert backend.aborted == [{"rid": "owned", "abort_all": False}]
                await until(lambda: app[APP_KEY].backends[0].active == 0)
                assert not app[APP_KEY].native_requests
                backend.gate.set()
    asyncio.run(check())


def test_queued_disconnect_and_timeout_never_reach_upstream(tmp_path):
    async def check():
        backend = FakeServer()
        backend.external = 1
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url], queue_timeout=0.2))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                task = asyncio.create_task(client.post(proxy + "/generate", json={"text": "cancel me"}))
                await until(lambda: app[APP_KEY].queue.pending == 1)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await until(lambda: app[APP_KEY].queue.pending == 0)
                r = await client.post(proxy + "/generate", json={"text": "timeout"})
                assert r.status == 504
                assert not backend.calls and not backend.aborted and not app[APP_KEY].native_requests
                backend.external = 0
                r = await client.post(proxy + "/generate", json={"text": "now fits"})
                assert r.status == 200 and len(backend.calls) == 1
    asyncio.run(check())


def test_balancing_pinning_identity_and_no_generation_retry(tmp_path):
    async def check():
        first, second = FakeServer(), FakeServer()
        first.external = 1
        async with listen(first.app()) as a, listen(second.app()) as b:
            app = create_app(config(tmp_path, [a, b], queue_timeout=0.1))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                r = await client.post(proxy + "/v1/chat/completions", json={"model": "public", "status": 503})
                assert r.status == 503 and len(second.calls) == 1 and not first.calls
                r = await client.post(proxy + "/generate", json={"text": "stateful", "session_params": {}})
                assert r.status == 400
                r = await client.post(proxy + "/backends/gpu0/generate", json={"text": "pinned"})
                assert r.status == 504 and not first.calls
                first.external = 0
                first.model = "/wrong-model"
                await until(lambda: not app[APP_KEY].backends[0].healthy)
                assert "identity changed" in app[APP_KEY].backends[0].error
    asyncio.run(check())


def test_config_and_batch_admission_reject_capacity_bypass(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        config(tmp_path, ["http://localhost:1", "http://localhost:1"])
    assert request_units("/generate", {"input_ids": [[1], [2]], "sampling_params": {"n": 2}}) == 4
    assert request_units("/v1/completions", {"prompt": [1, 2], "n": 2}) == 2
    with pytest.raises(ValueError):
        request_units("/v1/chat/completions", {"n": True})


def test_residual_body_over_old_64_mib_limit_reaches_backend(tmp_path):
    async def check():
        backend = FakeServer()
        received = []

        async def receive(request):
            data = await request.json()
            values = data["custom_params"]["specforge_config"]["residual"]["summary_features"]
            received.append((len(values), values[0], values[-1]))
            return web.json_response({"ok": True})

        # Avoid retaining or echoing the large body in the fake server.
        upstream = web.Application(client_max_size=1024 * 1024 * 1024)
        upstream.router.add_get("/get_server_info", backend.info)
        upstream.router.add_get("/get_load", backend.load)
        upstream.router.add_post("/generate", receive)
        count = 3_000_000
        body = (b'{"input_ids":[1],"custom_params":{"specforge_config":{"residual":{"summary_features":['
                + b'0.1234567890123456789012,' * (count - 1)
                + b'0.1234567890123456789012]}}}}')
        assert len(body) > 64 * 1024 * 1024
        async with listen(upstream) as url:
            cfg = config(tmp_path, [url], request_timeout=30)
            assert cfg["max_body_bytes"] == 1024 * 1024 * 1024
            app = create_app(cfg)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                response = await client.post(proxy + "/groups/tb21/generate", data=body)
                assert response.status == 200
                assert await response.json() == {"ok": True}
                assert received == [(count, 0.1234567890123456789012, 0.1234567890123456789012)]
                assert app[APP_KEY].backends[0].active == 0
    asyncio.run(check())


def test_configured_body_limit_reports_413_without_dispatch_or_capacity_leak(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url], max_body_bytes=1024))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                response = await client.post(proxy + "/generate", json={"text": "x" * 2048})
                assert response.status == 413
                assert (await response.json())["error"]["code"] == "request_body_too_large"
                assert not backend.calls and not backend.aborted
                gateway = app[APP_KEY]
                assert gateway.uploading == gateway.queue.pending == gateway.backends[0].active == 0
                assert gateway.failed == 1 and not gateway.native_requests
                status = await (await client.get(proxy + "/queue/status")).json()
                assert status["max_body_bytes"] == 1024
                assert status["inflight_handlers"] == 0
                response = await client.post(proxy + "/generate", json={"text": "ok"})
                assert response.status == 200 and len(backend.calls) == 1
    asyncio.run(check())


def test_state_cleanup_bypasses_full_queue_and_responses_are_explicitly_stateless(tmp_path):
    async def check():
        backend = FakeServer()
        backend.external = 1
        async with listen(backend.app(modern_load=True)) as url:
            app = create_app(config(tmp_path, [url]))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                assert app[APP_KEY].backends[0].observed_load == 1
                queued = asyncio.create_task(client.post(proxy + "/generate", json={"text": "wait"}))
                await until(lambda: app[APP_KEY].queue.pending == 1)
                r = await client.post(proxy + "/backends/gpu0/close_session", json={"session_id": "s"})
                assert r.status == 200 and app[APP_KEY].queue.pending == 1
                r = await client.post(proxy + "/v1/responses", json={"model": "public", "input": "x"})
                assert r.status == 400
                backend.external = 0
                assert (await queued).status == 200
                r = await client.post(proxy + "/v1/responses", json={"model": "public", "input": "x", "store": False})
                assert r.status == 200
                r = await client.post(proxy + "/backends/gpu0/v1/responses", json={"input": "x", "background": True})
                assert r.status == 400
    asyncio.run(check())


def test_remote_memory_is_pinned_queued_and_lease_can_bypass_full_queue(tmp_path):
    async def check():
        backend = FakeServer()
        backend.external = 1
        upstream = backend.app(modern_load=True)
        for path in ("/remory/memory", "/remory/memory/lease", "/release_memory_handle"):
            upstream.router.add_post(path, backend.generate)
        async with listen(upstream) as url:
            app = create_app(config(tmp_path, [url]))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                unpinned = await client.post(proxy + "/remory/memory", json={"payload": {"operation": "save"}})
                assert unpinned.status == 400
                nested = {"sampling_params": {"custom_params": {"hook": {"memory_handle_publish": {}}}}}
                assert (await client.post(proxy + "/generate", json=nested)).status == 400
                queued = asyncio.create_task(client.post(proxy + "/groups/main/backends/gpu0/remory/memory",
                    json={"payload": {"operation": "compose"}}))
                await until(lambda: app[APP_KEY].queue.pending == 1)
                for path, body in (("/remory/memory/lease", {"payload": {"operation": "renew", "handles": []}}),
                                   ("/release_memory_handle", {"handle": {}})):
                    reply = await client.post(proxy + "/groups/main/backends/gpu0" + path, json=body)
                    assert reply.status == 200
                    assert app[APP_KEY].queue.pending == 1
                backend.external = 0
                assert (await queued).status == 200
                assert backend.calls[-1][0] == "/remory/memory"
    asyncio.run(check())


def test_native_abort_owns_rid_and_request_timeout_releases_capacity(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url], request_timeout=0.1))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                response = await client.post(proxy + "/generate", json={"text": "x", "wait": True, "rid": "timeout"})
                assert response.status == 504
                await until(lambda: bool(backend.aborted))
                assert backend.aborted == [{"rid": "timeout", "abort_all": False}]
                await until(lambda: app[APP_KEY].backends[0].active == 0)
                backend.external = 1
                await until(lambda: app[APP_KEY].backends[0].observed_load == 1)
                pending = asyncio.create_task(client.post(proxy + "/generate", json={"text": "wait", "rid": "pending"}, headers={"X-Queue-Group": "owner"}))
                await until(lambda: app[APP_KEY].queue.pending == 1)
                r = await client.post(proxy + "/abort_request", json={"rid": "pending"})
                assert r.status == 404
                r = await client.post(proxy + "/abort_request", json={"rid": "pending"}, headers={"X-Queue-Group": "owner"})
                assert r.status == 200
                with contextlib.suppress(aiohttp.ClientError):
                    await pending
                await until(lambda: app[APP_KEY].queue.pending == 0)
                assert len(backend.aborted) == 1  # queued cancellation never hits SGLang
    asyncio.run(check())


def test_group_base_url_carries_generation_metadata_and_cancellation(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url]))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                base = proxy + "/groups/jobbench"
                assert (await client.get(base + "/health")).status == 200
                metadata = await (await client.get(base + "/get_server_info")).json()
                r = await client.post(base + "/v1/chat/completions", json={"model": metadata["served_model_name"]})
                assert r.status == 200 and (await r.json())["body"]["model"] == "native"
                r = await client.post(base + "/generate", json={"text": "x"}, headers={"X-Queue-Group": "conflict"})
                assert r.status == 400
                backend.external = 1
                await until(lambda: app[APP_KEY].backends[0].observed_load == 1)
                pending = asyncio.create_task(client.post(base + "/generate", json={"text": "x", "rid": "group-rid"}))
                await until(lambda: app[APP_KEY].queue.pending == 1)
                assert list(app[APP_KEY].queue.groups) == ["jobbench"]
                r = await client.post(base + "/abort_request", json={"rid": "group-rid"})
                assert r.status == 200
                with contextlib.suppress(aiohttp.ClientError):
                    await pending
                assert not backend.aborted
    asyncio.run(check())


@pytest.mark.parametrize("batched", [False, True])
def test_local_encoder_versions_wait_for_local_registry_while_plain_requests_use_remote(tmp_path, batched):
    async def check():
        local = FakeServer(capacity=2, capabilities=["remory_encoder_versions_v1"])
        # Even an advertised remote capability does not imply a shared registry.
        remote = FakeServer(capacity=2, capabilities=["remory_encoder_versions_v1"])
        local.external = 2
        async with listen(local.app()) as a, listen(remote.app()) as b:
            cfg = config(tmp_path, [a, b])
            for row in cfg["backends"]:
                row["max_inflight"] = 2
            cfg["backends"][1]["supports_local_encoder_versions"] = False
            app = create_app(cfg)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                sampling = {"custom_params": {"remory_encoder_version": "local-trained-v7"}}
                payload = {"input_ids": [[1], [2]] if batched else [1],
                           "sampling_params": [{}, sampling] if batched else sampling}
                pending = asyncio.create_task(client.post(proxy + "/groups/versioned/generate", json=payload))
                try:
                    await until(lambda: app[APP_KEY].queue.pending == 1)
                    response = await client.post(proxy + "/groups/plain/generate", json={"text": "plain"})
                    assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu1"
                    assert len(remote.calls) == 1 and not local.calls and not pending.done()
                    local.external = 0
                    response = await pending
                    assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu0"
                    assert local.calls[0][1]["sampling_params"] == payload["sampling_params"]
                    assert len(remote.calls) == 1
                finally:
                    if not pending.done():
                        pending.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pending
    asyncio.run(check())


def test_version_capability_metadata_is_routable_and_other_capabilities_remain_intersection(tmp_path):
    async def check():
        capability = "remory_encoder_versions_v1"
        local = FakeServer(capabilities=[capability, "common", "local_only"])
        remote = FakeServer(capabilities=["common", "remote_only"])
        async with listen(local.app()) as a, listen(remote.app()) as b:
            cfg = config(tmp_path, [a, b])
            cfg["backends"][1]["supports_local_encoder_versions"] = False
            app = create_app(cfg)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                info = await (await client.get(proxy + "/get_server_info")).json()
                assert info["request_capabilities"] == ["common", capability]
                # Capability is removed if no healthy registry-sharing backend remains.
                local.model = "/changed"
                await until(lambda: not app[APP_KEY].backends[0].healthy)
                info = await (await client.get(proxy + "/get_server_info")).json()
                assert capability not in info["request_capabilities"]
    asyncio.run(check())


@pytest.mark.parametrize("shared_registry,advertised", [(False, True), (True, False)])
@pytest.mark.parametrize("route", ["/generate", "/close_session"])
def test_pinned_encoder_version_requires_registry_configuration_and_advertised_capability(
        tmp_path, shared_registry, advertised, route):
    async def check():
        backend = FakeServer(capabilities=["remory_encoder_versions_v1"] if advertised else [])
        async with listen(backend.app()) as url:
            cfg = config(tmp_path, [url])
            cfg["backends"][0]["supports_local_encoder_versions"] = shared_registry
            app = create_app(cfg)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                response = await client.post(proxy + "/backends/gpu0" + route, json={
                    "text": "versioned", "sampling_params": {
                        "custom_params": {"remory_encoder_version": "unknown-local-version"}}})
                assert response.status == 400
                assert "registry" in (await response.json())["error"]["message"]
                assert not backend.calls and app[APP_KEY].queue.pending == 0
                info = await (await client.get(proxy + "/backends/gpu0/get_server_info")).json()
                assert "remory_encoder_versions_v1" not in info["request_capabilities"]
    asyncio.run(check())


def test_encoder_registry_config_requires_boolean(tmp_path):
    with pytest.raises(ValueError, match="supports_local_encoder_versions must be boolean"):
        config(tmp_path, [], backends=[{"name": "remote", "url": "http://localhost:1",
            "model": "public", "max_inflight": 1, "supports_local_encoder_versions": "false"}])


class PortableServer(FakeServer):
    def __init__(self, *, compressor="a", contract="b", target="c"):
        super().__init__(capabilities=["remory_encoder_versions_v1", "remory_server_memory_v1",
                                      "qwen38_summary_residual_sglang_v1", "common"])
        self.artifact = {"schema": "remory_artifact_identity_v1", "compressor_sha256": compressor * 64,
                         "contract_sha256": contract * 64, "target_metadata_sha256": target * 64}
        self.dtype = "bfloat16"

    async def info(self, request):
        response = await super().info(request)
        info = json.loads(response.text)
        info.update(remory_artifact_identity=self.artifact, dtype=self.dtype, quantization=None,
                    kv_cache_dtype="auto", forward_hooks=[{
                        "hook_factory": "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook",
                        "config": {"checkpoint": "/fixture/compressor", "residual_serving_abi": "fixture"}}])
        return web.json_response(info)


def stateless_app(tmp_path, local, urls):
    cfg = config(tmp_path, urls)
    cfg["backends"][1].update(stateless_only=True, supports_local_encoder_versions=False)
    app = create_app(cfg)
    # Inject the verified checkpoint metadata, isolating routing tests from model files.
    app[APP_KEY].local_contracts["gpu1"] = {
        "checkpoint": "/fixture/local", "target": "/model", "identity": dict(local.artifact)}
    return app


def test_stateless_remote_accepts_plain_generation_but_queues_compressor_requests_locally(tmp_path):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        local.external = 1
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                payload = {"input_ids": [1], "sampling_params": {"custom_params": {
                    "specforge_context_compressor": {"operation": "encode"}}}}
                pending = asyncio.create_task(client.post(proxy + "/groups/memory/generate", json=payload))
                try:
                    await until(lambda: app[APP_KEY].queue.pending == 1)
                    for route, body in [("/generate", {"text": "Summarize this paragraph"}),
                                        ("/v1/chat/completions", {"model": "public", "messages": []})]:
                        response = await client.post(proxy + "/groups/plain" + route, json=body)
                        assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu1"
                    assert len(remote.calls) == 2 and not pending.done() and not local.calls
                    local.external = 0
                    response = await pending
                    assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu0"
                    assert local.calls[0][1]["sampling_params"] == payload["sampling_params"]
                    info = await (await client.get(proxy + "/get_server_info")).json()
                    assert info["remory_artifact_identity"] == local.artifact
                    assert "qwen38_summary_residual_sglang_v1" in info["request_capabilities"]
                    assert "remory_encoder_versions_v1" in info["request_capabilities"]
                    assert [row["name"] for row in info["shared_gateway"]["memory_backends"]] == ["gpu0"]
                    remote_info = await (await client.get(proxy + "/backends/gpu1/get_server_info")).json()
                    assert remote_info["remory_artifact_identity"] == remote.artifact
                    assert remote_info["stateless_only"] is True and remote_info["request_capabilities"] == []
                finally:
                    if not pending.done():
                        pending.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pending
    asyncio.run(check())


@pytest.mark.parametrize("route,payload", [
    ("/generate", {"sampling_params": {"custom_params": {"specforge_context_compressor": {}}}}),
    ("/generate", {"sampling_params": [{"custom_params": {"memory_embeddings": [[1.0]]}}]}),
    ("/generate", {"nested": [{"summary_handle": {}}]}),
    ("/generate", {"sampling_params": {"custom_params": {"remory_encoder_version": "v2"}}}),
    ("/generate", {"session_params": {}}),
    ("/generate", {"input_embeds": [[1.0]]}),
    ("/generate", {"return_hidden_states": True}),
    ("/encode", {"input_ids": [1]}),
    ("/remory/memory", {"payload": {"operation": "save"}}),
    ("/close_session", {"session_id": "s"}),
    ("/v1/responses", {"store": True}),
])
def test_stateless_remote_rejects_pinned_memory_and_tensor_requests(tmp_path, route, payload):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                response = await client.post(proxy + "/backends/gpu1" + route, json=payload)
                assert response.status == 400
                assert not remote.calls and not local.calls
                assert app[APP_KEY].queue.pending == 0
    asyncio.run(check())


@pytest.mark.parametrize("mismatch", ["target", "dtype", "no_local_anchor"])
def test_stateless_remote_bad_base_identity_is_quarantined_without_poisoning_local(tmp_path, mismatch):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        if mismatch == "target":
            remote.artifact["target_metadata_sha256"] = "f" * 64
        elif mismatch == "dtype":
            remote.dtype = "float16"
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            if mismatch == "no_local_anchor":
                app[APP_KEY].local_contracts.clear()
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                gateway = app[APP_KEY]
                await gateway.probe(gateway.backends[1])
                assert gateway.backends[0].healthy and not gateway.backends[1].healthy
                assert "stateless_only" in gateway.backends[1].error
                response = await client.post(proxy + "/generate", json={"text": "plain"})
                assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu0"
                assert not remote.calls
    asyncio.run(check())


def test_stateless_config_requires_explicit_boolean_and_checkpoint_anchor(tmp_path):
    for value, message in [("true", "must be boolean"), (True, "local_checkpoint")]:
        with pytest.raises(ValueError, match=message):
            config(tmp_path, [], backends=[{"name": "remote", "url": "http://localhost:1",
                "model": "public", "max_inflight": 1, "stateless_only": value}])


def test_stateless_remote_does_not_lower_full_backend_residual_capacity_metadata(tmp_path, monkeypatch):
    from pathlib import Path

    workspace = Path(__file__).resolve().parents[1] / "src/recursive_memory/stack/payload/workspace"
    monkeypatch.syspath_prepend(str(workspace))
    from shared_runtime.residual_preflight import assert_residual_concurrency

    class LimitedServer(PortableServer):
        async def info(self, request):
            response = await super().info(request)
            info = json.loads(response.text)
            info.update(context_length=self.capacity * 1000,
                        max_total_num_tokens=self.capacity * 2000,
                        max_req_input_len=self.capacity * 500)
            return web.json_response(info)

    async def check():
        local, remote = LimitedServer(), LimitedServer(compressor="d", contract="e")
        local.capacity, remote.capacity = 4, 2
        local.capabilities.append("qwen38_residual_cpu_concurrent_v1")
        remote.capabilities = ["common"]
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                info = await (await client.get(proxy + "/get_server_info")).json()
                limits = ("max_running_requests", "context_length", "max_total_num_tokens", "max_req_input_len")
                assert {k: info[k] for k in limits} == dict(zip(limits, (4, 4000, 8000, 2000)))
                assert_residual_concurrency(info, 4)
                remote_info = await (await client.get(proxy + "/backends/gpu1/get_server_info")).json()
                assert {k: remote_info[k] for k in limits} == dict(zip(limits, (2, 2000, 4000, 1000)))
                assert remote_info["stateless_only"] is True
    asyncio.run(check())


def tensor_recover_payload():
    return {"input_ids": [1, 2, 3], "cache_policy": "bypass", "custom_logit_processor": "fixture-noop",
            "sampling_params": {"custom_params": {
        "specforge_context_compressor": {"version": 1, "mode": "recover",
            "memory_positions": [1], "memory_embeddings": [[0.25, -0.5]]}}}}


def enable_tensor_recover(app, remote):
    remote.capabilities.extend(["qwen38_residual_cpu_concurrent_v1", "request_cache_bypass_v1"])
    backend = app[APP_KEY].backends[1]
    backend.allow_tensor_recover = True
    backend.tensor_recover_artifact_identity = dict(remote.artifact)


def test_verified_tensor_recover_can_use_opted_in_remote_while_encode_stays_local(tmp_path):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        local.external = 1
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            enable_tensor_recover(app, remote)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                payload = tensor_recover_payload()
                response = await client.post(proxy + "/generate", json=payload)
                assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu1"
                assert remote.calls[0][1]["sampling_params"] == payload["sampling_params"]
                info = await (await client.get(proxy + "/backends/gpu1/get_server_info")).json()
                assert info["remory_artifact_identity"] == remote.artifact
                assert info["tensor_recover_only"] is True and info["request_capabilities"] == []
                encode = tensor_recover_payload()
                encode["sampling_params"]["custom_params"]["specforge_context_compressor"]["mode"] = "encode"
                pending = asyncio.create_task(client.post(proxy + "/groups/encode/generate", json=encode))
                try:
                    await until(lambda: app[APP_KEY].queue.pending == 1)
                    assert len(remote.calls) == 1
                    local.external = 0
                    response = await pending
                    assert response.status == 200 and response.headers["X-SGLang-Backend"] == "gpu0"
                finally:
                    if not pending.done():
                        pending.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pending
    asyncio.run(check())


@pytest.mark.parametrize("variant", ["opt_out", "missing_capability", "encoder_version", "encode", "handle",
                                     "extra_field", "nan", "ragged", "bad_position", "cache", "batch", "no_processor"])
def test_tensor_recover_exception_rejects_unverified_or_stateful_requests(tmp_path, variant):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            enable_tensor_recover(app, remote)
            if variant == "opt_out":
                app[APP_KEY].backends[1].allow_tensor_recover = False
            if variant == "missing_capability":
                remote.capabilities.remove("request_cache_bypass_v1")
            payload = tensor_recover_payload()
            custom = payload["sampling_params"]["custom_params"]
            recover = custom["specforge_context_compressor"]
            if variant == "encoder_version":
                custom["remory_encoder_version"] = "local-v2"
            elif variant == "encode":
                recover["mode"] = "encode"
            elif variant == "handle":
                recover["memory_handle"] = {}
            elif variant == "extra_field":
                recover["extra"] = True
            elif variant == "nan":
                recover["memory_embeddings"][0][0] = float("nan")
            elif variant == "ragged":
                recover["memory_positions"] = [1, 2]
                recover["memory_embeddings"].append([0.1])
            elif variant == "bad_position":
                recover["memory_positions"] = [3]
            elif variant == "cache":
                payload["cache_policy"] = "soft_reuse"
            elif variant == "batch":
                payload["sampling_params"] = [payload["sampling_params"]]
            elif variant == "no_processor":
                payload.pop("custom_logit_processor")
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                response = await client.post(proxy + "/backends/gpu1/generate", json=payload)
                assert response.status == 400
                assert not remote.calls and not local.calls
    asyncio.run(check())


def test_tensor_recover_raw_artifact_pin_change_quarantines_remote(tmp_path):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            enable_tensor_recover(app, remote)
            async with listen(app):
                await until(lambda: app[APP_KEY].backends[1].healthy)
                remote.artifact["compressor_sha256"] = "f" * 64
                await app[APP_KEY].probe(app[APP_KEY].backends[1])
                assert not app[APP_KEY].backends[1].healthy
                assert "pinned raw artifact" in app[APP_KEY].backends[1].error
                assert app[APP_KEY].backends[0].healthy
    asyncio.run(check())


def test_tensor_recover_configuration_requires_explicit_artifact_pin(tmp_path):
    row = {"name": "remote", "url": "http://localhost:1", "model": "public", "max_inflight": 1,
           "stateless_only": True, "local_checkpoint": str(tmp_path), "allow_tensor_recover": True}
    with pytest.raises(ValueError, match="exact tensor_recover_artifact_identity pin"):
        config(tmp_path, [], backends=[row])
    identity = {"schema": "remory_artifact_identity_v1", "compressor_sha256": "a" * 64,
                "contract_sha256": "b" * 64, "target_metadata_sha256": "c" * 64}
    row["tensor_recover_artifact_identity"] = identity
    assert config(tmp_path, [], backends=[row])["backends"][0]["allow_tensor_recover"] is True
    with pytest.raises(ValueError, match="exact tensor_recover_artifact_identity pin"):
        config(tmp_path, [], backends=[{**row, "stateless_only": False}])
    with pytest.raises(ValueError, match="allow_tensor_recover must be boolean"):
        config(tmp_path, [], backends=[{**row, "allow_tensor_recover": "true"}])


def test_verified_remote_recover_strips_encoder_version_only_on_remote_dispatch(tmp_path):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor="d", contract="e")
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            enable_tensor_recover(app, remote)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                payload = tensor_recover_payload()
                custom = payload["sampling_params"]["custom_params"]
                version = "f" * 64
                custom["remory_encoder_version"] = version
                response = await client.post(proxy + "/backends/gpu1/generate", json=payload)
                assert response.status == 200
                assert response.headers["X-Remory-Encoder-Version-Handling"] == "ignored-for-tensor-recover"
                remote_custom = remote.calls[0][1]["sampling_params"]["custom_params"]
                assert "remory_encoder_version" not in remote_custom
                assert remote_custom["specforge_context_compressor"] == custom["specforge_context_compressor"]
                response = await client.post(proxy + "/backends/gpu0/generate", json=payload)
                assert response.status == 200
                assert "X-Remory-Encoder-Version-Handling" not in response.headers
                assert local.calls[0][1]["sampling_params"]["custom_params"] == custom
                # The same version never grants a remote encode or ordinary request.
                for next_custom in ({"remory_encoder_version": version},
                        {**custom, "specforge_context_compressor": {
                            **custom["specforge_context_compressor"], "mode": "encode"}}):
                    response = await client.post(proxy + "/backends/gpu1/generate", json={
                        **payload, "sampling_params": {"custom_params": next_custom}})
                    assert response.status == 400
                assert len(remote.calls) == 1 and len(local.calls) == 1
    asyncio.run(check())


def test_completed_abort_is_idempotent_but_unknown_and_cross_group_are_not(tmp_path):
    async def check():
        backend = FakeServer()
        async with listen(backend.app()) as url:
            app = create_app(config(tmp_path, [url], request_timeout=None, queue_timeout=None))
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                r = await client.post(proxy + '/groups/research/generate', json={'text':'ok','rid':'finished'})
                assert r.status == 200
                await r.read()
                await until(lambda: not app[APP_KEY].native_requests)
                r = await client.post(proxy + '/groups/research/abort_request', json={'rid':'finished'})
                assert r.status == 200 and (await r.json())['already_finished']
                for group, rid in [('other','finished'), ('research','unknown')]:
                    r = await client.post(proxy + f'/groups/{group}/abort_request', json={'rid':rid})
                    assert r.status == 404
    asyncio.run(check())


def test_group_route_policy_applies_to_metadata_admission_and_pins(tmp_path):
    async def check():
        first, second = FakeServer(), FakeServer()
        async with listen(first.app()) as a, listen(second.app()) as b:
            settings = config(tmp_path, [a,b])
            settings['backends'][1]['excluded_groups'] = ['research']
            app = create_app(settings)
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                r = await client.get(proxy + '/groups/research/get_server_info')
                assert (await r.json())['shared_gateway']['backends'] == ['gpu0']
                r = await client.post(proxy + '/groups/research/generate', json={'text':'ok'})
                assert r.status == 200 and r.headers['X-SGLang-Backend'] == 'gpu0'
                r = await client.post(proxy + '/groups/research/backends/gpu1/generate', json={'text':'no'})
                assert r.status == 404
                r = await client.post(proxy + '/groups/other/backends/gpu1/generate', json={'text':'ok'})
                assert r.status == 200
    asyncio.run(check())


def test_native_request_bytes_are_preserved_without_reencoding(tmp_path):
    async def check():
        backend=FakeServer()
        seen=[]
        async def capture(request):
            seen.append(await request.read())
            return web.json_response({'ok':True})
        app=backend.app()
        app.router.add_post('/encode',capture)
        async with listen(app) as url:
            proxy_app=create_app(config(tmp_path,[url]))
            async with listen(proxy_app) as proxy,aiohttp.ClientSession() as client:
                body=b'{ "input_ids" : [1, 2], "sampling_params" : {"max_new_tokens":0} }'
                response=await client.post(proxy+'/encode',data=body,headers={'Content-Type':'application/json'})
                assert response.status==200
                assert seen==[body]
    asyncio.run(check())


def test_slow_request_inspection_does_not_block_health_or_cancellation(tmp_path,monkeypatch):
    import threading
    from recursive_memory.stack import gateway
    inspect=gateway.inspect_request
    began,release=threading.Event(),threading.Event()
    def slow(raw,path,eligible):
        began.set()
        assert release.wait(5)
        return inspect(raw,path,eligible)
    monkeypatch.setattr(gateway,'inspect_request',slow)
    async def check():
        backend=FakeServer()
        async with listen(backend.app()) as url:
            app=create_app(config(tmp_path,[url]))
            async with listen(app) as proxy,aiohttp.ClientSession() as client:
                request=asyncio.create_task(client.post(proxy+'/generate',json={'text':'large body'}))
                try:
                    await until(began.is_set)
                    async with asyncio.timeout(.5):
                        response=await client.get(proxy+'/health')
                        assert response.status==200
                finally:
                    release.set()
                response=await request
                assert response.status==200
    asyncio.run(check())


@pytest.mark.parametrize('recover', [False, True])
def test_large_research_requests_stay_local_small_requests_use_remote(tmp_path, recover):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor='d', contract='e')
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            enable_tensor_recover(app, remote)
            app[APP_KEY].backends[1].max_request_body_bytes = 1024
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                large = tensor_recover_payload() if recover else {'text': 'plain'}
                large['input_ids'] = [1] * 1200
                response = await client.post(proxy + '/groups/research/generate', json=large)
                assert response.status == 200 and response.headers['X-SGLang-Backend'] == 'gpu0'
                assert not remote.calls
                small = tensor_recover_payload() if recover else {'text': 'plain'}
                response = await client.post(proxy + '/groups/research/generate', json=small)
                assert response.status == 200 and response.headers['X-SGLang-Backend'] == 'gpu1'
                status = await (await client.get(proxy + '/queue/status')).json()
                row = status['backends'][1]
                assert row['routed_requests'] == row['successful_requests'] == 1
                assert row['failed_requests'] == 0 and row['last_request_group'] == 'research'
                assert row['last_request_kind'] == ('tensor_recover' if recover else 'generation')
    asyncio.run(check())


def test_rewritten_body_is_rerouted_before_any_remote_dispatch(tmp_path):
    async def check():
        local, remote = PortableServer(), PortableServer(compressor='d', contract='e')
        async with listen(local.app()) as a, listen(remote.app()) as b:
            app = stateless_app(tmp_path, local, [a, b])
            app[APP_KEY].backends[1].max_request_body_bytes = 150
            async with listen(app) as proxy, aiohttp.ClientSession() as client:
                await until(lambda: app[APP_KEY].backends[1].healthy)
                # The compact UTF-8 request fits; model alias rewriting expands
                # its Unicode strings. Prefer remote so the late check is tested.
                app[APP_KEY].backends[0].starts = 10
                data = {'model': 'public', 'messages': [{'role': 'user', 'content': '文' * 20}]}
                raw = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()
                assert len(raw) < 150 < len(json.dumps(data).encode())
                response = await client.post(proxy + '/v1/chat/completions', data=raw,
                                             headers={'Content-Type': 'application/json'})
                assert response.status == 200 and response.headers['X-SGLang-Backend'] == 'gpu0'
                assert not remote.calls and len(local.calls) == 1
                assert local.calls[0][1]['messages'] == data['messages']
    asyncio.run(check())


@pytest.mark.parametrize('limit', [True, 0, -1, 1.5, '1024'])
def test_backend_body_limit_validation(tmp_path, limit):
    with pytest.raises(ValueError, match='max_request_body_bytes'):
        config(tmp_path, [], backends=[dict(name='remote', url='http://localhost:1',
            model='public', max_inflight=1, max_request_body_bytes=limit)])
