"""Exercise authenticated HTTP, two real gateway hops, and portable identity."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error

import aiohttp
from aiohttp import web
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/recursive_memory/stack/payload/workspace"))
from context_compression.grepseek_sglang_client import GrepSeekSGLangClient
from context_compression.sglang_http import open_request, request_headers
from context_compression.remory_identity import artifact_identity, check_server_identity, FIELD, SCHEMA
from recursive_memory.stack.gateway import APP_KEY, create_app, Gateway
from shared_runtime.residual_preflight import _read_health, _read_server_info
from shared_runtime.sglang_streaming import GenerationStreamingMixin
from test_stack_gateway import FakeServer, listen, config, until


class Client(GenerationStreamingMixin, GrepSeekSGLangClient):
    pass


def test_auth_all_client_routes_key_rotation_and_no_redirects(tmp_path, monkeypatch):
    async def check():
        seen, leaks = [], []
        key = tmp_path / "remote.key"
        key.write_text("first-private-key")

        async def sink(request):
            leaks.append(dict(request.headers))
            return web.json_response({})

        sink_app = web.Application()
        sink_app.router.add_route("*", "/{route:.*}", sink)
        async with listen(sink_app) as other:
            async def endpoint(request):
                assert request.headers.get("Authorization") == "Bearer " + key.read_text().strip()
                seen.append(request.path)
                if request.path == "/redirect":
                    raise web.HTTPFound(other + "/leak")
                data = await request.json() if request.method == "POST" else {}
                assert "private-key" not in json.dumps(data)
                if request.path == "/get_load":
                    return web.json_response([dict(dp_rank=0, num_reqs=1, num_waiting_reqs=0, num_tokens=4, ts_tic=1)])
                if data.get("stream"):
                    frame = dict(output_ids=[42], text="answer", meta_info={"finish_reason": {"type": "stop"}})
                    return web.Response(body=b"data: " + json.dumps(frame).encode() + b"\n\n",
                                        content_type="text/event-stream")
                return web.json_response({"ok": True})

            app = web.Application()
            app.router.add_route("*", "/{route:.*}", endpoint)
            async with listen(app) as url:
                monkeypatch.setenv("REMORY_SGLANG_URL", url)
                monkeypatch.setenv("REMORY_SGLANG_API_KEY_FILE", str(key))
                assert not request_headers(other + "/generate")
                assert request_headers(url + "/groups/g/backends/b/generate")["Authorization"].endswith("first-private-key")

                def exercise():
                    c = Client(url)
                    c.health()
                    assert c.server_info()["ok"]
                    _read_health(url + "/health", 1)
                    assert _read_server_info(url + "/get_server_info", 1)["ok"]
                    for route in ("/generate", "/remory/memory", "/remory/memory/lease", "/release_memory_handle"):
                        assert c._post(route, {})["ok"]
                    c._abort_request("ours")
                    key.write_text("rotated-private-key")
                    c.configure_generation_streaming(artifact_dir=tmp_path)
                    assert c._post("/generate", dict(rid="stream", input_ids=[1], sampling_params={"max_new_tokens": 2}))["output_ids"] == [42]
                    assert c._stream_scheduler_load(1)[0][2] == 4
                    with pytest.raises(urllib.error.HTTPError) as exc:
                        open_request(url + "/redirect", timeout=1)
                    assert exc.value.code == 302
                await asyncio.to_thread(exercise)
                assert not leaks
                assert {"/generate", "/abort_request", "/get_load", "/health", "/remory/memory/lease", "/release_memory_handle"} <= set(seen)
    asyncio.run(check())


def test_credentials_require_origin_and_reject_header_injection(monkeypatch):
    monkeypatch.delenv("REMORY_SGLANG_URL", raising=False)
    monkeypatch.delenv("REMORY_SGLANG_API_KEY_FILE", raising=False)
    monkeypatch.setenv("REMORY_SGLANG_API_KEY", "key")
    with pytest.raises(ValueError, match="scope"):
        request_headers("https://remote.example/generate")
    with pytest.raises(ValueError, match="invalid"):
        request_headers("https://remote.example", api_key="bad\nkey")


def test_local_pool_to_authenticated_remote_gateway_pins_groups_cleanup_and_abort(tmp_path):
    async def check():
        backend = FakeServer()
        async def load(request):
            return web.json_response([dict(num_reqs=backend.active, num_waiting_reqs=0, num_tokens=7)])
        # Override the handler on a fresh fake application to include token progress.
        native = web.Application()
        native.router.add_get("/get_server_info", backend.info)
        native.router.add_get("/get_load", load)
        native.router.add_post("/abort_request", backend.abort)
        for path in ("/generate", "/remory/memory", "/remory/memory/lease", "/release_memory_handle"):
            native.router.add_post(path, backend.generate)
        key = tmp_path / "remote.key"
        key.write_text("remote-gateway-key")
        async with listen(native) as url:
            remote = create_app(config(tmp_path, [url]), api_key=key.read_text())
            async with listen(remote) as public:
                cfg = config(tmp_path, [public + "/backends/gpu0"], max_pending=1)
                cfg["backends"][0]["api_key_file"] = str(key)
                local = create_app(cfg)
                async with listen(local) as pool, aiohttp.ClientSession() as session:
                    assert local[APP_KEY].backends[0].healthy
                    r = await session.post(public + "/backends/gpu0/generate", json={})
                    assert r.status == 401
                    r = await session.post(pool + "/groups/project/backends/gpu0/remory/memory", json={"payload": {"operation": "save"}})
                    assert r.status == 200
                    assert backend.calls[-1][2]["X-Queue-Group"] == "project"
                    assert "Authorization" not in backend.calls[-1][2]
                    response = await session.post(pool + "/groups/project/backends/gpu0/generate",
                        json={"stream": True, "rid": "owned-remote"})
                    assert b"first" in await response.content.readline()
                    queued = asyncio.create_task(session.post(pool + "/groups/project/backends/gpu0/generate",
                        json={"rid": "queued-remote"}))
                    await until(lambda: local[APP_KEY].queue.pending == 1)
                    assert local[APP_KEY].uploading == 2  # full handler/admission limit
                    # Cleanup must pass both full queues without waiting for generation.
                    for path in ("/remory/memory/lease", "/release_memory_handle"):
                        r = await session.post(pool + "/groups/project/backends/gpu0" + path, json={})
                        assert r.status == 200
                    queued.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await queued
                    await until(lambda: local[APP_KEY].queue.pending == 0)
                    response.close()
                    await until(lambda: bool(backend.aborted))
                    assert backend.aborted == [{"rid": "owned-remote", "abort_all": False}]
                    await until(lambda: not remote[APP_KEY].native_requests and not local[APP_KEY].native_requests)
                    backend.gate.set()
    asyncio.run(check())


def test_portable_identity_ignores_location_and_rejects_different_artifacts(tmp_path):
    identities = []
    for name in ("host-a", "host-b"):
        target = tmp_path / name
        target.mkdir()
        (target / "config.json").write_text(json.dumps(dict(model_type="qwen3_5", hidden_size=5120, num_hidden_layers=64)))
        (target / "tokenizer.json").write_text("tokenizer")
        checkpoint = target / "compressor"
        checkpoint.mkdir()
        (checkpoint / "manifest.json").write_text(json.dumps({"compressor_sha256": "a" * 64}))
        cfg = dict(target_model=str(target), summary_contract={"tokenizer_files": {
            "tokenizer.json": hashlib.sha256(b"tokenizer").hexdigest()}}, compressor={"geometry": "fixed"})
        identities.append(artifact_identity(checkpoint, cfg))
    assert identities[0] == identities[1]
    assert check_server_identity({FIELD: identities[0], "model_path": "/remote/model"}, checkpoint, cfg)
    for field in ("compressor_sha256", "contract_sha256", "target_metadata_sha256"):
        with pytest.raises(RuntimeError, match="differs"):
            check_server_identity({FIELD: {**identities[0], field: "b" * 64}}, checkpoint, cfg)
    assert not check_server_identity({}, checkpoint, cfg)  # legacy path checks remain required
    (target / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="tokenizer identity"):
        artifact_identity(checkpoint, cfg)


def test_gateway_legacy_mapping_matches_relocated_new_server(tmp_path):
    gw = Gateway(config(tmp_path, ["http://localhost:1", "http://localhost:2"]))
    identity = dict(schema=SCHEMA, compressor_sha256="a"*64, contract_sha256="b"*64, target_metadata_sha256="c"*64)
    factory = "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook"
    gw.local_contracts["gpu0"] = dict(checkpoint="/local/serving", target="/local/target", identity=identity)
    def info(checkpoint, target):
        return dict(model_path=target, dtype="bfloat16", quantization=None, kv_cache_dtype="auto",
            forward_hooks=[dict(hook_factory=factory, config=dict(checkpoint=checkpoint, residual_serving_abi="same"))])
    local = info("/local/serving", "/local/target")
    remote = {**info("/remote/serving", "/remote/target"), FIELD: identity}
    assert gw.model_identity(gw.backends[0], local) == gw.model_identity(gw.backends[1], remote)
    with pytest.raises(ValueError, match="paths differ"):
        gw.model_identity(gw.backends[0], info("/wrong/checkpoint", "/local/target"))


def test_prepare_pool_config_keeps_source_and_uses_key_file(tmp_path):
    spec = importlib.util.spec_from_file_location("add_remote", ROOT / "tools/add_remote_backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config(tmp_path, ["http://127.0.0.1:1"])
    source = tmp_path / "gateway.json"
    before = source.read_bytes()
    checkpoint = tmp_path / "serving"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"target_model":"/local/model"}')
    kwargs = dict(key_file=tmp_path / "not-created-yet.key", checkpoint=checkpoint, name="h20", model="public", capacity=2)
    value = module.build_config(source, url="https://remote.example/backends/h20-tp4/v1", **kwargs)
    assert value["backends"][-1]["url"] == "https://remote.example/backends/h20-tp4"
    assert value["backends"][0]["local_checkpoint"] == str(checkpoint)
    assert value["backends"][-1]["supports_local_encoder_versions"] is False
    assert value["backends"][-1]["stateless_only"] is False
    assert "api_key" not in value["backends"][-1] and source.read_bytes() == before
    stateless = module.build_config(source, url="http://remote.example/backends/h20/v1",
                                   allow_http=True, stateless_only=True, **kwargs)
    assert stateless["backends"][-1]["stateless_only"] is True
    assert stateless["backends"][-1]["url"] == "http://remote.example/backends/h20"
    pinned = module.build_config(source, url="https://remote.example/backends/h20",
                                 memory_pinned_only=True, **kwargs)
    assert pinned["backends"][-1]["memory_pinned_only"] is True
    assert not pinned["backends"][-1].get("allow_tensor_recover", False)
    assert "verified_artifact_pair" not in pinned["backends"][-1]
    policy = "h20_online250_verified_tensor_pair_20261109_v1"
    verified = module.build_config(source, url="https://remote.example/backends/h20",
                                   memory_pinned_only=True, verified_artifact_pair=policy, **kwargs)
    assert verified["backends"][-1]["verified_artifact_pair"] == policy
    for options in ({"memory_pinned_only": True, "stateless_only": True},
                    {"verified_artifact_pair": policy}):
        with pytest.raises(ValueError):
            module.build_config(source, url="https://remote.example/backends/h20", **options, **kwargs)
    assert source.read_bytes() == before
    for bad in ("https://remote.example/v1", "http://remote.example/backends/a", "https://user:password@remote.example/backends/a"):
        with pytest.raises(ValueError):
            module.build_config(source, url=bad, **kwargs)


def test_prepare_runtime_verifies_bytes_and_relocates_only_target(tmp_path):
    from context_compression.qwen38_residual_serving_contract import canonical_sha256
    spec = importlib.util.spec_from_file_location("prepare_remote", ROOT / "tools/prepare_h20_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "remote-model"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(dict(model_type="qwen3_5", hidden_size=5120, num_hidden_layers=64)))
    original = json.loads((ROOT / "handoff/h20/compressor-metadata/config.json").read_text())
    for name in original["summary_contract"]["tokenizer_files"]:
        (target / name).write_text(name)
        original["summary_contract"]["tokenizer_files"][name] = hashlib.sha256(name.encode()).hexdigest()
    original["target_model"] = "/original/target/on/another/host"
    manifest = json.loads((ROOT / "handoff/h20/compressor-metadata/manifest.json").read_text())
    manifest.update(compressor_sha256=hashlib.sha256(b"fixture weights").hexdigest(), config_sha256=canonical_sha256(original))
    (source / "config.json").write_text(json.dumps(original))
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "compressor.pt").write_bytes(b"fixture weights")
    pins = dict(source_config_sha256=manifest["config_sha256"], target={"revision": "fixture"},
        artifact_identity=artifact_identity(source, {**original, "target_model": str(target)}))
    prefix = tmp_path / "stack"
    for name in ("server/bin/python", "sglang/python/sglang/launch_server.py"):
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    destination = tmp_path / "deployment"
    state_file = module.prepare(checkpoint=source, target=target, prefix=prefix, destination=destination, pins=pins)
    state = json.loads(state_file.read_text())
    new_config = json.loads((destination / "serving/config.json").read_text())
    assert new_config == {**original, "target_model": str(target)}
    assert json.loads((source / "config.json").read_text()) == original
    assert (destination / "serving/compressor.pt").resolve() == source / "compressor.pt"
    assert state["server"]["tp"] == 4 and state["target"] == str(target)
    (source / "compressor.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.prepare(checkpoint=source, target=target, prefix=prefix, destination=tmp_path / "wrong", pins=pins)
    assert not (tmp_path / "wrong").exists()
