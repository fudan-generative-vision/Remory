"""Fair request queue for existing SGLang servers; never owns models or tasks.

The OpenAI and native SGLang wire protocols pass through to compatible backends.
Generation POSTs are never retried: a lost response must not duplicate work.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
import uuid
import sys
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

from .request_queue import AdmissionQueue, Backend, QueueFull

LOG = logging.getLogger(__name__)
APP_KEY = web.AppKey("gateway", object)
INFERENCE_PATHS = {
    "/generate", "/encode", "/classify", "/v1/chat/completions",
    "/v1/completions", "/v1/responses", "/v1/embeddings", "/v1/score",
    "/v1/rerank", "/rerank", "/specforge/memory/chain", "/remory/memory",
}
PINNED_PATHS = {"/open_session", "/close_session", "/release_memory_handle", "/remory/memory/lease"}
INFO_PATHS = {"/get_server_info", "/server_info", "/get_model_info", "/model_info"}
NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
BACKEND_ROUTE = re.compile(r"(?:/groups/[A-Za-z0-9_.:-]{1,128})?(?:/backends/[A-Za-z0-9_.:-]{1,128})?")


def error(status, message, code="gateway_error"):
    return web.json_response({"error": {"message": message, "type": code, "code": code}}, status=status)


def positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite")
    return value


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("backends"), list) or not config["backends"]:
        raise ValueError("config needs a nonempty backends array")
    names, urls = set(), set()
    for backend in config["backends"]:
        name, model, url = backend["name"], backend["model"], backend["url"].rstrip("/")
        parsed = urlsplit(url)
        if (not NAME.fullmatch(name) or name in names or url in urls
                or not isinstance(model, str) or not model or len(model) > 256
                or parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or not BACKEND_ROUTE.fullmatch(parsed.path)
                or parsed.query or parsed.fragment):
            raise ValueError("invalid or duplicate backend name/URL/model")
        if backend.get("api_key_env") and backend.get("api_key_file"):
            raise ValueError("use either api_key_env or api_key_file for an upstream")
        if "api_key" in backend:
            raise ValueError("store upstream keys in api_key_file or api_key_env, not gateway JSON")
        for field in ("api_key_file", "local_checkpoint"):
            if field in backend and (not isinstance(backend[field], str) or not Path(backend[field]).is_absolute()):
                raise ValueError(field + " must be an absolute path")
        positive(backend["max_inflight"], "max_inflight", integer=True)
        if not isinstance(backend.get("probe_load", True), bool):
            raise ValueError("probe_load must be boolean")
        if not isinstance(backend.get("supports_local_encoder_versions", True), bool):
            raise ValueError("supports_local_encoder_versions must be boolean")
        if not isinstance(backend.get("stateless_only", False), bool):
            raise ValueError("stateless_only must be boolean")
        if backend.get("stateless_only") and not backend.get("local_checkpoint"):
            raise ValueError("stateless_only requires a local_checkpoint base identity anchor")
        backend["url"] = url
        names.add(name)
        urls.add(url)
    # Residual /generate calls include summary hidden states as JSON floats;
    # a roughly 2K-token summary already exceeds the old 64 MiB limit.
    for name, default in (("max_pending", 256), ("max_body_bytes", 1024 * 1024 * 1024)):
        config[name] = positive(config.get(name, default), name, integer=True)
    for name, default in (("queue_timeout", 1800), ("request_timeout", 3600), ("probe_interval", 1)):
        config[name] = positive(config.get(name, default), name)
    return config


def scrub_metadata(value):
    if isinstance(value, dict):
        return {k: scrub_metadata(v) for k, v in value.items()
                if k not in {"api_key", "admin_api_key", "authorization"}}
    if isinstance(value, list):
        return [scrub_metadata(v) for v in value]
    return value


def request_units(path, data):
    """Charge batched prompts and n completions against request capacity."""
    prompts = 1
    if path in {"/generate", "/encode", "/classify", "/v1/completions", "/v1/embeddings"}:
        for key in ("text", "prompt", "input_ids", "input"):
            value = data.get(key)
            if isinstance(value, list) and value and isinstance(value[0], (list, str)):
                prompts = len(value)
                break
    sampling = data.get("sampling_params", {})
    if isinstance(sampling, list):
        if len(sampling) != prompts:
            raise ValueError("batched sampling parameters must match the prompt count")
        return sum(positive(row.get("n", 1), "n", integer=True) for row in sampling)
    n = data.get("n", sampling.get("n", 1) if isinstance(sampling, dict) else 1)
    return prompts * positive(n, "n", integer=True)


def needs_local_encoder_versions(data):
    """Recognize native custom params, including per-prompt sampling params."""
    sampling = data.get("sampling_params", {})
    rows = sampling if isinstance(sampling, list) else [sampling]
    return any(isinstance(row, dict) and isinstance(row.get("custom_params"), dict)
               and "remory_encoder_version" in row["custom_params"]
               for row in [data, *rows])


def needs_full_backend(path, data):
    # Only ordinary generation has been admitted for replicas with an
    # incompatible compressor. /encode and all memory/control routes stay local.
    if path not in {"/generate", "/v1/chat/completions", "/v1/completions", "/v1/responses"}:
        return True
    if needs_pin(path, data) or (path == "/v1/responses" and data.get("store", True)):
        return True

    def contains_memory(value):
        if isinstance(value, dict):
            # CUSTOM_PARAM_KEY in context_compression.sglang_compressor_hook is
            # specforge_context_compressor. Also reject tensor inputs and legacy
            # config wrappers, recursively including batched sampling params.
            return any(k in {"specforge_context_compressor", "specforge_config",
                             "input_embeds", "input_embeddings", "return_hidden_states",
                             "hidden_states", "summary_features", "previous_response_id"}
                       or any(part in k.lower() for part in ("memory", "handle", "session", "tensor", "remory", "residual"))
                       or contains_memory(v) for k, v in value.items())
        return isinstance(value, list) and any(contains_memory(v) for v in value)

    return contains_memory(data)


def memory_capability(capability):
    return any(part in capability for part in ("remory", "residual", "memory", "compress", "hidden_state", "handle", "session"))


def needs_pin(path, data):
    # Native custom_params holds handles below sampling_params, including
    # batched requests. Check before dispatch; never balance an owned tensor.
    def contains_handle(value):
        if isinstance(value, dict):
            return any(k in {"memory_handle", "memory_handle_publish", "summary_handle"}
                       or contains_handle(v) for k, v in value.items())
        return isinstance(value, list) and any(contains_handle(v) for v in value)
    if path == "/remory/memory" or contains_handle(data):
        return True
    return (path in PINNED_PATHS or path == "/specforge/memory/chain"
            or path.startswith("/v1/responses/")
            or any(data.get(k) is not None for k in (
                "session_params", "session_id", "previous_response_id", "memory_handle",
                "memory_handles", "input_memory_handle", "input_memory_handles",
                "soft_memory_handle"))
            or data.get("return_memory_handle") is True
            or data.get("store") is True or data.get("background") is True)


class Gateway:
    def __init__(self, config, *, api_key=None):
        self.config, self.api_key = config, api_key
        self.backends = [Backend(row["name"], row["url"], row["model"], row["max_inflight"],
                                 supports_local_encoder_versions=row.get("supports_local_encoder_versions", True),
                                 stateless_only=row.get("stateless_only", False))
                         for row in config["backends"]]
        self.settings = {row["name"]: row for row in config["backends"]}
        self.queue = AdmissionQueue(self.backends, max_pending=config["max_pending"])
        self.model_identities = {}
        self.session = None
        self.poller = None
        self.completed = 0
        self.failed = 0
        self.uploading = 0
        self.native_requests = {}
        self.local_contracts = {}
        # Optional, explicitly configured bridge for existing legacy engines
        # which expose paths instead of the new portable artifact identity.
        for row in config["backends"]:
            if row.get("local_checkpoint"):
                workspace = str(Path(__file__).parent / "payload/workspace")
                if workspace not in sys.path:
                    sys.path.insert(0, workspace)
                from context_compression.remory_identity import artifact_identity
                from context_compression.qwen38_residual_serving_contract import load_residual_serving_contract
                checkpoint = Path(row["local_checkpoint"]).resolve()
                local = load_residual_serving_contract(checkpoint)
                self.local_contracts[row["name"]] = dict(checkpoint=str(checkpoint),
                    target=local["target_model"], identity=artifact_identity(checkpoint, local))

    def headers(self, backend):
        filename = self.settings[backend.name].get("api_key_file")
        if filename:
            value = Path(filename).read_text().strip()
            if not value or "\n" in value or "\r" in value:
                raise ValueError("invalid upstream API key file")
            return {"Authorization": f"Bearer {value}"}
        variable = self.settings[backend.name].get("api_key_env")
        if variable:
            value = os.environ.get(variable)
            if not value or "\n" in value or "\r" in value:
                raise ValueError(f"missing upstream credential environment variable {variable}")
            return {"Authorization": f"Bearer {value}"}
        return {}

    def model_identity(self, backend, info):
        artifact = info.get("remory_artifact_identity")
        local = self.local_contracts.get(backend.name)
        hooks = info.get("forward_hooks") or []
        reference = None
        expected_factory = "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook"
        if backend.stateless_only and (artifact is None or local is None):
            raise ValueError("stateless_only requires portable identity and a local checkpoint anchor")
        if artifact is None and local is not None:
            if (len(hooks) != 1 or hooks[0].get("hook_factory") != expected_factory
                    or str(Path(hooks[0].get("config", {}).get("checkpoint", "")).resolve()) != local["checkpoint"]
                    or info.get("model_path") != local["target"]):
                raise ValueError("legacy backend paths differ from its configured local checkpoint")
            artifact = local["identity"]
        if artifact is not None:
            if (not isinstance(artifact, dict) or set(artifact) != {
                    "schema", "compressor_sha256", "contract_sha256", "target_metadata_sha256"}
                    or artifact["schema"] != "remory_artifact_identity_v1"
                    or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                           for k, v in artifact.items() if k != "schema")
                    or len(hooks) != 1 or hooks[0].get("hook_factory") != expected_factory):
                raise ValueError("invalid portable residual identity")
            if backend.stateless_only:
                if artifact["target_metadata_sha256"] != local["identity"]["target_metadata_sha256"]:
                    raise ValueError("stateless_only target metadata differs from the local checkpoint")
                reference = next((b for b in self.backends if b.model == backend.model
                                  and b.healthy and not b.stateless_only), None)
                if reference is None:
                    raise ValueError("stateless_only needs a healthy full backend base identity anchor")
                for key in ("dtype", "quantization", "kv_cache_dtype"):
                    if key not in info or key not in reference.info or info[key] != reference.info[key]:
                        raise ValueError("stateless_only base " + key + " differs from the full backend")
                # INTERNAL base-model grouping only: the remote compressor and
                # its contract are incompatible and never used on this route.
                # Admission and queue pumping exclude all memory/state requests.
                # backend.info retains the remote's real portable identity.
                artifact = local["identity"]
            elif local is not None and artifact != local["identity"]:
                raise ValueError("backend residual identity differs from its configured checkpoint")
            return dict(artifact=artifact, dtype=info.get("dtype"), quantization=info.get("quantization"),
                        kv_cache_dtype=info.get("kv_cache_dtype"),
                        residual_serving_abi=(reference.info.get("forward_hooks", [{}])[0]
                            if reference is not None else hooks[0]).get("config", {}).get("residual_serving_abi"))
        return {k: info.get(k) for k in (
            "model_path", "revision", "dtype", "quantization", "kv_cache_dtype", "forward_hooks")}

    def upstream_headers(self, backend, group):
        match = re.match(r"/groups/([^/]+)(?:/|$)", urlsplit(backend.url).path)
        return {"X-Queue-Group": match[1] if match else group, **self.headers(backend)}

    async def read_json(self, backend, path):
        async with self.session.get(backend.url + path,
                                    headers={"Accept-Encoding": "identity", **self.headers(backend)},
                                    timeout=ClientTimeout(total=5), allow_redirects=False) as response:
            response.raise_for_status()
            if response.status != 200:
                raise ValueError("backend readiness returned a redirect/non-200 response")
            content = bytearray()
            async for part in response.content.iter_chunked(65536):
                content.extend(part)
                if len(content) > 4 * 1024 * 1024:
                    raise ValueError("backend metadata exceeds limit")
            return json.loads(content)

    async def probe(self, backend):
        sequence = backend.starts
        try:
            info = await self.read_json(backend, "/get_server_info")
            if type(info.get("max_running_requests")) is not int or info["max_running_requests"] < backend.capacity:
                raise ValueError("configured admission exceeds SGLang request capacity")
            wanted = self.settings[backend.name].get("model_path")
            if wanted is not None and info.get("model_path") != wanted:
                raise ValueError("backend model_path differs from configuration")
            identity = json.dumps(self.model_identity(backend, info), sort_keys=True)
            fingerprint = hashlib.sha256(identity.encode()).hexdigest()
            if backend.identity is not None and fingerprint != backend.identity:
                raise ValueError("backend model identity changed; restart the gateway after draining")
            previous = self.model_identities.setdefault(backend.model, fingerprint)
            if fingerprint != previous:
                raise ValueError("backends for one public model have different model/checkpoint identities")
            load = 0
            load_info = {}
            if self.settings[backend.name].get("probe_load", True):
                try:
                    report = await self.read_json(backend, "/v1/loads")
                    values = report["loads"]
                    running = "num_running_reqs"
                except ClientError as exc:
                    if getattr(exc, "status", None) != 404:
                        raise
                    values = await self.read_json(backend, "/get_load")
                    running = "num_reqs"
                if (not isinstance(values, list) or not values
                        or any(type(v.get(running)) is not int or v[running] < 0
                               or type(v.get("num_waiting_reqs")) is not int or v["num_waiting_reqs"] < 0 for v in values)):
                    raise ValueError("backend does not provide valid load counters")
                # Legacy /get_load.num_reqs already includes waiting requests.
                load = sum(v[running] + (v["num_waiting_reqs"] if running == "num_running_reqs" else 0)
                           for v in values)
                tokens = [v.get("num_used_tokens", v.get("num_tokens")) for v in values]
                if all(type(value) is int and value >= 0 for value in tokens):
                    load_info = {"num_reqs": load,
                                 "num_waiting_reqs": sum(v["num_waiting_reqs"] for v in values),
                                 "num_tokens": sum(tokens), "ts_tic": time.monotonic()}
            backend.info = scrub_metadata(info)
            backend.native_model = str(info.get("served_model_name") or backend.model)
            backend.identity = fingerprint
            backend.observed_load, backend.observed_starts = load, sequence
            backend.load_info = load_info
            backend.healthy, backend.error = True, None
        except (ClientError, TimeoutError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            backend.healthy = False
            backend.error = type(exc).__name__ + ": " + str(exc)
        self.queue.pump()

    async def start(self, app):
        # Large residual JSON parsing/serialization can outlast an upstream
        # idle keep-alive timeout. Open a fresh connection instead of reusing
        # a socket whose close notification is still waiting on this loop.
        # POSTs remain single-dispatch; transport failures are never retried.
        self.session = ClientSession(trust_env=False, auto_decompress=False,
            connector=TCPConnector(limit=0, force_close=True), timeout=ClientTimeout(total=None, sock_connect=10,
                sock_read=self.config["request_timeout"]))
        await asyncio.gather(*(self.probe(b) for b in self.backends))
        async def poll():
            while True:
                await asyncio.sleep(self.config["probe_interval"])
                await asyncio.gather(*(self.probe(b) for b in self.backends))
        self.poller = asyncio.create_task(poll())

    async def shutdown(self, app):
        self.queue.close()

    async def cleanup(self, app):
        if self.poller:
            self.poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.poller
        if self.session:
            await self.session.close()

    def authorized(self, request):
        return self.api_key is None or hmac.compare_digest(
            request.headers.get("Authorization", "").encode(), ("Bearer " + self.api_key).encode())

    def status(self):
        return {"status": "ready" if any(b.healthy for b in self.backends) else "unavailable",
                "max_body_bytes": self.config["max_body_bytes"],
                "inflight_handlers": self.uploading,
                "pending": self.queue.pending, "groups": {k: len(v) for k, v in self.queue.groups.items()},
                "admitted": self.queue.admitted, "completed": self.completed, "failed": self.failed,
                "backends": [{"name": b.name, "model": b.model, "healthy": b.healthy,
                              "active": b.active, "observed_load": b.observed_load,
                              "admission_load": b.load, "capacity": b.capacity, "error": b.error}
                             for b in self.backends]}

    async def handle(self, request):
        if request.path == "/health" and request.method == "GET":
            ready = any(b.healthy for b in self.backends)
            return web.json_response({"status": "ready" if ready else "unavailable"}, status=200 if ready else 503)
        if not self.authorized(request):
            return error(401, "valid gateway Bearer token required", "authentication_error")
        path = request.path
        group = request.headers.get("X-Queue-Group")
        if path.startswith("/groups/"):
            parts = path.split("/", 3)
            if len(parts) != 4 or not NAME.fullmatch(parts[2]) or (group and group != parts[2]):
                return error(400, "invalid or conflicting queue group selection")
            group, path = parts[2], "/" + parts[3]
        group = group or "default"
        pinned = request.headers.get("X-SGLang-Backend")
        if path.startswith("/backends/"):
            parts = path.split("/", 3)
            if len(parts) != 4 or (pinned and pinned != parts[2]):
                return error(400, "invalid or conflicting backend selection")
            pinned, path = parts[2], "/" + parts[3]
        if pinned and pinned not in self.settings:
            return error(404, "unknown backend")
        if path == "/health" and request.method == "GET":
            ready = any(b.healthy and (not pinned or b.name == pinned) for b in self.backends)
            return web.json_response({"status": "ready" if ready else "unavailable"}, status=200 if ready else 503)
        if path == "/queue/status" and request.method == "GET":
            return web.json_response(self.status())
        if path == "/get_load" and request.method == "GET":
            choices = [b for b in self.backends if b.healthy and (not pinned or b.name == pinned)]
            if not choices or any(not b.load_info for b in choices):
                return error(503, "backend token progress counters are unavailable")
            # Native streaming clients distinguish queued peer progress from
            # a dead stream using token counters, not just HTTP readiness.
            result = {key: sum(b.load_info[key] for b in choices)
                      for key in ("num_reqs", "num_waiting_reqs", "num_tokens")}
            return web.json_response([{**result, "dp_rank": 0,
                                       "ts_tic": min(b.load_info["ts_tic"] for b in choices)}])
        if path == "/v1/models" and request.method == "GET":
            return web.json_response({"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "shared-sglang"}
                for model in sorted({b.model for b in self.backends if not pinned or b.name == pinned})]})
        if path in INFO_PATHS and request.method == "GET":
            choices = [b for b in self.backends if b.healthy and (not pinned or b.name == pinned)]
            if not choices:
                return error(503, "selected backend is unavailable")
            if not pinned and len({b.model for b in self.backends}) != 1:
                return error(400, "multiple models: select X-SGLang-Backend or /backends/NAME prefix")
            full_choices = [b for b in choices if not b.stateless_only]
            metadata_backend = (full_choices or choices)[0]
            if path in {"/get_server_info", "/server_info"}:
                info = dict(metadata_backend.info)
                # Native metadata consumers (e.g. BrowseComp's local judge)
                # must send the gateway model alias on their next OpenAI call.
                info["served_model_name"] = metadata_backend.model
                if metadata_backend.stateless_only:
                    info["stateless_only"] = True
                    info["request_capabilities"] = []
                    info.pop("specforge_gpu_memory_handle", None)
                # This capability is routable when at least one eligible
                # healthy backend shares the gateway's local registry. Other
                # capabilities still require support throughout the pool.
                version_capability = "remory_encoder_versions_v1"
                capabilities = set(info.get("request_capabilities", []))
                if not metadata_backend.can_route_local_encoder_versions:
                    capabilities.discard(version_capability)
                info["request_capabilities"] = sorted(capabilities)
                if not pinned:
                    for key in ("max_running_requests", "context_length", "max_total_num_tokens", "max_req_input_len"):
                        values = [b.info.get(key) for b in choices]
                        if all(type(value) is int for value in values):
                            info[key] = min(values)
                    common = set(choices[0].info.get("request_capabilities", []))
                    for b in choices[1:]:
                        common.intersection_update(b.info.get("request_capabilities", []))
                    # REMORY contracts/capabilities describe only compatible
                    # full backends; stateless replicas cannot execute them.
                    common = {c for c in common if not memory_capability(c)}
                    if full_choices:
                        full_common = set(full_choices[0].info.get("request_capabilities", []))
                        for b in full_choices[1:]:
                            full_common.intersection_update(b.info.get("request_capabilities", []))
                        common.update(c for c in full_common if memory_capability(c))
                    common.discard(version_capability)
                    if any(b.can_route_local_encoder_versions for b in choices):
                        common.add(version_capability)
                    # Server-owned handles/sessions require an explicit route;
                    # only request-local capabilities may be advertised here.
                    info["request_capabilities"] = sorted(c for c in common if "handle" not in c and "session" not in c and c != "remory_server_memory_v1")
                    info.pop("specforge_gpu_memory_handle", None)
                    info["shared_gateway"] = {"backends": [b.name for b in choices],
                                              "max_inflight": sum(b.capacity for b in choices),
                                              "stateless_only_backends": [b.name for b in choices if b.stateless_only],
                                              "memory_backends": [dict(name=b.name, load=b.load, capacity=b.capacity)
                                                  for b in full_choices if "remory_server_memory_v1" in b.info.get("request_capabilities", [])]}
                return web.json_response(info)
            try:
                return web.json_response(scrub_metadata(await self.read_json(metadata_backend, path)))
            except (ClientError, TimeoutError, ValueError):
                return error(503, "backend metadata is temporarily unavailable")
        if path == "/abort_request" and request.method == "POST":
            try:
                payload = await request.json()
                rid = payload.get("rid")
                if payload.get("abort_all") or not isinstance(rid, str):
                    return error(400, "only one gateway-owned native rid may be cancelled")
                owner = self.native_requests.get(rid)
                if owner is None or owner["group"] != group:
                    return error(404, "no active request with that rid in this queue group")
                owner["task"].cancel()
                return web.json_response({"aborted": True, "rid": rid})
            except (ValueError, TypeError, AttributeError):
                return error(400, "invalid cancellation request")
        response_control = bool(re.fullmatch(r"/v1/responses/[A-Za-z0-9_-]+(?:/cancel)?", path))
        if (path not in INFERENCE_PATHS | PINNED_PATHS and not response_control
                or request.method not in {"POST", "PUT", "GET"}
                or request.method == "GET" and not response_control):
            return error(404, "this gateway exposes inference routes, not server administration")
        limit = self.config["max_pending"] + sum(b.capacity for b in self.backends)
        if self.uploading >= limit and path not in PINNED_PATHS and not response_control:
            return error(429, "gateway request capacity is full", "queue_full")
        self.uploading += 1
        ticket = None
        response = None
        native_ids = []
        owned_rid = None
        try:
            raw = await request.read()
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            if data.get("background") is True:
                raise ValueError("background inference is not supported; keep the request/stream open")
            if path == "/v1/responses" and data.get("store", True) and not pinned:
                raise ValueError("stored Responses require a pinned backend; use store=false for stateless routing")
            if needs_pin(path, data) and not pinned:
                raise ValueError("server-side state requires X-SGLang-Backend or /backends/NAME prefix")
            models = {b.model for b in self.backends if not pinned or b.name == pinned}
            model = data.get("model")
            if model is None and len(models) == 1:
                model = next(iter(models))
            if not isinstance(model, str) or model not in models:
                return error(404, "select a model advertised by /v1/models", "model_not_found")
            if not NAME.fullmatch(group):
                raise ValueError("X-Queue-Group must be 1..128 letters, digits, _, ., : or -")
            units = request_units(path, data)
            requires_local_encoder_versions = needs_local_encoder_versions(data)
            requires_full_backend = needs_full_backend(path, data)
            if path == "/generate":
                rid = data.get("rid")
                if units > 1:
                    if rid is not None:
                        raise ValueError("omit rid for batched generation; cancel by disconnecting")
                else:
                    rid = rid or "gw-" + uuid.uuid4().hex
                    if not isinstance(rid, str) or not NAME.fullmatch(rid):
                        raise ValueError("invalid native rid")
                    if rid in self.native_requests:
                        return error(409, "native rid is already active")
                    self.native_requests[rid] = {"group": group, "task": asyncio.current_task()}
                    owned_rid = rid
                    data["rid"] = rid
            started = time.monotonic()
            if path in PINNED_PATHS or response_control:
                backend = next(b for b in self.backends if b.name == pinned)
                if requires_full_backend and backend.stateless_only:
                    raise ValueError("memory/state requests cannot use a stateless_only backend")
                if requires_local_encoder_versions and not backend.can_route_local_encoder_versions:
                    raise ValueError("selected backend does not support the local remory_encoder_version registry")
                if not backend.healthy:
                    return error(503, "selected backend is unavailable")
            else:
                ticket = self.queue.submit(group=group, model=model, pinned=pinned, units=units,
                                           requires_local_encoder_versions=requires_local_encoder_versions,
                                           requires_full_backend=requires_full_backend)
                backend = await asyncio.wait_for(asyncio.shield(ticket.future), self.config["queue_timeout"])
            waited = time.monotonic() - started
            # OpenAI public aliases map to the backend's own served model name.
            if path.startswith("/v1/") and "model" in data:
                data["model"] = backend.native_model
            if owned_rid:
                native_ids = [owned_rid]
            headers = {"Content-Type": "application/json", "Accept-Encoding": "identity",
                       **self.upstream_headers(backend, group)}
            # Preserve SSE bytes, status and content type. No retry/failover
            # occurs after dispatch, including before the first response byte.
            async with asyncio.timeout(self.config["request_timeout"]), self.session.request(request.method, backend.url + path,
                    params=request.query, data=json.dumps(data).encode(), headers=headers,
                    allow_redirects=False) as upstream:
                response = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "application/json"),
                    "X-SGLang-Backend": backend.name, "X-Queue-Wait-Seconds": f"{waited:.6f}",
                    "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                    **{k: upstream.headers[k] for k in ("Content-Encoding", "Retry-After") if k in upstream.headers},
                })
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                self.completed += 1
                native_ids = []
                return response
        except web.HTTPRequestEntityTooLarge:
            self.failed += 1
            return error(413, f"request body exceeds gateway max_body_bytes={self.config['max_body_bytes']}; "
                         "residual hidden-state payloads may require a larger configured limit",
                         "request_body_too_large")
        except QueueFull:
            return error(429, "request queue is full", "queue_full")
        except (ValueError, TypeError, KeyError) as exc:
            return error(400, str(exc), "invalid_request_error")
        except TimeoutError:
            self.failed += 1
            if response is not None and response.prepared:
                raise
            return error(504, "queue or upstream request timed out", "timeout")
        except (ClientError, ConnectionError) as exc:
            self.failed += 1
            LOG.warning("upstream transport failed: %s (errno=%s)", type(exc).__name__, getattr(exc, "errno", None))
            if ticket is not None and ticket.backend is not None:
                ticket.backend.healthy = False
            if response is not None and response.prepared:
                raise
            return error(502, "upstream transport failed; generation was not retried", "upstream_error")
        finally:
            if native_ids and ticket is not None and ticket.backend is not None:
                # Keep ownership until abort is sent, or its short deadline
                # expires. Do not block the event loop or abort another caller.
                async def abort_upstream():
                    async with self.session.post(ticket.backend.url + "/abort_request",
                            json={"rid": native_ids[0], "abort_all": False},
                            headers=self.upstream_headers(ticket.backend, group),
                            timeout=ClientTimeout(total=3), allow_redirects=False) as aborted:
                        await aborted.read()
                # Nested gateways can receive both a disconnect and an explicit
                # abort. A second cancellation must not interrupt cleanup that
                # the first cancellation already started.
                cleanup = asyncio.create_task(abort_upstream())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                    except (ClientError, TimeoutError, OSError, ValueError):
                        break
                with contextlib.suppress(ClientError, TimeoutError, asyncio.CancelledError, OSError, ValueError):
                    cleanup.result()
            if ticket is not None:
                self.queue.release(ticket)
            if owned_rid:
                self.native_requests.pop(owned_rid, None)
            self.uploading -= 1


def create_app(config, *, api_key=None):
    gateway = Gateway(config, api_key=api_key)
    app = web.Application(client_max_size=config["max_body_bytes"])
    app[APP_KEY] = gateway
    app.on_startup.append(gateway.start)
    app.on_shutdown.append(gateway.shutdown)
    app.on_cleanup.append(gateway.cleanup)
    app.router.add_route("*", "/{path:.*}", gateway.handle)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--check", action="store_true", help="validate configuration without contacting servers")
    args = parser.parse_args()
    config = load_config(args.config)
    api_key = args.api_key_file.read_text().strip() if args.api_key_file else None
    if api_key is not None and len(api_key) < 24:
        parser.error("gateway token must have at least 24 characters")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and api_key is None:
        parser.error("a non-loopback listener requires --api-key-file")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be in 1024..65535")
    if args.check:
        print(json.dumps({"status": "ok", "backends": len(config["backends"]),
                          "capacity": sum(b["max_inflight"] for b in config["backends"])}))
        return
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(config, api_key=api_key), host=args.host, port=args.port,
                handler_cancellation=True, access_log=None, shutdown_timeout=60)


if __name__ == "__main__":
    main()
