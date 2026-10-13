"""Native SGLang HTTP serving with Remory compaction on the same model."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .engine import Remory
from .server import create_app
from .store import MemoryStore


def restore_generation(engine, obj, owner):
    """Expand a saved checkpoint before SGLang tokenizes and schedules the request."""
    ref = obj.remory
    if not isinstance(ref, dict) or set(ref) != {"handle"} or not isinstance(ref["handle"], str):
        raise ValueError("remory must contain a memory handle")
    for field in ("text", "input_embeds", "image_data", "audio_data", "video_data", "session_params", "lora_path"):
        if getattr(obj, field, None) is not None:
            raise ValueError(f"{field} cannot be combined with a Remory handle; use continuation input_ids")
    params = obj.sampling_params or {}
    if not isinstance(params, dict):
        raise ValueError("Remory generation requires one sampling_params object")
    if "custom_params" in params or obj.custom_logit_processor is not None:
        raise ValueError("Remory handles supply their own embedding overrides")
    count = params.get("max_new_tokens", 128)
    if type(count) is not int or count < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    source = engine._prepare(owner=owner, handle=ref["handle"], continuation_ids=obj.input_ids or [])
    params = {"temperature": 0.0, **params, "max_new_tokens": count}
    params["stop_token_ids"] = list(dict.fromkeys([
        *(params.get("stop_token_ids") or []), engine.contract.placeholder_id]))
    body = engine.backend.build_request(source, params,
        {"mode": "recover", **engine.backend._memory(source)})
    obj.input_ids = body["input_ids"]
    obj.sampling_params = body["sampling_params"]
    obj.custom_logit_processor = body.get("custom_logit_processor")
    if "cache_salt" in body:
        obj.rid = body["rid"]
        obj.cache_salt = body["cache_salt"]
    obj.remory = None


def extend_app(http_server, engine):
    from sglang.srt.managers.io_struct import GenerateReqInput

    @dataclass
    class GenerateWithMemory(GenerateReqInput):
        remory: dict | None = None

    app = http_server.app
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/generate"]

    @app.api_route("/generate", methods=["POST", "PUT"])
    async def generate(obj: GenerateWithMemory, request: Request):
        defaults = getattr(engine.backend, "generation_defaults", {})
        if defaults and (obj.sampling_params is None or isinstance(obj.sampling_params, dict)):
            obj.sampling_params = {**defaults, **(obj.sampling_params or {})}
        if obj.remory is not None:
            try:
                await run_in_threadpool(restore_generation, engine, obj,
                                       request.headers.get("x-remory-session"))
            except KeyError as exc:
                raise HTTPException(404, "memory not found for this session") from exc
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return await http_server.generate_request(obj, request)

    memory_app = create_app(engine)
    app.router.routes.extend(r for r in memory_app.routes if getattr(r, "path", "") in {
        "/v1/compact", "/v1/memories/{handle}", "/v1/remory/info"})
    app.openapi_schema = None


def launch(args):
    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    from .backends.sglang_http import SGLangHTTPBackend
    from .backends.sglang_setup import verify_installation
    from .backends.sglang_worker import engine_settings
    from .models.load import is_glm, resolve_actor, resolve_checkpoint

    verify_installation()
    checkpoint, config = resolve_checkpoint(args.checkpoint)
    actor = resolve_actor(config, args.actor)
    model_config = json.loads((actor / "config.json").read_text())
    limit = args.context_limit if args.context_limit is not None else model_config.get(
        "text_config", model_config)["max_position_embeddings"]
    if type(limit) is not int or limit < 2048:
        raise ValueError("context limit must be at least 2048")
    if not 0 < args.memory_fraction < 1 or args.gpu < 0:
        raise ValueError("invalid GPU index or memory fraction")
    settings = dict(actor=str(actor), checkpoint=str(checkpoint), context_limit=limit,
                    memory_fraction=args.memory_fraction, gpu=args.gpu, tp_size=args.tp_size,
                    model_family="glm" if is_glm(config) else "qwen")
    host = "127.0.0.1" if args.host == "0.0.0.0" else ("::1" if args.host == "::" else args.host)
    if ":" in host:
        host = f"[{host}]"
    if is_glm(config):
        from .backends.glm_http import GlmSGLangHTTPBackend
        from .glm_engine import GlmRemory
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        cache = Path(args.store).resolve().parent / "glm-memory"
        os.environ.update(GLM53_COMPRESSOR_CHECKPOINT=str(checkpoint),
            GLM53_MEMORY_CACHE_DIR=str(cache), GLM53_SERVER_ID=f"remory-{args.port}")
        settings["weights_sha256"] = manifest["safetensors_sha256"]
        backend = GlmSGLangHTTPBackend(f"http://{host}:{args.port}", config, actor,
            context_limit=limit, cache_dir=cache, weights_sha256=manifest["safetensors_sha256"])
        engine = GlmRemory(backend, MemoryStore(args.store))
    else:
        backend = SGLangHTTPBackend(f"http://{host}:{args.port}", config, actor, context_limit=limit)
        engine = Remory(backend, MemoryStore(args.store))
    engine.model_config = config
    extend_app(http_server, engine)
    server_args = ServerArgs(**engine_settings(settings), host=args.host, port=args.port)
    print(f"[Remory] Starting SGLang at http://{host}:{args.port}; context window: {limit:,} tokens.", flush=True)
    try:
        http_server.launch_server(server_args)
    finally:
        backend.close()
        kill_process_tree(parent_pid=os.getpid(), include_parent=False)
