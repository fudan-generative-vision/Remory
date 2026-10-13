"""Memory routes for SGLang and a small Transformers reference server."""
import logging
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .engine import Remory


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompactRequest(Request):
    prefix_ids: list[StrictInt]
    history_ids: list[StrictInt]
    summary_ids: list[StrictInt]
    previous: str | None = None
    branches: list[str] = Field(default_factory=list, max_length=16)


class MemoryReference(Request):
    handle: str


class NativeGenerateRequest(Request):
    input_ids: list[StrictInt]
    sampling_params: dict = Field(default_factory=dict)
    remory: MemoryReference | None = None


def create_app(engine: Remory) -> FastAPI:
    app = FastAPI(title="Remory", version="0.3.0")

    def session(x_remory_session: Annotated[str | None, Header()] = None):
        try:
            engine.store._key(x_remory_session)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return x_remory_session

    def call(fn, **kwargs):
        try:
            return fn(**kwargs)
        except KeyError as error:
            raise HTTPException(404, "memory not found for this session") from error
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except (RuntimeError, httpx.HTTPError) as error:
            logging.getLogger(__name__).exception("Residual backend failed")
            raise HTTPException(502, "residual backend failed; retain original history") from error

    @app.get("/health")
    def health():
        return {"status": "ok", "schema": "remory.v1"}

    @app.get("/v1/remory/info")
    def info():
        config = getattr(engine, "model_config", None)
        if config is None:
            raise HTTPException(503, "model metadata is unavailable")
        return {"schema": "remory.server.v1", "config": config,
                "identity": engine.contract.identity,
                "context_limit": engine.backend.context_limit}

    @app.post("/v1/compact")
    def compact(request: CompactRequest, owner: str = Depends(session)):
        args = request.model_dump()
        if not args["branches"]:
            args.pop("branches")
        memory = call(engine.compact, owner=owner, **args)
        return {"handle": memory.handle, "summary_ids": list(memory.summary_ids),
                "receipt": memory.receipt}

    @app.post("/generate")
    def native_generate(request: NativeGenerateRequest, owner: str = Depends(session)):
        params = dict(request.sampling_params)
        count = params.pop("max_new_tokens", 1024)
        result = call(engine.generate, owner=owner, max_new_tokens=count, sampling=params,
            handle=request.remory.handle if request.remory else None,
            input_ids=None if request.remory else request.input_ids,
            continuation_ids=request.input_ids if request.remory else ())
        return {"text": result.text, "output_ids": result.output_ids,
                "meta_info": {**result.usage, "finish_reason": result.finish_reason}}

    @app.delete("/v1/memories/{handle}")
    def delete(handle: str, owner: str = Depends(session)):
        if not call(engine.store.delete, owner=owner, handle=handle):
            raise HTTPException(404, "memory not found for this session")
        return {"deleted": True}

    return app
