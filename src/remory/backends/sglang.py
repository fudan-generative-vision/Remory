"""Managed SGLang inference. One owned worker, one public Remory endpoint."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import uuid

import numpy as np

from ..types import Contract, Generation, token_ids


class SGLangBackend:
    def __init__(self, config, checkpoint, actor, *, context_limit=None,
                 memory_fraction=0.8, gpu=0, tp_size=1, timeout=600, startup_timeout=900):
        if config.get("schema") == "remory_glm53_local_evaluation_v1":
            raise ValueError("GLM uses the native HTTP runtime; start python -m remory.launch_server")
        from .sglang_worker import run_worker
        if context_limit is None:
            from transformers import AutoConfig
            actor_config = AutoConfig.from_pretrained(
                actor, local_files_only=True, trust_remote_code=False)
            context_limit = actor_config.get_text_config().max_position_embeddings
        if type(context_limit) is not int or context_limit < 2048:
            raise ValueError("context limit must be at least 2048")
        if not 0 < memory_fraction < 1 or type(gpu) is not int or gpu < 0:
            raise ValueError("invalid GPU index or memory fraction")
        if type(tp_size) is not int or tp_size < 1:
            raise ValueError("tensor parallel size must be a positive integer")
        self.contract = Contract.from_config(config, identity=
            f"sglang:{Path(actor).resolve()}:{config['target_revision']}")
        self.context_limit, self.timeout = context_limit - 1, timeout
        self.lock, self.closed = threading.RLock(), False
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=run_worker, args=(child, {
            "checkpoint": str(Path(checkpoint).resolve()), "actor": str(Path(actor).resolve()),
            "context_limit": context_limit, "memory_fraction": memory_fraction, "gpu": gpu,
            "tp_size": tp_size,
        }), name="remory-sglang")
        self.process.start()
        child.close()
        try:
            self.info = self._receive(startup_timeout)
            if not self.info.get("disable_radix_cache") or self.info.get("max_running_requests") != 1:
                raise RuntimeError("worker did not start with isolated residual inference")
            self.context_limit = min(self.context_limit, self.info["context_length"] - 1)
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_pretrained(cls, checkpoint="mocoV3/Qwen3.8-27B-REMORY-1.9B", *, actor_path=None, **kwargs):
        from ..models.load import resolve_checkpoint, resolve_actor
        path, config = resolve_checkpoint(checkpoint)
        return cls(config, path, resolve_actor(config, actor_path), **kwargs)

    def _receive(self, timeout):
        if not self.pipe.poll(timeout):
            self.close()
            raise RuntimeError("SGLang worker timed out and was stopped; retain original history")
        try:
            message = self.pipe.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise RuntimeError("SGLang worker exited; inspect its startup logs") from exc
        if not message["ok"]:
            raise RuntimeError(message["error"])
        return message["result"]

    def build_request(self, source, sampling, payload=None):
        if len(source.input_ids) + sampling["max_new_tokens"] > self.context_limit:
            raise ValueError("request exceeds context limit; refusing truncation")
        from .sglang_hook import parameter_carrier
        request = {"input_ids": list(source.input_ids), "sampling_params": sampling,
                   "rid": "remory-" + uuid.uuid4().hex}
        if payload is not None:
            request["sampling_params"] = {**sampling, "custom_params": {"remory": payload}}
            request["custom_logit_processor"] = parameter_carrier()
            request["return_hidden_states"] = payload["mode"] == "encode"
        return request

    def _request(self, source, sampling, payload=None):
        request = self.build_request(source, sampling, payload)
        with self.lock:
            if self.closed:
                raise RuntimeError("SGLang worker is closed")
            try:
                self.pipe.send(request)
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError("SGLang worker disconnected; retain original history") from exc
            return self._receive(self.timeout)

    def _memory(self, source):
        if source.memory is None:
            if source.memory_positions:
                raise ValueError("memory positions require embeddings")
            return {}
        memory = np.asarray(source.memory, dtype=np.float32)
        if memory.shape != (len(source.memory_positions), self.contract.hidden_size):
            raise ValueError("invalid memory shape")
        return {"positions": list(source.memory_positions), "memory": memory.tolist()}

    def encode(self, source, *, start, end, operation, summary=None, input_depth=0, block_depths=()):
        from .sglang_hook import validate_payload
        from ..online.common import ENCODER_VERSION
        payload = {"mode": "encode", "operation": operation, "start": start, "end": end,
                   "depth": input_depth, "block_depths": list(block_depths), **self._memory(source)}
        if ENCODER_VERSION.get() is not None:
            payload["encoder_version"] = ENCODER_VERSION.get()
        if summary is not None:
            payload["summary"] = np.asarray(summary, dtype=np.float32).tolist()
        validate_payload(payload, len(source.input_ids), self.contract.hidden_size,
                         block=self.contract.block_tokens, max_depth=self.contract.max_depth)
        result = self._request(source, {"temperature": 0.0, "max_new_tokens": 0}, payload)
        chunks = result.get("meta_info", {}).get("hidden_states")
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise RuntimeError("worker did not return one complete encoding")
        memory = np.asarray(chunks[0], dtype=np.float32)
        rows = end-start if operation == "summary" else (
            (end-start+self.contract.block_tokens-1)//self.contract.block_tokens
            * (self.contract.block_tokens//self.contract.ratio))
        if memory.shape != (rows, self.contract.hidden_size) or not np.isfinite(memory).all():
            raise RuntimeError(f"worker returned incomplete or nonfinite {operation} memory: "
                               f"shape {memory.shape}, expected {(rows, self.contract.hidden_size)}")
        return memory

    def generate(self, source, *, max_new_tokens, sampling=None):
        from .sglang_hook import validate_payload
        allowed = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                   "repetition_penalty", "sampling_seed", "json_schema"}
        if set(sampling or {}) - allowed:
            raise ValueError("unsupported sampling options")
        params = {"temperature": 0.0, **(sampling or {}), "max_new_tokens": max_new_tokens,
                  "stop_token_ids": [self.contract.placeholder_id], "skip_special_tokens": False,
                  "no_stop_trim": True}
        payload = {"mode": "recover", **self._memory(source)} if source.memory is not None else None
        if payload:
            validate_payload(payload, len(source.input_ids), self.contract.hidden_size)
        result = self._request(source, params, payload)
        meta = result.get("meta_info", {})
        reason = meta.get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") in {"abort", "error"}:
            raise RuntimeError("SGLang aborted generation")
        return Generation(result.get("text", ""), list(token_ids(result.get("output_ids"),
            "output_ids", empty=True)), {k: meta[k] for k in
            ("prompt_tokens", "completion_tokens", "cached_tokens") if k in meta}, reason)

    def attest(self):
        return {"backend": "sglang", "context_limit": self.context_limit,
                "identity": self.contract.identity, **self.info}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            # The worker can have exited while a scheduler child still exists.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.terminate()
            self.process.join(15)
            if self.process.is_alive():
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    self.process.kill()
                self.process.join(5)
            self.pipe.close()
