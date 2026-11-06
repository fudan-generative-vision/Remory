# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Standalone DASYNC v3 draft engine.

Run this module as a separate process from the verifier SGLang server.  It owns
its draft model, draft KV/cache state, CUDA context, and any future distributed
world it creates.  The only verifier-facing interface is the JSON-lines Unix
socket protocol in :mod:`sglang.srt.speculative.dasync_ipc`.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional

import torch

from sglang.srt.speculative.dasync_ipc import DAsyncUnixSocketServer
from sglang.srt.speculative.dasync_draft_runtime import (
    _DraftRequestState,
    _dasync_clip_list,
    _dasync_decode_ids,
    _dasync_dump_event,
    _dasync_dump_max_items,
    _dasync_env_flag,
    _dasync_load_draft_model,
    _dasync_load_dump_tokenizer,
    _dasync_normalize_proposal_mode,
    _generate_proposal,
    _generate_specforge_isomorphic_proposals_batched,
    _resolve_dtype,
)

logger = logging.getLogger(__name__)


def _send_error(ipc: DAsyncUnixSocketServer, exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ipc.send({"type": "error", "traceback": tb})
    _dasync_dump_event({"role": "draft", "event": "draft_error", "traceback": tb})


def _ready_message(
    *,
    device: str,
    model_path: str,
    draft_backend: str,
    base_model_path: Optional[str],
    tokenizer_path: Optional[str],
    eos_token_id: Optional[int],
    ipc_path: str,
    condition_mode: str,
    condition_layer_ids: Optional[list[int]],
    proposal_mode: str,
) -> Dict[str, Any]:
    return {
        "type": "ready",
        "device": device,
        "model_path": model_path,
        "draft_backend": draft_backend,
        "base_model_path": base_model_path,
        "tokenizer_path": tokenizer_path,
        "eos_token_id": eos_token_id,
        "ipc_path": ipc_path,
        "protocol": "dasync-v3-jsonl-unix",
        "condition_mode": condition_mode,
        "condition_layer_ids": condition_layer_ids or [],
        "proposal_mode": proposal_mode,
    }


def run_draft_engine(config: Dict[str, Any]) -> None:
    ipc_path = str(config["ipc_path"])
    ipc = DAsyncUnixSocketServer(
        ipc_path, queue_size=int(config.get("queue_size", 4096))
    )
    ipc.start()
    logger.info("DASYNC draft engine listening on unix://%s", ipc_path)

    ready: Optional[Dict[str, Any]] = None
    shutdown_requested = False

    def request_shutdown(signum, frame):  # noqa: ARG001
        nonlocal shutdown_requested
        shutdown_requested = True

    old_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    old_sigint = signal.signal(signal.SIGINT, request_shutdown)

    try:
        device = str(config["device"])
        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                torch.cuda.set_device(int(device.split(":", 1)[1]))
            else:
                torch.cuda.set_device(0)

        condition_mode = str(config.get("condition_mode") or "none").strip().lower().replace("-", "_")
        if condition_mode in {"off", "disable", "disabled"}:
            condition_mode = "none"
        condition_layer_ids = [int(x) for x in (config.get("condition_layer_ids") or [])]
        if condition_mode != "none":
            raise RuntimeError(
                "DASYNC condition_mode=%s was requested, but this draft engine is still the "
                "v3 token-only runtime. Train/load a condition-capable DASYNC drafter and "
                "implement the v4 condition transport before enabling this flag."
                % condition_mode
            )
        proposal_mode = _dasync_normalize_proposal_mode(config.get("proposal_mode"))

        dtype = _resolve_dtype(config.get("dtype"))
        model, eos_token_id, draft_backend = _dasync_load_draft_model(
            config, device=device, dtype=dtype
        )
        tokenizer_path = (
            config.get("tokenizer_path")
            or config.get("target_model_path")
            or config.get("base_model_path")
            or getattr(model, "_dasync_base_model_path", None)
            or config.get("model_path")
        )
        tokenizer = _dasync_load_dump_tokenizer(
            tokenizer_path,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )

        ready = _ready_message(
            device=device,
            model_path=str(config["model_path"]),
            draft_backend=draft_backend,
            base_model_path=getattr(model, "_dasync_base_model_path", None),
            tokenizer_path=tokenizer_path,
            eos_token_id=eos_token_id,
            ipc_path=ipc_path,
            condition_mode=condition_mode,
            condition_layer_ids=condition_layer_ids,
            proposal_mode=proposal_mode,
        )
        ipc.send(ready)
        _dasync_dump_event(
            {
                "role": "draft",
                "event": "draft_model_ready",
                "device": device,
                "model_path": config["model_path"],
                "draft_backend": draft_backend,
                "base_model_path": getattr(model, "_dasync_base_model_path", None),
                "tokenizer_path": tokenizer_path,
                "eos_token_id": eos_token_id,
                "ipc_path": ipc_path,
                "condition_mode": condition_mode,
                "condition_layer_ids": condition_layer_ids,
                "proposal_mode": proposal_mode,
            }
        )

        states: Dict[str, _DraftRequestState] = {}
        idle_sleep_s = float(config.get("idle_sleep_s", 0.0005))
        max_tokens = int(config.get("max_tokens", 32))
        min_tokens = int(config.get("min_tokens", 0))
        confidence_threshold = float(config.get("confidence_threshold", 0.0))
        temperature = float(config.get("temperature", 0.0))
        top_p = float(config.get("top_p", 1.0))
        publish_stride = int(config.get("publish_stride", 4))
        ar_anchor_enabled = bool(config.get("ar_anchor_enabled", False))
        ar_anchor_threshold = float(config.get("ar_anchor_threshold", 0.0))
        ar_anchor_max_steps = int(config.get("ar_anchor_max_steps", 0))
        batch_size = max(1, int(config.get("batch_size", 1)))
        if ar_anchor_enabled and proposal_mode != "hybrid_ar":
            raise RuntimeError(
                "DASYNC AR anchor requires proposal_mode=hybrid_ar. "
                "masked_block is the training-consistent default."
            )

        while not shutdown_requested:
            drained_any = False
            while True:
                try:
                    cmd = ipc.recv_nowait()
                except queue.Empty:
                    break
                drained_any = True
                typ = cmd.get("type")
                if typ == DAsyncUnixSocketServer.CLIENT_CONNECTED:
                    if ready is not None:
                        ipc.send(ready)
                    continue
                if typ == DAsyncUnixSocketServer.CLIENT_DISCONNECTED:
                    continue
                if typ == "error":
                    logger.warning("DASYNC IPC server event: %s", cmd.get("traceback"))
                    continue

                op = cmd.get("op")
                if op == "shutdown":
                    shutdown_requested = True
                    break
                if op == "clear":
                    states.clear()
                    if device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if op == "remove":
                    states.pop(str(cmd.get("rid")), None)
                    continue
                if op == "upsert":
                    rid = str(cmd["rid"])
                    epoch = int(cmd["epoch"])
                    prefix_ids = [int(x) for x in cmd.get("prefix_ids", [])]
                    old = states.get(rid)
                    if old is not None and epoch < int(old.epoch):
                        _dasync_dump_event(
                            {
                                "role": "draft",
                                "event": "draft_drop_stale_upsert",
                                "rid": rid,
                                "epoch": int(epoch),
                                "latest_epoch": int(old.epoch),
                                "prefix_len": len(prefix_ids),
                            }
                        )
                        continue
                    prefix_tail = prefix_ids[-_dasync_dump_max_items() :]
                    evt = {
                        "role": "draft",
                        "event": "draft_recv_upsert",
                        "rid": rid,
                        "epoch": int(epoch),
                        "prefix_len": len(prefix_ids),
                        "prefix_tail_ids": prefix_tail,
                        "prefix_tail_text": _dasync_decode_ids(tokenizer, prefix_tail),
                    }
                    if _dasync_env_flag("SGLANG_DASYNC_DUMP_FULL_PREFIX", False):
                        evt["prefix_ids"] = _dasync_clip_list(prefix_ids, len(prefix_ids))
                    _dasync_dump_event(evt)
                    if old is None:
                        states[rid] = _DraftRequestState(
                            rid=rid,
                            epoch=epoch,
                            prefix_ids=prefix_ids,
                            cache_token_ids=[],
                        )
                    else:
                        old.epoch = epoch
                        old.prefix_ids = prefix_ids
                        old.last_proposal_epoch = None
                    continue

            if shutdown_requested:
                break

            made_progress = False
            pending_items = [
                (rid, state)
                for rid, state in list(states.items())
                if state.last_proposal_epoch != state.epoch
            ]

            def make_publish(rid, proposal_epoch, proposal_prefix_len, proposal_prefix_tail):
                def publish(tokens_snapshot, probs_snapshot):
                    msg = {
                        "type": "proposal",
                        "rid": rid,
                        "epoch": proposal_epoch,
                        "prefix_len": proposal_prefix_len,
                        "prefix_tail_ids": [int(x) for x in proposal_prefix_tail],
                        "tokens": [int(x) for x in tokens_snapshot],
                        "probs": [float(x) for x in probs_snapshot],
                    }
                    ipc.send(msg)
                    _dasync_dump_event(
                        {
                            "role": "draft",
                            "event": "draft_publish_proposal",
                            "rid": rid,
                            "epoch": int(proposal_epoch),
                            "prefix_len": int(proposal_prefix_len),
                            "prefix_tail_ids": _dasync_clip_list(proposal_prefix_tail),
                            "prefix_tail_text": _dasync_decode_ids(tokenizer, proposal_prefix_tail),
                            "proposal_len": len(tokens_snapshot),
                            "token_ids": _dasync_clip_list(tokens_snapshot),
                            "token_text": _dasync_decode_ids(tokenizer, tokens_snapshot),
                            "probs": _dasync_clip_list(probs_snapshot),
                            "truncated": len(tokens_snapshot) > _dasync_dump_max_items(),
                            "max_tokens": int(max_tokens),
                            "min_tokens": int(min_tokens),
                            "confidence_threshold": float(confidence_threshold),
                            "draft_backend": draft_backend,
                            "proposal_mode": proposal_mode,
                            "ipc_path": ipc_path,
                        }
                    )
                return publish

            if (
                len(pending_items) > 1
                and batch_size > 1
                and getattr(model, "_dasync_block_drafter", False)
                and proposal_mode == "masked_block"
                and not ar_anchor_enabled
            ):
                chunk = pending_items[:batch_size]
                publish_cbs = []
                meta = []
                for rid, state in chunk:
                    proposal_epoch = state.epoch
                    proposal_prefix_len = len(state.prefix_ids)
                    proposal_prefix_tail = state.prefix_ids[-_dasync_dump_max_items() :]
                    meta.append((rid, state, proposal_epoch, proposal_prefix_len, proposal_prefix_tail))
                    publish_cbs.append(make_publish(rid, proposal_epoch, proposal_prefix_len, proposal_prefix_tail))
                    ipc.send({"type": "draft_started", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len)})
                results = _generate_specforge_isomorphic_proposals_batched(
                    [state for _, state, _, _, _ in meta],
                    model=model,
                    device=device,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    confidence_threshold=confidence_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    publish_stride=publish_stride,
                    publish_cbs=publish_cbs,
                )
                for idx, ((rid, state, proposal_epoch, proposal_prefix_len, _), (tokens, probs)) in enumerate(zip(meta, results, strict=True)):
                    state.last_proposal_epoch = proposal_epoch
                    if len(tokens) == 0:
                        publish_cbs[idx]([], [])
                    ipc.send({"type": "draft_done", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len), "proposal_len": len(tokens)})
                    _dasync_dump_event({"role": "draft", "event": "draft_rollout_done", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len), "proposal_len": len(tokens), "token_ids": _dasync_clip_list(tokens), "token_text": _dasync_decode_ids(tokenizer, tokens), "probs": _dasync_clip_list(probs), "draft_backend": draft_backend, "proposal_mode": proposal_mode, "ipc_path": ipc_path, "batched": True})
                made_progress = True
            elif pending_items:
                rid, state = pending_items[0]
                proposal_epoch = state.epoch
                proposal_prefix_len = len(state.prefix_ids)
                proposal_prefix_tail = state.prefix_ids[-_dasync_dump_max_items() :]
                publish = make_publish(rid, proposal_epoch, proposal_prefix_len, proposal_prefix_tail)
                ipc.send({"type": "draft_started", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len)})
                tokens, probs = _generate_proposal(
                    state,
                    model=model,
                    device=device,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    confidence_threshold=confidence_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    publish_stride=publish_stride,
                    publish_cb=publish,
                    proposal_mode=proposal_mode,
                    ar_anchor_enabled=ar_anchor_enabled,
                    ar_anchor_threshold=ar_anchor_threshold,
                    ar_anchor_max_steps=ar_anchor_max_steps,
                )
                state.last_proposal_epoch = proposal_epoch
                if len(tokens) == 0:
                    publish([], [])
                ipc.send({"type": "draft_done", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len), "proposal_len": len(tokens)})
                _dasync_dump_event({"role": "draft", "event": "draft_rollout_done", "rid": rid, "epoch": int(proposal_epoch), "prefix_len": int(proposal_prefix_len), "proposal_len": len(tokens), "token_ids": _dasync_clip_list(tokens), "token_text": _dasync_decode_ids(tokenizer, tokens), "probs": _dasync_clip_list(probs), "draft_backend": draft_backend, "proposal_mode": proposal_mode, "ipc_path": ipc_path, "batched": False})
                made_progress = True

            if not made_progress and not drained_any:
                time.sleep(idle_sleep_s)
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            logger.exception("DASYNC draft engine failed")
            _send_error(ipc, exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        ipc.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone DASYNC v3 draft engine")
    parser.add_argument("--socket-path", dest="ipc_path", default=os.environ.get("SGLANG_DASYNC_IPC_PATH"), required=os.environ.get("SGLANG_DASYNC_IPC_PATH") is None)
    parser.add_argument("--draft-model-path", dest="model_path", required=True)
    parser.add_argument("--target-model-path", dest="target_model_path", default=None)
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", default=None)
    parser.add_argument("--revision", dest="revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--publish-stride", type=int, default=4)
    parser.add_argument("--proposal-mode", default=os.environ.get("DASYNC_DRAFT_PROPOSAL_MODE", "masked_block"))
    parser.add_argument("--ar-anchor-enabled", action="store_true")
    parser.add_argument("--ar-anchor-threshold", type=float, default=0.0)
    parser.add_argument("--ar-anchor-max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DASYNC_DRAFT_BATCH_SIZE", "8")))
    parser.add_argument("--condition-mode", choices=["none", "target_hidden"], default="none")
    parser.add_argument("--condition-layer-ids", type=int, nargs="*", default=None)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--idle-sleep-s", type=float, default=0.0005)
    parser.add_argument("--log-level", default=os.environ.get("DASYNC_DRAFT_LOG_LEVEL", "INFO"))
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = vars(args)
    if config["min_tokens"] is None:
        config["min_tokens"] = config["max_tokens"]
    if int(config["max_tokens"]) < 0:
        parser.error("--max-tokens must be non-negative")
    if not (0 <= int(config["min_tokens"]) <= int(config["max_tokens"])):
        parser.error("--min-tokens must be in [0, --max-tokens]")
    if not (0.0 <= float(config["confidence_threshold"]) <= 1.0):
        parser.error("--confidence-threshold must be in [0, 1]")
    if not (0.0 < float(config["top_p"]) <= 1.0):
        parser.error("--top-p must be in (0, 1]")
    if int(config["publish_stride"]) <= 0:
        parser.error("--publish-stride must be positive")
    if int(config["batch_size"]) <= 0:
        parser.error("--batch-size must be positive")
    try:
        config["proposal_mode"] = _dasync_normalize_proposal_mode(
            config.get("proposal_mode")
        )
    except ValueError as exc:
        parser.error(str(exc))
    condition_layer_ids = config.get("condition_layer_ids") or []
    if any(int(x) < 0 for x in condition_layer_ids):
        parser.error("--condition-layer-ids must be non-negative integers")
    if len(set(int(x) for x in condition_layer_ids)) != len(condition_layer_ids):
        parser.error("--condition-layer-ids must not contain duplicates")
    if config.get("condition_mode") == "target_hidden" and not condition_layer_ids:
        parser.error("--condition-mode=target_hidden requires --condition-layer-ids")
    if not (0.0 <= float(config["ar_anchor_threshold"]) <= 1.0):
        parser.error("--ar-anchor-threshold must be in [0, 1]")
    if int(config["ar_anchor_max_steps"]) < 0:
        parser.error("--ar-anchor-max-steps must be non-negative")
    if config.get("ar_anchor_enabled") and config["proposal_mode"] != "hybrid_ar":
        parser.error("--ar-anchor-enabled requires --proposal-mode=hybrid_ar")
    run_draft_engine(config)


if __name__ == "__main__":
    main(sys.argv[1:])
