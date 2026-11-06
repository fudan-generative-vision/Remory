from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional

import torch

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


def _put_drop_oldest(q: "queue.Queue[Dict[str, Any]]", msg: Dict[str, Any]) -> None:
    try:
        q.put_nowait(msg)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(msg)
    except queue.Full:
        pass


class DAsyncLocalDraftEngine:
    """In-process DASYNC draft engine.

    This is the transition path away from the two-engine Unix socket topology.
    It intentionally keeps the same scalar message protocol as the external
    engine, while running the drafter in a local background thread owned by the
    target worker process. Tensor-conditioned and TP-sharded drafter execution
    should replace this class' raw-token model path, not the verifier protocol.
    """

    def __init__(self, config: Dict[str, Any], *, queue_size: int) -> None:
        self.config = dict(config)
        qsize = max(4, int(queue_size))
        self._cmd_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=qsize)
        self._rx_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=qsize)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dasync-local-draft",
            daemon=True,
        )
        self._thread.start()

    def send(self, msg: Dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        _put_drop_oldest(self._cmd_q, dict(msg))

    def recv_nowait(self) -> Dict[str, Any]:
        return self._rx_q.get_nowait()

    def recv(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._rx_q.get(timeout=timeout)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        _put_drop_oldest(self._cmd_q, {"op": "shutdown"})
        self._thread.join(timeout=1.0)

    def _put_result(self, msg: Dict[str, Any]) -> None:
        if self._closed.is_set() and msg.get("type") != "error":
            return
        _put_drop_oldest(self._rx_q, msg)

    def _send_error(self, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._put_result({"type": "error", "traceback": tb})
        _dasync_dump_event({"role": "draft", "event": "draft_error", "traceback": tb})

    @staticmethod
    def _normalize_condition(config: Dict[str, Any]) -> tuple[str, list[int]]:
        condition_mode = (
            str(config.get("condition_mode") or "none").strip().lower().replace("-", "_")
        )
        if condition_mode in {"off", "disable", "disabled"}:
            condition_mode = "none"
        condition_layer_ids = [int(x) for x in (config.get("condition_layer_ids") or [])]
        return condition_mode, condition_layer_ids

    def _run(self) -> None:
        config = self.config
        try:
            device = str(config["device"])
            if device.startswith("cuda") and torch.cuda.is_available():
                if ":" in device:
                    torch.cuda.set_device(int(device.split(":", 1)[1]))
                else:
                    torch.cuda.set_device(0)

            condition_mode, condition_layer_ids = self._normalize_condition(config)
            if condition_mode != "none":
                raise RuntimeError(
                    "DASYNC local engine currently supports token-only drafting. "
                    "target_hidden conditioning needs a condition-trained drafter "
                    "and TP-local tensor plumbing before it can be enabled."
                )

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
            proposal_mode = _dasync_normalize_proposal_mode(
                config.get("proposal_mode")
            )

            ready = {
                "type": "ready",
                "device": device,
                "model_path": str(config["model_path"]),
                "draft_backend": draft_backend,
                "base_model_path": getattr(model, "_dasync_base_model_path", None),
                "tokenizer_path": tokenizer_path,
                "eos_token_id": eos_token_id,
                "protocol": "dasync-v4-local-thread",
                "condition_mode": condition_mode,
                "condition_layer_ids": condition_layer_ids,
                "proposal_mode": proposal_mode,
            }
            self._put_result(ready)
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
                    "protocol": ready["protocol"],
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
                    "DASYNC local AR anchor requires proposal_mode=hybrid_ar. "
                    "masked_block is the training-consistent default."
                )

            while not self._closed.is_set():
                drained_any = self._drain_commands(states, device, tokenizer)
                if self._closed.is_set():
                    break

                made_progress = self._run_one_draft_step(
                    states=states,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                    draft_backend=draft_backend,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    confidence_threshold=confidence_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    publish_stride=publish_stride,
                    proposal_mode=proposal_mode,
                    ar_anchor_enabled=ar_anchor_enabled,
                    ar_anchor_threshold=ar_anchor_threshold,
                    ar_anchor_max_steps=ar_anchor_max_steps,
                    batch_size=batch_size,
                )
                if not made_progress and not drained_any:
                    time.sleep(idle_sleep_s)
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                logger.exception("DASYNC local draft engine failed")
                self._send_error(exc)

    def _drain_commands(self, states, device: str, tokenizer) -> bool:
        drained_any = False
        while True:
            try:
                cmd = self._cmd_q.get_nowait()
            except queue.Empty:
                break
            drained_any = True
            op = cmd.get("op")
            if op == "shutdown":
                self._closed.set()
                break
            if op == "clear":
                states.clear()
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if op == "remove":
                states.pop(str(cmd.get("rid")), None)
                continue
            if op != "upsert":
                continue

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
                        "engine": "local",
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
                "engine": "local",
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
        return drained_any

    def _make_publish(
        self,
        *,
        rid: str,
        proposal_epoch: int,
        proposal_prefix_len: int,
        proposal_prefix_tail,
        tokenizer,
        max_tokens: int,
        min_tokens: int,
        confidence_threshold: float,
        draft_backend: str,
        proposal_mode: str,
    ):
        def publish(tokens_snapshot, probs_snapshot):
            msg = {
                "type": "proposal",
                "rid": rid,
                "epoch": int(proposal_epoch),
                "prefix_len": int(proposal_prefix_len),
                "prefix_tail_ids": [int(x) for x in proposal_prefix_tail],
                "tokens": [int(x) for x in tokens_snapshot],
                "probs": [float(x) for x in probs_snapshot],
            }
            self._put_result(msg)
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
                    "engine": "local",
                }
            )

        return publish

    def _run_one_draft_step(
        self,
        *,
        states,
        model,
        device: str,
        tokenizer,
        eos_token_id,
        draft_backend: str,
        max_tokens: int,
        min_tokens: int,
        confidence_threshold: float,
        temperature: float,
        top_p: float,
        publish_stride: int,
        proposal_mode: str,
        ar_anchor_enabled: bool,
        ar_anchor_threshold: float,
        ar_anchor_max_steps: int,
        batch_size: int,
    ) -> bool:
        pending_items = [
            (rid, state)
            for rid, state in list(states.items())
            if state.last_proposal_epoch != state.epoch
        ]
        if not pending_items:
            return False

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
                proposal_epoch = int(state.epoch)
                proposal_prefix_len = len(state.prefix_ids)
                proposal_prefix_tail = state.prefix_ids[-_dasync_dump_max_items() :]
                meta.append(
                    (
                        rid,
                        state,
                        proposal_epoch,
                        proposal_prefix_len,
                        proposal_prefix_tail,
                    )
                )
                publish_cbs.append(
                    self._make_publish(
                        rid=rid,
                        proposal_epoch=proposal_epoch,
                        proposal_prefix_len=proposal_prefix_len,
                        proposal_prefix_tail=proposal_prefix_tail,
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        min_tokens=min_tokens,
                        confidence_threshold=confidence_threshold,
                        draft_backend=draft_backend,
                        proposal_mode=proposal_mode,
                    )
                )
                self._put_result(
                    {
                        "type": "draft_started",
                        "rid": rid,
                        "epoch": int(proposal_epoch),
                        "prefix_len": int(proposal_prefix_len),
                    }
                )
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
            for idx, (
                (rid, state, proposal_epoch, proposal_prefix_len, _),
                (tokens, probs),
            ) in enumerate(zip(meta, results, strict=True)):
                state.last_proposal_epoch = proposal_epoch
                if len(tokens) == 0:
                    publish_cbs[idx]([], [])
                self._put_result(
                    {
                        "type": "draft_done",
                        "rid": rid,
                        "epoch": int(proposal_epoch),
                        "prefix_len": int(proposal_prefix_len),
                        "proposal_len": len(tokens),
                    }
                )
                self._dump_rollout_done(
                    tokenizer=tokenizer,
                    rid=rid,
                    proposal_epoch=proposal_epoch,
                    proposal_prefix_len=proposal_prefix_len,
                    tokens=tokens,
                    probs=probs,
                    draft_backend=draft_backend,
                    proposal_mode=proposal_mode,
                    batched=True,
                )
            return True

        rid, state = pending_items[0]
        proposal_epoch = int(state.epoch)
        proposal_prefix_len = len(state.prefix_ids)
        proposal_prefix_tail = state.prefix_ids[-_dasync_dump_max_items() :]
        publish = self._make_publish(
            rid=rid,
            proposal_epoch=proposal_epoch,
            proposal_prefix_len=proposal_prefix_len,
            proposal_prefix_tail=proposal_prefix_tail,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            confidence_threshold=confidence_threshold,
            draft_backend=draft_backend,
            proposal_mode=proposal_mode,
        )
        self._put_result(
            {
                "type": "draft_started",
                "rid": rid,
                "epoch": int(proposal_epoch),
                "prefix_len": int(proposal_prefix_len),
            }
        )
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
        self._put_result(
            {
                "type": "draft_done",
                "rid": rid,
                "epoch": int(proposal_epoch),
                "prefix_len": int(proposal_prefix_len),
                "proposal_len": len(tokens),
            }
        )
        self._dump_rollout_done(
            tokenizer=tokenizer,
            rid=rid,
            proposal_epoch=proposal_epoch,
            proposal_prefix_len=proposal_prefix_len,
            tokens=tokens,
            probs=probs,
            draft_backend=draft_backend,
            proposal_mode=proposal_mode,
            batched=False,
        )
        return True

    @staticmethod
    def _dump_rollout_done(
        *,
        tokenizer,
        rid: str,
        proposal_epoch: int,
        proposal_prefix_len: int,
        tokens,
        probs,
        draft_backend: str,
        proposal_mode: str,
        batched: bool,
    ) -> None:
        _dasync_dump_event(
            {
                "role": "draft",
                "event": "draft_rollout_done",
                "rid": rid,
                "epoch": int(proposal_epoch),
                "prefix_len": int(proposal_prefix_len),
                "proposal_len": len(tokens),
                "token_ids": _dasync_clip_list(tokens),
                "token_text": _dasync_decode_ids(tokenizer, tokens),
                "probs": _dasync_clip_list(probs),
                "draft_backend": draft_backend,
                "proposal_mode": proposal_mode,
                "engine": "local",
                "batched": bool(batched),
            }
        )
