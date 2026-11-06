"""Verified CPU tensor transport for summary-conditioned residual SGLang.

The v1 implementation trades JSON transport overhead for a small explicit ABI.
No raw source or summary hidden states are persisted as public telemetry.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from context_compression.grepseek_sglang_client import (
    GrepSeekSGLangClient, SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1,
)
from context_compression.qwen38_residual_pyramid import plan_pyramid
from context_compression.qwen38_residual_sglang_hook import (
    RESIDUAL_HOOK_FACTORY, RESIDUAL_SGLANG_ABI, RESIDUAL_RECURSIVE_CAPABILITY,
    RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
    validate_residual_encode_payload,
)
from context_compression.sglang_compressor_hook import CUSTOM_PARAM_KEY


RESIDUAL_DEPTH_POLICY = "unbounded_logical_depth_saturating_embedding_v1"


def _ids(values: Sequence[int], name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be nonempty token IDs")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return list(values)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SummaryResidualSGLangClient(GrepSeekSGLangClient):
    server_memory = False

    def valid_memory(self, value):
        if self.server_memory:
            from context_compression.remory_memory import RemoteMemory
            return isinstance(value, RemoteMemory) and value.backend == self.base_url
        return (isinstance(value, torch.Tensor) and value.ndim == 2
                and value.is_floating_point() and bool(torch.isfinite(value).all()))

    def compose_memory(self, values):
        return torch.cat(values)

    def _slice_memory(self, value, start, stop):
        return value[start:stop]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sglang_cache_policy", SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1)
        kwargs.setdefault("enable_request_cancellation", True)
        super().__init__(*args, **kwargs)
        self._residual_stopping = threading.Event()
        self._residual_attestation = None
        self._residual_config = None
        self._residual_context_limit = None

    def _validate_generate_lifecycle_locked(self):
        if self._residual_stopping.is_set():
            raise RuntimeError("residual client is stopping; no further model request may start")

    def abort_active_requests(self):
        # Seal the multi-request build before taking the inherited snapshot.
        # The same lock guards registration, eliminating the between-node race.
        with self._request_cancellation_lock:
            self._residual_stopping.set()
        return super().abort_active_requests()

    def attest_residual_server(self, checkpoint: str | Path, server_info=None) -> dict[str, Any]:
        from context_compression.qwen38_residual_serving_contract import load_residual_serving_contract
        checkpoint = Path(checkpoint).expanduser().resolve()
        config = load_residual_serving_contract(checkpoint, verify_weights=False)
        info = self.server_info() if server_info is None else server_info
        if RESIDUAL_SGLANG_ABI not in (info.get("request_capabilities") or []):
            raise RuntimeError("SGLang scheduler does not attest the residual encode ABI")
        if RESIDUAL_RECURSIVE_CAPABILITY not in (info.get("request_capabilities") or []):
            raise RuntimeError("SGLang lacks recursive residual source encoding; restart the SGLang server")
        if RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY not in (info.get("request_capabilities") or []):
            raise RuntimeError("SGLang still has fixed residual summary caps; restart the SGLang server")
        hooks = info.get("forward_hooks") or []
        matching = [h for h in hooks if isinstance(h, Mapping) and h.get("hook_factory") == RESIDUAL_HOOK_FACTORY]
        if len(matching) != 1:
            raise RuntimeError("SGLang has no unique SummaryResidualCompressor hook; do not use repeat checkpoint server")
        hook = matching[0].get("config")
        if not isinstance(hook, Mapping) or hook.get("residual_serving_abi") != RESIDUAL_SGLANG_ABI:
            raise RuntimeError("SGLang residual serving ABI mismatch")
        from context_compression.remory_identity import check_server_identity
        portable = check_server_identity(info, checkpoint, config)
        if not portable and (not isinstance(hook.get("checkpoint"), str) or Path(hook["checkpoint"]).resolve() != checkpoint):
            raise RuntimeError("SGLang residual compressor checkpoint mismatch")
        if not portable and Path(str(info.get("model_path", ""))).resolve() != Path(config["target_model"]).resolve():
            raise RuntimeError("SGLang target model differs from residual training target")
        cache = self.attest_sglang_cache_policy(info)
        limits = [info.get("context_length"), info.get("max_total_tokens"),
                  info.get("max_total_num_tokens"), info.get("max_req_input_len")]
        for worker in info.get("internal_states", []) or []:
            if isinstance(worker, Mapping):
                limits.extend([worker.get("max_total_num_tokens"), worker.get("max_req_input_len")])
        observed = [int(v) for v in limits if isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 1]
        if not observed:
            raise RuntimeError("SGLang does not advertise a residual raw-input context ceiling")
        self._residual_context_limit = min(observed) - 1
        receipt = {
            "abi": RESIDUAL_SGLANG_ABI, "hook_factory": RESIDUAL_HOOK_FACTORY,
            "checkpoint": str(checkpoint), "config_sha256": _sha(config),
            "context_limit": self._residual_context_limit, "cache": cache,
            "transport": "cpu_tensor_json_v1", "source_modality": "text_only",
            "recursive_compact_capability": RESIDUAL_RECURSIVE_CAPABILITY,
            "dynamic_summary_capability": RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
        }
        self._residual_config = config
        self._residual_attestation = receipt
        return dict(receipt)

    def _encode_residual(self, ids, *, start, end, operation, input_depth=0,
                         summary_features=None, child_memory=None, expected_rows,
                         source_memory_positions=None, block_input_depths=None):
        if self._residual_attestation is None or not self._isolated_cache_policy_active():
            raise RuntimeError("residual encoding requires an attested residual server and isolated cache")
        self._validate_generate_lifecycle_locked()
        if len(ids) > self._residual_context_limit:
            raise ValueError(f"residual {operation} source length {len(ids)} exceeds SGLang ceiling "
                             f"{self._residual_context_limit}; full raw history is never truncated")
        residual = {"abi": RESIDUAL_SGLANG_ABI, "operation": operation, "input_depth": input_depth}
        if summary_features is not None:
            residual["summary_features"] = summary_features.detach().float().cpu().tolist()
        if block_input_depths is not None:
            residual["block_input_depths"] = list(block_input_depths)
        config = {"version": 1, "mode": "encode", "evidence_spans": [[start, end]],
                  "source_token_lengths": [end - start], "residual": residual}
        if child_memory is not None:
            config.update(memory_positions=(list(source_memory_positions)
                                           if source_memory_positions is not None else list(range(start, end))),
                          memory_embeddings=child_memory.detach().float().cpu().tolist())
        width = int(self._residual_config["compressor"]["target_hidden_size"])
        validate_residual_encode_payload(config, token_count=len(ids), hidden_size=width,
                                         max_depth=self._residual_config["compressor"]["max_depth"])
        request = {
            "input_ids": ids, "extra_key": "residual-" + uuid.uuid4().hex,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0,
                                "custom_params": {CUSTOM_PARAM_KEY: config}},
            "return_hidden_states": True, "custom_logit_processor": self._processor_string(),
            "cache_policy": "bypass",
        }
        response = self._post_generate(request)
        self._require_cache_bypass_receipt(response)
        chunks = (response.get("meta_info") or {}).get("hidden_states")
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise RuntimeError("residual encode requires one complete hidden-state chunk")
        memory = torch.tensor(chunks[0], dtype=torch.float32)
        if memory.shape != (expected_rows, width) or not bool(torch.isfinite(memory).all()):
            raise RuntimeError("residual encode returned invalid shape or nonfinite rows")
        return memory

    def build_residual_memory(self, prefix_ids, history_ids, summary_ids, *, contract,
                              memory_budget=4096, block_tokens=1024, max_depth=8,
                              image_data=None, source_memory=None,
                              source_memory_positions=(), source_memory_depths=()):
        """Re-encode the active causal source, conditioned on the new summary.

        At recursive compacts history_ids contains old S + soft placeholders +
        new raw rounds. Old memory is scattered at absolute source positions,
        not decoded as placeholder text or copied into the new frontier.

        Returns ``memory_embeddings`` [slots,hidden] on CPU and a JSON receipt.
        Every partial raw leaf receives 64 slots, without synthetic source rows.
        Logical depths remain unbounded and exact. Only the checkpoint embedding
        coordinate saturates at max_depth-1, explicitly recorded as an inference
        extension. Source embeddings, positions and text are never discarded.
        """
        if image_data:
            raise ValueError("residual source ABI is text-only; image source compression is unsupported")
        self._validate_generate_lifecycle_locked()
        if self._residual_config is None:
            raise RuntimeError("attest_residual_server must precede residual memory construction")
        prefix = _ids(prefix_ids, "prefix_ids")
        history = _ids(history_ids, "history_ids")
        summary = _ids(summary_ids, "summary_ids")
        c = self._residual_config
        if dict(contract) != c["summary_contract"]:
            raise ValueError("residual summary envelope differs from attested checkpoint")
        if ((block_tokens, max_depth) != (c["residual"]["block_tokens"], c["compressor"]["max_depth"])
                or (not self.server_memory and memory_budget != c["residual"]["memory_budget"])):
            raise ValueError("residual pyramid geometry differs from attested checkpoint")
        if self._residual_attestation.get("dynamic_summary_capability") != RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY:
            raise RuntimeError("dynamic residual summary requires an attested server capability")
        source = prefix + history
        recursive = source_memory is not None
        leaf_input_depths = [0] * ((len(history) + block_tokens - 1) // block_tokens)
        if recursive:
            if self._residual_attestation.get("recursive_compact_capability") != RESIDUAL_RECURSIVE_CAPABILITY:
                raise RuntimeError("recursive residual encoding requires an attested server capability")
            positions = list(source_memory_positions)
            depths = list(source_memory_depths)
            if (not self.valid_memory(source_memory)
                    or source_memory.shape != (len(positions), c["compressor"]["target_hidden_size"])
                    or not positions or len(depths) != len(positions)
                    or any(isinstance(p, bool) or not isinstance(p, int)
                           or not len(prefix) <= p < len(source) for p in positions)
                    or positions != sorted(set(positions))
                    or any(isinstance(d, bool) or not isinstance(d, int)
                           or d <= 0 for d in depths)):
                raise ValueError("invalid recursive source memory coordinates/logical depths")
            for position, depth in zip(positions, depths):
                block_index = (position - len(prefix)) // block_tokens
                leaf_input_depths[block_index] = max(leaf_input_depths[block_index], depth)
        elif source_memory_positions or source_memory_depths:
            raise ValueError("source memory coordinates require source embeddings")
        summary_prefix = prefix + list(contract["student_summary_before_ids"])
        encoded_summary = summary_prefix + summary + list(contract["student_summary_after_ids"])
        # Validate all source widths before any encode requests; never clip H.
        if max(len(source), len(encoded_summary), len(prefix) + block_tokens) > self._residual_context_limit:
            raise ValueError("full residual source/summary/parent exceeds SGLang context ceiling; no truncation")
        started = time.perf_counter()
        plan = plan_pyramid(len(history), budget=memory_budget, block=block_tokens,
                            ratio=c["residual"]["compression_ratio"], max_depth=None,
                            leaf_input_depths=leaf_input_depths)
        embedding_depths = [min(d, max_depth - 1) for d in leaf_input_depths]
        summary_features = self._encode_residual(
            encoded_summary, start=len(summary_prefix), end=len(summary_prefix) + len(summary),
            operation="summary", expected_rows=len(summary),
        )
        leaves = [node for node in plan.nodes if not node.children]
        leaf_memory = self._encode_residual(
            source, start=len(prefix), end=len(source),
            operation="recursive_leaves" if recursive else "leaves",
            summary_features=summary_features, expected_rows=len(leaves) * plan.slots_per_node,
            **(dict(child_memory=source_memory, source_memory_positions=positions,
                    block_input_depths=embedding_depths) if recursive else {}),
        )
        if len(leaves) == len(plan.nodes):
            memory = leaf_memory
        else:
            outputs = {node.index: self._slice_memory(leaf_memory, i * plan.slots_per_node, (i + 1) * plan.slots_per_node)
                       for i, node in enumerate(leaves)}
            del leaf_memory
            for node in plan.nodes:
                if not node.children:
                    continue
                children = self.compose_memory([outputs.pop(k) for k in node.children])
                if children.shape[0] != block_tokens:
                    raise RuntimeError("residual parent requires exactly one physical soft block")
                parent_ids = prefix + [self.eos_token_id] * block_tokens
                outputs[node.index] = self._encode_residual(
                    parent_ids, start=len(prefix), end=len(parent_ids), operation="parent",
                    input_depth=min(node.depth - 1, max_depth - 1), summary_features=summary_features,
                    child_memory=children, expected_rows=plan.slots_per_node,
                )
            memory = self.compose_memory([outputs[k] for k in plan.frontier])
        if memory.shape != (plan.soft_tokens, c["compressor"]["target_hidden_size"]):
            raise RuntimeError("residual frontier geometry mismatch")
        return {"memory_embeddings": memory, "receipt": {
            **self._residual_attestation, "prefix_sha256": _sha(prefix),
            "history_sha256": _sha(history), "summary_sha256": _sha(summary),
            "source_tokens": len(history), "summary_tokens": len(summary),
            "fixed_summary_token_cap": None,
            "source_policy": "previous_summary_soft_memory_raw_tail_v1" if recursive else "initial_raw_history_v1",
            "source_memory_slots": int(source_memory.shape[0]) if recursive else 0,
            "block_input_depths": leaf_input_depths,
            "block_embedding_input_depths": embedding_depths,
            "recursive_depth_policy": RESIDUAL_DEPTH_POLICY,
            "logical_depth_limit": None,
            "checkpoint_embedding_depth_count": max_depth,
            "embedding_depth_saturated": any(node.depth > max_depth for node in plan.nodes),
            "parent_embedding_input_depths": [min(node.depth - 1, max_depth - 1)
                                              for node in plan.nodes if node.children],
            "slot_depths": [plan.nodes[k].depth for k in plan.frontier
                            for _ in range(plan.slots_per_node)],
            "cross_summary_reconditioning": recursive,
            "soft_tokens": plan.soft_tokens, "block_tokens": block_tokens,
            "memory_budget": memory_budget, "max_depth": plan.max_depth,
            "leaf_count": len(leaves), "parent_count": len(plan.nodes) - len(leaves),
            "frontier": list(plan.frontier), "nodes": [asdict(n) for n in plan.nodes],
            "encode_requests": 2 + len(plan.nodes) - len(leaves),
            "elapsed_seconds": time.perf_counter() - started,
        }}
