"""Explicit, fail-closed SGLang ABI for summary-conditioned residual memory.

Ordinary recovery reuses the existing soft-input scatter operation. Encoding
does not reuse the repeat compressor: it requires a residual checkpoint,
captures causal source layers, and conditions every node on one fixed summary.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from context_compression.sglang_compressor_hook import (
    CUSTOM_PARAM_KEY, CompressorCheckpointMetadata, SpecForgeCompressorContractError,
    SpecForgeContextCompressorHook, GENERATION_DEGENERATION_GUARD_SERVER_ABI,
    _compute_hook_abi_digest, GPU_MEMORY_HANDLE_POLICY, GPU_MEMORY_HANDLE_META_KEY,
    normalize_memory_handle_descriptor, normalize_memory_handle_publish_claim,
    compute_memory_handle_identity,
)

RESIDUAL_SGLANG_ABI = "qwen38_summary_residual_sglang_v1"
RESIDUAL_RECURSIVE_CAPABILITY = "qwen38_recursive_summary_residual_v1"
RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY = "qwen38_residual_dynamic_summary_v1"
RESIDUAL_HOOK_FACTORY = (
    "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook"
)
RESIDUAL_CONTEXT_ORGANIZATION = "qwen38_summary_residual_v1"


def _load_summary_residual_compressor(metadata, device, dtype):
    """Assign mmap weights without leaving nonpersistent RoPE buffers on meta."""
    from context_compression.qwen38_residual_serving_contract import build_summary_residual_compressor

    model = build_summary_residual_compressor(metadata.raw_config, device="meta", checkpoint=metadata.checkpoint)
    if metadata.state_path.name == "model.safetensors":
        from safetensors.torch import load_file
        state = load_file(str(metadata.state_path), device="cpu")
    else:
        state = torch.load(metadata.state_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    # RoPE frequencies are deterministic, nonpersistent buffers, so they are
    # deliberately absent from the checkpoint. Rebuild the complete parameter-
    # free module, including original_inv_freq and its cached sequence metadata.
    # Do not use to_empty: that would discard the learned weights just assigned.
    rotary = model.rotary_emb
    if any(value.is_meta for value in rotary.buffers()):
        if rotary.state_dict() or any(True for _ in rotary.parameters()):
            raise RuntimeError("residual rotary module unexpectedly contains checkpoint state")
        with torch.device("cpu"):
            model.rotary_emb = type(rotary)(model.config, device="cpu")
    remaining = [name for name, value in (*model.named_parameters(), *model.named_buffers())
                 if value.is_meta]
    if remaining:
        raise RuntimeError(f"residual checkpoint left unexpected meta tensors: {remaining}")
    return model.to(device=device, dtype=dtype).requires_grad_(False).eval()


def validate_residual_encode_payload(payload: Mapping[str, Any], *, token_count: int,
                                     hidden_size: int, max_depth: int = 8) -> dict:
    """Shared scheduler/model validation, before any collective model forward."""
    if payload.get("version") != 1 or payload.get("mode") != "encode":
        raise ValueError("residual requests require encode version 1")
    residual = payload.get("residual")
    if not isinstance(residual, Mapping) or residual.get("abi") != RESIDUAL_SGLANG_ABI:
        raise ValueError("missing or unsupported residual serving ABI")
    operation = residual.get("operation")
    if operation not in {"summary", "leaves", "parent", "recursive_leaves"}:
        raise ValueError("unsupported residual encode operation")
    expected = {"version", "mode", "evidence_spans", "source_token_lengths", "residual"}
    residual_fields = {"abi", "operation", "input_depth"}
    if "memory_handle_publish" in payload:
        normalize_memory_handle_publish_claim(payload["memory_handle_publish"])
        expected.add("memory_handle_publish")
    if operation != "summary":
        residual_fields.add("summary_handle" if "summary_handle" in residual else "summary_features")
    if operation in {"parent", "recursive_leaves"}:
        expected |= {"memory_positions", "memory_handle" if "memory_handle" in payload else "memory_embeddings"}
    if operation == "recursive_leaves":
        residual_fields.add("block_input_depths")
    if set(payload) != expected or set(residual) != residual_fields:
        raise ValueError("residual encode contains missing or unsupported fields")
    spans = payload.get("evidence_spans")
    lengths = payload.get("source_token_lengths")
    if not isinstance(spans, (list, tuple)) or len(spans) != 1:
        raise ValueError("residual encode requires one physical source span")
    span = spans[0]
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError("residual source span must be [start,end]")
    start, end = span
    if any(isinstance(v, bool) or not isinstance(v, int) for v in span):
        raise ValueError("residual source coordinates must be integers")
    if not 0 < start < end <= token_count or lengths != [end - start]:
        raise ValueError("residual source must exactly name valid physical rows after prefix")
    depth = residual["input_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth < max_depth:
        raise ValueError("residual input_depth exceeds checkpoint geometry")
    if operation in {"summary", "leaves", "recursive_leaves"} and depth != 0:
        raise ValueError("summary and raw leaves require input_depth=0")
    if operation != "summary" and end != token_count:
        raise ValueError("residual source must cover the complete post-prefix sequence")

    def matrix(value, rows=None):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("residual features require nonempty rows")
        if rows is not None and len(value) != rows:
            raise ValueError("residual feature row count mismatch")
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != hidden_size:
                raise ValueError("residual feature width mismatch")
            if any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) for v in row):
                raise ValueError("residual features must be finite floats")

    def handle(value, rows=None):
        d = normalize_memory_handle_descriptor(value)
        if d["shape"][1] != hidden_size or (rows is not None and d["shape"][0] != rows):
            raise ValueError("residual memory handle shape mismatch")

    def source_matrix(rows):
        if "memory_handle" in payload:
            handle(payload["memory_handle"], rows)
        else:
            matrix(payload["memory_embeddings"], rows)

    if "memory_handle_publish" in payload and ("memory_embeddings" in payload or "summary_features" in residual):
        raise ValueError("server memory transitions must not carry client tensors")
    if operation != "summary":
        if "summary_handle" in residual:
            handle(residual["summary_handle"])
        else:
            matrix(residual["summary_features"])
    if operation == "parent":
        if depth == 0 or end - start != 1024:
            raise ValueError("residual parent consumes exactly 1024 soft rows at depth>0")
        positions = payload["memory_positions"]
        if positions != list(range(start, end)) or any(isinstance(v, bool) for v in positions):
            raise ValueError("residual parent memory must cover every source row")
        source_matrix(1024)
    if operation == "recursive_leaves":
        positions = payload["memory_positions"]
        if not isinstance(positions, list) or not positions or any(
            isinstance(p, bool) or not isinstance(p, int) or not start <= p < end
            for p in positions
        ) or positions != sorted(set(positions)):
            raise ValueError("recursive memory positions must be distinct ordered post-prefix rows")
        source_matrix(len(positions))
        depths = residual["block_input_depths"]
        blocks = (end - start + 1023) // 1024
        if not isinstance(depths, list) or len(depths) != blocks or any(
            isinstance(d, bool) or not isinstance(d, int) or not 0 <= d < max_depth
            for d in depths
        ):
            raise ValueError("recursive block depths exceed checkpoint geometry")
        soft_blocks = {(p - start) // 1024 for p in positions}
        if any((d > 0) != (i in soft_blocks) for i, d in enumerate(depths)):
            raise ValueError("recursive block depths must distinguish raw and soft-containing blocks")
    return dict(residual)


class SummaryResidualSGLangHook(SpecForgeContextCompressorHook):
    def __init__(self, config, *, compressor_loader=None):
        from context_compression.qwen38_residual_serving_contract import (
            load_residual_serving_contract, resolve_residual_serving_contract,
        )
        allowed = {"checkpoint", "residual_serving_abi",
                   "generation_degeneration_guard_server_abi", "sync_token_ids_across_tp",
                   "gpu_memory_handle_policy", "gpu_memory_handle_ttl_seconds",
                   "gpu_memory_handle_max_entries_per_tp_rank", "gpu_memory_handle_max_bytes_per_tp_rank"}
        if not isinstance(config, Mapping) or set(config) - allowed:
            raise ValueError("unsupported residual hook config fields")
        if config.get("residual_serving_abi") != RESIDUAL_SGLANG_ABI:
            raise ValueError("residual hook requires explicit residual_serving_abi")
        checkpoint = Path(config["checkpoint"]).expanduser().resolve()
        raw = load_residual_serving_contract(checkpoint, verify_weights=True)
        from context_compression.remory_identity import register_loaded_identity
        register_loaded_identity(checkpoint, raw)
        self.residual_contract = resolve_residual_serving_contract(raw)
        c = raw["compressor"]
        from context_compression.remory_public_release import serving_weight_path
        self.metadata = CompressorCheckpointMetadata(
            checkpoint=checkpoint, state_path=serving_weight_path(checkpoint), raw_config=raw,
            target_layer_ids=tuple(c["target_layer_ids"]), target_hidden_size=c["target_hidden_size"],
            compression_ratio=c["compression_ratio"], local_window=c["local_window"],
            output_embedding_rms=float(c["output_embedding_rms"]),
            context_organization=RESIDUAL_CONTEXT_ORGANIZATION, checkpoint_schema=raw["schema"],
        )
        self.hook_abi_digest = _compute_hook_abi_digest(self.metadata)
        guard = config.get("generation_degeneration_guard_server_abi")
        if guard not in {None, GENERATION_DEGENERATION_GUARD_SERVER_ABI}:
            raise ValueError("unsupported generation degeneration guard ABI")
        sync = config.get("sync_token_ids_across_tp")
        if sync is not None and (sync is not True or os.environ.get("SYNC_TOKEN_IDS_ACROSS_TP") != "1"):
            raise ValueError("residual TP synchronization requires SYNC_TOKEN_IDS_ACROSS_TP=1")
        self.generation_degeneration_guard_server_abi = guard
        self.sync_token_ids_across_tp = sync
        self.gpu_memory_handle_policy = config.get("gpu_memory_handle_policy")
        if self.gpu_memory_handle_policy not in {None, GPU_MEMORY_HANDLE_POLICY}:
            raise ValueError("unsupported GPU memory policy")
        self.gpu_memory_handle_ttl_seconds = float(config.get("gpu_memory_handle_ttl_seconds", 3600))
        self.gpu_memory_handle_max_entries_per_tp_rank = config.get("gpu_memory_handle_max_entries_per_tp_rank", 256)
        self.gpu_memory_handle_max_bytes_per_tp_rank = config.get("gpu_memory_handle_max_bytes_per_tp_rank", 2 << 30)
        if not math.isfinite(self.gpu_memory_handle_ttl_seconds) or self.gpu_memory_handle_ttl_seconds <= 0:
            raise ValueError("invalid memory lease")
        for v in (self.gpu_memory_handle_max_entries_per_tp_rank, self.gpu_memory_handle_max_bytes_per_tp_rank):
            if type(v) is not int or v <= 0:
                raise ValueError("invalid GPU memory capacity")
        self._memory_handle_registry = None
        self._target_model = None
        self.compressor = None

        self._compressor_loader = compressor_loader or _load_summary_residual_compressor

    def _request_payload(self, forward_batch):
        params = getattr(getattr(forward_batch, "sampling_info", None), "custom_params", None)
        if params is None:
            return None
        if not isinstance(params, (list, tuple)):
            raise SpecForgeCompressorContractError("custom_params must be request-aligned")
        marked = [p[CUSTOM_PARAM_KEY] for p in params if isinstance(p, Mapping) and CUSTOM_PARAM_KEY in p]
        if not marked:
            return None
        if all(isinstance(p, Mapping) and p.get("mode") == "recover" for p in marked):
            return super()._request_payload(forward_batch)
        if len(params) != 1 or len(marked) != 1 or not isinstance(marked[0], Mapping):
            raise SpecForgeCompressorContractError("residual encode is a singleton")
        return marked[0]

    def __call__(self, module, inputs, output):
        del module
        if not isinstance(inputs, tuple) or len(inputs) < 4:
            raise SpecForgeCompressorContractError("residual hook belongs on logits_processor")
        forward_batch = inputs[3]
        payload = self._request_payload(forward_batch)
        if payload is None:
            return output
        _, aux = self._validate_encode_forward(inputs, forward_batch, allow_cpu_aux=True)
        try:
            residual = validate_residual_encode_payload(
                payload, token_count=int(inputs[0].numel()), hidden_size=self.metadata.target_hidden_size,
                max_depth=self.residual_contract.max_depth,
            )
        except (ValueError, TypeError) as exc:
            raise SpecForgeCompressorContractError(str(exc)) from exc
        start, end = payload["evidence_spans"][0]
        if residual["operation"] == "summary":
            # The logits processor receives post-final-RMSNorm target states.
            hidden = inputs[1]
            if not isinstance(hidden, torch.Tensor) or hidden.shape != (inputs[0].numel(), self.metadata.target_hidden_size):
                raise SpecForgeCompressorContractError("summary requires complete normalized final target states")
            packed = hidden[start:end]
        else:
            compressor_device = next(self.compressor.parameters()).device
            # Long summaries stay on CPU and are staged by inference-only
            # tiled cross-attention. A tile size is not a summary length cap.
            if "summary_handle" in residual:
                params = forward_batch.sampling_info.custom_params[0]
                req = params.get("__req__")
                if getattr(req, "remory_summary_descriptor", None) != residual["summary_handle"]:
                    raise SpecForgeCompressorContractError("summary memory was not admitted")
                summary = req.remory_summary_tensor_ref[None]
            else:
                summary_device = "cpu" if len(residual["summary_features"]) > 4096 else compressor_device
                summary = torch.as_tensor(residual["summary_features"], dtype=aux[0].dtype,
                                          device=summary_device)[None]
            summary_mask = torch.ones(summary.shape[:2], dtype=torch.bool, device=summary.device)
            block = self.residual_contract.block_tokens
            pieces = []
            with torch.inference_mode():
                for block_index, left in enumerate(range(start, end, block)):
                    right = min(end, left + block)
                    # Captured layers may live on CPU to bound the target's
                    # full-history prefill peak. Stage only this exact block;
                    # never copy all captured layers back to CUDA at once.
                    features = torch.cat([h[left:right] for h in aux], dim=-1).to(
                        device=compressor_device
                    )[None]
                    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                    lengths = torch.tensor([right - left], device=features.device)
                    fixed_slots = torch.tensor([block], device=features.device)
                    encoded = self.compressor(
                        features, mask, source_token_lengths=lengths, slot_token_lengths=fixed_slots,
                        summary_features=summary, summary_attention_mask=summary_mask,
                        input_depth=(residual["block_input_depths"][block_index]
                                     if residual["operation"] == "recursive_leaves"
                                     else residual["input_depth"]),
                    )
                    memory = encoded.embeddings[encoded.attention_mask]
                    if memory.shape != (block // self.metadata.compression_ratio, self.metadata.target_hidden_size):
                        raise SpecForgeCompressorContractError("residual node violated fixed 1024-to-64 geometry")
                    pieces.append(memory)
            packed = torch.cat(pieces)
        if not bool(torch.isfinite(packed).all()):
            raise SpecForgeCompressorContractError("residual encode produced nonfinite states")
        if "memory_handle_publish" in payload:
            from context_compression.remory_memory import transition_digest
            from sglang.srt.remory_memory import collective_publish
            registry = self._memory_handle_registry
            claim = payload["memory_handle_publish"]
            def gather(value):
                if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
                    return [value]
                results = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(results, value)
                return results
            # Token IDs are small metadata; hidden states never leave the worker.
            request_digest = transition_digest(inputs[0].tolist(), payload)
            valid = registry is not None and claim["request_digest"] == request_digest
            if valid:
                valid = claim["memory_identity"] == compute_memory_handle_identity(
                    server_epoch=registry.server_epoch, request_digest=request_digest,
                    hook_abi_digest=self.hook_abi_digest)
            if not all(gather(valid)):
                raise SpecForgeCompressorContractError("residual publication identity mismatch")
            # A summary is a view into full prompt states. Detach its storage.
            tensor = packed.clone() if residual["operation"] == "summary" else packed
            descriptor = collective_publish(registry, tensor=tensor, op_id=claim["op_id"],
                                             request_digest=request_digest, gather=gather)
            output.customized_info = {**(getattr(output, "customized_info", None) or {}),
                                      GPU_MEMORY_HANDLE_META_KEY: [descriptor]}
            output.hidden_states = None
        else:
            output.hidden_states = packed.detach()
        return output


def make_summary_residual_compressor_hook(config):
    # Keep the attested factory ABI; scheduler advertises versions only with
    # this registry switch. Missing version support must fail startup closed.
    import os
    if os.environ.get("REMORY_VERSION_REGISTRY"):
        from remory_version_hook import VersionedResidualHook
        return VersionedResidualHook(config)
    return SummaryResidualSGLangHook(config)
