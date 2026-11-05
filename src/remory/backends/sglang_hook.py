"""Qwen's model hook: capture source states and scatter soft memory.

The deployment uses one request at a time, a complete prefill, no shared prefix
cache, and no CUDA graphs. Ordinary autoregressive KV caching remains enabled.
"""
from __future__ import annotations

import json
import os
from functools import wraps
import numpy as np
import torch

PARAM = "remory"
HOOK = "remory.backends.sglang_hook:make_hook"


class CarryParameters:
    """SGLang carries custom_params only when a custom processor is present."""

    def __call__(self, logits, custom_param_list=None):
        return logits


def parameter_carrier():
    # A fixed import reference, never user-supplied serialized Python code.
    reference = b"cremory.backends.sglang_hook\nCarryParameters\n."
    return json.dumps({"callable": reference.hex()})


def validate_payload(payload, tokens, width, *, block=1024, max_depth=8):
    if not isinstance(payload, dict) or payload.get("mode") not in {"encode", "recover"}:
        raise ValueError("invalid Remory request")
    positions = payload.get("positions", [])
    if (not isinstance(positions, list) or any(type(i) is not int or not 0 <= i < tokens
                                              for i in positions)
            or positions != sorted(set(positions))):
        raise ValueError("memory positions must be distinct, ordered input positions")
    memory = payload.get("memory")
    if positions:
        a = np.asarray(memory, dtype=np.float32)
        if a.shape != (len(positions), width) or not np.isfinite(a).all():
            raise ValueError("invalid memory embeddings")
    elif memory is not None:
        raise ValueError("memory requires positions")
    if payload["mode"] == "recover":
        if set(payload) - {"mode", "positions", "memory"}:
            raise ValueError("unsupported recovery fields")
        return
    if set(payload) - {"mode", "operation", "start", "end", "summary", "depth",
                       "block_depths", "positions", "memory", "encoder_version"}:
        raise ValueError("unsupported encode fields")
    if "encoder_version" in payload:
        from ..online.common import version_id
        version_id(payload["encoder_version"])
    op, start, end = (payload.get(k) for k in ("operation", "start", "end"))
    if op not in {"summary", "leaves", "recursive_leaves", "parent"}:
        raise ValueError("invalid encode operation")
    if any(type(n) is not int for n in (start, end)) or not 0 <= start < end <= tokens:
        raise ValueError("invalid source span")
    depth = payload.get("depth", 0)
    if type(depth) is not int or not 0 <= depth < max_depth:
        raise ValueError("invalid memory depth")
    if op in {"summary", "leaves", "recursive_leaves"} and depth != 0:
        raise ValueError("raw source requires depth zero")
    if op != "summary":
        summary = np.asarray(payload.get("summary"), dtype=np.float32)
        if (summary.ndim != 2 or summary.shape[1] != width or not len(summary)
                or not np.isfinite(summary).all() or end != tokens):
            raise ValueError("encoding requires a complete source and finite summary states")
    if op in {"summary", "leaves"} and positions:
        raise ValueError("raw source cannot contain memory overrides")
    if op == "parent" and (depth == 0 or end - start != block
                           or positions != list(range(start, end))):
        raise ValueError("parent must cover exactly one block of soft memory")
    if op == "recursive_leaves":
        depths = payload.get("block_depths")
        count = (end - start + block - 1) // block
        if (not positions or any(i < start for i in positions)
                or not isinstance(depths, list) or len(depths) != count
                or any(type(d) is not int or not 0 <= d < max_depth for d in depths)):
            raise ValueError("invalid recursive memory depths")
        soft = {(p - start) // block for p in positions}
        if any((d > 0) != (i in soft) for i, d in enumerate(depths)):
            raise ValueError("recursive depths must identify the blocks containing memory")


class ResidualHook:
    def __init__(self, config):
        self.checkpoint = config["checkpoint"]
        self.compressor = None
        self.payload = None
        self.captured = {}
        self.encoder_root = os.environ.get("REMORY_ENCODER_DIRECTORY")
        self.active_version = None

    def setup_model(self, model):
        from ..models.load import load_compressor
        self.backbone = model.model
        embedding = self.backbone.embed_tokens.weight
        self.compressor, config = load_compressor(
            self.checkpoint, device=embedding.device, dtype=embedding.dtype)
        self.base_config = config
        if self.encoder_root:
            from ..online.common import checkpoint
            self.base_version = checkpoint(self.checkpoint)["id"]
            self.active_version = self.base_version
        self.width = config["compressor"]["target_hidden_size"]
        self.block = config["residual"]["block_tokens"]
        self.ratio = config["residual"]["compression_ratio"]
        if embedding.shape[1] != self.width:
            raise ValueError("actor and memory hidden dimensions differ")
        self.layers = tuple(self.compressor.target_layer_ids)
        if any(i >= len(self.backbone.layers) - 1 for i in self.layers):
            raise ValueError("memory requires interior decoder output layers")
        # ModelRunner calls model.forward directly, bypassing Module.__call__.
        # Wrap that boundary so payload setup/cleanup also runs on that path.
        original_forward = model.forward
        @wraps(original_forward)
        def forward(*args, **kwargs):
            try:
                self.begin(model, args, kwargs)
                return original_forward(*args, **kwargs)
            finally:
                self.end(model, args, None)
        model.forward = forward
        self.backbone.embed_tokens.register_forward_hook(self.scatter)
        for i in self.layers:
            self.backbone.layers[i].register_forward_hook(self.capture(i))

    def begin(self, module, args, kwargs):
        self.payload, self.captured = None, {}
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        params = getattr(getattr(batch, "sampling_info", None), "custom_params", None)
        if not params or not any(isinstance(p, dict) and PARAM in p for p in params):
            return
        if len(params) != 1 or batch.batch_size != 1:
            raise RuntimeError("Remory requires singleton forwards")
        # Decode uses the private KV/recurrent state from the already injected
        # prefill. The absolute input positions do not apply to decode rows.
        if batch.forward_mode.is_decode():
            return
        ids = kwargs.get("input_ids", args[0] if args else None)
        if (not batch.forward_mode.is_extend() or batch.extend_prefix_lens_cpu != [0]
                or batch.extend_seq_lens_cpu != [ids.numel()]):
            raise RuntimeError("Remory requires a complete prefill without prefix reuse")
        payload = params[0][PARAM]
        validate_payload(payload, ids.numel(), self.width, block=self.block,
                         max_depth=self.compressor.max_depth)
        if payload["mode"] == "encode" and not batch.is_prefill_only:
            raise RuntimeError("encoding must be a prefill-only request")
        if payload["mode"] == "encode":
            self.select_encoder(payload.get("encoder_version"))
        self.payload = payload

    def select_encoder(self, version):
        if not self.encoder_root:
            if version is not None:
                raise ValueError("Online checkpoint loading is disabled")
            return
        version = version or self.base_version
        if version == self.active_version:
            return
        from pathlib import Path
        from safetensors.torch import load_file
        from ..online.common import resolve_version
        selected = resolve_version(self.encoder_root, self.checkpoint, version, verify=True)
        # Keep one encoder on the GPU. Generation only scatters saved memory;
        # an encoder is selected independently for every complete encode request.
        weights = load_file(str(Path(selected["path"]) / "model.safetensors"), device="cpu")
        current = self.compressor.state_dict()
        if (set(weights) != set(current)
            or any(weights[k].shape != current[k].shape or not torch.isfinite(weights[k]).all()
                   for k in weights)):
            raise ValueError("Invalid checkpoint tensor names, shapes, or values")
        self.compressor.load_state_dict(weights, strict=True)
        self.active_version = version

    def end(self, module, args, output):
        self.payload, self.captured = None, {}

    def scatter(self, module, args, output):
        p = self.payload
        if p and p.get("positions"):
            output = output.clone()
            output[p["positions"]] = torch.as_tensor(p["memory"], device=output.device,
                                                      dtype=output.dtype)
        return output

    def capture(self, index):
        def save(module, args, output):
            p = self.payload
            if p is None or p["mode"] != "encode" or p["operation"] == "summary":
                return
            # SGLang fuses residual addition into the NEXT layer's norm.
            # Reconstruct the complete decoder output used by HF hidden_states.
            hidden, residual = output
            value = hidden if residual is None else hidden + residual
            self.captured[index] = value[p["start"]:p["end"]].detach().to("cpu", copy=True)
        return save

    @torch.inference_mode()
    def __call__(self, module, args, output):
        p = self.payload
        if p is None or p["mode"] != "encode":
            return output
        if p["operation"] == "summary":
            packed = args[1][p["start"]:p["end"]]
        else:
            if set(self.captured) != set(self.layers):
                raise RuntimeError("actor did not expose all selected source layers")
            weight = next(self.compressor.parameters())
            summary = torch.as_tensor(p["summary"], dtype=weight.dtype)[None]
            summary_mask = torch.ones(summary.shape[:2], dtype=torch.bool)
            pieces = []
            for index, left in enumerate(range(0, p["end"] - p["start"], self.block)):
                right = min(left + self.block, p["end"] - p["start"])
                features = torch.cat([self.captured[i][left:right] for i in self.layers], -1)
                features = features.to(weight)[None]
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=weight.device)
                memory = self.compressor(features, mask,
                    source_token_lengths=torch.tensor([right-left], device=weight.device),
                    slot_token_lengths=torch.tensor([self.block], device=weight.device),
                    summary_features=summary, summary_attention_mask=summary_mask,
                    input_depth=(p["block_depths"][index] if p["operation"] == "recursive_leaves"
                                 else p["depth"]))
                rows = memory.embeddings[memory.attention_mask]
                if rows.shape != (self.block // self.ratio, self.width):
                    raise RuntimeError("compressor returned unexpected memory geometry")
                pieces.append(rows)
            packed = torch.cat(pieces)
        if not torch.isfinite(packed).all():
            raise RuntimeError("nonfinite residual encoding")
        output.hidden_states = packed.detach()
        output.remory_packed_hidden_states = True
        return output


def make_hook(config):
    return ResidualHook(config)
