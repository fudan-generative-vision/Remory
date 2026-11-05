"""Summary-conditioned residual compressor; repeat backbone keys stay unchanged."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from context_compression.grepseek_modeling import (
    CompressedContextOutput,
    GrepSeekDFlashCompressor,
    build_local_compression_layout,
)


class SummaryCrossAttention(nn.Module):
    """Slots query detached summary states, independently of source/slot attention."""

    def __init__(self, config, gate_init=1e-3):
        super().__init__()
        d, h, k = config.hidden_size, config.num_attention_heads, config.num_key_value_heads
        self.heads, self.kv_heads, self.head_dim = h, k, config.head_dim
        self.query_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.summary_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(d, h * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, k * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, k * self.head_dim, bias=False)
        self.o_proj = nn.Linear(h * self.head_dim, d, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # FSDP does not support scalar (zero-dimensional) parameters.
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, slots, summary, mask, *, stream_summary=False, chunk_tokens=1024):
        b, n, _ = slots.shape
        s = summary.shape[1]
        q = self.q_proj(self.query_norm(slots)).view(b, n, self.heads, self.head_dim)
        if stream_summary:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("streaming summary attention is inference-only")
            if type(chunk_tokens) is not int or chunk_tokens < 1:
                raise ValueError("summary attention tile must be positive")
            q = self.q_norm(q).transpose(1, 2)
            maximum = torch.full((*q.shape[:-1], 1), -torch.inf, device=q.device, dtype=torch.float32)
            normalizer = torch.zeros_like(maximum)
            numerator = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
            # Exact online softmax over every summary row: no independent
            # chunk softmax, no dropped tokens, no changed checkpoint weights.
            for start in range(0, s, chunk_tokens):
                stop = min(s, start + chunk_tokens)
                part = self.summary_norm(summary[:, start:stop].detach().to(slots))
                k = self.k_proj(part).view(b, stop - start, self.kv_heads, self.head_dim)
                v = self.v_proj(part).view(b, stop - start, self.kv_heads, self.head_dim)
                k = self.k_norm(k).transpose(1, 2).repeat_interleave(self.heads // self.kv_heads, 1)
                v = v.transpose(1, 2).repeat_interleave(self.heads // self.kv_heads, 1)
                scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
                scores.masked_fill_(~mask[:, None, None, start:stop].to(q.device), -torch.inf)
                updated = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
                safe_max = torch.where(torch.isfinite(updated), updated, torch.zeros_like(updated))
                rescale = torch.exp(maximum - safe_max)
                probabilities = torch.exp(scores - safe_max)
                numerator = numerator * rescale + torch.matmul(probabilities, v.float())
                normalizer = normalizer * rescale + probabilities.sum(dim=-1, keepdim=True)
                maximum = updated
            value = (numerator / normalizer.clamp_min(torch.finfo(torch.float32).tiny)).to(q.dtype)
            value = value.transpose(1, 2).reshape(b, n, -1)
            return slots + self.gate.tanh() * self.o_proj(value)
        summary = self.summary_norm(summary.detach())
        k = self.k_proj(summary).view(b, s, self.kv_heads, self.head_dim)
        v = self.v_proj(summary).view(b, s, self.kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2).repeat_interleave(self.heads // self.kv_heads, 1)
        v = v.transpose(1, 2).repeat_interleave(self.heads // self.kv_heads, 1)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                              dropout_p=0.0, is_causal=False)
        value = value.transpose(1, 2).reshape(b, n, -1)
        return slots + self.gate.tanh() * self.o_proj(value)


class SummaryResidualCompressor(GrepSeekDFlashCompressor):
    def __init__(self, *args, max_depth=8, summary_gate_init=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        if max_depth < 1 or not math.isfinite(summary_gate_init):
            raise ValueError("invalid residual depth/gate")
        if float(getattr(self.config, "attention_dropout", 0)) != 0:
            raise ValueError("detached frontier VJP replay requires dropout=0")
        self.max_depth = int(max_depth)
        self.summary_gate_init = float(summary_gate_init)
        self.summary_attention = nn.ModuleList([
            SummaryCrossAttention(self.config, summary_gate_init) for _ in self.layers[:-1]
        ])
        self.summary_attention.apply(
            lambda module: self._initialize_module(module, self.config.initializer_range)
        )
        # Depth denotes INPUT depth: raw=0, first-generation soft tokens=1, ...
        self.depth_embedding = nn.Embedding(max_depth, self.config.hidden_size)
        nn.init.zeros_(self.depth_embedding.weight)

    @staticmethod
    def is_new_parameter(name):
        return name.startswith(("summary_attention.", "depth_embedding."))

    def load_repeat_state(self, state):
        expected = {k for k in self.state_dict() if not self.is_new_parameter(k)}
        if set(state) != expected:
            raise ValueError(f"repeat state keys differ: missing={expected-set(state)}, extra={set(state)-expected}")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not self.is_new_parameter(k) for k in result.missing_keys):
            raise ValueError("non-residual parameters missing from repeat initialization")
        return result

    def export_config(self):
        return {**super().export_config(), "max_depth": self.max_depth,
                "summary_gate_init": self.summary_gate_init,
                "summary_conditioning": "independent_slot_Q_summary_KV_after_source_attention",
                "depth_coordinate": "input_depth_raw_is_zero"}

    def forward(self, source_features, source_attention_mask, source_token_lengths=None,
                slot_token_lengths=None, *, summary_features, summary_attention_mask,
                input_depth=0):
        if not isinstance(input_depth, int) or not 0 <= input_depth < self.max_depth:
            raise ValueError("input depth exceeds configured residual pyramid")
        width = len(self.target_layer_ids) * self.target_hidden_size
        if source_features.ndim != 3 or source_features.shape[-1] != width:
            raise ValueError("invalid residual source features")
        if source_attention_mask.shape != source_features.shape[:2]:
            raise ValueError("source mask shape mismatch")
        if (summary_features.ndim != 3 or summary_features.shape[0] != source_features.shape[0]
                or summary_features.shape[-1] != self.target_hidden_size
                or summary_attention_mask.shape != summary_features.shape[:2]
                or summary_attention_mask.dtype != torch.bool
                or not bool(summary_attention_mask.any(-1).all())):
            raise ValueError("each source needs a nonempty summary with a boolean mask")
        features = source_features.detach().to(self.fc.weight)
        stream_summary = (not self.training and not torch.is_grad_enabled()
                          and summary_features.device.type == "cpu" and summary_features.shape[1] > 4096)
        summary = (summary_features.detach().to(dtype=self.fc.weight.dtype) if stream_summary
                   else summary_features.detach().to(self.fc.weight))
        mask = summary_attention_mask.to(summary.device)
        layout = build_local_compression_layout(
            source_attention_mask, compression_ratio=self.compression_ratio,
            local_window=self.local_window, source_token_lengths=source_token_lengths,
            slot_token_lengths=slot_token_lengths,
        )
        b, slots = layout.slot_attention_mask.shape
        if not slots:
            return CompressedContextOutput(features.new_zeros(b, 0, self.target_hidden_size),
                                           layout.slot_attention_mask, layout.slot_counts)
        source = self.hidden_norm(self.fc(features))
        hidden = self.mask_embedding.expand(b, slots, -1) + self.depth_embedding.weight[input_depth]
        slot_cos, slot_sin = self.rotary_emb(hidden, layout.slot_position_ids)
        source_cos, source_sin = self.rotary_emb(source, layout.source_position_ids)
        rows = layout.slot_attention_mask.unsqueeze(-1)

        def local_layer(hidden, source, summary, layer, cross):
            hidden = hidden + layer.self_attn(layer.input_layernorm(hidden), source,
                layout.local_keep_mask, slot_cos, slot_sin, source_cos, source_sin)
            hidden = cross(hidden, summary, mask, stream_summary=stream_summary)
            return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        for layer, cross in zip(self.layers[:-1], self.summary_attention):
            # Bind modules as arguments, never late-bind loop variables in a checkpoint closure.
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(local_layer, hidden, source, summary, layer, cross,
                                    use_reentrant=False)
            else:
                hidden = local_layer(hidden, source, summary, layer, cross)
            hidden = hidden * rows
        final_args = (hidden, None, layout.global_keep_mask, slot_cos, slot_sin, None, None)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            hidden = checkpoint(self.layers[-1], *final_args, use_reentrant=False)
        else:
            hidden = self.layers[-1](*final_args)
        embeddings = self.norm(hidden * rows) * self.output_embedding_rms.to(hidden) * rows
        return CompressedContextOutput(embeddings, layout.slot_attention_mask, layout.slot_counts)
