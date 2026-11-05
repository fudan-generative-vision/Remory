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


from context_compression.qwen38_residual_modeling import SummaryCrossAttention
from context_compression.qwen38_dflash2_compressor import DFlash2Compressor, TARGET_LAYER_IDS


class DFlash2SummaryResidualCompressor(DFlash2Compressor):
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
        return {**super().export_config(), "architecture": "DFlash2SummaryResidualCompressor", "max_depth": self.max_depth,
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
            residual = hidden
            hidden, kernel = layer.attention_conv.prepare(layer.input_layernorm(hidden))
            hidden = layer.self_attn(hidden, source, layout.local_keep_mask,
                                     slot_cos, slot_sin, source_cos, source_sin)
            hidden = residual + layer.attention_conv.finish(hidden, kernel)
            hidden = cross(hidden, summary, mask, stream_summary=stream_summary)
            residual = hidden
            hidden, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(hidden))
            return residual + layer.mlp_conv.finish(layer.mlp(hidden), kernel)

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


def build_residual(config_path, *, output_embedding_rms, gradient_checkpointing=False):
    import json
    from pathlib import Path
    from transformers import Qwen3Config
    data = json.loads(Path(config_path).read_text())
    if (data['hidden_size'], data['num_hidden_layers'], data['intermediate_size']) != (5120, 5, 17408):
        raise ValueError('unexpected pretrained DFlash2 geometry')
    if tuple(data['dflash_config']['target_layer_ids']) != TARGET_LAYER_IDS:
        raise ValueError('unexpected DFlash2 feature layers')
    config = Qwen3Config.from_dict(data)
    config.layer_types = ['sliding_attention'] * 4 + ['full_attention']
    return DFlash2SummaryResidualCompressor(config, target_hidden_size=5120,
        target_layer_ids=TARGET_LAYER_IDS, output_embedding_rms=output_embedding_rms,
        gradient_checkpointing=gradient_checkpointing)


def load_pretrained_residual(model, checkpoint, target_path):
    from safetensors import safe_open
    from context_compression.qwen38_dflash2_compressor import SELECTOR_KEYS, load_target_mask_embedding
    own = model.state_dict()
    expected = {k for k in own if not model.is_new_parameter(k)} - {'mask_embedding', 'output_embedding_rms'}
    loaded = []
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        if keys != expected | SELECTOR_KEYS:
            raise ValueError(f'pretrained keys differ: missing={expected-keys}, extra={keys-expected-SELECTOR_KEYS}')
        with torch.no_grad():
            for name in sorted(expected):
                value = source.get_tensor(name)
                if value.shape != own[name].shape:
                    raise ValueError(f'pretrained shape mismatch: {name}')
                own[name].copy_(value)
                if not torch.equal(own[name], value.to(own[name])):
                    raise RuntimeError(f'pretrained copy mismatch: {name}')
                loaded.append(dict(name=name, parameters=value.numel(), shape=list(value.shape)))
            mask = load_target_mask_embedding(target_path, 248070)
            model.mask_embedding.copy_(mask)
            if not torch.equal(model.mask_embedding, mask.to(model.mask_embedding)):
                raise RuntimeError('target mask embedding copy mismatch')
    return dict(all_backbone_tensors_loaded_exactly=True, loaded=loaded,
        loaded_parameters=sum(x['parameters'] for x in loaded),
        excluded=sorted(SELECTOR_KEYS), mask_embedding_equal_target=True,
        new_residual_parameters=[k for k in own if model.is_new_parameter(k)],
        stage1_checkpoint_used=False)


def load_stage1_residual(model, checkpoint):
    """Transfer the complete reconstruction backbone; keep new residual modules fresh."""
    from pathlib import Path
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    result = model.load_repeat_state(state)
    own = model.state_dict()
    loaded = []
    for name, value in state.items():
        if not torch.isfinite(value).all() or not torch.equal(own[name], value.to(own[name])):
            raise ValueError(f"Stage I tensor transfer failed: {name}")
        loaded.append(dict(name=name, parameters=value.numel(), shape=list(value.shape)))
    return dict(stage1_checkpoint_used=True, source_checkpoint=str(checkpoint),
        all_stage1_tensors_loaded_exactly=True, loaded=loaded,
        loaded_parameters=sum(value.numel() for name, value in state.items()
                              if name != "output_embedding_rms"),
        mask_embedding_copied_from_stage1=True,
        new_residual_parameters=sorted(result.missing_keys))
