# Adapted from Recursive Memory and SpecForge (MIT); see NOTICES.md.
"""DFlash2 residual memory inference with the released grouped convolutions."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .grouped_conv import DFlashGroupedConv
from .layers import CompressedContextOutput, GrepSeekDFlashCompressor, build_local_compression_layout
from .compressor import SummaryCrossAttention

class PaddedGroupedConv(DFlashGroupedConv):
    """Right padding supports short final slots without cross-example mixing."""

    def prepare(self, hidden_states):
        length = hidden_states.shape[1]
        padded = F.pad(hidden_states, (0, 0, 0, -length % self.block_size))
        value, coefficients = super().prepare(padded)
        return (value[:, :length], coefficients)

    def finish(self, hidden_states, coefficients):
        length = hidden_states.shape[1]
        padded = F.pad(hidden_states, (0, 0, 0, coefficients.shape[1] - length))
        return super().finish(padded, coefficients)[:, :length]

class DFlash2CompressionLayer(nn.Module):

    def __init__(self, layer, config):
        super().__init__()
        for name in ('self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'):
            setattr(self, name, getattr(layer, name))
        d = config.dflash_config
        for name in ('attention_conv', 'mlp_conv'):
            setattr(self, name, PaddedGroupedConv(config.hidden_size, d['block_size'], d['conv_kernel_size'], d['conv_group_size']))

    def forward(self, hidden, source, mask, slot_cos, slot_sin, source_cos, source_sin):
        residual = hidden
        hidden, kernel = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = self.self_attn(hidden, source, mask, slot_cos, slot_sin, source_cos, source_sin)
        hidden = residual + self.attention_conv.finish(hidden, kernel)
        residual = hidden
        hidden, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return residual + self.mlp_conv.finish(self.mlp(hidden), kernel)

class DFlash2Compressor(GrepSeekDFlashCompressor):

    def __init__(self, config, *, target_hidden_size, target_layer_ids, compression_ratio=16, local_window=2048, output_embedding_rms=0.02, gradient_checkpointing=True):
        super().__init__(config, target_hidden_size=target_hidden_size, target_layer_ids=target_layer_ids, compression_ratio=compression_ratio, local_window=local_window, output_embedding_rms=output_embedding_rms, gradient_checkpointing=gradient_checkpointing)
        self.layers = nn.ModuleList((DFlash2CompressionLayer(layer, self.config) for layer in self.layers))

class DFlash2SummaryResidualCompressor(DFlash2Compressor):

    def __init__(self, *args, max_depth=8, summary_gate_init=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        if max_depth < 1 or not math.isfinite(summary_gate_init):
            raise ValueError('invalid residual depth/gate')
        if float(getattr(self.config, 'attention_dropout', 0)) != 0:
            raise ValueError('detached frontier VJP replay requires dropout=0')
        self.max_depth = int(max_depth)
        self.summary_gate_init = float(summary_gate_init)
        self.summary_attention = nn.ModuleList([SummaryCrossAttention(self.config, summary_gate_init) for _ in self.layers[:-1]])
        self.summary_attention.apply(lambda module: self._initialize_module(module, self.config.initializer_range))
        self.depth_embedding = nn.Embedding(max_depth, self.config.hidden_size)
        nn.init.zeros_(self.depth_embedding.weight)

    def forward(self, source_features, source_attention_mask, source_token_lengths=None, slot_token_lengths=None, *, summary_features, summary_attention_mask, input_depth=0):
        if not isinstance(input_depth, int) or not 0 <= input_depth < self.max_depth:
            raise ValueError('input depth exceeds configured residual pyramid')
        width = len(self.target_layer_ids) * self.target_hidden_size
        if source_features.ndim != 3 or source_features.shape[-1] != width:
            raise ValueError('invalid residual source features')
        if source_attention_mask.shape != source_features.shape[:2]:
            raise ValueError('source mask shape mismatch')
        if summary_features.ndim != 3 or summary_features.shape[0] != source_features.shape[0] or summary_features.shape[-1] != self.target_hidden_size or (summary_attention_mask.shape != summary_features.shape[:2]) or (summary_attention_mask.dtype != torch.bool) or (not bool(summary_attention_mask.any(-1).all())):
            raise ValueError('each source needs a nonempty summary with a boolean mask')
        features = source_features.detach().to(self.fc.weight)
        stream_summary = not self.training and (not torch.is_grad_enabled()) and (summary_features.device.type == 'cpu') and (summary_features.shape[1] > 4096)
        summary = summary_features.detach().to(dtype=self.fc.weight.dtype) if stream_summary else summary_features.detach().to(self.fc.weight)
        mask = summary_attention_mask.to(summary.device)
        layout = build_local_compression_layout(source_attention_mask, compression_ratio=self.compression_ratio, local_window=self.local_window, source_token_lengths=source_token_lengths, slot_token_lengths=slot_token_lengths)
        b, slots = layout.slot_attention_mask.shape
        if not slots:
            return CompressedContextOutput(features.new_zeros(b, 0, self.target_hidden_size), layout.slot_attention_mask, layout.slot_counts)
        source = self.hidden_norm(self.fc(features))
        hidden = self.mask_embedding.expand(b, slots, -1) + self.depth_embedding.weight[input_depth]
        slot_cos, slot_sin = self.rotary_emb(hidden, layout.slot_position_ids)
        source_cos, source_sin = self.rotary_emb(source, layout.source_position_ids)
        rows = layout.slot_attention_mask.unsqueeze(-1)

        def local_layer(hidden, source, summary, layer, cross):
            residual = hidden
            hidden, kernel = layer.attention_conv.prepare(layer.input_layernorm(hidden))
            hidden = layer.self_attn(hidden, source, layout.local_keep_mask, slot_cos, slot_sin, source_cos, source_sin)
            hidden = residual + layer.attention_conv.finish(hidden, kernel)
            hidden = cross(hidden, summary, mask, stream_summary=stream_summary)
            residual = hidden
            hidden, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(hidden))
            return residual + layer.mlp_conv.finish(layer.mlp(hidden), kernel)
        for layer, cross in zip(self.layers[:-1], self.summary_attention):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(local_layer, hidden, source, summary, layer, cross, use_reentrant=False)
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
