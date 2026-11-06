"""Pretrained DFlash2 backbone adapted to continuous context compression.

Keep all 78 backbone tensors, including both dynamic convolutions per layer.
The candidate selector scores discrete vocabulary paths and is not a compressor.
Attention retains the established four local reads + one global slot mixer.
Convolution groups are eight consecutive slots, matching the pretrained block.
"""
from pathlib import Path
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3Config

from context_compression.dflash2_grouped_conv import DFlashGroupedConv
from context_compression.grepseek_modeling import GrepSeekDFlashCompressor

TARGET_LAYER_IDS = (5, 19, 33, 47, 61)
SELECTOR_KEYS = {
    'candidate_selector.hidden_projection.weight',
    'candidate_selector.predecessor_codebook',
    'candidate_selector.successor_codebook',
}


class PaddedGroupedConv(DFlashGroupedConv):
    """Right padding supports short final slots without cross-example mixing."""
    def prepare(self, hidden_states):
        length = hidden_states.shape[1]
        padded = F.pad(hidden_states, (0, 0, 0, (-length) % self.block_size))
        value, coefficients = super().prepare(padded)
        return value[:, :length], coefficients

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
            setattr(self, name, PaddedGroupedConv(config.hidden_size, d['block_size'],
                                                 d['conv_kernel_size'], d['conv_group_size']))

    def forward(self, hidden, source, mask, slot_cos, slot_sin, source_cos, source_sin):
        residual = hidden
        hidden, kernel = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = self.self_attn(hidden, source, mask, slot_cos, slot_sin, source_cos, source_sin)
        hidden = residual + self.attention_conv.finish(hidden, kernel)
        residual = hidden
        hidden, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return residual + self.mlp_conv.finish(self.mlp(hidden), kernel)


class DFlash2Compressor(GrepSeekDFlashCompressor):
    def __init__(self, config, *, target_hidden_size, target_layer_ids,
                 compression_ratio=16, local_window=2048, output_embedding_rms=0.02,
                 gradient_checkpointing=True):
        super().__init__(config, target_hidden_size=target_hidden_size,
                         target_layer_ids=target_layer_ids, compression_ratio=compression_ratio,
                         local_window=local_window, output_embedding_rms=output_embedding_rms,
                         gradient_checkpointing=gradient_checkpointing)
        self.layers = nn.ModuleList(DFlash2CompressionLayer(layer, self.config) for layer in self.layers)

    def export_config(self):
        return {**super().export_config(), 'architecture': 'DFlash2Compressor',
                'dflash2': dict(self.config.dflash_config),
                'attention_layout': 'four_local_source_reads_then_global_slot_only',
                'convolution_layout': 'causal_groups_of_8_consecutive_slots_right_padded',
                'candidate_selector': False}


def build_compressor(config_path, *, output_embedding_rms, gradient_checkpointing=True):
    data = json.loads(Path(config_path).read_text())
    if (data['hidden_size'], data['num_hidden_layers'], data['intermediate_size']) != (5120, 5, 17408):
        raise ValueError('unexpected pretrained DFlash2 geometry')
    if tuple(data['dflash_config']['target_layer_ids']) != TARGET_LAYER_IDS:
        raise ValueError('unexpected DFlash2 target feature layers')
    config = Qwen3Config.from_dict(data)
    # Only layout changes: all learned attention/MLP matrices keep their geometry.
    config.layer_types = ['sliding_attention'] * 4 + ['full_attention']
    return DFlash2Compressor(config, target_hidden_size=5120, target_layer_ids=TARGET_LAYER_IDS,
                            output_embedding_rms=output_embedding_rms,
                            gradient_checkpointing=gradient_checkpointing)


def load_pretrained_backbone(model, checkpoint):
    own = model.state_dict()
    new = {'mask_embedding', 'output_embedding_rms'}
    expected = set(own) - new
    loaded = []
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        if keys != expected | SELECTOR_KEYS:
            raise ValueError(f'pretrained keys differ: missing={expected-keys}, extra={keys-expected-SELECTOR_KEYS}')
        with torch.no_grad():
            for name in sorted(expected):
                value = source.get_tensor(name)
                if value.shape != own[name].shape:
                    raise ValueError(f'pretrained shape differs: {name}')
                own[name].copy_(value)
                if not torch.equal(own[name], value.to(own[name])):
                    raise RuntimeError(f'pretrained copy differs: {name}')
                loaded.append({'name': name, 'shape': list(value.shape), 'parameters': value.numel()})
        excluded = [{'name': name, 'shape': list(source.get_slice(name).get_shape())}
                    for name in sorted(SELECTOR_KEYS)]
    return {'all_backbone_tensors_loaded_exactly': True, 'loaded': loaded,
            'loaded_parameters': sum(row['parameters'] for row in loaded),
            'excluded': excluded, 'excluded_reason': 'discrete_token_path_selector_not_used_for_soft_memory',
            'new_parameters': ['mask_embedding'], 'new_buffers': ['output_embedding_rms']}


def load_target_mask_embedding(target_path, token_id):
    """Read one frozen embedding row without loading the 27B model into CPU RAM."""
    root = Path(target_path)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    names = [name for name in index if name.endswith('embed_tokens.weight')]
    if len(names) != 1:
        raise ValueError('ambiguous target token embedding')
    with safe_open(str(root / index[names[0]]), framework='pt', device='cpu') as source:
        return source.get_slice(names[0])[token_id:token_id+1].float().reshape(1, 1, -1)
