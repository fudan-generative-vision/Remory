# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""

import logging
from functools import lru_cache
from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import triton

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)

# Configs
from sglang.srt.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5TextConfig,
)

# Distributed
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

# Layers - Attention
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)

# Layers - Others
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType

# Layers - Linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    general_mm_embed_routine as _dflash_general_mm_embed_routine,
)
from sglang.srt.mem_cache.soft_memory_cache_contract import (
    plan_packed_soft_memory_rows,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock

# Models
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.server_args import get_global_server_args

# Utils
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_npu,
    make_layers,
    set_weight_attrs,
)
from sglang.srt.utils.hf_transformers_utils import get_processor, get_rope_config

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_gfx95 = is_gfx95_supported()
_is_amx_available = cpu_has_amx_support()


cached_get_processor = lru_cache(get_processor)


GDN_ACTIVATION_LIFETIME_CAPABILITY = "qwen38_gdn_early_activation_release_v1"


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkvz", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_ba", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        # Override weight loaders for packed checkpoint format.
        # Important: for FP8, this must cover not only `.weight` but also
        # `weight_scale_inv` / `weight_scale` / `input_scale` if present.
        self._bind_packed_weight_loaders(self.in_proj_qkvz)
        self._bind_packed_weight_loaders(self.in_proj_ba)

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self._override_weight_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ],
                self.attn_tp_size,
                self.attn_tp_rank,
            ),
        )

        # State parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

    @staticmethod
    def _override_weight_loader(param, loader):
        """Robustly override loader for:
        1) BasevLLMParameter subclasses: real storage is `_weight_loader`
        2) regular Parameters that already have mutable `weight_loader`
        3) regular Parameters without `weight_loader` yet
        """
        if hasattr(param, "_weight_loader"):
            # FP8 / quantized BasevLLMParameter path
            param._weight_loader = loader
            return

        if hasattr(param, "weight_loader"):
            # Regular parameter/tensor that already has a mutable attr.
            # Do NOT call set_weight_attrs here, because it asserts when
            # overwriting an existing attribute.
            param.weight_loader = loader
            return

        # Fresh attribute on a normal tensor/Parameter
        set_weight_attrs(param, {"weight_loader": loader})

    def _bind_packed_weight_loaders(self, module):
        """Bind packed-checkpoint-aware loaders to all relevant params of a merged module."""
        for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale"):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            wrapped_loader = self._make_packed_weight_loader(module, original_loader)
            self._override_weight_loader(param, wrapped_loader)

    @staticmethod
    def _get_split_sizes_for_param(module, param, loaded_shard_id):
        """Return checkpoint-side split sizes for this param type."""
        if isinstance(param, BlockQuantScaleParameter):
            # Split by output blocks, not raw output sizes.
            block_n, _ = module.quant_method.quant_config.weight_block_size
            block_n = 1 if getattr(param, "format_ue8m0", False) else block_n
            return [
                (module.output_sizes[idx] + block_n - 1) // block_n
                for idx in loaded_shard_id
            ]

        if isinstance(param, PerTensorScaleParameter):
            # One logical scale per logical shard.
            return [1 for _ in loaded_shard_id]

        # Normal weight / non-block quant tensor
        return [module.output_sizes[idx] for idx in loaded_shard_id]

    @classmethod
    def _make_packed_weight_loader(cls, module, original_weight_loader):
        """Wrap the param's original loader so split checkpoints:
          - in_proj_qkv + in_proj_z -> merged in_proj_qkvz
          - in_proj_b + in_proj_a   -> merged in_proj_ba
        can load correctly for both normal and FP8 params.
        """

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            # Only intercept split-checkpoint tuple shards.
            # int shard_id and None should preserve original behavior.
            if isinstance(loaded_shard_id, tuple):
                split_sizes = cls._get_split_sizes_for_param(
                    module, param, loaded_shard_id
                )

                if len(loaded_weight.shape) == 0:
                    # Scalar only makes sense for a single logical shard.
                    assert len(split_sizes) == 1 and split_sizes[0] == 1, (
                        f"Unexpected scalar for tuple shard load: "
                        f"{loaded_shard_id=}, {split_sizes=}"
                    )
                    chunks = [loaded_weight.reshape(1)]
                else:
                    split_dim = getattr(param, "output_dim", 0)
                    chunks = loaded_weight.split(split_sizes, dim=split_dim)

                assert len(chunks) == len(loaded_shard_id), (
                    f"Chunk/shard mismatch: {len(chunks)=}, "
                    f"{len(loaded_shard_id)=}, {split_sizes=}"
                )

                for idx, chunk in zip(loaded_shard_id, chunks):
                    # Delegate each chunk to the param's original int-shard loader.
                    original_weight_loader(param, chunk, idx)
                return

            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        k_tp = self.key_dim // self.attn_tp_size
        v_tp = self.value_dim // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size

        # Directly split, no head group reshape
        query, key, value, z = mixed_qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
        b, a = mixed_ba.split([nv_tp, nv_tp], dim=-1)

        # value / z reshape to (seq, num_v_heads/tp, head_v_dim)
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        return query, key, value, z, b, a

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        if (
            _is_cpu
            or _is_npu
            or not get_global_server_args().disable_piecewise_cuda_graph
        ):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        seq_len, _ = hidden_states.shape
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_qkvz, projected_states_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        # _forward_input_proj can use an alternate CUDA stream during graph
        # capture below 1024 rows. Preserve those short/decode lifetimes; only
        # release inference-prefill temporaries on the single-stream path.
        release_prefill_buffers = (
            hidden_states.shape[0] >= 1024 and not torch.is_grad_enabled()
        )
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(
            hidden_states
        )

        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        elif _is_cpu and _is_amx_available:
            mixed_qkv, z, b, a = (
                torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_cpu(
                    projected_states_qkvz,
                    projected_states_ba,
                    self.num_k_heads // self.attn_tp_size,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            query, key, value = map(
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
            if release_prefill_buffers:
                del query, key, value

        # Fused split owns its outputs. Drop the full-history projection buffers
        # before attention; in the fallback path z/a/b views retain their bases
        # until the last consumer naturally releases them. This is important for
        # unchunked long prefills, not just the bounded GDN kernel workspace.
        if release_prefill_buffers:
            del projected_states_qkvz, projected_states_ba
        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )
        if release_prefill_buffers:
            del mixed_qkv, a, b

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        # Add padding for DP-Attn
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad
            if release_prefill_buffers:
                del core_attn_out_pad

        core_attn_out = self.norm(core_attn_out, z)
        # Gate and QKV storage must not coexist with the next full-history
        # hidden-width output allocation. No values/attention states change.
        if release_prefill_buffers:
            del z
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        linear_attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, layer_id, linear_attn_quant_config, alt_stream, prefix
        )

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
                is_nextn=is_nextn,
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )
        else:
            hidden_states = self.mlp(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        # If rope_scaling doesn't specify a scaling type, treat as no scaling
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=attn_quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=attn_quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        elif config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
                is_nextn=is_nextn,
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Q/K normalization with optional alt_stream overlap."""
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Full attention forward pass."""
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )
        else:
            hidden_states = self.mlp(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Model with support for dense variant."""

    if _is_gfx95:
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        layer_types = getattr(config, "layers_block_type", None)
        if layer_types is None:
            # Native HF Qwen3.5 text configs expose ``layer_types`` and spell
            # full-attention blocks as ``full_attention``.  The multimodal
            # SGLang config exposes the equivalent computed
            # ``layers_block_type`` list with ``attention`` values.
            layer_types = getattr(config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != int(
            config.num_hidden_layers
        ):
            raise ValueError(
                "Qwen3.5 layer types must contain exactly num_hidden_layers entries"
            )
        layer_types = [
            "attention" if layer_type == "full_attention" else layer_type
            for layer_type in layer_types
        ]
        invalid_layer_types = sorted(set(layer_types) - set(ALL_DECODER_LAYER_TYPES))
        if invalid_layer_types:
            raise ValueError(
                f"unsupported Qwen3.5 layer types: {invalid_layer_types}"
            )

        # Embedding layer
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        def get_layer(idx: int, prefix: str):
            layer_type = layer_types[idx]
            layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
            if layer_type == "attention":
                prefix = add_prefix("self_attn", prefix)
            else:
                prefix = add_prefix("linear_attn", prefix)
            return layer_class(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )

        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self):
        return self.embed_tokens

    @property
    def start_layer(self) -> int:
        return self._start_layer

    @property
    def end_layer(self) -> int:
        return self._end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # Initialize hidden states
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # Pass through decoder layers
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )

            # Process deepstack embeddings if provided
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        # Return intermediate tensors for pipeline parallelism
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        # Apply final normalization
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # HF Qwen3.5 MoE always renormalizes the selected top-k router
        # probabilities, but its config intentionally has no
        # ``norm_topk_prob`` field.  The reused Qwen2 MoE block expects that
        # compatibility field, so inject the architecture-defined value while
        # rejecting a conflicting override instead of silently changing router
        # semantics.
        norm_topk_prob = getattr(config, "norm_topk_prob", None)
        if norm_topk_prob is not None and norm_topk_prob is not True:
            raise ValueError(
                "Qwen3.5 MoE requires top-k router probability renormalization"
            )
        config.norm_topk_prob = True
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        return loaded_params


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    if _is_gfx95:
        packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
        hf_to_sglang_mapper = None

    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ):
        super().__init__(config, quant_config, prefix, language_model_cls)

        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    @property
    def start_layer(self) -> int:
        return getattr(getattr(self, "model", None), "start_layer", 0)

    @property
    def end_layer(self) -> int:
        model = getattr(self, "model", None)
        end_layer = getattr(model, "end_layer", None)
        if end_layer is not None:
            return end_layer
        cfg = getattr(model, "config", None)
        return int(getattr(cfg, "num_hidden_layers", 0))

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "visual" in name or "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                # print(name, loaded_weight.shape)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""

    if _is_gfx95:
        packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
        hf_to_sglang_mapper = None

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            "_weight_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if name.endswith("experts.gate_up_proj") or name.endswith(
                    "experts.down_proj"
                ):
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )

        return loaded_params

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


# ``Qwen3_5ForCausalLM`` above is the language-tower backbone used by the
# multimodal wrappers.  Hugging Face's text-only Qwen3.5 checkpoints use the
# same architecture string for the complete causal LM, however, so preserve
# the backbone under an explicit internal name and provide a complete SGLang
# entry class below.  Giving the entry class the HF architecture name at the
# end of the definition lets the registry select it without changing existing
# multimodal construction.
_Qwen3_5TextBackbone = Qwen3_5ForCausalLM
_Qwen3_5MoeTextBackbone = Qwen3_5MoeForCausalLM


class Qwen3_5TextForCausalLM(nn.Module):
    """Complete text-only Qwen3.5 causal LM backed by the native SGLang tower."""

    backbone_cls = _Qwen3_5TextBackbone
    packed_modules_mapping = getattr(
        _Qwen3_5TextBackbone, "packed_modules_mapping", {}
    )

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self.backbone_cls(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_dflash_layers_to_capture(self, layer_ids):
        return _dflash_qwen35_text_set_layers_to_capture(self, layer_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        return _dflash_qwen35_text_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_params: Set[str] = set()
        lm_head = self.lm_head

        def backbone_weights():
            for name, loaded_weight in weights:
                if name == "lm_head.weight":
                    if not self.pp_group.is_last_rank:
                        continue
                    weight_loader = getattr(
                        lm_head.weight, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head.weight, loaded_weight)
                    loaded_params.add(name)
                    continue
                # Some text-only snapshots retain the tensor names from the
                # original multimodal shard files even though their published
                # index/config expose a text causal LM.  Normalize the actual
                # safetensors key, not merely the index key.
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                if name.startswith("model."):
                    name = name[len("model.") :]
                yield name, loaded_weight

        for name in self.model.load_weights(backbone_weights()):
            loaded_params.add(add_prefix(name, "model"))
        return loaded_params


class Qwen3_5MoeTextForCausalLM(Qwen3_5TextForCausalLM):
    """Complete text-only Qwen3.5 MoE causal LM."""

    backbone_cls = _Qwen3_5MoeTextBackbone
    packed_modules_mapping = getattr(
        _Qwen3_5MoeTextBackbone, "packed_modules_mapping", {}
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_params = super().load_weights(weights)
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )
        return loaded_params

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return _Qwen3_5MoeTextBackbone.get_model_config_for_expert_location(config)


# ModelRegistry keys entry classes by ``__name__``.  The implementation has a
# distinct Python symbol to avoid shadowing the language-tower backbone above.
Qwen3_5TextForCausalLM.__name__ = "Qwen3_5ForCausalLM"
Qwen3_5MoeTextForCausalLM.__name__ = "Qwen3_5MoeForCausalLM"


EntryClass = [
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextForCausalLM,
    Qwen3_5MoeTextForCausalLM,
]

# ================= DFLASH_QWEN35_BACKPORT_MARKER BEGIN =================
# Backport DFlash aux-hidden-state capture support for Qwen3.5 on older SGLang.
# This patch avoids pulling newer PR23000/CUDA13 dependencies.


_SPECFORGE_CONTEXT_COMPRESSOR_KEY = "specforge_context_compressor"
_SPECFORGE_CONTEXT_COMPRESSOR_VERSION = 1
_SPECFORGE_CONTEXT_COMPRESSOR_MODES = frozenset({"encode", "recover"})
_SPECFORGE_RESIDUAL_MLP_CHUNK_TOKENS = 4096


def _dflash_qwen35_need_aux_capture(forward_batch) -> bool:
    capture_hidden_mode = getattr(forward_batch, "capture_hidden_mode", None)
    return capture_hidden_mode is not None and capture_hidden_mode.need_capture()


def _dflash_qwen35_get_context_compressor_configs(forward_batch):
    """Return a request-aligned compressor config list.

    Compressor encode remains a strict singleton.  Recover performs no hook
    work and may be aligned with other recover or ordinary text requests.
    """
    sampling_info = getattr(forward_batch, "sampling_info", None)
    custom_params = getattr(sampling_info, "custom_params", None)
    if custom_params is None:
        return None
    if not isinstance(custom_params, (list, tuple)):
        raise TypeError("sampling_info.custom_params must be a request-aligned list")
    batch_size = int(getattr(forward_batch, "batch_size", -1))
    if batch_size <= 0 or len(custom_params) != batch_size:
        raise ValueError(
            "sampling_info.custom_params must match forward_batch.batch_size"
        )
    configs = []
    marked_indices = []
    encode_indices = []
    for index, params in enumerate(custom_params):
        if not isinstance(params, dict) or (
            _SPECFORGE_CONTEXT_COMPRESSOR_KEY not in params
        ):
            configs.append(None)
            continue
        config = params[_SPECFORGE_CONTEXT_COMPRESSOR_KEY]
        if not isinstance(config, dict):
            raise TypeError(
                f"{_SPECFORGE_CONTEXT_COMPRESSOR_KEY} must be a dictionary"
            )
        version = config.get("version")
        if version != _SPECFORGE_CONTEXT_COMPRESSOR_VERSION:
            raise ValueError(
                "SpecForge context compressor version must be "
                f"{_SPECFORGE_CONTEXT_COMPRESSOR_VERSION}, got {version!r}"
            )
        mode = config.get("mode")
        if (
            not isinstance(mode, str)
            or mode not in _SPECFORGE_CONTEXT_COMPRESSOR_MODES
        ):
            raise ValueError(
                "SpecForge context compressor mode must be one of "
                f"{sorted(_SPECFORGE_CONTEXT_COMPRESSOR_MODES)}, got {mode!r}"
            )
        configs.append(config)
        marked_indices.append(index)
        if mode == "encode":
            encode_indices.append(index)
    if not marked_indices:
        return None
    if encode_indices and (
        batch_size != 1 or marked_indices != [0] or encode_indices != [0]
    ):
        raise ValueError("SpecForge compressor encode requires batch_size=1")
    return configs


def _dflash_qwen35_get_context_compressor_config(forward_batch):
    """Compatibility view retained for batch-one tests and integrations."""

    configs = _dflash_qwen35_get_context_compressor_configs(forward_batch)
    if configs is None:
        return None
    if len(configs) != 1:
        raise ValueError("single compressor config view requires batch_size=1")
    return configs[0]


def _dflash_qwen35_is_residual_encode(forward_batch):
    configs = _dflash_qwen35_get_context_compressor_configs(forward_batch)
    config = configs[0] if configs is not None and len(configs) == 1 else None
    residual = config.get("residual") if config is not None else None
    return (
        config is not None
        and config.get("mode") == "encode"
        and isinstance(residual, dict)
        and residual.get("abi") == "qwen38_summary_residual_sglang_v1"
    )


def _dflash_qwen35_dense_mlp_forward(
    mlp, hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
):
    """Bound dense inference-prefill MLP temporaries without splitting history.

    Residual leaf encoding intentionally prefills the complete original history
    and captures its causal target features.  A dense MLP is token-local, so its
    gate/up/activation/down work can be tiled without changing attention, KV or
    recurrent state, capture boundaries, or the 1024-to-64 compressor geometry.
    Generation, summary, recovery and packed batches need the same bound.
    One-token decode, gradient-enabled execution and sparse MoE are unchanged.
    """
    chunk_tokens = _SPECFORGE_RESIDUAL_MLP_CHUNK_TOKENS
    chunked = (
        isinstance(mlp, Qwen2MoeMLP)
        and forward_batch.forward_mode.is_extend()
        and hidden_states.ndim == 2
        and hidden_states.shape[0] > chunk_tokens
        and not torch.is_grad_enabled()
    )
    if not chunked:
        return mlp(hidden_states, should_allreduce_fusion, use_reduce_scatter)

    # Keep at most one chunk of wide intermediates live.  Do not collect chunk
    # outputs in a list and cat: that would double the full output allocation.
    # Preserve the original down-projection collective flags on every TP rank;
    # fused/deferred reductions still receive a single assembled output tensor.
    output = None
    for left in range(0, hidden_states.shape[0], chunk_tokens):
        right = min(left + chunk_tokens, hidden_states.shape[0])
        piece = mlp(
            hidden_states[left:right], should_allreduce_fusion, use_reduce_scatter
        )
        if output is None:
            output = piece.new_empty((hidden_states.shape[0], *piece.shape[1:]))
        output[left:right].copy_(piece)
        del piece
    return output


def _dflash_qwen35_scatter_memory_embeddings(
    input_embeds, config, *, protected_token_mask=None
):
    """Scatter memory at server-expanded packed-EXTEND token coordinates."""

    if not isinstance(input_embeds, torch.Tensor):
        raise TypeError("target input embeddings must be a torch.Tensor")
    if input_embeds.ndim != 2:
        raise ValueError("target input embeddings must have shape [tokens, hidden]")
    if not input_embeds.is_floating_point():
        raise TypeError("target input embeddings must use a floating-point dtype")

    if "memory_positions" not in config:
        raise ValueError("context compressor memory override requires memory_positions")
    has_memory_embeddings = "memory_embeddings" in config
    has_memory_handle = "memory_handle" in config
    if has_memory_embeddings == has_memory_handle:
        raise ValueError(
            "context compressor memory override requires exactly one of "
            "memory_embeddings or memory_handle"
        )

    try:
        memory_positions = torch.as_tensor(
            config["memory_positions"], device=input_embeds.device
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("memory_positions must be tensor-convertible") from exc
    if memory_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("memory_positions must use int32 or int64")
    if memory_positions.ndim == 1:
        token_positions = memory_positions.to(dtype=torch.long)
    elif memory_positions.ndim == 2 and memory_positions.shape[1] == 2:
        if memory_positions.numel() and bool(
            (memory_positions[:, 0] != 0).any().item()
        ):
            raise ValueError("batch coordinates in memory_positions must all be zero")
        token_positions = memory_positions[:, 1].to(dtype=torch.long)
    else:
        raise ValueError("memory_positions must have shape [N] or [N, 2]")

    if has_memory_handle:
        try:
            from sglang.srt.memory_handle_registry import (
                get_global_memory_handle_registry,
            )

            resolved_memory = get_global_memory_handle_registry().resolve(
                config["memory_handle"]
            )
            memory_embeddings = resolved_memory.tensor
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"memory_handle resolution failed: {exc}") from exc
        if memory_embeddings.device != input_embeds.device:
            raise ValueError(
                "resolved memory_handle tensor is not local to the target GPU"
            )
    else:
        try:
            memory_embeddings = torch.as_tensor(
                config["memory_embeddings"], device=input_embeds.device
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("memory_embeddings must be tensor-convertible") from exc
    if not memory_embeddings.is_floating_point():
        raise TypeError("memory_embeddings must use a floating-point dtype")
    if memory_embeddings.ndim != 2:
        raise ValueError("memory_embeddings must have shape [N, hidden]")
    if memory_embeddings.shape[0] != token_positions.numel():
        raise ValueError(
            "memory_positions and memory_embeddings must contain the same number "
            "of rows"
        )
    if memory_embeddings.shape[1] != input_embeds.shape[1]:
        raise ValueError(
            "memory_embeddings hidden size must match target input embeddings"
        )
    if not bool(torch.isfinite(memory_embeddings).all().item()):
        raise ValueError("memory_embeddings must contain only finite values")

    if not token_positions.numel():
        raise ValueError("memory_positions must not be empty")
    if bool(
        ((token_positions < 0) | (token_positions >= input_embeds.shape[0]))
        .any()
        .item()
    ):
        raise ValueError(f"memory_positions contains an out-of-range position: rows={input_embeds.shape[0]}, min={int(token_positions.min())}, max={int(token_positions.max())}")
    if token_positions.numel() > 1 and bool(
        (token_positions[1:] <= token_positions[:-1]).any().item()
    ):
        raise ValueError("memory_positions must be unique and strictly increasing")

    if protected_token_mask is not None:
        if not isinstance(protected_token_mask, torch.Tensor):
            raise TypeError("protected_token_mask must be a torch.Tensor")
        if protected_token_mask.dtype != torch.bool:
            raise TypeError("protected_token_mask must use boolean dtype")
        if protected_token_mask.ndim != 1 or protected_token_mask.shape[0] != (
            input_embeds.shape[0]
        ):
            raise ValueError(
                "protected_token_mask must have shape [tokens] matching input embeddings"
            )
        if bool(protected_token_mask[token_positions].any().item()):
            raise ValueError(
                "memory_positions must not overlap a multimodal placeholder or "
                "boundary span"
            )

    scattered = input_embeds.clone()
    scattered[token_positions] = memory_embeddings.to(dtype=scattered.dtype)
    return scattered


def _dflash_qwen35_request_lengths(value, *, batch_size, name):
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a request-aligned sequence")
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain exactly {batch_size} values")
    result = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name}[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative")
        result.append(int(item))
    return result


def _dflash_qwen35_memory_positions(config):
    raw = config.get("memory_positions")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("memory_positions must be a nonempty integer list")
    positions = []
    previous = -1
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"memory_positions[{index}] must be an integer")
        value = int(value)
        if value <= previous:
            raise ValueError("memory_positions must be unique and strictly increasing")
        positions.append(value)
        previous = value
    return positions


def _dflash_qwen35_resolve_memory_rows(input_embeds, config, *, request=None):
    has_embeddings = "memory_embeddings" in config
    has_handle = "memory_handle" in config
    if has_embeddings == has_handle:
        raise ValueError(
            "memory override requires exactly one of memory_embeddings or memory_handle"
        )
    if has_handle:
        memory_embeddings = getattr(request, "memory_handle_tensor_ref", None)
        if memory_embeddings is None:
            from sglang.srt.memory_handle_registry import (
                get_global_memory_handle_registry,
            )

            memory_embeddings = get_global_memory_handle_registry().resolve(
                config["memory_handle"]
            ).tensor
        if memory_embeddings.device != input_embeds.device:
            raise ValueError("resolved memory handle is not local to the target GPU")
    else:
        memory_embeddings = torch.as_tensor(
            config["memory_embeddings"], device=input_embeds.device
        )
    if (
        not memory_embeddings.is_floating_point()
        or memory_embeddings.ndim != 2
        or not bool(torch.isfinite(memory_embeddings).all().item())
    ):
        raise ValueError("memory rows must be a finite rank-two floating tensor")
    if memory_embeddings.shape[1] != input_embeds.shape[1]:
        raise ValueError("memory hidden size must match target input embeddings")
    return memory_embeddings


def _dflash_qwen35_scatter_batched_memory_embeddings(
    input_embeds, configs, forward_batch
):
    """Scatter only uncached memory rows into packed multi-request EXTEND."""

    if not isinstance(input_embeds, torch.Tensor) or input_embeds.ndim != 2:
        raise ValueError("packed target input embeddings must have shape [tokens, hidden]")
    batch_size = int(getattr(forward_batch, "batch_size", -1))
    if batch_size <= 0 or len(configs) != batch_size:
        raise ValueError("compressor configs must be request-aligned")
    prefix_lens = _dflash_qwen35_request_lengths(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        batch_size=batch_size,
        name="extend_prefix_lens_cpu",
    )
    extend_lens = _dflash_qwen35_request_lengths(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        batch_size=batch_size,
        name="extend_seq_lens_cpu",
    )
    if sum(extend_lens) != int(input_embeds.shape[0]):
        # Preserve the historical batch-one wrapper contract used by local
        # direct-forward integrations that did not populate EXTEND metadata.
        # Real scheduler batches, and every concurrent batch, remain strict.
        if batch_size == 1 and prefix_lens == [0]:
            extend_lens = [int(input_embeds.shape[0])]
        else:
            raise ValueError("packed EXTEND lengths disagree with input embeddings")
    positions_per_request = [
        None
        if config is None or "memory_positions" not in config
        else _dflash_qwen35_memory_positions(config)
        for config in configs
    ]
    custom_params = getattr(
        getattr(forward_batch, "sampling_info", None), "custom_params", None
    )
    # Chunked EXTEND may stop before later soft rows. Validate against the
    # admitted full request, and scatter only rows in this physical chunk.
    total_lens = []
    for index, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
        params = custom_params[index] if isinstance(custom_params, (list, tuple)) else None
        req = params.get("__req__") if isinstance(params, dict) else None
        total_lens.append(len(req.origin_input_ids) + len(req.output_ids) if req is not None else prefix + length)
    plans = plan_packed_soft_memory_rows(
        memory_positions_per_request=positions_per_request,
        extend_prefix_lens=prefix_lens,
        extend_seq_lens=extend_lens,
        request_total_lens=total_lens,
    )
    scattered = input_embeds
    copied = False
    for plan in plans:
        if not plan.embedding_row_indices:
            continue
        config = configs[plan.request_index]
        assert config is not None
        request = None
        if isinstance(custom_params, (list, tuple)):
            params = custom_params[plan.request_index]
            if isinstance(params, dict):
                request = params.get("__req__")
        memory_rows = _dflash_qwen35_resolve_memory_rows(
            input_embeds, config, request=request
        )
        positions = positions_per_request[plan.request_index]
        assert positions is not None
        if memory_rows.shape[0] != len(positions):
            raise ValueError("memory positions and rows must have equal length")
        if not copied:
            scattered = input_embeds.clone()
            copied = True
        row_indices = torch.tensor(
            plan.embedding_row_indices,
            dtype=torch.long,
            device=memory_rows.device,
        )
        packed_positions = torch.tensor(
            plan.packed_token_positions,
            dtype=torch.long,
            device=scattered.device,
        )
        scattered[packed_positions] = memory_rows.index_select(
            0, row_indices
        ).to(dtype=scattered.dtype)
    return scattered


def _dflash_qwen35_single_cpu_length(value, *, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one request length")
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one request length")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must contain an integer request length")
    return int(value)


def _dflash_qwen35_multimodal_placeholder_mask(input_ids, forward_batch):
    """Return multimodal placeholder and adjacent boundary rows to protect.

    SpecForge continuous memories are intentionally cache-free.  Multimodal item
    offsets use full-request coordinates, while the model sees packed EXTEND
    rows.  In SGLang, ``extend_prefix_lens_cpu`` counts cached rows and
    ``extend_seq_lens_cpu`` counts the new rows packed into ``input_ids``.
    Accepting a nonzero cached prefix would make the declared server-expanded
    memory positions ambiguous and, more importantly, could reuse KV/hybrid
    state built without those continuous vectors.  The serving client therefore
    flushes radix/hybrid caches before every continuous-memory request.
    """

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError("multimodal memory override requires rank-one input_ids")
    prefix_length = _dflash_qwen35_single_cpu_length(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        name="extend_prefix_lens_cpu",
    )
    extend_length = _dflash_qwen35_single_cpu_length(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        name="extend_seq_lens_cpu",
    )
    if prefix_length != 0 or extend_length != int(input_ids.numel()):
        raise ValueError(
            "multimodal memory override requires one complete, cache-free EXTEND; "
            "flush SGLang radix/hybrid caches before sending continuous memory"
        )

    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if not isinstance(mm_inputs, (list, tuple)) or len(mm_inputs) != 1:
        raise ValueError(
            "multimodal memory override requires one request-aligned multimodal input"
        )
    mm_input = mm_inputs[0]
    mm_items = None if mm_input is None else getattr(mm_input, "mm_items", None)
    if not isinstance(mm_items, (list, tuple)) or not mm_items:
        raise ValueError("multimodal memory override requires multimodal item offsets")

    protected = torch.zeros(
        int(input_ids.numel()), dtype=torch.bool, device=input_ids.device
    )
    observed_offset = False
    for item in mm_items:
        offsets = getattr(item, "offsets", None)
        if not isinstance(offsets, (list, tuple)) or not offsets:
            raise ValueError(
                "multimodal memory override requires every item to expose offsets"
            )
        for offset in offsets:
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError("multimodal item offsets must contain [start, end]")
            start, end = offset
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end >= extend_length
            ):
                raise ValueError(
                    "multimodal item offsets must be valid inclusive token ranges"
                )
            # Qwen's processor records the inclusive image/video/audio
            # placeholder range.  Its immediately adjacent rows are the native
            # vision/audio start and end structure tokens; replacing either with
            # a continuous-memory vector would leave an invalid multimodal
            # envelope even though the feature rows themselves stayed intact.
            protected[max(0, start - 1) : min(extend_length, end + 2)] = True
            observed_offset = True
    if not observed_offset or not bool(protected.any().item()):
        raise ValueError(
            "multimodal memory override found no multimodal placeholder span"
        )
    return protected


class _DFlashQwen35MemoryScatterLanguageModel:
    """Compose native multimodal embeddings with SpecForge soft memory.

    ``general_mm_embed_routine`` remains the sole owner of vision encoding,
    placeholder replacement, deepstack construction, feature offload, and CUDA
    graph buffer selection.  This narrow proxy intercepts its final language
    model call, after all of that native work, and replaces only declared,
    non-multimodal text rows with continuous memory. ``memory_positions`` must
    already use the server-expanded packed-EXTEND coordinate space.
    """

    def __init__(
        self,
        language_model,
        *,
        config,
        protected_token_mask,
    ):
        self._language_model = language_model
        self._config = config
        self._protected_token_mask = protected_token_mask
        self.pp_group = language_model.pp_group

    def get_input_embeddings(self):
        return self._language_model.get_input_embeddings()

    def __call__(self, *, input_ids, forward_batch, input_embeds, **kwargs):
        if input_embeds is not None:
            input_embeds = _dflash_qwen35_scatter_memory_embeddings(
                input_embeds,
                self._config,
                protected_token_mask=self._protected_token_mask,
            )
            stable_input_embeds = getattr(forward_batch, "input_embeds", None)
            if stable_input_embeds is not None:
                if stable_input_embeds.shape != input_embeds.shape:
                    raise ValueError(
                        "preallocated input embedding buffer shape changed during memory scatter"
                    )
                stable_input_embeds.copy_(input_embeds)
                input_embeds = stable_input_embeds
            # The native routine records the pre-transform multimodal tensor
            # before invoking the language model.  Keep that request-local
            # receipt consistent with the actual post-scatter tensor.
            forward_batch.mm_input_embeds = input_embeds
        return self._language_model(
            input_ids=input_ids,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )


def _dflash_qwen35_validate_public_input_embeds(
    input_ids, input_embeds, expected_hidden_size
):
    if not isinstance(input_embeds, torch.Tensor):
        raise TypeError("input_embeds must be a torch.Tensor")
    if input_embeds.ndim != 2:
        raise ValueError("input_embeds must have shape [tokens, hidden]")
    if not input_embeds.is_floating_point():
        raise TypeError("input_embeds must use a floating-point dtype")
    if input_ids is not None and input_embeds.shape[0] != input_ids.numel():
        raise ValueError("input_embeds token count must match flattened input_ids")
    if input_embeds.shape[1] != int(expected_hidden_size):
        raise ValueError("input_embeds hidden size does not match the target model")


def _dflash_prepare_attn_and_capture_last_layer_outputs(
    self,
    hidden_states,
    residual,
    forward_batch,
    captured_last_layer_outputs=None,
    post_residual_addition=None,
):
    hidden_states, residual = self.prepare_attn(
        hidden_states,
        residual,
        forward_batch,
        post_residual_addition=post_residual_addition,
    )

    if captured_last_layer_outputs is not None:
        gathered_last_layer_output = self._communicate_simple_fn(
            hidden_states=residual,
            forward_batch=forward_batch,
            context=self._context,
        )
        if _dflash_qwen35_is_residual_encode(forward_batch):
            # Keep exact causal features, but do not retain several full-history
            # CUDA tensors while later attention/MLP layers allocate workspace.
            # This must be a completed copy before residual can be reused in
            # place. The residual hook stages only a 1024-token feature block
            # back to the compressor's device. No source tokens are discarded.
            gathered_last_layer_output = gathered_last_layer_output.to(
                device="cpu", copy=True, non_blocking=False
            )
            forward_batch.specforge_residual_aux_cpu = True
        elif gathered_last_layer_output is residual:
            # Avoid in-place RMSNorm / residual reuse modifying captured states.
            gathered_last_layer_output = residual.clone()
        captured_last_layer_outputs.append(gathered_last_layer_output)

    return hidden_states, residual


LayerCommunicator.prepare_attn_and_capture_last_layer_outputs = (
    _dflash_prepare_attn_and_capture_last_layer_outputs
)


def _dflash_qwen35_linear_decoder_forward(self, hidden_states, residual, **kwargs):
    forward_batch = kwargs.get("forward_batch", None)

    hidden_states, residual = (
        self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states,
            residual,
            forward_batch,
            captured_last_layer_outputs=kwargs.get("captured_last_layer_outputs", None),
        )
    )

    if not forward_batch.forward_mode.is_idle():
        hidden_states = self.linear_attn(
            hidden_states,
            forward_batch,
        )

    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )

    use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
        forward_batch
    )

    should_allreduce_fusion = (
        self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)
    )

    if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        )
    else:
        hidden_states = _dflash_qwen35_dense_mlp_forward(
            self.mlp, hidden_states, forward_batch,
            should_allreduce_fusion, use_reduce_scatter
        )

    if should_allreduce_fusion:
        hidden_states._sglang_needs_allreduce_fusion = True
    else:
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

    return hidden_states, residual


def _dflash_qwen35_attention_decoder_forward(
    self,
    positions,
    hidden_states,
    residual,
    forward_batch,
    **kwargs,
):
    hidden_states, residual = (
        self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states,
            residual,
            forward_batch,
            captured_last_layer_outputs=kwargs.get("captured_last_layer_outputs", None),
        )
    )

    if not forward_batch.forward_mode.is_idle():
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )

    use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
        forward_batch
    )

    should_allreduce_fusion = (
        self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)
    )

    if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        )
    else:
        hidden_states = _dflash_qwen35_dense_mlp_forward(
            self.mlp, hidden_states, forward_batch,
            should_allreduce_fusion, use_reduce_scatter
        )

    if should_allreduce_fusion:
        hidden_states._sglang_needs_allreduce_fusion = True
    else:
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

    return hidden_states, residual


def _dflash_qwen35_model_set_layers_to_capture(self, layers_to_capture):
    self.layers_to_capture = [int(x) for x in layers_to_capture]

    for layer_id in self.layers_to_capture:
        # PP-safe guard. For pp_size=1, this simply sets the requested layers.
        if self.start_layer <= layer_id < self.end_layer:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)


@torch.no_grad()
def _dflash_qwen35_model_forward(
    self,
    input_ids,
    positions,
    forward_batch,
    input_embeds=None,
    pp_proxy_tensors=None,
    input_deepstack_embeds=None,
):
    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    should_capture_aux = bool(getattr(self, "layers_to_capture", [])) and (
        _dflash_qwen35_need_aux_capture(forward_batch)
    )
    aux_hidden_states = [] if should_capture_aux else None

    for layer_idx in range(self.start_layer, self.end_layer):
        layer = self.layers[layer_idx]
        with get_global_expert_distribution_recorder().with_current_layer(layer_idx):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                captured_last_layer_outputs=(
                    aux_hidden_states
                    if should_capture_aux
                    and getattr(layer, "_is_layer_to_capture", False)
                    else None
                ),
            )

        # Keep existing Qwen3.5 deepstack behavior.
        if (
            input_deepstack_embeds is not None
            and input_deepstack_embeds.numel() > 0
            and layer_idx < 3
        ):
            sep = self.hidden_size * layer_idx
            hidden_states.add_(input_deepstack_embeds[:, sep : sep + self.hidden_size])

    if not self.pp_group.is_last_rank:
        return PPProxyTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )

    if hidden_states.shape[0] != 0:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

    if aux_hidden_states is None:
        return hidden_states

    return hidden_states, aux_hidden_states


def _dflash_qwen35_conditional_set_layers_to_capture(self, layer_ids):
    if not self.pp_group.is_last_rank:
        return

    if layer_ids is None:
        raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")

    self.capture_aux_hidden_states = True

    # Match newer SGLang behavior: capture output of previous target layer
    # by marking layer_id + 1.
    self.model.set_dflash_layers_to_capture([int(val) + 1 for val in layer_ids])


def _dflash_qwen35_text_set_layers_to_capture(self, layer_ids):
    if not self.pp_group.is_last_rank:
        return
    if layer_ids is None:
        raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")

    self.capture_aux_hidden_states = True
    # Match the HF hidden-state tuple used during compressor training: capture
    # the residual stream immediately before ``layer_id + 1``.
    self.model.set_dflash_layers_to_capture([int(val) + 1 for val in layer_ids])


@torch.no_grad()
def _dflash_qwen35_conditional_forward(
    self,
    input_ids,
    positions,
    forward_batch,
    get_embedding=False,
    pp_proxy_tensors=None,
    input_embeds=None,
):
    if self.is_mrope_enabled:
        positions = forward_batch.mrope_positions

    if not (
        forward_batch.forward_mode.is_decode()
        or not forward_batch.contains_image_inputs()
    ):
        if self.is_mrope_enabled:
            assert positions.ndim == 2 and positions.size(0) == 3, (
                "multimodal section rotary embedding requires "
                f"(3, seq_len) positions, but got {positions.size()}"
            )

    context_compressor_configs = _dflash_qwen35_get_context_compressor_configs(
        forward_batch
    )
    has_recover = bool(context_compressor_configs) and any(
        config is not None and config.get("mode") == "recover"
        for config in context_compressor_configs
    )
    has_encode_memory = bool(context_compressor_configs) and any(
        config is not None
        and config.get("mode") == "encode"
        and "memory_positions" in config
        for config in context_compressor_configs
    )
    has_memory_override = has_recover or has_encode_memory

    if input_embeds is not None:
        if forward_batch.contains_mm_inputs():
            raise ValueError(
                "public input_embeds is only supported for text-only input"
            )
        if has_recover:
            raise ValueError("recover mode cannot be combined with public input_embeds")
        _dflash_qwen35_validate_public_input_embeds(
            input_ids, input_embeds, self.model.hidden_size
        )
        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
    elif has_memory_override:
        if forward_batch.forward_mode.is_decode():
            if has_encode_memory:
                raise ValueError("encode memory override is prefill-only")
            # The soft memories have already been committed to KV during EXTEND.
            # Decode must use only the newly sampled token embedding.
            hidden_states = _dflash_general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                use_deepstack=self.use_deepstack,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode != ForwardMode.EXTEND:
            raise ValueError(
                "context compressor memory override requires EXTEND or recover DECODE"
            )
        elif forward_batch.contains_mm_inputs():
            if len(context_compressor_configs) != 1:
                raise ValueError(
                    "multimodal memory override remains a batch-one legacy ABI"
                )
            context_compressor_config = context_compressor_configs[0]
            assert context_compressor_config is not None
            protected_token_mask = _dflash_qwen35_multimodal_placeholder_mask(
                input_ids, forward_batch
            )
            memory_scatter_model = _DFlashQwen35MemoryScatterLanguageModel(
                self.model,
                config=context_compressor_config,
                protected_token_mask=protected_token_mask,
            )
            hidden_states = _dflash_general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=memory_scatter_model,
                multimodal_model=self,
                positions=positions,
                use_deepstack=self.use_deepstack,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            if self.pp_group.is_first_rank:
                target_input_embeds = self.model.get_input_embeddings()(input_ids)
                target_input_embeds = (
                    _dflash_qwen35_scatter_batched_memory_embeddings(
                        target_input_embeds,
                        context_compressor_configs,
                        forward_batch,
                    )
                )
            else:
                target_input_embeds = None
            hidden_states = self.model(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=target_input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
    else:
        # Includes ordinary requests and first-transition SpecForge encode
        # requests without previous memory. Keep the existing multimodal/
        # DFlash embedding behavior for both.
        hidden_states = _dflash_general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    aux_hidden_states = None
    if getattr(self, "capture_aux_hidden_states", False) and (
        _dflash_qwen35_need_aux_capture(forward_batch)
    ):
        hidden_states, aux_hidden_states = hidden_states

    if self.pp_group.is_last_rank:
        if not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return self.pooler(hidden_states, forward_batch)
    else:
        return hidden_states


@torch.no_grad()
def _dflash_qwen35_text_forward(
    self,
    input_ids,
    positions,
    forward_batch,
    input_embeds=None,
    get_embedding=False,
    pp_proxy_tensors=None,
):
    """Text-only forward with the same SpecForge soft-memory ABI as VL."""

    context_compressor_configs = _dflash_qwen35_get_context_compressor_configs(
        forward_batch
    )
    has_recover = bool(context_compressor_configs) and any(
        config is not None and config.get("mode") == "recover"
        for config in context_compressor_configs
    )
    has_encode_memory = bool(context_compressor_configs) and any(
        config is not None
        and config.get("mode") == "encode"
        and "memory_positions" in config
        for config in context_compressor_configs
    )
    has_memory_override = has_recover or has_encode_memory

    if input_embeds is not None:
        if has_recover:
            raise ValueError("recover mode cannot be combined with public input_embeds")
        _dflash_qwen35_validate_public_input_embeds(
            input_ids, input_embeds, self.model.hidden_size
        )
        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
    elif has_memory_override:
        if forward_batch.forward_mode.is_decode():
            if has_encode_memory:
                raise ValueError("encode memory override is prefill-only")
            # The soft rows were committed to KV during EXTEND.  Decode only
            # embeds the newly sampled discrete token.
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode != ForwardMode.EXTEND:
            raise ValueError(
                "context compressor memory override requires EXTEND or recover DECODE"
            )
        else:
            if self.pp_group.is_first_rank:
                target_input_embeds = self.model.get_input_embeddings()(input_ids)
                target_input_embeds = (
                    _dflash_qwen35_scatter_batched_memory_embeddings(
                        target_input_embeds,
                        context_compressor_configs,
                        forward_batch,
                    )
                )
            else:
                target_input_embeds = None
            hidden_states = self.model(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=target_input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
    else:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    aux_hidden_states = None
    if self.capture_aux_hidden_states and _dflash_qwen35_need_aux_capture(
        forward_batch
    ):
        hidden_states, aux_hidden_states = hidden_states

    if self.pp_group.is_last_rank:
        if not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        return self.pooler(hidden_states, forward_batch)
    return hidden_states


Qwen3_5LinearDecoderLayer.forward = _dflash_qwen35_linear_decoder_forward
Qwen3_5AttentionDecoderLayer.forward = _dflash_qwen35_attention_decoder_forward

Qwen3_5ForCausalLM.set_dflash_layers_to_capture = (
    _dflash_qwen35_model_set_layers_to_capture
)
Qwen3_5ForCausalLM.forward = _dflash_qwen35_model_forward

Qwen3_5ForConditionalGeneration.set_dflash_layers_to_capture = (
    _dflash_qwen35_conditional_set_layers_to_capture
)
Qwen3_5ForConditionalGeneration.forward = _dflash_qwen35_conditional_forward

# ================= DFLASH_QWEN35_BACKPORT_MARKER END =================
