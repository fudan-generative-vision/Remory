# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or a full LM head; DFlash uses the target
# model's embedding/lm_head. Experimental checkpoints may add an unmerged
# low-rank residual to that shared head.

from __future__ import annotations

import logging
from copy import deepcopy
from collections.abc import Mapping
from typing import Iterable, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)


def _validate_dflash_parameter_load_coverage(
    parameter_names: Iterable[str],
    loaded_param_parts: Mapping[str, set[object]],
) -> None:
    """Require every runtime parameter and every fused checkpoint shard."""

    missing: list[str] = []
    incomplete: list[str] = []
    for param_name in sorted(parameter_names):
        parts = loaded_param_parts.get(param_name)
        if not parts:
            missing.append(param_name)
            continue
        if "full" in parts:
            continue
        required_parts: Optional[set[object]] = None
        if ".qkv_proj." in param_name:
            required_parts = {"q", "k", "v"}
        elif ".gate_up_proj." in param_name:
            required_parts = {0, 1}
        if required_parts is None or not required_parts.issubset(parts):
            incomplete.append(
                f"{param_name} shards={sorted(str(part) for part in parts)}"
            )

    if missing or incomplete:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if incomplete:
            details.append(f"incomplete={incomplete}")
        raise ValueError(
            "DFLASH checkpoint did not completely initialize the runtime model. "
            "Refusing to serve uninitialized or partially fused parameters: "
            + "; ".join(details)
        )


def resolve_dflash_rope_config(config) -> tuple[float, Optional[dict]]:
    """Resolve both Transformers v4 and v5 RoPE config schemas.

    Transformers v5 stores ``rope_theta`` inside ``rope_parameters`` and may
    no longer expose the legacy top-level attribute.  Silently using the
    fallback theta here changes every cached context key and breaks numerical
    parity with the HF/SpecForge training implementation.
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        rope_scaling = dict(rope_parameters)
        rope_theta = rope_scaling.get("rope_theta", 1_000_000.0)
    else:
        rope_theta = getattr(config, "rope_theta", 1_000_000.0)
        legacy_scaling = getattr(config, "rope_scaling", None)
        rope_scaling = (
            dict(legacy_scaling)
            if isinstance(legacy_scaling, Mapping)
            else legacy_scaling
        )
    return float(rope_theta), rope_scaling


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = resolve_dflash_rope_config(config)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        # DFlash uses non-causal attention over the draft block.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Use a minimal dummy query (1 head) to avoid doing full-Q work.
        dummy_q = k.new_empty((k.shape[0], self.head_dim))
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / full lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
      - optional `head_lora_{a,b}.weight` for an unmerged draft-only residual
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        layer_id_offset: int = 0,
        _allow_joint_judge: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layer_id_offset = int(layer_id_offset)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(config=config, layer_id=self.layer_id_offset + i)
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)
        self.head_lora_rank = int(draft_config.head_lora_rank)
        self.head_lora_alpha = float(draft_config.head_lora_alpha)
        self.head_lora_a = None
        self.head_lora_b = None
        if self.head_lora_rank > 0:
            self.head_lora_a = ReplicatedLinear(
                hidden_size,
                self.head_lora_rank,
                bias=False,
                quant_config=None,
                prefix="head_lora_a",
            )
            # B follows the exact padded vocab partition used by ParallelLMHead.
            self.head_lora_b = ParallelLMHead(
                int(config.vocab_size),
                self.head_lora_rank,
                bias=False,
                quant_config=None,
                prefix="head_lora_b",
            )

        self.joint_judge_enabled = bool(
            draft_config.joint_judge_enabled and _allow_joint_judge
        )
        self.has_multiple_dflash_kv_groups = self.joint_judge_enabled
        self.joint_judge_head_hidden_size = int(
            draft_config.joint_judge_head_hidden_size
        )
        self.joint_judge_prior = float(draft_config.joint_judge_prior)
        self.joint_judge_model: Optional[DFlashDraftModel] = None
        self.joint_judge_head: Optional[nn.Module] = None
        if self.joint_judge_enabled:
            judge_config = deepcopy(config)
            judge_dflash_config = dict(
                getattr(judge_config, "dflash_config", {}) or {}
            )
            # Match SpecForge's checkpoint ABI: the judge is exactly one
            # independent native DFlash tower, without recursively creating
            # generator-side experimental modules.
            for key in (
                "joint_judge_enabled",
                "accept_head_enabled",
                "accept_head_hidden_size",
                "accept_head_prior",
                "head_lora_rank",
                "head_lora_alpha",
                "mixture_copula_rank",
                "mixture_copula_dim",
                "state_copula_rank",
                "state_copula_dim",
                "kd_potential_dim",
                "sparse_drafter",
            ):
                judge_dflash_config.pop(key, None)
            judge_dflash_config["joint_judge_enabled"] = False
            judge_config.dflash_config = judge_dflash_config
            self.joint_judge_model = DFlashDraftModel(
                config=judge_config,
                quant_config=quant_config,
                prefix=(
                    f"{prefix}.joint_judge_model"
                    if prefix
                    else "joint_judge_model"
                ),
                layer_id_offset=self.layer_id_offset + num_layers,
                _allow_joint_judge=False,
            )
            if self.joint_judge_head_hidden_size == 0:
                self.joint_judge_head = nn.Linear(hidden_size, 1)
            else:
                self.joint_judge_head = nn.Sequential(
                    nn.Linear(
                        hidden_size,
                        self.joint_judge_head_hidden_size,
                        bias=False,
                    ),
                    nn.SiLU(),
                    nn.Linear(self.joint_judge_head_hidden_size, 1),
                )

            # ModelRunner sizes the KV pool from these bounds.  The generator
            # owns layers [0, L), while the judge owns [L, 2L).
            self.start_layer = self.layer_id_offset
            self.end_layer = self.layer_id_offset + 2 * num_layers
            # The existing fused materializer assumes one projection/tower.
            # The generic path below deliberately materializes both towers.
            self.supports_fused_kv_materialization = False

    @property
    def head_lora_scaling(self) -> float:
        if self.head_lora_rank <= 0:
            return 0.0
        return self.head_lora_alpha / float(self.head_lora_rank)

    def apply_head_lora_to_local_logits(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        output_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """Apply the local vocab shard of B(Ah) without merging either head."""
        if self.head_lora_a is None or self.head_lora_b is None:
            return base_logits
        low_rank, _ = self.head_lora_a(hidden_states)
        b_weight = self.head_lora_b.weight
        if output_slice is not None:
            b_weight = b_weight[output_slice]
        if int(base_logits.shape[-1]) != int(b_weight.shape[0]):
            raise RuntimeError(
                "DFLASH Head-LoRA local vocab shape mismatch: "
                f"base_logits={tuple(base_logits.shape)}, B={tuple(b_weight.shape)}"
            )
        if low_rank.dtype != b_weight.dtype:
            low_rank = low_rank.to(b_weight.dtype)
        return torch.addmm(
            base_logits,
            low_rank,
            b_weight.T,
            beta=1.0,
            alpha=self.head_lora_scaling,
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def iter_dflash_kv_groups(
        self, target_hidden: torch.Tensor
    ) -> Iterator[Tuple[Iterable[DFlashDecoderLayer], torch.Tensor]]:
        """Yield every independent tower that needs target-context KV."""
        yield self.layers, self.project_target_hidden(target_hidden)
        if self.joint_judge_enabled:
            assert self.joint_judge_model is not None
            yield (
                self.joint_judge_model.layers,
                self.joint_judge_model.project_target_hidden(target_hidden),
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def joint_judge_forward(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the isomorphic judge and return one conditional logit per token."""
        if self.joint_judge_model is None or self.joint_judge_head is None:
            raise RuntimeError(
                "This DFlash checkpoint does not contain an enabled joint judge."
            )
        judge_output = self.joint_judge_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        judge_hidden = judge_output.hidden_states
        if judge_hidden is None:
            raise RuntimeError("Joint DFlash judge returned no hidden states.")
        return self.joint_judge_head(judge_hidden).squeeze(-1)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        require_complete: bool = True,
    ) -> set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        loaded_params = set()
        loaded_param_parts: dict[str, set[object]] = {}
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(resolved_name)
                loaded_param_parts.setdefault(resolved_name, set()).add(shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(resolved_name)
                loaded_param_parts.setdefault(resolved_name, set()).add("full")

        if self.head_lora_rank > 0:
            required = {"head_lora_a.weight", "head_lora_b.weight"}
            missing = required - loaded_params
            if missing:
                raise ValueError(
                    "DFLASH config enables Head-LoRA but checkpoint weights are "
                    f"missing {sorted(missing)}. Refusing to use uninitialized weights."
                )
        if self.joint_judge_enabled:
            missing_joint: list[str] = []
            incomplete_joint: list[str] = []
            for param_name in params_dict:
                if not (
                    param_name.startswith("joint_judge_model.")
                    or param_name.startswith("joint_judge_head.")
                ):
                    continue
                parts = loaded_param_parts.get(param_name)
                if not parts:
                    missing_joint.append(param_name)
                    continue
                if ".qkv_proj." in param_name and "full" not in parts:
                    required_parts: set[object] = {"q", "k", "v"}
                    if not required_parts.issubset(parts):
                        incomplete_joint.append(
                            f"{param_name} shards={sorted(str(x) for x in parts)}"
                        )
                if ".gate_up_proj." in param_name and "full" not in parts:
                    required_parts = {0, 1}
                    if not required_parts.issubset(parts):
                        incomplete_joint.append(
                            f"{param_name} shards={sorted(str(x) for x in parts)}"
                        )
            if missing_joint or incomplete_joint:
                details = []
                if missing_joint:
                    details.append(f"missing={sorted(missing_joint)}")
                if incomplete_joint:
                    details.append(f"incomplete={sorted(incomplete_joint)}")
                raise ValueError(
                    "DFLASH joint judge is enabled but its checkpoint is "
                    "incomplete. Refusing to serve randomly initialized judge "
                    "weights: "
                    + "; ".join(details)
                )
        if require_complete:
            _validate_dflash_parameter_load_coverage(
                params_dict.keys(), loaded_param_parts
            )
        return loaded_params


def _rms_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.float()
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


class SparseDFlashDraftModel(nn.Module):
    """SGLang runtime for SpecForge's top-1 full-tower DFlash router.

    Each expert owns an independent attention stack and therefore independent
    KV-cache layer IDs.  For correctness under CUDA graphs and mixed batches,
    the first implementation evaluates every expert and selects each request's
    block output with a tensor mask.  This preserves the trained routing
    semantics; sparse dispatch can be added independently as an optimization.

    ``input_ids`` is an internal route carrier for this embedding-free draft
    model: DFlashWorker places the selected expert ID in the first position of
    each block while passing the real token embeddings through ``input_embeds``.
    The original token IDs remain in the worker for target verification.
    """

    supports_fused_kv_materialization = False
    has_multiple_dflash_kv_groups = True
    uses_input_ids_for_expert_routing = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.num_experts = int(draft_config.sparse_num_experts)
        self.expert_mode = str(draft_config.sparse_expert_mode)
        self.router_temperature = float(draft_config.sparse_router_temperature)
        if self.num_experts < 2:
            raise ValueError("SparseDFlashDraftModel requires sparse_num_experts >= 2.")
        if self.expert_mode != "full_tower":
            raise ValueError(
                "SGLang SparseDFlashDraftModel currently supports only "
                f"expert_mode='full_tower', got {self.expert_mode!r}."
            )
        if int(draft_config.head_lora_rank) > 0:
            raise ValueError(
                "Sparse full-tower DFlash does not support Head-LoRA checkpoints."
            )

        hidden_size = int(config.hidden_size)
        num_layers = int(draft_config.require_num_layers())
        self.experts = nn.ModuleList(
            [
                DFlashDraftModel(
                    config=config,
                    quant_config=quant_config,
                    prefix=(
                        f"{prefix}.experts.{expert_idx}"
                        if prefix
                        else f"experts.{expert_idx}"
                    ),
                    layer_id_offset=expert_idx * num_layers,
                )
                for expert_idx in range(self.num_experts)
            ]
        )
        first_expert = self.experts[0]
        self.block_size = int(first_expert.block_size)
        self.num_context_features = int(first_expert.num_context_features)
        router_input_size = (self.num_context_features + 1) * hidden_size
        self.router = ReplicatedLinear(
            router_input_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.router" if prefix else "router",
        )
        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )

        # ModelRunner uses these bounds to size the draft KV pool.  The HF
        # config depth remains the per-expert depth for target feature capture.
        self.start_layer = 0
        self.end_layer = num_layers * self.num_experts
        self.head_lora_rank = 0
        self.head_lora_alpha = 0.0
        self._last_expert_ids: Optional[torch.Tensor] = None

    @property
    def layers(self):
        """Flattened attention layers, primarily for diagnostics."""
        return tuple(layer for expert in self.experts for layer in expert.layers)

    @torch.no_grad()
    def select_experts(
        self,
        *,
        anchor_embeddings: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the trained leakage-free block router."""
        if anchor_embeddings.ndim != 2:
            raise ValueError(
                "Sparse DFlash anchor embeddings must be [batch, hidden], got "
                f"{tuple(anchor_embeddings.shape)}."
            )
        expected_context = self.num_context_features * int(self.config.hidden_size)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected_context:
            raise ValueError(
                "Sparse DFlash router target context must be "
                f"[batch, {expected_context}], got {tuple(target_hidden.shape)}."
            )
        if int(target_hidden.shape[0]) != int(anchor_embeddings.shape[0]):
            raise ValueError(
                "Sparse DFlash router batch mismatch: "
                f"context={target_hidden.shape[0]}, anchors={anchor_embeddings.shape[0]}."
            )
        router_input = torch.cat(
            [_rms_normalize(target_hidden), _rms_normalize(anchor_embeddings)],
            dim=-1,
        )
        router_logits = (
            F.linear(router_input, self.router.weight.float()) / self.router_temperature
        )
        routing_scores = router_logits + self.expert_bias.float().view(1, -1)
        expert_ids = routing_scores.argmax(dim=-1).to(torch.long)
        self._last_expert_ids = expert_ids.detach()
        return expert_ids

    def iter_dflash_kv_groups(
        self, target_hidden: torch.Tensor
    ) -> Iterator[Tuple[Iterable[DFlashDecoderLayer], torch.Tensor]]:
        """Yield each expert's layers with its own projected target context."""
        for expert in self.experts:
            yield expert.layers, expert.project_target_hidden(target_hidden)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "SparseDFlashDraftModel requires target-model input_embeds."
            )
        if input_ids.numel() != input_embeds.shape[0]:
            raise ValueError(
                "Sparse DFlash route carrier/input embedding length mismatch: "
                f"{input_ids.numel()} vs {input_embeds.shape[0]}."
            )
        if input_ids.numel() % self.block_size != 0:
            raise ValueError(
                "Sparse DFlash route carrier length must be divisible by block_size="
                f"{self.block_size}, got {input_ids.numel()}."
            )

        batch_size = input_ids.numel() // self.block_size
        expert_ids = input_ids.view(batch_size, self.block_size)[:, 0]
        token_expert_ids = expert_ids.repeat_interleave(self.block_size)
        routed_hidden = torch.zeros_like(input_embeds)
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            ).hidden_states
            route_mask = (token_expert_ids == expert_idx).unsqueeze(-1)
            routed_hidden = torch.where(route_mask, expert_output, routed_hidden)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=routed_hidden,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        loaded_router = False
        loaded_expert_bias = False
        loaded_checkpoint_names: set[str] = set()
        loaded_expert_tensors = [0] * self.num_experts
        for name, loaded_weight in weights:
            normalized_name = (
                name[len("model.") :] if name.startswith("model.") else name
            )
            if normalized_name == "router.weight":
                weight_loader = getattr(
                    self.router.weight, "weight_loader", default_weight_loader
                )
                weight_loader(self.router.weight, loaded_weight)
                loaded_router = True
                loaded_checkpoint_names.add(normalized_name)
                continue
            if normalized_name == "expert_bias":
                self.expert_bias.copy_(
                    loaded_weight.to(
                        device=self.expert_bias.device,
                        dtype=self.expert_bias.dtype,
                    )
                )
                loaded_expert_bias = True
                loaded_checkpoint_names.add(normalized_name)
                continue
            if not normalized_name.startswith("experts."):
                continue
            parts = normalized_name.split(".", 2)
            if len(parts) != 3:
                continue
            try:
                expert_idx = int(parts[1])
            except ValueError:
                continue
            if expert_idx < 0 or expert_idx >= self.num_experts:
                raise ValueError(
                    f"Sparse DFlash checkpoint contains expert index {expert_idx}, "
                    f"but config declares {self.num_experts} experts."
                )
            expert = self.experts[expert_idx]
            if isinstance(expert, DFlashDraftModel):
                # Sparse dispatch streams one tensor at a time into each
                # expert. The sparse wrapper owns its checkpoint-level
                # validation, so do not apply the ordinary full-checkpoint
                # gate to this internal subset call.
                loaded_params = expert.load_weights(
                    [(parts[2], loaded_weight)], require_complete=False
                )
            else:
                # Preserve lightweight test/custom expert compatibility.
                loaded_params = expert.load_weights([(parts[2], loaded_weight)])
            if not loaded_params:
                raise ValueError(
                    "Sparse DFlash checkpoint tensor does not map to a runtime "
                    f"parameter: {name!r}."
                )
            loaded_expert_tensors[expert_idx] += 1
            loaded_checkpoint_names.add(normalized_name)

        if not loaded_router:
            raise ValueError(
                "Sparse DFlash checkpoint is missing required router.weight."
            )
        if not loaded_expert_bias:
            logger.warning("Sparse DFlash checkpoint has no expert_bias; using zeros.")
        missing_experts = [
            expert_idx
            for expert_idx, count in enumerate(loaded_expert_tensors)
            if count == 0
        ]
        if missing_experts:
            raise ValueError(
                "Sparse DFlash checkpoint did not load any tensors for experts "
                f"{missing_experts}."
            )
        logger.info(
            "Loaded Sparse DFlash checkpoint: experts=%s tensors_per_expert=%s "
            "router=%s expert_bias=%s",
            self.num_experts,
            loaded_expert_tensors,
            loaded_router,
            loaded_expert_bias,
        )
        return loaded_checkpoint_names


EntryClass = [DFlashDraftModel, SparseDFlashDraftModel]
