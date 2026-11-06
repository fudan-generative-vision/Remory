"""DFlash-shaped continuous context compressor and frozen target runtime.

The source text is visible only to :class:`DFlashKVCompressor`: a frozen target
prefill turns the source into KV tensors, and five query-only blocks reduce
those tensors to sixteen continuous memory embeddings. Reconstruction uses a
fresh target-model context. In particular, the source cache is never passed
to the recovery forward.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from context_compression.layout import (
    COMPRESSED_QUERY_COUNT,
    SOURCE_KV_WIDTH,
    build_dflash_16_query_layout,
)
from context_compression.losses import (
    ChunkedTokenCEOutput,
    frozen_lm_head_chunked_token_ce,
)

DEFAULT_TARGET_LAYER_IDS = (1, 9, 17, 25, 33)


def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return hidden * cos + _rotate_half(hidden) * sin


def _repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    batch, kv_heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, kv_heads, repeats, length, head_dim)
    return hidden.reshape(batch, kv_heads * repeats, length, head_dim)


class DFlashKVCompressionBlock(nn.Module):
    """One DFlash-style query block attending directly to one target KV layer."""

    def __init__(self, target_config: Any) -> None:
        super().__init__()
        self.hidden_size = int(target_config.hidden_size)
        self.num_attention_heads = int(target_config.num_attention_heads)
        self.num_key_value_heads = int(target_config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                target_config,
                "head_dim",
                self.hidden_size // self.num_attention_heads,
            )
        )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        attention_bias = bool(getattr(target_config, "attention_bias", False))
        self.attention_dropout = float(getattr(target_config, "attention_dropout", 0.0))
        eps = float(target_config.rms_norm_eps)

        self.input_layernorm = Qwen3RMSNorm(self.hidden_size, eps=eps)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(self.hidden_size, eps=eps)
        self.mlp = Qwen3MLP(target_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_key: torch.Tensor,
        source_value: torch.Tensor,
        keep_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, _ = hidden_states.shape
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)

        query = self.q_proj(normalized).view(
            batch, query_count, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        query_key = self.k_proj(normalized).view(
            batch, query_count, self.num_key_value_heads, self.head_dim
        )
        query_key = self.k_norm(query_key).transpose(1, 2)
        query_value = self.v_proj(normalized).view(
            batch, query_count, self.num_key_value_heads, self.head_dim
        )
        query_value = query_value.transpose(1, 2)

        # Target cache keys already contain RoPE at their source positions.  We
        # rotate only the newly-created memory queries before concatenation.
        query = _apply_rotary(query, cos, sin)
        query_key = _apply_rotary(query_key, cos, sin)
        key = torch.cat((source_key, query_key), dim=-2)
        value = torch.cat((source_value, query_value), dim=-2)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=keep_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_count, -1)
        hidden_states = residual + self.o_proj(attended)
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class DFlashKVCompressor(nn.Module):
    """Compress a frozen target's fixed-width KV cache into 16 embeddings.

    There is one trainable DFlash-style block for each selected target cache
    layer.  The source K/V tensors are consumed directly; they are not copied
    into a trainable target-model wrapper or later reused by recovery.
    """

    def __init__(
        self,
        target_config: Any,
        *,
        target_layer_ids: Sequence[int] = DEFAULT_TARGET_LAYER_IDS,
        source_tokens: int = SOURCE_KV_WIDTH,
        output_embedding_rms: Optional[float] = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        layer_ids = tuple(int(layer_id) for layer_id in target_layer_ids)
        if not layer_ids:
            raise ValueError("target_layer_ids must not be empty")
        target_layers = int(target_config.num_hidden_layers)
        if any(layer_id < 0 or layer_id >= target_layers for layer_id in layer_ids):
            raise ValueError(f"target_layer_ids must be in [0, {target_layers - 1}]")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("target_layer_ids must be unique")
        if isinstance(source_tokens, bool) or not isinstance(source_tokens, int):
            raise TypeError("source_tokens must be an integer")
        if source_tokens <= 0:
            raise ValueError("source_tokens must be positive")

        self.target_layer_ids = layer_ids
        self.source_tokens = source_tokens
        self.hidden_size = int(target_config.hidden_size)
        self.num_key_value_heads = int(target_config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                target_config,
                "head_dim",
                self.hidden_size // int(target_config.num_attention_heads),
            )
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.memory_queries = nn.Parameter(
            torch.empty(1, COMPRESSED_QUERY_COUNT, self.hidden_size)
        )
        # Keep compressor construction isolated from the caller's live target
        # config: Qwen3MLP holds the config and must never observe later edits.
        compressor_config = copy.deepcopy(target_config)
        self.layers = nn.ModuleList(
            [DFlashKVCompressionBlock(compressor_config) for _ in layer_ids]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(compressor_config)
        self.norm = Qwen3RMSNorm(
            self.hidden_size, eps=float(target_config.rms_norm_eps)
        )

        initializer_range = float(getattr(target_config, "initializer_range", 0.02))
        self.apply(lambda module: self._initialize_module(module, initializer_range))
        nn.init.normal_(self.memory_queries, mean=0.0, std=initializer_range)
        initial_scale = (
            float(output_embedding_rms)
            if output_embedding_rms is not None
            else initializer_range
        )
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("output_embedding_rms must be finite and positive")
        self.log_output_scale = nn.Parameter(
            torch.tensor(math.log(initial_scale), dtype=torch.float32)
        )

    @staticmethod
    def _initialize_module(module: nn.Module, std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def from_target_model(
        cls,
        target_model: nn.Module,
        *,
        target_layer_ids: Sequence[int] = DEFAULT_TARGET_LAYER_IDS,
        source_tokens: int = SOURCE_KV_WIDTH,
        gradient_checkpointing: bool = True,
    ) -> "DFlashKVCompressor":
        embedding_weight = target_model.get_input_embeddings().weight.detach()
        sample = embedding_weight[: min(4096, embedding_weight.shape[0])]
        embedding_rms = sample.float().square().mean().sqrt().item()
        return cls(
            target_model.config,
            target_layer_ids=target_layer_ids,
            source_tokens=source_tokens,
            output_embedding_rms=embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )

    @staticmethod
    def _cache_layer(
        source_cache: Any, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # transformers <=4.57 exposes legacy ``cache[layer_id]`` indexing;
        # transformers 5.x moved public storage to ``cache.layers``.
        try:
            layers = getattr(source_cache, "layers", None)
            if layers is not None:
                layer = layers[layer_id]
                key = getattr(layer, "keys")
                value = getattr(layer, "values")
            else:
                key, value = source_cache[layer_id]
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"source cache has no target layer {layer_id}") from exc
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("source cache entries must be tensor (key, value) pairs")
        return key, value

    def forward(
        self,
        source_cache: Any,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_attention_mask = source_attention_mask.bool()
        layout = build_dflash_16_query_layout(
            source_attention_mask,
            source_tokens=self.source_tokens,
        )
        batch = source_attention_mask.shape[0]
        hidden_states = self.memory_queries.expand(batch, -1, -1)
        query_positions = layout.position_ids[:, -COMPRESSED_QUERY_COUNT:]
        cos, sin = self.rotary_emb(hidden_states, query_positions)

        for block, layer_id in zip(self.layers, self.target_layer_ids):
            source_key, source_value = self._cache_layer(source_cache, layer_id)
            expected = (
                batch,
                self.num_key_value_heads,
                self.source_tokens,
                self.head_dim,
            )
            if (
                tuple(source_key.shape) != expected
                or tuple(source_value.shape) != expected
            ):
                raise ValueError(
                    f"target layer {layer_id} KV shape must be {expected}, got "
                    f"{tuple(source_key.shape)} and {tuple(source_value.shape)}"
                )
            if source_key.device != hidden_states.device:
                raise ValueError("source cache and compressor must share a device")
            if source_key.dtype != hidden_states.dtype:
                source_key = source_key.to(dtype=hidden_states.dtype)
                source_value = source_value.to(dtype=hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    source_key,
                    source_value,
                    layout.keep_mask,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    source_key,
                    source_value,
                    layout.keep_mask,
                    cos,
                    sin,
                )

        scale = self.log_output_scale.exp().to(dtype=hidden_states.dtype)
        return self.norm(hidden_states) * scale

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": type(self).__name__,
            "source_tokens": self.source_tokens,
            "memory_tokens": COMPRESSED_QUERY_COUNT,
            "target_layer_ids": list(self.target_layer_ids),
            "hidden_size": self.hidden_size,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


class RecoveryForwardOutput(NamedTuple):
    hidden_states: torch.Tensor
    prefix_width: int
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class RecoveryLossOutput:
    loss: torch.Tensor
    token_accuracy: torch.Tensor
    token_count: torch.Tensor
    prefix_width: int


def freeze_target_model(target_model: nn.Module) -> nn.Module:
    """Freeze target parameters without disabling autograd through its inputs."""

    target_model.requires_grad_(False)
    for parameter in target_model.parameters():
        parameter.grad = None
    return target_model


class FrozenTargetRuntime:
    """Differentiable recovery operations through a parameter-frozen target."""

    def __init__(
        self,
        target_model: nn.Module,
        *,
        source_tokens: int = SOURCE_KV_WIDTH,
        gradient_checkpointing: bool = True,
    ) -> None:
        if isinstance(source_tokens, bool) or not isinstance(source_tokens, int):
            raise TypeError("source_tokens must be an integer")
        if source_tokens <= 0:
            raise ValueError("source_tokens must be positive")
        self.target = freeze_target_model(target_model)
        self.source_tokens = source_tokens
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            try:
                self.target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:  # transformers compatibility
                self.target.gradient_checkpointing_enable()
            # HF checkpoint layers activate only in training mode.  Qwen3 has
            # zero dropout by default, so this does not randomize recovery.
            self.target.train()
        else:
            self.target.eval()

    @property
    def backbone(self) -> nn.Module:
        if not hasattr(self.target, "model"):
            raise TypeError("target model must expose its decoder as `.model`")
        return self.target.model

    @property
    def lm_head(self) -> nn.Module:
        head = self.target.get_output_embeddings()
        if head is None:
            raise TypeError("target model has no output embedding / lm_head")
        return head

    def encode_source(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> Any:
        """Create ordinary (not inference-mode) frozen KV tensors."""

        if tuple(source_input_ids.shape) != tuple(source_attention_mask.shape):
            raise ValueError("source ids and attention mask shapes differ")
        if (
            source_input_ids.ndim != 2
            or source_input_ids.shape[1] != self.source_tokens
        ):
            raise ValueError(
                f"source input must have shape [batch, {self.source_tokens}]"
            )
        was_training = self.target.training
        self.target.eval()  # use_cache is disabled by HF checkpointing in train mode
        positions = source_attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~source_attention_mask.bool(), 0)
        try:
            with torch.no_grad():
                output = self.target(
                    input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                    position_ids=positions,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
        finally:
            self.target.train(was_training)
        if output.past_key_values is None:
            raise RuntimeError("frozen target prefill did not return a KV cache")
        return output.past_key_values

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.target.get_input_embeddings()(input_ids)

    def assemble_teacher_forcing(
        self,
        memory_embeddings: torch.Tensor,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        task_input_ids: torch.Tensor,
        task_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Assemble ``system + memory + task + gold[:-1]`` embeddings."""

        batch = memory_embeddings.shape[0]
        if (
            memory_embeddings.ndim != 3
            or memory_embeddings.shape[1] != COMPRESSED_QUERY_COUNT
            or memory_embeddings.shape[2] != self.target.config.hidden_size
        ):
            raise ValueError(
                "memory embeddings must have shape [batch, 16, target_hidden_size]"
            )
        tensors = (
            prefix_input_ids,
            prefix_attention_mask,
            task_input_ids,
            task_attention_mask,
            target_input_ids,
            target_labels,
        )
        if any(tensor.ndim != 2 or tensor.shape[0] != batch for tensor in tensors):
            raise ValueError("all recovery tensors must be rank-2 with one batch size")
        if target_input_ids.shape != target_labels.shape:
            raise ValueError("target ids and labels shapes differ")
        if prefix_input_ids.shape != prefix_attention_mask.shape:
            raise ValueError("prefix ids and attention mask shapes differ")
        if task_input_ids.shape != task_attention_mask.shape:
            raise ValueError("task ids and attention mask shapes differ")
        if not bool(task_attention_mask[:, -1].bool().all().item()):
            raise ValueError(
                "every task suffix must end in a valid token; left-pad variable tasks"
            )
        if target_input_ids.shape[1] < 1:
            raise ValueError("target sequence must contain at least one token")
        target_valid = target_labels != int(ignore_index)
        target_lengths = target_valid.sum(dim=-1)
        expected_target_valid = (
            torch.arange(target_labels.shape[1], device=target_labels.device)[None, :]
            < target_lengths[:, None]
        )
        if bool((target_lengths <= 0).any().item()) or not torch.equal(
            target_valid, expected_target_valid
        ):
            raise ValueError("target labels must be a nonempty right-padded prefix")

        target_teacher_ids = target_input_ids[:, :-1]
        target_teacher_mask = target_labels[:, :-1] != int(ignore_index)
        prefix_embeds = self._embed(prefix_input_ids)
        task_embeds = self._embed(task_input_ids)
        teacher_embeds = self._embed(target_teacher_ids)
        inputs_embeds = torch.cat(
            (prefix_embeds, memory_embeddings, task_embeds, teacher_embeds), dim=1
        )
        memory_mask = torch.ones(
            (batch, memory_embeddings.shape[1]),
            dtype=torch.bool,
            device=memory_embeddings.device,
        )
        attention_mask = torch.cat(
            (
                prefix_attention_mask.bool(),
                memory_mask,
                task_attention_mask.bool(),
                target_teacher_mask,
            ),
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        prefix_width = (
            prefix_input_ids.shape[1]
            + memory_embeddings.shape[1]
            + task_input_ids.shape[1]
        )
        return inputs_embeds, attention_mask, position_ids, prefix_width

    def recovery_hidden(
        self,
        memory_embeddings: torch.Tensor,
        **batch: torch.Tensor,
    ) -> RecoveryForwardOutput:
        inputs_embeds, attention_mask, position_ids, prefix_width = (
            self.assemble_teacher_forcing(memory_embeddings, **batch)
        )
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return RecoveryForwardOutput(
            hidden_states=output.last_hidden_state,
            prefix_width=prefix_width,
            attention_mask=attention_mask,
        )

    def token_loss(
        self,
        memory_embeddings: torch.Tensor,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        task_input_ids: torch.Tensor,
        task_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        chunk_size: int = 128,
        ignore_index: int = -100,
    ) -> RecoveryLossOutput:
        recovered = self.recovery_hidden(
            memory_embeddings,
            prefix_input_ids=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            task_input_ids=task_input_ids,
            task_attention_mask=task_attention_mask,
            target_input_ids=target_input_ids,
            target_labels=target_labels,
            ignore_index=ignore_index,
        )
        loss_output: ChunkedTokenCEOutput = frozen_lm_head_chunked_token_ce(
            recovered.hidden_states,
            target_labels,
            self.lm_head,
            prefix_width=recovered.prefix_width,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )
        return RecoveryLossOutput(
            loss=loss_output.loss,
            token_accuracy=loss_output.token_accuracy,
            token_count=loss_output.token_count,
            prefix_width=recovered.prefix_width,
        )

    @torch.inference_mode()
    def greedy_generate(
        self,
        memory_embeddings: torch.Tensor,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        task_input_ids: torch.Tensor,
        task_attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Generate from a fresh recovery cache; no source cache is accepted."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        batch = memory_embeddings.shape[0]
        if (
            memory_embeddings.ndim != 3
            or memory_embeddings.shape[1] != COMPRESSED_QUERY_COUNT
            or memory_embeddings.shape[2] != self.target.config.hidden_size
        ):
            raise ValueError(
                "memory embeddings must have shape [batch, 16, target_hidden_size]"
            )
        if prefix_input_ids.shape != prefix_attention_mask.shape:
            raise ValueError("prefix ids and attention mask shapes differ")
        if task_input_ids.shape != task_attention_mask.shape:
            raise ValueError("task ids and attention mask shapes differ")
        if not bool(task_attention_mask[:, -1].bool().all().item()):
            raise ValueError(
                "every task suffix must end in a valid token; left-pad variable tasks"
            )
        memory_mask = torch.ones(
            (batch, memory_embeddings.shape[1]),
            dtype=torch.bool,
            device=memory_embeddings.device,
        )
        inputs_embeds = torch.cat(
            (
                self._embed(prefix_input_ids),
                memory_embeddings,
                self._embed(task_input_ids),
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                prefix_attention_mask.bool(),
                memory_mask,
                task_attention_mask.bool(),
            ),
            dim=1,
        )
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~attention_mask, 0)

        was_training = self.target.training
        self.target.eval()
        try:
            output = self.target(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = [next_token]
            finished = torch.zeros(batch, dtype=torch.bool, device=next_token.device)
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1) == int(eos_token_id)
            recovery_cache = output.past_key_values

            for _ in range(1, max_new_tokens):
                if bool(finished.all().item()):
                    break
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch, 1),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                next_position = attention_mask.sum(dim=-1, keepdim=True) - 1
                output = self.target(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=recovery_cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, int(eos_token_id)),
                        next_token,
                    )
                generated.append(next_token)
                recovery_cache = output.past_key_values
                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1) == int(eos_token_id)
        finally:
            self.target.train(was_training)
        return torch.cat(generated, dim=1)


__all__ = [
    "DEFAULT_TARGET_LAYER_IDS",
    "DFlashKVCompressionBlock",
    "DFlashKVCompressor",
    "FrozenTargetRuntime",
    "RecoveryForwardOutput",
    "RecoveryLossOutput",
    "freeze_target_model",
]
