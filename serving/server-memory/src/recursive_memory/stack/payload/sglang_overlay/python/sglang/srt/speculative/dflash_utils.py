from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def get_active_dflash_verify_width_buckets(
    *,
    block_size: int,
    max_verify_width: int,
) -> tuple[int, ...]:
    """Return the legal target widths for an active DFlash proposal chain."""

    block_size = int(block_size)
    max_verify_width = int(max_verify_width)
    if block_size < 2:
        raise ValueError(f"Active DFLASH block_size must be >= 2, got {block_size}.")
    if max_verify_width < block_size:
        raise ValueError(
            "Active DFLASH max_verify_width must cover at least one block, got "
            f"block_size={block_size}, max_verify_width={max_verify_width}."
        )

    proposal_stride = block_size - 1
    if (max_verify_width - 1) % proposal_stride != 0:
        raise ValueError(
            "Active DFLASH max_verify_width must be 1 + N * (block_size - 1), "
            f"got block_size={block_size}, max_verify_width={max_verify_width}."
        )
    return tuple(
        1 + block_index * proposal_stride
        for block_index in range(1, (max_verify_width - 1) // proposal_stride + 1)
    )


def select_active_dflash_verify_width(
    *,
    max_accumulated_draft_tokens: int,
    block_size: int,
    max_verify_width: int,
) -> int:
    """Choose the smallest legal width containing a batch's real proposals.

    Width includes the shared root token, while ``max_accumulated_draft_tokens``
    counts only tokens appended behind that root.  Even an empty proposal uses
    the one-block width so the first implementation has no root-only target
    path.
    """

    max_accumulated_draft_tokens = int(max_accumulated_draft_tokens)
    if max_accumulated_draft_tokens < 0:
        raise ValueError(
            "max_accumulated_draft_tokens must be non-negative, got "
            f"{max_accumulated_draft_tokens}."
        )
    if max_accumulated_draft_tokens >= int(max_verify_width):
        raise ValueError(
            "Active DFLASH accumulated proposal does not fit max_verify_width: "
            f"proposal_tokens={max_accumulated_draft_tokens}, "
            f"max_verify_width={max_verify_width}."
        )

    required_width = max_accumulated_draft_tokens + 1
    for width in get_active_dflash_verify_width_buckets(
        block_size=block_size,
        max_verify_width=max_verify_width,
    ):
        if required_width <= width:
            return width
    raise AssertionError(
        "Active DFLASH verify width selection exhausted validated buckets."
    )


def compact_active_dflash_verify_tensors(
    *,
    verify_tokens: torch.Tensor,
    verify_positions: torch.Tensor,
    proposal_hidden: Optional[torch.Tensor],
    verify_width: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Narrow max-capacity active buffers into contiguous target inputs."""

    verify_width = int(verify_width)
    if verify_tokens.ndim != 2 or verify_positions.ndim != 2:
        raise ValueError("Active DFLASH verify token/position buffers must be 2-D.")
    if verify_tokens.shape != verify_positions.shape:
        raise ValueError(
            "Active DFLASH verify token/position shapes must match, got "
            f"tokens={tuple(verify_tokens.shape)}, "
            f"positions={tuple(verify_positions.shape)}."
        )
    if not 1 <= verify_width <= int(verify_tokens.shape[1]):
        raise ValueError(
            "Active DFLASH verify_width is outside the source buffer, got "
            f"verify_width={verify_width}, source_width={verify_tokens.shape[1]}."
        )

    compact_tokens = verify_tokens[:, :verify_width].contiguous()
    compact_positions = verify_positions[:, :verify_width].contiguous()
    compact_hidden = proposal_hidden
    if proposal_hidden is not None:
        if proposal_hidden.ndim != 3:
            raise ValueError("Active DFLASH proposal_hidden must be 3-D.")
        if int(proposal_hidden.shape[0]) != int(verify_tokens.shape[0]):
            raise ValueError(
                "Active DFLASH proposal_hidden batch size must match verify tokens."
            )
        proposal_width = verify_width - 1
        if proposal_width > int(proposal_hidden.shape[1]):
            raise ValueError(
                "Active DFLASH proposal_hidden does not cover verify_width, got "
                f"proposal_width={proposal_width}, "
                f"hidden_width={proposal_hidden.shape[1]}."
            )
        compact_hidden = proposal_hidden[:, :proposal_width].contiguous()
    return compact_tokens, compact_positions, compact_hidden


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
        if cfg is None:
            cfg = config.get("dflow_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
        if cfg is None:
            cfg = getattr(config, "dflow_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]
    head_lora_rank: int
    head_lora_alpha: float
    sparse_num_experts: int
    sparse_expert_mode: str
    sparse_router_temperature: float
    joint_judge_enabled: bool
    joint_judge_head_hidden_size: int
    joint_judge_prior: float

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_kv_num_layers(self) -> int:
        """Return the number of independent attention-layer KV caches.

        A full-tower sparse drafter owns one complete DFlash KV stack per
        expert.  The value intentionally differs from ``num_hidden_layers``:
        the latter still describes the depth of each tower and therefore the
        number of target hidden-state features used by the DFlash projector.
        """
        num_layers = self.require_num_layers()
        if self.joint_judge_enabled:
            if self.sparse_num_experts > 1:
                raise ValueError(
                    "Joint DFLASH drafter/judge checkpoints cannot also enable "
                    "sparse drafter experts."
                )
            # The isomorphic judge is a second, independent DFlash tower.  Its
            # attention layers use IDs [num_layers, 2 * num_layers), so both
            # stacks need their own KV-cache storage.
            return num_layers * 2
        if self.sparse_num_experts <= 1:
            return num_layers
        if self.sparse_expert_mode == "full_tower":
            return num_layers * self.sparse_num_experts
        if self.sparse_expert_mode == "last_layer":
            return (num_layers - 1) + self.sparse_num_experts
        raise ValueError(
            f"Unsupported DFLASH sparse expert mode {self.sparse_expert_mode!r}."
        )

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            # ``target_num_layers`` is a contextual-x0 sentinel for the
            # post-final-norm hidden state. Ordinary layer ids remain in
            # [0, target_num_layers).
            if val < 0 or val > target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, supported range="
                    f"[0, {target_num_layers}] (the upper endpoint selects the "
                    "post-final-norm hidden state)."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    head_lora_rank = _parse_optional_int(
        dflash_cfg.get("head_lora_rank", 0),
        field_name="DFLASH head_lora_rank",
        min_value=0,
    )
    head_lora_rank = int(head_lora_rank or 0)
    raw_head_lora_alpha = dflash_cfg.get("head_lora_alpha", None)
    if raw_head_lora_alpha is None:
        head_lora_alpha = float(head_lora_rank)
    else:
        try:
            head_lora_alpha = float(raw_head_lora_alpha)
        except Exception as e:
            raise ValueError(
                f"Invalid DFLASH head_lora_alpha={raw_head_lora_alpha!r}."
            ) from e
    if not math.isfinite(head_lora_alpha):
        raise ValueError(
            f"DFLASH head_lora_alpha must be finite, got {head_lora_alpha}."
        )
    if head_lora_rank > 0 and head_lora_alpha <= 0.0:
        raise ValueError(
            "DFLASH head_lora_alpha must be positive when Head-LoRA is enabled, "
            f"got {head_lora_alpha}."
        )

    raw_sparse_cfg = dflash_cfg.get("sparse_drafter", {}) or {}
    if not isinstance(raw_sparse_cfg, dict):
        try:
            raw_sparse_cfg = dict(raw_sparse_cfg)
        except Exception as e:
            raise ValueError(
                "DFLASH dflash_config.sparse_drafter must be a mapping, got "
                f"{type(raw_sparse_cfg).__name__}."
            ) from e
    sparse_num_experts = _parse_optional_int(
        raw_sparse_cfg.get("num_experts", 1),
        field_name="DFLASH sparse num_experts",
        min_value=1,
    )
    sparse_num_experts = int(sparse_num_experts or 1)
    sparse_expert_mode = str(raw_sparse_cfg.get("expert_mode", "full_tower"))
    if sparse_expert_mode not in {"full_tower", "last_layer"}:
        raise ValueError(
            "DFLASH sparse expert_mode must be 'full_tower' or 'last_layer', "
            f"got {sparse_expert_mode!r}."
        )
    try:
        sparse_router_temperature = float(raw_sparse_cfg.get("router_temperature", 1.0))
    except Exception as e:
        raise ValueError(
            "Invalid DFLASH sparse router_temperature="
            f"{raw_sparse_cfg.get('router_temperature')!r}."
        ) from e
    if not math.isfinite(sparse_router_temperature) or sparse_router_temperature <= 0:
        raise ValueError(
            "DFLASH sparse router_temperature must be positive and finite, "
            f"got {sparse_router_temperature}."
        )

    joint_judge_enabled = dflash_cfg.get("joint_judge_enabled", False)
    if not isinstance(joint_judge_enabled, bool):
        raise ValueError(
            "DFLASH dflash_config.joint_judge_enabled must be a boolean, "
            f"got {joint_judge_enabled!r}."
        )
    joint_judge_head_hidden_size = _parse_optional_int(
        dflash_cfg.get("joint_judge_head_hidden_size", 256),
        field_name="DFLASH joint_judge_head_hidden_size",
        min_value=0,
    )
    joint_judge_head_hidden_size = int(joint_judge_head_hidden_size or 0)
    try:
        joint_judge_prior = float(dflash_cfg.get("joint_judge_prior", 0.5))
    except Exception as e:
        raise ValueError(
            "Invalid DFLASH joint_judge_prior="
            f"{dflash_cfg.get('joint_judge_prior')!r}."
        ) from e
    if not math.isfinite(joint_judge_prior) or not 0.0 < joint_judge_prior < 1.0:
        raise ValueError(
            "DFLASH joint_judge_prior must be finite and in (0, 1), "
            f"got {joint_judge_prior}."
        )
    if joint_judge_enabled and sparse_num_experts > 1:
        raise ValueError(
            "Joint DFLASH drafter/judge checkpoints cannot also enable sparse "
            f"drafter experts (num_experts={sparse_num_experts})."
        )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
        head_lora_rank=head_lora_rank,
        head_lora_alpha=head_lora_alpha,
        sparse_num_experts=sparse_num_experts,
        sparse_expert_mode=sparse_expert_mode,
        sparse_router_temperature=sparse_router_temperature,
        joint_judge_enabled=joint_judge_enabled,
        joint_judge_head_hidden_size=joint_judge_head_hidden_size,
        joint_judge_prior=joint_judge_prior,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    valid_draft_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.
        valid_draft_lens: Optional real proposal count for each row when
            ``candidates`` is padded to a fixed active-chain verify width.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index accept_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    if valid_draft_lens is not None:
        valid_lens = valid_draft_lens.to(
            device=candidates.device, dtype=torch.int64
        ).clamp(min=0, max=max(0, block_size - 1))
        if valid_lens.ndim != 1 or int(valid_lens.numel()) != int(bs):
            raise ValueError(
                "valid_draft_lens must have shape [batch], got "
                f"{tuple(valid_lens.shape)} for batch={bs}."
            )
        proposal_pos = torch.arange(
            block_size - 1, dtype=torch.int64, device=candidates.device
        ).unsqueeze(0)
        matches = matches & (proposal_pos < valid_lens.unsqueeze(1))
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)


def normalize_joint_judge_decoder_mode(mode: str) -> str:
    """Normalize active-judge decoder names while retaining readable aliases."""

    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "per_token": "threshold",
        "token_threshold": "threshold",
        "cost": "expected_cost",
        "hazard": "expected_cost",
        "asymmetric_cost": "expected_cost",
        "first_fail_quantile": "quantile",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"threshold", "expected_cost", "quantile"}:
        raise ValueError(
            "joint judge decoder mode must be one of "
            "['threshold', 'expected_cost', 'quantile'], got "
            f"{mode!r}."
        )
    return normalized


def compute_joint_judge_first_fail_distribution(
    *,
    judge_logits: torch.Tensor,
    valid_horizons: Optional[Union[int, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert conditional-survival logits into a first-fail distribution.

    For proposal conditional-survival probabilities ``q_1..q_H``, boundary
    ``K=k`` means that the first ``k`` proposals survive and proposal ``k+1``
    is the first failure.  ``K=H`` is the terminal all-survive event.

    Returns:
        prefix_survival: ``[batch, H + 1]`` where column ``k`` is ``P(K >= k)``.
        first_fail_probability: ``[batch, H + 1]`` containing ``P(K = k)``.
    """

    if judge_logits.ndim != 2:
        raise ValueError(
            "judge_logits must have shape [batch, block_size], got "
            f"{tuple(judge_logits.shape)}."
        )
    batch_size, block_size = judge_logits.shape
    if int(batch_size) <= 0 or int(block_size) <= 0:
        raise ValueError(
            "judge_logits must have positive batch and block dimensions, got "
            f"{tuple(judge_logits.shape)}."
        )

    horizon = int(block_size) - 1
    if valid_horizons is None:
        valid_horizons_tensor = torch.full(
            (batch_size,),
            horizon,
            dtype=torch.int64,
            device=judge_logits.device,
        )
    else:
        valid_horizons_tensor = torch.as_tensor(
            valid_horizons,
            dtype=torch.int64,
            device=judge_logits.device,
        )
        if valid_horizons_tensor.ndim == 0:
            valid_horizons_tensor = valid_horizons_tensor.expand(batch_size)
        if valid_horizons_tensor.ndim != 1 or int(valid_horizons_tensor.numel()) != int(
            batch_size
        ):
            raise ValueError(
                "valid_horizons must be scalar or have shape [batch], got "
                f"{tuple(valid_horizons_tensor.shape)} for batch={batch_size}."
            )
        valid_horizons_tensor = valid_horizons_tensor.clamp(min=0, max=horizon)

    conditional_survival = torch.sigmoid(judge_logits[:, 1:].float())
    raw_prefix_survival = torch.cat(
        (
            torch.ones(
                (batch_size, 1),
                dtype=conditional_survival.dtype,
                device=conditional_survival.device,
            ),
            torch.cumprod(conditional_survival, dim=-1),
        ),
        dim=-1,
    )

    boundary = torch.arange(
        horizon + 1,
        dtype=torch.int64,
        device=judge_logits.device,
    ).unsqueeze(0)
    within_horizon = boundary <= valid_horizons_tensor.unsqueeze(1)
    prefix_survival = torch.where(
        within_horizon,
        raw_prefix_survival,
        torch.zeros_like(raw_prefix_survival),
    )

    reject_probability = torch.cat(
        (
            1.0 - conditional_survival,
            torch.ones(
                (batch_size, 1),
                dtype=conditional_survival.dtype,
                device=conditional_survival.device,
            ),
        ),
        dim=-1,
    )
    before_terminal = boundary < valid_horizons_tensor.unsqueeze(1)
    at_terminal = boundary == valid_horizons_tensor.unsqueeze(1)
    first_fail_probability = torch.where(
        before_terminal,
        raw_prefix_survival * reject_probability,
        torch.where(
            at_terminal,
            raw_prefix_survival,
            torch.zeros_like(raw_prefix_survival),
        ),
    )
    return prefix_survival, first_fail_probability


def compute_joint_judge_accept_len(
    *,
    judge_logits: torch.Tensor,
    threshold: float = 0.5,
    mode: str = "threshold",
    valid_horizons: Optional[Union[int, torch.Tensor]] = None,
    early_cost: float = 1.0,
    late_cost: float = 1.0,
    quantile: float = 0.5,
) -> torch.Tensor:
    """Decode one sequence-level first-fail boundary per judge block.

    ``judge_logits[:, 0]`` belongs to the already-known anchor and is never a
    proposal. Positions 1..B-1 parameterize conditional survival.

    ``threshold`` preserves the legacy consecutive per-token decisions.
    ``expected_cost`` minimizes asymmetric expected absolute boundary error,
    and ``quantile`` selects the requested first-fail categorical quantile.
    """
    mode = normalize_joint_judge_decoder_mode(mode)
    if judge_logits.ndim != 2:
        raise ValueError(
            "judge_logits must have shape [batch, block_size], got "
            f"{tuple(judge_logits.shape)}."
        )
    batch_size, block_size = judge_logits.shape
    if int(batch_size) <= 0 or int(block_size) <= 0:
        raise ValueError(
            "judge_logits must have positive batch and block dimensions, got "
            f"{tuple(judge_logits.shape)}."
        )
    horizon = int(block_size) - 1

    if mode == "threshold":
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"joint judge threshold must be finite and in [0, 1], got {threshold}."
            )
        accept = torch.sigmoid(judge_logits[:, 1:].float()) >= threshold
        if valid_horizons is not None:
            valid_horizons_tensor = torch.as_tensor(
                valid_horizons,
                dtype=torch.int64,
                device=judge_logits.device,
            )
            if valid_horizons_tensor.ndim == 0:
                valid_horizons_tensor = valid_horizons_tensor.expand(batch_size)
            if valid_horizons_tensor.ndim != 1 or int(
                valid_horizons_tensor.numel()
            ) != int(batch_size):
                raise ValueError(
                    "valid_horizons must be scalar or have shape [batch], got "
                    f"{tuple(valid_horizons_tensor.shape)} for batch={batch_size}."
                )
            valid_horizons_tensor = valid_horizons_tensor.clamp(min=0, max=horizon)
            proposal_position = torch.arange(
                horizon,
                dtype=torch.int64,
                device=judge_logits.device,
            ).unsqueeze(0)
            accept &= proposal_position < valid_horizons_tensor.unsqueeze(1)
        return accept.to(torch.int32).cumprod(dim=1).sum(dim=1)

    _, first_fail_probability = compute_joint_judge_first_fail_distribution(
        judge_logits=judge_logits,
        valid_horizons=valid_horizons,
    )
    num_boundaries = horizon + 1

    if valid_horizons is None:
        valid_horizons_tensor = torch.full(
            (batch_size,),
            horizon,
            dtype=torch.int64,
            device=judge_logits.device,
        )
    else:
        valid_horizons_tensor = torch.as_tensor(
            valid_horizons,
            dtype=torch.int64,
            device=judge_logits.device,
        )
        if valid_horizons_tensor.ndim == 0:
            valid_horizons_tensor = valid_horizons_tensor.expand(batch_size)
        valid_horizons_tensor = valid_horizons_tensor.clamp(min=0, max=horizon)

    boundary = torch.arange(
        num_boundaries,
        dtype=torch.int64,
        device=judge_logits.device,
    )
    valid_boundary = boundary.unsqueeze(0) <= valid_horizons_tensor.unsqueeze(1)

    if mode == "quantile":
        quantile = float(quantile)
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError(
                "joint judge boundary quantile must be finite and in [0, 1], "
                f"got {quantile}."
            )
        cumulative_probability = first_fail_probability.cumsum(dim=-1)
        reached = cumulative_probability >= quantile
        # Floating-point summation can leave the terminal CDF infinitesimally
        # below one. The valid terminal boundary must nevertheless be selected.
        reached |= boundary.unsqueeze(0) == valid_horizons_tensor.unsqueeze(1)
        reached &= valid_boundary
        return reached.to(torch.int32).argmax(dim=-1)

    early_cost = float(early_cost)
    late_cost = float(late_cost)
    if (
        not math.isfinite(early_cost)
        or not math.isfinite(late_cost)
        or early_cost < 0.0
        or late_cost < 0.0
        or early_cost + late_cost <= 0.0
    ):
        raise ValueError(
            "joint judge early/late costs must be finite, non-negative, and "
            f"not both zero, got early={early_cost}, late={late_cost}."
        )

    action = boundary.view(1, num_boundaries, 1)
    outcome = boundary.view(1, 1, num_boundaries)
    delta = action - outcome
    boundary_cost = (
        float(late_cost) * delta.clamp_min(0).float()
        + float(early_cost) * (-delta).clamp_min(0).float()
    )
    expected_cost = (boundary_cost * first_fail_probability.unsqueeze(1)).sum(dim=-1)
    expected_cost.masked_fill_(~valid_boundary, torch.inf)
    return expected_cost.argmin(dim=-1)


def append_joint_judge_accepted_prefix_(
    *,
    verify_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    accumulated_lens: torch.Tensor,
    active: torch.Tensor,
    accepted_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one block's accepted prefix to an active-chain verify tape.

    The block anchor is already present in the accumulated tape, so only
    ``draft_tokens[:, 1 : k + 1]`` is appended.  The next block anchor is
    ``draft_tokens[:, k]``; therefore it is the last accepted token when
    ``k > 0`` and the unchanged anchor when ``k == 0``.
    """
    if verify_tokens.ndim != 2 or draft_tokens.ndim != 2:
        raise ValueError("verify_tokens and draft_tokens must both be rank-2.")
    bs, block_size = draft_tokens.shape
    if int(verify_tokens.shape[0]) != int(bs):
        raise ValueError(
            "active-chain batch mismatch: verify_tokens="
            f"{tuple(verify_tokens.shape)}, draft_tokens={tuple(draft_tokens.shape)}."
        )
    for name, value in (
        ("accumulated_lens", accumulated_lens),
        ("active", active),
        ("accepted_lens", accepted_lens),
    ):
        if value.ndim != 1 or int(value.numel()) != int(bs):
            raise ValueError(
                f"{name} must have shape [batch], got {tuple(value.shape)}."
            )

    effective = torch.where(
        active,
        accepted_lens.to(torch.long),
        torch.zeros_like(accepted_lens, dtype=torch.long),
    )
    proposal_offsets = torch.arange(
        block_size - 1, dtype=torch.long, device=draft_tokens.device
    ).unsqueeze(0)
    dst_cols = (
        1 + accumulated_lens.to(torch.long).unsqueeze(1) + proposal_offsets
    )
    take = proposal_offsets < effective.unsqueeze(1)
    rows = torch.arange(bs, dtype=torch.long, device=draft_tokens.device).unsqueeze(1)
    rows = rows.expand(-1, block_size - 1)
    verify_tokens[rows[take], dst_cols[take]] = draft_tokens[:, 1:][take]

    next_root = draft_tokens.gather(1, effective.unsqueeze(1)).squeeze(1)
    next_accumulated = accumulated_lens + effective.to(accumulated_lens.dtype)
    next_active = active & (effective > 0)
    return effective, next_root, next_accumulated, next_active


def compute_dflash_sampling_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            f"next_token_logits must be 2D, got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # Build target distribution once over all verify rows.
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        max_top_k = int(repeated_top_ks.max().item())

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if not torch.all(repeated_top_ks == max_top_k):
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    accept_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, accept_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return accept_len, bonus
