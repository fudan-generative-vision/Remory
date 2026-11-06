from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DASYNC_PROPOSAL_MODE_ALIASES = {
    "masked": "masked_block",
    "maskedblock": "masked_block",
    "masked_block": "masked_block",
    "block": "masked_block",
    "blockwise": "masked_block",
    "block_wise": "masked_block",
    "training": "masked_block",
    "train": "masked_block",
    "hybrid": "hybrid_ar",
    "hybridar": "hybrid_ar",
    "hybrid_ar": "hybrid_ar",
    "chunked": "hybrid_ar",
    "chunked_ar": "hybrid_ar",
    "legacy_hybrid": "hybrid_ar",
}


def _dasync_normalize_proposal_mode(value: Optional[str]) -> str:
    raw = str(value or os.environ.get("DASYNC_DRAFT_PROPOSAL_MODE") or "masked_block")
    key = raw.strip().lower().replace("-", "_")
    normalized = _DASYNC_PROPOSAL_MODE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            "DASYNC draft proposal mode must be one of "
            "['masked_block', 'hybrid_ar'], got "
            f"{value!r}."
        )
    return normalized

def _dasync_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dasync_dump_enabled() -> bool:
    return _dasync_env_flag("SGLANG_DASYNC_DUMP", False)


def _dasync_dump_path() -> str:
    return os.environ.get("SGLANG_DASYNC_DUMP_FILE", "/tmp/sglang_dasync_trace.jsonl")


def _dasync_dump_max_items() -> int:
    try:
        return max(1, int(os.environ.get("SGLANG_DASYNC_DUMP_MAX_ITEMS", "128")))
    except Exception:
        return 128


def _dasync_clip_list(values, limit: Optional[int] = None):
    if values is None:
        return []
    if limit is None:
        limit = _dasync_dump_max_items()
    out = []
    for x in list(values)[: int(limit)]:
        try:
            if isinstance(x, float):
                out.append(float(x))
            else:
                out.append(int(x))
        except Exception:
            out.append(x)
    return out


def _dasync_decode_ids(tokenizer, ids) -> Optional[List[str]]:
    if tokenizer is None:
        return None
    if not _dasync_env_flag("SGLANG_DASYNC_DUMP_DECODE", True):
        return None
    try:
        return [
            tokenizer.decode([int(x)], skip_special_tokens=False)
            for x in _dasync_clip_list(ids)
        ]
    except Exception as exc:
        return [f"<decode-error:{type(exc).__name__}>"]


def _dasync_dump_event(event: Dict[str, Any]) -> None:
    if not _dasync_dump_enabled():
        return
    try:
        path = _dasync_dump_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", time.time())
        event.setdefault("pid", os.getpid())
        with open(path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.debug("failed to write DASYNC dump event", exc_info=True)


def _dasync_load_dump_tokenizer(tokenizer_path: Optional[str], *, trust_remote_code: bool):
    if not _dasync_dump_enabled():
        return None
    if not _dasync_env_flag("SGLANG_DASYNC_DUMP_DECODE", True):
        return None
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(trust_remote_code),
        )
        _dasync_dump_event(
            {
                "role": "tokenizer",
                "event": "tokenizer_ready",
                "tokenizer_path": tokenizer_path,
                "vocab_size": len(tokenizer),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            }
        )
        return tokenizer
    except Exception:
        _dasync_dump_event(
            {
                "role": "tokenizer",
                "event": "tokenizer_load_failed",
                "tokenizer_path": tokenizer_path,
                "traceback": traceback.format_exc(),
            }
        )
        return None


@dataclass
class _DraftRequestState:
    rid: str
    epoch: int
    prefix_ids: List[int]
    cache_token_ids: List[int]
    past_key_values: Any = None
    next_logits: Optional[torch.Tensor] = None
    last_proposal_epoch: Optional[int] = None


def _resolve_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None
    name = str(dtype_name).lower()
    if name in {"auto", "none"}:
        return "auto"
    if name in {"half", "float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float", "float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported DASYNC draft dtype: {dtype_name!r}")


def _select_token(logits: torch.Tensor, *, temperature: float, top_p: float):
    logits = logits.float().view(-1)
    if temperature <= 0.0:
        probs = torch.softmax(logits, dim=-1)
        prob, token = torch.max(probs, dim=-1)
        return int(token.item()), float(prob.item())

    probs = torch.softmax(logits / float(temperature), dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= float(top_p)
        keep[0] = True
        filtered_probs = sorted_probs[keep]
        filtered_idx = sorted_idx[keep]
        filtered_probs = filtered_probs / filtered_probs.sum()
        sample = torch.multinomial(filtered_probs, num_samples=1)
        token = filtered_idx[sample]
        prob = probs[token]
        return int(token.item()), float(prob.item())

    token = torch.multinomial(probs, num_samples=1)
    prob = probs[token]
    return int(token.item()), float(prob.item())


def _dasync_pick_text_components(model: nn.Module) -> Tuple[nn.Module, nn.Module, Any]:
    """Extract the Qwen3.5 text backbone and LM head from common wrappers."""

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise TypeError(f"{type(model).__name__} has no lm_head")

    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    language_model = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and hasattr(candidate, "layers")
            and hasattr(candidate, "embed_tokens")
            and hasattr(candidate, "rotary_emb")
        ),
        None,
    )
    if language_model is None:
        raise TypeError(
            "Could not locate the Qwen3.5 text backbone. Expected one of "
            "model.language_model, language_model, or model to expose "
            "layers/embed_tokens/rotary_emb."
        )
    return (
        language_model,
        lm_head,
        getattr(language_model, "config", getattr(model, "config", None)),
    )


def _dasync_load_transformers_model(
    pretrained_model_name_or_path: str,
    *,
    dtype: Optional[torch.dtype],
    attention_backend: Optional[str],
    trust_remote_code: bool,
    revision: Optional[str] = None,
):
    """Load Qwen3.5 through whichever Transformers Auto class exists locally."""

    import transformers

    auto_class_names = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
    )
    errors: List[str] = []
    for class_name in auto_class_names:
        auto_cls = getattr(transformers, class_name, None)
        if auto_cls is None:
            continue
        base_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        if revision is not None:
            base_kwargs["revision"] = revision
        if dtype is not None and dtype != "auto":
            base_kwargs["torch_dtype"] = dtype
        attempts: List[Dict[str, Any]] = []
        if attention_backend:
            attempts.append({**base_kwargs, "attn_implementation": attention_backend})
        attempts.append(base_kwargs)

        for kwargs in attempts:
            try:
                return auto_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception as exc:
                errors.append(
                    f"{class_name}({sorted(kwargs)}): {type(exc).__name__}: {exc}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    joined = (
        "\n  - ".join(errors) if errors else "no compatible Auto class was installed"
    )
    raise RuntimeError(
        "Unable to load the Qwen3.5 checkpoint through Transformers:\n  - " + joined
    )


def _dasync_create_prefix_block_additive_mask(
    *,
    padding_mask: torch.Tensor,
    block_starts: torch.LongTensor,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Prefix causal + current block bidirectional additive mask."""

    batch_size, seq_len = padding_mask.shape
    device = padding_mask.device
    q = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    k = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    starts = block_starts.view(batch_size, 1, 1)
    ends = starts + int(block_size)

    q_is_prefix = q < starts
    q_is_block = (q >= starts) & (q < ends)
    k_is_prefix = k < starts
    k_is_block = (k >= starts) & (k < ends)

    prefix_causal = q_is_prefix & k_is_prefix & (k <= q)
    block_visible = q_is_block & (k_is_prefix | k_is_block)
    allowed = prefix_causal | block_visible

    valid = padding_mask.to(torch.bool)
    allowed = allowed & valid[:, :, None] & valid[:, None, :]

    additive = torch.zeros((batch_size, seq_len, seq_len), dtype=dtype, device=device)
    additive.masked_fill_(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)


@dataclass
class _DAsyncIsomorphicBlockOutput:
    logits: torch.Tensor


class _DAsyncIsomorphicBlockDrafter(nn.Module):
    """SpecForge isomorphic Qwen3.5 block drafter loaded from draft_model.pt."""

    def __init__(self, language_model: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.config = getattr(language_model, "config", None)
        self._dasync_block_drafter = True

        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "linear_attention" if hasattr(layer, "linear_attn") else "full_attention"
                for layer in self.language_model.layers
            ]
        if len(layer_types) != len(self.language_model.layers):
            raise ValueError(
                "config.layer_types and language_model.layers have different lengths"
            )
        self.layer_types = list(layer_types)

        for layer, layer_type in zip(self.language_model.layers, self.layer_types):
            if layer_type == "full_attention":
                attention = getattr(layer, "self_attn", None)
                if attention is not None:
                    attention.is_causal = False

    @classmethod
    def from_base_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        *,
        dtype: Optional[torch.dtype],
        attention_backend: Optional[str],
        trust_remote_code: bool,
        revision: Optional[str] = None,
    ) -> "_DAsyncIsomorphicBlockDrafter":
        if attention_backend is not None and str(attention_backend).lower() not in {
            "sdpa",
            "eager",
        }:
            # The block mask is a dense 4-D additive mask. Standard FA/FA3 causal
            # entrypoints cannot express the bidirectional block topology.
            logger.warning(
                "DASYNC SpecForge block drafter forces attn_implementation='sdpa' "
                "instead of %r.",
                attention_backend,
            )
            attention_backend = "sdpa"
        if attention_backend is None:
            attention_backend = "sdpa"

        full_model = _dasync_load_transformers_model(
            pretrained_model_name_or_path,
            dtype=dtype,
            attention_backend=attention_backend,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        language_model, lm_head, _ = _dasync_pick_text_components(full_model)
        if getattr(language_model, "config", None) is not None:
            language_model.config._attn_implementation = attention_backend
        for layer in language_model.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None and getattr(attention, "config", None) is not None:
                attention.config._attn_implementation = attention_backend
        drafter = cls(language_model=language_model, lm_head=lm_head)
        del full_model
        gc.collect()
        return drafter

    @property
    def vocab_size(self) -> int:
        return int(self.lm_head.weight.shape[0])

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        text_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = {
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "position_ids": text_position_ids,
            "use_cache": False,
        }
        signature = inspect.signature(layer.forward)
        if "past_key_values" in signature.parameters:
            kwargs["past_key_values"] = None
        elif "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = None
        output = layer(hidden_states, **kwargs)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        padding_mask: torch.Tensor,
        full_attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        block_starts: torch.LongTensor,
        block_size: int,
    ) -> _DAsyncIsomorphicBlockOutput:
        hidden_states = self.language_model.embed_tokens(input_ids)
        text_position_ids = position_ids
        rope_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.language_model.rotary_emb(
            hidden_states, rope_position_ids
        )

        for layer_idx, layer in enumerate(self.language_model.layers):
            layer_mask = (
                padding_mask
                if self.layer_types[layer_idx] == "linear_attention"
                else full_attention_mask
            )
            hidden_states = self._call_layer(
                layer,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                text_position_ids=text_position_ids,
            )

        hidden_states = self.language_model.norm(hidden_states)
        offsets = torch.arange(int(block_size), device=input_ids.device).view(1, -1)
        block_indices = block_starts.view(-1, 1) + offsets
        gather_index = block_indices.unsqueeze(-1).expand(
            -1, -1, hidden_states.shape[-1]
        )
        block_hidden = torch.gather(hidden_states, dim=1, index=gather_index)
        return _DAsyncIsomorphicBlockOutput(logits=self.lm_head(block_hidden))

    def forward_causal_next_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return next-token logits for a raw-token prefix using causal attention."""

        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, seq_len>=1]")
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = self.language_model.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).view(1, -1)
        position_ids = position_ids.expand(batch_size, -1)
        rope_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.language_model.rotary_emb(
            hidden_states, rope_position_ids
        )
        padding_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        causal_mask = torch.zeros(
            (batch_size, seq_len, seq_len),
            dtype=hidden_states.dtype,
            device=device,
        )
        causal_allowed = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        causal_mask.masked_fill_(~causal_allowed.view(1, seq_len, seq_len), torch.finfo(hidden_states.dtype).min)
        causal_mask = causal_mask.unsqueeze(1)

        for layer_idx, layer in enumerate(self.language_model.layers):
            layer_mask = (
                padding_mask
                if self.layer_types[layer_idx] == "linear_attention"
                else causal_mask
            )
            hidden_states = self._call_layer(
                layer,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                text_position_ids=position_ids,
            )

        hidden_states = self.language_model.norm(hidden_states)
        return self.lm_head(hidden_states[:, -1, :])


def _dasync_extract_eos_token_id(model) -> Optional[int]:
    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(getattr(model, "language_model", None), "config", None), "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    return int(eos_token_id) if eos_token_id is not None else None


def _dasync_resolve_specforge_mask_token_id(metadata: Dict[str, Any], vocab_size: int) -> int:
    mask_id = metadata.get("mask_token_id")
    if mask_id is None:
        # SpecForge Qwen3.5 training default: an existing reserved vocabulary entry.
        mask_id = 248070
    mask_id = int(mask_id)
    if not 0 <= mask_id < int(vocab_size):
        raise ValueError(
            f"mask_token_id={mask_id} is outside draft vocabulary size {vocab_size}"
        )
    return mask_id




def _dasync_first_int_attr(objects, *names: str) -> Optional[int]:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _dasync_get_submodule(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    obj: Any = root
    for part in dotted_name.split("."):
        if part == "":
            continue
        try:
            if part.isdigit() and isinstance(obj, (list, tuple, nn.ModuleList)):
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        except (AttributeError, IndexError, TypeError):
            return None
    return obj if isinstance(obj, nn.Module) else None


def _dasync_infer_qwen35_gdn_layout(
    model: nn.Module,
    *,
    prefix: str,
    qkv_shape: torch.Size,
    z_shape: torch.Size,
    b_shape: torch.Size,
) -> Tuple[int, int, int, int]:
    """Infer Qwen3.5 GDN projection layout for split/fused checkpoint conversion."""

    if len(qkv_shape) != 2 or len(z_shape) != 2 or len(b_shape) != 2:
        raise ValueError(
            f"Unexpected GDN projection ranks for {prefix}: "
            f"qkv={tuple(qkv_shape)}, z={tuple(z_shape)}, b={tuple(b_shape)}"
        )
    if qkv_shape[1] != z_shape[1] or qkv_shape[1] != b_shape[1]:
        raise ValueError(
            f"Unexpected GDN projection hidden dims for {prefix}: "
            f"qkv={tuple(qkv_shape)}, z={tuple(z_shape)}, b={tuple(b_shape)}"
        )

    value_dim = int(z_shape[0])
    key_twice = int(qkv_shape[0]) - value_dim
    if key_twice <= 0 or key_twice % 2 != 0:
        raise ValueError(
            f"Cannot infer key_dim for {prefix}: qkv={tuple(qkv_shape)}, z={tuple(z_shape)}"
        )
    key_dim = key_twice // 2

    module_name = prefix[:-1] if prefix.endswith(".") else prefix
    module = _dasync_get_submodule(model, module_name)
    config = getattr(model, "config", None)
    language_config = getattr(getattr(model, "language_model", None), "config", None)
    objects = (module, config, language_config)

    num_key_heads = _dasync_first_int_attr(
        objects,
        "linear_num_key_heads",
        "num_key_heads",
        "num_kv_heads",
    )
    num_value_heads = _dasync_first_int_attr(
        objects,
        "linear_num_value_heads",
        "num_value_heads",
    )
    key_head_dim = _dasync_first_int_attr(
        objects,
        "linear_key_head_dim",
        "key_head_dim",
    )
    value_head_dim = _dasync_first_int_attr(
        objects,
        "linear_value_head_dim",
        "value_head_dim",
    )

    if num_value_heads is None:
        # Split Qwen3.5 checkpoints store B/A as one row per value head.
        num_value_heads = int(b_shape[0])
    if value_head_dim is None:
        if value_dim % num_value_heads != 0:
            raise ValueError(
                f"Cannot infer value_head_dim for {prefix}: "
                f"value_dim={value_dim}, num_value_heads={num_value_heads}"
            )
        value_head_dim = value_dim // num_value_heads
    if num_key_heads is None:
        if key_head_dim is not None:
            if key_dim % key_head_dim != 0:
                raise ValueError(
                    f"Cannot infer num_key_heads for {prefix}: "
                    f"key_dim={key_dim}, key_head_dim={key_head_dim}"
                )
            num_key_heads = key_dim // key_head_dim
        else:
            # Qwen3.5 configs normally expose linear_num_key_heads.  This fallback
            # is conservative: key heads must group value heads exactly.
            candidates = [
                h
                for h in range(1, num_value_heads + 1)
                if num_value_heads % h == 0 and key_dim % h == 0
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot uniquely infer num_key_heads for {prefix}; candidates={candidates}. "
                    "Please install the matching Qwen3.5/FLA backend or expose linear_num_key_heads in the config."
                )
            num_key_heads = candidates[0]
    if key_head_dim is None:
        if key_dim % num_key_heads != 0:
            raise ValueError(
                f"Cannot infer key_head_dim for {prefix}: "
                f"key_dim={key_dim}, num_key_heads={num_key_heads}"
            )
        key_head_dim = key_dim // num_key_heads

    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            f"Invalid GDN heads for {prefix}: num_value_heads={num_value_heads}, "
            f"num_key_heads={num_key_heads}"
        )
    if num_key_heads * key_head_dim != key_dim:
        raise ValueError(
            f"Invalid GDN key layout for {prefix}: heads={num_key_heads}, "
            f"head_dim={key_head_dim}, key_dim={key_dim}"
        )
    if num_value_heads * value_head_dim != value_dim:
        raise ValueError(
            f"Invalid GDN value layout for {prefix}: heads={num_value_heads}, "
            f"head_dim={value_head_dim}, value_dim={value_dim}"
        )
    if int(b_shape[0]) != num_value_heads:
        raise ValueError(
            f"Unexpected B/A projection rows for {prefix}: got {int(b_shape[0])}, "
            f"expected {num_value_heads}"
        )
    return num_key_heads, num_value_heads, key_head_dim, value_head_dim


def _dasync_merge_qwen35_gdn_projections(
    qkv_weight: torch.Tensor,
    z_weight: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_heads_per_key = num_value_heads // num_key_heads
    hidden_size = qkv_weight.shape[1]
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim

    query, key, value = torch.split(qkv_weight, [key_dim, key_dim, value_dim], dim=0)
    query = query.view(num_key_heads, key_head_dim, hidden_size)
    key = key.view(num_key_heads, key_head_dim, hidden_size)
    value = value.view(num_key_heads, value_heads_per_key * value_head_dim, hidden_size)
    z = z_weight.view(num_key_heads, value_heads_per_key * value_head_dim, hidden_size)
    qkvz = torch.cat((query, key, value, z), dim=1).reshape(-1, hidden_size)

    b = b_weight.view(num_key_heads, value_heads_per_key, hidden_size)
    a = a_weight.view(num_key_heads, value_heads_per_key, hidden_size)
    ba = torch.cat((b, a), dim=1).reshape(-1, hidden_size)
    return qkvz.contiguous(), ba.contiguous()


def _dasync_split_qwen35_gdn_projections(
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    value_heads_per_key = num_value_heads // num_key_heads
    hidden_size = qkvz_weight.shape[1]
    qkvz_group = 2 * key_head_dim + 2 * value_heads_per_key * value_head_dim
    ba_group = 2 * value_heads_per_key

    expected_qkvz = (num_key_heads * qkvz_group, hidden_size)
    expected_ba = (num_key_heads * ba_group, hidden_size)
    if tuple(qkvz_weight.shape) != expected_qkvz:
        raise ValueError(
            f"Unexpected fused in_proj_qkvz shape {tuple(qkvz_weight.shape)}; "
            f"expected {expected_qkvz}"
        )
    if tuple(ba_weight.shape) != expected_ba:
        raise ValueError(
            f"Unexpected fused in_proj_ba shape {tuple(ba_weight.shape)}; "
            f"expected {expected_ba}"
        )

    qkvz = qkvz_weight.view(num_key_heads, qkvz_group, hidden_size)
    query, key, value, z = torch.split(
        qkvz,
        [
            key_head_dim,
            key_head_dim,
            value_heads_per_key * value_head_dim,
            value_heads_per_key * value_head_dim,
        ],
        dim=1,
    )
    qkv = torch.cat(
        (query.reshape(-1, hidden_size), key.reshape(-1, hidden_size), value.reshape(-1, hidden_size)),
        dim=0,
    )
    z = z.reshape(-1, hidden_size)

    ba = ba_weight.view(num_key_heads, ba_group, hidden_size)
    b, a = torch.split(ba, [value_heads_per_key, value_heads_per_key], dim=1)
    return qkv.contiguous(), z.contiguous(), b.reshape(-1, hidden_size).contiguous(), a.reshape(-1, hidden_size).contiguous()


def _dasync_normalize_specforge_gdn_state_dict(
    model: nn.Module, state: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Convert Qwen3.5 GDN split/fused projection names to the runtime backend.

    SpecForge checkpoints may be written with FLA fast-path fused projections
    (in_proj_qkvz/in_proj_ba), while the runtime environment may instantiate the
    PyTorch fallback split projections (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a),
    or vice versa.  The math is the same; only parameter packing differs.
    """

    target = model.state_dict()
    state = dict(state)
    converted: List[str] = []
    candidate_prefixes = sorted(
        {
            key[: -len("in_proj_qkv.weight")]
            for key in target.keys()
            if key.endswith("linear_attn.in_proj_qkv.weight")
        }
        | {
            key[: -len("in_proj_qkvz.weight")]
            for key in target.keys()
            if key.endswith("linear_attn.in_proj_qkvz.weight")
        }
        | {
            key[: -len("in_proj_qkv.weight")]
            for key in state.keys()
            if key.endswith("linear_attn.in_proj_qkv.weight")
        }
        | {
            key[: -len("in_proj_qkvz.weight")]
            for key in state.keys()
            if key.endswith("linear_attn.in_proj_qkvz.weight")
        }
    )

    for prefix in candidate_prefixes:
        split_keys = (
            prefix + "in_proj_qkv.weight",
            prefix + "in_proj_z.weight",
            prefix + "in_proj_b.weight",
            prefix + "in_proj_a.weight",
        )
        fused_keys = (
            prefix + "in_proj_qkvz.weight",
            prefix + "in_proj_ba.weight",
        )
        target_has_split = all(key in target for key in split_keys)
        target_has_fused = all(key in target for key in fused_keys)
        state_has_split = all(key in state for key in split_keys)
        state_has_fused = all(key in state for key in fused_keys)

        if target_has_split and state_has_fused and not state_has_split:
            layout = _dasync_infer_qwen35_gdn_layout(
                model,
                prefix=prefix,
                qkv_shape=target[split_keys[0]].shape,
                z_shape=target[split_keys[1]].shape,
                b_shape=target[split_keys[2]].shape,
            )
            qkv, z, b, a = _dasync_split_qwen35_gdn_projections(
                state[fused_keys[0]], state[fused_keys[1]],
                num_key_heads=layout[0],
                num_value_heads=layout[1],
                key_head_dim=layout[2],
                value_head_dim=layout[3],
            )
            state[split_keys[0]] = qkv
            state[split_keys[1]] = z
            state[split_keys[2]] = b
            state[split_keys[3]] = a
            del state[fused_keys[0]]
            del state[fused_keys[1]]
            converted.append(prefix + "fused->split")

        elif target_has_fused and state_has_split and not state_has_fused:
            layout = _dasync_infer_qwen35_gdn_layout(
                model,
                prefix=prefix,
                qkv_shape=state[split_keys[0]].shape,
                z_shape=state[split_keys[1]].shape,
                b_shape=state[split_keys[2]].shape,
            )
            qkvz, ba = _dasync_merge_qwen35_gdn_projections(
                state[split_keys[0]], state[split_keys[1]], state[split_keys[2]], state[split_keys[3]],
                num_key_heads=layout[0],
                num_value_heads=layout[1],
                key_head_dim=layout[2],
                value_head_dim=layout[3],
            )
            state[fused_keys[0]] = qkvz
            state[fused_keys[1]] = ba
            for key in split_keys:
                del state[key]
            converted.append(prefix + "split->fused")

    return state, converted


def _dasync_load_specforge_isomorphic_checkpoint(
    config: Dict[str, Any],
    *,
    device: str,
    dtype: Optional[torch.dtype],
):
    checkpoint_dir = str(config["model_path"])
    state_path = os.path.join(checkpoint_dir, "draft_model.pt")
    metadata_path = os.path.join(checkpoint_dir, "training_config.json")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(state_path)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    base_model_path = (
        metadata.get("model_path")
        or metadata.get("pretrained_model_name_or_path")
        or metadata.get("base_model_path")
        or config.get("target_model_path")
    )
    if base_model_path is None:
        raise ValueError(
            "SpecForge DASYNC checkpoint requires training_config.json to contain "
            "model_path, or the server config to provide target_model_path."
        )

    attention_backend = (
        config.get("attn_implementation")
        or metadata.get("attention_backend")
        or "sdpa"
    )
    base_revision = (
        metadata.get("model_revision")
        or metadata.get("revision")
        or config.get("revision")
    )
    model = _DAsyncIsomorphicBlockDrafter.from_base_checkpoint(
        str(base_model_path),
        dtype=dtype,
        attention_backend=str(attention_backend),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        revision=base_revision,
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(
            f"SpecForge DASYNC checkpoint {state_path} should contain a state_dict, "
            f"got {type(state).__name__}"
        )
    state, converted_gdn = _dasync_normalize_specforge_gdn_state_dict(model, state)
    if converted_gdn:
        logger.info(
            "DASYNC normalized %d Qwen3.5 GDN projection groups for runtime backend; first=%s",
            len(converted_gdn),
            converted_gdn[0],
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_unexpected_prefixes = ("accept_head.",)
    blocking_unexpected = [
        key
        for key in unexpected
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    if missing or blocking_unexpected:
        raise RuntimeError(
            "SpecForge DASYNC checkpoint mismatch after GDN projection normalization. "
            f"missing={missing[:10]}, unexpected={blocking_unexpected[:10]}"
        )
    ignored_unexpected = [
        key for key in unexpected if key.startswith(allowed_unexpected_prefixes)
    ]
    if ignored_unexpected:
        logger.info(
            "Ignoring %d training-only DASYNC checkpoint keys; first=%s",
            len(ignored_unexpected),
            ignored_unexpected[0],
        )

    model._dasync_mask_token_id = _dasync_resolve_specforge_mask_token_id(
        metadata, model.vocab_size
    )
    model._dasync_pad_token_id = int(metadata.get("pad_token_id") or 0)
    model._dasync_trained_max_future_tokens = int(
        metadata.get("current_num_draft_tokens")
        or metadata.get("num_draft_tokens_start")
        or max(1, int(metadata.get("current_block_size_including_anchor") or 33) - 1)
    )
    model._dasync_checkpoint_dir = checkpoint_dir
    model._dasync_base_model_path = str(base_model_path)
    model.to(device)
    model.eval()
    return model, _dasync_extract_eos_token_id(model), "specforge_isomorphic_block"


def _dasync_load_hf_causal_draft_model(
    config: Dict[str, Any],
    *,
    device: str,
    dtype: Optional[torch.dtype],
):
    model = _dasync_load_transformers_model(
        str(config["model_path"]),
        dtype=dtype,
        attention_backend=config.get("attn_implementation"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        revision=config.get("revision"),
    )
    model.to(device)
    model.eval()
    return model, _dasync_extract_eos_token_id(model), "hf_causal"


def _dasync_load_draft_model(
    config: Dict[str, Any],
    *,
    device: str,
    dtype: Optional[torch.dtype],
):
    model_path = str(config["model_path"])
    if os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "draft_model.pt")
    ):
        return _dasync_load_specforge_isomorphic_checkpoint(
            config, device=device, dtype=dtype
        )
    return _dasync_load_hf_causal_draft_model(config, device=device, dtype=dtype)


def _generate_specforge_isomorphic_proposal(
    state: _DraftRequestState,
    *,
    model: _DAsyncIsomorphicBlockDrafter,
    device: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    publish_stride: int,
    publish_cb=None,
    ar_anchor_enabled: bool = False,
    ar_anchor_threshold: float = 0.0,
    ar_anchor_max_steps: int = 0,
) -> Tuple[List[int], List[float]]:
    rolling_prefix = [int(x) for x in state.prefix_ids]
    if len(rolling_prefix) == 0 or max_tokens <= 0:
        return [], []

    trained_chunk = int(getattr(model, "_dasync_trained_max_future_tokens", max_tokens))
    trained_chunk = max(1, min(int(max_tokens), trained_chunk))
    mask_token_id = int(getattr(model, "_dasync_mask_token_id", 248070))
    tokens: List[int] = []
    probs: List[float] = []
    last_published = 0
    publish_stride = max(1, int(publish_stride))
    ar_anchor_threshold = float(ar_anchor_threshold)
    ar_anchor_remaining = max(0, int(ar_anchor_max_steps)) if ar_anchor_enabled else 0
    ar_anchor_ct = 0
    block_chunk_ct = 0

    def maybe_publish(force: bool = False) -> None:
        nonlocal last_published
        if publish_cb is None:
            return
        if force or len(tokens) - last_published >= publish_stride:
            publish_cb(tokens[:], probs[:])
            last_published = len(tokens)

    # SpecForge isomorphic checkpoints are trained with a fixed future block K.
    # To expose longer proposal tapes, run multiple trained-width blocks and feed
    # the previously selected draft tokens back as raw-token prefix. This keeps
    # each forward on-distribution instead of asking a K=32 checkpoint to denoise
    # an unseen K=64 block in one shot.
    while len(tokens) < int(max_tokens):
        remaining = int(max_tokens) - len(tokens)
        future_tokens = min(trained_chunk, remaining)
        block_size = future_tokens + 1
        seq_len = len(rolling_prefix) + future_tokens
        anchor = len(rolling_prefix) - 1

        input_ids = torch.full(
            (1, seq_len),
            fill_value=mask_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids[0, : len(rolling_prefix)] = torch.tensor(
            rolling_prefix, dtype=torch.long, device=device
        )
        padding_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).view(1, -1)
        block_starts = torch.tensor([anchor], dtype=torch.long, device=device)
        mask_dtype = next(model.parameters()).dtype
        full_attention_mask = _dasync_create_prefix_block_additive_mask(
            padding_mask=padding_mask,
            block_starts=block_starts,
            block_size=block_size,
            dtype=mask_dtype,
        )

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                padding_mask=padding_mask,
                full_attention_mask=full_attention_mask,
                position_ids=position_ids,
                block_starts=block_starts,
                block_size=block_size,
            )
        future_logits = output.logits[0, 1 : future_tokens + 1, :]
        block_chunk_ct += 1

        produced_this_chunk = 0
        for row_logits in future_logits:
            token_id, prob = _select_token(
                row_logits, temperature=float(temperature), top_p=float(top_p)
            )
            if prob < float(confidence_threshold) and len(tokens) >= int(min_tokens):
                if ar_anchor_remaining <= 0:
                    maybe_publish(force=True)
                    _dasync_dump_event(
                        {
                            "role": "draft",
                            "event": "draft_hybrid_stop_low_confidence",
                            "prefix_len": len(state.prefix_ids),
                            "proposal_len": len(tokens),
                            "prob": float(prob),
                            "confidence_threshold": float(confidence_threshold),
                            "ar_anchor_count": int(ar_anchor_ct),
                            "block_chunk_count": int(block_chunk_ct),
                        }
                    )
                    return tokens, probs

                with torch.inference_mode():
                    prefix_tensor = torch.tensor(
                        [rolling_prefix], dtype=torch.long, device=device
                    )
                    ar_logits = model.forward_causal_next_logits(prefix_tensor)[0]
                ar_token_id, ar_prob = _select_token(
                    ar_logits, temperature=float(temperature), top_p=float(top_p)
                )
                if ar_prob < ar_anchor_threshold:
                    maybe_publish(force=True)
                    _dasync_dump_event(
                        {
                            "role": "draft",
                            "event": "draft_hybrid_stop_low_ar_anchor",
                            "prefix_len": len(state.prefix_ids),
                            "proposal_len": len(tokens),
                            "block_prob": float(prob),
                            "ar_anchor_prob": float(ar_prob),
                            "ar_anchor_threshold": float(ar_anchor_threshold),
                            "ar_anchor_count": int(ar_anchor_ct),
                            "block_chunk_count": int(block_chunk_ct),
                        }
                    )
                    return tokens, probs

                token_id = int(ar_token_id)
                tokens.append(token_id)
                probs.append(float(ar_prob))
                rolling_prefix.append(token_id)
                produced_this_chunk += 1
                ar_anchor_remaining -= 1
                ar_anchor_ct += 1
                maybe_publish(force=True)
                _dasync_dump_event(
                    {
                        "role": "draft",
                        "event": "draft_hybrid_ar_anchor",
                        "prefix_len": len(state.prefix_ids),
                        "proposal_len": len(tokens),
                        "block_prob": float(prob),
                        "ar_anchor_token_id": int(token_id),
                        "ar_anchor_prob": float(ar_prob),
                        "ar_anchor_count": int(ar_anchor_ct),
                    }
                )
                if eos_token_id is not None and token_id == int(eos_token_id):
                    maybe_publish(force=True)
                    return tokens, probs
                break
            token_id = int(token_id)
            tokens.append(token_id)
            probs.append(float(prob))
            rolling_prefix.append(token_id)
            produced_this_chunk += 1
            maybe_publish(force=(len(tokens) == 1))
            if eos_token_id is not None and token_id == int(eos_token_id):
                maybe_publish(force=True)
                return tokens, probs

        if produced_this_chunk == 0:
            break

    maybe_publish(force=True)
    if ar_anchor_ct > 0:
        _dasync_dump_event(
            {
                "role": "draft",
                "event": "draft_hybrid_rollout_done",
                "prefix_len": len(state.prefix_ids),
                "proposal_len": len(tokens),
                "ar_anchor_count": int(ar_anchor_ct),
                "block_chunk_count": int(block_chunk_ct),
            }
        )

    return tokens, probs


def _generate_specforge_isomorphic_proposals_batched(
    states: List[_DraftRequestState],
    *,
    model: _DAsyncIsomorphicBlockDrafter,
    device: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    publish_stride: int,
    publish_cbs: List[Any],
) -> List[Tuple[List[int], List[float]]]:
    """Generate one trained-width block for several requests in one forward."""

    if not states:
        return []
    if max_tokens <= 0:
        return [([], []) for _ in states]

    trained_chunk = int(getattr(model, "_dasync_trained_max_future_tokens", max_tokens))
    future_tokens = min(int(max_tokens), max(1, trained_chunk))
    mask_token_id = int(getattr(model, "_dasync_mask_token_id", 248070))
    prefixes = [[int(x) for x in state.prefix_ids] for state in states]
    if any(len(prefix) == 0 for prefix in prefixes):
        return [([], []) for _ in states]

    block_size = future_tokens + 1
    seq_lens = [len(prefix) + future_tokens for prefix in prefixes]
    max_seq_len = max(seq_lens)
    batch_size = len(prefixes)

    input_ids = torch.full(
        (batch_size, max_seq_len),
        fill_value=mask_token_id,
        dtype=torch.long,
        device=device,
    )
    padding_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
    block_starts = torch.empty((batch_size,), dtype=torch.long, device=device)
    for i, prefix in enumerate(prefixes):
        prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
        input_ids[i, : len(prefix)] = prefix_tensor
        padding_mask[i, : seq_lens[i]] = True
        block_starts[i] = len(prefix) - 1

    position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device).view(1, -1)
    position_ids = position_ids.expand(batch_size, -1)
    mask_dtype = next(model.parameters()).dtype
    full_attention_mask = _dasync_create_prefix_block_additive_mask(
        padding_mask=padding_mask,
        block_starts=block_starts,
        block_size=block_size,
        dtype=mask_dtype,
    )

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            padding_mask=padding_mask,
            full_attention_mask=full_attention_mask,
            position_ids=position_ids,
            block_starts=block_starts,
            block_size=block_size,
        )

    results: List[Tuple[List[int], List[float]]] = []
    publish_stride = max(1, int(publish_stride))
    for row in range(batch_size):
        tokens: List[int] = []
        probs: List[float] = []
        last_published = 0

        def maybe_publish(force: bool = False) -> None:
            nonlocal last_published
            cb = publish_cbs[row] if row < len(publish_cbs) else None
            if cb is None:
                return
            if force or len(tokens) - last_published >= publish_stride:
                cb(tokens[:], probs[:])
                last_published = len(tokens)

        for row_logits in output.logits[row, 1 : future_tokens + 1, :]:
            token_id, prob = _select_token(
                row_logits, temperature=float(temperature), top_p=float(top_p)
            )
            if prob < float(confidence_threshold) and len(tokens) >= int(min_tokens):
                break
            token_id = int(token_id)
            tokens.append(token_id)
            probs.append(float(prob))
            maybe_publish(force=(len(tokens) == 1))
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
        maybe_publish(force=True)
        results.append((tokens, probs))

    _dasync_dump_event(
        {
            "role": "draft",
            "event": "draft_batched_block_forward",
            "proposal_mode": "masked_block",
            "batch_size": int(batch_size),
            "future_tokens": int(future_tokens),
            "max_seq_len": int(max_seq_len),
            "proposal_lens": [len(tokens) for tokens, _ in results],
        }
    )
    return results


def _generate_specforge_isomorphic_masked_block_proposal(
    state: _DraftRequestState,
    *,
    model: _DAsyncIsomorphicBlockDrafter,
    device: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    publish_stride: int,
    publish_cb=None,
) -> Tuple[List[int], List[float]]:
    """Generate exactly one training-shaped masked block proposal.

    SpecForge isomorphic checkpoints are trained on
    verified_prefix + [MASK] * K.  This path intentionally does not append
    selected draft tokens to the prefix to request another chunk: Qwen3.5's
    linear-attention recurrence would then scan generated raw tokens that were
    never present in training.
    """

    results = _generate_specforge_isomorphic_proposals_batched(
        [state],
        model=model,
        device=device,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        publish_stride=publish_stride,
        publish_cbs=[publish_cb],
    )
    return results[0] if results else ([], [])


def _advance_cache_with_token(state: _DraftRequestState, model, device: str, token_id: int) -> None:
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, past_key_values=state.past_key_values, use_cache=True)
    state.past_key_values = out.past_key_values
    state.cache_token_ids.append(int(token_id))
    state.next_logits = out.logits[:, -1, :]


def _ensure_prefix_cache(state: _DraftRequestState, model, device: str) -> None:
    prefix = [int(x) for x in state.prefix_ids]
    if len(prefix) == 0:
        state.past_key_values = None
        state.next_logits = None
        state.cache_token_ids = []
        return

    if state.cache_token_ids == prefix and state.next_logits is not None:
        return

    if (
        state.past_key_values is not None
        and len(state.cache_token_ids) <= len(prefix)
        and prefix[: len(state.cache_token_ids)] == state.cache_token_ids
        and state.next_logits is not None
    ):
        for token_id in prefix[len(state.cache_token_ids) :]:
            _advance_cache_with_token(state, model, device, int(token_id))
        return

    # Rejections generally make the verified prefix diverge from the optimistic
    # rollout.  A robust reset is cheaper than trying to crop every possible HF
    # cache representation correctly.
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    state.past_key_values = out.past_key_values
    state.cache_token_ids = list(prefix)
    state.next_logits = out.logits[:, -1, :]


def _generate_proposal(
    state: _DraftRequestState,
    *,
    model,
    device: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    publish_stride: int,
    publish_cb=None,
    proposal_mode: str = "masked_block",
    ar_anchor_enabled: bool = False,
    ar_anchor_threshold: float = 0.0,
    ar_anchor_max_steps: int = 0,
) -> Tuple[List[int], List[float]]:
    if getattr(model, "_dasync_block_drafter", False):
        proposal_mode = _dasync_normalize_proposal_mode(proposal_mode)
        if proposal_mode == "masked_block":
            if ar_anchor_enabled:
                raise ValueError(
                    "DASYNC AR anchor is only valid with proposal_mode=hybrid_ar. "
                    "masked_block keeps inference input identical to SpecForge "
                    "training: verified_prefix + [MASK] * K."
                )
            return _generate_specforge_isomorphic_masked_block_proposal(
                state,
                model=model,
                device=device,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                confidence_threshold=confidence_threshold,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
                publish_stride=publish_stride,
                publish_cb=publish_cb,
            )
        return _generate_specforge_isomorphic_proposal(
            state,
            model=model,
            device=device,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            publish_stride=publish_stride,
            publish_cb=publish_cb,
            ar_anchor_enabled=ar_anchor_enabled,
            ar_anchor_threshold=ar_anchor_threshold,
            ar_anchor_max_steps=ar_anchor_max_steps,
        )

    _ensure_prefix_cache(state, model, device)
    if state.next_logits is None or max_tokens <= 0:
        return [], []

    tokens: List[int] = []
    probs: List[float] = []
    last_published = 0
    publish_stride = max(1, int(publish_stride))

    def maybe_publish(force: bool = False) -> None:
        nonlocal last_published
        if publish_cb is None or len(tokens) == 0:
            return
        if force or len(tokens) - last_published >= publish_stride:
            publish_cb(tokens[:], probs[:])
            last_published = len(tokens)

    for _ in range(int(max_tokens)):
        token_id, prob = _select_token(
            state.next_logits, temperature=float(temperature), top_p=float(top_p)
        )
        if prob < float(confidence_threshold) and len(tokens) >= int(min_tokens):
            break
        tokens.append(int(token_id))
        probs.append(float(prob))
        maybe_publish(force=(len(tokens) == 1))
        _advance_cache_with_token(state, model, device, int(token_id))
        if eos_token_id is not None and int(token_id) == int(eos_token_id):
            break
    maybe_publish(force=True)
    return tokens, probs
