"""Native plain-DFlow runtime for Stage-L causal Domino checkpoints.

This module intentionally contains no DFly layer-fusion parameters or compute.
The DFlow backbone remains unchanged; the three replicated Domino modules are
consumed by :mod:`sglang.srt.speculative.dflow_domino_worker` after the sole
block forward.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Tuple

import torch
from torch import nn

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflow import DFlowDraftModel


DOMINO_ARCHITECTURE = "DFlowDominoDraftModel"
DOMINO_REMOTE_AUTO_MODEL = "dflow_dfly_domino.DFlowDominoDraftModel"
DOMINO_PREFIX_HIDDEN_SIZE = 1024
DOMINO_RESIDUAL_RANK = 256
DOMINO_LATENT_ABI = {
    "full_tail_gaussian": True,
    "stage_l_padding_policy": "whole_block_zero_only",
    "stage_l_loss_mask_scope": "supervision_only_not_latent",
    "stage_l_block_attention": (
        "prefix_lt_anchor_plus_bidirectional_same_block"
    ),
}
DOMINO_HEAD_KEYS = frozenset(
    {
        "prefix_gru.weight_ih_l0",
        "prefix_gru.weight_hh_l0",
        "residual_down.weight",
        "residual_up.weight",
    }
)

_PLAIN_FORBIDDEN_WEIGHT_PREFIXES = (
    "layer_fusion_",
    "hidden_correction.",
    "embed_proj.",
)


def expected_dflow_domino_head_shapes(
    *,
    hidden_size: int,
    vocab_size: int,
    prefix_hidden_size: int = DOMINO_PREFIX_HIDDEN_SIZE,
    residual_rank: int = DOMINO_RESIDUAL_RANK,
) -> dict[str, tuple[int, ...]]:
    """Return the exact unsharded, replicated Stage-L head inventory."""

    return {
        "prefix_gru.weight_ih_l0": (3 * int(prefix_hidden_size), int(hidden_size)),
        "prefix_gru.weight_hh_l0": (
            3 * int(prefix_hidden_size),
            int(prefix_hidden_size),
        ),
        "residual_down.weight": (
            int(residual_rank),
            int(hidden_size) + int(prefix_hidden_size),
        ),
        "residual_up.weight": (int(vocab_size), int(residual_rank)),
    }


def _config_value_equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, Integral)
            and not isinstance(actual, bool)
            and int(actual) == expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _config_value_equal(item, expected_item)
                for item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def validate_plain_dflow_domino_config(config) -> None:
    """Fail closed unless ``config`` is the exact plain Stage-L serving ABI."""

    mismatches = {}
    architectures = getattr(config, "architectures", None)
    if architectures != [DOMINO_ARCHITECTURE]:
        mismatches["architectures"] = (architectures, [DOMINO_ARCHITECTURE])

    auto_map = getattr(config, "auto_map", None)
    auto_model = auto_map.get("AutoModel") if isinstance(auto_map, Mapping) else None
    if auto_model != DOMINO_REMOTE_AUTO_MODEL:
        mismatches["auto_map.AutoModel"] = (
            auto_model,
            DOMINO_REMOTE_AUTO_MODEL,
        )

    raw_flow = getattr(config, "dflow_config", None)
    if not isinstance(raw_flow, Mapping):
        raise ValueError(
            "Plain DFlow-Domino checkpoint requires a mapping dflow_config."
        )
    flow = dict(raw_flow)
    expected = {
        "stage_l_domino": True,
        "stage_l_source_arm": "base",
        "pure_draft_prefix_len": 1,
        "stage_l_correction_start_slot": 2,
        "one_step_direct": True,
        "serving_drafter_forwards": 1,
        "stage_l_head_storage": "unified_model_safetensors",
        "stage_l_head_state_keys": sorted(DOMINO_HEAD_KEYS),
        "stage_l_prefix_hidden_size": DOMINO_PREFIX_HIDDEN_SIZE,
        "stage_l_residual_rank": DOMINO_RESIDUAL_RANK,
        "stage_l_slot1_base_only": True,
        **DOMINO_LATENT_ABI,
    }
    mismatches.update(
        {
            f"dflow_config.{key}": (flow.get(key), value)
            for key, value in expected.items()
            if not _config_value_equal(flow.get(key), value)
        }
    )

    forbidden_markers = sorted(
        key
        for key in flow
        if key.startswith("layer_fusion_")
        or key
        in {
            "hybrid_target_conditioning",
            "stage_e_dfly_hybrid",
            "full_dfly_adaptation",
            "enable_hidden_correction",
        }
        and flow.get(key) not in (None, False)
    )
    if forbidden_markers:
        mismatches["plain_forbidden_markers"] = (
            forbidden_markers,
            [],
        )

    expected_shapes = expected_dflow_domino_head_shapes(
        hidden_size=int(config.hidden_size),
        vocab_size=int(config.vocab_size),
    )
    configured_shapes = flow.get("stage_l_head_shapes")
    if configured_shapes is not None:
        normalized_shapes = (
            {
                str(key): [int(dim) for dim in value]
                for key, value in configured_shapes.items()
            }
            if isinstance(configured_shapes, Mapping)
            else configured_shapes
        )
        wanted_shapes = {key: list(shape) for key, shape in expected_shapes.items()}
        if normalized_shapes != wanted_shapes:
            mismatches["dflow_config.stage_l_head_shapes"] = (
                normalized_shapes,
                wanted_shapes,
            )

    if mismatches:
        raise ValueError(
            "Plain DFlow-Domino config contract mismatch (actual, expected): "
            f"{mismatches}."
        )


def _normalize_checkpoint_weight_name(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def partition_plain_dflow_domino_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    hidden_size: int,
    vocab_size: int,
) -> tuple[list[Tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Separate and validate the four full, replicated Domino tensors.

    Validation completes before the caller mutates any model parameter.  In
    particular, a plain checkpoint cannot silently carry DFly/HSC tensors and a
    base DFlow checkpoint cannot serve with an uninitialized Domino head.
    """

    expected_shapes = expected_dflow_domino_head_shapes(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    base_weights: list[Tuple[str, torch.Tensor]] = []
    head_weights: dict[str, torch.Tensor] = {}
    normalized_names: set[str] = set()
    for raw_name, value in weights:
        name = _normalize_checkpoint_weight_name(raw_name)
        if name in normalized_names:
            raise ValueError(
                "Plain DFlow-Domino checkpoint contains duplicate normalized "
                f"tensor {name!r}."
            )
        normalized_names.add(name)

        if name in DOMINO_HEAD_KEYS:
            actual = tuple(int(dim) for dim in value.shape)
            expected = expected_shapes[name]
            if actual != expected:
                raise ValueError(
                    f"Plain DFlow-Domino {name} shape mismatch: expected "
                    f"{expected}, got {actual}."
                )
            head_weights[name] = value
            continue

        if name.startswith(("prefix_gru.", "residual_down.", "residual_up.")):
            raise ValueError(
                "Plain DFlow-Domino checkpoint contains unsupported head tensor "
                f"{name!r}."
            )
        if name.startswith(_PLAIN_FORBIDDEN_WEIGHT_PREFIXES):
            raise ValueError(
                "Plain DFlow-Domino checkpoint contains a forbidden fusion/HSC "
                f"tensor {name!r}."
            )
        base_weights.append((raw_name, value))

    missing = DOMINO_HEAD_KEYS - head_weights.keys()
    if missing:
        raise ValueError(
            "Plain DFlow-Domino checkpoint is missing required head tensors: "
            f"{sorted(missing)}."
        )
    return base_weights, head_weights


class DFlowDominoDraftModel(DFlowDraftModel):
    """Plain one-step DFlow backbone plus a replicated causal Domino head."""

    is_dflow_domino_model = True
    domino_source_arm = "base"

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        validate_plain_dflow_domino_config(config)
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.pure_draft_prefix_len = 1
        self.prefix_hidden_size = DOMINO_PREFIX_HIDDEN_SIZE
        self.residual_rank = DOMINO_RESIDUAL_RANK

        hidden_size = int(config.hidden_size)
        vocab_size = int(config.vocab_size)
        self.prefix_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=self.prefix_hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.residual_down = nn.Linear(
            hidden_size + self.prefix_hidden_size,
            self.residual_rank,
            bias=False,
        )
        # Full vocabulary by design: this projection is replicated on every TP
        # rank, unlike the verifier's vocab-parallel LM head.
        self.residual_up = nn.Linear(
            self.residual_rank,
            vocab_size,
            bias=False,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        base_weights, head_weights = partition_plain_dflow_domino_weights(
            weights,
            hidden_size=int(self.config.hidden_size),
            vocab_size=int(self.config.vocab_size),
        )
        loaded = super().load_weights(base_weights)
        params = dict(self.named_parameters())
        for name in sorted(DOMINO_HEAD_KEYS):
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, head_weights[name])
            loaded.add(name)
        return loaded


EntryClass = DFlowDominoDraftModel


__all__ = [
    "DOMINO_ARCHITECTURE",
    "DOMINO_HEAD_KEYS",
    "DOMINO_PREFIX_HIDDEN_SIZE",
    "DOMINO_REMOTE_AUTO_MODEL",
    "DOMINO_RESIDUAL_RANK",
    "DFlowDominoDraftModel",
    "EntryClass",
    "expected_dflow_domino_head_shapes",
    "partition_plain_dflow_domino_weights",
    "validate_plain_dflow_domino_config",
]
