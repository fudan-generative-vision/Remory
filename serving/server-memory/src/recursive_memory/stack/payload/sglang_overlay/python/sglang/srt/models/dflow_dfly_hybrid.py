"""Native SGLang runtime for the zero-gated DFlow/DFly hybrid.

The Stage-I hybrid keeps the ordinary :class:`DFlowDraftModel` backbone and
changes only the target context presented to each draft layer.  For draft
layer ``i`` the context is

``RMSNorm(fc(concat(target_views)) + gate[i] * weighted_view[i])``.

Target features are already captured by the verifier for DFlow.  This module
therefore uses the existing multi-KV-group worker ABI to materialize one
layer-specific view into each layer's KV cache.  It does not add another
drafter forward and deliberately does not implement DFly hidden-state
correction (HSC).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from numbers import Integral
from typing import Tuple

import torch
from torch import nn

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflow import DFlowDraftModel
from sglang.srt.models.dflash import DFlashDecoderLayer
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


_FUSION_WEIGHT_KEYS = frozenset({"layer_fusion_weights", "layer_fusion_gates"})
_HYBRID_ARCHITECTURE = "DFlowDFlyHybridDraftModel"


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


def _normalize_checkpoint_weight_name(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _partition_hybrid_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    num_draft_layers: int,
    num_target_features: int,
) -> tuple[list[Tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Separate and validate the two replicated hybrid tensors.

    Validation happens before any model parameter is modified.  In
    particular, a base checkpoint cannot be accidentally served by the hybrid
    architecture with randomly initialized fusion parameters.
    """

    expected_shapes = {
        "layer_fusion_weights": (
            int(num_draft_layers),
            int(num_target_features),
        ),
        "layer_fusion_gates": (int(num_draft_layers),),
    }
    base_weights: list[Tuple[str, torch.Tensor]] = []
    fusion_weights: dict[str, torch.Tensor] = {}
    for raw_name, value in weights:
        name = _normalize_checkpoint_weight_name(raw_name)
        if name not in _FUSION_WEIGHT_KEYS:
            if name.startswith("layer_fusion_"):
                raise ValueError(
                    "DFlow/DFly checkpoint contains unsupported fusion tensor "
                    f"{name!r}."
                )
            if name.startswith("hidden_correction."):
                raise ValueError(
                    "DFlow/DFly layer-fusion runtime does not implement HSC; "
                    f"unsupported tensor {name!r}."
                )
            base_weights.append((raw_name, value))
            continue
        if name in fusion_weights:
            raise ValueError(
                "DFlow/DFly checkpoint contains duplicate tensor " f"{name!r}."
            )
        expected = expected_shapes[name]
        actual = tuple(int(dim) for dim in value.shape)
        if actual != expected:
            raise ValueError(
                f"DFlow/DFly {name} shape mismatch: expected {expected}, "
                f"got {actual}."
            )
        fusion_weights[name] = value

    missing = _FUSION_WEIGHT_KEYS - fusion_weights.keys()
    if missing:
        raise ValueError(
            "DFlow/DFly checkpoint is missing required runtime tensors: "
            f"{sorted(missing)}"
        )
    return base_weights, fusion_weights


def _validate_hybrid_config(
    config,
    *,
    num_draft_layers: int,
    target_layer_ids: tuple[int, ...],
) -> None:
    """Fail closed on a checkpoint that only resembles the Stage-I ABI."""

    architectures = getattr(config, "architectures", None)
    if architectures != [_HYBRID_ARCHITECTURE]:
        raise ValueError(
            "DFlow/DFly native runtime requires architectures="
            f"[{_HYBRID_ARCHITECTURE!r}], got {architectures!r}."
        )

    raw_dflow_config = getattr(config, "dflow_config", None)
    if not isinstance(raw_dflow_config, Mapping):
        raise ValueError("DFlow/DFly checkpoint requires a mapping dflow_config.")
    dflow_config = dict(raw_dflow_config)
    expected = {
        "hybrid_target_conditioning": "dfly_residual",
        "layer_fusion_warm_start": "zero_gate_exact_base",
        "layer_fusion_num_target_layers": len(target_layer_ids),
        "layer_fusion_target_layer_ids": list(target_layer_ids),
        "layer_fusion_preserves_target_feature_abi": True,
        "serving_drafter_forwards": 1,
    }
    mismatches = {
        key: (dflow_config.get(key), value)
        for key, value in expected.items()
        if not _config_value_equal(dflow_config.get(key), value)
    }
    if not _config_value_equal(
        getattr(config, "num_hidden_layers", None), int(num_draft_layers)
    ):
        mismatches["num_hidden_layers"] = (
            getattr(config, "num_hidden_layers", None),
            int(num_draft_layers),
        )

    # Stage-I checkpoints carry additional arm markers.  If any Stage-I
    # marker is present, require the complete relevant contract rather than
    # treating a partially rewritten base-arm config as a hybrid checkpoint.
    claims_stage_i = any(key.startswith("stage_i_") for key in dflow_config)
    if claims_stage_i:
        stage_i_expected = {
            "stage_i_matched_pair": True,
            "stage_i_arm": "zero_gated_layer_fusion",
            "stage_i_only_semantic_delta": "drafter_architecture",
            "stage_i_zero_gate_exact_base": True,
            "training_stage": "one_forward_stage_i_matched_architecture_pair",
        }
        mismatches.update(
            {
                key: (dflow_config.get(key), value)
                for key, value in stage_i_expected.items()
                if not _config_value_equal(dflow_config.get(key), value)
            }
        )

    # A full DFly/HSC checkpoint needs a different serving ABI (previous-token
    # embeddings after decode).  Never silently discard those weights here.
    for unsupported_marker in (
        "full_dfly_adaptation",
        "enable_hidden_correction",
    ):
        if dflow_config.get(unsupported_marker, False) is not False:
            mismatches[unsupported_marker] = (
                dflow_config.get(unsupported_marker),
                False,
            )

    if mismatches:
        raise ValueError(
            "DFlow/DFly hybrid config contract mismatch (actual, expected): "
            f"{mismatches}."
        )


class DFlowDFlyHybridDraftModel(DFlowDraftModel):
    """DFlow with a learned target-layer mixture for every draft layer."""

    # The existing worker's generic multi-group path accepts arbitrary layer
    # subsets.  Each layer owns a different projected context, so the shared
    # fused KV materializer is not applicable.
    has_multiple_dflash_kv_groups = True
    supports_fused_kv_materialization = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # DFlashDraftModel sets this instance field from joint-judge state;
        # restore the hybrid's layer-specific KV contract after base init.
        self.has_multiple_dflash_kv_groups = True
        self.supports_fused_kv_materialization = False
        if getattr(self, "joint_judge_enabled", False):
            raise ValueError(
                "DFlow/DFly layer fusion does not support the joint-judge " "tower."
            )

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else int(config.num_hidden_layers)
        )
        target_layer_ids = tuple(
            int(layer_id)
            for layer_id in draft_config.resolve_target_layer_ids(
                target_num_layers=target_num_layers,
                draft_num_layers=int(config.num_hidden_layers),
            )
        )
        if len(target_layer_ids) != int(self.num_context_features):
            raise ValueError(
                "DFlow/DFly resolved target-feature count mismatch: "
                f"ids={target_layer_ids}, num_context_features="
                f"{self.num_context_features}."
            )
        _validate_hybrid_config(
            config,
            num_draft_layers=len(self.layers),
            target_layer_ids=target_layer_ids,
        )
        self.target_layer_ids = target_layer_ids

        # These 54 scalars for Qwen3-4B (6*8 mixture logits + 6 gates) are
        # replicated on every TP rank.  The loader below requires full shapes
        # on every rank; they are intentionally not tensor-parallel shards.
        self.layer_fusion_weights = nn.Parameter(
            torch.zeros(len(self.layers), len(target_layer_ids))
        )
        self.layer_fusion_gates = nn.Parameter(torch.zeros(len(self.layers)))
        # Construction may happen on the meta device in non-default loaders.
        # The initialized values are known without reading tensor storage; the
        # checkpoint loader refreshes this cache from the actual gate payload.
        self._layer_fusion_zero_gates = (True,) * len(self.layers)

    def _refresh_zero_gate_cache(self, gates: torch.Tensor | None = None) -> None:
        gates = self.layer_fusion_gates.detach() if gates is None else gates.detach()
        if gates.is_meta:
            raise RuntimeError(
                "Cannot resolve DFlow/DFly zero gates from a meta tensor."
            )
        self._layer_fusion_zero_gates = tuple(
            bool(value == 0) for value in gates.float().cpu().tolist()
        )

    def _validate_target_hidden(self, target_hidden: torch.Tensor) -> None:
        expected = int(self.num_context_features) * int(self.config.hidden_size)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFlow/DFly target_hidden feature dim mismatch. Expected "
                f"[N, {expected}] = [N, {self.num_context_features} * "
                f"{int(self.config.hidden_size)}], got "
                f"{tuple(target_hidden.shape)}."
            )

    def project_target_hidden_for_layer(
        self,
        target_hidden: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Build the exact training-side target context for one draft layer."""

        self._validate_target_hidden(target_hidden)
        if not 0 <= int(layer_idx) < len(self.layers):
            raise IndexError(f"DFlow/DFly draft layer index out of range: {layer_idx}.")

        base_context = self.fc(target_hidden)
        target_views = target_hidden.reshape(
            int(target_hidden.shape[0]),
            int(self.num_context_features),
            int(self.config.hidden_size),
        )
        return self._build_layer_context(
            target_views=target_views,
            base_context=base_context,
            layer_idx=int(layer_idx),
        )

    def _build_layer_context(
        self,
        *,
        target_views: torch.Tensor,
        base_context: torch.Tensor,
        layer_idx: int,
        zero_gate_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse one view while preserving an exact base-only fast path."""

        # This explicit branch is the functional warm-start guarantee: when a
        # gate is exactly zero, execute the same operation sequence as native
        # DFlow and avoid even a multiply/add by zero.  Hence the materialized
        # K/V, backbone hidden states, LM-head logits and greedy IDs can remain
        # bitwise identical to the base arm.
        if self._layer_fusion_zero_gates[layer_idx]:
            if zero_gate_context is not None:
                return zero_gate_context
            return self.hidden_norm(base_context)

        fusion_probs = torch.softmax(
            self.layer_fusion_weights[layer_idx].float(), dim=-1
        ).to(dtype=base_context.dtype)
        residual_context = torch.einsum(
            "t,nth->nh",
            fusion_probs,
            target_views.to(dtype=base_context.dtype),
        )
        gate = self.layer_fusion_gates[layer_idx].to(dtype=base_context.dtype)
        return self.hidden_norm(base_context + gate * residual_context)

    def iter_dflash_kv_groups(
        self, target_hidden: torch.Tensor
    ) -> Iterator[Tuple[Iterable[DFlashDecoderLayer], torch.Tensor]]:
        """Yield one target-context KV materialization group per draft layer."""

        self._validate_target_hidden(target_hidden)
        base_context = self.fc(target_hidden)
        target_views = target_hidden.reshape(
            int(target_hidden.shape[0]),
            int(self.num_context_features),
            int(self.config.hidden_size),
        )
        # Native base DFlow evaluates fc+RMSNorm once, then reuses the result
        # for all layers.  Share that exact tensor across zero-gated layers so
        # the all-zero checkpoint takes the same projection path and cost.
        zero_gate_context = (
            self.hidden_norm(base_context)
            if any(self._layer_fusion_zero_gates)
            else None
        )
        for layer_idx, layer in enumerate(self.layers):
            yield (
                (layer,),
                self._build_layer_context(
                    target_views=target_views,
                    base_context=base_context,
                    layer_idx=layer_idx,
                    zero_gate_context=zero_gate_context,
                ),
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load base DFlow tensors plus the two required replicated tensors."""

        base_weights, fusion_weights = _partition_hybrid_weights(
            weights,
            num_draft_layers=len(self.layers),
            num_target_features=int(self.num_context_features),
        )
        loaded = super().load_weights(base_weights)
        params = dict(self.named_parameters())
        for name in sorted(_FUSION_WEIGHT_KEYS):
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, fusion_weights[name])
            loaded.add(name)
        self._refresh_zero_gate_cache(fusion_weights["layer_fusion_gates"])
        return loaded


EntryClass = DFlowDFlyHybridDraftModel


__all__ = [
    "DFlowDFlyHybridDraftModel",
    "EntryClass",
]
