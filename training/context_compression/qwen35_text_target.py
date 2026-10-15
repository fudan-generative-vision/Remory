"""Strict local loader dispatch for Qwen3.5-family text-only targets.

Qwen3.5-family releases store the language backbone below a multimodal
checkpoint, while the prepared snapshots expose a native causal-LM config.
Dense and MoE snapshots use different Transformers classes and modeling
registries, so callers must dispatch from the derived ``model_type`` instead
of assuming that every Qwen3.5-family text model is dense.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias

from transformers import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe


Qwen35TextTarget: TypeAlias = Qwen3_5ForCausalLM | Qwen3_5MoeForCausalLM

TEXT_TARGET_CLASS_BY_MODEL_TYPE = {
    "qwen3_5_text": Qwen3_5ForCausalLM,
    "qwen3_5_moe_text": Qwen3_5MoeForCausalLM,
}
TEXT_TARGET_ARCHITECTURE_BY_MODEL_TYPE = {
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
}
TEXT_TARGET_MODELING_BY_MODEL_TYPE = {
    "qwen3_5_text": modeling_qwen3_5,
    "qwen3_5_moe_text": modeling_qwen3_5_moe,
}


def qwen35_text_config(target_model: Path) -> dict[str, Any]:
    """Read and validate a prepared local Qwen3.5-family text config."""

    config_path = target_model.expanduser().resolve() / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid target text config: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError("target text config must be a JSON object")
    model_type = config.get("model_type")
    expected_architecture = TEXT_TARGET_ARCHITECTURE_BY_MODEL_TYPE.get(model_type)
    if expected_architecture is None:
        raise ValueError(
            "unsupported Qwen3.5-family text model_type: " f"{model_type!r}"
        )
    architectures = config.get("architectures")
    if architectures != [expected_architecture]:
        raise ValueError(
            "target text architecture does not match model_type: "
            f"model_type={model_type!r}, architectures={architectures!r}, "
            f"expected={[expected_architecture]!r}"
        )
    return config


def qwen35_text_model_type(target_model: Path) -> str:
    """Return the validated dense/MoE text model type for a local snapshot."""

    return str(qwen35_text_config(target_model)["model_type"])


def qwen35_text_target_class(target_model: Path) -> type[Qwen35TextTarget]:
    """Resolve the exact Transformers causal-LM class for a local snapshot."""

    return TEXT_TARGET_CLASS_BY_MODEL_TYPE[qwen35_text_model_type(target_model)]


def qwen35_modeling_module_for_model_type(model_type: str) -> ModuleType:
    """Resolve the matching Transformers modeling registry, failing closed."""

    try:
        return TEXT_TARGET_MODELING_BY_MODEL_TYPE[model_type]
    except KeyError as error:
        raise ValueError(
            f"unsupported Qwen3.5-family text model_type: {model_type!r}"
        ) from error


def qwen35_modeling_module(model: Qwen35TextTarget) -> ModuleType:
    """Resolve the modeling registry for an already-loaded target."""

    return qwen35_modeling_module_for_model_type(str(model.config.model_type))


def load_qwen35_text_target(
    target_model: Path,
    **from_pretrained_kwargs: Any,
) -> Qwen35TextTarget:
    """Load a local dense or MoE Qwen3.5-family text-only target.

    The snapshot config is validated before Transformers allocates any model
    tensors.  Remote fallback is deliberately disabled because training binds
    its run contract to the local snapshot contents.
    """

    model_class = qwen35_text_target_class(target_model)
    if from_pretrained_kwargs.get("local_files_only") is False:
        raise ValueError("Qwen3.5-family target loading cannot enable remote fallback")
    from_pretrained_kwargs["local_files_only"] = True
    return model_class.from_pretrained(target_model, **from_pretrained_kwargs)


__all__ = [
    "Qwen35TextTarget",
    "TEXT_TARGET_ARCHITECTURE_BY_MODEL_TYPE",
    "TEXT_TARGET_CLASS_BY_MODEL_TYPE",
    "load_qwen35_text_target",
    "qwen35_modeling_module",
    "qwen35_modeling_module_for_model_type",
    "qwen35_text_config",
    "qwen35_text_model_type",
    "qwen35_text_target_class",
]
