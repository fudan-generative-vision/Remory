"""Independent long-context compression research module for SpecForge.

The public modeling names remain available from this package, but are imported
lazily.  This keeps CPU-only artifact tooling (for example the Streamlit replay
demo) independent of Torch while preserving the existing training API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_compression.layout import (
        COMPRESSED_QUERY_COUNT,
        SOURCE_KV_WIDTH,
    )
    from context_compression.modeling import (
        DEFAULT_TARGET_LAYER_IDS,
        DFlashKVCompressor,
        FrozenTargetRuntime,
    )

__all__ = [
    "COMPRESSED_QUERY_COUNT",
    "DEFAULT_TARGET_LAYER_IDS",
    "DFlashKVCompressor",
    "FrozenTargetRuntime",
    "SOURCE_KV_WIDTH",
]


def __getattr__(name: str) -> Any:
    if name in {"COMPRESSED_QUERY_COUNT", "SOURCE_KV_WIDTH"}:
        from context_compression import layout

        value = getattr(layout, name)
    elif name in {
        "DEFAULT_TARGET_LAYER_IDS",
        "DFlashKVCompressor",
        "FrozenTargetRuntime",
    }:
        from context_compression import modeling

        value = getattr(modeling, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
