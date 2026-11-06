"""Benchmark-independent configuration for residual server admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    harness_mode: str = "native_compact"
    working_memory_turns: int = 12
    serving_eviction_policy: str = "checkpoint_1024_v1"
    native_compaction_text_summary: bool = True
    compressor_enabled: bool = True
    residual_compaction: bool = True
    residual_checkpoint: str = ""

    def validate(self):
        if self.harness_mode != "native_compact":
            raise ValueError("This release supports only native_compact")
        for name in ("native_compaction_text_summary", "compressor_enabled", "residual_compaction"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.compressor_enabled != self.residual_compaction or not self.native_compaction_text_summary:
            raise ValueError("native compact requires text summary and paired Compressor/residual switches")
        # Both arms pin the export: text-only reads the summary metadata but
        # never instantiates or calls the encoder.
        if not isinstance(self.residual_checkpoint, str) or not Path(self.residual_checkpoint).is_absolute():
            raise ValueError("both memory arms require an absolute summary export path")
        if type(self.working_memory_turns) is not int or self.working_memory_turns < 1:
            raise ValueError("working_memory_turns must be a positive integer")
        if self.serving_eviction_policy != "checkpoint_1024_v1":
            raise ValueError("native compaction requires checkpoint_1024_v1")
        return self
