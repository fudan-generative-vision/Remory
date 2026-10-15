"""Public compact prompt verified against the reproduction's Codex CLI binary.

This is a new training extension, not part of the historical source snapshot.
It pins the CLI's local compaction instruction, not OpenAI's remote compactor.
"""

import hashlib
import copy
from pathlib import Path

CODEX_VERSION = "0.149.0"
CODEX_SHA256 = "bbc3341e44c9ead340ed9570c17be936e37870f570751a941699ffd04d672827"
PROMPT_SHA256 = "ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07"
HANDOFF_FIELDS = (
    "current_progress", "key_decisions", "important_context_constraints_preferences",
    "next_steps", "critical_data_examples_references",
)
HANDOFF_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "array", "items": {"type": "string"}, "maxItems": 12,
                          "minItems": 1 if name in {
                              "current_progress", "important_context_constraints_preferences"
                          } else 0}
                   for name in HANDOFF_FIELDS},
    "required": list(HANDOFF_FIELDS), "additionalProperties": False,
}
PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""


def verify_binary(path):
    """Read an explicitly supplied binary; never execute it or log its contents."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != CODEX_SHA256:
        raise ValueError("Codex binary differs from pinned CLI 0.149.0")
    # The stripped release binary embeds the same literal at several call sites.
    if PROMPT.encode() not in data:
        raise ValueError("pinned compact prompt is absent from binary")


def summary_contract(tokenizer, **kwargs):
    from context_compression.qwen38_residual_data import summary_contract as legacy

    if hashlib.sha256(PROMPT.encode()).hexdigest() != PROMPT_SHA256:
        raise ValueError("Codex compact prompt bytes changed")
    maximum = kwargs.get("max_summary_tokens", 4096)
    # Preserve the export ABI's positive target field. It is a budget ceiling,
    # not an additional textual request or a 512-token target in this variant.
    contract = legacy(tokenizer, **{**kwargs, "max_summary_tokens": maximum,
                                   "target_summary_tokens": maximum})
    suffix = ("\n<|im_start|>user\n" + PROMPT +
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    contract.update(
        prompt=PROMPT,
        generation_suffix_ids=tokenizer.encode(suffix, add_special_tokens=False),
        prompt_source={"kind": "codex_cli_builtin_local_compact", "version": CODEX_VERSION,
                       "binary_sha256": CODEX_SHA256, "prompt_sha256": PROMPT_SHA256},
        summary_length_policy="generation_cap_only_no_prompt_target",
    )
    return contract


def structured_summary_contract(tokenizer, **kwargs):
    """Same prompt and causal request, with an explicit handoff output grammar.

    Local Qwen can otherwise continue an old task instead of making a handoff.
    This is an output-format adaptation, not the native CLI decoding algorithm.
    The distinct contract prevents mixing unconstrained text into this cache.
    """
    contract = summary_contract(tokenizer, **kwargs)
    return {**contract, "output_format": "json_handoff_v1",
            "output_schema": copy.deepcopy(HANDOFF_SCHEMA)}
