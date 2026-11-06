"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


from typing import Any, Mapping, Optional, Sequence


from context_compression.grepseek_generation import (
    ANSWER_CLOSE_TOKEN_IDS,
    ANSWER_OPEN_TOKEN_IDS,
    EOS_TOKEN_ID,
)


def _assert_token_contract(tokenizer: Any) -> None:
    if int(tokenizer.eos_token_id) != EOS_TOKEN_ID:
        raise ValueError(
            f"GrepSeek EOS must be <|im_end|>={EOS_TOKEN_ID}, got "
            f"{tokenizer.eos_token_id}"
        )
    contract = {
        "<think>": [248068],
        "</think>": [248069],
        "<tool_call>": [248058],
        "</tool_call>": [248059],
        "<answer>": list(ANSWER_OPEN_TOKEN_IDS),
        "</answer>": list(ANSWER_CLOSE_TOKEN_IDS),
        "<|im_end|>": [EOS_TOKEN_ID],
    }
    for text, expected in contract.items():
        observed = list(tokenizer.encode(text, add_special_tokens=False))
        if observed != expected:
            raise ValueError(
                f"GrepSeek delimiter tokenization changed for {text!r}: "
                f"expected={expected}, observed={observed}"
            )
