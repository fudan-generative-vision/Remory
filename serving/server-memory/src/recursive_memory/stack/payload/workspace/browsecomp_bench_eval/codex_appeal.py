"""Pure web-appeal prompt and parser; no legacy pool/reviewer imports."""
from __future__ import annotations
import json
import urllib.parse
from dataclasses import dataclass
from typing import Mapping
from .codex_appeal_contract import (CODEX_APPEAL_DECISIONS, CODEX_APPEAL_INSTRUCTIONS, MAX_CODEX_APPEAL_REASON_CHARS)
MAX_PROMPT_FIELD_CHARS = 128 * 1024
MAX_CODEX_OUTPUT_BYTES = 256 * 1024

class CodexAppealError(RuntimeError):
    """An appeal artifact, contract, or invocation failed."""

class CodexInvocationError(CodexAppealError):
    """The isolated Codex subprocess did not return a valid decision."""

@dataclass(frozen=True, slots=True)
class StructuredAppeal:
    decision: str
    prediction_satisfies_question: bool | None
    ground_truth_issue: str
    reason: str
    primary_sources: tuple[Mapping[str, str], ...]

def build_codex_appeal_prompt(
    *, question: str, prediction: str, reference_answer: str, qwen_reason: str
) -> str:
    evidence = {
        "question": question,
        "candidate_prediction": prediction,
        "reference_answer": reference_answer,
        "qwen_incorrect_reason": qwen_reason,
    }
    for field, value in evidence.items():
        if not isinstance(value, str) or len(value) > MAX_PROMPT_FIELD_CHARS:
            raise CodexAppealError(f"appeal prompt field {field} is invalid")
    return (
        CODEX_APPEAL_INSTRUCTIONS
        + "\n\nUNTRUSTED EVIDENCE JSON:\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )

def parse_codex_appeal_output(payload: str | bytes) -> StructuredAppeal:
    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
    else:
        encoded = bytes(payload)
    if len(encoded) > MAX_CODEX_OUTPUT_BYTES:
        raise CodexInvocationError("Codex appeal output is too large")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodexInvocationError("Codex appeal output is not valid JSON") from exc
    required = {
        "decision",
        "prediction_satisfies_question",
        "ground_truth_issue",
        "reason",
        "primary_sources",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CodexInvocationError("Codex appeal output shape is invalid")
    decision = value.get("decision")
    satisfies = value.get("prediction_satisfies_question")
    ground_truth_issue = value.get("ground_truth_issue")
    reason = value.get("reason")
    sources = value.get("primary_sources")
    if decision not in CODEX_APPEAL_DECISIONS:
        raise CodexInvocationError("Codex appeal decision is invalid")
    if satisfies is not None and not isinstance(satisfies, bool):
        raise CodexInvocationError("Codex appeal satisfaction field is invalid")
    if decision == "request_overturn" and satisfies is not True:
        raise CodexInvocationError("overturn requires a positive condition finding")
    if decision == "uphold" and satisfies is not False:
        raise CodexInvocationError("uphold requires a negative condition finding")
    if ground_truth_issue not in {
        "none",
        "typo",
        "alias",
        "transliteration",
        "normalization",
        "incomplete",
        "other",
        "uncertain",
    }:
        raise CodexInvocationError("Codex appeal ground-truth finding is invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise CodexInvocationError("Codex appeal reason is invalid")
    reason = reason.strip()
    if len(reason) > MAX_CODEX_APPEAL_REASON_CHARS:
        raise CodexInvocationError("Codex appeal reason is too large")
    if not isinstance(sources, list) or len(sources) > 8:
        raise CodexInvocationError("Codex appeal sources are invalid")
    normalized_sources: list[Mapping[str, str]] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"url", "title", "support"}:
            raise CodexInvocationError("Codex appeal source shape is invalid")
        url = source.get("url")
        title = source.get("title")
        support = source.get("support")
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or len(url) > 2048
            or not isinstance(title, str)
            or not title.strip()
            or len(title) > 512
            or not isinstance(support, str)
            or not support.strip()
            or len(support) > 2048
        ):
            raise CodexInvocationError("Codex appeal source is invalid")
        normalized_sources.append(
            {"url": url, "title": title.strip(), "support": support.strip()}
        )
    if decision != "inconclusive" and not normalized_sources:
        raise CodexInvocationError("a conclusive appeal requires a primary source")
    return StructuredAppeal(
        decision=str(decision),
        prediction_satisfies_question=satisfies,
        ground_truth_issue=str(ground_truth_issue),
        reason=reason,
        primary_sources=tuple(normalized_sources),
    )
