"""Strict, shared parsing for model-reported BrowseComp confidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


DEFAULT_CONFIDENCE_THRESHOLD = Decimal("90")

_CONFIDENCE_FIELD_RE = re.compile(
    r"""
    ^\s*(?:[-+*]\s+)?
    (?:(?:\*\*|__)\s*)?
    confidence\s*
    (?:(?:\*\*|__)\s*)?
    :\s*
    (?:(?:\*\*|__)\s*)?
    (?P<body>.*)$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PERCENT_RE = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%(?!\w)"
)


@dataclass(frozen=True, slots=True)
class ConfidenceParse:
    """One explicit confidence-field parse with an auditable fallback."""

    state: str
    value: Decimal | None
    field_count: int

    @property
    def effective_value(self) -> Decimal:
        # Missing, invalid, and ambiguous values are deliberately treated as
        # zero by both the in-episode gate and the external requeue policy.
        return self.value if self.value is not None else Decimal("0")


def parse_reported_confidence(prediction: Any) -> ConfidenceParse:
    """Parse exactly one explicit ``Confidence: N%`` final-answer field.

    Markdown-wrapped labels, Unicode full-width punctuation after NFKC
    normalization, and trailing explanatory prose are accepted.  Multiple
    fields or percentages are rejected rather than guessed.
    """

    text = unicodedata.normalize(
        "NFKC", prediction if isinstance(prediction, str) else ""
    )
    field_bodies: list[str] = []
    for line in text.splitlines():
        match = _CONFIDENCE_FIELD_RE.match(line)
        if match is not None:
            field_bodies.append(match.group("body"))
    if not field_bodies:
        return ConfidenceParse("missing", None, 0)
    if len(field_bodies) != 1:
        return ConfidenceParse("ambiguous", None, len(field_bodies))

    matches = list(_PERCENT_RE.finditer(field_bodies[0]))
    if len(matches) > 1:
        return ConfidenceParse("ambiguous", None, 1)
    if not matches:
        return ConfidenceParse("invalid", None, 1)
    try:
        value = Decimal(matches[0].group("number"))
    except InvalidOperation:
        return ConfidenceParse("invalid", None, 1)
    if not value.is_finite() or value < 0 or value > 100:
        return ConfidenceParse("invalid", None, 1)
    return ConfidenceParse("parsed", value, 1)
