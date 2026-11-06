"""BrowseComp-only generation policy; no benchmark answers enter this module."""
from decimal import Decimal

from .confidence import parse_reported_confidence

POLICY_ID = "browsecomp_confidence_gt95_system_v1"
REQUEUE_THRESHOLD = Decimal("95")
SYSTEM_INSTRUCTIONS = """BrowseComp research and final-answer policy:
Only give a final answer when your evidence-based confidence is strictly greater
than 95%. If confidence is 95% or lower, keep researching and checking the
question's conditions instead of ending the task. Do not inflate or fabricate
confidence to satisfy this requirement: improve the evidence first. Confidence
is your calibrated estimate, not a guarantee of correctness.
When ready, finish with Explanation: <brief evidence and sources>,
Exact Answer: <answer>, and exactly one Confidence: <0-100%> field.
"""


def confidence_receipt(answer):
    parsed = parse_reported_confidence(answer)
    return {"confidence_percent": float(parsed.effective_value),
            "confidence_parse_state": parsed.state}


def needs_confidence_requeue(answer):
    # Existing results exactly at 95 are deliberately retained. The new prompt
    # asks for >95; the user's historical requeue instruction is strictly <95.
    return parse_reported_confidence(answer).effective_value < REQUEUE_THRESHOLD
