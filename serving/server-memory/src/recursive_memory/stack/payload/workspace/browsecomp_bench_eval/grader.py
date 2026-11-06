"""Post-generation BrowseComp judge with a fixed upstream bug corrected."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


# OpenAI simple-evals BrowseComp grader prompt. Gold is passed only to this
# post-hoc component after the browsing trajectory has terminated. Preserve
# the upstream runtime string byte-for-byte, including its typo, blank lines,
# and escaped-percent notation. Pinned at simple-evals commit 652c89d0:
# https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/browsecomp_eval.py
GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

OPENAI_SIMPLE_EVALS_GRADER_TEMPLATE_SHA256 = (
    "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
)
if hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest() != (
    OPENAI_SIMPLE_EVALS_GRADER_TEMPLATE_SHA256
):
    raise RuntimeError("OpenAI simple-evals BrowseComp grader prompt drifted")

_GRADE_PATTERN = re.compile(r"(?im)^\s*correct\s*:\s*(yes|no)\b")


def parse_grade(text: str) -> str:
    """Return ``yes``/``no``; fail closed when the judge format is invalid.

    Upstream simple-evals currently returns ``match.group(0)`` and then
    compares it with ``"yes"``.  That reports zero accuracy even for a judge
    response containing ``correct: yes``.  The captured group is authoritative.
    """

    match = _GRADE_PATTERN.search(str(text))
    return match.group(1).lower() if match else "no"


@dataclass(frozen=True)
class GradeResult:
    verdict: str
    correct: bool
    response_text: str
    elapsed_seconds: float
    model: str

    def to_private_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "correct": self.correct,
            "response_text": self.response_text,
            "elapsed_seconds": self.elapsed_seconds,
            "model": self.model,
        }


class OpenAICompatibleGrader:
    """Minimal chat-completions judge; no OpenAI SDK dependency is required."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        disable_thinking: bool = False,
        strict: bool = False,
        bypass_proxy: bool = False,
    ) -> None:
        if not str(base_url).strip() or not str(model).strip():
            raise ValueError("grader base_url and model must be nonempty")
        if timeout_seconds <= 0:
            raise ValueError("grader timeout must be positive")
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.disable_thinking = disable_thinking
        self.strict = strict
        self.bypass_proxy = bypass_proxy

    def grade(
        self, *, question: str, correct_answer: str, response: str
    ) -> GradeResult:
        prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
                **({"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
                   if self.disable_thinking else {}),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            opener = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                      if self.bypass_proxy else urllib.request.urlopen)
            with opener(
                request, timeout=self.timeout_seconds
            ) as response_handle:
                body = response_handle.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"BrowseComp grader request failed: {exc}") from exc
        elapsed = time.monotonic() - started
        try:
            decoded: Mapping[str, Any] = json.loads(body)
            choices = decoded["choices"]
            text = str(choices[0]["message"]["content"])
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "BrowseComp grader returned an invalid response"
            ) from exc
        verdict = parse_grade(text)
        if self.strict and (len(_GRADE_PATTERN.findall(text)) != 1 or
                            choices[0].get("finish_reason") == "length"):
            raise RuntimeError("BrowseComp judge output is truncated or has no unique verdict")
        return GradeResult(
            verdict=verdict,
            correct=verdict == "yes",
            response_text=text,
            elapsed_seconds=elapsed,
            model=self.model,
        )
