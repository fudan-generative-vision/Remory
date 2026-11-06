"""Pin Qwen's template effort per tokenizer instance, not a Codex API knob.

No global monkeypatch, tokenizer-file edits, sampling changes or token caps.
The ordinary evaluators pass None and retain their original tokenizer object.
"""
import hashlib


EFFORTS = ("low", "medium", "xhigh")
POLICY = "qwen_native_template_explicit_reasoning_effort_v1"


class EffortTokenizer:
    def __init__(self, tokenizer, effort):
        if effort not in EFFORTS:
            raise ValueError("Qwen reasoning effort must be low, medium, or xhigh")
        self._tokenizer, self.effort = tokenizer, effort

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] != self.effort:
            raise ValueError("Cannot override this cell's pinned Qwen reasoning effort")
        return self._tokenizer.apply_chat_template(*args, **{**kwargs, "reasoning_effort": self.effort})


def bind_reasoning_effort(tokenizer, effort):
    if effort is None:
        return tokenizer, None
    if effort not in EFFORTS:
        raise ValueError("Qwen reasoning effort must be low, medium, or xhigh")
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or "reasoning_effort" not in template:
        raise RuntimeError("Loaded Qwen template has no native reasoning_effort support")
    messages = [{"role": "system", "content": "Reasoning-effort template probe."},
                {"role": "user", "content": "Compute 2+2."}]
    tools = [{"type": "function", "function": {"name": "probe", "description": "Probe only",
              "parameters": {"type": "object", "properties": {}}}}]
    # Test the actual tool-aware branch used by the harness, not metadata labels.
    options = dict(tokenize=False, enable_thinking=True, preserve_thinking=True,
                   add_generation_prompt=True, tools=tools)
    rendered = {level: tokenizer.apply_chat_template(messages, reasoning_effort=level, **options)
                for level in EFFORTS}
    marker = "Reasoning effort is set to "
    if (len(set(rendered.values())) != 3
        or marker + "low." not in rendered["low"]
        or marker + "xhigh." not in rendered["xhigh"]
        or marker in rendered["medium"]
        or any(not text.endswith("<think>\n") for text in rendered.values())):
        raise RuntimeError("Loaded Qwen template did not produce the three distinct native thinking policies")
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    receipt = {"policy": POLICY, "reasoning_effort": effort,
        "chat_template_sha256": digest(template), "enable_thinking": True, "preserve_thinking": True,
        "probe_sha256_by_effort": {level: digest(text) for level, text in rendered.items()},
        "default_matches_xhigh": tokenizer.apply_chat_template(messages, **options) == rendered["xhigh"],
        "fixed_thinking_token_budget": None,
        "summary_decoding_policy": "unchanged_nonthinking_suffix"}
    return EffortTokenizer(tokenizer, effort), receipt
