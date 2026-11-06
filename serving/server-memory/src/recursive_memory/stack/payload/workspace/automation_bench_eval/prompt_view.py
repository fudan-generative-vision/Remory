"""Private initial model-input receipts and offline legacy prompt rendering.

Legacy public-task prompts omit Codex's own instructions and tool schemas. They
can be rendered for inspection, but must never be presented as exact run inputs.
This module's CLI loads only a local tokenizer; it never contacts model serving.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


FILENAME = "system-user-chat-template.json"
CAPTURE_POLICY = "first_decode_system_user_input_v1"
PROJECT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initial_prompt_observer(execution, *, model_path, reasoning_effort):
    """Record a copy of the exact initial input IDs, before the serving request."""
    def record(tokenizer, input_ids):
        from shared_runtime.storage import atomic_json, utcnow
        text = tokenizer.decode(input_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        atomic_json(Path(execution) / "gateway" / FILENAME, {
            "schema": CAPTURE_POLICY, "provenance": "exact_initial_generate_input",
            "recorded_at": utcnow(), "model_path": str(model_path),
            "reasoning_effort": reasoning_effort, "includes_generation_prefix": True,
            "chat_template_sha256": digest(tokenizer.get_chat_template()),
            "input_ids": list(input_ids), "input_tokens": len(input_ids),
            "text": text, "text_sha256": digest(text),
        })
    return record


def render_legacy(tokenizer, messages, *, reasoning_effort=None, expected_template_hash=None):
    if (not isinstance(messages, list) or not messages
        or any(not isinstance(m, dict) or m.get("role") not in {"system", "user"}
               or not isinstance(m.get("content"), str) for m in messages)
        or not any(m["role"] == "user" for m in messages)):
        raise ValueError("No saved textual system/user prompt")
    if reasoning_effort not in (None, "low", "medium", "xhigh"):
        raise ValueError("Invalid saved reasoning effort")
    template_hash = digest(tokenizer.get_chat_template())
    if expected_template_hash and template_hash != expected_template_hash:
        raise ValueError("Local chat template differs from this execution's recorded template")
    options = dict(tokenize=False, add_generation_prompt=True,
                   enable_thinking=True, preserve_thinking=True)
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    text = tokenizer.apply_chat_template(messages, **options)
    return {"text": text, "chat_template_sha256": template_hash,
            "provenance": "reconstructed_public_task_only",
            "template_verified": bool(expected_template_hash)}


@lru_cache(maxsize=64)
def _render_local(model_path, fingerprint, messages_json, effort, template_hash):
    # Streamlit may use an older Transformers environment. Use the same local
    # Python environment as evaluation, with no GPU and no Hub downloads.
    python = PROJECT / "eval/terminal_bench/compressor_venv/bin/python"
    if not python.is_file():
        raise ValueError("Evaluation tokenizer Python environment is unavailable")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        result = subprocess.run([str(python), "-m", "automation_bench_eval.prompt_view"],
            cwd=PROJECT, env=env, input=json.dumps({"model_path": model_path,
                "messages": json.loads(messages_json), "reasoning_effort": effort,
                "expected_template_hash": template_hash}),
            text=True, capture_output=True, timeout=45, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Local tokenizer rendering timed out") from exc
    if result.returncode:
        raise ValueError("Local tokenizer renderer could not start")
    output = json.loads(result.stdout)
    if output.get("error"):
        raise ValueError(output["error"])
    return output


def reconstruct_public_prompt(metadata, public, reasoning_policy=None):
    model_path = metadata.get("model_path")
    if not isinstance(model_path, str) or not Path(model_path).is_dir():
        raise ValueError("This execution's local tokenizer is unavailable")
    model = Path(model_path)
    # Refresh the cache if tokenizer/template files change, without loading
    # Transformers or the tokenizer into the monitoring process itself.
    fingerprint = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for name in ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "config.json")
        if (p := model / name).is_file())
    policy = reasoning_policy or {}
    effort = metadata.get("reasoning_effort", policy.get("reasoning_effort"))
    if policy.get("reasoning_effort") not in (None, effort):
        raise ValueError("Saved reasoning-effort records disagree")
    return _render_local(str(model), fingerprint,
        json.dumps(public.get("prompt"), ensure_ascii=False, sort_keys=True),
        effort, policy.get("chat_template_sha256"))


def main():
    try:
        from transformers import AutoTokenizer
        request = json.load(sys.stdin)
        tokenizer = AutoTokenizer.from_pretrained(request.pop("model_path"),
            local_files_only=True, trust_remote_code=False)
        output = render_legacy(tokenizer, **request)
    except Exception as exc:
        output = {"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
