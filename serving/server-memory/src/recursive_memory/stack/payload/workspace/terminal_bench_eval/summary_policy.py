"""Serving-capacity budgets and explicit completion receipts for all summaries."""
from __future__ import annotations

from copy import deepcopy

SUMMARY_BUDGET_POLICY = "remaining_sglang_context_no_fixed_summary_cap_v1"


def summary_output_budget(*, input_tokens, server_info, post_compact_overhead=0):
    from context_compression.grepseek_sglang_live import _generation_token_budget
    try:
        budget = _generation_token_budget(
            # This positive placeholder is ignored in remaining-context mode;
            # it is neither a one-token cap nor the ordinary-round limit.
            prefix_token_count=input_tokens, configured_max_new_tokens=1,
            server_context_length=server_info["context_length"],
            server_max_total_num_tokens=server_info["max_total_num_tokens"],
            server_max_req_input_len=server_info["max_req_input_len"],
            use_remaining_context=True, remaining_context_output_ceiling_tokens=None,
        )
        # The result must also fit when installed with its envelopes, soft
        # memory and retained WM. This is a physical bound, not a summary cap.
        installed_limit = min(server_info[k] for k in
                              ("context_length", "max_total_num_tokens", "max_req_input_len"))
        budget = min(budget, installed_limit - post_compact_overhead - 2)
        if budget < 1:
            raise ValueError("no space remains for a summary and its installed context")
        return budget
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"summary capacity exceeded: {exc}") from exc


def summary_completion(result, *, stop_token_ids=()):
    reason = deepcopy(getattr(result, "finish_reason", None))
    kind = reason.get("type", reason.get("reason")) if isinstance(reason, dict) else reason
    meta = getattr(result, "meta_info", {}) or {}
    guard = meta.get("generation_degeneration_guard", {}) or {}
    # The degeneration guard can return an explicitly marked useful prefix.
    # It is not a naturally finished summary, but must not be regenerated.
    stopped_for_loop = bool(guard.get("detections"))
    natural = bool(result.finished_on_eos)
    matched = reason.get("matched") if isinstance(reason, dict) else None
    # SGLang also stops Qwen on <|endoftext|>, whereas the agent parser's
    # finished_on_eos means <|im_end|> only. A summary quoting e.g. GPT-2's
    # terminator can hit the former mid-sentence. Keep that useful prefix,
    # explicitly truncated, instead of retrying it or failing the whole task.
    alternate_stop = (kind == "stop" and type(matched) is int
                      and matched in stop_token_ids and not natural)
    interrupted = kind in {"abort", "aborted", "cancel", "canceled", "cancelled", "error"}
    length_limited = kind in {"length", "max_tokens", "max_new_tokens"}
    accepted = not interrupted and (natural or alternate_stop or length_limited or stopped_for_loop)
    return dict(summary_budget_policy=SUMMARY_BUDGET_POLICY, fixed_summary_token_cap=None,
                finish_reason=reason, finished_on_eos=natural,
                summary_truncated=not natural, summary_completion_accepted=accepted,
                summary_stop_class=("interrupted" if interrupted else "natural" if natural else
                                    "alternate_stop_token" if alternate_stop else
                                    "degeneration_guard" if stopped_for_loop else "length" if length_limited else "unknown"))


def prepare_summary_output(result, *, tokenizer, end_token_id=None):
    """Classify a summary and strip only its terminal, known control token.

    Do not accept arbitrary stop strings, all special tokens, or a tokenizer's
    unknown-token fallback. Raw generated IDs stay unchanged for the archive.
    Callers still validate nonempty text and publish/install transactionally.
    """
    stop_ids = {value for value in (end_token_id, getattr(tokenizer, "eos_token_id", None))
                if type(value) is int and value >= 0}
    lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
    inverse = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(lookup) and callable(inverse):
        alternate = lookup("<|endoftext|>")
        if (type(alternate) is int and alternate >= 0
                and inverse(alternate) == "<|endoftext|>"):
            stop_ids.add(alternate)
    receipt = summary_completion(result, stop_token_ids=stop_ids)
    body = list(result.token_ids)
    removed = [body.pop()] if body and body[-1] in stop_ids else []
    receipt.update(summary_stop_token_ids=sorted(stop_ids),
                   summary_removed_stop_token_ids=removed)
    return body, receipt


def require_summary_completion(receipt):
    if not receipt["summary_completion_accepted"]:
        raise RuntimeError(f"summary generation did not complete safely: {receipt['summary_stop_class']}; "
                           f"finish_reason={receipt['finish_reason']!r}")
