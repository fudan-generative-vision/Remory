"""Local text summaries paired with Codex's opaque remote-compaction checkpoints.

Summary decoding is deliberately not a normal agent round: no receipt, cursor,
memory handle, or transcript is committed by this operation. The signed gateway
checkpoint installs the summary and recursive-memory frontier together later.
"""
from __future__ import annotations

import time
from typing import Any

from .summary_policy import summary_output_budget, prepare_summary_output, require_summary_completion
SUMMARY_FILENAME = "codex-native-compact-summary.json"
SUMMARY_PROMPT = (
    "Create a concise context checkpoint for another instance continuing this task. "
    "Do not perform the task or call tools. Preserve the user goal and constraints, "
    "completed work, current files/artifacts, exact paths/identifiers, important "
    "evidence, failed approaches, unresolved issues and concrete next steps. "
    "Distinguish observations from guesses. Merge the previous checkpoint if any; "
    "do not repeat the whole transcript. Return only the checkpoint summary."
)


def message_suffix(generator: Any, initial: list, content: Any) -> list[int]:
    kwargs = {"tool_names": generator._tool_names, "add_generation_prompt": False}
    prefix = list(generator._apply_native_chat_ids(initial, **kwargs))
    rendered = list(generator._apply_native_chat_ids(
        initial + [{"role": "user", "content": content}], **kwargs
    ))
    if rendered[:len(prefix)] != prefix:
        raise RuntimeError("native compaction summary changed persistent P")
    return rendered[len(prefix):]


def generate_summary(generator: Any, *, messages, episode_id: str,
                     round_index: int, raw_suffix_images=()) -> dict[str, Any]:
    # Lazy import keeps the controller's non-serving tests lightweight.
    from context_compression.grepseek_sglang_live import (
        _select_raw_suffix_images,
        _raw_suffix_image_context_reserve_tokens,
    )

    started = time.perf_counter()
    with generator._state_lock:
        if generator._episode_id != episode_id:
            raise RuntimeError("cannot summarize an inactive native episode")
        rendered = generator._render_token_clock_history(
            messages, episode_id=episode_id, round_index=round_index
        )
        initial = [dict(role=role, content=content)
                   for role, content in rendered.guided_messages[:2]]
        prefix = list(rendered.raw_prefix_input_ids)
        memory, handle = generator._memory, generator._memory_handle
        slots = int(memory.shape[0]) if memory is not None else (
            int(handle.shape[0]) if handle is not None else 0
        )
        positions = list(range(len(prefix), len(prefix) + slots))
        prefix.extend([int(generator.tokenizer.pad_token_id)] * slots)
        prefix.extend(getattr(generator, "_codex_native_summary_input_ids", ()))
        prefix.extend(rendered.history_input_ids[generator._consumed_history_tokens:])
        images = _select_raw_suffix_images(
            raw_suffix_images, consumed_round_count=generator._consumed_round_count,
            completed_round_count=round_index - 1,
        )
        content = []
        for index, img in enumerate(images):
            content.extend([
                {"type": "text", "text": f"Visual observation, round {img.completed_round_index}:"},
                {"type": "image", "image": f"request-local://image/{index}"},
            ])
        content.append({"type": "text", "text": SUMMARY_PROMPT})
        prefix.extend(message_suffix(generator, initial, content if images else SUMMARY_PROMPT))
        # Close the thinking prefix; summary decode uses the remaining serving
        # window, independently of any ordinary-round output limit.
        prefix.extend(generator.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        ))
        budget = summary_output_budget(
            input_tokens=len(prefix) + (generator._image_token_budget.reserve(images)
                if getattr(generator, "_image_token_budget", None) is not None
                else _raw_suffix_image_context_reserve_tokens(len(images))),
            server_info=dict(context_length=generator.server_context_length,
                max_total_num_tokens=generator.server_max_total_num_tokens,
                max_req_input_len=generator.server_max_req_input_len),
            post_compact_overhead=len(rendered.raw_prefix_input_ids) + slots + 1024,
        )
        kwargs = dict(max_new_tokens=budget, sampling_overrides={"temperature": 0.0})
        if images:
            kwargs["image_data"] = [img.image_data for img in images]
        if handle is not None:
            result = generator.client.generate_with_memory_handle(
                prefix, memory_positions=positions, memory_handle=handle,
                expected_hidden_size=generator.target_hidden_size, **kwargs
            )
        elif memory is not None:
            result = generator.client.generate_with_memory(
                prefix, memory_positions=positions, memory_embeddings=memory,
                expected_hidden_size=generator.target_hidden_size, **kwargs
            )
        else:
            result = generator.client.generate_discrete(prefix, **kwargs)
        body, completion = prepare_summary_output(result, tokenizer=generator.tokenizer)
        require_summary_completion(completion)
        text = generator.tokenizer.decode(body, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("native compaction returned an empty text summary")
        if not result.finished_on_eos:
            text += "\n[Checkpoint output was truncated; consult the exact memory archive when details matter.]"
        overlay = message_suffix(generator, initial,
            "[Context checkpoint: summary of earlier work, not a new user request]\n" + text)
        return dict(text=text, input_ids=overlay, input_tokens=len(prefix),
                    output_tokens=len(result.token_ids), output_budget_tokens=budget, **completion,
                    seconds=time.perf_counter() - started)
