"""Shared whole-round window decisions and head-preserving tool admission."""
from __future__ import annotations

from dataclasses import replace

POLICY = "wm_accumulation_oversized_capacity_v1"
TRUNCATION_MARKER = "\n[tool output truncated to fit recursive semantic-memory capacity]"


def accumulation_target(ends, consumed, nominal, threshold):
    old = ends[consumed - 1] if consumed else 0
    eligible = ends[nominal - 1] - old if nominal > consumed else 0
    return (max(consumed, nominal) if eligible >= threshold else consumed), eligible


def oversized_rounds(ends, consumed, total_ceiling, working_memory_turns):
    threshold = total_ceiling // (working_memory_turns + 1)
    if threshold <= 0:
        raise ValueError("serving pool cannot support the working-memory window")
    found = [i + 1 for i in range(consumed, len(ends))
             if ends[i] - (ends[i - 1] if i else 0) > threshold]
    return found, threshold


def plan_window(*, ends, consumed, working_memory_turns, accumulation_tokens,
                total_ceiling, prefix_ceiling, project, maximum_target=None):
    if not 0 <= consumed <= len(ends) or any(b <= a for a, b in zip([0, *ends], ends)):
        raise ValueError("invalid whole-round frontier")
    nominal = max(0, len(ends) - working_memory_turns)
    target, eligible = accumulation_target(ends, consumed, nominal, accumulation_tokens)
    reasons = ["accumulation"] if target > consumed else []
    found, threshold = oversized_rounds(ends, target, total_ceiling, working_memory_turns)
    nonlatest = [i for i in found if i < len(ends)]
    if nonlatest and nonlatest[-1] > target:
        target = nonlatest[-1]
        reasons.append("oversized_complete_round")
    maximum = len(ends) if maximum_target is None else maximum_target
    if target > maximum:
        target = max(consumed, maximum)
    before_pressure = target
    while project(target) > prefix_ceiling and target < maximum:
        target += 1
    if target > before_pressure:
        reasons.append("capacity_pressure")
    return dict(policy=POLICY, target=target, nominal_target=nominal,
        accumulated_old_text_tokens=eligible, reasons=reasons,
        oversized_round_threshold_tokens=threshold,
        oversized_complete_round_indices=found,
        projected_prefix_tokens=project(target), maximum_prefix_tokens=prefix_ceiling,
        fits=project(target) <= prefix_ceiling,
        forced_subthreshold_flush=(target > consumed and
            ends[target - 1] - (ends[consumed - 1] if consumed else 0) < accumulation_tokens))


def admit_tool_output(result, *, tokenizer, fits, marker=TRUNCATION_MARKER):
    """Keep stdout first, then stderr; never execute or mutate compression here."""
    if fits(result):
        return result
    stdout = tokenizer.encode(result.stdout, add_special_tokens=False)
    stderr = tokenizer.encode(result.stderr, add_special_tokens=False)

    def candidate(nout, nerr, with_marker=True):
        def crop(original, tokens, count):
            if count >= len(tokens):
                return original
            return tokenizer.decode(tokens[:count], skip_special_tokens=False).rstrip() + (marker if with_marker else "")
        out, err = crop(result.stdout, stdout, nout), crop(result.stderr, stderr, nerr)
        return replace(result, stdout=out, stderr=err, truncated=True,
                       information_lines=[line for line in out.splitlines() if line.strip()])

    empty = candidate(0, 0, False)
    if not fits(empty):
        raise RuntimeError("assistant/tool metadata cannot fit even with empty stdout/stderr")
    full_stdout = candidate(len(stdout), 0)
    keep_stdout = fits(full_stdout)
    selected = full_stdout if keep_stdout else empty
    low, high = 0, len(stderr) if keep_stdout else len(stdout)
    while low <= high:
        middle = (low + high) // 2
        value = candidate(len(stdout), middle) if keep_stdout else candidate(middle, 0)
        if fits(value):
            selected, low = value, middle + 1
        else:
            high = middle - 1
    return selected


def attach_regular_runtime_policy(generator, server_info):
    # Inject a small policy interface rather than making standalone SpecForge
    # depend on the benchmark package. Residual uses the same pure functions.
    from types import SimpleNamespace
    from shared_runtime.vision_budget import ImageTokenBudget
    generator._runtime_window_policy = SimpleNamespace(
        accumulation_target=accumulation_target, oversized_rounds=oversized_rounds)
    generator._runtime_admit_tool_output = admit_tool_output
    generator._image_token_budget = ImageTokenBudget.from_server_info(server_info)

    def visual_budget(consumed, completed):
        from context_compression.grepseek_sglang_live import _select_raw_suffix_images
        images = _select_raw_suffix_images(getattr(generator, "_runtime_images", ()),
            consumed_round_count=consumed, completed_round_count=completed)
        if not images:
            return 0
        initial = generator._runtime_initial
        content = []
        for index, image in enumerate(images, 1):
            content.extend([
                {"type": "text", "text": f"Retained visual tool observation from completed round {image.completed_round_index}, image {index}:"},
                {"type": "image", "image": f"request-local://image/{index}"},
            ])
        kwargs = dict(tool_names=generator._tool_names, add_generation_prompt=False)
        before = generator._apply_native_chat_ids(initial, **kwargs)
        after = generator._apply_native_chat_ids(initial + [{"role": "user", "content": content}], **kwargs)
        if after[:len(before)] != before:
            raise RuntimeError("visual admission changed the persistent prefix")
        return len(after) - len(before) + generator._image_token_budget.reserve(images)
    generator._runtime_visual_budget = visual_budget
