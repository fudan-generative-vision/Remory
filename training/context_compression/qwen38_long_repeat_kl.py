"""Long-source extension of the frozen-reader full-response repeat KL objective.

The historical 16K objective remains unchanged in qwen38_repeat_full_kl.py.
"""
from __future__ import annotations

import contextlib
import torch

from context_compression import train_qwen38_mmfine_joint as previous
from context_compression.qwen38_multimodal_target import render_qwen38_repeat_target
from context_compression.qwen38_reverse_kl import chunked_reverse_kl


def render_full_repeat(processor, source_ids):
    """Exact source IDs, full response plus terminator, no prefilled answer."""
    end = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(end) != 1:
        raise ValueError("expected one native turn terminator")
    return render_qwen38_repeat_target(
        processor, source_token_ids=source_ids,
        target_token_ids=tuple(source_ids) + tuple(end),
        target_prefix_token_ids=(),
    )


def full_repeat_kl(rendered, *, target, compressor, source_offset=0,
                   projection_tokens=64, offload_after_tokens=131072, maximum_source_tokens=65536):
    """One causal reader forward covering EVERY response position.

    Teacher: raw X + instruction + empty thinking + teacher-forced X/EOT.
    Student: C(X) + identical instruction/empty thinking + forced X/EOT.
    Only preceding response tokens are visible at each prediction position.
    Source chunks retain their original position coordinate for weighting.
    """
    if rendered.source_span.width > maximum_source_tokens:
        raise ValueError("source exceeds the explicit long-repeat budget")
    if rendered.target_prefix_ids:
        raise ValueError("repeat must not prefill an unscored answer prefix")
    if any(p.requires_grad for p in target.model.parameters()):
        raise ValueError("the whole reader/teacher must be frozen, without LoRA")
    if any("lora" in name.lower() for name, _ in target.model.named_parameters()):
        raise ValueError("repeat does not allow reader LoRA parameters")
    maximum = target.language_model.config.max_position_embeddings
    if rendered.raw_model_inputs["input_ids"].shape[1] > maximum:
        raise ValueError("full raw repeat exceeds native context; no truncation allowed")
    was_training = target.model.training
    target.model.eval()
    try:
        with torch.no_grad():
            raw = target.raw_source_and_teacher(rendered)
    finally:
        target.model.train(was_training)
    # Raw teacher has no graph and cannot receive gradients from KL.
    teacher = raw.teacher_hidden.detach()
    offload = (rendered.source_span.width >= offload_after_tokens)
    saving = (torch.autograd.graph.save_on_cpu(pin_memory=True)
              if offload else contextlib.nullcontext())
    with saving:
        memory, source_tokens, padded = previous._compress_rendered(
            rendered, raw.source_features.detach(), compressor=compressor,
            block_tokens=1024, maximum_blocks=(maximum_source_tokens + 1023) // 1024, compression_ratio=16,
        )
        student = target.student_from_memory(rendered, memory, grad=True)
    count = len(rendered.target_ids)
    start = student.prefix_width - 1
    aligned = student.hidden_states[:, start:start + count]
    weights = previous.bounded_causal_position_weights(
        count, absolute_start=source_offset, alpha=1.0, tau=64.0, power=0.5,
        device=aligned.device,
    )
    result = chunked_reverse_kl(
        aligned, teacher, target.lm_head, weights=weights,
        chunk_tokens=projection_tokens,
    )
    return result.loss, {
        "reverse_kl": float(result.loss.detach()),
        "top1_agreement": float(result.top1_agreement),
        "tokens": count, "source_tokens": source_tokens,
        "padded_source_tokens": padded, "soft_tokens": memory.shape[1],
        "teacher_tokens": rendered.raw_model_inputs["input_ids"].shape[1],
        "target_offset": source_offset, "target_prefix_tokens": 0,
    }
