"""Frozen Qwen3.8 multimodal target helpers for context compression.

The existing recursive-compressor trainer deliberately consumes a derived
text-only snapshot.  MMFineReasoning needs the released multimodal checkpoint
instead: the raw branch must replace image placeholder tokens with visual
features, while the compressed student branch consumes ordinary continuous
tokens in the language backbone.  Keeping those two paths explicit prevents a
soft-memory forward from accidentally reusing image inputs.

This module is intentionally independent of the legacy trainer.  It contains
only rendering/alignment helpers and a thin frozen-target facade; task sampling,
loss weighting, FSDP, and checkpoint policy remain the caller's responsibility.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import nn
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)

QWEN38_MULTIMODAL_MODEL_TYPE = "qwen3_5"
QWEN38_MULTIMODAL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
DEFAULT_QWEN38_TARGET_LAYER_IDS = (1, 16, 31, 46, 61)
QWEN38_REPEAT_INSTRUCTION = "\nRepeat the compressed text exactly."
QWEN38_EMPTY_THINKING_PREFIX = "<think>\n\n</think>\n\n"

Qwen38MultimodalModel: TypeAlias = Qwen3_5ForConditionalGeneration


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token span in a single rendered sequence."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.stop, bool)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("TokenSpan must be a non-empty half-open span")

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class RenderedQwen38Target:
    """One raw source/target sequence and its compressed-student layout.

    ``raw_model_inputs`` contains the complete teacher-forced multimodal
    sequence.  The continuous student replaces ``source_span`` with memory and
    uses the two discrete prefix fragments recorded here.  ``target_span`` is
    expressed in the raw sequence; its first token is predicted by hidden row
    ``target_span.start - 1``.
    """

    raw_model_inputs: Mapping[str, Any]
    source_span: TokenSpan
    target_span: TokenSpan
    student_prefix_before_memory_ids: tuple[int, ...]
    student_prefix_after_memory_ids: tuple[int, ...]
    assistant_prefix_ids: tuple[int, ...]
    target_prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        input_ids = self.raw_model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("raw_model_inputs.input_ids must be [batch, sequence]")
        if input_ids.shape[0] != 1:
            raise ValueError("RenderedQwen38Target currently requires batch size one")
        sequence_width = int(input_ids.shape[1])
        if self.source_span.stop > sequence_width:
            raise ValueError("source span is outside raw input_ids")
        if self.target_span.stop > sequence_width:
            raise ValueError("target span is outside raw input_ids")
        if self.target_span.start <= self.source_span.stop:
            raise ValueError("target span must follow the source span")
        if self.target_span.width != len(self.target_ids):
            raise ValueError("target span and target_ids widths differ")
        if not self.student_prefix_before_memory_ids:
            raise ValueError("student prefix before memory must not be empty")
        if not self.student_prefix_after_memory_ids:
            raise ValueError("student prefix after memory must not be empty")

    @property
    def target_width(self) -> int:
        return len(self.target_ids)


@dataclass(frozen=True)
class Qwen38RawForwardOutput:
    """Detached source features and raw-teacher prediction rows."""

    source_features: torch.Tensor
    teacher_hidden: torch.Tensor


@dataclass(frozen=True)
class Qwen38StudentForwardOutput:
    """Student hidden states plus explicit causal alignment metadata."""

    hidden_states: torch.Tensor
    target_ids: torch.Tensor
    prefix_width: int


def _plain_token_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError(f"{label} must have batch size one")
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f"{label} must be a one-dimensional token sequence")
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a token sequence")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} contains an invalid token id")
        result.append(int(token))
    return tuple(result)


def _encode(tokenizer: Any, text: str, *, label: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise TypeError(f"{label} must be text")
    value = tokenizer.encode(text, add_special_tokens=False)
    ids = _plain_token_ids(value, label=label)
    if not ids:
        raise ValueError(f"{label} tokenized to an empty sequence")
    return ids


def _unique_subsequence_start(
    sequence: Sequence[int],
    needle: Sequence[int],
    *,
    label: str,
    floor: int = 0,
) -> int:
    if not needle:
        raise ValueError(f"{label} marker must not be empty")
    starts = [
        index
        for index in range(floor, len(sequence) - len(needle) + 1)
        if tuple(sequence[index : index + len(needle)]) == tuple(needle)
    ]
    if len(starts) != 1:
        raise ValueError(
            f"rendered sequence must contain exactly one {label} marker after "
            f"position {floor}; observed {len(starts)}"
        )
    return starts[0]


def qwen38_user_content_span(input_ids: Any, tokenizer: Any) -> TokenSpan:
    """Locate the sole user-content span in a one-turn Qwen3.8 render.

    The caller renders without a generation prompt or an explicit system
    message.  Marker IDs are derived from the tokenizer rather than hard-coded,
    and image-placeholder expansion is therefore harmless to the alignment.
    """

    ids = _plain_token_ids(input_ids, label="input_ids")
    user_header = _encode(tokenizer, "<|im_start|>user\n", label="Qwen3.8 user header")
    message_end = _encode(tokenizer, "<|im_end|>", label="Qwen3.8 message end")
    header_start = _unique_subsequence_start(ids, user_header, label="user header")
    content_start = header_start + len(user_header)
    end_start = _unique_subsequence_start(
        ids,
        message_end,
        label="user message end",
        floor=content_start,
    )
    return TokenSpan(content_start, end_start)


def strip_leading_image_marker(question: str) -> str:
    """Remove the dataset protocol's single leading ``<image>`` marker."""

    if not isinstance(question, str):
        raise TypeError("question must be text")
    stripped = question.lstrip()
    if stripped.startswith("<image>"):
        stripped = stripped[len("<image>") :].lstrip()
    if not stripped:
        raise ValueError("question is empty after removing the image marker")
    return stripped


def _copy_processor_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        try:
            value = dict(value)
        except (TypeError, ValueError) as error:
            raise TypeError("processor must return a mapping") from error
    result = dict(value)
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        tensor = result.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"processor output {name} must be [batch, sequence]")
        if tensor.shape[0] != 1:
            raise ValueError("Qwen3.8 rendering currently requires batch size one")
    width = result["input_ids"].shape[1]
    if any(
        result[name].shape[1] != width
        for name in ("attention_mask", "mm_token_type_ids")
    ):
        raise ValueError("processor sequence tensors have inconsistent widths")
    return result


def _append_text_tokens(
    model_inputs: Mapping[str, Any],
    appended_ids: Sequence[int],
) -> dict[str, Any]:
    ids = _plain_token_ids(appended_ids, label="appended_ids")
    if not ids:
        raise ValueError("appended_ids must not be empty")
    result = dict(model_inputs)
    input_ids = result["input_ids"]
    suffix = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)[None]
    result["input_ids"] = torch.cat((input_ids, suffix), dim=1)
    attention_mask = result["attention_mask"]
    result["attention_mask"] = torch.cat(
        (
            attention_mask,
            torch.ones(
                (1, len(ids)),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ),
        dim=1,
    )
    token_types = result["mm_token_type_ids"]
    result["mm_token_type_ids"] = torch.cat(
        (
            token_types,
            torch.zeros(
                (1, len(ids)),
                dtype=token_types.dtype,
                device=token_types.device,
            ),
        ),
        dim=1,
    )
    return result


def _insert_text_tokens(
    model_inputs: Mapping[str, Any],
    inserted_ids: Sequence[int],
    *,
    position: int,
) -> dict[str, Any]:
    """Insert ordinary text tokens into the three sequence-aligned inputs."""

    ids = _plain_token_ids(inserted_ids, label="inserted_ids")
    if not ids:
        raise ValueError("inserted_ids must not be empty")
    result = dict(model_inputs)
    input_ids = result["input_ids"]
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        or position > input_ids.shape[1]
    ):
        raise ValueError("text-token insertion position is outside input_ids")
    suffix = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)[None]
    result["input_ids"] = torch.cat(
        (input_ids[:, :position], suffix, input_ids[:, position:]), dim=1
    )
    for name, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
        tensor = result[name]
        inserted = torch.full(
            (1, len(ids)), fill, dtype=tensor.dtype, device=tensor.device
        )
        result[name] = torch.cat(
            (tensor[:, :position], inserted, tensor[:, position:]), dim=1
        )
    return result


def _replace_text_tokens(
    model_inputs: Mapping[str, Any],
    replacement_ids: Sequence[int] | torch.Tensor,
    *,
    span: TokenSpan,
) -> dict[str, Any]:
    """Replace one text span in every sequence-aligned processor tensor."""

    ids = _plain_token_ids(replacement_ids, label="replacement_ids")
    if not ids:
        raise ValueError("replacement_ids must not be empty")
    result = dict(model_inputs)
    input_ids = result["input_ids"]
    if span.stop > input_ids.shape[1]:
        raise ValueError("text-token replacement span is outside input_ids")
    replacement = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)[
        None
    ]
    result["input_ids"] = torch.cat(
        (input_ids[:, : span.start], replacement, input_ids[:, span.stop :]), dim=1
    )
    for name, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
        tensor = result[name]
        inserted = torch.full(
            (1, len(ids)), fill, dtype=tensor.dtype, device=tensor.device
        )
        result[name] = torch.cat(
            (tensor[:, : span.start], inserted, tensor[:, span.stop :]), dim=1
        )
    return result


def _resolve_target_ids(
    tokenizer: Any,
    *,
    target_text: str | None,
    target_token_ids: Sequence[int] | torch.Tensor | None,
    target_prefix_token_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (target_text is None) == (target_token_ids is None):
        raise ValueError("provide exactly one of target_text or target_token_ids")
    prefix_ids = (
        ()
        if target_prefix_token_ids is None
        else _plain_token_ids(
            target_prefix_token_ids,
            label="target_prefix_token_ids",
        )
    )
    if target_text is not None:
        if prefix_ids:
            raise ValueError(
                "target_prefix_token_ids require the token-id target interface"
            )
        return (), _encode(tokenizer, target_text, label="target_text")
    target_ids = _plain_token_ids(target_token_ids, label="target_token_ids")
    if not target_ids:
        raise ValueError("target_token_ids must not be empty")
    return prefix_ids, target_ids


def _render_qwen38_target(
    processor: Any,
    *,
    content: Sequence[Mapping[str, Any]],
    target_text: str | None,
    target_token_ids: Sequence[int] | torch.Tensor | None,
    target_prefix_token_ids: Sequence[int] | torch.Tensor | None,
    shared_user_suffix_text: str | None,
    assistant_prefix_text: str | None,
    source_token_ids: Sequence[int] | torch.Tensor | None,
    template_kwargs: Mapping[str, Any] | None,
) -> RenderedQwen38Target:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor has no tokenizer")
    kwargs = dict(template_kwargs or {})
    forbidden = {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "enable_thinking",
    }.intersection(kwargs)
    if forbidden:
        raise ValueError(
            "template_kwargs cannot override fixed rendering controls: "
            + ", ".join(sorted(forbidden))
        )
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": [dict(item) for item in content]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        # With add_generation_prompt=False this suppresses the checkpoint's
        # synthetic reasoning-system message without inserting a closed think
        # block.  The stored target retains its own <think>...</think> markup.
        enable_thinking=False,
        **kwargs,
    )
    source_inputs = _copy_processor_output(rendered)
    source_ids = _plain_token_ids(source_inputs["input_ids"], label="source input_ids")
    source_span = qwen38_user_content_span(source_ids, tokenizer)
    if source_token_ids is not None:
        exact_source_ids = _plain_token_ids(source_token_ids, label="source_token_ids")
        if not exact_source_ids:
            raise ValueError("source_token_ids must not be empty")
        source_inputs = _replace_text_tokens(
            source_inputs, exact_source_ids, span=source_span
        )
        source_span = TokenSpan(
            source_span.start, source_span.start + len(exact_source_ids)
        )
        source_ids = _plain_token_ids(
            source_inputs["input_ids"], label="source input_ids with exact tokens"
        )
        if tuple(source_ids[source_span.start : source_span.stop]) != exact_source_ids:
            raise RuntimeError("exact source-token replacement changed token IDs")
    shared_user_suffix_ids = (
        _encode(
            tokenizer,
            shared_user_suffix_text,
            label="shared_user_suffix_text",
        )
        if shared_user_suffix_text is not None
        else ()
    )
    if shared_user_suffix_ids:
        # Keep the source span restricted to the text being compressed.  The
        # shared instruction is inserted immediately afterward, before the
        # user-message terminator, and is therefore identical for raw teacher
        # and compressed student.
        source_inputs = _insert_text_tokens(
            source_inputs,
            shared_user_suffix_ids,
            position=source_span.stop,
        )
        source_ids = _plain_token_ids(
            source_inputs["input_ids"], label="source input_ids with shared suffix"
        )
    assistant_header = _encode(
        tokenizer,
        "<|im_start|>assistant\n",
        label="Qwen3.8 assistant header",
    )
    assistant_prefix_ids = (
        _encode(
            tokenizer,
            assistant_prefix_text,
            label="assistant_prefix_text",
        )
        if assistant_prefix_text is not None
        else ()
    )
    target_prefix_ids, target_ids = _resolve_target_ids(
        tokenizer,
        target_text=target_text,
        target_token_ids=target_token_ids,
        target_prefix_token_ids=target_prefix_token_ids,
    )
    raw_inputs = _append_text_tokens(
        source_inputs,
        (
            *assistant_header,
            *assistant_prefix_ids,
            *target_prefix_ids,
            *target_ids,
        ),
    )
    target_start = (
        len(source_ids)
        + len(assistant_header)
        + len(assistant_prefix_ids)
        + len(target_prefix_ids)
    )
    target_span = TokenSpan(target_start, target_start + len(target_ids))
    before_memory = tuple(source_ids[: source_span.start])
    after_memory = (
        *source_ids[source_span.stop :],
        *assistant_header,
        *assistant_prefix_ids,
        *target_prefix_ids,
    )
    return RenderedQwen38Target(
        raw_model_inputs=raw_inputs,
        source_span=source_span,
        target_span=target_span,
        student_prefix_before_memory_ids=before_memory,
        student_prefix_after_memory_ids=tuple(after_memory),
        assistant_prefix_ids=assistant_prefix_ids,
        target_prefix_ids=target_prefix_ids,
        target_ids=target_ids,
    )


def render_qwen38_multimodal_target(
    processor: Any,
    *,
    image: Any,
    question: str,
    target_text: str | None = None,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    target_prefix_token_ids: Sequence[int] | torch.Tensor | None = None,
    template_kwargs: Mapping[str, Any] | None = None,
) -> RenderedQwen38Target:
    """Render ``Image + Question -> target`` with an explicit source span.

    Long responses can be passed as an already-tokenized deterministic window:
    ``target_prefix_token_ids`` supplies the unscored causal history and
    ``target_token_ids`` supplies the scored suffix.  Both teacher and student
    consume those exact IDs; no decode/re-tokenize round trip is performed.
    """

    if image is None:
        raise ValueError("image must not be None")
    clean_question = strip_leading_image_marker(question)
    return _render_qwen38_target(
        processor,
        content=(
            {"type": "image", "image": image},
            {"type": "text", "text": clean_question},
        ),
        target_text=target_text,
        target_token_ids=target_token_ids,
        target_prefix_token_ids=target_prefix_token_ids,
        shared_user_suffix_text=None,
        assistant_prefix_text=None,
        source_token_ids=None,
        template_kwargs=template_kwargs,
    )


def render_qwen38_text_target(
    processor: Any,
    *,
    source_text: str | None = None,
    source_token_ids: Sequence[int] | torch.Tensor | None = None,
    target_text: str | None = None,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    target_prefix_token_ids: Sequence[int] | torch.Tensor | None = None,
    shared_user_suffix_text: str | None = None,
    assistant_prefix_text: str | None = None,
    template_kwargs: Mapping[str, Any] | None = None,
) -> RenderedQwen38Target:
    """Render a text-only compression/reconstruction event.

    ``shared_user_suffix_text`` is placed after the raw source or continuous
    memory and before the user-message terminator.  ``assistant_prefix_text``
    is placed after the assistant header.  Both are shared exactly by raw
    teacher and compressed student and are excluded from the scored target.
    """

    if (source_text is None) == (source_token_ids is None):
        raise ValueError("provide exactly one of source_text or source_token_ids")
    if source_text is not None and (
        not isinstance(source_text, str) or not source_text.strip()
    ):
        raise ValueError("source_text must be non-empty")
    # Token chunks must enter the model byte-for-byte as their original IDs.
    # A harmless placeholder is rendered only to obtain the checkpoint's exact
    # chat-template envelope; _render_qwen38_target replaces its content span.
    template_source = source_text if source_text is not None else "x"
    return _render_qwen38_target(
        processor,
        content=({"type": "text", "text": template_source},),
        target_text=target_text,
        target_token_ids=target_token_ids,
        target_prefix_token_ids=target_prefix_token_ids,
        shared_user_suffix_text=shared_user_suffix_text,
        assistant_prefix_text=assistant_prefix_text,
        source_token_ids=source_token_ids,
        template_kwargs=template_kwargs,
    )


def render_qwen38_repeat_target(
    processor: Any,
    *,
    source_text: str | None = None,
    source_token_ids: Sequence[int] | torch.Tensor | None = None,
    target_text: str | None = None,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    target_prefix_token_ids: Sequence[int] | torch.Tensor | None = None,
    template_kwargs: Mapping[str, Any] | None = None,
) -> RenderedQwen38Target:
    """Render the fixed MMFineReasoning exact-repeat distillation protocol.

    Raw teacher and compressed student receive the same repeat instruction in
    the user turn.  Their assistant turns both start with an empty, already
    closed thinking segment, and scoring begins only on the repeated text.
    """

    return render_qwen38_text_target(
        processor,
        source_text=source_text,
        source_token_ids=source_token_ids,
        target_text=target_text,
        target_token_ids=target_token_ids,
        target_prefix_token_ids=target_prefix_token_ids,
        shared_user_suffix_text=QWEN38_REPEAT_INSTRUCTION,
        assistant_prefix_text=QWEN38_EMPTY_THINKING_PREFIX,
        template_kwargs=template_kwargs,
    )


def qwen38_multimodal_config(target_model: str | Path) -> dict[str, Any]:
    """Read and validate the released Qwen3.8 multimodal config."""

    root = Path(target_model).expanduser().resolve()
    path = root / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Qwen3.8 multimodal config: {path}") from error
    if not isinstance(config, dict):
        raise TypeError("Qwen3.8 multimodal config must be a JSON object")
    if config.get("model_type") != QWEN38_MULTIMODAL_MODEL_TYPE:
        raise ValueError("target must use the qwen3_5 multimodal model type")
    if config.get("architectures") != [QWEN38_MULTIMODAL_ARCHITECTURE]:
        raise ValueError("target must use Qwen3_5ForConditionalGeneration")
    text_config = config.get("text_config")
    vision_config = config.get("vision_config")
    if not isinstance(text_config, dict) or not isinstance(vision_config, dict):
        raise TypeError("target config must contain text_config and vision_config")
    text_hidden = text_config.get("hidden_size")
    vision_output = vision_config.get("out_hidden_size")
    if (
        isinstance(text_hidden, bool)
        or not isinstance(text_hidden, int)
        or text_hidden <= 0
        or vision_output != text_hidden
    ):
        raise ValueError("vision merger output must match the text hidden size")
    return config


def load_qwen38_multimodal_model(
    target_model: str | Path,
    **from_pretrained_kwargs: Any,
) -> Qwen38MultimodalModel:
    """Load the full local checkpoint without permitting remote fallback."""

    qwen38_multimodal_config(target_model)
    if from_pretrained_kwargs.get("local_files_only") is False:
        raise ValueError("Qwen3.8 multimodal loading cannot enable remote fallback")
    from_pretrained_kwargs["local_files_only"] = True
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        target_model,
        **from_pretrained_kwargs,
    )


def load_qwen38_processor(
    target_model: str | Path,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load the checkpoint's local Qwen3-VL processor."""

    qwen38_multimodal_config(target_model)
    if from_pretrained_kwargs.get("local_files_only") is False:
        raise ValueError("Qwen3.8 processor loading cannot enable remote fallback")
    from_pretrained_kwargs["local_files_only"] = True
    return AutoProcessor.from_pretrained(target_model, **from_pretrained_kwargs)


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(value, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else value


def _last_hidden_state(output: Any, *, label: str) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError(f"{label} must return [batch, sequence, hidden] states")
    return _full_tensor(hidden)


class FrozenQwen38MultimodalTarget:
    """Frozen full Qwen3.8 target with separate raw and soft-memory paths."""

    def __init__(
        self,
        model: Qwen38MultimodalModel,
        *,
        target_layer_ids: Sequence[int] = DEFAULT_QWEN38_TARGET_LAYER_IDS,
        gradient_checkpointing: bool = True,
    ) -> None:
        if getattr(model.config, "model_type", None) != QWEN38_MULTIMODAL_MODEL_TYPE:
            raise ValueError("model is not a Qwen3.8 multimodal target")
        core = getattr(model, "model", None)
        language_model = getattr(core, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise ValueError("multimodal target has no language_model.layers")
        ids = tuple(int(layer_id) for layer_id in target_layer_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(layer_id < 0 or layer_id >= len(layers) for layer_id in ids)
        ):
            raise ValueError(
                "target_layer_ids are incompatible with the language model"
            )
        text_config = getattr(model.config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise ValueError("multimodal target has no valid text hidden size")
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, nn.Module):
            raise TypeError("multimodal target has no lm_head")

        self.model = model.requires_grad_(False)
        self.core = core
        self.language_model = language_model
        self.target_layer_ids = ids
        self._hidden_size = hidden_size
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            self.model.train()
        else:
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def device(self) -> torch.device:
        embedding = self.model.get_input_embeddings().weight
        return torch.device(embedding.device)

    def _inputs_on_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }

    def embed_ids(self, ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        values = _plain_token_ids(ids, label="ids")
        if not values:
            raise ValueError("ids must not be empty")
        tensor = torch.tensor(values, dtype=torch.long, device=self.device)[None]
        return _full_tensor(self.model.get_input_embeddings()(tensor))

    def raw_source_features(
        self,
        rendered: RenderedQwen38Target,
    ) -> torch.Tensor:
        """Capture frozen text-source features without evaluating the target.

        Repeat CE needs the frozen reader as a feature extractor and LM head,
        but has no teacher-distribution branch.  Because the backbone is
        causal, tokens after ``source_span.stop`` cannot affect source rows;
        truncating there avoids an otherwise wasted raw target forward.
        """

        inputs = self._inputs_on_device(rendered.raw_model_inputs)
        if "pixel_values" in inputs or "image_grid_thw" in inputs:
            raise ValueError("raw_source_features is restricted to text-only inputs")
        full_width = int(inputs["input_ids"].shape[1])
        source = rendered.source_span
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            tensor = inputs.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.shape != (1, full_width)
            ):
                raise ValueError(f"raw source input {name} has an invalid shape")
            inputs[name] = tensor[:, : source.stop]

        captures: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_id in self.target_layer_ids:

            def capture(_module, _inputs, output, *, layer_id=layer_id):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError(
                        "target decoder layer returned non-tensor hidden state"
                    )
                captures[layer_id] = (
                    _full_tensor(hidden)[:, source.start : source.stop, :]
                    .clone()
                    .detach()
                )

            hooks.append(
                self.language_model.layers[layer_id].register_forward_hook(capture)
            )

        self.core.rope_deltas = None
        try:
            with torch.no_grad():
                self.core(**inputs, use_cache=False, return_dict=True)
        finally:
            for hook in hooks:
                hook.remove()
            self.core.rope_deltas = None
        if tuple(captures) != self.target_layer_ids:
            raise RuntimeError(
                f"selected target captures are incomplete: {tuple(captures)}"
            )
        source_features = torch.cat(
            [captures[layer_id] for layer_id in self.target_layer_ids], dim=-1
        )
        expected_shape = (
            1,
            source.width,
            len(self.target_layer_ids) * self.hidden_size,
        )
        if source_features.shape != expected_shape:
            raise RuntimeError("raw source feature capture produced the wrong shape")
        return source_features

    def raw_source_and_teacher(
        self,
        rendered: RenderedQwen38Target,
    ) -> Qwen38RawForwardOutput:
        """Run one no-grad multimodal pass for source features and teacher rows."""

        inputs = self._inputs_on_device(rendered.raw_model_inputs)
        captures: dict[int, torch.Tensor] = {}
        hooks = []
        source = rendered.source_span
        for layer_id in self.target_layer_ids:

            def capture(_module, _inputs, output, *, layer_id=layer_id):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError(
                        "target decoder layer returned non-tensor hidden state"
                    )
                captures[layer_id] = (
                    _full_tensor(hidden)[:, source.start : source.stop, :]
                    .clone()
                    .detach()
                )

            hooks.append(
                self.language_model.layers[layer_id].register_forward_hook(capture)
            )

        # Qwen3.8 caches a request-specific mRoPE delta on the multimodal core.
        # Clear it both before and after every independent raw example.
        self.core.rope_deltas = None
        try:
            with torch.no_grad():
                output = self.core(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
            final_hidden = _last_hidden_state(output, label="multimodal core")
        finally:
            for hook in hooks:
                hook.remove()
            self.core.rope_deltas = None
        if tuple(captures) != self.target_layer_ids:
            raise RuntimeError(
                f"selected target captures are incomplete: {tuple(captures)}"
            )
        target = rendered.target_span
        prediction_start = target.start - 1
        prediction_stop = target.stop - 1
        teacher_hidden = (
            final_hidden[:, prediction_start:prediction_stop, :].clone().detach()
        )
        if teacher_hidden.shape != (1, target.width, self.hidden_size):
            raise RuntimeError("raw teacher target alignment produced the wrong shape")
        source_features = torch.cat(
            [captures[layer_id] for layer_id in self.target_layer_ids], dim=-1
        )
        expected_source_shape = (
            1,
            source.width,
            len(self.target_layer_ids) * self.hidden_size,
        )
        if source_features.shape != expected_source_shape:
            raise RuntimeError("raw source feature capture produced the wrong shape")
        return Qwen38RawForwardOutput(
            source_features=source_features,
            teacher_hidden=teacher_hidden,
        )

    def student_hidden(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        grad: bool = True,
    ) -> torch.Tensor:
        """Run only the text backbone on continuous-token student inputs."""

        if (
            not isinstance(inputs_embeds, torch.Tensor)
            or inputs_embeds.ndim != 3
            or not torch.is_floating_point(inputs_embeds)
            or inputs_embeds.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "inputs_embeds must be floating [batch, sequence, hidden_size]"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        elif (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != inputs_embeds.shape[:2]
            or attention_mask.device != inputs_embeds.device
        ):
            raise ValueError(
                "attention_mask must match inputs_embeds [batch, sequence]"
            )
        mode = contextlib.nullcontext() if grad else torch.no_grad()
        with mode:
            output = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return _last_hidden_state(output, label="Qwen3.8 language model")

    def student_from_memory(
        self,
        rendered: RenderedQwen38Target,
        memory: torch.Tensor,
        *,
        grad: bool = True,
    ) -> Qwen38StudentForwardOutput:
        """Replace the raw source span with memory and teacher-force the target."""

        if (
            not isinstance(memory, torch.Tensor)
            or memory.ndim != 3
            or memory.shape[0] != 1
            or memory.shape[1] <= 0
            or memory.shape[2] != self.hidden_size
            or not torch.is_floating_point(memory)
        ):
            raise ValueError("memory must be floating [1, slots, hidden_size]")
        if memory.device != self.device:
            raise ValueError("memory and frozen target must share a device")
        before = self.embed_ids(rendered.student_prefix_before_memory_ids)
        after = self.embed_ids(rendered.student_prefix_after_memory_ids)
        target_ids = torch.tensor(
            rendered.target_ids,
            dtype=torch.long,
            device=self.device,
        )[None]
        target_embeddings = _full_tensor(self.model.get_input_embeddings()(target_ids))
        prefix_width = before.shape[1] + memory.shape[1] + after.shape[1]
        inputs_embeds = torch.cat(
            (before, memory, after, target_embeddings),
            dim=1,
        )
        hidden = self.student_hidden(inputs_embeds, grad=grad)
        return Qwen38StudentForwardOutput(
            hidden_states=hidden,
            target_ids=target_ids,
            prefix_width=int(prefix_width),
        )


__all__ = [
    "DEFAULT_QWEN38_TARGET_LAYER_IDS",
    "QWEN38_EMPTY_THINKING_PREFIX",
    "QWEN38_MULTIMODAL_ARCHITECTURE",
    "QWEN38_MULTIMODAL_MODEL_TYPE",
    "QWEN38_REPEAT_INSTRUCTION",
    "FrozenQwen38MultimodalTarget",
    "Qwen38RawForwardOutput",
    "Qwen38StudentForwardOutput",
    "RenderedQwen38Target",
    "TokenSpan",
    "load_qwen38_multimodal_model",
    "load_qwen38_processor",
    "qwen38_multimodal_config",
    "qwen38_user_content_span",
    "render_qwen38_multimodal_target",
    "render_qwen38_repeat_target",
    "render_qwen38_text_target",
    "strip_leading_image_marker",
]
