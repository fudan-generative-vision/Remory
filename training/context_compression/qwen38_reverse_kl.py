"""Full-response KL(student || raw teacher), with detached recursive memory.

The reader runs the complete answer once, without 256-token scoring windows.
Only the vocabulary projection is chunked/checkpointed. All historical source
folds detach their previous memory; no reverse tape or cross-fold BPTT exists.
"""

from __future__ import annotations

import contextlib
from typing import NamedTuple

import torch
from torch.utils.checkpoint import checkpoint

from context_compression.qwen38_allturn_training import RecursiveCEObjective
from context_compression.train_qwen38_mmfine_joint import (
    bounded_causal_position_weights,
)


class ReverseKLOutput(NamedTuple):
    loss: torch.Tensor
    top1_agreement: torch.Tensor
    token_count: int


def chunked_reverse_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    weights: torch.Tensor,
    chunk_tokens: int = 64,
) -> ReverseKLOutput:
    """Exact full-vocabulary reverse KL at T=1, no CE or top-k approximation.

    Inputs are already next-token aligned [1, response_tokens, hidden_size].
    Weights are normalized over the WHOLE response, not projection chunks.
    Teacher probabilities are detached; student probabilities stay in autograd.
    """
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[0] != 1:
        raise ValueError(
            "student and teacher must have matching [1, tokens, hidden] shapes"
        )
    if student.device != teacher.device or weights.device != student.device:
        raise ValueError("reverse KL inputs must share a device")
    count = student.shape[1]
    if count == 0 or weights.shape != (count,) or chunk_tokens <= 0:
        raise ValueError("invalid response weights/projection chunk")
    if any(p.requires_grad for p in lm_head.parameters()):
        raise ValueError("LM head must be frozen")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("weights must be finite and positive")
    weights = weights.detach()

    def project(s, t, w):
        student_logits = lm_head(s).float()
        student_logp = student_logits.log_softmax(-1)
        with torch.no_grad():
            teacher_logits = lm_head(t.detach()).float()
            teacher_logp = teacher_logits.log_softmax(-1)
            agreement = (
                student_logits.detach().argmax(-1) == teacher_logits.argmax(-1)
            ).sum()
        per_token = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
        return (per_token * w).sum(), agreement

    numerators = []
    correct = torch.zeros((), device=student.device, dtype=torch.long)
    for start in range(0, count, chunk_tokens):
        stop = min(start + chunk_tokens, count)
        args = (
            student[0, start:stop],
            teacher[0, start:stop].detach(),
            weights[start:stop],
        )
        if torch.is_grad_enabled() and student.requires_grad:
            numerator, agree = checkpoint(project, *args, use_reentrant=False)
        else:
            numerator, agree = project(*args)
        numerators.append(numerator)
        correct += agree.detach()
    return ReverseKLOutput(
        torch.stack(numerators).sum() / weights.sum(), correct.float() / count, count
    )


def response_inputs(example: dict):
    """Yield exact raw-teacher prefixes and incremental compression sources."""
    history = list(example["prefix_ids"])
    pending: list[int] = []
    for index, segment in enumerate(example["segments"]):
        history += segment["context_ids"]
        pending += segment["context_ids"]
        yield {
            "turn": index,
            "source_ids": list(pending),
            "teacher_prefix_ids": history + segment["opening_ids"],
            "opening_ids": segment["opening_ids"],
            "target_ids": segment["response_ids"],
        }
        pending = segment["opening_ids"] + segment["response_ids"]
        history += pending


class FullResponseRKLObjective(RecursiveCEObjective):
    def __init__(
        self, *args, offload_after_tokens=4096, projection_tokens=64, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.offload_after_tokens = offload_after_tokens
        self.projection_tokens = projection_tokens

    def raw_teacher(self, prefix_ids, target_ids):
        """Native monolithic teacher. No KV-cache approximation or truncation."""
        total = len(prefix_ids) + len(target_ids)
        maximum = self.target.language_model.config.max_position_embeddings
        if total > maximum:
            raise ValueError(f"raw teacher exceeds native context: {total} > {maximum}")
        was_training = self.target.model.training
        self.target.model.eval()
        try:
            with torch.no_grad():
                embeddings = self._embeddings(prefix_ids + target_ids)
                hidden = self.target.student_hidden(embeddings, grad=False)
                start = len(prefix_ids) - 1
                # CPU keeps the raw teacher from overlapping student activations.
                aligned = (
                    hidden[:, start : start + len(target_ids)].detach().clone().cpu()
                )
        finally:
            self.target.model.train(was_training)
        return aligned

    def fold(self, event, previous):
        # Detach recurrent input and frozen source extraction, not the current
        # compressor output. The current fold remains directly trainable.
        with torch.no_grad():
            prefix = self._embeddings(self.prefix_ids)
            suffix = self._embeddings(event["ids"])
            memory = previous.detach() if previous is not None else None
            embeddings = torch.cat(
                [prefix] + ([memory] if memory is not None else []) + [suffix], dim=1
            )
            captures, hooks = {}, []
            for layer_id in self.target.target_layer_ids:

                def capture(_module, _inputs, output, *, layer_id=layer_id):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hasattr(hidden, "full_tensor"):
                        hidden = hidden.full_tensor()
                    captures[layer_id] = hidden[:, prefix.shape[1] :].detach().clone()

                hooks.append(
                    self.target.language_model.layers[layer_id].register_forward_hook(
                        capture
                    )
                )
            try:
                self.target.student_hidden(embeddings, grad=False)
            finally:
                for hook in hooks:
                    hook.remove()
            features = torch.cat(
                [captures[k] for k in self.target.target_layer_ids], dim=-1
            )
        padded = (0 if previous is None else previous.shape[1]) + event[
            "padded_new_tokens"
        ]
        result = self.compressor(
            features,
            torch.ones(features.shape[:2], dtype=torch.bool, device=features.device),
            source_token_lengths=torch.tensor(
                [features.shape[1]], device=features.device
            ),
            slot_token_lengths=torch.tensor([padded], device=features.device),
        )
        memory = result.embeddings[result.attention_mask][None]
        if memory.shape[1] != (padded + 15) // 16:
            raise RuntimeError("recursive slot geometry changed")
        return memory

    def full_response(self, event, memory, teacher):
        before = self._embeddings(self.prefix_ids)
        opening = self._embeddings(event["opening_ids"])
        inputs = torch.cat(
            [before, memory, opening, self._embeddings(event["target_ids"])], dim=1
        )
        # Non-reentrant per-layer checkpoint inputs can reside on CPU for long
        # answers. This is activation offload, not sequence/gradient truncation.
        offload = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if len(event["target_ids"]) > self.offload_after_tokens
            else contextlib.nullcontext()
        )
        with offload:
            hidden = self.target.student_hidden(inputs, grad=True)
        start = before.shape[1] + memory.shape[1] + opening.shape[1] - 1
        aligned = hidden[:, start : start + len(event["target_ids"])]
        weights = bounded_causal_position_weights(
            len(event["target_ids"]),
            absolute_start=0,
            alpha=self.alpha,
            tau=self.tau,
            power=self.power,
            device=memory.device,
        )
        return chunked_reverse_kl(
            aligned,
            teacher.to(memory.device),
            self.target.lm_head,
            weights=weights,
            chunk_tokens=self.projection_tokens,
        )


def backward_full_responses(example, objective, *, block=1024, report=None):
    memory = None
    metrics = {
        "loss_sum": 0.0,
        "correct_sum": 0.0,
        "tokens": 0,
        "reader_forwards": 0,
        "teacher_forwards": 0,
        "folds": 0,
        "responses": 0,
        "max_soft_tokens": 0,
        "max_teacher_tokens": 0,
        "max_response_tokens": 0,
    }
    for event in response_inputs(example):
        tokens = len(event["target_ids"])
        if report is not None:
            report(
                {
                    "phase": "response_start",
                    "turn": event["turn"],
                    "response_tokens": tokens,
                    "raw_teacher_tokens": len(event["teacher_prefix_ids"]) + tokens,
                }
            )
        teacher = objective.raw_teacher(
            event["teacher_prefix_ids"], event["target_ids"]
        )
        if not bool(torch.isfinite(teacher).all()):
            raise FloatingPointError("nonfinite teacher hidden states")
        source = event["source_ids"]
        if not source:
            raise ValueError("response has no incremental causal source")
        chunks = [source[i : i + block] for i in range(0, len(source), block)]
        for index, chunk in enumerate(chunks):
            last = index == len(chunks) - 1
            mode = contextlib.nullcontext() if last else torch.no_grad()
            with mode:
                memory = objective.fold(
                    {"ids": chunk, "padded_new_tokens": block},
                    memory.detach() if memory is not None else None,
                )
            metrics["folds"] += 1
        result = objective.full_response(event, memory, teacher)
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("nonfinite reverse KL")
        (result.loss * tokens).backward()
        metrics["loss_sum"] += float(result.loss.detach()) * tokens
        metrics["correct_sum"] += float(result.top1_agreement) * tokens
        metrics["tokens"] += tokens
        metrics["responses"] += 1
        metrics["reader_forwards"] += 1
        metrics["teacher_forwards"] += 1
        metrics["max_soft_tokens"] = max(metrics["max_soft_tokens"], memory.shape[1])
        metrics["max_teacher_tokens"] = max(
            metrics["max_teacher_tokens"], len(event["teacher_prefix_ids"]) + tokens
        )
        metrics["max_response_tokens"] = max(metrics["max_response_tokens"], tokens)
        memory = memory.detach()
        del result, teacher
    return metrics
