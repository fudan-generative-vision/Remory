"""Exact reverse-replay BPTT with only one recursive fold live on the GPU.

The forward tape stores soft states and their adjoints on CPU. Each scored
window backpropagates into its memory leaf. A reverse sweep recomputes every
fold, accumulates compressor parameter gradients, and propagates its input
memory adjoint to the preceding fold. This is the chain rule, NOT truncated
BPTT: no trajectory-depth cutoff, and no optimizer update inside a trajectory.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import torch

from context_compression.losses import frozen_lm_head_chunked_token_ce
from context_compression.qwen38_allturn_data import event_plan
from context_compression.train_qwen38_mmfine_joint import (
    bounded_causal_position_weights,
)


@dataclass
class FoldState:
    event: dict
    previous: torch.Tensor | None
    output: torch.Tensor
    adjoint: torch.Tensor
    rng_cpu: torch.Tensor
    rng_cuda: torch.Tensor | None


def backward_trajectory(
    example: dict,
    *,
    fold: Callable,
    objective: Callable,
    device: torch.device,
    block: int = 1024,
    window: int = 256,
    verify_replay: bool = False,
) -> dict:
    """Accumulate the token-summed CE gradient for one complete trajectory."""
    states: list[FoldState] = []
    memory = None
    metrics = {
        "loss_sum": 0.0,
        "correct_sum": 0.0,
        "tokens": 0,
        "windows": 0,
        "folds": 0,
        "responses": len(example["segments"]),
        "max_soft_tokens": 0,
    }
    for event in event_plan(example, block=block, window=window):
        if event["kind"] == "fold":
            cpu_rng = torch.get_rng_state()
            cuda_rng = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            )
            previous = memory.detach().cpu() if memory is not None else None
            with torch.no_grad():
                memory = fold(event, memory)
            if not bool(torch.isfinite(memory).all()):
                raise FloatingPointError("nonfinite forward memory")
            output = memory.detach().cpu()
            states.append(
                FoldState(
                    event,
                    previous,
                    output,
                    torch.zeros_like(
                        output,
                        dtype=(
                            torch.float64
                            if output.dtype == torch.float64
                            else torch.float32
                        ),
                    ),
                    cpu_rng,
                    cuda_rng,
                )
            )
            metrics["folds"] += 1
            metrics["max_soft_tokens"] = max(
                metrics["max_soft_tokens"], memory.shape[1]
            )
        else:
            if memory is None:
                raise ValueError("assistant target has no compressed causal history")
            leaf = memory.detach().requires_grad_(True)
            loss, accuracy = objective(event, leaf)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite CE")
            count = len(event["target_ids"])
            (loss * count).backward()
            if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
                raise FloatingPointError("missing/nonfinite memory adjoint")
            states[-1].adjoint.add_(
                leaf.grad.detach().cpu().to(states[-1].adjoint.dtype)
            )
            metrics["loss_sum"] += float(loss.detach()) * count
            metrics["correct_sum"] += float(accuracy) * count
            metrics["tokens"] += count
            metrics["windows"] += 1
            del loss, leaf
    del memory
    for index in range(len(states) - 1, -1, -1):
        state = states[index]
        previous = (
            state.previous.to(device).requires_grad_(True)
            if state.previous is not None
            else None
        )
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(state.rng_cpu)
            if state.rng_cuda is not None:
                torch.cuda.set_rng_state(state.rng_cuda, device)
            output = fold(state.event, previous)
            if verify_replay:
                torch.testing.assert_close(
                    output.detach().cpu(), state.output, rtol=0.02, atol=0.001
                )
            output.backward(state.adjoint.to(device=device, dtype=output.dtype))
        if previous is not None:
            if previous.grad is None or not bool(torch.isfinite(previous.grad).all()):
                raise FloatingPointError("recursive gradient edge was lost")
            states[index - 1].adjoint.add_(
                previous.grad.detach().cpu().to(states[index - 1].adjoint.dtype)
            )
        del output, previous
        # Release the CPU tape as well, once its reverse edge has been consumed.
        states.pop()
    return metrics


class RecursiveCEObjective:
    def __init__(
        self,
        target: Any,
        compressor: Any,
        prefix_ids: list[int],
        *,
        alpha: float,
        tau: float,
        power: float,
        ce_chunk_size: int = 64,
    ):
        self.target = target
        self.compressor = compressor
        self.prefix_ids = prefix_ids
        self.alpha, self.tau, self.power = alpha, tau, power
        self.ce_chunk_size = ce_chunk_size

    def _embeddings(self, ids: list[int]) -> torch.Tensor:
        if ids:
            return self.target.embed_ids(ids)
        return self.target.model.get_input_embeddings().weight.new_empty(
            (1, 0, self.target.hidden_size)
        )

    def fold(self, event: dict, previous: torch.Tensor | None) -> torch.Tensor:
        prefix = self._embeddings(self.prefix_ids)
        suffix = self._embeddings(event["ids"])
        parts = [prefix] + ([previous] if previous is not None else []) + [suffix]
        embeddings = torch.cat(parts, dim=1)
        captures = {}
        hooks = []
        for layer_id in self.target.target_layer_ids:

            def capture(_module, _inputs, output, *, layer_id=layer_id):
                hidden = output[0] if isinstance(output, tuple) else output
                if hasattr(hidden, "full_tensor"):
                    hidden = hidden.full_tensor()
                captures[layer_id] = hidden[:, prefix.shape[1] :, :].clone()

            hooks.append(
                self.target.language_model.layers[layer_id].register_forward_hook(
                    capture
                )
            )
        try:
            self.target.student_hidden(embeddings, grad=torch.is_grad_enabled())
        finally:
            for hook in hooks:
                hook.remove()
        if set(captures) != set(self.target.target_layer_ids):
            raise RuntimeError("recursive source feature capture incomplete")
        features = torch.cat(
            [captures[k] for k in self.target.target_layer_ids], dim=-1
        )
        physical = features.shape[1]
        previous_width = previous.shape[1] if previous is not None else 0
        padded = previous_width + event["padded_new_tokens"]
        result = self.compressor(
            features,
            torch.ones(features.shape[:2], device=features.device, dtype=torch.bool),
            source_token_lengths=torch.tensor([physical], device=features.device),
            slot_token_lengths=torch.tensor([padded], device=features.device),
        )
        memory = result.embeddings[result.attention_mask][None]
        if memory.shape[1] != (padded + 15) // 16:
            raise RuntimeError("recursive physical-rate slot count mismatch")
        return memory

    def objective(
        self, event: dict, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        before = self._embeddings(self.prefix_ids)
        after = self._embeddings(event["prefix_ids"])
        target_ids = torch.tensor(
            event["target_ids"], device=memory.device, dtype=torch.long
        )[None]
        embeddings = torch.cat(
            [before, memory, after, self._embeddings(event["target_ids"])], dim=1
        )
        hidden = self.target.student_hidden(embeddings, grad=True)
        weights = bounded_causal_position_weights(
            target_ids.shape[1],
            absolute_start=event["position"],
            alpha=self.alpha,
            tau=self.tau,
            power=self.power,
            device=memory.device,
        )[None]
        loss = frozen_lm_head_chunked_token_ce(
            hidden,
            target_ids,
            self.target.lm_head,
            prefix_width=before.shape[1] + memory.shape[1] + after.shape[1],
            token_weights=weights,
            chunk_size=self.ce_chunk_size,
        )
        return loss.loss, loss.token_accuracy
