"""Full teacher forcing with compact-summary + detached pyramid residual memory.

The reader backpropagates once to the concatenated frontier. Each retained C
node is then replayed for its exact local VJP. This keeps one C activation graph
resident and trains EVERY retained node without tree BPTT or reader rollouts.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from context_compression.qwen38_residual_data import residual_events, summary_envelope
from context_compression.qwen38_residual_pyramid import plan_pyramid
from context_compression.qwen38_reverse_kl import chunked_reverse_kl
from context_compression.train_qwen38_mmfine_joint import bounded_causal_position_weights


@dataclass
class ResidualMemory:
    embeddings: torch.Tensor
    summary_sha256: str
    history_sha256: str
    plan: object
    frontier_features: dict

    def require_identity(self, event, summary):
        if (self.summary_sha256 != summary["summary_sha256"]
                or self.history_sha256 != event.history_sha256):
            raise ValueError("memory was built for a different history/summary; rebuild from raw history")


class ResidualObjective:
    def __init__(self, target, compressor, summary_contract, *, memory_budget=4096,
                 block=1024, ratio=16, max_depth=8, alpha=1.0, tau=64.0, power=0.5,
                 projection_tokens=64, offload_after_tokens=4096,
                 feature_residency="cpu", node_batch_size=1,
                 reader_checkpoint_after_tokens=0):
        if any(p.requires_grad for p in target.model.parameters()):
            raise ValueError("residual teacher/student reader must be frozen")
        if any("lora" in name.lower() for name, _ in target.model.named_parameters()):
            raise ValueError("LoRA is not permitted in residual teacher or student")
        self.target, self.compressor, self.contract = target, compressor, summary_contract
        self.memory_budget, self.block, self.ratio, self.max_depth = memory_budget, block, ratio, max_depth
        self.alpha, self.tau, self.power = alpha, tau, power
        self.projection_tokens, self.offload_after_tokens = projection_tokens, offload_after_tokens
        if feature_residency not in ("cpu", "gpu") or node_batch_size < 1:
            raise ValueError("invalid residual execution configuration")
        self.feature_residency = feature_residency
        self.node_batch_size = node_batch_size
        if reader_checkpoint_after_tokens < 0:
            raise ValueError("reader checkpoint threshold must be nonnegative")
        self.reader_checkpoint_after_tokens = reader_checkpoint_after_tokens
        self._reader_checkpoint_enabled = None

    def _cache(self, value, *, copy=False):
        device = "cpu" if self.feature_residency == "cpu" else self.target.device
        return value.detach().to(device=device, copy=copy)

    def _embed(self, ids):
        if not ids:
            weight = self.target.model.get_input_embeddings().weight
            return torch.empty(1, 0, self.target.hidden_size, device=self.target.device, dtype=weight.dtype)
        return self.target.embed_ids(ids).detach()

    def _check_length(self, total, branch):
        maximum = self.target.language_model.config.max_position_embeddings
        if total > maximum:
            raise ValueError(f"{branch} exceeds native context: {total} > {maximum}; no truncation")

    @contextlib.contextmanager
    def _frozen_eval(self):
        training = self.target.model.training
        self.target.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.target.model.train(training)

    def _capture(self, embeddings, start, stop):
        """Retain ONLY causal source rows on the configured cache device."""
        self._check_length(embeddings.shape[1], "source/teacher")
        captures, hooks = {}, []
        for layer_id in self.target.target_layer_ids:
            def capture(_module, _args, value, *, layer_id=layer_id):
                value = value[0] if isinstance(value, tuple) else value
                if hasattr(value, "full_tensor"):
                    value = value.full_tensor()
                captures[layer_id] = self._cache(value[:, start:stop], copy=True)
            hooks.append(self.target.language_model.layers[layer_id].register_forward_hook(capture))
        try:
            hidden = self.target.student_hidden(embeddings, grad=False)
        finally:
            for hook in hooks:
                hook.remove()
        features = torch.cat([captures[k] for k in self.target.target_layer_ids], dim=-1)
        return features, hidden

    def raw_teacher_and_source(self, event):
        # No summary, soft memory, altered prefix, or adapter enters this branch.
        prefix = event.prefix_ids + event.history_ids + event.opening_ids
        ids = prefix + event.target_ids
        self._check_length(len(ids), "raw teacher")
        with self._frozen_eval():
            features, hidden = self._capture(self._embed(ids), len(event.prefix_ids),
                                             len(event.prefix_ids) + len(event.history_ids))
            start = len(prefix) - 1
            teacher = self._cache(hidden[:, start:start + len(event.target_ids)], copy=True)
        return features, teacher

    def summary_features(self, event, summary):
        ids = list(event.prefix_ids) + summary_envelope(summary["ids"], self.contract)
        self._check_length(len(ids), "summary encoder")
        before = len(event.prefix_ids) + len(self.contract["student_summary_before_ids"])
        with self._frozen_eval():
            hidden = self.target.student_hidden(self._embed(ids), grad=False)
            return self._cache(hidden[:, before:before + len(summary["ids"])], copy=True)

    def parent_features(self, prefix_ids, child_memory):
        with self._frozen_eval():
            prefix = self._embed(prefix_ids)
            inputs = torch.cat([prefix, child_memory.detach().to(prefix)], dim=1)
            features, _ = self._capture(inputs, prefix.shape[1], inputs.shape[1])
        return features

    def compress_node(self, features, summary_features, input_depth):
        return self.compress_nodes([features], summary_features, input_depth)

    def compress_nodes(self, items, summary_features, input_depth):
        """Batch independent nodes at one depth; padding is masked, never scored."""
        device = self.target.device
        lengths = [x.shape[1] for x in items]
        features = torch.nn.utils.rnn.pad_sequence(
            [x[0].to(device) for x in items], batch_first=True)
        mask = torch.arange(features.shape[1], device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        summary_features = summary_features.to(device).expand(len(items), -1, -1)
        output = self.compressor(features,
            mask,
            source_token_lengths=torch.tensor(lengths, device=device),
            slot_token_lengths=torch.full((len(items),), self.block, device=device),
            summary_features=summary_features,
            summary_attention_mask=torch.ones(summary_features.shape[:2], dtype=torch.bool, device=device),
            input_depth=input_depth)
        result = output.embeddings
        if not bool(output.attention_mask.all()):
            raise ValueError("virtual block padding must retain every allocated slot")
        if result.shape[1] != self.block // self.ratio:
            raise ValueError("compressor/pyramid slot geometry mismatch")
        return result

    def build_memory(self, event, summary, raw_features, summary_features):
        """Rebuild at each compact, using ONE fixed S for all levels (no stale-S reuse)."""
        if raw_features.shape[1] != len(event.history_ids):
            raise ValueError("raw features do not cover complete history")
        plan = plan_pyramid(len(event.history_ids), budget=self.memory_budget,
                            block=self.block, ratio=self.ratio, max_depth=self.max_depth)
        frontier = set(plan.frontier)
        outputs, replay = {}, {}
        with torch.no_grad():
            position = 0
            while position < len(plan.nodes):
                first = plan.nodes[position]
                group = []
                for node in plan.nodes[position:position + self.node_batch_size]:
                    if node.depth != first.depth or bool(node.children) != bool(first.children):
                        break
                    if node.children and any(k not in outputs for k in node.children):
                        break
                    group.append(node)
                if not group:
                    raise RuntimeError("pyramid batch has an unresolved dependency")
                if first.children:
                    children = torch.cat([torch.cat([outputs[k] for k in node.children], dim=1)
                                          for node in group], dim=0)
                    if children.shape[1] != self.block:
                        raise ValueError("parent must consume exactly one full physical block")
                    with self._frozen_eval():
                        prefix = self._embed(event.prefix_ids).expand(len(group), -1, -1)
                        inputs = torch.cat([prefix, children.to(prefix)], dim=1)
                        encoded, _ = self._capture(inputs, prefix.shape[1], inputs.shape[1])
                    features = list(encoded.split(1))
                    for node in group:
                        for k in node.children:
                            del outputs[k]
                else:
                    features = [raw_features[:, node.start:node.stop] for node in group]
                compressed = self.compress_nodes(features, summary_features, first.depth - 1)
                for node, feature, value in zip(group, features, compressed.split(1)):
                    outputs[node.index] = self._cache(value)
                    if node.index in frontier:
                        replay[node.index] = self._cache(feature)
                position += len(group)
            memory = torch.cat([outputs[k] for k in plan.frontier], dim=1).to(self.target.device)
        if not bool(torch.isfinite(memory).all()):
            raise FloatingPointError("nonfinite residual memory")
        return ResidualMemory(memory, summary["summary_sha256"], event.history_sha256, plan, replay)

    def compress_history(self, event, summary):
        """Inference-compatible rebuild: never accesses target_ids or runs the teacher."""
        ids = event.prefix_ids + event.history_ids
        self._check_length(len(ids), "raw compression source")
        with self._frozen_eval():
            features, _ = self._capture(self._embed(ids), len(event.prefix_ids), len(ids))
        encoded_summary = self.summary_features(event, summary)
        memory = self.build_memory(event, summary, features, encoded_summary)
        memory.frontier_features.clear()  # serving has no replay tape
        return memory

    def response_loss(self, event, summary, memory, teacher):
        before = (list(event.prefix_ids) + summary_envelope(summary["ids"], self.contract)
                  + self.contract["student_memory_before_ids"])
        opening = event.opening_ids if event.student_opening_ids is None else event.student_opening_ids
        after = self.contract["student_memory_after_ids"] + list(opening)
        start = len(before) + memory.shape[1] + len(after) - 1
        self._check_length(start + 1 + len(event.target_ids), "residual student")
        inputs = torch.cat([self._embed(before), memory, self._embed(after + list(event.target_ids))], dim=1)
        if self.reader_checkpoint_after_tokens:
            enabled = inputs.shape[1] > self.reader_checkpoint_after_tokens
            if enabled != self._reader_checkpoint_enabled:
                if enabled:
                    self.target.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    self.target.model.gradient_checkpointing_disable()
                self._reader_checkpoint_enabled = enabled
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=memory.is_cuda)
                   if inputs.shape[1] > self.offload_after_tokens else contextlib.nullcontext())
        with offload:
            hidden = self.target.student_hidden(inputs, grad=True)
        student = hidden[:, start:start + len(event.target_ids)]
        weights = bounded_causal_position_weights(len(event.target_ids), absolute_start=event.target_offset,
            alpha=self.alpha, tau=self.tau, power=self.power, device=memory.device)
        return chunked_reverse_kl(student, teacher.to(memory.device), self.target.lm_head,
                                  weights=weights, chunk_tokens=self.projection_tokens)

    def backward_response(self, event, summary):
        raw_features, teacher = self.raw_teacher_and_source(event)
        if not bool(torch.isfinite(teacher).all()):
            raise FloatingPointError("nonfinite raw teacher")
        summary_features = self.summary_features(event, summary)
        memory = self.build_memory(event, summary, raw_features, summary_features)
        memory.require_identity(event, summary)
        leaf = memory.embeddings.detach().requires_grad_(True)
        result = self.response_loss(event, summary, leaf, teacher)
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("nonfinite residual KL")
        # Token-sum accumulation; the optimizer divides by total scored tokens.
        (result.loss * result.token_count).backward()
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
            raise FloatingPointError("missing/nonfinite reader-to-memory gradient")
        gradient = self._cache(leaf.grad)
        metrics = {"loss_sum": float(result.loss.detach()) * result.token_count,
                   "correct_sum": float(result.top1_agreement) * result.token_count,
                   "tokens": result.token_count, "responses": 1, "reader_forwards": 1,
                   "teacher_forwards": 1, "summary_forwards": 1,
                   "parent_forwards": sum(bool(n.children) for n in memory.plan.nodes),
                   "compressor_forwards": len(memory.plan.nodes) + len(memory.plan.frontier),
                   "frontier_vjps": len(memory.plan.frontier),
                   "source_tokens": len(event.history_ids), "covered_source_tokens": memory.plan.source_tokens,
                   "max_soft_tokens": memory.plan.soft_tokens, "max_depth": memory.plan.max_depth,
                   "max_teacher_tokens": len(event.prefix_ids + event.history_ids + event.opening_ids + event.target_ids),
                   "max_response_tokens": result.token_count}
        del result, leaf, teacher
        width = memory.plan.slots_per_node
        position = 0
        while position < len(memory.plan.frontier):
            indices = []
            depth = memory.plan.nodes[memory.plan.frontier[position]].depth
            for index in memory.plan.frontier[position:position + self.node_batch_size]:
                if memory.plan.nodes[index].depth != depth:
                    break
                indices.append(index)
            replay = self.compress_nodes([memory.frontier_features.pop(i) for i in indices],
                                         summary_features, depth - 1)
            gradients = torch.cat([gradient[:, k*width:(k+1)*width]
                                   for k in range(position, position + len(indices))], dim=0)
            replay.backward(gradients.to(replay))
            position += len(indices)
        return metrics


def backward_residual_responses(example, objective, summary_cache, *, report=None):
    metrics = {}
    for event in residual_events(example):
        summary = summary_cache.get(event)
        if report:
            report({"phase": "residual_response", "turn": event.turn,
                    "history_tokens": len(event.history_ids), "response_tokens": len(event.target_ids)})
        row = objective.backward_response(event, summary)
        for key, value in row.items():
            metrics[key] = max(metrics.get(key, 0), value) if key.startswith("max_") else metrics.get(key, 0) + value
    return metrics
