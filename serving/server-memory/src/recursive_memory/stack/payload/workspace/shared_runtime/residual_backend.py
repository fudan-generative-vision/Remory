"""Native compact with summary-conditioned residual memory, not repeat memory.

Inference pins native system/tools + original user task as the prefix. Training
used system/tools only: this is an explicit inference-prefix extension, not a
checkpoint conversion. Both the summary and encoder condition on that prefix.
The first compact uses raw H; later compacts read old S + old M + new raw rounds.
Both summary generation and residual encoding use that same active context.
Cross-summary recursive reconditioning is an inference extension to training.
Logical recursion is unbounded; serving saturates only the learned depth
embedding coordinate, without resetting lineage or dropping old soft memory.
Ordinary agent generations keep the paired tools/parser/recovery/archive
behavior. REMORY_SERVER_MEMORY=1 selects server-owned references on an explicitly
pinned endpoint; legacy sessions retain their CPU-memory transport.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

from terminal_bench_eval.codex_r2eact_gateway import (
    LiveR2EActBackend, _InjectedCodexQwenContract, activate_pinned_host_rg,
)
from .summary_only_backend import SummaryOnlyGenerator
from .subagent_memory import SubagentMemoryMixin
from .sglang_streaming import GenerationStreamingMixin, POLICY as STREAMING_POLICY
from terminal_bench_eval.summary_policy import (
    summary_output_budget, prepare_summary_output, require_summary_completion,
)
from terminal_bench_eval.incremental_history import IncrementalHistory, POLICY as HISTORY_POLICY


POLICY = "codex_native_recursive_summary_residual_persistent_task_v3"
VISUAL_POLICY = "recursive_visual_summary_and_raw_tail_v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class ResidualGenerator(SubagentMemoryMixin, SummaryOnlyGenerator):
    policy = POLICY

    def __init__(self, tokenizer, client, server_info, config, tools, deployment):
        super().__init__(tokenizer, client, server_info, config, tools)
        self.text_only = not config.compressor_enabled
        if self.text_only:
            self.policy = "codex_native_text_summary_without_residual_paired_v1"
        self.deployment = deployment
        self.summary_contract = deployment["summary_contract"]
        self.residual = deployment["residual"]
        self.server_memory = getattr(client, "server_memory", False)
        self._deferred_retire = {}
        self.memory = None
        self.summary_ids = None
        self.consumed_ids = None
        self.memory_positions = []
        self.memory_depths = []
        self.initialize_subagent_memory()
        self._prepared = None
        self._installed_identity = None
        self._new_compression = False
        self._receipt = {}
        self._images = {}
        self._closing_check = lambda: False
        self._summary_observer = lambda receipt: None
        self._summary_request_index = 0
        self._summary_budget = None
        self._runtime_event = lambda receipt: None
        self._initial_prompt_observer = None
        self._initial_prompt_seen = False
        self._history = IncrementalHistory(self._canonical, self._chat)
        # Keep the exported training contract immutable; override only the
        # serving prompt, removing its numerical target as well as its cap.
        if self.summary_contract.get("prompt"):
            self._summary_suffix_ids = list(tokenizer.encode(
                "\n<|im_start|>user\n" + self.summary_contract["prompt"]
                + " Refer to the persistent system instructions and original user task above. "
                "Summarize the subsequent work, evidence and next steps; do not replace "
                "or override those original instructions."
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False))
        else:
            self._summary_suffix_ids = list(self.summary_contract["generation_suffix_ids"])

    def _ensure_open(self):
        if self._closing_check():
            raise RuntimeError("residual backend is shutting down")

    def _canonical(self, messages, episode_id):
        from context_compression.grepseek_sglang_live import _canonicalize_tools
        if self.episode_id not in (None, episode_id):
            raise RuntimeError("residual decoder cannot switch episodes")
        if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["system", "user"]:
            raise ValueError("residual history requires initial system/user messages")
        return _canonicalize_tools(episode_id, [dict(m) for m in messages])[0]

    def _chat(self, messages, *, opening=False):
        from context_compression.grepseek_sglang_live import _apply_qwen36_chat_ids
        return list(_apply_qwen36_chat_ids(self.tokenizer, messages, tool_names=self.tools,
                    add_generation_prompt=opening, model_display_name="Qwen3.8"))

    def _text_history(self, messages, episode_id):
        return self._history.render(messages, episode_id)

    def _remember_images(self, images):
        occurrences = {}
        for image in images:
            # The gateway supplies content-addressed, validated private bytes.
            # Admission and generation may carry different receipt metadata
            # for the same pixels. That must not duplicate archived images.
            # An ordinal preserves genuine repeated observations in one round.
            item = (dict(image) if isinstance(image, Mapping) else {
                "completed_round_index": image.completed_round_index,
                "image_data": image.image_data,
            })
            identity = (item["completed_round_index"], _digest(item["image_data"]))
            occurrence = occurrences.get(identity, 0)
            occurrences[identity] = occurrence + 1
            self._images.setdefault((*identity, occurrence), deepcopy(item))

    def _image_selection(self, images, completed_rounds, consumed_rounds):
        from context_compression.grepseek_sglang_live import _select_raw_suffix_images
        # Validate the supplied frontier, then project the private archive to
        # it. WM-triggered summaries end before the currently retained tail;
        # future-to-that-summary pixels must not leak into its checkpoint.
        _select_raw_suffix_images(images, consumed_round_count=0,
                                  completed_round_count=completed_rounds)
        self._remember_images(images)
        ordered = sorted((item for item in self._images.values()
                          if item["completed_round_index"] <= completed_rounds),
                         key=lambda item: item["completed_round_index"])
        return _select_raw_suffix_images(ordered, consumed_round_count=consumed_rounds,
                                        completed_round_count=completed_rounds)

    def _visual_suffix(self, canonical, images):
        if not images:
            return []
        content = []
        for index, image in enumerate(images):
            content.extend([
                {"type": "text", "text": f"Original visual observation from completed round {image.completed_round_index}, image {index + 1}:"},
                {"type": "image", "image": f"request-local://image/{index}"},
            ])
        before = self._chat(canonical[:2])
        after = self._chat([*canonical[:2], {"role": "user", "content": content}])
        if after[:len(before)] != before:
            raise RuntimeError("visual appendix changed the residual text history")
        return after[len(before):]

    def _source_identity(self, episode_id, closed):
        return _digest(dict(episode_id=episode_id, closed=closed,
                            previous_checkpoint=self._installed_identity,
                            child_folds=self._fold_source_identity()))

    def _active_text_context(self, prefix, closed):
        active = self._base_active_text_context(prefix, closed)
        return self._with_subagent_memory(prefix, closed, active)

    def _base_active_text_context(self, prefix, closed):
        # closed remains an immutable CPU audit/frontier, never re-expanded in
        # model input after installation. Soft rows keep their exact positions.
        self.memory_positions = []
        if self.summary_ids is None:
            return list(closed)
        if self.text_only:
            if closed[:len(self.consumed_ids)] != self.consumed_ids:
                raise RuntimeError("already summarized text history changed")
            contract = self.summary_contract
            return (prefix + list(contract["student_summary_before_ids"]) + self.summary_ids
                    + list(contract["student_summary_after_ids"]) + closed[len(self.consumed_ids):])
        if closed[:len(self.consumed_ids)] != self.consumed_ids:
            raise RuntimeError("already compacted residual text history changed")
        contract = self.summary_contract
        ids = (prefix + list(contract["student_summary_before_ids"]) + self.summary_ids
               + list(contract["student_summary_after_ids"])
               + list(contract["student_memory_before_ids"]))
        slots = int(self.memory.shape[0])
        self.memory_positions = list(range(len(ids), len(ids) + slots))
        return (ids + [int(self.tokenizer.pad_token_id)] * slots
                + list(contract["student_memory_after_ids"])
                + closed[len(self.consumed_ids):])

    def _project_memory_slots(self, prefix, closed):
        source_tokens = len(self._active_text_context(prefix, closed)) - len(prefix)
        block = self.residual["block_tokens"]
        ratio = self.residual.get("compression_ratio", 16)
        # Parent merges only reduce the leaf total. This is exact before any
        # merge and a safe bound afterwards, not an unconditional 4096 charge.
        slots = ((source_tokens + block - 1) // block) * (block // ratio)
        return slots if self.memory_budget is None else min(self.memory_budget, slots)

    def _render(self, messages, *, episode_id, raw_suffix_images=(), summary_request=False):
        canonical, prefix, closed = self._text_history(messages, episode_id)
        images = self._image_selection(raw_suffix_images, (len(messages) - 2) // 2,
                                       self.consumed_rounds)
        visual = self._visual_suffix(canonical, images)
        active = self._active_text_context(prefix, closed)
        if summary_request:
            ids = active + visual + self._summary_suffix_ids
        else:
            ids = active + visual + self._history.opening
        return ids, images, self.image_token_budget.reserve(images)

    def _decode(self, ids, images, reserve, *, summary_request=False, post_compact_overhead=0):
        from context_compression.grepseek_sglang_live import _generation_token_budget, QWEN36_GENERATION_SAMPLING
        self._ensure_open()
        if summary_request:
            budget = summary_output_budget(input_tokens=len(ids) + reserve,
                server_info=self.server_info, post_compact_overhead=post_compact_overhead)
            self._summary_budget = budget
        else:
            budget = _generation_token_budget(prefix_token_count=len(ids) + reserve,
                configured_max_new_tokens=self.config.max_new_tokens,
                server_context_length=self.server_info["context_length"],
                server_max_total_num_tokens=self.server_info["max_total_num_tokens"],
                server_max_req_input_len=self.server_info["max_req_input_len"],
                use_remaining_context=True,
                remaining_context_output_ceiling_tokens=self.config.remaining_context_output_ceiling_tokens)
        kwargs = dict(max_new_tokens=budget, sampling_overrides=(
            {"temperature": self.summary_contract["temperature"]} if summary_request else QWEN36_GENERATION_SAMPLING))
        if images:
            kwargs["image_data"] = [image.image_data for image in images]
        stream_context = getattr(self.client, "set_generation_stream_context", None)
        if stream_context is not None:
            stream_context(phase="summary" if summary_request else "generation",
                episode_id=self.episode_id,
                round_index=getattr(self, "_observed_round_index", None))
        if not summary_request and not self._initial_prompt_seen:
            self._initial_prompt_seen = True
            # An opt-in monitor receipt, not a second template application.
            # Never mistake a resumed history or a compact summary for the
            # original system/user input. Observer failure cannot affect decode.
            if (self._initial_prompt_observer is not None and not images
                and self.memory is None and len(self._history.canonical) == 2
                and list(ids) == self._history.prefix + self._history.opening):
                try:
                    self._initial_prompt_observer(self.tokenizer, list(ids))
                except Exception as exc:
                    try:
                        self._runtime_event(dict(event="initial_prompt_capture_failed",
                                                 error_type=type(exc).__name__))
                    except Exception:
                        pass
        if self._context_memory is None:
            return self.client.generate_discrete(ids, **kwargs)
        return self.client.generate_with_memory(ids, memory_positions=self.memory_positions,
            memory_embeddings=self._context_memory,
            expected_hidden_size=self.deployment["compressor"]["target_hidden_size"], **kwargs)

    @property
    def memory_budget(self):
        # Request-scoped remory_budget overrides must be observed dynamically.
        return self.residual["memory_budget"]

    def summarize(self, **kwargs):
        if not self.server_memory:
            return self._summarize_impl(**kwargs)
        before = set(self.client.owned_memory())
        try:
            return self._summarize_impl(**kwargs)
        except BaseException:
            # New failed candidates were never published in a caller journal.
            owned = self.client.owned_memory()
            for key in set(owned) - before:
                try:
                    self.client.release_memory(owned[key])
                except Exception as exc:
                    self._runtime_event(dict(event="uncommitted_memory_cleanup_failed", error=str(exc)))
            self._context_memory = None
            self._context_memory_key = None
            raise

    def _summarize_impl(self, *, messages, episode_id, round_index, raw_suffix_images=(), retained_context_tokens=0):
        self._ensure_open()
        if len(messages) != 2 * round_index:
            raise ValueError("residual compact requires a complete causal round boundary")
        started = time.perf_counter()
        self._runtime_event(dict(event="compression_started", episode_id=episode_id,
            round_index=getattr(self, "_observed_round_index", round_index),
            payload=dict(target_completed_rounds=round_index - 1, mode=self.policy, phase="summary")))
        _, prefix, closed = self._text_history(messages, episode_id)
        source_identity = self._source_identity(episode_id, closed)
        ids, images, reserve = self._render(messages, episode_id=episode_id,
                                          raw_suffix_images=raw_suffix_images, summary_request=True)
        overhead = (len(prefix) + sum(len(self.summary_contract[k]) for k in (
            "student_summary_before_ids", "student_summary_after_ids",
            "student_memory_before_ids", "student_memory_after_ids"))
            + self._project_memory_slots(prefix, closed) + retained_context_tokens
            + len(self._summary_suffix_ids) + self.config.capacity_pressure_reserve_tokens)
        result = self._decode(ids, images, reserve, summary_request=True, post_compact_overhead=overhead)
        body, completion = prepare_summary_output(result, tokenizer=self.tokenizer,
            end_token_id=self.summary_contract["end_token_id"])
        self._summary_request_index += 1
        # Persist even rejected output/finish reasons before any encode. A disk
        # failure does not replace the old active summary or memory.
        self._summary_observer(dict(request_index=self._summary_request_index,
            episode_id=episode_id, target_completed_rounds=round_index - 1,
            source_identity=source_identity, token_ids=list(result.token_ids),
            text=self.tokenizer.decode(body, skip_special_tokens=True),
            input_tokens=len(ids) + reserve, output_tokens=len(result.token_ids),
            output_budget_tokens=self._summary_budget, post_compact_overhead_tokens=overhead,
            seconds=time.perf_counter() - started, **completion))
        require_summary_completion(completion)
        if not body or any(token in completion["summary_stop_token_ids"] for token in body):
            raise RuntimeError("residual compact requires a nonempty valid summary body")
        text = self.tokenizer.decode(body, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("residual compact produced an empty summary")
        summary_identity = _digest(dict(source_identity=source_identity, body=body))
        self._ensure_open()
        active = self._active_text_context(prefix, closed)
        if self.text_only:
            # No encoder API, embedding tensor, memory handle or soft-token input.
            memory, depths, receipt = None, [], {}
        else:
            self._runtime_event(dict(event="compression_progress", episode_id=episode_id,
                round_index=getattr(self, "_observed_round_index", round_index),
                payload=dict(mode=self.policy, phase="residual_encode",
                    summary_tokens=len(body), active_source_tokens=len(active) - len(prefix))))
            recursive = (dict(source_memory=self._context_memory,
                              source_memory_positions=list(self.memory_positions),
                              source_memory_depths=list(self._context_depths))
                         if self._context_memory is not None else {})
            built = self.client.build_residual_memory(prefix, active[len(prefix):], body,
                contract=self.summary_contract, memory_budget=self.residual["memory_budget"],
                block_tokens=self.residual["block_tokens"], max_depth=self.deployment["compressor"]["max_depth"],
                **recursive)
            memory = built["memory_embeddings"]
            # Check shape/finite values before publishing a signed compact handle.
            import torch
            if (memory.ndim != 2 or memory.shape[1] != self.deployment["compressor"]["target_hidden_size"]
                    or memory.shape[0] <= 0
                    or (self.memory_budget is not None and memory.shape[0] > self.memory_budget)
                    or (self.server_memory and not self.client.valid_memory(memory))
                    or (not self.server_memory and not bool(torch.isfinite(memory).all()))):
                raise RuntimeError("invalid residual frontier tensor")
            receipt = dict(built["receipt"])
            depths = receipt.get("slot_depths")
            if (not isinstance(depths, list) or len(depths) != memory.shape[0]
                    or any(type(d) is not int or d <= 0 for d in depths)):
                raise RuntimeError("invalid residual frontier depth receipt")
        summary = dict(text=text, body_input_ids=body, input_tokens=len(ids) + reserve,
            output_tokens=len(result.token_ids), output_budget_tokens=self._summary_budget,
            post_compact_overhead_tokens=overhead, **completion,
            seconds=time.perf_counter() - started, policy=self.policy,
            target_completed_rounds=round_index - 1, source_identity=source_identity,
            summary_identity=summary_identity, raw_history_tokens=len(closed) - len(prefix),
            active_source_tokens=len(active) - len(prefix),
            new_raw_tokens=len(closed) - (len(self.consumed_ids) if self.consumed_ids is not None else len(prefix)),
            previous_memory_slots=int(self._context_memory.shape[0]) if self._context_memory is not None else 0,
            folded_subagents=self._fold_source_identity(),
            recursive_depth_policy=receipt.get("recursive_depth_policy"),
            logical_memory_max_depth=max(depths, default=0),
            embedding_depth_saturated=receipt.get("embedding_depth_saturated", False),
            compaction_generation=self._receipt.get("compaction_generation", 0) + 1,
            memory_slots=int(memory.shape[0]) if memory is not None else 0, residual=receipt,
            compressor_calls=0 if self.text_only else receipt.get("compressor_calls"),
            compressor_enabled=not self.text_only, residual_compaction=not self.text_only,
            persistent_prefix_policy="system_and_initial_user_v1",
            prefix_tokens=len(prefix), history_render_policy=HISTORY_POLICY,
            visual_policy=VISUAL_POLICY, summary_image_count=len(images),
            image_token_budget=self.image_token_budget.estimate(images).to_dict())
        self._ensure_open()
        # Atomic prepare/install: a failure above leaves the old active context
        # intact. The tensor never goes into JSON or the opaque Codex handle.
        if self.server_memory and self._prepared is not None:
            self._release_retired([self._prepared["memory"]])
        self._prepared = dict(identity=_digest(summary), episode_id=episode_id,
            target=round_index - 1, closed=closed, memory=memory, body=body,
            depths=list(depths), receipt=receipt, summary=deepcopy(summary), messages=deepcopy(messages),
            fold_ids=self._fold_source_identity())
        return summary

    def _release_retired(self, values):
        """Defer all old-handle retirement until the parent journal is durable."""
        if not self.server_memory:
            return
        from context_compression.remory_memory import RemoteMemory
        if not hasattr(self, '_deferred_retire'):
            self._deferred_retire = {}
        for value in values:
            if isinstance(value, RemoteMemory):
                self._deferred_retire[value.descriptor['handle_id']] = value

    def commit_remote_retirements(self):
        if not self.server_memory:
            return
        from context_compression.remory_memory import RemoteMemory
        live = [self.memory, self._context_memory,
                *((getattr(self, '_prepared', None) or {}).get(k) for k in ['memory']),
                *(e['memory'] for e in self._folded_branches)]
        retained = {v.descriptor['handle_id'] for v in live if isinstance(v, RemoteMemory)}
        for key, value in list(self._deferred_retire.items()):
            if key in retained:
                self._deferred_retire.pop(key, None)
                continue
            try:
                self.client.release_memory(value)
                self._deferred_retire.pop(key, None)
            except Exception as exc:
                self._runtime_event(dict(event='memory_release_failed', error=str(exc)))

    def install(self, *, episode_id, target_completed_rounds, summary):
        self._ensure_open()
        identity = _digest(summary)
        if self._installed_identity == (episode_id, target_completed_rounds, identity):
            return
        prepared = self._prepared
        if (prepared is None or identity != prepared["identity"]
                or episode_id != prepared["episode_id"] or target_completed_rounds != prepared["target"]
                or target_completed_rounds < self.consumed_rounds):
            raise RuntimeError("residual compact summary/history identity mismatch")
        retired = [self.memory, self._context_memory]
        retired.extend(e["memory"] for e in self._folded_branches if e["fold_id"] in prepared.get("fold_ids", ()))
        self._history.commit(prepared["messages"], episode_id)
        self._consume_prepared_folds(prepared)
        self.memory, self.summary_ids = prepared["memory"], list(prepared["body"])
        self.memory_depths = list(prepared["depths"])
        self.consumed_ids = list(prepared["closed"])
        self.consumed_rounds = target_completed_rounds
        self.summary = summary["text"]
        self._receipt = deepcopy(summary)
        self._installed_identity = (episode_id, target_completed_rounds, identity)
        self._new_compression = True
        self._prepared = None
        if self.server_memory:
            self._context_memory = None
            self._context_memory_key = None
            self._context_depths = []
            self.memory_positions = []
            self._release_retired([v for v in retired if v is not self.memory])
        self._runtime_event(dict(event="native_text_summary_installed" if self.text_only else "residual_memory_installed", episode_id=episode_id,
            round_index=getattr(self, "_observed_round_index", target_completed_rounds + 1), mode=self.policy,
            consumed_round_count=target_completed_rounds,
            memory_tokens=int(self.memory.shape[0]) if self.memory is not None else 0, compression_seconds=summary["seconds"],
            summary_tokens=len(self.summary_ids), summary_identity=summary["summary_identity"]))

    def generate_next(self, messages, *, episode_id, round_index, event_sink=None, raw_suffix_images=()):
        from context_compression.grepseek_sglang_live import _finish_generation
        started = time.perf_counter()
        self._observed_round_index = round_index
        ids, images, reserve = self._render(messages, episode_id=episode_id, raw_suffix_images=raw_suffix_images)
        sink = event_sink or self._runtime_event
        sink(dict(event="compression_finished", episode_id=episode_id, round_index=round_index,
            payload={"compression": dict(mode=self.policy,
                student_prefix_tokens=len(ids), generation_server_budget_prefix_tokens=len(ids) + reserve,
                memory_tokens=int(self._context_memory.shape[0]) if self._context_memory is not None else 0,
                summary_tokens=len(self.summary_ids or ()), consumed_round_count=self.consumed_rounds,
                working_memory_turns=None if self.config.native_compaction else self.config.working_memory_turns,
                compression_triggered=self._new_compression,
                compression_seconds=self._receipt.get("seconds", 0.0) if self._new_compression else 0.0,
                target_generation_seconds=None, history_render_policy=HISTORY_POLICY)}))
        result = self._decode(ids, images, reserve)
        self._history.commit(messages, episode_id)
        self.episode_id = episode_id
        self.pending_image_count = len(images)
        self.configure_pending_tool_images(images=(
            {"completed_round_index": image.completed_round_index, "image_data": image.image_data}
            for image in images
        ))
        slots = int(self._context_memory.shape[0]) if self._context_memory is not None else 0
        compression = dict(backend="sglang-residual", mode=self.policy, compressor_enabled=True,
            residual_compaction=True, native_compaction_text_summary=self.config.native_compaction_text_summary,
            memory_slots=slots, memory_tokens=slots, student_prefix_tokens=len(ids),
            generation_server_budget_prefix_tokens=len(ids) + reserve,
            compression_triggered=self._new_compression,
            compression_seconds=self._receipt.get("seconds", 0.0) if self._new_compression else 0.0,
            target_generation_seconds=time.perf_counter() - started,
            consumed_round_count=self.consumed_rounds,
            compressed_react_round_indices=list(range(1, self.consumed_rounds + 1)),
            compressed_message_indices=list(range(2, 2 + 2 * self.consumed_rounds)),
            archived_message_indices=list(range(2, 2 + 2 * self.consumed_rounds)),
            memory_archive_policy="exact_rounds_archived_recursive_summary_residual_v2",
            memory_transport_policy="remory_server_memory_v1" if self.server_memory else "cpu_tensor_roundtrip_v1",
            residual_receipt={k: deepcopy(self._receipt[k]) for k in (
                "summary_identity", "compaction_generation", "summary_tokens", "memory_slots", "seconds",
                "recursive_depth_policy", "logical_memory_max_depth", "embedding_depth_saturated"
            ) if k in self._receipt},
            history_render_policy=HISTORY_POLICY, persistent_prefix_policy="system_and_initial_user_v1",
            persistent_prefix_tokens=len(self._history.prefix),
            summary_tokens=len(self.summary_ids or ()),
            raw_working_memory_tokens=(len(self._history.ids) - len(self.consumed_ids or self._history.prefix)),
            visual_policy=VISUAL_POLICY,
            image_token_budget=self.image_token_budget.estimate(images).to_dict())
        if self.text_only:
            compression.update(backend="sglang-discrete", compressor_enabled=False,
                residual_compaction=False, compressor_calls=0,
                compressed_react_round_indices=[], compressed_message_indices=[],
                memory_archive_policy="exact_rounds_replaced_by_text_summary_v1",
                memory_transport_policy="none", residual_receipt={})
        if not self.config.native_compaction:
            compression.update(working_memory_residual=True,
                working_memory_turns=self.config.working_memory_turns,
                retained_raw_rounds=(len(messages) - 2) // 2 - self.consumed_rounds,
                accumulation_target_tokens=8192,
                trigger_receipt=({k: deepcopy(v) for k, v in self._trigger_receipt.items()
                                  if k != "summary"} if self._new_compression else None))
        stream_receipt = (getattr(result, "meta_info", {}) or {}).get("generation_stream")
        if stream_receipt is not None:
            compression["generation_stream"] = deepcopy(stream_receipt)
        generated = _finish_generation(tokenizer=self.tokenizer, result=result,
            eos_token_id=self.tokenizer.eos_token_id, compression=compression, qwen_native_family="qwen38")
        self._new_compression = False
        return generated


class ResidualBackend(LiveR2EActBackend):
    # Tool admission and generation reserve actual image pixels.
    supports_pixel_image_budget = True
    serving_concurrency = 1

    def _acquire_serving_lock(self, *, concurrent: bool):
        # Shared readers may interleave only with the attested no-flush,
        # request-isolated server ABI. Legacy exclusive users still exclude us.
        self.config.process_lock.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock_handle = self.config.process_lock.open("a+")
        operation = fcntl.LOCK_SH if concurrent else fcntl.LOCK_EX
        while True:
            if self._closing.is_set():
                raise RuntimeError("residual backend shut down while waiting for SGLang")
            try:
                fcntl.flock(self._process_lock_handle.fileno(), operation | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                self._closing.wait(0.1)

    def configure(self, *, episode_id, tools):
        if not (self.config.residual_compaction == self.config.compressor_enabled
                and self.config.native_compaction and self.config.native_compaction_text_summary):
            raise RuntimeError("residual backend requires native compact with summary and soft memory")
        if self._generator is not None:
            if self._episode_id != episode_id or self._tools != tuple(tools):
                raise RuntimeError("residual backend cannot switch sessions/tools")
            return
        self._episode_id, self._tools = episode_id, tuple(deepcopy(tools))
        client = None
        try:
            from .residual_preflight import (
                get_server_info, wait_for_health, RESIDUAL_CONCURRENCY_CAPABILITY,
                assert_residual_concurrency,
            )
            from .discrete_concurrency import validate_count
            import os
            server_memory_required = os.environ.get("REMORY_SERVER_MEMORY") == "1"
            concurrency = validate_count(int(os.environ.get("REMORY_SERVING_CONCURRENCY", self.serving_concurrency)))
            serving_info = get_server_info(self.config.sglang_url, cancel_event=self._closing)
            if concurrency > 1:
                assert_residual_concurrency(serving_info, concurrency)
            concurrent = RESIDUAL_CONCURRENCY_CAPABILITY in (serving_info.get("request_capabilities") or [])
            if not self.config.compressor_enabled:
                from .discrete_concurrency import assert_discrete_concurrency
                assert_discrete_concurrency(serving_info, concurrency)
                concurrent = concurrency > 1 or concurrent
            self._acquire_serving_lock(concurrent=concurrent)
            root = str(self.config.specforge_root)
            if root not in sys.path:
                sys.path.insert(0, root)
            host_rg = activate_pinned_host_rg()
            self._contract = _InjectedCodexQwenContract(self._tools)
            self._contract.__enter__()
            from transformers import AutoTokenizer
            from context_compression.qwen38_residual_serving_contract import load_residual_serving_contract
            from context_compression.qwen38_residual_sglang_client import SummaryResidualSGLangClient
            from context_compression.grepseek_data import parse_tool_response, _canonical_tool_content
            from context_compression.grepseek_tool import GrepSeekCorpusToolRunner, ToolResult
            deployment = load_residual_serving_contract(self.config.checkpoint)
            from context_compression.remory_sglang_client import ServerResidualSGLangClient
            from context_compression.remory_memory import CAPABILITY
            if server_memory_required and CAPABILITY not in serving_info.get("request_capabilities", []):
                raise RuntimeError("Pinned backend does not support required server memory")
            residual_client = ServerResidualSGLangClient if server_memory_required else SummaryResidualSGLangClient
            class StreamingResidualClient(GenerationStreamingMixin, residual_client):
                pass

            client_type = StreamingResidualClient
            if not self.config.compressor_enabled:
                from .discrete_sglang_client import StreamingDiscreteSGLangClient
                client_type = StreamingDiscreteSGLangClient
            client = client_type(self.config.sglang_url,
                request_timeout=self.config.request_timeout_seconds, enable_request_cancellation=True)
            wait_for_health(self.config.sglang_url, cancel_event=self._closing)
            # Re-read after acquiring the serving lock; never reuse a stale
            # pre-lock identity. Use the same bounded/cancellable transport.
            info = get_server_info(self.config.sglang_url, cancel_event=self._closing)
            if concurrency > 1:
                assert_residual_concurrency(info, concurrency)
            residual_attestation = (client.attest_residual_server(self.config.checkpoint, info)
                if self.config.compressor_enabled else None)
            client.configure_generation_streaming(artifact_dir=self.artifact_dir,
                incremental=info.get("incremental_streaming_output", False),
                event_sink=self._event, cancelled=self._closing.is_set)
            if concurrent and RESIDUAL_CONCURRENCY_CAPABILITY not in (info.get("request_capabilities") or []):
                raise RuntimeError("SGLang lost its residual concurrency attestation during initialization")
            for key in ("context_length", "max_total_num_tokens", "max_req_input_len"):
                if type(info.get(key)) is not int or info[key] <= 0:
                    raise RuntimeError(f"SGLang is missing a valid {key}")
            model = Path(deployment["target_model"])
            model_config = json.loads((model / "config.json").read_text())
            if "Qwen3_5ForConditionalGeneration" not in model_config.get("architectures", []):
                raise RuntimeError("residual experiment requires the paired Qwen3.8 model")
            if self.config.vision_enabled and (not model_config.get("vision_config") or info.get("language_only") is True or info.get("enable_multimodal") is False):
                raise RuntimeError("residual experiment requires the paired vision capability")
            tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
            for name, expected in deployment["summary_contract"]["tokenizer_files"].items():
                if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"residual tokenizer differs from training: {name}")
            from .qwen_reasoning import bind_reasoning_effort
            tokenizer, reasoning_receipt = bind_reasoning_effort(
                tokenizer, getattr(self.config, "qwen_reasoning_effort", None))
            if reasoning_receipt is not None:
                from .storage import atomic_json
                atomic_json(self.artifact_dir / "qwen-reasoning-policy.json", reasoning_receipt)
                self._event(dict(event="qwen_reasoning_effort_configured", **reasoning_receipt))
            client.eos_token_id = tokenizer.eos_token_id
            cache = client.attest_sglang_cache_policy(info)
            client.configure_generation_degeneration_guard(True, check_interval_tokens=256,
                minimum_periodic_span_tokens=1024, maximum_period_tokens=256,
                minimum_repeat_count=8, max_retries=1, return_detected_generation=True)
            if not self.config.native_compaction:
                raise RuntimeError("The release does not include the working-memory trigger")
            self._generator = ResidualGenerator(tokenizer, client, info, self.config,
                                                [t["name"] for t in self._tools], deployment)
            self._generator._closing_check = self._closing.is_set
            self._generator._summary_observer = self._record_summary_generation
            self._generator._runtime_event = self._event
            self._generator._initial_prompt_observer = getattr(self, "initial_prompt_observer", None)
            self._memory_runner = GrepSeekCorpusToolRunner(corpus_dir=self.config.corpus_dir,
                tokenizer=tokenizer, timeout=180, max_output_tokens=2048).restricted(("shell",), shell_corpus="react_memory")
            self._memory_runner.begin_episode(episode_id)
            self._tool_result_type, self._parse_tool_response = ToolResult, parse_tool_response
            self._canonical_tool_content = _canonical_tool_content
            self._persist_memory()
            self._event(dict(event="gateway_backend_configured", episode_id=episode_id,
                tools=[t["name"] for t in self._tools], native_compaction=self.config.native_compaction,
                native_compaction_text_summary=self.config.native_compaction_text_summary,
                residual_compaction=self.config.residual_compaction, compressor_enabled=self.config.compressor_enabled,
                compressor_calls=0, memory_slots=0,
                compressor_checkpoint=str(self.config.checkpoint) if self.config.compressor_enabled else None,
                summary_contract_source=str(self.config.checkpoint), policy=self._generator.policy,
                working_memory_turns=(None if self.config.native_compaction else self.config.working_memory_turns),
                accumulation_target_tokens=(None if self.config.native_compaction else 8192),
                model_path=str(model), host_rg=host_rg, sglang_cache_attestation=cache,
                residual_attestation=residual_attestation, visual_policy=VISUAL_POLICY,
                prefill_memory_capabilities=[cap for cap in (info.get("request_capabilities") or [])
                    if cap in {"qwen38_bounded_gdn_all_prefill_v2", "qwen38_bounded_mlp_all_prefill_v2"}],
                generation_transport_policy=STREAMING_POLICY,
                serving_lock_mode="shared" if concurrent else "exclusive",
                serving_concurrency=concurrency,
                max_running_requests=info.get("max_running_requests")))
        except BaseException:
            if self._generator is None and client is not None and getattr(client, "server_memory", False):
                client.close_memory()
            self.close()
            raise

    def _record_summary_generation(self, receipt):
        from .storage import atomic_json
        path = self.artifact_dir / "summary-generations" / f"{receipt['request_index']:04d}.json"
        atomic_json(path, receipt)
        self._event(dict(event="residual_summary_generated", artifact=str(path),
            **{k: v for k, v in receipt.items() if k not in {"text", "token_ids"}}))

    def summarize_native_compaction(self, **kwargs):
        if not self.config.native_compaction:
            raise RuntimeError("Codex native compaction is disabled for the working-memory residual experiment")
        summary = self._generator.summarize(**kwargs)
        self._event(dict(event="residual_compaction_prepared", episode_id=self._episode_id, summary=summary))
        return summary

    def stage_native_compaction(self, *, episode_id, target_completed_rounds, handle_digest, summary=None):
        if summary is None:
            raise RuntimeError("residual compaction cannot install memory without its summary")
        self._generator.install(episode_id=episode_id, target_completed_rounds=target_completed_rounds, summary=summary)
        self._event(dict(event="residual_compaction_installed", episode_id=episode_id,
            target_completed_rounds=target_completed_rounds, handle_digest=handle_digest, policy=self._generator.policy))

    def reserve_pending_tool_images(self, count):
        self._generator.pending_image_count = count
        self._generator.pending_images = None

    def close(self):
        generator = self._generator
        try:
            super().close()
        finally:
            if generator is not None and getattr(generator, 'server_memory', False):
                generator.client.close_memory()

    def configure_pending_tool_images(self, *, images):
        self._generator.configure_pending_tool_images(images=images)

    def prepare_tool_message_for_native_compaction(self, **kwargs):
        return self._prepare_tool_message(**kwargs, native_compaction_request=True)
