"""Codex-native text compaction using only SGLang discrete generation.

Shares Qwen framing, tool parsing, receipts, recovery and the JSONL archive with
the paired backend. Never loads a Compressor bundle/checkpoint or allocates soft
memory. A server with an idle Compressor hook can also serve these requests.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import sys
import time

from terminal_bench_eval.codex_native_summary import SUMMARY_PROMPT
from terminal_bench_eval.summary_policy import summary_output_budget, prepare_summary_output, require_summary_completion
from terminal_bench_eval.codex_r2eact_gateway import (
    LiveR2EActBackend, _InjectedCodexQwenContract, activate_pinned_host_rg,
    CODEX_NATIVE_TEXT_COMPACTION_POLICY,
)
from .vision_budget import ImageTokenBudget


POLICY = CODEX_NATIVE_TEXT_COMPACTION_POLICY


class SummaryOnlyGenerator:
    """A discrete decoder with a summary frontier, not a recursive generator."""

    def __init__(self, tokenizer, client, server_info, config, tools):
        self.tokenizer, self.client, self.config = tokenizer, client, config
        self.tools = tuple(tools)
        self.server_info = server_info
        self.consumed_rounds = 0
        self.summary = None
        self.last_summary_target = None
        self.episode_id = None
        self.pending_image_count = 0
        self.pending_images = None
        self.image_token_budget = ImageTokenBudget.from_server_info(server_info)
        self._initial_prompt_seen = False
        self._initial_prompt_observer = None
        self._runtime_event = lambda event: None

    def _render(self, messages, *, episode_id, raw_suffix_images=(), summary_request=False):
        from context_compression.grepseek_sglang_live import (
            _canonicalize_tools, _apply_qwen36_chat_ids, _select_raw_suffix_images,
        )
        if self.episode_id not in (None, episode_id):
            raise RuntimeError("summary-only decoder cannot switch episodes")
        # Slice before canonicalizing: past rounds remain on disk, not in the
        # target's prompt. Keep original system/task exactly once.
        frontier = 2 + 2 * self.consumed_rounds
        if len(messages) < frontier:
            raise RuntimeError("text summary frontier exceeds the transcript")
        selected = [dict(m) for m in messages[:2]]
        if self.summary is not None:
            selected.append({"role": "user", "content":
                "[Context checkpoint: summary of earlier work, not a new user request]\n" + self.summary})
        selected.extend(dict(m) for m in messages[frontier:])
        selected, _, _ = _canonicalize_tools(episode_id, selected)
        images = _select_raw_suffix_images(raw_suffix_images,
            consumed_round_count=self.consumed_rounds,
            completed_round_count=(len(messages) - 2) // 2)
        if images:
            content = []
            for i, image in enumerate(images, 1):
                content.extend([
                    {"type": "text", "text": f"Retained visual tool observation from completed round {image.completed_round_index}, image {i}:"},
                    {"type": "image", "image": f"request-local://image/{i}"},
                ])
            selected.append({"role": "user", "content": content})
        if summary_request:
            selected.append({"role": "user", "content": SUMMARY_PROMPT})
        ids = _apply_qwen36_chat_ids(self.tokenizer, selected, tool_names=self.tools,
            add_generation_prompt=not summary_request, model_display_name="Qwen3.8")
        if summary_request:
            ids.extend(self.tokenizer.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False))
        reserve = self.image_token_budget.reserve(images)
        return ids, images, reserve

    def _decode(self, ids, images, reserve, *, summary_request=False):
        from context_compression.grepseek_sglang_live import _generation_token_budget, QWEN36_GENERATION_SAMPLING
        if summary_request:
            budget = summary_output_budget(input_tokens=len(ids) + reserve, server_info=self.server_info,
                post_compact_overhead=self.config.capacity_pressure_reserve_tokens)
            self._summary_budget = budget
        else:
            budget = _generation_token_budget(prefix_token_count=len(ids) + reserve,
                configured_max_new_tokens=self.config.max_new_tokens,
                server_context_length=self.server_info.get("context_length"),
                server_max_total_num_tokens=self.server_info.get("max_total_num_tokens"),
                server_max_req_input_len=self.server_info.get("max_req_input_len"),
                use_remaining_context=True,
                remaining_context_output_ceiling_tokens=self.config.remaining_context_output_ceiling_tokens)
        kwargs = {"max_new_tokens": budget,
            "sampling_overrides": {"temperature": 0.0} if summary_request else QWEN36_GENERATION_SAMPLING}
        if images:
            kwargs["image_data"] = [image.image_data for image in images]
        # This is deliberately the ONLY model-operation call in this backend.
        return self.client.generate_discrete(ids, **kwargs)

    def generate_next(self, messages, *, episode_id, round_index, event_sink=None, raw_suffix_images=()):
        from context_compression.grepseek_sglang_live import _finish_generation
        started = time.perf_counter()
        ids, images, reserve = self._render(messages, episode_id=episode_id, raw_suffix_images=raw_suffix_images)
        if not self._initial_prompt_seen:
            self._initial_prompt_seen = True
            if (self._initial_prompt_observer is not None and not images
                and self.summary is None and self.consumed_rounds == 0 and len(messages) == 2):
                try:
                    self._initial_prompt_observer(self.tokenizer, list(ids))
                except Exception as exc:
                    try:
                        self._runtime_event(dict(event="initial_prompt_capture_failed", error_type=type(exc).__name__))
                    except Exception:
                        pass  # Telemetry must not change model execution.
        result = self._decode(ids, images, reserve)
        self.episode_id = episode_id
        self.pending_image_count = len(images)
        self.configure_pending_tool_images(images=(
            {"completed_round_index": image.completed_round_index, "image_data": image.image_data}
            for image in images
        ))
        compression = {
            "backend": "sglang-discrete", "mode": POLICY, "compressor_enabled": False,
            "compressor_calls": 0, "memory_slots": 0, "memory_tokens": 0,
            "compression_triggered": False, "compression_seconds": 0.0,
            "student_prefix_tokens": len(ids), "generation_server_budget_prefix_tokens": len(ids) + reserve,
            "target_generation_seconds": time.perf_counter() - started,
            "consumed_round_count": self.consumed_rounds,
            "compressed_react_round_indices": [], "compressed_message_indices": [],
            "archived_message_indices": list(range(2, 2 + 2 * self.consumed_rounds)),
            "memory_archive_policy": "exact_rounds_replaced_by_text_summary_v1",
            "native_compaction_text_summary": True,
            "image_token_budget": self.image_token_budget.estimate(images).to_dict(),
        }
        return _finish_generation(tokenizer=self.tokenizer, result=result,
            eos_token_id=self.tokenizer.eos_token_id, compression=compression, qwen_native_family="qwen38")

    def summarize(self, *, messages, episode_id, round_index, raw_suffix_images=()):
        started = time.perf_counter()
        ids, images, reserve = self._render(messages, episode_id=episode_id,
            raw_suffix_images=raw_suffix_images, summary_request=True)
        result = self._decode(ids, images, reserve, summary_request=True)
        body, completion = prepare_summary_output(result, tokenizer=self.tokenizer)
        require_summary_completion(completion)
        text = self.tokenizer.decode(body, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("native compaction returned an empty text summary")
        if not result.finished_on_eos:
            text += "\n[Checkpoint output was truncated; consult the exact memory archive when details matter.]"
        # Preparing a summary does not change the active context. Install only
        # when Codex sends back the gateway's authenticated compaction handle.
        self.last_summary_target = (episode_id, round_index - 1, text)
        return dict(text=text, input_tokens=len(ids) + reserve, output_tokens=len(result.token_ids),
            output_budget_tokens=self._summary_budget, **completion, seconds=time.perf_counter() - started,
            compressor_calls=0, memory_slots=0, policy=POLICY,
            image_token_budget=self.image_token_budget.estimate(images).to_dict())

    def install(self, *, episode_id, target_completed_rounds, summary):
        expected = (episode_id, target_completed_rounds, summary["text"])
        if self.last_summary_target != expected or target_completed_rounds < self.consumed_rounds:
            raise RuntimeError("summary-only native compaction receipt mismatch")
        self.consumed_rounds = target_completed_rounds
        self.summary = summary["text"]

    def prepare_tool_result_for_next_round(self, *, messages, tool_result, episode_id, round_index):
        # Keep the existing head-preserving admission behavior without running
        # compression during tool admission. Reserve a checkpoint decode too.
        from context_compression.grepseek_sglang_live import _raw_suffix_image_context_reserve_tokens
        limits = [self.server_info.get(key) for key in ("context_length", "max_total_num_tokens", "max_req_input_len")]
        ceiling = min(value for value in limits if type(value) is int and value > 0)
        def fits(candidate):
            ids, _, image_reserve = self._render(
                [*messages, {"role": "tool", "content": candidate.to_payload()}],
                episode_id=episode_id, raw_suffix_images=self.pending_images or (),
            )
            if self.pending_images is None:
                image_reserve = _raw_suffix_image_context_reserve_tokens(self.pending_image_count)
            return len(ids) + image_reserve + self.config.capacity_pressure_reserve_tokens + 2 < ceiling
        if fits(tool_result):
            return tool_result
        tokens = self.tokenizer.encode(tool_result.stdout, add_special_tokens=False)
        def clipped(n):
            text = self.tokenizer.decode(tokens[:n], skip_special_tokens=False) + "\n[Tool output truncated to fit the remaining context.]"
            return replace(tool_result, stdout=text, stderr="", truncated=True,
                information_lines=[line for line in text.splitlines() if line.strip()])
        low, high = 0, len(tokens)
        if not fits(clipped(0)):
            raise RuntimeError("native context is full before tool admission; no hidden compression was applied")
        while low < high:
            middle = (low + high + 1) // 2
            if fits(clipped(middle)):
                low = middle
            else:
                high = middle - 1
        return clipped(low)

    def configure_pending_tool_images(self, *, images):
        self.pending_images = tuple(images)
        self.pending_image_count = len(self.pending_images)

    def prepare_tool_result_for_native_compaction(self, *, messages, tool_result, episode_id, round_index):
        # This is not admission for another ordinary generation: Codex has
        # already requested compact. Preserve the complete source, including
        # pending observations, for the summary's own capacity check.
        if self.episode_id not in (None, episode_id):
            raise RuntimeError("native compact cannot switch episodes")
        return tool_result


class SummaryOnlyBackend(LiveR2EActBackend):
    """Reuse tool/recovery/trace plumbing, but never configure a live Compressor."""

    serving_concurrency = 1

    def _acquire_serving_lock(self, *, concurrent=False):
        self.config.process_lock.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock_handle = self.config.process_lock.open("a+")
        mode = fcntl.LOCK_SH if concurrent else fcntl.LOCK_EX
        while True:
            if self._closing.is_set():
                raise RuntimeError("summary-only backend shut down while waiting for SGLang")
            try:
                fcntl.flock(self._process_lock_handle.fileno(), mode | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                self._closing.wait(0.1)

    def configure(self, *, episode_id, tools):
        if self.config.compressor_enabled or not self.config.native_compaction_text_summary:
            raise RuntimeError("summary-only backend requires explicit no-Compressor config")
        if self._generator is not None:
            if self._episode_id != episode_id or self._tools != tuple(tools):
                raise RuntimeError("summary-only backend cannot switch sessions/tools")
            return
        self._episode_id, self._tools = episode_id, tuple(deepcopy(tools))
        try:
            from .discrete_concurrency import validate_count, assert_discrete_concurrency
            concurrency = validate_count(self.serving_concurrency)
            if concurrency > 1:
                from .residual_preflight import get_server_info
                assert_discrete_concurrency(get_server_info(self.config.sglang_url), concurrency)
            self._acquire_serving_lock(concurrent=concurrency > 1)
            host_rg = activate_pinned_host_rg()
            root = str(self.config.specforge_root)
            if root not in sys.path:
                sys.path.insert(0, root)
            self._contract = _InjectedCodexQwenContract(self._tools)
            self._contract.__enter__()
            from transformers import AutoTokenizer
            from .discrete_sglang_client import DiscreteSGLangClient
            from context_compression.grepseek_data import parse_tool_response, _canonical_tool_content
            from context_compression.grepseek_tool import GrepSeekCorpusToolRunner, ToolResult
            client = DiscreteSGLangClient(self.config.sglang_url,
                request_timeout=self.config.request_timeout_seconds, enable_request_cancellation=True)
            client.health()
            info = client.server_info()
            if concurrency > 1:
                assert_discrete_concurrency(info, concurrency)
            for key in ("context_length", "max_total_num_tokens", "max_req_input_len"):
                if type(info.get(key)) is not int or info[key] <= 0:
                    raise RuntimeError(f"SGLang is missing a valid {key}")
            model = Path(str(info.get("model_path", "")))
            model_config = json.loads((model / "config.json").read_text())
            if "Qwen3_5ForConditionalGeneration" not in model_config.get("architectures", []):
                raise RuntimeError("summary-only control requires the paired Qwen3.8 model")
            if self.config.vision_enabled and (not model_config.get("vision_config") or info.get("language_only") is True or info.get("enable_multimodal") is False):
                raise RuntimeError("summary-only control requires the paired vision capability")
            tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
            from .qwen_reasoning import bind_reasoning_effort
            tokenizer, reasoning_receipt = bind_reasoning_effort(
                tokenizer, getattr(self.config, "qwen_reasoning_effort", None))
            if reasoning_receipt is not None:
                from .storage import atomic_json
                atomic_json(self.artifact_dir / "qwen-reasoning-policy.json", reasoning_receipt)
                self._event(dict(event="qwen_reasoning_effort_configured", **reasoning_receipt))
            client.eos_token_id = tokenizer.eos_token_id
            cache = client.attest_sglang_cache_policy(info)
            # A reused hooked server supports the paired loop guard. On a pure
            # server, ordinary EOS/length recovery still uses the same gateway.
            # Hook-free SGLang returns JSON null, not necessarily an empty
            # list. Optional hook configs can also be null. Neither advertises
            # the Compressor-provided guard; ordinary EOS/length handling stays
            # enabled without adding custom processors to the plain server.
            guard = any(
                h["config"].get("generation_degeneration_guard_server_abi")
                for h in (info.get("forward_hooks") or [])
                if isinstance(h, dict) and isinstance(h.get("config"), dict)
            )
            if guard:
                client.configure_generation_degeneration_guard(True, check_interval_tokens=256,
                    minimum_periodic_span_tokens=1024, maximum_period_tokens=256,
                    minimum_repeat_count=8, max_retries=1, return_detected_generation=True)
            self._generator = SummaryOnlyGenerator(tokenizer, client, info, self.config, [t["name"] for t in self._tools])
            self._generator._initial_prompt_observer = getattr(self, "initial_prompt_observer", None)
            self._generator._runtime_event = self._event
            self._memory_runner = GrepSeekCorpusToolRunner(corpus_dir=self.config.corpus_dir,
                tokenizer=tokenizer, timeout=180, max_output_tokens=2048).restricted(("shell",), shell_corpus="react_memory")
            self._memory_runner.begin_episode(episode_id)
            self._tool_result_type = ToolResult
            self._parse_tool_response = parse_tool_response
            self._canonical_tool_content = _canonical_tool_content
            self._persist_memory()
            self._event(dict(event="gateway_backend_configured", episode_id=episode_id,
                tools=[t["name"] for t in self._tools], native_compaction=True,
                native_compaction_text_summary=True, compressor_enabled=False,
                compressor_checkpoint=None, compressor_calls=0, memory_slots=0, policy=POLICY,
                serving_concurrency=concurrency,
                serving_lock_mode="shared" if concurrency > 1 else "exclusive",
                model_path=str(model), host_rg=host_rg, sglang_cache_attestation=cache,
                generation_guard_enabled=bool(guard)))
        except BaseException:
            self.close()
            raise

    def summarize_native_compaction(self, **kwargs):
        return self._generator.summarize(**kwargs)

    def reserve_pending_tool_images(self, count):
        self._generator.pending_image_count = count
        self._generator.pending_images = None

    def configure_pending_tool_images(self, *, images):
        self._generator.configure_pending_tool_images(images=images)

    def prepare_tool_message_for_native_compaction(self, **kwargs):
        return self._prepare_tool_message(**kwargs, native_compaction_request=True)

    def stage_native_compaction(self, *, episode_id, target_completed_rounds, handle_digest, summary=None):
        if summary is None:
            raise RuntimeError("text-only compaction cannot install an empty checkpoint")
        self._generator.install(episode_id=episode_id, target_completed_rounds=target_completed_rounds, summary=summary)
        self._event(dict(event="codex_native_text_compaction_staged", episode_id=episode_id,
            target_completed_rounds=target_completed_rounds, handle_digest=handle_digest,
            compressor_calls=0, memory_slots=0, policy=POLICY))
