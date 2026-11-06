"""Terminus r^2eAct 0.5.3 serving-accumulation canary.

This additive harness leaves the frozen 0.5.0/0.5.1/0.5.2 protocols intact.
It retains the v0.5.2 history, cache, and TP-GPU transport contracts while
pinning the Qwen3.8 serving-only 8,192-token accumulation policy for every
episode.  Configuration and the per-generation compression receipt are both
verified fail closed; the checkpoint-native 1,024-token policy can therefore
never be reported as a successful v0.5.3 rollout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from harbor.agents.terminus_2.terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
from harbor.llms.base import BaseLLM

from terminal_bench_eval.terminus_r2eact_agent_v050 import (
    TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
    TERMINUS_R2EACT_V050_NAME,
    _MemoryPersistingGenerator,
    _terminal_system_prompt_v050,
    _tool_result_with_stdout,
)
from terminal_bench_eval.terminus_r2eact_agent_v052 import (
    TerminusR2EActAgentV052,
    TerminusR2EActLLMV052,
)


TERMINUS_R2EACT_V053_VERSION = "0.5.3"
TERMINUS_R2EACT_V053_NAME = TERMINUS_R2EACT_V050_NAME
TERMINUS_R2EACT_V053_CONTEXT_ABI = (
    "qwen_round_recursive_event_durable_incremental_history_"
    "tp_gpu_chain_soft_prefix_batch_serving_8192_accumulation_"
    "oversized_answer_head1024_recovery_v2"
)
TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS = 12
TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY = (
    "qwen38_serving_8192_accumulation_canary_v1"
)
TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS = 8_192
TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY = (
    "postdecode_nonaction_answer_soft_threshold_head_projection_continue_v1"
)
TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS = 12_288
TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS = 1_024
TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY = (
    "terminus_json_valid_command_or_completion_preserved_v1"
)
TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY = (
    "terminus_json_noop_task_incomplete_v1"
)
TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY = (
    "next_round_controller_cue_prefix_v1"
)
_OVERFLOW_CUE_MARKER = "[r^2eAct controller output overflow]"
_TERMINUS_JSON_PARSER = TerminusJSONPlainParser()


def _controller_noop_response() -> str:
    """Return a deterministic, non-executing Terminus response.

    Harbor must never parse an incidental JSON object from the retained model
    prefix.  The real prefix is kept only in the recursive transcript; this
    separate response advances Terminus without executing a command or marking
    the task complete.
    """

    return json.dumps(
        {
            "analysis": (
                "The controller rejected an overlong response that contained "
                "no valid terminal action or completion signal."
            ),
            "plan": (
                "Continue from the retained prefix and current terminal state, "
                "then issue concise JSON."
            ),
            "commands": [],
            "task_complete": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _decode_projected_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if tokenizer is None:
        raise TypeError("oversized answer recovery requires a tokenizer")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("oversized answer recovery tokenizer lacks decode()")
    values = [int(value) for value in token_ids]
    try:
        return str(
            decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        try:
            return str(decode(values, skip_special_tokens=False))
        except TypeError:
            return str(decode(values))


def _terminus_response_content(generated: Any) -> str:
    parsed = getattr(generated, "parsed", None)
    answer = getattr(parsed, "answer_text", None)
    if isinstance(answer, str):
        return answer
    content = str(getattr(generated, "assistant_content", ""))
    close = content.find("</think>")
    return content[close + len("</think>") :].strip() if close >= 0 else content


def _has_valid_terminus_action(generated: Any) -> bool:
    """Protect real actions before considering a post-decode projection.

    SpecForge classifies Terminus's text JSON as an ``answer`` rather than a
    native tool call.  Reusing only BrowseComp's parsed-kind check would
    therefore discard legitimate shell commands.  Harbor's own parser is the
    authority for this harness.
    """

    parsed = getattr(generated, "parsed", None)
    if getattr(parsed, "kind", None) == "tool_call" and bool(
        getattr(parsed, "tool_schema_valid", False)
    ):
        return True
    tool_calls = getattr(parsed, "tool_calls", ())
    if isinstance(tool_calls, (list, tuple)) and any(
        bool(getattr(call, "tool_schema_valid", False)) for call in tool_calls
    ):
        return True
    result = _TERMINUS_JSON_PARSER.parse_response(
        _terminus_response_content(generated)
    )
    return not result.error and (bool(result.commands) or result.is_task_complete)


def _parsed_projection(parsed: Any, controller_response: str) -> Any:
    if not dataclasses.is_dataclass(parsed) or isinstance(parsed, type):
        raise TypeError("oversized answer recovery requires parsed dataclass output")
    available = {field.name for field in dataclasses.fields(parsed)}
    required = {"answer_text", "think_text"}
    if not required.issubset(available):
        raise TypeError("oversized answer recovery parsed output is incompatible")
    candidates: dict[str, Any] = {
        "kind": "answer",
        "eos_reached": True,
        "structure_valid": True,
        "complete_valid": True,
        "think_text": "",
        "think_nonempty": False,
        "tool_payload": None,
        "tool_json_valid": False,
        "tool_schema_valid": False,
        "tool_command": None,
        "answer_text": controller_response,
        "errors": ("controller_round_output_overflow",),
        "tool_schema_error": None,
        "tool_calls": (),
        "tool_preamble_text": "",
    }
    return dataclasses.replace(
        parsed,
        **{key: value for key, value in candidates.items() if key in available},
    )


def _recover_oversized_answer_generation(
    generated: Any,
    *,
    tokenizer: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Admit only the head of a runaway, non-action EOS answer.

    Physical token IDs stay on the returned object for accurate Harbor usage
    accounting.  The recursive transcript is bounded by replacing only
    ``assistant_content`` with the first 1,024 decoded body tokens.  Valid
    Terminus commands and completion signals are never intercepted.
    """

    parsed = getattr(generated, "parsed", None)
    token_ids_value = getattr(generated, "token_ids", ())
    if not isinstance(token_ids_value, (list, tuple)):
        return generated, None
    token_ids = tuple(int(value) for value in token_ids_value)
    if (
        not bool(getattr(generated, "finished_on_eos", False))
        or bool(getattr(generated, "hit_generation_cap", False))
        or not bool(getattr(parsed, "eos_reached", False))
        or len(token_ids) <= TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS
        or _has_valid_terminus_action(generated)
    ):
        return generated, None
    if not dataclasses.is_dataclass(generated) or isinstance(generated, type):
        raise TypeError("oversized answer recovery requires generation dataclass")

    raw_body_ids = token_ids[:-1]
    retained_body_ids = raw_body_ids[
        :TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS
    ]
    eos_token_id = token_ids[-1]
    projected_token_ids = tuple(retained_body_ids) + (eos_token_id,)
    decoded_body = _decode_projected_token_ids(tokenizer, retained_body_ids)
    original_assistant = str(getattr(generated, "assistant_content", ""))
    assistant_prefix = (
        "<think>\n"
        if original_assistant.startswith("<think>\n")
        and not decoded_body.startswith("<think>")
        else ""
    )
    projected_assistant = assistant_prefix + decoded_body
    controller_response = _controller_noop_response()
    recovery = {
        "policy": TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY,
        "trigger_kind": "eos_nonaction_answer_above_soft_threshold",
        "soft_threshold_tokens": (
            TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS
        ),
        "physical_raw_token_count": len(token_ids),
        "physical_raw_body_token_count": len(raw_body_ids),
        "context_admitted_body_token_count": len(retained_body_ids),
        "context_admitted_token_count_including_eos": len(projected_token_ids),
        "discarded_body_token_count": len(raw_body_ids) - len(retained_body_ids),
        "retained_side": "head",
        "raw_assistant_sha256": hashlib.sha256(
            original_assistant.encode("utf-8")
        ).hexdigest(),
        "raw_generation_persisted_in_private_events": True,
        "physical_token_ids_retained_for_usage_accounting": True,
        "valid_terminus_actions_preserved": True,
        "valid_action_policy": TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY,
        "harbor_response_policy": (
            TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY
        ),
        "next_round_observation_policy": (
            TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY
        ),
        "request_generation_limit_unchanged": True,
    }
    compression = getattr(generated, "compression", {})
    projected_compression = (
        dict(compression) if isinstance(compression, Mapping) else {}
    )
    projected_compression["oversized_answer_recovery"] = dict(recovery)
    projected = dataclasses.replace(
        generated,
        # Keep raw token_ids for physical accounting.  Only these decoded
        # fields enter the recursive transcript / Harbor response plane.
        generated_text=_decode_projected_token_ids(tokenizer, projected_token_ids),
        assistant_content=projected_assistant,
        parsed=_parsed_projection(parsed, controller_response),
        compression=projected_compression,
    )
    return projected, recovery


class _V053MemoryPersistingGenerator(_MemoryPersistingGenerator):
    """V0.5.0 durability plus v0.5.3-only output admission recovery."""

    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Any = None,
    ) -> Any:
        generated = super().generate_next(
            messages,
            episode_id=episode_id,
            round_index=round_index,
            event_sink=event_sink,
        )
        projected, recovery = _recover_oversized_answer_generation(
            generated,
            tokenizer=getattr(self._generator, "tokenizer", None),
        )
        if recovery is None:
            return projected
        raw_to_json = getattr(generated, "to_json", None)
        if not callable(raw_to_json):
            raise TypeError("oversized answer generation must provide to_json()")
        sink = event_sink if event_sink is not None else self._owner._event_sink
        sink(
            {
                "event": "round_output_overflow_raw_generation",
                "episode_id": str(episode_id),
                "round_index": int(round_index),
                "payload": {**dict(recovery), "raw_generation": raw_to_json()},
            }
        )
        sink(
            {
                "event": "round_output_overflow_recovered",
                "episode_id": str(episode_id),
                "round_index": int(round_index),
                "payload": dict(recovery),
            }
        )
        return projected

    def prepare_tool_result_for_next_round(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tool_result: Any,
        episode_id: str,
        round_index: int,
    ) -> Any:
        recovery = self._owner._v053_pending_overflow_for_round(round_index)
        if recovery is not None:
            original_stdout = getattr(tool_result, "stdout", None)
            if not isinstance(original_stdout, str):
                raise TypeError("overflow recovery observation must be a string")
            cue = self._owner._v053_overflow_cue(recovery)
            tool_result = _tool_result_with_stdout(
                tool_result,
                cue if not original_stdout else cue + "\n\n" + original_stdout,
            )
            self._owner._v053_note_overflow_cue_admission(round_index)
        value = super().prepare_tool_result_for_next_round(
            messages=messages,
            tool_result=tool_result,
            episode_id=episode_id,
            round_index=round_index,
        )
        if recovery is not None:
            prepared = value[0] if isinstance(value, tuple) else value
            prepared_stdout = getattr(prepared, "stdout", None)
            if not isinstance(prepared_stdout, str) or _OVERFLOW_CUE_MARKER not in (
                prepared_stdout
            ):
                raise RuntimeError(
                    "tool-result admission cropped the output-overflow cue"
                )
        return value


def _terminal_system_prompt_v053(working_memory_turns: int) -> str:
    """Render v0.5.3 with an explicitly unbounded total episode."""

    inherited = _terminal_system_prompt_v050(working_memory_turns)
    bounded = (
        "- There is no fixed round limit. The external trial timeout is the complete\n"
        "  budget."
    )
    unbounded = (
        "- There is no fixed round limit and no total episode time budget. Work until\n"
        "  the task is complete; once solved, return the final JSON promptly instead of\n"
        "  doing redundant terminal work."
    )
    if inherited.count(bounded) != 1:
        raise RuntimeError("inherited v0.5.0 timeout policy text changed")
    return inherited.replace(bounded, unbounded)


def _require_v053_working_memory(value: Any) -> int:
    if type(value) is not int or value != TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS:
        raise ValueError(
            "working_memory_turns must equal "
            f"{TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS} for v0.5.3"
        )
    return value


def _require_v053_canary(value: Any) -> bool:
    if type(value) is not bool or value is not True:
        raise ValueError(
            "qwen38_serving_8192_accumulation_canary must be true for v0.5.3"
        )
    return True


class TerminusR2EActLLMV053(TerminusR2EActLLMV052):
    """V0.5.2 plus a per-episode, receipt-verified serving canary."""

    def _expected_static_system_prompt(self) -> str:
        return _terminal_system_prompt_v053(self.working_memory_turns)

    def __init__(
        self,
        *,
        working_memory_turns: int = TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS,
        qwen38_serving_8192_accumulation_canary: bool = True,
        **kwargs: Any,
    ) -> None:
        self.qwen38_serving_8192_accumulation_canary = _require_v053_canary(
            qwen38_serving_8192_accumulation_canary
        )
        self._v053_episode_policy_configured = False
        self._v053_configured_generator: Any | None = None
        self._v053_observed_serving_eviction_policy: str | None = None
        self._v053_observed_accumulation_target_tokens: int | None = None
        self._v053_overflow_lock = threading.RLock()
        self._v053_pending_overflow: dict[str, Any] | None = None
        self._v053_overflow_counts = {
            "recovered_rounds": 0,
            "successor_cue_admissions": 0,
            "successor_cues_consumed": 0,
        }
        super().__init__(
            working_memory_turns=_require_v053_working_memory(working_memory_turns),
            **kwargs,
        )

    def begin_run(self) -> None:
        super().begin_run()
        self._v053_episode_policy_configured = False
        self._v053_configured_generator = None
        self._v053_observed_serving_eviction_policy = None
        self._v053_observed_accumulation_target_tokens = None
        with self._v053_overflow_lock:
            self._v053_pending_overflow = None
            for key in self._v053_overflow_counts:
                self._v053_overflow_counts[key] = 0

    @staticmethod
    def _v053_overflow_cue(recovery: Mapping[str, Any]) -> str:
        raw = int(recovery["physical_raw_token_count"])
        retained = int(recovery["context_admitted_body_token_count"])
        return (
            f"{_OVERFLOW_CUE_MARKER}\n"
            f"The previous round decoded {raw} tokens without a valid terminal "
            f"action or completion signal. Only its first {retained} body tokens "
            "were admitted to context; no command or completion from that round "
            "was accepted. Continue from the current terminal state and respond "
            "with concise valid JSON."
        )

    def _v053_pending_overflow_for_round(
        self, round_index: int
    ) -> dict[str, Any] | None:
        with self._v053_overflow_lock:
            pending = self._v053_pending_overflow
            if pending is None:
                return None
            if pending.get("source_round_index") != int(round_index):
                raise RuntimeError(
                    "pending output-overflow cue does not match this observation"
                )
            recovery = pending.get("recovery")
            if not isinstance(recovery, Mapping):
                raise RuntimeError("pending output-overflow receipt is invalid")
            return dict(recovery)

    def _v053_note_overflow_cue_admission(self, round_index: int) -> None:
        with self._v053_overflow_lock:
            pending = self._v053_pending_overflow
            if pending is None or pending.get("source_round_index") != int(round_index):
                raise RuntimeError("output-overflow cue admission has no owner")
            self._v053_overflow_counts["successor_cue_admissions"] += 1

    def _v053_validate_commit_transition(
        self,
        *,
        round_index: int,
        recovery: Mapping[str, Any] | None,
    ) -> None:
        with self._v053_overflow_lock:
            pending = self._v053_pending_overflow
            if pending is not None and pending.get("source_round_index") != (
                int(round_index) - 1
            ):
                raise RuntimeError(
                    "stale output-overflow cue survived past its successor round"
                )
            if recovery is not None and recovery.get("policy") != (
                TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY
            ):
                raise RuntimeError("generation returned an unknown overflow policy")

    def _commit_generation(self, pending: Any) -> Any:
        raw_recovery = pending.compression.get("oversized_answer_recovery")
        recovery = dict(raw_recovery) if isinstance(raw_recovery, Mapping) else None
        self._v053_validate_commit_transition(
            round_index=int(pending.round_index), recovery=recovery
        )
        response = super()._commit_generation(pending)
        with self._v053_overflow_lock:
            previous = self._v053_pending_overflow
            if previous is not None:
                self._v053_pending_overflow = None
                self._v053_overflow_counts["successor_cues_consumed"] += 1
            if recovery is not None:
                self._v053_pending_overflow = {
                    "source_round_index": int(pending.round_index),
                    "recovery": dict(recovery),
                }
                self._v053_overflow_counts["recovered_rounds"] += 1
        return response

    def _configure_v053_serving_eviction_policy(self, bundle: Any) -> None:
        generator = self._underlying_generator(bundle.generator)
        configure = getattr(generator, "configure_serving_eviction_policy", None)
        if not callable(configure):
            raise RuntimeError("SpecForge lacks the v0.5.3 serving-eviction policy API")
        if (
            not self._v053_episode_policy_configured
            or self._v053_configured_generator is not generator
        ):
            configure(TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY)
            self._v053_episode_policy_configured = True
            self._v053_configured_generator = generator
        observed = getattr(generator, "serving_eviction_policy", None)
        if observed != TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY:
            self._v053_episode_policy_configured = False
            raise RuntimeError(
                "SpecForge did not retain the v0.5.3 serving-eviction policy"
            )

    def _load_bundle(self) -> Any:
        bundle = super()._load_bundle()
        if not isinstance(bundle.generator, _V053MemoryPersistingGenerator):
            if not isinstance(bundle.generator, _MemoryPersistingGenerator):
                raise RuntimeError(
                    "v0.5.3 expected the durable generator wrapper before "
                    "output-recovery installation"
                )
            wrapped = _V053MemoryPersistingGenerator(
                bundle.generator.wrapped_generator,
                self,
            )
            if dataclasses.is_dataclass(bundle):
                bundle = dataclasses.replace(bundle, generator=wrapped)
            else:
                setattr(bundle, "generator", wrapped)
            self._bundle = bundle
        self._configure_v053_serving_eviction_policy(bundle)
        return bundle

    def _attest_v053_compression_receipt(self, compression: Mapping[str, Any]) -> None:
        observed_policy = compression.get("serving_eviction_policy")
        observed_target = compression.get("serving_eviction_accumulation_target_tokens")
        if observed_policy != TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY:
            raise RuntimeError(
                "v0.5.3 generation did not use the pinned serving-eviction policy"
            )
        if (
            type(observed_target) is not int
            or observed_target
            != TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS
        ):
            raise RuntimeError(
                "v0.5.3 generation did not retain the 8,192-token accumulation target"
            )
        self._v053_observed_serving_eviction_policy = observed_policy
        self._v053_observed_accumulation_target_tokens = observed_target

    def _response_extra(
        self,
        *,
        compression: Mapping[str, Any],
        round_index: int,
        tool_result_admission: Mapping[str, Any] | None,
        tool_observation_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._attest_v053_compression_receipt(compression)
        value = super()._response_extra(
            compression=compression,
            round_index=round_index,
            tool_result_admission=tool_result_admission,
            tool_observation_projection=tool_observation_projection,
        )
        value.update(
            {
                "r2eact_context_abi": TERMINUS_R2EACT_V053_CONTEXT_ABI,
                "r2eact_serving_eviction_policy": (
                    TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY
                ),
                "r2eact_serving_eviction_accumulation_target_tokens": (
                    TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS
                ),
                "r2eact_oversized_answer_recovery": (
                    dict(compression["oversized_answer_recovery"])
                    if isinstance(
                        compression.get("oversized_answer_recovery"), Mapping
                    )
                    else None
                ),
            }
        )
        return value

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        receipt_attested = (
            self._v053_observed_serving_eviction_policy
            == TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY
            and self._v053_observed_accumulation_target_tokens
            == TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS
        )
        with self._v053_overflow_lock:
            pending_successor_cue = self._v053_pending_overflow is not None
            overflow_counts = dict(self._v053_overflow_counts)
        value.update(
            {
                "agent_version": TERMINUS_R2EACT_V053_VERSION,
                "context_abi": TERMINUS_R2EACT_V053_CONTEXT_ABI,
                "qwen38_serving_8192_accumulation_canary": True,
                "serving_eviction_policy": {
                    "requested": TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY,
                    "configured": self._v053_episode_policy_configured,
                    "effective": self._v053_observed_serving_eviction_policy,
                    "accumulation_target_tokens": (
                        TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS
                    ),
                    "receipt_attested": receipt_attested,
                },
                "oversized_answer_recovery": {
                    "enabled": True,
                    "policy": (
                        TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY
                    ),
                    "soft_threshold_tokens": (
                        TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS
                    ),
                    "retained_body_tokens": (
                        TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS
                    ),
                    "retained_side": "head",
                    "scope": "finished_eos_nonaction_answer_only",
                    "valid_action_policy": (
                        TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY
                    ),
                    "harbor_response_policy": (
                        TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY
                    ),
                    "next_round_observation_policy": (
                        TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY
                    ),
                    "physical_token_accounting": "raw_generation",
                    "context_admission_accounting": "projected_head_plus_eos",
                    "pending_successor_cue": pending_successor_cue,
                    **overflow_counts,
                },
            }
        )
        return value


class TerminusR2EActAgentV053(TerminusR2EActAgentV052):
    """Additive WM12/8,192-token canary; earlier harnesses stay frozen."""

    @staticmethod
    def name() -> str:
        return TERMINUS_R2EACT_V053_NAME

    def version(self) -> str:
        return TERMINUS_R2EACT_V053_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        *,
        working_memory_turns: int = TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS,
        qwen38_serving_8192_accumulation_canary: bool = True,
        **kwargs: Any,
    ) -> None:
        self._v053_working_memory_turns = _require_v053_working_memory(
            working_memory_turns
        )
        self._v053_qwen38_serving_8192_accumulation_canary = _require_v053_canary(
            qwen38_serving_8192_accumulation_canary
        )
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            working_memory_turns=self._v053_working_memory_turns,
            **kwargs,
        )

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        del args, kwargs
        bootstrap = dict(self._r2eact_bootstrap)
        bootstrap["system_prompt"] = _terminal_system_prompt_v053(
            int(bootstrap["working_memory_turns"])
        )
        bootstrap.update(
            {
                "expected_served_model": self._v050_expected_served_model,
                "expected_model_path": self._v050_expected_model_path,
                "expected_target_family": self._v050_expected_target_family,
                "expected_checkpoint_schema": self._v050_expected_checkpoint_schema,
                "sync_token_ids_across_tp": self._v050_sync_token_ids_across_tp,
                "native_history_policy": self._v051_native_history_policy,
                "sglang_cache_policy": self._v051_sglang_cache_policy,
                "memory_transport_policy": self._v052_memory_transport_policy,
                "qwen38_serving_8192_accumulation_canary": (
                    self._v053_qwen38_serving_8192_accumulation_canary
                ),
            }
        )
        self._r2eact_bootstrap = bootstrap
        return TerminusR2EActLLMV053(**bootstrap)


__all__ = [
    "TERMINUS_R2EACT_V053_CONTEXT_ABI",
    "TERMINUS_R2EACT_V053_NAME",
    "TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY",
    "TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY",
    "TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS",
    "TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS",
    "TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY",
    "TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY",
    "TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS",
    "TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY",
    "TERMINUS_R2EACT_V053_VERSION",
    "TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS",
    "_terminal_system_prompt_v053",
    "TerminusR2EActAgentV053",
    "TerminusR2EActLLMV053",
]
