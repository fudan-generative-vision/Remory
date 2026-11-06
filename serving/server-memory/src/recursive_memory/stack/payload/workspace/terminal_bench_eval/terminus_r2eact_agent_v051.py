"""Terminus r^2eAct 0.5.1 performance-policy extension.

The released 0.4.12 base and 0.5.0 official harness remain byte-for-byte
frozen.  This additive module enables two independently versioned SpecForge
policies before the first model request:

* append-only, token-exact native-history rendering; and
* request-local soft-cache bypass with ordinary discrete radix-prefix reuse.

Both policies are fail-closed.  The second additionally requires a live
server capability receipt, so a v0.5.1 job cannot silently fall back to the
legacy global-flush path while claiming the optimized protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harbor.llms.base import BaseLLM

from terminal_bench_eval.terminus_r2eact_agent_v050 import (
    TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
    TERMINUS_R2EACT_V050_NAME,
    TerminusR2EActAgentV050,
    TerminusR2EActLLMV050,
    _terminal_system_prompt_v050,
)


TERMINUS_R2EACT_V051_VERSION = "0.5.1"
TERMINUS_R2EACT_V051_NAME = TERMINUS_R2EACT_V050_NAME
TERMINUS_R2EACT_V051_CONTEXT_ABI = (
    "qwen_round_recursive_event_durable_incremental_history_isolated_soft_cache_v1"
)
TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY = "append_only_token_exact_v1"
TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY = "isolated_soft_no_flush_discrete_prefix_v1"
TERMINUS_R2EACT_V051_CACHE_CAPABILITY = "request_cache_bypass_v1"


def _require_policy(value: str, *, name: str, expected: str) -> str:
    if value != expected:
        raise ValueError(f"{name} must equal {expected!r} for v0.5.1")
    return value


class TerminusR2EActLLMV051(TerminusR2EActLLMV050):
    """V050 semantics plus strictly attested incremental/cache policies."""

    def __init__(
        self,
        *,
        native_history_policy: str = TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY,
        sglang_cache_policy: str = TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY,
        **kwargs: Any,
    ) -> None:
        self.native_history_policy = _require_policy(
            native_history_policy,
            name="native_history_policy",
            expected=TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY,
        )
        self.sglang_cache_policy = _require_policy(
            sglang_cache_policy,
            name="sglang_cache_policy",
            expected=TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY,
        )
        self._v051_policy_attested = False
        self._v051_cache_policy_receipt: dict[str, Any] | None = None
        super().__init__(**kwargs)

    def _attest_v051_policies(self, bundle: Any) -> None:
        generator = self._underlying_generator(bundle.generator)
        client = getattr(generator, "client", None)
        configure_cache = getattr(client, "configure_sglang_cache_policy", None)
        attest_cache = getattr(client, "attest_sglang_cache_policy", None)
        server_info_method = getattr(client, "server_info", None)
        if not self._v051_policy_attested:
            configure_history = getattr(
                generator, "configure_native_history_policy", None
            )
            if not callable(configure_history):
                raise RuntimeError(
                    "SpecForge lacks the v0.5.1 native-history policy API"
                )
            configure_history(self.native_history_policy)
            observed_history = getattr(generator, "native_history_policy", None)
            if observed_history != self.native_history_policy:
                raise RuntimeError("SpecForge did not retain the native-history policy")
            if not callable(configure_cache):
                raise RuntimeError("SpecForge lacks the v0.5.1 cache-policy API")
            configure_cache(self.sglang_cache_policy)
        if not all(callable(value) for value in (attest_cache, server_info_method)):
            raise RuntimeError("SpecForge lacks the v0.5.1 cache-policy API")
        # Re-attest on every round boundary.  If SGLang restarts after the
        # first request, the next retry must fail closed instead of following
        # the client's rollout-safe legacy fallback.
        self._v051_policy_attested = False
        self._v051_cache_policy_receipt = None
        server_info = server_info_method()
        if not isinstance(server_info, Mapping):
            raise RuntimeError("SGLang server_info must be an object")
        receipt = attest_cache(server_info)
        expected_receipt = {
            "requested": self.sglang_cache_policy,
            "effective": self.sglang_cache_policy,
            "request_cache_bypass_capability": True,
        }
        if not isinstance(receipt, Mapping) or dict(receipt) != expected_receipt:
            raise RuntimeError("SGLang returned an invalid cache-policy receipt")
        capabilities = server_info.get("request_capabilities")
        if (
            not isinstance(capabilities, list)
            or TERMINUS_R2EACT_V051_CACHE_CAPABILITY not in capabilities
        ):
            raise RuntimeError("SGLang cache-bypass capability disappeared")
        self._v051_cache_policy_receipt = {
            **expected_receipt,
            "capability": TERMINUS_R2EACT_V051_CACHE_CAPABILITY,
            "max_running_requests": server_info.get("max_running_requests"),
            "soft_requests": "bypass_lookup_and_insert_ack_required",
            "discrete_requests": "default_radix_prefix_cache",
            "legacy_global_flush": False,
        }
        self._v051_policy_attested = True

    def _load_bundle(self) -> Any:
        bundle = super()._load_bundle()
        self._attest_v051_policies(bundle)
        return bundle

    def _response_extra(
        self,
        *,
        compression: Mapping[str, Any],
        round_index: int,
        tool_result_admission: Mapping[str, Any] | None,
        tool_observation_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = super()._response_extra(
            compression=compression,
            round_index=round_index,
            tool_result_admission=tool_result_admission,
            tool_observation_projection=tool_observation_projection,
        )
        value["r2eact_context_abi"] = TERMINUS_R2EACT_V051_CONTEXT_ABI
        value["r2eact_native_history_policy"] = self.native_history_policy
        value["r2eact_sglang_cache_policy"] = self.sglang_cache_policy
        return value

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "agent_version": TERMINUS_R2EACT_V051_VERSION,
                "context_abi": TERMINUS_R2EACT_V051_CONTEXT_ABI,
                "native_history_policy": {
                    "requested": self.native_history_policy,
                    "effective": (
                        self.native_history_policy
                        if self._v051_policy_attested
                        else None
                    ),
                    "attested": self._v051_policy_attested,
                },
                "sglang_cache_policy": (
                    {
                        "requested": self.sglang_cache_policy,
                        "effective": None,
                        "attested": False,
                    }
                    if self._v051_cache_policy_receipt is None
                    else {
                        **dict(self._v051_cache_policy_receipt),
                        "attested": True,
                    }
                ),
            }
        )
        return value


class TerminusR2EActAgentV051(TerminusR2EActAgentV050):
    """Additive v0.5.1 agent; frozen v0.5.0 remains unchanged."""

    @staticmethod
    def name() -> str:
        return TERMINUS_R2EACT_V051_NAME

    def version(self) -> str:
        return TERMINUS_R2EACT_V051_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        *,
        native_history_policy: str = TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY,
        sglang_cache_policy: str = TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY,
        **kwargs: Any,
    ) -> None:
        self._v051_native_history_policy = _require_policy(
            native_history_policy,
            name="native_history_policy",
            expected=TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY,
        )
        self._v051_sglang_cache_policy = _require_policy(
            sglang_cache_policy,
            name="sglang_cache_policy",
            expected=TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY,
        )
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        del args, kwargs
        bootstrap = dict(self._r2eact_bootstrap)
        bootstrap["system_prompt"] = _terminal_system_prompt_v050(
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
            }
        )
        self._r2eact_bootstrap = bootstrap
        return TerminusR2EActLLMV051(**bootstrap)


__all__ = [
    "TERMINUS_R2EACT_V051_CACHE_CAPABILITY",
    "TERMINUS_R2EACT_V051_CONTEXT_ABI",
    "TERMINUS_R2EACT_V051_NAME",
    "TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY",
    "TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY",
    "TERMINUS_R2EACT_V051_VERSION",
    "TerminusR2EActAgentV051",
    "TerminusR2EActLLMV051",
]
