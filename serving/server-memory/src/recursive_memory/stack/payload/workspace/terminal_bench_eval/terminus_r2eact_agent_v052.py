"""Terminus r^2eAct 0.5.2 GPU-memory performance extension.

This additive harness keeps the frozen 0.5.0/0.5.1 protocols intact while
requiring one coherent SpecForge/SGLang transport contract:

* append-only, token-exact native-history rendering;
* request-local cache bypass for compressor encodes and ordinary discrete
  radix-prefix reuse;
* TP-rank-local GPU memory handles;
* scheduler-serialized recursive compression chains; and
* identity-bound soft-memory prefix reuse with recover batching.

Every round re-attests the live server epoch and hook ABI.  A restart or a
partial deployment therefore fails closed instead of silently returning to
CPU tensor transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harbor.llms.base import BaseLLM

from terminal_bench_eval.terminus_r2eact_agent_v050 import (
    TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
    TERMINUS_R2EACT_V050_NAME,
    _terminal_system_prompt_v050,
)
from terminal_bench_eval.terminus_r2eact_agent_v051 import (
    TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY,
    TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY,
    TerminusR2EActAgentV051,
    TerminusR2EActLLMV051,
    _require_policy,
)


TERMINUS_R2EACT_V052_VERSION = "0.5.2"
TERMINUS_R2EACT_V052_NAME = TERMINUS_R2EACT_V050_NAME
TERMINUS_R2EACT_V052_CONTEXT_ABI = (
    "qwen_round_recursive_event_durable_incremental_history_"
    "tp_gpu_chain_soft_prefix_batch_v1"
)
TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY = (
    TERMINUS_R2EACT_V051_NATIVE_HISTORY_POLICY
)
TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY = (
    TERMINUS_R2EACT_V051_SGLANG_CACHE_POLICY
)
TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY = (
    "tp_gpu_handle_scheduler_chain_v1"
)
TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES = frozenset(
    {
        "request_cache_bypass_v1",
        "soft_memory_prefix_reuse_batch_v1",
        "specforge_recursive_compression_chain_v1",
        "specforge_tp_gpu_memory_handle_v1",
    }
)


class TerminusR2EActLLMV052(TerminusR2EActLLMV051):
    """V0.5.1 history/cache semantics plus strict TP-GPU memory transport."""

    def __init__(
        self,
        *,
        memory_transport_policy: str = (
            TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY
        ),
        **kwargs: Any,
    ) -> None:
        self.memory_transport_policy = _require_policy(
            memory_transport_policy,
            name="memory_transport_policy",
            expected=TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
        )
        self._v052_policy_configured = False
        self._v052_memory_transport_receipt: dict[str, Any] | None = None
        super().__init__(**kwargs)

    def _attest_v051_policies(self, bundle: Any) -> None:
        """Replace the v0.5.1 batch-one attestation with the v0.5.2 ABI."""

        generator = self._underlying_generator(bundle.generator)
        client = getattr(generator, "client", None)
        configure_history = getattr(
            generator, "configure_native_history_policy", None
        )
        configure_cache = getattr(client, "configure_sglang_cache_policy", None)
        configure_transport = getattr(
            generator, "configure_memory_transport_policy", None
        )
        attest_cache = getattr(client, "attest_sglang_cache_policy", None)
        attest_transport = getattr(
            generator, "attest_memory_transport_policy", None
        )
        server_info_method = getattr(client, "server_info", None)
        required_methods = (
            configure_history,
            configure_cache,
            configure_transport,
            attest_cache,
            attest_transport,
            server_info_method,
        )
        if not all(callable(method) for method in required_methods):
            raise RuntimeError("SpecForge lacks the v0.5.2 policy API")

        if not self._v052_policy_configured:
            configure_history(self.native_history_policy)
            if (
                getattr(generator, "native_history_policy", None)
                != self.native_history_policy
            ):
                raise RuntimeError(
                    "SpecForge did not retain the native-history policy"
                )
            configure_cache(self.sglang_cache_policy)
            configure_transport(self.memory_transport_policy)
            self._v052_policy_configured = True

        # The same HTTP snapshot is used for both receipts.  Repeating this at
        # every load/round boundary detects a restart-scoped server epoch.
        previous_transport_receipt = self._v052_memory_transport_receipt
        self._v051_policy_attested = False
        self._v051_cache_policy_receipt = None
        self._v052_memory_transport_receipt = None
        server_info = server_info_method()
        if not isinstance(server_info, Mapping):
            raise RuntimeError("SGLang server_info must be an object")

        capabilities = server_info.get("request_capabilities")
        if (
            not isinstance(capabilities, list)
            or any(not isinstance(value, str) for value in capabilities)
            or not TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES.issubset(
                capabilities
            )
        ):
            raise RuntimeError("SGLang v0.5.2 capabilities are incomplete")
        max_running = server_info.get("max_running_requests")
        if (
            isinstance(max_running, bool)
            or not isinstance(max_running, int)
            or max_running < 2
        ):
            raise RuntimeError("v0.5.2 requires max_running_requests >= 2")

        cache_receipt = attest_cache(server_info)
        expected_cache_receipt = {
            "requested": self.sglang_cache_policy,
            "effective": self.sglang_cache_policy,
            "request_cache_bypass_capability": True,
        }
        if (
            not isinstance(cache_receipt, Mapping)
            or dict(cache_receipt) != expected_cache_receipt
        ):
            raise RuntimeError("SGLang returned an invalid cache-policy receipt")

        transport_receipt = attest_transport(server_info)
        if not isinstance(transport_receipt, Mapping):
            raise RuntimeError("SGLang returned no memory-transport receipt")
        required = transport_receipt.get("required_capabilities")
        observed = transport_receipt.get("observed_capabilities")
        handle_info = server_info.get("specforge_gpu_memory_handle")
        if (
            transport_receipt.get("requested") != self.memory_transport_policy
            or transport_receipt.get("effective") != self.memory_transport_policy
            or not isinstance(required, list)
            or set(required) != TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES
            or not isinstance(observed, list)
            or set(observed) != set(capabilities)
            or transport_receipt.get("max_running_requests") != max_running
            or not isinstance(handle_info, Mapping)
            or transport_receipt.get("server_epoch")
            != handle_info.get("server_epoch")
            or transport_receipt.get("hook_abi_digest")
            != handle_info.get("hook_abi_digest")
        ):
            raise RuntimeError(
                "SGLang returned an invalid memory-transport receipt"
            )
        if previous_transport_receipt is not None and (
            previous_transport_receipt.get("server_epoch")
            != transport_receipt.get("server_epoch")
            or previous_transport_receipt.get("hook_abi_digest")
            != transport_receipt.get("hook_abi_digest")
        ):
            raise RuntimeError(
                "SGLang memory-transport epoch/ABI changed during this run"
            )

        self._v051_cache_policy_receipt = {
            **expected_cache_receipt,
            "capability": "request_cache_bypass_v1",
            "max_running_requests": max_running,
            "soft_requests": "identity_bound_prefix_reuse_batch",
            "discrete_requests": "default_radix_prefix_cache",
            "legacy_global_flush": False,
        }
        self._v052_memory_transport_receipt = dict(transport_receipt)
        self._v051_policy_attested = True

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
        value["r2eact_context_abi"] = TERMINUS_R2EACT_V052_CONTEXT_ABI
        value["r2eact_memory_transport_policy"] = self.memory_transport_policy
        return value

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "agent_version": TERMINUS_R2EACT_V052_VERSION,
                "context_abi": TERMINUS_R2EACT_V052_CONTEXT_ABI,
                "memory_transport_policy": (
                    {
                        "requested": self.memory_transport_policy,
                        "effective": None,
                        "attested": False,
                    }
                    if self._v052_memory_transport_receipt is None
                    else {
                        **dict(self._v052_memory_transport_receipt),
                        "attested": True,
                    }
                ),
            }
        )
        return value


class TerminusR2EActAgentV052(TerminusR2EActAgentV051):
    """Additive v0.5.2 agent; earlier official harnesses remain frozen."""

    @staticmethod
    def name() -> str:
        return TERMINUS_R2EACT_V052_NAME

    def version(self) -> str:
        return TERMINUS_R2EACT_V052_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        *,
        memory_transport_policy: str = (
            TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY
        ),
        **kwargs: Any,
    ) -> None:
        self._v052_memory_transport_policy = _require_policy(
            memory_transport_policy,
            name="memory_transport_policy",
            expected=TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
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
                "memory_transport_policy": self._v052_memory_transport_policy,
            }
        )
        self._r2eact_bootstrap = bootstrap
        return TerminusR2EActLLMV052(**bootstrap)


__all__ = [
    "TERMINUS_R2EACT_V052_CONTEXT_ABI",
    "TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY",
    "TERMINUS_R2EACT_V052_NAME",
    "TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY",
    "TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES",
    "TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY",
    "TERMINUS_R2EACT_V052_VERSION",
    "TerminusR2EActAgentV052",
    "TerminusR2EActLLMV052",
]
