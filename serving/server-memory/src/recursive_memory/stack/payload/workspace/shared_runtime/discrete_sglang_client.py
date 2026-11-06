"""Ordinary token/vision generation, without the soft-memory cache contract.

Imported after the runner selects its SpecForge checkout. Reuse its generation
parsing, cancellation and optional loop guard, but prohibit Compressor traffic.
"""
from collections.abc import Mapping

from context_compression.grepseek_sglang_client import GrepSeekSGLangClient


DISCRETE_CACHE_POLICY = "native_discrete_prefix_v1"


class DiscreteSGLangClient(GrepSeekSGLangClient):
    def attest_sglang_cache_policy(self, server_info=None):
        if server_info is not None and not isinstance(server_info, Mapping):
            raise TypeError("server_info must be a mapping")
        # This is a client-side wire policy, not a claim that the server has
        # attested soft-memory isolation. No special server capability is used.
        return {
            "requested": DISCRETE_CACHE_POLICY,
            "effective": DISCRETE_CACHE_POLICY,
            "request_cache_bypass_capability": False,
            "compressor_enabled": False,
        }

    def _isolated_cache_policy_active(self):
        # In the shared decoder, this branch emits ordinary discrete requests
        # without a unique key or global flush. Its soft-memory branch is NOT
        # safe on a plain server, so both entry and wire guards below reject it.
        return True

    def _generate(self, input_ids, *, custom_params, memory_transport_cache_policy=None, **kwargs):
        if custom_params is not None or memory_transport_cache_policy is not None:
            raise RuntimeError("discrete-only SGLang client forbids soft-memory generation")
        return super()._generate(input_ids, custom_params=None, **kwargs)

    def _post(self, route, payload, *, request_timeout=None):
        # Also catch inherited encode/chain calls before they reach the server.
        if route != "/generate":
            raise RuntimeError("discrete-only SGLang client forbids Compressor operations")
        allowed = {"input_ids", "sampling_params", "image_data", "rid", "custom_logit_processor"}
        if set(payload) - allowed:
            raise RuntimeError("discrete-only SGLang client forbids soft-memory wire fields")
        sampling = payload.get("sampling_params", {})
        custom = sampling.get("custom_params")
        guard = self._generation_guard_custom_params()
        if custom is not None and (guard is None or custom != guard):
            raise RuntimeError("discrete-only SGLang client forbids Compressor custom params")
        return super()._post(route, payload, request_timeout=request_timeout)

    def flush_cache(self):
        raise RuntimeError("discrete-only generation must not flush the shared SGLang cache")


# Keep the discrete wire validator BEFORE the streaming transport in the MRO.
# Streaming must never bypass the prohibition on encoder and soft-memory calls.
from .sglang_streaming import GenerationStreamingMixin


class _StreamingDiscreteTransport(GenerationStreamingMixin, GrepSeekSGLangClient):
    pass


class StreamingDiscreteSGLangClient(DiscreteSGLangClient, _StreamingDiscreteTransport):
    pass
