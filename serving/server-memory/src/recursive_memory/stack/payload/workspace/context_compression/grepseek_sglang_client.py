"""Small fail-closed client for the GrepSeek SGLang compressor protocol.

The source request returns only compressed memory slots through SGLang's
``return_hidden_states`` channel.  A second, fresh request carries those slots
as sparse embedding overrides for the recovery prompt.  Raw target features
never cross the worker boundary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from context_compression.r2eact_compression_chain import (
    GPU_MEMORY_HANDLE_META_KEY,
    GPU_MEMORY_HANDLE_RELEASE_ROUTE,
    GPU_MEMORY_HANDLE_RELEASE_SCHEMA,
    MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1,
    MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1,
    RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY,
    RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY,
    RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA,
    RECURSIVE_COMPRESSION_CHAIN_ROUTE,
    RECURSIVE_COMPRESSION_POLICIES,
    SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY,
    SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY,
    MemoryTransportAttestation,
    RecursiveCompressionChainRequest,
    RecursiveCompressionChainResult,
    TPGPUMemoryHandle,
    attest_memory_transport_server_info,
    parse_recursive_compression_chain_response,
)


GENERATION_DEGENERATION_GUARD_POLICY = (
    "exact_periodic_tail_force_eos_same_sampling_retry_v1"
)
DEFAULT_GENERATION_GUARD_CHECK_INTERVAL_TOKENS = 256
DEFAULT_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS = 1_024
DEFAULT_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS = 256
DEFAULT_GENERATION_GUARD_MINIMUM_REPEAT_COUNT = 8
DEFAULT_GENERATION_GUARD_MAX_RETRIES = 1

SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1 = "idle_flush_unique_key_v1"
SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1 = (
    "isolated_soft_no_flush_discrete_prefix_v1"
)
SGLANG_REQUEST_CACHE_BYPASS_CAPABILITY = "request_cache_bypass_v1"
SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY = "bypass"
_SGLANG_CACHE_POLICIES = frozenset(
    {
        SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1,
        SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1,
    }
)

_RECURSIVE_COMPRESSION_CHAIN_LIMITS_SCHEMA = (
    "specforge_recursive_compression_chain_limits_v1"
)
_RECURSIVE_COMPRESSION_CHAIN_LIMIT_FIELDS = frozenset(
    {
        "schema",
        "execution_timeout_seconds",
        "scheduler_request_timeout_seconds",
        "handle_release_timeout_seconds",
        "abort_ack_timeout_seconds",
        "orphan_grace_seconds",
        "response_timeout_upper_bound_seconds",
    }
)
_RETRYABLE_RECURSIVE_CHAIN_SERVER_ERRORS = (
    "recursive compression chain exceeded its server execution deadline",
    "recursive compression chain scheduler request timed out",
    "recursive compression chain handle release timed out",
)
_LEGACY_RECURSIVE_CHAIN_RECONNECT_GRACE_SECONDS = 90.0

logger = logging.getLogger(__name__)


def _integer_list(values: Sequence[int], *, name: str) -> list[int]:
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


_GENERATION_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "sampling_seed",
        "json_schema",
    }
)


def _generation_sampling_overrides(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the small sampling surface allowed by the live harness."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("sampling_overrides must be a mapping or None")
    unknown = set(value) - _GENERATION_SAMPLING_KEYS
    if unknown:
        raise ValueError(
            "unsupported generation sampling overrides: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    result = dict(value)
    for name in (
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    ):
        if name not in result:
            continue
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"{name} must be finite")
        result[name] = item
    if "temperature" in result and result["temperature"] < 0.0:
        raise ValueError("temperature must be non-negative")
    if "top_p" in result and not 0.0 < result["top_p"] <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if "min_p" in result and not 0.0 <= result["min_p"] <= 1.0:
        raise ValueError("min_p must be in [0, 1]")
    if "presence_penalty" in result and not -2.0 <= result["presence_penalty"] <= 2.0:
        raise ValueError("presence_penalty must be in [-2, 2]")
    if (
        "repetition_penalty" in result
        and not 0.0 <= result["repetition_penalty"] <= 2.0
    ):
        raise ValueError("repetition_penalty must be in [0, 2]")
    if "top_k" in result:
        top_k = result["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
    if "sampling_seed" in result:
        seed = result["sampling_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampling_seed must be a non-negative integer")
    if "json_schema" in result:
        schema = result["json_schema"]
        if not isinstance(schema, str):
            raise TypeError("json_schema must be a serialized JSON object")
        if not isinstance(json.loads(schema), dict):
            raise ValueError("json_schema must describe a JSON object")
    return result


@dataclass(frozen=True)
class SGLangGeneration:
    token_ids: tuple[int, ...]
    finished_on_eos: bool
    finish_reason: Any
    meta_info: Mapping[str, Any]


@dataclass(frozen=True)
class ExactPeriodicTailDetection:
    period_tokens: int
    repeat_count: int
    periodic_suffix_tokens: int
    loop_onset_token: int
    observed_output_tokens: int

    def to_json(self) -> dict[str, int]:
        return {
            "period_tokens": int(self.period_tokens),
            "repeat_count": int(self.repeat_count),
            "periodic_suffix_tokens": int(self.periodic_suffix_tokens),
            "loop_onset_token": int(self.loop_onset_token),
            "observed_output_tokens": int(self.observed_output_tokens),
        }


class DegenerateGenerationError(RuntimeError):
    """A high-confidence exact periodic generation exhausted safe retries."""

    def __init__(self, telemetry: Mapping[str, Any]) -> None:
        self.telemetry = dict(telemetry)
        detections = self.telemetry.get("detections", [])
        last = detections[-1] if isinstance(detections, list) and detections else {}
        super().__init__(
            "SGLang generation entered an exact periodic token loop "
            f"(period={last.get('period_tokens')}, "
            f"onset={last.get('loop_onset_token')}, "
            f"observed={last.get('observed_output_tokens')}); "
            "same-sampling retry budget exhausted"
        )


def detect_exact_periodic_tail(
    token_ids: Sequence[int],
    *,
    minimum_periodic_span_tokens: int = (
        DEFAULT_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
    ),
    maximum_period_tokens: int = DEFAULT_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS,
    minimum_repeat_count: int = DEFAULT_GENERATION_GUARD_MINIMUM_REPEAT_COUNT,
) -> ExactPeriodicTailDetection | None:
    """Return a conservative exact-period detector for the current suffix."""

    values = [int(value) for value in token_ids]
    minimum_span = int(minimum_periodic_span_tokens)
    maximum_period = int(maximum_period_tokens)
    minimum_repeats = int(minimum_repeat_count)
    if minimum_span <= 0 or maximum_period <= 0 or minimum_repeats < 2:
        raise ValueError("invalid exact-periodic-tail detector configuration")
    token_count = len(values)
    if token_count < minimum_span:
        return None
    largest_period = min(maximum_period, token_count // minimum_repeats)
    for period in range(1, largest_period + 1):
        required_repeats = max(
            minimum_repeats,
            (minimum_span + period - 1) // period,
        )
        if token_count < period * required_repeats:
            continue
        pattern = values[token_count - period : token_count]
        repeat_count = 1
        while token_count >= period * (repeat_count + 1):
            end = token_count - period * repeat_count
            start = end - period
            if values[start:end] != pattern:
                break
            repeat_count += 1
        if repeat_count < required_repeats:
            continue
        periodic_tokens = repeat_count * period
        return ExactPeriodicTailDetection(
            period_tokens=period,
            repeat_count=repeat_count,
            periodic_suffix_tokens=periodic_tokens,
            loop_onset_token=token_count - periodic_tokens,
            observed_output_tokens=token_count,
        )
    return None


def _finish_reason_kind(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Mapping):
        for key in ("type", "reason", "finish_reason"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()
    return ""


class GrepSeekSGLangClient:
    """Serialized batch-one access to the SGLang compressor hook."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 600.0,
        api_key: str | None = None,
        eos_token_id: int = 248046,
        enable_request_cancellation: bool = False,
        abort_timeout: float = 5.0,
        sglang_cache_policy: str = SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1,
    ) -> None:
        normalized = str(base_url).rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        if not math.isfinite(float(request_timeout)) or float(request_timeout) <= 0:
            raise ValueError("request_timeout must be positive and finite")
        if not isinstance(enable_request_cancellation, bool):
            raise TypeError("enable_request_cancellation must be boolean")
        if (
            isinstance(abort_timeout, bool)
            or not isinstance(abort_timeout, (int, float))
            or not math.isfinite(float(abort_timeout))
            or float(abort_timeout) <= 0
        ):
            raise ValueError("abort_timeout must be a positive finite number")
        if sglang_cache_policy not in _SGLANG_CACHE_POLICIES:
            raise ValueError(
                "unsupported sglang_cache_policy: " f"{sglang_cache_policy!r}"
            )
        self.base_url = normalized
        self._api_key = api_key
        self.request_timeout = float(request_timeout)
        self.eos_token_id = int(eos_token_id)
        self.abort_timeout = float(abort_timeout)
        self._custom_processor: str | None = None
        self._request_lock = threading.Lock()
        self._cache_capability_lock = threading.Lock()
        self._sglang_cache_policy = sglang_cache_policy
        self._cache_bypass_capability: bool | None = None
        self._memory_transport_lock = threading.Lock()
        self._memory_transport_policy = MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1
        self._memory_transport_attestation: MemoryTransportAttestation | None = None
        self._recursive_chain_response_timeout_upper_bound: float | None = None
        self._request_cancellation_lock = threading.Lock()
        self._request_cancellation_enabled = enable_request_cancellation
        self._active_request_ids: set[str] = set()
        self._abort_retry_state: dict[str, tuple[int, float]] = {}
        self._abort_requests_in_flight: set[str] = set()
        self._request_id_namespace: str | None = None
        self._request_id_sequence = 0
        self._generation_guard_enabled = False
        self._generation_guard_check_interval_tokens = (
            DEFAULT_GENERATION_GUARD_CHECK_INTERVAL_TOKENS
        )
        self._generation_guard_minimum_periodic_span_tokens = (
            DEFAULT_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
        )
        self._generation_guard_maximum_period_tokens = (
            DEFAULT_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS
        )
        self._generation_guard_minimum_repeat_count = (
            DEFAULT_GENERATION_GUARD_MINIMUM_REPEAT_COUNT
        )
        self._generation_guard_max_retries = DEFAULT_GENERATION_GUARD_MAX_RETRIES
        self._generation_guard_return_detected_generation = False
        lock_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        self._process_lock_path = Path(
            f"/tmp/specforge-sglang-{os.getuid()}-{lock_digest}.lock"
        )

    def configure_sglang_cache_policy(self, policy: str) -> None:
        """Select a versioned cache policy for requests started afterwards.

        The isolated policy is doubly gated: selecting it here is only the
        client opt-in, and the server must separately advertise the exact
        request-cache-bypass capability.  Otherwise the client keeps the
        legacy idle-flush/unique-key behavior.
        """

        if policy not in _SGLANG_CACHE_POLICIES:
            raise ValueError(f"unsupported sglang_cache_policy: {policy!r}")
        with self._cache_capability_lock:
            self._sglang_cache_policy = policy
            self._cache_bypass_capability = None

    def configure_memory_transport_policy(self, policy: str) -> None:
        """Select one coherent CPU or TP-GPU memory transport policy.

        The legacy CPU round-trip remains the default.  Selecting the GPU
        chain is only a local opt-in; a harness must call
        :meth:`attest_memory_transport_policy` and persist its receipt before
        starting an episode.  Unknown or incomplete server generations fail
        closed rather than silently falling back to tensor serialization.
        """

        if policy not in RECURSIVE_COMPRESSION_POLICIES:
            raise ValueError(f"unsupported memory transport policy: {policy!r}")
        with self._memory_transport_lock:
            self._memory_transport_policy = policy
            self._memory_transport_attestation = None
            self._recursive_chain_response_timeout_upper_bound = None

    def attest_memory_transport_policy(
        self,
        server_info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the strict, persistable handshake for the selected policy."""

        with self._memory_transport_lock:
            requested = self._memory_transport_policy
        chain_response_timeout_upper_bound: float | None = None
        if requested == MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1:
            attestation = attest_memory_transport_server_info(
                {}, requested_policy=requested
            )
        else:
            if server_info is None:
                server_info = self.server_info()
            attestation = attest_memory_transport_server_info(
                server_info,
                requested_policy=requested,
            )
            chain_response_timeout_upper_bound = (
                self._attest_recursive_compression_chain_limits(server_info)
            )
        with self._memory_transport_lock:
            if requested != self._memory_transport_policy:
                raise RuntimeError(
                    "memory transport policy changed during server attestation"
                )
            self._memory_transport_attestation = attestation
            self._recursive_chain_response_timeout_upper_bound = (
                chain_response_timeout_upper_bound
            )
        return attestation.to_json()

    def _attest_recursive_compression_chain_limits(
        self,
        server_info: Mapping[str, Any],
    ) -> float | None:
        """Require a server response deadline below the HTTP client timeout."""

        limits = server_info.get("specforge_recursive_compression_chain")
        if limits is None:
            # Compatibility for already-running v1 servers.  The request path
            # uses a bounded two-phase timeout and reconnect grace for these
            # servers so rolling out this client does not strand live pools.
            logger.warning(
                "GPU-chain server does not advertise recursive-chain timeout "
                "limits; enabling bounded legacy retry mode"
            )
            return None
        if not isinstance(limits, Mapping) or set(limits) != (
            _RECURSIVE_COMPRESSION_CHAIN_LIMIT_FIELDS
        ):
            raise RuntimeError(
                "GPU-chain server_info has no exact recursive-chain timeout limits"
            )
        if limits["schema"] != _RECURSIVE_COMPRESSION_CHAIN_LIMITS_SCHEMA:
            raise RuntimeError("unsupported recursive-chain timeout limits schema")

        values: dict[str, float] = {}
        for name in _RECURSIVE_COMPRESSION_CHAIN_LIMIT_FIELDS - {"schema"}:
            value = limits[name]
            allow_zero = name == "orphan_grace_seconds"
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or (not allow_zero and float(value) == 0.0)
            ):
                qualifier = "nonnegative" if allow_zero else "positive"
                raise RuntimeError(
                    f"recursive-chain {name} must be finite and {qualifier}"
                )
            values[name] = float(value)

        execution_timeout = values["execution_timeout_seconds"]
        for name in (
            "scheduler_request_timeout_seconds",
            "handle_release_timeout_seconds",
            "abort_ack_timeout_seconds",
        ):
            if values[name] >= execution_timeout:
                raise RuntimeError(
                    f"recursive-chain {name} must be below its execution deadline"
                )
        conservative_response_bound = (
            execution_timeout
            + values["abort_ack_timeout_seconds"]
            + 2.0 * values["handle_release_timeout_seconds"]
        )
        if values["response_timeout_upper_bound_seconds"] < (
            conservative_response_bound
        ):
            raise RuntimeError(
                "recursive-chain response timeout bound omits abort/release cleanup"
            )
        if self.request_timeout <= values["response_timeout_upper_bound_seconds"]:
            raise RuntimeError(
                "SGLang request_timeout must be strictly greater than the server "
                "recursive-chain response deadline"
            )
        return values["response_timeout_upper_bound_seconds"]

    def _require_gpu_memory_transport_attestation(
        self,
    ) -> MemoryTransportAttestation:
        with self._memory_transport_lock:
            policy = self._memory_transport_policy
            attestation = self._memory_transport_attestation
        if policy != MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1:
            raise RuntimeError(
                "GPU memory transport requires explicit "
                f"{MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1!r} policy selection"
            )
        if attestation is None or attestation.effective != policy:
            raise RuntimeError("GPU memory transport was not attested for this client")
        if attestation.server_epoch is None or attestation.hook_abi_digest is None:
            raise RuntimeError("GPU memory transport attestation is incomplete")
        return attestation

    @staticmethod
    def _server_supports_request_cache_bypass(
        server_info: Mapping[str, Any],
    ) -> bool:
        capabilities = server_info.get("request_capabilities")
        max_running_requests = server_info.get("max_running_requests")
        capabilities_valid = isinstance(capabilities, list) and all(
            isinstance(value, str) for value in capabilities
        )
        batch_contract = (
            isinstance(max_running_requests, int)
            and not isinstance(max_running_requests, bool)
            and (
                max_running_requests == 1
                or (
                    max_running_requests >= 2
                    and capabilities_valid
                    and (
                        SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY in capabilities
                        or (
                            "qwen38_residual_cpu_concurrent_v1" in capabilities
                            and "qwen38_summary_residual_sglang_v1" in capabilities
                        )
                    )
                )
            )
        )
        return (
            capabilities_valid
            and SGLANG_REQUEST_CACHE_BYPASS_CAPABILITY in capabilities
            # Legacy servers remain exact batch-one.  A concurrent server is
            # accepted only when it advertises the encode-singleton/recover-
            # batch contract implemented by the scheduler and model wrapper.
            and batch_contract
        )

    def attest_sglang_cache_policy(
        self,
        server_info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fail closed when an explicitly requested policy is unavailable.

        Individual client calls retain their rollout-safe legacy fallback so a
        live server restart cannot corrupt cache state.  A versioned harness,
        however, must not silently claim the isolated policy while executing
        the legacy flush path.  Loaders call this method once during preflight
        and persist the returned requested/effective receipt.
        """

        requested = self._sglang_cache_policy
        if requested == SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1:
            return {
                "requested": requested,
                "effective": requested,
                "request_cache_bypass_capability": False,
            }
        if server_info is None:
            server_info = self.server_info()
        if not isinstance(server_info, Mapping):
            raise TypeError("server_info must be a mapping")
        supported = self._server_supports_request_cache_bypass(server_info)
        with self._cache_capability_lock:
            self._cache_bypass_capability = supported
        if not supported:
            raise RuntimeError(
                "requested SGLang cache policy is unavailable: server must "
                f"advertise {SGLANG_REQUEST_CACHE_BYPASS_CAPABILITY!r} with "
                "max_running_requests=1, or advertise the soft-memory batch "
                "or residual CPU concurrency contract with max_running_requests>=2"
            )
        return {
            "requested": requested,
            "effective": requested,
            "request_cache_bypass_capability": True,
        }

    def _isolated_cache_policy_active(self) -> bool:
        if (
            self._sglang_cache_policy
            != SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1
        ):
            return False
        with self._cache_capability_lock:
            if self._cache_bypass_capability is not None:
                return self._cache_bypass_capability
            try:
                info = self.server_info()
            except RuntimeError:
                # A capability lookup failure cannot authorize bypass.  Do
                # not cache a transient transport failure; a later operation
                # may safely retry the handshake.
                return False
            supported = self._server_supports_request_cache_bypass(info)
            self._cache_bypass_capability = supported
            return supported

    def _require_cache_bypass_receipt(self, response: Mapping[str, Any]) -> None:
        meta = response.get("meta_info")
        if (
            isinstance(meta, Mapping)
            and meta.get("cache_policy") == SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY
        ):
            return

        error = RuntimeError(
            "SGLang did not acknowledge request-local cache bypass; "
            "refusing the response"
        )
        with self._cache_capability_lock:
            # Treat a missing receipt as evidence that this server generation
            # cannot honor bypass.  Stay on the legacy path until an explicit
            # reconfigure (or a fresh client) performs a new handshake.
            self._cache_bypass_capability = False
        # A server can restart between capability discovery and /generate.
        # Repair any cache entry an older server may have inserted before
        # surfacing the fail-closed protocol error.
        try:
            with self._serialized_request():
                self.flush_cache()
        except BaseException as repair_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"best-effort cache repair also failed: {repair_error}")
        raise error

    @staticmethod
    def _require_soft_reuse_receipt(response: Mapping[str, Any]) -> None:
        meta = response.get("meta_info")
        if (
            isinstance(meta, Mapping)
            and meta.get("cache_policy") == SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY
        ):
            return
        raise RuntimeError(
            "SGLang did not acknowledge identity-bound soft-memory prefix reuse; "
            "refusing the response"
        )

    def configure_request_cancellation(self, enabled: bool = True) -> None:
        """Opt in or out of explicit SGLang request IDs and cancellation.

        Cancellation is disabled by default so legacy ``/generate`` payloads
        remain exactly unchanged.  Enabling it affects requests that begin
        afterwards.  Disabling it does not discard an already-active request;
        that request remains abortable until its transport call exits.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        with self._request_cancellation_lock:
            self._request_cancellation_enabled = enabled

    def configure_generation_degeneration_guard(
        self,
        enabled: bool = True,
        *,
        check_interval_tokens: int = (DEFAULT_GENERATION_GUARD_CHECK_INTERVAL_TOKENS),
        minimum_periodic_span_tokens: int = (
            DEFAULT_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
        ),
        maximum_period_tokens: int = (DEFAULT_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS),
        minimum_repeat_count: int = DEFAULT_GENERATION_GUARD_MINIMUM_REPEAT_COUNT,
        max_retries: int = DEFAULT_GENERATION_GUARD_MAX_RETRIES,
        return_detected_generation: bool = False,
    ) -> None:
        """Configure exact-period loop termination and recovery handoff.

        The legacy default discards a detected generation and retries it with
        identical sampling.  A harness may instead request the stopped first
        generation so it can project a safe, non-executable partial round.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        if not isinstance(return_detected_generation, bool):
            raise TypeError("return_detected_generation must be boolean")
        values = {
            "check_interval_tokens": check_interval_tokens,
            "minimum_periodic_span_tokens": minimum_periodic_span_tokens,
            "maximum_period_tokens": maximum_period_tokens,
            "minimum_repeat_count": minimum_repeat_count,
            "max_retries": max_retries,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if check_interval_tokens <= 0:
            raise ValueError("check_interval_tokens must be positive")
        if minimum_periodic_span_tokens <= 0:
            raise ValueError("minimum_periodic_span_tokens must be positive")
        if maximum_period_tokens <= 0:
            raise ValueError("maximum_period_tokens must be positive")
        if minimum_repeat_count < 2:
            raise ValueError("minimum_repeat_count must be at least 2")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._generation_guard_enabled = enabled
        self._generation_guard_check_interval_tokens = check_interval_tokens
        self._generation_guard_minimum_periodic_span_tokens = (
            minimum_periodic_span_tokens
        )
        self._generation_guard_maximum_period_tokens = maximum_period_tokens
        self._generation_guard_minimum_repeat_count = minimum_repeat_count
        self._generation_guard_max_retries = max_retries
        self._generation_guard_return_detected_generation = return_detected_generation

    def _validate_generate_lifecycle_locked(self) -> None:
        """Optional specialized-client shutdown gate; cancellation lock is held."""

    def _post_generate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST one optionally abortable generation and track its lifetime."""

        request_id: str | None = None
        outbound_payload = payload
        with self._request_cancellation_lock:
            self._validate_generate_lifecycle_locked()
            if self._request_cancellation_enabled:
                if self._request_id_namespace is None:
                    self._request_id_namespace = uuid.uuid4().hex
                self._request_id_sequence += 1
                request_id = (
                    f"specforge-{os.getpid()}-{self._request_id_namespace}-"
                    f"{self._request_id_sequence}"
                )
                if (
                    request_id in self._active_request_ids
                    or request_id in self._abort_retry_state
                    or request_id in self._abort_requests_in_flight
                ):  # pragma: no cover
                    raise RuntimeError("generated a duplicate active SGLang request ID")
                self._active_request_ids.add(request_id)
                outbound_payload = dict(payload)
                outbound_payload["rid"] = request_id
        try:
            return self._post("/generate", outbound_payload)
        except BaseException as error:
            # A client-side socket timeout/disconnect does not prove that the
            # scheduler stopped the request.  Abort the known rid before it is
            # removed from the active set so a long 81,920-token generation
            # cannot continue orphaned after the rollout has already failed.
            if request_id is not None:
                try:
                    self._abort_request(request_id)
                except BaseException as abort_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "best-effort SGLang abort after generation transport "
                            f"failure also failed: {abort_error}"
                        )
            raise
        finally:
            if request_id is not None:
                with self._request_cancellation_lock:
                    self._active_request_ids.discard(request_id)
                    self._abort_retry_state.pop(request_id, None)
                    self._abort_requests_in_flight.discard(request_id)

    def _abort_request(self, request_id: str) -> None:
        """Send one control-plane abort without taking the generation lock."""

        body = json.dumps(
            {"rid": request_id},
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/abort_request",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open_request(
                request, timeout=self.abort_timeout
            ) as response:
                raw = response.read()
                status = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "SGLang /abort_request returned HTTP "
                f"{error.code} for rid {request_id!r}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"SGLang /abort_request failed for rid {request_id!r}: {error}"
            ) from error
        except OSError as error:
            # ``urlopen`` may surface socket timeouts directly rather than
            # wrapping them in URLError on some Python/platform combinations.
            raise RuntimeError(
                f"SGLang /abort_request failed for rid {request_id!r}: {error}"
            ) from error
        if not 200 <= status < 300:
            detail = raw.decode("utf-8", errors="replace")
            raise RuntimeError(
                "SGLang /abort_request returned HTTP "
                f"{status} for rid {request_id!r}: {detail}"
            )
        if raw:
            try:
                result = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # SGLang versions differ between an empty response and a
                # short plain-text acknowledgement.  The HTTP status is the
                # authoritative control-plane acknowledgement in both cases.
                result = None
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(
                    "SGLang /abort_request failed for rid "
                    f"{request_id!r}: {result['error']}"
                )

    def abort_active_requests(self) -> tuple[str, ...]:
        """Abort a stable snapshot of this client's active ``/generate`` calls.

        The active-ID lock is held only while taking the snapshot.  In
        particular, this method never acquires ``_request_lock`` or the
        cross-process generation lock, so a Harbor cancellation callback can
        reach SGLang while a long request is still in flight.  Returned IDs
        have received a successful HTTP acknowledgement; SGLang's endpoint is
        asynchronous, so that acknowledgement is not proof of GPU-side drain.
        In particular, SGLang also returns HTTP 200 when an abort races ahead
        of server-side request registration.  Acknowledged IDs are therefore
        retried while their generate transport remains active, with a per-ID
        monotonic cooldown of 0.25, 0.5, 1, then at most 2 seconds.  Newly
        active IDs remain independently discoverable throughout that cooldown.
        """

        with self._request_cancellation_lock:
            now = time.monotonic()
            candidates = tuple(
                sorted(
                    request_id
                    for request_id in self._active_request_ids
                    if request_id not in self._abort_requests_in_flight
                    and (
                        request_id not in self._abort_retry_state
                        or self._abort_retry_state[request_id][1] <= now
                    )
                )
            )
        aborted: list[str] = []
        failures: list[tuple[str, RuntimeError]] = []
        for request_id in candidates:
            with self._request_cancellation_lock:
                retry_state = self._abort_retry_state.get(request_id)
                if (
                    request_id not in self._active_request_ids
                    or request_id in self._abort_requests_in_flight
                    or (retry_state is not None and retry_state[1] > time.monotonic())
                ):
                    continue
                self._abort_requests_in_flight.add(request_id)
            try:
                self._abort_request(request_id)
            except RuntimeError as error:
                failures.append((request_id, error))
            else:
                aborted.append(request_id)
                with self._request_cancellation_lock:
                    # The generation transport may have completed between the
                    # active snapshot and the abort acknowledgement. Do not
                    # leave stale retry state behind in that race.
                    if request_id in self._active_request_ids:
                        previous_attempts = self._abort_retry_state.get(
                            request_id, (0, 0.0)
                        )[0]
                        attempts = previous_attempts + 1
                        cooldown = 0.25 * (2 ** min(attempts - 1, 3))
                        self._abort_retry_state[request_id] = (
                            attempts,
                            time.monotonic() + cooldown,
                        )
                    else:
                        self._abort_retry_state.pop(request_id, None)
            finally:
                with self._request_cancellation_lock:
                    self._abort_requests_in_flight.discard(request_id)
        if failures:
            detail = "; ".join(
                f"{request_id}: {error}" for request_id, error in failures
            )
            raise RuntimeError(
                "failed to abort one or more active SGLang requests: " + detail
            ) from failures[0][1]
        return tuple(aborted)

    @contextmanager
    def _serialized_request(self):
        """Serialize one flush/request pair across threads and processes."""

        with self._request_lock:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._process_lock_path, flags, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _processor_string(self) -> str:
        if self._custom_processor is None:
            from context_compression.sglang_compressor_hook import (
                compressor_metadata_logit_processor_string,
            )

            self._custom_processor = compressor_metadata_logit_processor_string()
        return self._custom_processor

    def _generation_guard_custom_params(self) -> dict[str, Any] | None:
        """Return the request-local, JSON-safe loop-guard contract."""

        if not self._generation_guard_enabled:
            return None
        from context_compression.sglang_compressor_hook import (
            GENERATION_DEGENERATION_GUARD_CUSTOM_PARAM_KEY,
        )

        return {
            GENERATION_DEGENERATION_GUARD_CUSTOM_PARAM_KEY: {
                "enabled": True,
                "policy": GENERATION_DEGENERATION_GUARD_POLICY,
                "check_interval_tokens": (self._generation_guard_check_interval_tokens),
                "minimum_periodic_span_tokens": (
                    self._generation_guard_minimum_periodic_span_tokens
                ),
                "maximum_period_tokens": (self._generation_guard_maximum_period_tokens),
                "minimum_repeat_count": self._generation_guard_minimum_repeat_count,
                "eos_token_id": self.eos_token_id,
            }
        }

    def _open_request(self, request, *, timeout):
        from context_compression.sglang_http import open_request
        return open_request(request, timeout=timeout, api_key=getattr(self, "_api_key", None))

    def _post(
        self,
        route: str,
        payload: Mapping[str, Any],
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = self.request_timeout if request_timeout is None else request_timeout
        try:
            with self._open_request(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang {route} returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"SGLang {route} request failed: {error}") from error
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError(f"SGLang {route} returned a non-object response")
        if "error" in result:
            raise RuntimeError(f"SGLang {route} failed: {result['error']}")
        return result

    @staticmethod
    def _recursive_chain_error_has_timeout(error: BaseException) -> bool:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, TimeoutError):
                return True
            for linked in (
                current.__cause__,
                current.__context__,
                getattr(current, "reason", None),
            ):
                if isinstance(linked, BaseException):
                    pending.append(linked)
        return False

    @staticmethod
    def _recursive_chain_error_is_retryable(error: BaseException) -> bool:
        """Identify failures where an exact singleflight replay is safe."""

        message = str(error).lower()
        if any(
            marker in message for marker in _RETRYABLE_RECURSIVE_CHAIN_SERVER_ERRORS
        ):
            return True
        if GrepSeekSGLangClient._recursive_chain_error_has_timeout(error):
            return True

        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, json.JSONDecodeError):
                return True
            if isinstance(current, urllib.error.URLError) and not isinstance(
                current, urllib.error.HTTPError
            ):
                return True
            for linked in (
                current.__cause__,
                current.__context__,
                getattr(current, "reason", None),
            ):
                if isinstance(linked, BaseException):
                    pending.append(linked)
        return False

    def health(self) -> None:
        request = urllib.request.Request(self.base_url + "/health", method="GET")
        try:
            with self._open_request(
                request, timeout=min(self.request_timeout, 30.0)
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"SGLang health returned HTTP {response.status}")
        except urllib.error.URLError as error:
            raise RuntimeError(f"SGLang health request failed: {error}") from error

    def server_info(self) -> dict[str, Any]:
        """Return the server's authoritative serving limits and configuration."""

        request = urllib.request.Request(self.base_url + "/server_info", method="GET")
        try:
            with self._open_request(
                request, timeout=min(self.request_timeout, 30.0)
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang /server_info returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"SGLang /server_info request failed: {error}"
            ) from error
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("SGLang /server_info returned a non-object")
        return value

    def flush_cache(self) -> None:
        """Synchronously clear prefix, KV, and hybrid-state caches.

        Recovery requests contain continuous memory embeddings whose values are
        not represented by their placeholder token IDs.  They must therefore
        never share a radix-cache entry with another request.  The Qwen3.5
        hybrid Mamba cache in the compatibility fork does not reliably isolate
        these entries with ``extra_key``, so use SGLang's idle-only cache flush
        as the correctness boundary.
        """

        request = urllib.request.Request(
            self.base_url + "/flush_cache?timeout=30",
            data=b"",
            method="POST",
        )
        try:
            # The server can legitimately defer this request for up to 30 s.
            # Keep the transport timeout above that even when generation uses
            # a shorter user-supplied timeout, otherwise a live pending flush
            # would outlast the client call.
            with self._open_request(request, timeout=45.0) as response:
                detail = response.read().decode("utf-8", errors="replace")
                if response.status != 200:
                    raise RuntimeError(
                        f"SGLang cache flush returned HTTP {response.status}: {detail}"
                    )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang cache flush returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"SGLang cache flush request failed: {error}") from error

    def encode_recursive_memory_chain(
        self,
        transitions: Sequence[Mapping[str, Any]],
        *,
        operation_id: str,
        compression_ratio: int,
        expected_hidden_size: int = 4096,
        parent_memory_handle: Mapping[str, Any] | TPGPUMemoryHandle | None = None,
    ) -> RecursiveCompressionChainResult:
        """Submit two or more recursive transitions in one HTTP request.

        The server executes the items one by one through its ordinary
        scheduler and returns only the final GPU-memory handle.  No legacy
        fallback exists here: callers must explicitly opt into and attest the
        unified GPU-chain transport policy first.
        """

        attestation = self._require_gpu_memory_transport_attestation()
        if isinstance(transitions, (str, bytes)) or not isinstance(
            transitions, Sequence
        ):
            raise TypeError("transitions must be a sequence of mappings")
        normalized_transitions: list[dict[str, Any]] = []
        for index, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                raise TypeError(f"transitions[{index}] must be a mapping")
            normalized_transitions.append(dict(transition))
        if isinstance(parent_memory_handle, TPGPUMemoryHandle):
            parent_json = parent_memory_handle.to_json()
        elif parent_memory_handle is None:
            parent_json = None
        elif isinstance(parent_memory_handle, Mapping):
            parent_json = dict(parent_memory_handle)
        else:
            raise TypeError("parent_memory_handle must be a descriptor mapping or None")
        payload = {
            "schema": RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA,
            "operation_id": operation_id,
            "execution_policy": RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY,
            "commit_policy": RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY,
            "server_epoch": attestation.server_epoch,
            "hook_abi_digest": attestation.hook_abi_digest,
            "expected_hidden_size": expected_hidden_size,
            "compression_ratio": compression_ratio,
            "custom_logit_processor": self._processor_string(),
            "cache_policy": SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY,
            "parent_memory_handle": parent_json,
            "transitions": normalized_transitions,
        }
        request = RecursiveCompressionChainRequest.parse(payload)
        wire_payload = request.to_json()
        payload_digest = hashlib.sha256(
            json.dumps(
                wire_payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self._memory_transport_lock:
            advertised_response_bound = (
                self._recursive_chain_response_timeout_upper_bound
            )
        legacy_retry_grace_seconds = 0.0
        if advertised_response_bound is None:
            # Old servers use a 30 s zero-waiter grace and have no stage
            # deadlines.  Reserve another 60 s for delayed ASGI disconnect
            # observation/event-loop scheduling before reconnecting.  Small
            # caller budgets scale this window down rather than making either
            # HTTP attempt nonpositive.  Both attempts plus the grace stay
            # within the caller's original request_timeout budget.
            legacy_retry_grace_seconds = min(
                _LEGACY_RECURSIVE_CHAIN_RECONNECT_GRACE_SECONDS,
                self.request_timeout * 0.25,
            )
            first_timeout = min(
                300.0,
                (self.request_timeout - legacy_retry_grace_seconds) * 0.5,
            )
            attempt_timeouts = (
                first_timeout,
                self.request_timeout - legacy_retry_grace_seconds - first_timeout,
            )
        else:
            attempt_timeouts = (self.request_timeout, self.request_timeout)
        for attempt, attempt_timeout in enumerate(attempt_timeouts, start=1):
            started = time.monotonic()
            logger.info(
                "recursive-chain HTTP attempt started: operation_id=%s "
                "payload_sha256=%s attempt=%d/2 transitions=%d timeout_seconds=%.3f",
                request.operation_id,
                payload_digest,
                attempt,
                len(request.transitions),
                attempt_timeout,
            )
            try:
                response = self._post(
                    RECURSIVE_COMPRESSION_CHAIN_ROUTE,
                    wire_payload,
                    request_timeout=attempt_timeout,
                )
            except Exception as error:
                elapsed = time.monotonic() - started
                if attempt == 1 and self._recursive_chain_error_is_retryable(error):
                    if (
                        advertised_response_bound is None
                        and legacy_retry_grace_seconds > 0.0
                        and self._recursive_chain_error_has_timeout(error)
                    ):
                        logger.warning(
                            "legacy recursive-chain request timed out; waiting "
                            "%.3fs for server orphan cleanup before exact retry: "
                            "operation_id=%s payload_sha256=%s",
                            legacy_retry_grace_seconds,
                            request.operation_id,
                            payload_digest,
                        )
                        time.sleep(legacy_retry_grace_seconds)
                    logger.warning(
                        "recursive-chain HTTP attempt failed; retrying exact "
                        "operation/payload once: operation_id=%s payload_sha256=%s "
                        "elapsed_seconds=%.3f error=%s",
                        request.operation_id,
                        payload_digest,
                        elapsed,
                        error,
                    )
                    continue
                logger.error(
                    "recursive-chain HTTP attempt failed: operation_id=%s "
                    "payload_sha256=%s attempt=%d/2 elapsed_seconds=%.3f error=%s",
                    request.operation_id,
                    payload_digest,
                    attempt,
                    elapsed,
                    error,
                )
                raise
            logger.info(
                "recursive-chain HTTP attempt completed: operation_id=%s "
                "payload_sha256=%s attempt=%d/2 elapsed_seconds=%.3f",
                request.operation_id,
                payload_digest,
                attempt,
                time.monotonic() - started,
            )
            break
        else:  # pragma: no cover - the two explicit branches always exit
            raise AssertionError("recursive-chain retry loop exhausted")
        return parse_recursive_compression_chain_response(
            response,
            request=request,
        )

    def encode_memory_handle(
        self,
        source_input_ids: Sequence[int],
        *,
        evidence_spans: Sequence[Sequence[int]],
        source_token_lengths: Sequence[int],
        operation_id: str,
        expected_slot_count: int,
        recursive_source_contract: str | None = None,
        memory_positions: Sequence[int] | None = None,
        parent_memory_handle: Mapping[str, Any] | TPGPUMemoryHandle | None = None,
        expected_hidden_size: int = 4096,
    ) -> TPGPUMemoryHandle:
        """Publish one encode result directly to the TP-local GPU registry."""

        from context_compression.sglang_compressor_hook import (
            GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
            build_recursive_encode_custom_params,
            compute_memory_handle_identity,
            compute_memory_handle_request_digest,
        )

        attestation = self._require_gpu_memory_transport_attestation()
        input_ids = _integer_list(source_input_ids, name="source_input_ids")
        if isinstance(parent_memory_handle, TPGPUMemoryHandle):
            parent = parent_memory_handle
        elif parent_memory_handle is None:
            parent = None
        else:
            parent = TPGPUMemoryHandle.parse(
                parent_memory_handle,
                expected_server_epoch=attestation.server_epoch,
                expected_hidden_size=expected_hidden_size,
            )
        if parent is not None and parent.server_epoch != attestation.server_epoch:
            raise RuntimeError("parent memory handle belongs to a stale server epoch")
        request_digest = compute_memory_handle_request_digest(
            input_ids=input_ids,
            evidence_spans=evidence_spans,
            source_token_lengths=source_token_lengths,
            memory_positions=memory_positions,
            previous_memory_handle=(None if parent is None else parent.to_json()),
            recursive_source_contract=recursive_source_contract,
        )
        assert attestation.server_epoch is not None
        assert attestation.hook_abi_digest is not None
        memory_identity = compute_memory_handle_identity(
            server_epoch=attestation.server_epoch,
            request_digest=request_digest,
            hook_abi_digest=attestation.hook_abi_digest,
        )
        publish = {
            "schema": GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
            "op_id": operation_id,
            "memory_identity": memory_identity,
            "request_digest": request_digest,
        }
        custom_params = build_recursive_encode_custom_params(
            evidence_spans=evidence_spans,
            source_token_lengths=source_token_lengths,
            memory_positions=memory_positions,
            memory_handle=(None if parent is None else parent.to_json()),
            memory_handle_publish=publish,
            recursive_source_contract=recursive_source_contract,
            hidden_size=expected_hidden_size,
        )
        payload = {
            "input_ids": input_ids,
            "extra_key": f"specforge-handle-encode-{operation_id}",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "custom_params": custom_params,
            },
            "return_hidden_states": False,
            "custom_logit_processor": self._processor_string(),
            "cache_policy": SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY,
        }
        response = self._post_generate(payload)
        meta = response.get("meta_info")
        if not isinstance(meta, Mapping):
            raise RuntimeError("SGLang handle encode response has no meta_info object")
        # First take cleanup ownership of any authentic descriptor.  Strict
        # transition checks happen afterwards so a wrong-shape/identity or
        # forbidden-tensor receipt cannot strand the server-published handle.
        candidate = TPGPUMemoryHandle.parse(
            meta.get(GPU_MEMORY_HANDLE_META_KEY),
            expected_server_epoch=attestation.server_epoch,
            expected_memory_identity=memory_identity,
            expected_request_digest=request_digest,
        )
        try:
            self._require_cache_bypass_receipt(response)
            if {"hidden_states", "memory_embeddings"}.intersection(meta):
                raise RuntimeError("SGLang handle encode leaked tensor payload")
            return TPGPUMemoryHandle.parse(
                candidate.to_json(),
                expected_server_epoch=attestation.server_epoch,
                expected_slots=expected_slot_count,
                expected_hidden_size=expected_hidden_size,
                expected_memory_identity=memory_identity,
                expected_request_digest=request_digest,
            )
        except BaseException as error:
            cleanup_operation_id = (
                "encode_cleanup_" + candidate.descriptor["descriptor_digest"][7:]
            )
            try:
                self.release_memory_handle(
                    candidate,
                    operation_id=cleanup_operation_id,
                )
            except BaseException as cleanup_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "published handle cleanup also failed: " + str(cleanup_error)
                    )
            raise

    def release_memory_handle(
        self,
        memory_handle: Mapping[str, Any] | TPGPUMemoryHandle,
        *,
        operation_id: str,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Idempotently release one descriptor from every TP rank."""

        handle = (
            memory_handle
            if isinstance(memory_handle, TPGPUMemoryHandle)
            else TPGPUMemoryHandle.parse(memory_handle)
        )
        response = self._post(
            GPU_MEMORY_HANDLE_RELEASE_ROUTE,
            {
                "schema": GPU_MEMORY_HANDLE_RELEASE_SCHEMA,
                "op_id": operation_id,
                "handle": handle.to_json(),
            },
            request_timeout=request_timeout,
        )
        expected_fields = {
            "success",
            "schema",
            "op_id",
            "server_epoch",
            "released",
            "already_released",
            "message",
        }
        if set(response) != expected_fields or response.get("success") is not True:
            raise RuntimeError("SGLang memory-handle release was not acknowledged")
        if (
            response["schema"] != GPU_MEMORY_HANDLE_RELEASE_SCHEMA
            or response["op_id"] != operation_id
            or response["server_epoch"] != handle.server_epoch
            or not isinstance(response["released"], bool)
            or not isinstance(response["already_released"], bool)
            or response["released"] == response["already_released"]
        ):
            raise RuntimeError(
                "SGLang returned an invalid memory-handle release receipt"
            )
        return response

    def encode_memory(
        self,
        source_input_ids: Sequence[int],
        *,
        evidence_spans: Sequence[Sequence[int]],
        source_token_lengths: Sequence[int],
        expected_slot_count: int | None = None,
        recursive: bool = False,
        recursive_source_contract: str | None = None,
        memory_positions: Sequence[int] | None = None,
        memory_embeddings: torch.Tensor | Sequence[Sequence[float]] | None = None,
        expected_hidden_size: int = 4096,
    ) -> torch.Tensor:
        """Return packed ``[total_slots, hidden]`` memory on CPU."""

        from context_compression.sglang_compressor_hook import (
            build_encode_custom_params,
            build_recursive_encode_custom_params,
        )

        input_ids = _integer_list(source_input_ids, name="source_input_ids")
        if (
            isinstance(expected_hidden_size, bool)
            or not isinstance(expected_hidden_size, int)
            or expected_hidden_size <= 0
        ):
            raise ValueError("expected_hidden_size must be a positive integer")
        if not isinstance(recursive, bool):
            raise TypeError("recursive must be boolean")
        if recursive:
            if isinstance(memory_embeddings, torch.Tensor):
                memory_values = memory_embeddings.detach().float().cpu().tolist()
            elif memory_embeddings is None:
                memory_values = None
            else:
                memory_values = [list(map(float, row)) for row in memory_embeddings]
            custom_params = build_recursive_encode_custom_params(
                evidence_spans=evidence_spans,
                source_token_lengths=source_token_lengths,
                memory_positions=memory_positions,
                memory_embeddings=memory_values,
                recursive_source_contract=recursive_source_contract,
                hidden_size=expected_hidden_size,
            )
        else:
            if (
                memory_positions is not None
                or memory_embeddings is not None
                or recursive_source_contract is not None
            ):
                raise ValueError("recursive source fields require recursive=True")
            custom_params = build_encode_custom_params(
                evidence_spans=evidence_spans,
                source_token_lengths=source_token_lengths,
            )
        payload = {
            "input_ids": input_ids,
            "extra_key": f"specforge-encode-{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "custom_params": custom_params,
            },
            "return_hidden_states": True,
            "custom_logit_processor": self._processor_string(),
        }
        isolated_cache = self._isolated_cache_policy_active()
        if isolated_cache:
            payload["cache_policy"] = SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY
            response = self._post_generate(payload)
            self._require_cache_bypass_receipt(response)
        else:
            with self._serialized_request():
                self.flush_cache()
                response = self._post_generate(payload)
        meta = response.get("meta_info")
        if not isinstance(meta, dict):
            raise RuntimeError("SGLang encode response has no meta_info object")
        chunks = meta.get("hidden_states")
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise RuntimeError(
                "SGLang encode must return exactly one prefill hidden-state chunk"
            )
        wire = torch.tensor(chunks[0], dtype=torch.float32)
        if wire.ndim != 2 or wire.shape[0] <= 0 or wire.shape[1] <= 0:
            raise RuntimeError(
                f"SGLang encode returned invalid memory shape {tuple(wire.shape)}"
            )
        if expected_slot_count is not None and (
            isinstance(expected_slot_count, bool) or int(expected_slot_count) <= 0
        ):
            raise ValueError("expected_slot_count must be positive or None")
        if expected_slot_count is not None and wire.shape[0] != int(
            expected_slot_count
        ):
            raise RuntimeError(
                "SGLang encode returned the wrong slot count: "
                f"expected={expected_slot_count}, observed={wire.shape[0]}"
            )
        if not bool(torch.isfinite(wire).all().item()):
            raise RuntimeError("SGLang encode returned non-finite memory")
        if wire.shape[1] != expected_hidden_size:
            raise RuntimeError(
                "SGLang encode returned the wrong memory width: "
                f"expected={expected_hidden_size}, observed={wire.shape[1]}"
            )
        return wire

    def generate_with_memory(
        self,
        prefix_input_ids: Sequence[int],
        *,
        memory_positions: Sequence[int],
        memory_embeddings: torch.Tensor | Sequence[Sequence[float]],
        max_new_tokens: int,
        expected_hidden_size: int = 4096,
        sampling_overrides: Mapping[str, Any] | None = None,
        image_data: Sequence[str] | None = None,
    ) -> SGLangGeneration:
        from context_compression.sglang_compressor_hook import (
            build_recover_custom_params,
        )

        input_ids = _integer_list(prefix_input_ids, name="prefix_input_ids")
        if isinstance(memory_embeddings, torch.Tensor):
            memory_values = memory_embeddings.detach().float().cpu().tolist()
        else:
            memory_values = [list(map(float, row)) for row in memory_embeddings]
        custom_params = build_recover_custom_params(
            memory_positions=memory_positions,
            memory_embeddings=memory_values,
            hidden_size=expected_hidden_size,
        )
        return self._generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            custom_params=custom_params,
            sampling_overrides=sampling_overrides,
            image_data=image_data,
        )

    def generate_with_memory_handle(
        self,
        prefix_input_ids: Sequence[int],
        *,
        memory_positions: Sequence[int],
        memory_handle: Mapping[str, Any] | TPGPUMemoryHandle,
        max_new_tokens: int,
        expected_hidden_size: int = 4096,
        sampling_overrides: Mapping[str, Any] | None = None,
        image_data: Sequence[str] | None = None,
    ) -> SGLangGeneration:
        """Recover from an identity-bound server GPU-memory handle."""

        from context_compression.sglang_compressor_hook import (
            build_recover_custom_params,
        )

        attestation = self._require_gpu_memory_transport_attestation()
        handle = (
            memory_handle
            if isinstance(memory_handle, TPGPUMemoryHandle)
            else TPGPUMemoryHandle.parse(memory_handle)
        )
        if handle.server_epoch != attestation.server_epoch:
            raise RuntimeError("recovery memory handle belongs to a stale server epoch")
        input_ids = _integer_list(prefix_input_ids, name="prefix_input_ids")
        custom_params = build_recover_custom_params(
            memory_positions=memory_positions,
            memory_handle=handle.to_json(),
            hidden_size=expected_hidden_size,
        )
        return self._generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            custom_params=custom_params,
            sampling_overrides=sampling_overrides,
            # SGLang deliberately rejects ``soft_reuse`` when native
            # multimodal inputs are present. Image-bearing recovery must be a
            # complete cache-free EXTEND so the model can compose native vision
            # embeddings with the continuous-memory rows. ``_generate``
            # enforces request-local bypass for that narrow combination.
            memory_transport_cache_policy=(
                None if image_data else SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY
            ),
            image_data=image_data,
        )

    def generate_discrete(
        self,
        prefix_input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        sampling_overrides: Mapping[str, Any] | None = None,
        image_data: Sequence[str] | None = None,
    ) -> SGLangGeneration:
        return self._generate(
            _integer_list(prefix_input_ids, name="prefix_input_ids"),
            max_new_tokens=max_new_tokens,
            custom_params=None,
            sampling_overrides=sampling_overrides,
            image_data=image_data,
        )

    def _generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        custom_params: Mapping[str, Any] | None,
        sampling_overrides: Mapping[str, Any] | None,
        memory_transport_cache_policy: str | None = None,
        image_data: Sequence[str] | None = None,
    ) -> SGLangGeneration:
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if memory_transport_cache_policy not in {
            None,
            SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY,
        }:
            raise ValueError("unsupported memory transport cache policy")
        if memory_transport_cache_policy is not None and custom_params is None:
            raise ValueError(
                "a memory transport cache policy requires soft-memory custom params"
            )
        normalized_images: list[str] = []
        if image_data is not None:
            if isinstance(image_data, (str, bytes)) or not isinstance(
                image_data, Sequence
            ):
                raise TypeError("image_data must be a sequence of nonempty strings")
            for index, value in enumerate(image_data):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"image_data[{index}] must be a nonempty string")
                normalized_images.append(value)
            if not normalized_images:
                raise ValueError("image_data must not be empty when provided")
        sampling: dict[str, Any] = {
            "temperature": 0.0,
            "max_new_tokens": int(max_new_tokens),
            "stop_token_ids": [self.eos_token_id],
            "skip_special_tokens": False,
            "no_stop_trim": True,
        }
        sampling.update(_generation_sampling_overrides(sampling_overrides))
        request_custom_params = dict(custom_params or {})
        guard_params = self._generation_guard_custom_params()
        if guard_params is not None:
            overlap = set(request_custom_params).intersection(guard_params)
            if overlap:  # pragma: no cover - reserved-key defense
                raise RuntimeError(
                    "generation custom_params use reserved guard key: "
                    + ", ".join(sorted(overlap))
                )
            request_custom_params.update(guard_params)
        if request_custom_params:
            sampling["custom_params"] = request_custom_params

        base_payload: dict[str, Any] = {
            "input_ids": [int(value) for value in input_ids],
            "sampling_params": sampling,
        }
        if normalized_images:
            # SGLang expands the Qwen image placeholders in ``input_ids`` and
            # owns image resizing, grid construction, mRoPE, and vision
            # embedding. Keeping the media on this native wire field avoids a
            # second client-side processor whose expansion could drift.
            base_payload["image_data"] = normalized_images
        if request_custom_params:
            base_payload["custom_logit_processor"] = self._processor_string()

        detections: list[dict[str, Any]] = []
        maximum_attempts = 1 + (
            self._generation_guard_max_retries if self._generation_guard_enabled else 0
        )
        soft_reuse = (
            memory_transport_cache_policy == SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY
        )
        soft_memory_request = custom_params is not None
        multimodal_soft_memory_request = bool(normalized_images) and soft_memory_request
        if soft_reuse and multimodal_soft_memory_request:  # pragma: no cover - defense
            raise ValueError("multimodal soft-memory recovery cannot use soft_reuse")
        isolated_cache = (
            True
            if multimodal_soft_memory_request
            else (False if soft_reuse else self._isolated_cache_policy_active())
        )
        request_boundary = (
            nullcontext()
            if isolated_cache or soft_reuse
            else self._serialized_request()
        )
        with request_boundary:
            for attempt_number in range(1, maximum_attempts + 1):
                payload = dict(base_payload)
                if soft_reuse:
                    payload["cache_policy"] = SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY
                elif multimodal_soft_memory_request:
                    # Qwen3.8's native vision + SpecForge memory composition
                    # requires prefix_len=0. Bypass is mandatory even when the
                    # client's configured text-only policy is legacy.
                    payload["extra_key"] = f"specforge-mm-recover-{uuid.uuid4().hex}"
                    payload["cache_policy"] = SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY
                elif isolated_cache:
                    if soft_memory_request:
                        # The server performs neither lookup nor insertion for
                        # continuous-memory requests.  A unique namespace is
                        # retained only as defense in depth and diagnostics.
                        payload["extra_key"] = f"specforge-recover-{uuid.uuid4().hex}"
                        payload["cache_policy"] = (
                            SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY
                        )
                    # Discrete requests intentionally carry neither bypass nor
                    # a unique cache key, restoring ordinary radix-prefix reuse.
                else:
                    # Legacy policy keeps the exact idle-flush/unique-key
                    # correctness boundary used by already-running jobs.
                    payload["extra_key"] = f"specforge-recover-{uuid.uuid4().hex}"
                    self.flush_cache()
                response = self._post_generate(payload)
                if soft_reuse:
                    self._require_soft_reuse_receipt(response)
                elif isolated_cache and soft_memory_request:
                    self._require_cache_bypass_receipt(response)

                token_ids_raw = response.get("output_ids")
                if not isinstance(token_ids_raw, list):
                    raise RuntimeError(
                        "SGLang generation response has no output_ids list"
                    )
                token_ids = tuple(int(token_id) for token_id in token_ids_raw)
                meta_raw = response.get("meta_info")
                if not isinstance(meta_raw, dict):
                    raise RuntimeError(
                        "SGLang generation response has no meta_info object"
                    )
                meta = dict(meta_raw)
                finish_kind = _finish_reason_kind(meta.get("finish_reason"))
                externally_aborted = finish_kind in {
                    "abort",
                    "aborted",
                    "cancel",
                    "canceled",
                    "cancelled",
                }
                eos_reached = self.eos_token_id in token_ids
                if eos_reached:
                    token_ids = token_ids[: token_ids.index(self.eos_token_id) + 1]
                body_ids = token_ids[:-1] if eos_reached else token_ids

                detection = None
                if self._generation_guard_enabled and not externally_aborted:
                    detection = detect_exact_periodic_tail(
                        body_ids,
                        minimum_periodic_span_tokens=(
                            self._generation_guard_minimum_periodic_span_tokens
                        ),
                        maximum_period_tokens=(
                            self._generation_guard_maximum_period_tokens
                        ),
                        minimum_repeat_count=(
                            self._generation_guard_minimum_repeat_count
                        ),
                    )
                if detection is not None:
                    detections.append(
                        {
                            "attempt_number": attempt_number,
                            **detection.to_json(),
                            "finished_on_eos": eos_reached,
                        }
                    )
                    if self._generation_guard_return_detected_generation:
                        telemetry = self._generation_guard_telemetry(
                            attempt_count=attempt_number,
                            detections=detections,
                            retry_exhausted=False,
                        )
                        telemetry.update(
                            {
                                "detected_generation_returned": True,
                                "action": "return_partial_for_controller_recovery",
                                "controller_recovery_required": True,
                                "retry_suppressed": True,
                                "retry_suppressed_reason": (
                                    "controller_partial_generation_recovery"
                                ),
                                "wire_eos_observed": eos_reached,
                                "wire_output_token_count": len(token_ids),
                                "returned_body_token_count": len(body_ids),
                            }
                        )
                        meta["generation_degeneration_guard"] = telemetry
                        return SGLangGeneration(
                            token_ids=body_ids,
                            finished_on_eos=False,
                            finish_reason=meta.get("finish_reason"),
                            meta_info=meta,
                        )
                    if attempt_number < maximum_attempts:
                        continue
                    raise DegenerateGenerationError(
                        self._generation_guard_telemetry(
                            attempt_count=attempt_number,
                            detections=detections,
                            retry_exhausted=True,
                        )
                    )

                if self._generation_guard_enabled:
                    meta["generation_degeneration_guard"] = (
                        self._generation_guard_telemetry(
                            attempt_count=attempt_number,
                            detections=detections,
                            retry_exhausted=False,
                        )
                    )
                    if externally_aborted:
                        meta["generation_degeneration_guard"][
                            "retry_suppressed_finish_reason"
                        ] = finish_kind
                return SGLangGeneration(
                    token_ids=token_ids,
                    finished_on_eos=eos_reached,
                    finish_reason=meta.get("finish_reason"),
                    meta_info=meta,
                )

        raise AssertionError("generation attempt loop did not return or raise")

    def _generation_guard_telemetry(
        self,
        *,
        attempt_count: int,
        detections: Sequence[Mapping[str, Any]],
        retry_exhausted: bool,
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "policy": GENERATION_DEGENERATION_GUARD_POLICY,
            "attempt_count": int(attempt_count),
            "retry_count": int(attempt_count) - 1,
            "maximum_retries": self._generation_guard_max_retries,
            "check_interval_tokens": (self._generation_guard_check_interval_tokens),
            "minimum_periodic_span_tokens": (
                self._generation_guard_minimum_periodic_span_tokens
            ),
            "maximum_period_tokens": self._generation_guard_maximum_period_tokens,
            "minimum_repeat_count": self._generation_guard_minimum_repeat_count,
            "sampling_unchanged_across_retries": True,
            "generation_budget_unchanged_across_retries": True,
            "detections": [dict(value) for value in detections],
            "retry_exhausted": bool(retry_exhausted),
        }


__all__ = [
    "DEFAULT_GENERATION_GUARD_CHECK_INTERVAL_TOKENS",
    "DEFAULT_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS",
    "DEFAULT_GENERATION_GUARD_MAX_RETRIES",
    "DEFAULT_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS",
    "DEFAULT_GENERATION_GUARD_MINIMUM_REPEAT_COUNT",
    "DegenerateGenerationError",
    "ExactPeriodicTailDetection",
    "GENERATION_DEGENERATION_GUARD_POLICY",
    "GrepSeekSGLangClient",
    "MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1",
    "MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1",
    "MemoryTransportAttestation",
    "RecursiveCompressionChainResult",
    "SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1",
    "SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1",
    "SGLANG_REQUEST_CACHE_BYPASS_CAPABILITY",
    "SGLANG_REQUEST_CACHE_BYPASS_WIRE_POLICY",
    "SGLangGeneration",
    "TPGPUMemoryHandle",
    "detect_exact_periodic_tail",
]
