"""Fail-closed, secret-free attestation for a direct r^2eAct SGLang server.

This module deliberately has no benchmark, Harbor, SpecForge, or third-party
dependencies.  It only inspects the live SGLang HTTP contract and returns a
small allowlisted report; the raw ``/server_info`` object is never returned
because it may contain API-key fields in some SGLang versions.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


R2EACT_FORWARD_HOOK_FACTORY = (
    "context_compression.sglang_compressor_hook:make_r2eact_compressor_hook"
)
RUNTIME_ATTESTATION_SCHEMA = "r2eact_direct_sglang_runtime_attestation_v1"
DEFAULT_MAXIMUM_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeAttestationSpec:
    """Expected identity and serving contract of one direct SGLang runtime.

    ``expected_checkpoint`` is compared after resolving both the expected and
    server-reported paths.  When ``expected_model_path`` is set, both
    server-info aliases must report a path resolving exactly to it.
    Authentication material is intentionally not part of this contract:
    ``sglang_url`` must be a credential-free HTTP(S) origin.
    """

    sglang_url: str
    expected_served_model: str
    expected_forward_hook_factory: str
    expected_checkpoint: str | os.PathLike[str]
    expected_generation_degeneration_guard_server_abi: str
    expected_model_path: str | os.PathLike[str] | None = None
    expected_sync_token_ids_across_tp: bool = True
    require_custom_logit_processor: bool = True
    request_timeout_seconds: float = 30.0
    maximum_response_bytes: int = DEFAULT_MAXIMUM_RESPONSE_BYTES


def _nonempty_text(value: Any, *, field: str, maximum_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"SGLang {field} is missing or is not a string")
    if len(value) > maximum_length or any(char in value for char in "\x00\r\n"):
        raise RuntimeError(f"SGLang {field} is not a safe public identifier")
    return value


def _expected_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if len(value) > 4096 or any(char in value for char in "\x00\r\n"):
        raise ValueError(f"{field} is not a safe public identifier")
    return value


def _resolved_path(value: Any, *, field: str, expected: bool) -> Path:
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        error_type = ValueError if expected else RuntimeError
        raise error_type(f"{field} must be a filesystem path") from exc
    if not isinstance(rendered, str) or not rendered.strip():
        error_type = ValueError if expected else RuntimeError
        raise error_type(f"{field} must be a nonempty filesystem path")
    if "\x00" in rendered or "\r" in rendered or "\n" in rendered:
        error_type = ValueError if expected else RuntimeError
        raise error_type(f"{field} is not a safe filesystem path")
    try:
        return Path(rendered).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        error_type = ValueError if expected else RuntimeError
        raise error_type(f"could not resolve {field}") from exc


def _validated_base_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sglang_url must be a nonempty string")
    try:
        parsed = urllib.parse.urlsplit(value)
        # Accessing .port performs its own validation.
        parsed.port
    except ValueError as exc:
        raise ValueError("sglang_url is not a valid HTTP(S) origin") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("sglang_url must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("sglang_url must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("sglang_url must not contain a path, query, or fragment")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, "", "", "")
    )


def _validate_spec(spec: RuntimeAttestationSpec) -> tuple[str, Path, Path | None]:
    if not isinstance(spec, RuntimeAttestationSpec):
        raise TypeError("spec must be a RuntimeAttestationSpec")
    base_url = _validated_base_url(spec.sglang_url)
    _expected_text(spec.expected_served_model, field="expected_served_model")
    _expected_text(
        spec.expected_forward_hook_factory,
        field="expected_forward_hook_factory",
    )
    _expected_text(
        spec.expected_generation_degeneration_guard_server_abi,
        field="expected_generation_degeneration_guard_server_abi",
    )
    expected_checkpoint = _resolved_path(
        spec.expected_checkpoint,
        field="expected_checkpoint",
        expected=True,
    )
    expected_model_path = (
        _resolved_path(
            spec.expected_model_path,
            field="expected_model_path",
            expected=True,
        )
        if spec.expected_model_path is not None
        else None
    )
    if spec.expected_sync_token_ids_across_tp is not True:
        raise ValueError("expected_sync_token_ids_across_tp must be true")
    if spec.require_custom_logit_processor is not True:
        raise ValueError("require_custom_logit_processor must be true")
    timeout = spec.request_timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("request_timeout_seconds must be finite and positive")
    maximum = spec.maximum_response_bytes
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum_response_bytes must be a positive integer")
    return base_url, expected_checkpoint, expected_model_path


def _request(
    base_url: str,
    route: str,
    *,
    timeout: float,
) -> tuple[Any, int]:
    url = base_url + route
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "r2eact-runtime-attestor/1",
        },
        method="GET",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"SGLang {route} returned HTTP {int(exc.code)}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"SGLang {route} request failed") from exc
    return response, int(getattr(response, "status", 0))


def _check_response_identity(response: Any, *, expected_url: str, route: str) -> None:
    geturl = getattr(response, "geturl", None)
    observed_url = geturl() if callable(geturl) else None
    if observed_url != expected_url:
        raise RuntimeError(f"SGLang {route} redirected unexpectedly")


def _health(
    base_url: str,
    *,
    timeout: float,
) -> int:
    response, status = _request(base_url, "/health", timeout=timeout)
    try:
        _check_response_identity(
            response,
            expected_url=base_url + "/health",
            route="/health",
        )
        if status != 200:
            raise RuntimeError(f"SGLang /health returned HTTP {status}")
    finally:
        response.close()
    return status


def _get_json(
    base_url: str,
    route: str,
    *,
    timeout: float,
    maximum_response_bytes: int,
) -> tuple[Mapping[str, Any], int]:
    response, status = _request(base_url, route, timeout=timeout)
    try:
        _check_response_identity(
            response,
            expected_url=base_url + route,
            route=route,
        )
        if status != 200:
            raise RuntimeError(f"SGLang {route} returned HTTP {status}")
        raw = response.read(maximum_response_bytes + 1)
    finally:
        response.close()
    if len(raw) > maximum_response_bytes:
        raise RuntimeError(f"SGLang {route} response exceeds the size limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SGLang {route} returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"SGLang {route} returned a non-object")
    return value, status


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"SGLang /server_info has no positive {field!r}")
    return value


def _model_ids(models: Mapping[str, Any]) -> list[str]:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("SGLang /v1/models has no nonempty data list")
    result: list[str] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise RuntimeError("SGLang /v1/models contains a malformed model")
        result.append(_nonempty_text(item.get("id"), field="model id"))
    return result


def _attest_hook(
    server_info: Mapping[str, Any],
    *,
    spec: RuntimeAttestationSpec,
    expected_checkpoint: Path,
    endpoint: str,
) -> dict[str, Any]:
    raw_hooks = server_info.get("forward_hooks")
    if not isinstance(raw_hooks, list):
        raise RuntimeError(f"SGLang {endpoint} has no forward_hooks list")
    matching = [
        hook
        for hook in raw_hooks
        if isinstance(hook, Mapping)
        and hook.get("hook_factory") == spec.expected_forward_hook_factory
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"SGLang {endpoint} must report exactly one expected forward-hook factory"
        )
    hook_config = matching[0].get("config")
    if not isinstance(hook_config, Mapping):
        raise RuntimeError(f"SGLang {endpoint} forward-hook config is missing")
    observed_checkpoint = _resolved_path(
        hook_config.get("checkpoint"),
        field=f"{endpoint} forward-hook checkpoint",
        expected=False,
    )
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(f"SGLang {endpoint} forward-hook checkpoint mismatch")
    observed_abi = hook_config.get("generation_degeneration_guard_server_abi")
    if observed_abi != spec.expected_generation_degeneration_guard_server_abi:
        raise RuntimeError(
            f"SGLang {endpoint} generation degeneration guard server ABI mismatch"
        )
    observed_sync = hook_config.get("sync_token_ids_across_tp")
    if observed_sync is not spec.expected_sync_token_ids_across_tp:
        raise RuntimeError(
            f"SGLang {endpoint} sync_token_ids_across_tp mismatch"
        )
    return {
        "factory": spec.expected_forward_hook_factory,
        "checkpoint": str(expected_checkpoint),
        "generation_degeneration_guard_server_abi": (
            spec.expected_generation_degeneration_guard_server_abi
        ),
        "sync_token_ids_across_tp": True,
    }


def _attest_server_info(
    server_info: Mapping[str, Any],
    *,
    spec: RuntimeAttestationSpec,
    expected_checkpoint: Path,
    expected_model_path: Path | None,
    endpoint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    served_model = _nonempty_text(
        server_info.get("served_model_name"),
        field="served_model_name",
    )
    if served_model != spec.expected_served_model:
        raise RuntimeError(f"SGLang {endpoint} served-model mismatch")
    if server_info.get("enable_custom_logit_processor") is not True:
        raise RuntimeError(
            f"SGLang {endpoint} does not enable custom logit processors"
        )
    hook = _attest_hook(
        server_info,
        spec=spec,
        expected_checkpoint=expected_checkpoint,
        endpoint=endpoint,
    )
    model_path = _nonempty_text(server_info.get("model_path"), field="model_path")
    if expected_model_path is not None:
        observed_model_path = _resolved_path(
            model_path,
            field=f"{endpoint} model_path",
            expected=False,
        )
        if observed_model_path != expected_model_path:
            raise RuntimeError(f"SGLang {endpoint} model_path mismatch")
        reported_model_path = str(observed_model_path)
    else:
        reported_model_path = model_path
    runtime = {
        "model_path": reported_model_path,
        "context_length": _positive_integer(
            server_info.get("context_length"), field="context_length"
        ),
        "max_total_num_tokens": _positive_integer(
            server_info.get("max_total_num_tokens"),
            field="max_total_num_tokens",
        ),
    }
    return hook, runtime


def attest_r2eact_runtime(spec: RuntimeAttestationSpec) -> dict[str, Any]:
    """Attest a direct SGLang runtime or raise without returning a partial report.

    Four endpoints are read: ``/health``, the requested compatibility endpoint
    ``/get_server_info``, canonical ``/server_info``, and ``/v1/models``.
    Security-critical fields must agree across the two server-info aliases.
    Only allowlisted, non-secret fields are copied into the success report.
    """

    base_url, expected_checkpoint, expected_model_path = _validate_spec(spec)
    timeout = float(spec.request_timeout_seconds)
    maximum = spec.maximum_response_bytes

    health_status = _health(base_url, timeout=timeout)
    compatibility_info, compatibility_status = _get_json(
        base_url,
        "/get_server_info",
        timeout=timeout,
        maximum_response_bytes=maximum,
    )
    canonical_info, canonical_status = _get_json(
        base_url,
        "/server_info",
        timeout=timeout,
        maximum_response_bytes=maximum,
    )
    models, models_status = _get_json(
        base_url,
        "/v1/models",
        timeout=timeout,
        maximum_response_bytes=maximum,
    )

    compatibility_hook, compatibility_runtime = _attest_server_info(
        compatibility_info,
        spec=spec,
        expected_checkpoint=expected_checkpoint,
        expected_model_path=expected_model_path,
        endpoint="/get_server_info",
    )
    canonical_hook, canonical_runtime = _attest_server_info(
        canonical_info,
        spec=spec,
        expected_checkpoint=expected_checkpoint,
        expected_model_path=expected_model_path,
        endpoint="/server_info",
    )
    if compatibility_hook != canonical_hook:
        raise RuntimeError("SGLang server-info forward-hook aliases disagree")
    if compatibility_runtime != canonical_runtime:
        raise RuntimeError("SGLang server-info runtime metadata aliases disagree")

    model_ids = _model_ids(models)
    if model_ids != [spec.expected_served_model]:
        raise RuntimeError("SGLang /v1/models does not exactly match the served model")

    return {
        "schema": RUNTIME_ATTESTATION_SCHEMA,
        "ok": True,
        "checked_at_unix": time.time(),
        "sglang_url": base_url,
        "endpoints": {
            "/health": {"status": health_status},
            "/get_server_info": {"status": compatibility_status},
            "/server_info": {"status": canonical_status},
            "/v1/models": {"status": models_status},
        },
        "served_model": {
            "expected": spec.expected_served_model,
            "observed_model_ids": model_ids,
        },
        "server": canonical_runtime,
        "forward_hook": {
            "attested": True,
            **canonical_hook,
        },
        "generation_degeneration_guard": {
            "custom_logit_processor_enabled": True,
            "server_abi": spec.expected_generation_degeneration_guard_server_abi,
        },
        "tensor_parallel": {
            "sync_token_ids_across_tp": True,
        },
    }


__all__ = [
    "R2EACT_FORWARD_HOOK_FACTORY",
    "RUNTIME_ATTESTATION_SCHEMA",
    "RuntimeAttestationSpec",
    "attest_r2eact_runtime",
]
