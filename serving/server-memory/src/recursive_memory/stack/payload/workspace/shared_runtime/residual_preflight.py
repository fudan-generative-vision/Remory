"""Read-only identity admission for the isolated residual Compressor experiment.

No cache flush, explicit /generate, server restart, or task claim is allowed
here. /health may run SGLang's own one-token readiness probe. A healthy
repeat4500/plain-Qwen endpoint is deliberately not a residual endpoint.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import ssl
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from .config import HarnessConfig


RESIDUAL_SERVING_ABI = "qwen38_summary_residual_sglang_v1"
RESIDUAL_MEMORY_CAPABILITY = "qwen38_residual_bounded_prefill_memory_v1"
RESIDUAL_GDN_CAPABILITY = "qwen38_residual_bounded_gdn_prefill_v1"
ALL_PREFILL_GDN_CAPABILITY = "qwen38_bounded_gdn_all_prefill_v2"
ALL_PREFILL_MLP_CAPABILITY = "qwen38_bounded_mlp_all_prefill_v2"
RESIDUAL_RECURSIVE_CAPABILITY = "qwen38_recursive_summary_residual_v1"
RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY = "qwen38_residual_dynamic_summary_v1"
RESIDUAL_CONCURRENCY_CAPABILITY = "qwen38_residual_cpu_concurrent_v1"
RESIDUAL_HOOK_FACTORY = (
    "context_compression.qwen38_residual_sglang_hook:make_summary_residual_compressor_hook"
)


def assert_residual_concurrency(info, count):
    """Reject shared soft-memory clients without request isolation/capacity."""
    from .discrete_concurrency import validate_count
    validate_count(count)
    capabilities = info.get("request_capabilities")
    limit = info.get("max_running_requests")
    if (not isinstance(capabilities, list) or RESIDUAL_CONCURRENCY_CAPABILITY not in capabilities
        or type(limit) is not int or limit < count):
        raise RuntimeError(
            f"{count} residual task slots require patched residual SGLang with "
            f"MAX_RUNNING_REQUESTS>={count}; no task claimed. "
            "Configure a compatible server on an idle endpoint."
        )


def load_checkpoint_contract(checkpoint: Path) -> dict:
    specforge = Path(os.environ.get("SPECFORGE_ROOT", os.environ.get(
        "TERMINAL_BENCH_POOL_SPECFORGE_ROOT",
        "/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge",
    )))
    if str(specforge) not in sys.path:
        sys.path.insert(0, str(specforge))
    module = importlib.import_module("context_compression.qwen38_residual_serving_contract")
    return module.load_residual_serving_contract(checkpoint, verify_weights=False)


def _read_server_info(url: str, timeout: float) -> dict:
    from context_compression.sglang_http import open_request
    with open_request(url, timeout=timeout) as response:
        payload = response.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise RuntimeError("residual SGLang server metadata exceeds the admission limit")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise RuntimeError("residual SGLang server metadata must be an object")
    return result


def _read_health(url: str, timeout: float) -> None:
    # Readiness is local to the explicitly configured model origin. Never send
    # it through the web proxy or accept a redirect/login page as readiness.
    from context_compression.sglang_http import open_request
    with open_request(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"SGLang health returned unexpected HTTP {response.status}")


def _bounded_get_attempt(url, timeout, cancel_event, *, reader, kind):
    # urllib's socket timeout is an inactivity timeout, not an overall deadline.
    # Bound the caller's wait (including DNS/body reads) and allow prompt Ctrl+C
    # cleanup. The daemon only performs a read-only GET, never installs state;
    # a late reply after timeout/cancellation is discarded.
    done = threading.Event()
    outcome = []

    def read():
        try:
            outcome.append((True, reader(url, timeout)))
        except BaseException as exc:
            outcome.append((False, exc))
        finally:
            done.set()

    deadline = time.monotonic() + timeout
    threading.Thread(target=read, name=f"sglang-{kind}-read", daemon=True).start()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError(f"residual {kind} admission cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"SGLang {kind} request timed out")
        if done.wait(min(0.1, remaining)):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError(f"residual {kind} admission cancelled")
            ok, value = outcome[0]
            if ok:
                return value
            raise value


def _retryable_metadata_error(exc):
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, URLError):
        return _retryable_metadata_error(exc.reason)
    if isinstance(exc, ssl.SSLError):
        return False  # Never retry away a TLS/certificate validation failure.
    return isinstance(exc, OSError)


def _get_with_retry(sglang_url: str, *, reader, path, kind, cancel_event=None,
                    total_timeout=180.0, request_timeout=30.0,
                    retry_backoff=1.0):
    """Bound an origin GET by attempts, wall time, and shutdown cancellation."""
    parsed = urlsplit(sglang_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not re.fullmatch(r"(?:/groups/[A-Za-z0-9_.:-]{1,128})?(?:/backends/[A-Za-z0-9_.:-]{1,128})?/?", parsed.path)):
        raise ValueError("residual SGLang URL must be a credential-free HTTP(S) origin or shared gateway route")
    # The SGLang endpoint is explicitly supplied by the task platform, not a
    # web tool URL. Do not send its private server metadata through a VPN proxy.
    if (not all(math.isfinite(v) for v in (total_timeout, request_timeout, retry_backoff))
        or total_timeout <= 0 or request_timeout <= 0 or retry_backoff < 0):
        raise ValueError(f"invalid SGLang {kind} retry timing")
    url = sglang_url.rstrip("/") + path
    deadline = time.monotonic() + total_timeout
    last_error = None
    for attempt in range(1, 7):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError(f"residual {kind} admission cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return _bounded_get_attempt(url, min(request_timeout, remaining), cancel_event,
                                        reader=reader, kind=kind)
        except Exception as exc:
            if not _retryable_metadata_error(exc):
                raise
            last_error = exc
            remaining = deadline - time.monotonic()
            if attempt == 6 or remaining <= 0:
                break
            logging.getLogger(__name__).warning(
                "SGLang %s temporarily unavailable (%s); retry %d/6, %.1fs remaining",
                kind, type(exc).__name__, attempt + 1, remaining,
            )
            delay = min(retry_backoff * 2 ** (attempt - 1), remaining)
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    raise RuntimeError(f"residual {kind} admission cancelled") from exc
            else:
                time.sleep(delay)
    raise TimeoutError(f"SGLang live {kind} admission exhausted its retry budget") from last_error


def get_server_info(sglang_url: str, *, cancel_event=None,
                    total_timeout=180.0, request_timeout=30.0,
                    retry_backoff=1.0) -> dict:
    """Read live identity with bounded transport retries, never stale metadata."""
    return _get_with_retry(sglang_url, reader=_read_server_info,
        path="/get_server_info", kind="metadata", cancel_event=cancel_event,
        total_timeout=total_timeout, request_timeout=request_timeout,
        retry_backoff=retry_backoff)


def wait_for_health(sglang_url: str, *, cancel_event=None,
                    total_timeout=180.0, request_timeout=30.0,
                    retry_backoff=1.0) -> None:
    """Require HTTP 200 readiness; transient failure is not an agent answer.

    SGLang's /health defaults to a generated probe with a 20-second scheduler
    response window. Busy prefill can outlast it even while metadata is healthy.
    Retry only transport/temporary HTTP errors, at most six attempts / 180s.
    Metadata/checkpoint attestation must still run after readiness succeeds.
    """
    _get_with_retry(sglang_url, reader=_read_health, path="/health", kind="health",
        cancel_event=cancel_event, total_timeout=total_timeout,
        request_timeout=request_timeout, retry_backoff=retry_backoff)


def assert_paired_server(config: HarnessConfig, sglang_url: str) -> dict:
    """Both memory arms use the same attested model, hook and summary export."""
    config.validate()
    admission = HarnessConfig(**{name: getattr(config, name)
                                 for name in HarnessConfig.__dataclass_fields__})
    if not admission.residual_compaction:
        from dataclasses import replace
        admission = replace(admission, residual_compaction=True, compressor_enabled=True)
    return assert_residual_server(admission, sglang_url)


def assert_residual_server(config: HarnessConfig, sglang_url: str) -> dict:
    config.validate()
    if not config.residual_compaction:
        raise ValueError("residual preflight requires the residual experiment config")
    checkpoint = Path(config.residual_checkpoint).resolve()
    contract = load_checkpoint_contract(checkpoint)
    if contract.get("deployment_export", {}).get("source_global_step") != 14925:
        raise RuntimeError("this evaluation requires the fixed Stage-II checkpoint 14925")
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    from context_compression.remory_public_release import RELEASE
    from context_compression.remory_identity import artifact_identity
    allowed = {'c9bfdf6f2f383d6905969506aecd71764bdf1b25265da84945296230365bc0f4',
               RELEASE['compressor_sha256']}
    identity = artifact_identity(checkpoint, contract)
    if (manifest.get('compressor_sha256') not in allowed
            or identity['contract_sha256'] != 'ea4598529c83cd14447c967210f62d357c304104e76d58f08421c4e854a2b3c3'
            or identity['target_metadata_sha256'] != '68dca6ab72fe8b44b54228bb020909c54ef960d6d8a779fbeff72bcdaa27e08c'):
        raise RuntimeError('checkpoint differs from the pinned Stage-II-14925 / online-250 releases')
    residual, summary = contract.get("residual", {}), contract.get("summary_contract", {})
    if (residual.get("block_tokens"), residual.get("compression_ratio"), residual.get("memory_budget"),
            summary.get("target_summary_tokens"), summary.get("max_summary_tokens")) != (1024, 16, 4096, 8192, 8192):
        raise RuntimeError("Stage-II-14925 requires trained 1024/64 blocks, 4096 soft tokens, and 8192 summary metadata")
    info = get_server_info(sglang_url)
    capabilities = info.get("request_capabilities") or []
    if not isinstance(capabilities, list) or RESIDUAL_SERVING_ABI not in capabilities:
        raise RuntimeError("SGLang scheduler does not attest the residual encode/cache-bypass ABI; no task was claimed")
    if RESIDUAL_MEMORY_CAPABILITY not in capabilities:
        raise RuntimeError(
            "Residual SGLang lacks the long-history OOM fix (bounded MLP + CPU feature staging). "
            "Restart the SGLang server, not only the worker; no task was claimed."
        )
    if config.harness_mode == "native_compact" and RESIDUAL_GDN_CAPABILITY not in capabilities:
        raise RuntimeError(
            "Native residual compact requires bounded GDN prefill to prevent long-history CUDA OOM. "
            "Restart the SGLang server, not only the worker; no task was claimed."
        )
    if config.harness_mode == "native_compact" and not all(
        cap in capabilities for cap in (ALL_PREFILL_GDN_CAPABILITY, ALL_PREFILL_MLP_CAPABILITY)
    ):
        raise RuntimeError(
            "Native residual compact requires bounded GDN + dense MLP for ALL prefills "
            "(generation, summary and encode); the encode-only OOM patch is insufficient. "
            "Restart the SGLang server, not only the worker; no task was claimed."
        )
    if RESIDUAL_RECURSIVE_CAPABILITY not in capabilities:
        raise RuntimeError(
            "Residual SGLang lacks recursive compact (previous summary + soft memory + new raw tail). "
            "Restart the SGLang server, not only the worker; no task was claimed."
        )
    if RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY not in capabilities:
        raise RuntimeError(
            "Residual SGLang still has fixed summary caps. "
            "Restart the SGLang server, not only the worker; no task was claimed."
        )
    hooks = info.get("forward_hooks") or []
    if not isinstance(hooks, list):
        raise RuntimeError("SGLang did not report a valid residual forward-hook list")
    matching = [hook for hook in hooks if isinstance(hook, dict)
                and hook.get("hook_factory") == RESIDUAL_HOOK_FACTORY]
    if len(matching) != 1:
        raise RuntimeError(
            "SGLang is healthy but is not the residual Compressor server. No task was claimed; "
            "use a free dedicated endpoint or explicitly replace the old server yourself."
        )
    settings = matching[0].get("config")
    if not isinstance(settings, dict) or settings.get("residual_serving_abi") != RESIDUAL_SERVING_ABI:
        raise RuntimeError("SGLang residual Compressor hook ABI does not match this experiment")
    reported = settings.get("checkpoint")
    from context_compression.remory_identity import check_server_identity
    portable = check_server_identity(info, checkpoint, contract)
    if not portable and (not isinstance(reported, str) or not Path(reported).is_absolute() or Path(reported).resolve() != checkpoint):
        raise RuntimeError("SGLang residual Compressor checkpoint differs from the immutable pool pin")
    if not portable and info.get("model_path") != contract.get("target_model"):
        raise RuntimeError("SGLang target model differs from the residual training target")
    concurrency = info.get("max_running_requests")
    concurrent = RESIDUAL_CONCURRENCY_CAPABILITY in capabilities
    if type(concurrency) is not int or concurrency < 1 or (concurrency > 1 and not concurrent):
        raise RuntimeError("residual concurrent serving requires qwen38_residual_cpu_concurrent_v1; restart the patched SGLang server")
    return {"residual_serving_abi": RESIDUAL_SERVING_ABI,
            "recursive_compact_capability": RESIDUAL_RECURSIVE_CAPABILITY,
            "dynamic_summary_capability": RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
            "prefill_memory_capability": RESIDUAL_MEMORY_CAPABILITY,
            "bounded_gdn_prefill_capability": RESIDUAL_GDN_CAPABILITY if RESIDUAL_GDN_CAPABILITY in capabilities else None,
            "all_prefill_gdn_capability": ALL_PREFILL_GDN_CAPABILITY if ALL_PREFILL_GDN_CAPABILITY in capabilities else None,
            "all_prefill_mlp_capability": ALL_PREFILL_MLP_CAPABILITY if ALL_PREFILL_MLP_CAPABILITY in capabilities else None,
            "checkpoint": str(checkpoint),
            "source_global_step": 14925,
            "online_step": 250 if manifest['compressor_sha256'] == RELEASE['compressor_sha256'] else None,
            "max_running_requests": concurrency,
            "compressor_sha256": manifest['compressor_sha256'],
            "config_sha256": manifest['config_sha256'],
            "concurrency_capability": RESIDUAL_CONCURRENCY_CAPABILITY if concurrent else None}
