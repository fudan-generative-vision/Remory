"""Harbor Codex CLI agent backed by the local r^2eAct Responses gateway."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, override

from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from terminal_bench_eval.codex_r2eact_artifacts import (
    CONTROLLER_ARTIFACT_DIRNAME,
)
from terminal_bench_eval.codex_r2eact_gateway import (
    QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY,
    CodexGatewayServer,
    CodexResponsesGateway,
    LiveR2EActBackend,
    MEMORY_GUEST_PATH,
    R2EActGatewayConfig,
    normalize_trusted_mcp_aliases,
    summarize_codex_output_for_trajectory,
)


CODEX_R2EACT_POC_VERSION = "0.1.0"
CODEX_R2EACT_VISION_VERSION = "0.1.0"
CODEX_R2EACT_NATIVE_TOOLS_VERSION = "0.1.0"
CODEX_NATIVE_COMPACT_CANARY_VERSION = "0.1.0"
CODEX_NATIVE_COMPACT_AUTO_LIMIT = 179_200
PINNED_CODEX_VERSION = "0.149.0"
PINNED_CODEX_BINARY = Path(os.environ["R2EACT_STACK_CODEX_BINARY"])
PINNED_CODEX_SHA256 = "bbc3341e44c9ead340ed9570c17be936e37870f570751a941699ffd04d672827"
CODEX_R2EACT_MODEL_SLUG = "r2eact-qwen38-27b-compressor-r16-step1000"
PINNED_CODEX_MODEL_CATALOG = Path(__file__).with_name(
    "codex_r2eact_model_catalog_v0149.json"
)
PINNED_CODEX_MODEL_CATALOG_SHA256 = (
    "d8d3e4dfa6c686cca670de1107c5782b3033df023d742e09e76224cfdc7b74c7"
)
PINNED_CODEX_BUNDLED_MODELS_SHA256 = (
    "8aafe33104f3d423769bdfc1df737e126bc230cb994650e07f237f74eba0a1c4"
)
CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH = (
    "/usr/local/share/r2eact-codex/model-catalog.json"
)


def _normalize_trusted_mcp_servers(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("trusted_mcp_servers must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for server_name, server_config in value.items():
        # Reuse the gateway's name validation while reserving its internal
        # Serper namespace. The alias value is immaterial here.
        normalize_trusted_mcp_aliases({server_name: server_name})
        if not isinstance(server_config, Mapping):
            raise TypeError(
                f"trusted MCP server {server_name!r} config must be a mapping"
            )
        normalized = deepcopy(dict(server_config))
        raw_command = normalized.get("command")
        raw_url = normalized.get("url")
        if raw_command is not None and (
            not isinstance(raw_command, str) or not raw_command.strip()
        ):
            raise ValueError(
                f"trusted MCP server {server_name!r} has an invalid stdio command"
            )
        if raw_url is not None and (
            not isinstance(raw_url, str)
            or not raw_url.startswith(("http://", "https://"))
        ):
            raise ValueError(
                f"trusted MCP server {server_name!r} has an invalid HTTP URL"
            )
        has_command = raw_command is not None
        has_url = raw_url is not None
        if has_command == has_url:
            raise ValueError(
                f"trusted MCP server {server_name!r} must configure exactly "
                "one stdio command or HTTP URL"
            )
        result[str(server_name)] = normalized
    return result


def _build_codex_runtime_config(
    base_url: str,
    *,
    native_compaction: bool = True,
    auto_compact_token_limit: int | None = None,
    web_search_enabled: bool = False,
    mcp_server_url: str | None = None,
    trusted_mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
    server_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not native_compaction:
        raise ValueError("Only Codex-native compact is supported by this release")
    provider_name = "r2eact_local"
    resolved_auto_limit = (
        CODEX_NATIVE_COMPACT_AUTO_LIMIT
        if native_compaction and auto_compact_token_limit is None
        else (
            261_120
            if auto_compact_token_limit is None
            else int(auto_compact_token_limit)
        )
    )
    if not 1 <= resolved_auto_limit < 262_144:
        raise ValueError("Codex auto-compact token limit is outside its context window")
    context_window = 262_144
    if server_info is not None:
        capacities = [server_info.get(name) for name in
                      ("context_length", "max_total_num_tokens", "max_req_input_len")]
        if any(type(value) is not int or value <= 16_384 for value in capacities):
            raise ValueError("Native compact requires valid live SGLang context capacities >16384")
        context_window = min(context_window, *capacities)
        # A 64K worker must compact before admission, not at the 179200-token
        # large-server trigger. Preserve headroom for tools/next generation.
        resolved_auto_limit = min(resolved_auto_limit, context_window - 16_384)
    if web_search_enabled and (
        not isinstance(mcp_server_url, str)
        or not mcp_server_url.startswith(("http://", "https://"))
    ):
        raise ValueError("web search requires an HTTP MCP server URL")
    normalized_mcp_servers = _normalize_trusted_mcp_servers(trusted_mcp_servers)

    config: dict[str, Any] = {
        "model_reasoning_effort": "xhigh",
        "model_provider": provider_name,
        "model_catalog_json": CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH,
        "model_context_window": context_window,
        "model_auto_compact_token_limit": resolved_auto_limit,
        "web_search": "disabled",
        "model_providers": {
            provider_name: {
                # Codex 0.149 gates Remote Compaction V2 on the provider's
                # OpenAI-compatible capability identity.  The provider id,
                # URL, credentials, and static model catalog all remain local.
                "name": (
                    "OpenAI" if native_compaction else "r2eAct local Responses gateway"
                ),
                "base_url": base_url,
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
                "request_max_retries": 0,
                "stream_max_retries": 0,
            }
        },
    }
    mcp_servers = dict(normalized_mcp_servers)
    if web_search_enabled:
        mcp_servers["r2eact_web_search"] = {
            "url": mcp_server_url,
            # Reuse the per-trial gateway credential. The Serper key stays
            # host-side and is never uploaded into the task environment.
            "bearer_token_env_var": "OPENAI_API_KEY",
            "startup_timeout_sec": 30,
            "tool_timeout_sec": 45,
            "required": True,
            "enabled_tools": ["search"],
        }
    if mcp_servers:
        config["mcp_servers"] = mcp_servers
    from shared_runtime.subagents import enabled
    if enabled():
        config["features"] = {"multi_agent": True, "multi_agent_v2": False}
        config["agents"] = {"max_threads": 3, "max_depth": 1}
    return config


def _build_merged_model_catalog(
    codex_binary: Path = PINNED_CODEX_BINARY,
) -> bytes:
    extension_raw = PINNED_CODEX_MODEL_CATALOG.read_bytes()
    extension_sha256 = hashlib.sha256(extension_raw).hexdigest()
    if extension_sha256 != PINNED_CODEX_MODEL_CATALOG_SHA256:
        raise RuntimeError(
            "pinned Codex model catalog checksum mismatch: "
            f"expected {PINNED_CODEX_MODEL_CATALOG_SHA256}, "
            f"observed {extension_sha256}"
        )

    with tempfile.TemporaryDirectory(prefix="r2eact-codex-home-") as codex_home:
        command_env = dict(os.environ)
        command_env["CODEX_HOME"] = codex_home
        completed = subprocess.run(
            [str(codex_binary), "debug", "models", "--bundled"],
            check=False,
            capture_output=True,
            timeout=30,
            env=command_env,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"could not read pinned Codex bundled catalog: {stderr}")
    bundled_sha256 = hashlib.sha256(completed.stdout).hexdigest()
    if bundled_sha256 != PINNED_CODEX_BUNDLED_MODELS_SHA256:
        raise RuntimeError(
            "pinned Codex bundled catalog checksum mismatch: "
            f"expected {PINNED_CODEX_BUNDLED_MODELS_SHA256}, "
            f"observed {bundled_sha256}"
        )

    try:
        bundled = json.loads(completed.stdout)
        extension = json.loads(extension_raw)
        bundled_models = bundled["models"]
        extension_models = extension["models"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("pinned Codex model catalog is malformed") from exc
    if not isinstance(bundled_models, list) or not isinstance(extension_models, list):
        raise RuntimeError("pinned Codex model catalog models must be lists")
    if len(extension_models) != 1:
        raise RuntimeError("r^2eAct model catalog must contain exactly one extension")
    existing_slugs = {
        model.get("slug") for model in bundled_models if isinstance(model, dict)
    }
    extension_slug = extension_models[0].get("slug")
    if extension_slug != CODEX_R2EACT_MODEL_SLUG:
        raise RuntimeError("r^2eAct model catalog extension slug is invalid")
    if extension_slug in existing_slugs:
        raise RuntimeError("r^2eAct model catalog extension collides with Codex")

    bundled_models.extend(extension_models)
    return (
        json.dumps(bundled, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


class CodexR2EActAgent(Codex):
    """Run stock Codex CLI in PRoot while r^2eAct serves model turns."""

    def __init__(
        self,
        *args: Any,
        specforge_root: str | Path,
        checkpoint: str | Path,
        corpus_dir: str | Path,
        sglang_url: str = "http://127.0.0.1:30000",
        codex_binary: str | Path = PINNED_CODEX_BINARY,
        codex_binary_sha256: str = PINNED_CODEX_SHA256,
        gateway_host: str = "127.0.0.1",
        gateway_port: int = 0,
        max_new_tokens: int = 81_920,
        remaining_context_output_ceiling_tokens: int = 81_920,
        working_memory_turns: int = 12,
        serving_eviction_policy: str = (
            "checkpoint_1024_v1"
        ),
        request_timeout: float = 1_800.0,
        capacity_pressure_reserve_tokens: int = 1_024,
        process_lock: str | Path = "/tmp/r2eact-terminal-agent-sglang.lock",
        native_compaction: bool = True,
        native_compaction_text_summary: bool = True,
        residual_compaction: bool = True,
        compressor_enabled: bool = True,
        serving_concurrency: int = 1,
        qwen_reasoning_effort: str | None = "xhigh",
        document_tools_enabled: bool = False,
        vision_enabled: bool = False,
        web_search_enabled: bool = False,
        update_plan_enabled: bool = False,
        trusted_mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        trusted_mcp_aliases: Mapping[str, str] | None = None,
        codex_auto_compact_token_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        if type(compressor_enabled) is not bool or type(residual_compaction) is not bool:
            raise TypeError("compression switches must be boolean")
        if not (native_compaction and native_compaction_text_summary
                and compressor_enabled == residual_compaction
                and qwen_reasoning_effort == "xhigh" and serving_eviction_policy == "checkpoint_1024_v1"):
            raise ValueError("This release supports only native compact + xhigh, with residual or text-only memory")
        from shared_runtime.discrete_concurrency import validate_count
        self._serving_concurrency = validate_count(serving_concurrency)
        if kwargs.get("config") is not None:
            raise ValueError(
                "CodexR2EActAgent owns its dynamic provider config; config must be omitted"
            )
        kwargs["version"] = PINNED_CODEX_VERSION
        kwargs.setdefault("reasoning_effort", None)
        kwargs.setdefault("reasoning_summary", "none")
        if kwargs.get("web_search") not in {None, "disabled"}:
            raise ValueError(
                "Codex provider-native web_search is unsupported; use the "
                "host-side web_search_enabled MCP tool"
            )
        kwargs["web_search"] = "disabled"
        super().__init__(*args, **kwargs)
        self._gateway_config = R2EActGatewayConfig(
            specforge_root=Path(specforge_root).expanduser().resolve(),
            checkpoint=Path(checkpoint).expanduser().resolve(),
            corpus_dir=Path(corpus_dir).expanduser().resolve(),
            sglang_url=str(sglang_url),
            max_new_tokens=int(max_new_tokens),
            remaining_context_output_ceiling_tokens=int(
                remaining_context_output_ceiling_tokens
            ),
            working_memory_turns=int(working_memory_turns),
            serving_eviction_policy=serving_eviction_policy,
            request_timeout_seconds=float(request_timeout),
            capacity_pressure_reserve_tokens=int(capacity_pressure_reserve_tokens),
            process_lock=Path(process_lock),
            native_compaction=bool(native_compaction),
            native_compaction_text_summary=bool(native_compaction_text_summary),
            residual_compaction=bool(residual_compaction),
            compressor_enabled=compressor_enabled,
            # This controls Qwen's real chat template, not the Codex provider's
            # reasoning field (which the local gateway does not use).
            qwen_reasoning_effort=qwen_reasoning_effort,
            vision_enabled=bool(vision_enabled),
            web_search_enabled=bool(web_search_enabled),
            update_plan_enabled=bool(update_plan_enabled),
        )
        self._codex_binary = Path(codex_binary).expanduser().resolve()
        self._document_tools_enabled = bool(document_tools_enabled)
        self._codex_binary_sha256 = str(codex_binary_sha256).lower()
        self._gateway_host = str(gateway_host)
        self._gateway_port = int(gateway_port)
        self._trusted_mcp_servers = _normalize_trusted_mcp_servers(trusted_mcp_servers)
        aliases = (
            {server_name: server_name for server_name in self._trusted_mcp_servers}
            if trusted_mcp_aliases is None
            else trusted_mcp_aliases
        )
        self._trusted_mcp_aliases = normalize_trusted_mcp_aliases(aliases)
        if set(self._trusted_mcp_aliases) != set(self._trusted_mcp_servers):
            raise ValueError(
                "trusted_mcp_aliases keys must exactly match trusted_mcp_servers"
            )
        # Trial snapshots agent.extra_env before run(), so proxy bypasses must
        # be present now rather than added only after the dynamic gateway starts.
        # These names cover both PRoot and LocalDocker gateway URLs.
        for key in ("NO_PROXY", "no_proxy"):
            entries = [
                value.strip()
                for value in str(
                    self._extra_env.get(key) or os.environ.get(key) or ""
                ).split(",")
                if value.strip()
            ]
            for host in (
                "127.0.0.1",
                "localhost",
                "host.docker.internal",
                self._gateway_host,
            ):
                if host and host not in entries:
                    entries.append(host)
            self._extra_env[key] = ",".join(entries)
        self._codex_auto_compact_token_limit = (
            CODEX_NATIVE_COMPACT_AUTO_LIMIT
            if native_compaction and codex_auto_compact_token_limit is None
            else (
                261_120
                if codex_auto_compact_token_limit is None
                else int(codex_auto_compact_token_limit)
            )
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        selected_model = (self.model_name or "").split("/")[-1]
        if selected_model != CODEX_R2EACT_MODEL_SLUG:
            raise RuntimeError(
                "Codex r^2eAct model catalog mismatch: "
                f"expected {CODEX_R2EACT_MODEL_SLUG!r}, observed {selected_model!r}"
            )
        if not self._codex_binary.is_file():
            raise FileNotFoundError(
                f"pinned Codex binary is unavailable: {self._codex_binary}"
            )
        observed = self._sha256(self._codex_binary)
        if observed != self._codex_binary_sha256:
            raise RuntimeError(
                "pinned Codex binary checksum mismatch: "
                f"expected {self._codex_binary_sha256}, observed {observed}"
            )
        if not PINNED_CODEX_MODEL_CATALOG.is_file():
            raise FileNotFoundError(
                "pinned Codex model catalog is unavailable: "
                f"{PINNED_CODEX_MODEL_CATALOG}"
            )
        merged_catalog = _build_merged_model_catalog(self._codex_binary)
        await environment.upload_file(self._codex_binary, "/usr/local/bin/codex")
        await self.exec_as_root(
            environment,
            command=(
                "chmod 0755 /usr/local/bin/codex && "
                "mkdir -p /usr/local/share/r2eact-codex && "
                "/usr/local/bin/codex --version"
            ),
        )
        with tempfile.TemporaryDirectory(prefix="r2eact-codex-catalog-") as temp_dir:
            merged_catalog_path = Path(temp_dir) / "model-catalog.json"
            merged_catalog_path.write_bytes(merged_catalog)
            await environment.upload_file(
                merged_catalog_path,
                CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH,
            )
        await self.exec_as_root(
            environment,
            command=(
                f"chmod 0644 {CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH} && "
                "/usr/local/bin/codex debug models "
                "-c 'model_catalog_json=\""
                f"{CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH}"
                "\"' "
                ">/dev/null"
            ),
        )
        result = await environment.exec(command="/usr/local/bin/codex --version")
        version = self.parse_version(result.stdout or "")
        if result.return_code != 0 or version != PINNED_CODEX_VERSION:
            raise RuntimeError(
                f"guest Codex version mismatch: expected {PINNED_CODEX_VERSION}, "
                f"observed {version!r}"
            )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self._document_tools_enabled:
            return await self._run_with_gateway(instruction, environment, context)
        # Document interpreters stay CPU-only on the host, with only explicitly
        # selected guest documents copied into their isolated staging workspace.
        # Never mount the host runtime/credentials or verifier tree into the guest.
        import asyncio
        from terminal_bench_eval.document_tools import DocumentTools
        rootfs_path = getattr(environment, "_rootfs_path", None)
        if not callable(rootfs_path):
            raise RuntimeError("TB document tools currently require the PRoot guest-file bridge")
        if self._trusted_mcp_servers:
            raise ValueError("document tools own the per-trial trusted MCP configuration")
        documents = DocumentTools(rootfs_path("/"), self.logs_dir / CONTROLLER_ARTIFACT_DIRNAME / "documents")
        old_env = dict(self._extra_env)
        try:
            await asyncio.to_thread(documents.start)
            self._document_services = documents
            self._trusted_mcp_servers = documents.configs()
            self._trusted_mcp_aliases = {name: name for name in self._trusted_mcp_servers}
            await self._run_with_gateway(instruction, environment, context)
        finally:
            await asyncio.to_thread(documents.close)
            self._trusted_mcp_servers = {}
            self._trusted_mcp_aliases = {}
            self._document_services = None
            self._extra_env.clear()
            self._extra_env.update(old_env)

    async def _run_with_gateway(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        rootfs_path = getattr(environment, "_rootfs_path", None)
        unbounded_wire_exec = getattr(environment, "unbounded_wire_exec", None)
        docker_gateway_host = getattr(environment, "r2eact_host_gateway", None)
        docker_memory_mirror = getattr(environment, "r2eact_memory_mirror_path", None)
        docker_mode = (
            not callable(rootfs_path)
            and bool(getattr(environment.capabilities, "mounted", False))
            and docker_gateway_host == "host.docker.internal"
            and isinstance(docker_memory_mirror, Path)
        )
        if callable(rootfs_path) and callable(unbounded_wire_exec):
            memory_target = Path(rootfs_path(MEMORY_GUEST_PATH))
            memory_target.parent.mkdir(parents=True, exist_ok=True)
            memory_target.write_bytes(b"")
            memory_target.chmod(0o444)
            memory_mirror_path: Path | None = memory_target
            command_scope = unbounded_wire_exec
            gateway_bind_host = self._gateway_host
            gateway_advertised_host: str | None = None
            gateway_token = secrets.token_urlsafe(32)
        elif docker_mode:
            # LocalDockerR2EActEnvironment owns a private host directory and
            # bind-mounts the whole directory read-only at /__r2eact__.  The
            # backend can atomically replace the host file while the guest
            # cannot unlink it or bypass the policy through /logs/agent.
            memory_target = docker_memory_mirror
            if (
                not memory_target.is_file()
                or memory_target.name != "memory.jsonl"
                or memory_target.is_symlink()
            ):
                raise RuntimeError("Docker r^2eAct memory mirror is not staged safely")
            memory_target.chmod(0o444)
            memory_mirror_path = memory_target
            command_scope = contextlib.nullcontext
            gateway_bind_host = "0.0.0.0"
            gateway_advertised_host = str(docker_gateway_host)
            # Binding on all interfaces is required for the Docker bridge, so
            # use a per-trial credential rather than exposing the model gateway
            # to unrelated processes that can reach the host port.
            gateway_token = secrets.token_urlsafe(32)
        else:
            raise RuntimeError(
                "Codex r^2eAct requires either the custom PRoot provider or "
                "the mounted LocalDockerR2EActEnvironment"
            )

        if self._document_tools_enabled:
            # Harbor snapshots extra_env before run(). Reuse the ephemeral
            # provider credential that Codex.run explicitly passes to the guest;
            # a new run-time-only environment variable would not reach its MCPs.
            self._document_services.set_token(gateway_token)
        controller_artifact_dir = self.logs_dir / CONTROLLER_ARTIFACT_DIRNAME
        if controller_artifact_dir.is_symlink():
            raise RuntimeError("Codex r^2eAct controller artifact directory is unsafe")
        controller_artifact_dir.mkdir(parents=True, exist_ok=True)
        if not controller_artifact_dir.is_dir():
            raise RuntimeError(
                "Codex r^2eAct controller artifact path is not a directory"
            )
        controller_artifact_dir.chmod(0o700)

        backend_type = LiveR2EActBackend
        if self._gateway_config.residual_compaction or not self._gateway_config.compressor_enabled:
            from shared_runtime.residual_backend import ResidualBackend
            backend_type = ResidualBackend
        backend = backend_type(
            self._gateway_config,
            # The live guest cannot see this hidden host-owned controller
            # directory.  A provider may copy agent logs after execution has
            # ended, but that cannot expose them to the model while it runs.
            controller_artifact_dir,
            memory_mirror_path=memory_mirror_path,
        )
        backend.serving_concurrency = self._serving_concurrency
        gateway_kwargs: dict[str, Any] = {
            "native_compaction": self._gateway_config.native_compaction,
            "working_memory_turns": self._gateway_config.working_memory_turns,
            "vision_enabled": self._gateway_config.vision_enabled,
        }
        if self._gateway_config.native_compaction_text_summary:
            gateway_kwargs["native_compaction_text_summary"] = True
        if self._trusted_mcp_aliases:
            gateway_kwargs["trusted_mcp_aliases"] = self._trusted_mcp_aliases
        if self._gateway_config.web_search_enabled:
            gateway_kwargs.update(
                {
                    "web_search_enabled": True,
                    "serper_api_key": os.environ.get("SERPER_API_KEY"),
                }
            )
        if self._gateway_config.update_plan_enabled:
            gateway_kwargs["update_plan_enabled"] = True
        from shared_runtime.subagents import gateway_class
        gateway = gateway_class()(
            backend, controller_artifact_dir, **gateway_kwargs
        )
        server = CodexGatewayServer(
            gateway,
            host=gateway_bind_host,
            port=self._gateway_port,
            bearer_token=gateway_token,
        )
        old_config = self._base_config
        old_extra_env = dict(self._extra_env)
        try:
            from shared_runtime.residual_preflight import get_server_info
            live_server_info = get_server_info(self._gateway_config.sglang_url)
            server.start()
            gateway_base_url = (
                server.base_url_for_host(gateway_advertised_host)
                if gateway_advertised_host is not None
                else server.base_url
            )
            gateway_mcp_url: str | None = None
            if self._gateway_config.web_search_enabled:
                gateway_mcp_url = (
                    server.mcp_url_for_host(gateway_advertised_host)
                    if gateway_advertised_host is not None
                    else server.mcp_url
                )
            self._base_config = _build_codex_runtime_config(
                gateway_base_url,
                native_compaction=self._gateway_config.native_compaction,
                auto_compact_token_limit=self._codex_auto_compact_token_limit,
                web_search_enabled=self._gateway_config.web_search_enabled,
                mcp_server_url=gateway_mcp_url,
                trusted_mcp_servers=self._trusted_mcp_servers,
                server_info=live_server_info,
            )
            if self._document_tools_enabled:
                from terminal_bench_eval.document_tools import INSTRUCTIONS
                self._base_config["developer_instructions"] = INSTRUCTIONS
            self._extra_env["OPENAI_API_KEY"] = gateway_token
            # Shadow any shell-global OpenAI endpoint so the dynamically selected
            # provider remains the sole model route inside the PRoot.
            self._extra_env["OPENAI_BASE_URL"] = ""
            # Codex is one long guest command containing many model/tool rounds.
            # Harbor owns cancellation of the whole agent phase (including the
            # pinned TB2.1 task-level budget); a second command-layer deadline
            # would race that authoritative boundary and obscure provenance.
            with command_scope():
                await super().run(instruction, environment, context)
        finally:
            server.close()
            self._base_config = old_config
            self._extra_env.clear()
            self._extra_env.update(old_extra_env)

    @override
    def version(self) -> str:
        arm = "residual-c2320" if self._gateway_config.compressor_enabled else "text-only"
        return f"codex-{PINNED_CODEX_VERSION}+native-compact-{arm}-xhigh-v1"

    @staticmethod
    def _parse_output_blob(raw: Any) -> tuple[str | None, dict[str, Any] | None]:
        """Keep image base64 out of Harbor's normalized ATIF trajectory."""

        if isinstance(raw, list):
            return summarize_codex_output_for_trajectory(raw)
        return Codex._parse_output_blob(raw)


class CodexNativeCompactR2EActAgent(CodexR2EActAgent):
    """Isolated canary: Codex V2 selects compact timing/range; Compressor folds it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("native_compaction") is False:
            raise ValueError(
                "CodexNativeCompactR2EActAgent cannot disable native compaction"
            )
        kwargs["native_compaction"] = True
        kwargs.setdefault(
            "codex_auto_compact_token_limit", CODEX_NATIVE_COMPACT_AUTO_LIMIT
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "CODEX_R2EACT_MODEL_CATALOG_GUEST_PATH",
    "CODEX_R2EACT_MODEL_SLUG",
    "CODEX_R2EACT_NATIVE_TOOLS_VERSION",
    "CODEX_R2EACT_VISION_VERSION",
    "CODEX_NATIVE_COMPACT_AUTO_LIMIT",
    "CODEX_NATIVE_COMPACT_CANARY_VERSION",
    "CodexNativeCompactR2EActAgent",
    "CodexR2EActAgent",
    "PINNED_CODEX_BINARY",
    "PINNED_CODEX_BUNDLED_MODELS_SHA256",
    "PINNED_CODEX_MODEL_CATALOG",
    "PINNED_CODEX_MODEL_CATALOG_SHA256",
    "PINNED_CODEX_SHA256",
    "PINNED_CODEX_VERSION",
    "_build_merged_model_catalog",
    "_build_codex_runtime_config",
]
