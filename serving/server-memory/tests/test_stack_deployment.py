"""CPU-only packaging/ownership tests. No GPUs, APIs, installs or pool claims."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from recursive_memory.stack import deploy as d
from recursive_memory.stack import runtime as r


@pytest.fixture
def config():
    return {
        "schema": "recursive_memory_stack_v1",
        "target": {"path": "/models/target"},
        "compressor": {"path": "/models/compressor"},
    }


@pytest.fixture
def state(tmp_path):
    return {
        "schema": "recursive_memory_stack_v1",
        "prefix": str(tmp_path / "installation"),
        "workspace": str(tmp_path / "installation/releases/source/workspace"),
        "target": "/models/target",
        "checkpoint": "/models/derived-compressor",
        "source_sha256": "a" * 64,
        "server": {"port": 31000, "devices": "0,1", "tp": 2, "memory_fraction": 0.55},
    }


def test_config_is_strict_and_secret_free(tmp_path, config):
    path = tmp_path / "config.json"
    d.atomic(path, config)
    assert d.load_config(path)["server"]["tp"] == "auto"
    d.atomic(path, {**config, "SERPER_API_KEY": "not-allowed"})
    with pytest.raises(ValueError, match="credentials"):
        d.load_config(path)


def test_tp8_concurrent_server_config(tmp_path, config):
    path = tmp_path / "4090.json"
    server = {
        "tp": 8,
        "devices": "0,1,2,3,4,5,6,7",
        "memory_fraction": 0.72,
        "max_running_requests": 2,
        "context_length": 65536,
        "max_total_tokens": 65536,
    }
    d.atomic(path, {**config, "server": server})
    result = d.load_config(path)["server"]
    assert result["tp"] == 8 and result["max_running_requests"] == 2
    for change in (
        {"devices": "0,1"},
        {"max_running_requests": True},
        {"max_total_tokens": 0},
    ):
        d.atomic(path, {**config, "server": {**server, **change}})
        with pytest.raises(ValueError):
            d.load_config(path)


@pytest.mark.parametrize("revision", ["main", "latest", "v1", "1234", "A" * 40])
def test_download_requires_immutable_revision(revision):
    with pytest.raises(ValueError, match="immutable"):
        d.validate_source({"repo_id": "org/model", "revision": revision})


@pytest.mark.parametrize(
    "name", ["../escape", "/tmp/escape", "a/../../b", "a\\b", "a//b", ""]
)
def test_artifact_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        d.relative(name)


def test_packaged_payload_checksums_and_portable_binaries():
    manifest = d.verified_payload()
    assert len(manifest["files"]) > 100
    for filename, variable in [
        ("codex_r2eact_agent.py", "R2EACT_STACK_CODEX_BINARY"),
        ("codex_r2eact_gateway.py", "R2EACT_STACK_RG_BINARY"),
    ]:
        source = (d.PAYLOAD / "workspace/terminal_bench_eval" / filename).read_text()
        assert f'os.environ["{variable}"]' in source


def test_dry_run_writes_nothing(tmp_path, config, capsys):
    path, prefix = tmp_path / "config.json", tmp_path / "not-created"
    d.atomic(path, config)
    d.install(path, prefix, dry_run=True)
    assert not prefix.exists()
    report = json.loads(capsys.readouterr().out)
    assert not report["claims_tasks"] and not report["starts_services"]


def test_sources_are_idempotent_and_tampering_is_rejected(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    (payload / "workspace").mkdir(parents=True)
    (payload / "workspace/example.py").write_text("x = 1\n")
    manifest = {
        "files": {
            "workspace/example.py": {
                "sha256": d.sha(payload / "workspace/example.py"),
                "executable": False,
            }
        }
    }
    monkeypatch.setattr(d, "PAYLOAD", payload)
    first = d.stage_sources(tmp_path / "install", manifest)
    assert d.stage_sources(tmp_path / "install", manifest) == first
    assert first.stat().st_mode & 0o005 == 0o005
    (first / "workspace/example.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="edited"):
        d.stage_sources(tmp_path / "install", manifest)


def test_checkpoint_relocation_changes_only_target(tmp_path, monkeypatch):
    source, target, destination = (
        tmp_path / name for name in ("source", "target", "derived")
    )
    source.mkdir()
    target.mkdir()
    (source / "compressor.pt").write_bytes(b"test-weights")
    (target / "tokenizer.json").write_bytes(b"test-tokenizer")
    config = {
        "target_model": "/original/location",
        "summary_contract": {
            "tokenizer_files": {"tokenizer.json": d.sha(target / "tokenizer.json")}
        },
    }
    manifest = {
        "compressor_file": "compressor.pt",
        "compressor_sha256": d.sha(source / "compressor.pt"),
        "config_sha256": d.canonical(config),
    }
    d.atomic(source / "config.json", config)
    d.atomic(source / "manifest.json", manifest)
    monkeypatch.setattr(d, "CONFIG_SHA", d.canonical(config))
    monkeypatch.setattr(d, "WEIGHT_SHA", manifest["compressor_sha256"])
    d.relocate_checkpoint(source, target, destination)
    assert d.read(source / "config.json") == config
    assert d.read(destination / "config.json") == {
        **config,
        "target_model": str(target),
    }
    assert d.sha(destination / "compressor.pt") == manifest["compressor_sha256"]
    d.relocate_checkpoint(source, target, destination)
    (target / "tokenizer.json").write_bytes(b"wrong-tokenizer")
    with pytest.raises(ValueError, match="tokenizer"):
        d.relocate_checkpoint(source, target, destination)


def test_tool_download_does_not_extract_tar_paths(tmp_path, monkeypatch):
    content = b"binary"
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo("../../outside/rg")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(
        d,
        "TOOLS",
        {
            "rg": (
                "https://example.invalid/rg.tgz",
                "rg",
                hashlib.sha256(content).hexdigest(),
            )
        },
    )
    monkeypatch.setattr(d.shutil, "which", lambda name: None)
    monkeypatch.setattr(d, "urlopen", lambda *a, **kw: io.BytesIO(data.getvalue()))
    d.install_tools(tmp_path)
    assert (tmp_path / "bin/rg").read_bytes() == content
    assert not (tmp_path.parent / "outside").exists()


def test_wrong_binary_never_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        d, "TOOLS", {"proot": ("https://example.invalid/proot", None, "0" * 64)}
    )
    monkeypatch.setattr(d.shutil, "which", lambda name: None)
    monkeypatch.setattr(d, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong"))
    with pytest.raises(RuntimeError, match="checksum"):
        d.install_tools(tmp_path)
    assert not (tmp_path / "bin/proot").exists()


def test_environment_does_not_inherit_old_experiment_switches(state, monkeypatch):
    monkeypatch.setenv("R2EACT_EVAL_MEMORY_PROFILE", "old-wm4")
    monkeypatch.setenv("TERMINAL_BENCH_POOL_SERVER_COMMAND", "old-engine")
    monkeypatch.setenv("SERPER_API_KEY", "test-private-key")
    before_home = d.os.environ.get("HOME")
    env = d.environment(state)
    assert "R2EACT_EVAL_MEMORY_PROFILE" not in env
    assert "TERMINAL_BENCH_POOL_SERVER_COMMAND" not in env
    assert env["SERPER_API_KEY"] == "test-private-key"
    assert env.get("HOME") == before_home
    assert "test-private-key" not in json.dumps(state)
    assert str(d.PACKAGE_ROOT) not in env["PYTHONPATH"].split(":")


def test_controller_source_does_not_include_bootstrap_site_packages(tmp_path):
    directory = d.stage_controller(tmp_path)
    actual = {str(p.relative_to(directory / "recursive_memory"))
              for p in (directory / "recursive_memory").rglob("*.py")}
    assert actual == set(d.controller_sources())
    assert {"stack/node.py", "stack/cuda.py", "stack/validation.py"} <= actual
    assert not (directory / "torch").exists()
    assert d.stage_controller(tmp_path) == directory


def test_foreign_attested_server_is_not_stopped(state, monkeypatch):
    monkeypatch.setattr(r, "occupied", lambda port: True)
    monkeypatch.setattr(r, "admit_server", lambda state: {"status": "ok"})
    monkeypatch.setattr(
        r, "stop_owned", lambda child: pytest.fail("foreign server was stopped")
    )
    with r.serving(state, start=True) as process:
        assert process is None


def test_foreign_wrong_checkpoint_is_not_replaced(state, monkeypatch):
    monkeypatch.setattr(r, "occupied", lambda port: True)

    def wrong(state):
        raise RuntimeError("checkpoint mismatch")

    monkeypatch.setattr(r, "admit_server", wrong)
    monkeypatch.setattr(
        r.subprocess, "Popen", lambda *a, **k: pytest.fail("replacement spawned")
    )
    with pytest.raises(RuntimeError, match="checkpoint"):
        with r.serving(state, start=True):
            pass


def test_empty_port_requires_explicit_server_start(state, monkeypatch):
    monkeypatch.setattr(r, "occupied", lambda port: False)
    with pytest.raises(RuntimeError, match="start-server"):
        with r.serving(state):
            pass


@pytest.mark.parametrize("benchmark", ["automationbench", "tb21"])
def test_queue_command_routing_and_once(state, benchmark):
    args = argparse.Namespace(
        benchmark=benchmark, task=[], job=None, once=True, worker_id="worker-test"
    )
    prepare, worker = r.evaluation_commands(state, args)
    assert prepare.count("--job-name") == worker.count("--job-name") == 1
    assert worker.count("worker") == 1
    assert "--worker-id" in worker
    if benchmark == "automationbench":
        assert "--prepare-only" in prepare
        assert worker[-4:-2] == ["--max-tasks", "1"]
    else:
        assert prepare[-3:] == ["--attempts", "1", "--all"]
        assert "--once" in worker and "recursive_memory.stack.preflight" in worker
        profile = d.read(prepare[prepare.index("--config") + 1])
        assert profile["n_attempts"] == 1
        kwargs = profile["agents"][0]["kwargs"]
        assert (
            kwargs["working_memory_turns"] == 12
            and kwargs["serving_eviction_policy"] == d.POLICY
        )
        assert all(
            kwargs[key]
            for key in (
                "residual_compaction",
                "document_tools_enabled",
                "vision_enabled",
                "web_search_enabled",
                "update_plan_enabled",
            )
        )


def test_cannot_replace_frozen_profile(state):
    args = argparse.Namespace(
        benchmark="tb21", task=[], job="fixed", once=False, worker_id=None
    )
    prepare, _ = r.evaluation_commands(state, args)
    config = Path(prepare[prepare.index("--config") + 1])
    changed = d.read(config)
    changed["n_attempts"] = 5
    d.atomic(config, changed)
    with pytest.raises(RuntimeError, match="immutable"):
        r.evaluation_commands(state, args)


def test_local_relocation_is_explicit(state, tmp_path):
    copied = tmp_path / "copied"
    copied.mkdir()
    d.atomic(copied / "installed.json", state)
    with pytest.raises(ValueError, match="moved"):
        d.load_state(copied)


def test_performance_server_config(tmp_path, config):
    path = tmp_path / "performance.json"
    server = dict(cuda_graph=True, overlap_schedule=True,
                  piecewise_cuda_graph=True, chunked_prefill_size=8192)
    d.atomic(path, {**config, "server": server})
    assert d.load_config(path)["server"]["chunked_prefill_size"] == 8192
    for change in ({"chunked_prefill_size": 0}, {"chunked_prefill_size": True},
                   {"overlap_schedule": "true"}, {"piecewise_cuda_graph": 1}):
        d.atomic(path, {**config, "server": {**server, **change}})
        with pytest.raises(ValueError):
            d.load_config(path)
