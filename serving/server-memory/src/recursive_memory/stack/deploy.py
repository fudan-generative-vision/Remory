"""Install isolated environments and verified sources; never alter old pools."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import urlopen

from .gpu import SUPPORTED_TP, validate_devices

PAYLOAD = Path(__file__).with_name("payload")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WEIGHT_SHA = "c2c6efde58350ddb9fb98c7bdaa0e8bb0576be46581751ecd70cc198e67a1764"
CONFIG_SHA = "2218a16a137e07217268b6d29f88f966d76d3e1e903e1a00b0bfe7418e545f2f"
POLICY = "checkpoint_1024_v1"
DATASET = "terminal-bench/terminal-bench-2-1"
DATASET_REF = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
TOOLS = {
    "codex": (
        "https://github.com/openai/codex/releases/download/rust-v0.149.0/codex-x86_64-unknown-linux-musl.tar.gz",
        "codex-x86_64-unknown-linux-musl",
        "bbc3341e44c9ead340ed9570c17be936e37870f570751a941699ffd04d672827",
    ),
    "rg": (
        "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz",
        "rg",
        "e62198eb19b136b88c330af83647b5a962cb99b6b1f066758568f12de1974849",
    ),
    "proot": (
        "https://github.com/proot-me/proot/releases/download/v5.4.1/proot",
        None,
        "19f44283f5c0e73091c60195f5fcd4f4c1165505e44410d434e2ab1b677c1a09",
    ),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe artifact path: {name!r}")
    return path


def verified_payload(root=PAYLOAD):
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != "recursive_memory_eval_source_v1":
        raise ValueError("unsupported runtime source manifest")
    for name, entry in manifest["files"].items():
        path = root / relative(name)
        if (
            any(p.is_symlink() for p in [path, *path.parents])
            or sha(path) != entry["sha256"]
        ):
            raise ValueError(f"source checksum/symlink mismatch: {name}")
    for name in manifest["sglang"]["removed"]:
        relative(name)
    return manifest


def validate_source(value):
    if not isinstance(value, dict):
        raise ValueError("model source must be an object")
    if set(value) == {"path"}:
        if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
            raise ValueError("local weight paths must be absolute")
    elif set(value) == {"repo_id", "revision"}:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", value["repo_id"]):
            raise ValueError("HF repo_id must be owner/repository, not a URL or token")
        if not re.fullmatch(r"[a-f0-9]{40}", value["revision"]):
            raise ValueError(
                "HF revisions must be full immutable commit SHAs, not main/latest"
            )
    else:
        raise ValueError("weight source requires either path or repo_id + revision")
    return value


def load_config(path):
    config = read(path)
    if (
        set(config) - {"schema", "target", "compressor", "server", "runtime", "downloads"}
        or config.get("schema") != "recursive_memory_stack_v1"
    ):
        raise ValueError(
            "unsupported deployment config; credentials belong in the environment"
        )
    for name in ("target", "compressor"):
        validate_source(config[name])
    runtime = {"cuda": "auto", "document_sandbox": "auto", **config.get("runtime", {})}
    if (set(runtime) != {"cuda", "document_sandbox"}
        or runtime["cuda"] not in {"auto", "cu128", "cu130"}
        or runtime["document_sandbox"] not in {"auto", "landlock", "bubblewrap"}):
        raise ValueError("invalid CUDA build or document sandbox selection")
    downloads = {"endpoint": "https://huggingface.co", "direct": False, **config.get("downloads", {})}
    origin = urlsplit(downloads["endpoint"])
    if (set(downloads) != {"endpoint", "direct"} or type(downloads["direct"]) is not bool
        or origin.scheme != "https" or not origin.hostname or origin.username or origin.password
        or origin.path not in {"", "/"} or origin.query or origin.fragment):
        raise ValueError("downloads requires a credential-free HTTPS origin and boolean direct")
    server = {
        "port": 30000,
        "devices": "auto",
        "tp": "auto",
        "memory_fraction": "auto",
        "max_running_requests": 1,
        "context_length": "auto",
        "max_total_tokens": "auto",
        "cuda_graph": "auto",
        "overlap_schedule": False,
        "piecewise_cuda_graph": False,
        "chunked_prefill_size": -1,
        **config.get("server", {}),
    }
    if set(server) != {
        "port",
        "devices",
        "tp",
        "memory_fraction",
        "max_running_requests",
        "context_length",
        "max_total_tokens",
        "cuda_graph",
        "overlap_schedule",
        "piecewise_cuda_graph",
        "chunked_prefill_size",
    }:
        raise ValueError("unknown server option")
    if (
        type(server["port"]) is not int
        or not 1024 <= server["port"] <= 65535
        or (
            server["tp"] != "auto"
            and (type(server["tp"]) is not int or server["tp"] not in SUPPORTED_TP)
        )
        or (
            server["memory_fraction"] != "auto"
            and (
                type(server["memory_fraction"]) not in {float, int}
                or not 0.1 <= server["memory_fraction"] <= 0.9
            )
        )
    ):
        raise ValueError(
            "requires auto or TP1/2/4/8, a valid memory fraction and loopback port"
        )
    if (
        any(
            server[key] != "auto" and (type(server[key]) is not int or server[key] < 1)
            for key in ("context_length", "max_total_tokens")
        )
        or type(server["max_running_requests"]) is not int
        or not 1 <= server["max_running_requests"] <= 32
        or (server["cuda_graph"] != "auto" and type(server["cuda_graph"]) is not bool)
        or type(server["overlap_schedule"]) is not bool
        or type(server["piecewise_cuda_graph"]) is not bool
        or type(server["chunked_prefill_size"]) is not int
        or (server["chunked_prefill_size"] != -1 and server["chunked_prefill_size"] <= 0)
    ):
        raise ValueError(
            "server concurrency must be 1..32 and context/token limits positive integers"
        )
    validate_devices(server["devices"])
    if (
        server["devices"] != "auto"
        and server["tp"] != "auto"
        and len(server["devices"].split(",")) != server["tp"]
    ):
        raise ValueError("device count must match configured TP")
    return {**config, "server": server, "runtime": runtime, "downloads": downloads}


def safe_prefix(path):
    prefix = Path(path).expanduser().absolute()
    if prefix.resolve() in {
        Path("/"),
        Path.home().resolve(),
        Path.cwd().resolve(),
        PACKAGE_ROOT.resolve(),
    }:
        raise ValueError(
            "choose a dedicated installation directory, not a home/workspace root"
        )
    if any(p.is_symlink() for p in [prefix, *prefix.parents]):
        raise ValueError("installation prefix must not traverse symlinks")
    return prefix


def run(argv, *, env=None, cwd=None, timeout=None):
    # Never print environment values: credentials/proxy auth remain private.
    print("[stack] " + " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, env=env, cwd=cwd, timeout=timeout)


def clean_env():
    env = dict(os.environ)
    # Ignore unrelated system-wide pip indexes/constraints. An explicitly
    # supplied PIP_CONFIG_FILE still wins for sites with their own package mirror.
    env.setdefault("PIP_CONFIG_FILE", os.devnull)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "PIP_CONSTRAINT",
        "UV_CONSTRAINT",
    ):
        env.pop(key, None)
    return env


def load_tool_env(path):
    """Read literal KEY=value pairs; never source a credentials file as code."""
    pending = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ")
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"invalid --env-file syntax at line {number}")
            if key.strip() not in {"SERPER_API_KEY", "MINERU_API_KEY"}:
                continue
            try:
                tokens = shlex.split(value, comments=True, posix=True)
                if len(tokens) > 1:
                    raise ValueError()
            except ValueError:
                raise ValueError(f"invalid --env-file syntax at line {number}") from None
            pending[key.strip()] = tokens[0] if tokens else ""
    os.environ.update(pending)


def stage_sources(prefix, manifest):
    generation = prefix / "releases" / canonical(manifest)[:16]
    marker = generation / "source.json"
    if marker.exists():
        if read(marker) != manifest:
            raise RuntimeError("existing source generation has a conflicting manifest")
        for name, item in manifest["files"].items():
            if sha(generation / name) != item["sha256"]:
                raise RuntimeError(
                    f"installed source was edited: {name}; use a new prefix"
                )
        return generation
    generation.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".source-", dir=generation.parent))
    for name, entry in manifest["files"].items():
        destination = staging / relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAYLOAD / name, destination)
        destination.chmod(0o755 if entry["executable"] else 0o644)
    atomic(staging / "source.json", manifest)
    staging.chmod(0o755)
    staging.rename(generation)
    return generation


def controller_sources():
    """Do not put a bootstrap venv's entire site-packages on runtime PYTHONPATH."""
    package = Path(__file__).resolve().parents[1]
    names = ["__init__.py", *sorted("stack/" + p.name for p in (package / "stack").glob("*.py"))]
    return {name: (package / name).read_bytes() for name in names}


def controller_manifest():
    return {
        name: hashlib.sha256(data).hexdigest()
        for name, data in controller_sources().items()
    }


def stage_controller(prefix):
    sources, manifest = controller_sources(), controller_manifest()
    directory = prefix / "controller-src" / canonical(manifest)[:16]
    if directory.exists():
        for name, expected in manifest.items():
            if sha(directory / "recursive_memory" / name) != expected:
                raise RuntimeError("installed controller source was edited")
        return directory
    directory.parent.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".controller-", dir=directory.parent))
    for name, data in sources.items():
        file = staging / "recursive_memory" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    staging.chmod(0o755)
    atomic(staging / "manifest.json", manifest)
    staging.rename(directory)
    return directory


def install_tools(prefix):
    directory = prefix / "bin"
    directory.mkdir(exist_ok=True)
    for name, (url, member_name, expected) in TOOLS.items():
        destination = directory / name
        if destination.exists():
            if not destination.is_symlink() and sha(destination) == expected:
                continue
            raise RuntimeError(
                f"refusing to replace a different executable: {destination}"
            )
        # Accept only the exact published binary, whether local or downloaded.
        local = shutil.which(name)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=prefix) as temp:
            payload = Path(temp) / "download"
            if local and sha(local) == expected:
                shutil.copyfile(local, payload)
            else:
                with urlopen(url, timeout=120) as response, payload.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if member_name:
                    with tarfile.open(payload) as archive:
                        members = [
                            m
                            for m in archive.getmembers()
                            if m.isfile() and PurePosixPath(m.name).name == member_name
                        ]
                        if len(members) != 1:
                            raise RuntimeError(f"unexpected {name} release archive")
                        extracted = Path(temp) / "binary"
                        with (
                            archive.extractfile(members[0]) as src,
                            extracted.open("wb") as out,
                        ):
                            shutil.copyfileobj(src, out)
                        payload = extracted
            if sha(payload) != expected:
                raise RuntimeError(f"{name} binary checksum mismatch; not installed")
            payload.chmod(0o755)
            payload.rename(destination)


def environment(state):
    prefix = Path(state["prefix"])
    workspace = Path(state["workspace"])
    env = clean_env()
    from .cuda import runtime_environment
    runtime_environment(state, env)
    sandbox = state.get("document_sandbox", {"backend": "landlock"})
    env["R2EACT_DOCUMENT_SANDBOX"] = sandbox["backend"]
    if sandbox.get("binary"):
        env["R2EACT_BWRAP_BINARY"] = sandbox["binary"]
    env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    env.update(
        {
            "PATH": f"{prefix / 'controller/bin'}:{prefix / 'bin'}:{env.get('PATH', os.defpath)}",
            "PYTHONPATH": os.pathsep.join(
                map(
                    str,
                    [
                        workspace,
                        state.get("controller_root", prefix / "controller-src"),
                    ],
                )
            ),
            "SPECFORGE_ROOT": str(workspace),
            "R2EACT_CORPUS_DIR": str(prefix / "empty-corpus"),
            "R2EACT_STACK_CODEX_BINARY": str(prefix / "bin/codex"),
            "R2EACT_STACK_RG_BINARY": str(prefix / "bin/rg"),
            "R2EACT_DOCUMENT_VENV": str(prefix / "documents"),
            "R2EACT_LOCAL_RUNTIME": (
                f"/tmp/r2eact-documents-{os.getuid()}-{canonical(str(prefix))[:12]}"
            ),
            "AUTOMATIONBENCH_RUNTIME_ROOT": str(prefix / "runs/automationbench"),
            "JOBBENCH_MONITOR_JOBS_ROOT": str(prefix / "runs/jobbench/jobs"),
            "BROWSECOMP_CODEX_MONITOR_JOBS_ROOT": str(prefix / "runs/browsecomp100/jobs"),
            "HARBOR_PROOT_BINARY": str(prefix / "bin/proot"),
            "R2EACT_STACK_STATE": str(prefix / "installed.json"),
            "R2EACT_TB_MONITOR_JOBS_ROOT": str(prefix / "runs/tb21/jobs"),
            "R2EACT_TB_MONITOR_POOL_PYTHON": str(prefix / "controller/bin/python"),
            "HARBOR_SHARED_OCI_CACHE": str(prefix / "runs/tb21/shared-oci-cache"),
            "AUTOMATIONBENCH_MONITOR_JOBS_ROOT": str(
                prefix / "runs/automationbench/jobs"
            ),
        }
    )
    # The engine is local and must not be reached via the web egress proxy.
    bypass = list(
        dict.fromkeys(
            [
                *env.get("NO_PROXY", "").split(","),
                *env.get("no_proxy", "").split(","),
                "localhost",
                "127.0.0.1",
                "::1",
            ]
        )
    )
    env["NO_PROXY"] = env["no_proxy"] = ",".join(p for p in bypass if p)
    # Don't inherit unrelated experiment switches, native-compaction modes or
    # source-job migration knobs from the caller's old worker environment.
    for key in list(env):
        if key.startswith(("TERMINAL_BENCH_POOL_", "R2EACT_EVAL_")):
            env.pop(key)
    return env


def install_venv(prefix, name, python, commands):
    directory = prefix / name
    recipe = {"python": str(python), "commands": commands}
    marker = directory / "stack-ready.json"
    if marker.exists():
        if read(marker) != recipe:
            raise RuntimeError(f"{name} recipe changed; install into a new prefix")
        return
    if not directory.exists():
        run([python, "-m", "venv", directory], env=clean_env())
    for args in commands:
        run([directory / "bin/python", "-m", "pip", "install", *args], env=clean_env())
    run([directory / "bin/python", "-m", "pip", "check"], env=clean_env())
    atomic(marker, recipe)


def download_source(source, controller, destination, settings=None):
    if "path" in source:
        path = Path(source["path"]).resolve()
        if not path.is_dir():
            raise ValueError(f"weight directory unavailable: {path}")
        return path
    # No credentials in argv/files; huggingface_hub reads HF_TOKEN normally.
    settings = settings or {"endpoint": "https://huggingface.co", "direct": False}
    env = clean_env()
    env["HF_ENDPOINT"] = settings["endpoint"]
    if settings["direct"]:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        env.update(NO_PROXY="*", no_proxy="*", HF_HUB_DISABLE_XET="1")
    code = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
        "endpoint=sys.argv[4], headers={'User-Agent': 'curl/8.5.0'}, max_workers=4)"
    )
    run(
        [controller, "-c", code, source["repo_id"], source["revision"], destination, settings["endpoint"]],
        env=env,
    )
    return destination


def relocate_checkpoint(source, target, destination):
    """Keep weights immutable; relocate only target_model in a derived export."""
    config, manifest = read(source / "config.json"), read(source / "manifest.json")
    if canonical(config) != CONFIG_SHA or manifest.get("config_sha256") != CONFIG_SHA:
        raise ValueError("this release requires the reviewed residual step2320 config")
    if (
        manifest.get("compressor_file") != "compressor.pt"
        or manifest.get("compressor_sha256") != WEIGHT_SHA
        or sha(source / "compressor.pt") != WEIGHT_SHA
    ):
        raise ValueError("residual step2320 weight checksum mismatch")
    for name, digest in config["summary_contract"]["tokenizer_files"].items():
        if sha(target / relative(name)) != digest:
            raise ValueError(
                f"target tokenizer differs from the training contract: {name}"
            )
    config = {**config, "target_model": str(target)}
    derived_manifest = {**manifest, "config_sha256": canonical(config)}
    if destination.exists():
        if (
            read(destination / "config.json") != config
            or read(destination / "manifest.json") != derived_manifest
            or sha(destination / "compressor.pt") != WEIGHT_SHA
        ):
            raise ValueError(
                "existing deployed checkpoint changed; refusing to overwrite"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        os.link(source / "compressor.pt", staging / "compressor.pt")
    except OSError:
        shutil.copyfile(source / "compressor.pt", staging / "compressor.pt")
    atomic(staging / "config.json", config)
    atomic(staging / "manifest.json", derived_manifest)
    atomic(
        staging / "relocation.json",
        {
            "source_config_sha256": CONFIG_SHA,
            "derived_config_sha256": canonical(config),
            "source": str(source),
            "changed_fields": ["target_model"],
            "weights_sha256": WEIGHT_SHA,
        },
    )
    staging.chmod(0o755)
    for name in ("config.json", "manifest.json", "relocation.json"):
        (staging / name).chmod(0o644)
    staging.rename(destination)


def install_engine(prefix, generation, manifest):
    engine = prefix / "sglang"
    marker = engine / ".stack-source.json"
    recipe = {"base": manifest["sglang"], "source": canonical(manifest)}
    if marker.exists():
        if read(marker) != recipe:
            raise RuntimeError(
                "SGLang recipe changed; choose a new installation prefix"
            )
        for name, item in manifest["files"].items():
            if (
                name.startswith("sglang_overlay/")
                and sha(engine / name.removeprefix("sglang_overlay/")) != item["sha256"]
            ):
                raise RuntimeError("installed SGLang overlay was edited")
        return
    if engine.exists():
        raise RuntimeError("unmanaged SGLang directory; it was not modified")
    # Publish only a complete checkout+overlay. An interrupted network fetch
    # leaves no half-initialized engine that poisons the next installation.
    with tempfile.TemporaryDirectory(prefix=".engine-", dir=prefix) as temporary:
        staging = Path(temporary) / "sglang"
        run(["git", "init", staging])
        run(
            [
                "git",
                "-C",
                staging,
                "remote",
                "add",
                "origin",
                manifest["sglang"]["repository"],
            ]
        )
        run(
            [
                "git",
                "-C",
                staging,
                "fetch",
                "--depth=1",
                "origin",
                manifest["sglang"]["base_commit"],
            ]
        )
        run(["git", "-C", staging, "checkout", "--detach", "FETCH_HEAD"])
        head = subprocess.check_output(
            ["git", "-C", str(staging), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != manifest["sglang"]["base_commit"]:
            raise RuntimeError(
                "SGLang checkout differs from the pinned upstream commit"
            )
        for name in manifest["files"]:
            if name.startswith("sglang_overlay/"):
                destination = staging / relative(name.removeprefix("sglang_overlay/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generation / name, destination)
        for name in manifest["sglang"]["removed"]:
            (staging / relative(name)).unlink(missing_ok=True)
        atomic(staging / ".stack-source.json", recipe)
        staging.rename(engine)


def install(config_file, prefix, *, system_packages=False, dry_run=False, target_path=None):
    config = load_config(config_file)
    if target_path is not None:
        config["target"] = {"path": str(Path(target_path).expanduser().resolve())}
    prefix = safe_prefix(prefix)
    manifest = verified_payload()
    if dry_run:
        print(
            json.dumps(
                {
                    "prefix": str(prefix),
                    "config": config,
                    "source_sha256": canonical(manifest),
                    "plan": [
                        "isolated Python environments",
                        "pinned Codex/rg/PRoot",
                        "pinned target + step2320 weights",
                        "patched SGLang",
                        "official AutomationBench 1.0.6 runtime",
                        "read-only doctor",
                    ],
                    "starts_services": False,
                    "claims_tasks": False,
                },
                indent=2,
            )
        )
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "serving/evaluation deployment requires Linux x86_64 (not macOS)"
        )
    for name in ("target", "compressor"):
        if "path" in config[name] and not Path(config[name]["path"]).is_dir():
            raise ValueError(
                f"{name} directory is unavailable; edit the deployment config before installing"
            )
    python = shutil.which("python3.12")
    if not python:
        raise RuntimeError("install Python 3.12 with venv support first")
    from . import cuda, sandbox
    build, driver = cuda.select_build(config["runtime"]["cuda"], python, env=clean_env())
    owner = prefix / ".stack-owner.json"
    if prefix.exists() and not owner.exists() and any(prefix.iterdir()):
        raise ValueError(
            "nonempty unmanaged prefix; existing data will not be overwritten"
        )
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not owner.exists():
            atomic(owner, {"schema": "recursive_memory_stack_v1"})
        contract = {
            "config": config,
            "source": canonical(manifest),
            "controller": canonical(controller_manifest()),
            "cuda_build": build,
        }
        pin = prefix / "deployment.json"
        if pin.exists() and read(pin) != contract:
            raise ValueError("immutable installation differs; use a new prefix")
        atomic(pin, contract)
        missing = [
            name
            for name in ("git", "gcc", "g++", "skopeo", "umoci", "curl", "flock", "dpkg-deb", "numactl")
            if not shutil.which(name)
        ]
        if system_packages:
            if os.geteuid() != 0 or not shutil.which("apt-get"):
                raise RuntimeError("--system-packages requires root on Debian/Ubuntu")
            run(
                [
                    python,
                    PAYLOAD / "workspace/terminal_bench_eval/host_apt_bootstrap.py",
                    "git",
                    "build-essential",
                    "python3.12-dev",
                    "skopeo",
                    "umoci",
                    "curl",
                    "util-linux",
                    "numactl",
                    "ca-certificates",
                    "file",
                    "xxd",
                    "libglib2.0-0",
                    "libseccomp2",
                    "libcap2",
                    "libselinux1",
                ]
            )
        elif missing:
            raise RuntimeError(
                f"missing host packages: {', '.join(missing)}; install them or use --system-packages"
            )
        run(
            [python, "-c", "import pathlib, sysconfig; "
             "header = pathlib.Path(sysconfig.get_path('include')) / 'Python.h'; "
             "assert header.is_file(), 'Python 3.12 development headers are required; install python3.12-dev or use --system-packages'"],
            env=clean_env(),
        )
        generation = stage_sources(prefix, manifest)
        controller_root = stage_controller(prefix)
        install_tools(prefix)
        document_sandbox = sandbox.install(prefix, config["runtime"]["document_sandbox"])
        install_venv(
            prefix,
            "controller",
            python,
            [
                [
                    "torch==2.9.1+cpu",
                    "torchvision==0.24.1+cpu",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                [
                    "-c", str(generation / "constraints/controller.txt"),
                    "harbor==0.21.0",
                    "httpx==0.28.1",
                    "transformers==5.3.0",
                    "huggingface_hub==1.28.0",
                    "safetensors==0.6.2",
                    "dill==0.3.9",
                    "numpy==1.26.4",
                    "pyarrow==19.0.1",
                    "streamlit==1.49.1",
                    "PyYAML==6.0.2",
                    "pytest==8.4.2",
                ],
            ],
        )
        controller = prefix / "controller/bin/python"
        target = download_source(
            config["target"], controller, prefix / "weights/target", config["downloads"]
        )
        source = download_source(
            config["compressor"], controller, prefix / "weights/compressor-source", config["downloads"]
        )
        checkpoint = prefix / "weights/residual-step2320"
        relocate_checkpoint(source, target, checkpoint)
        install_engine(prefix, generation, manifest)
        install_venv(
            prefix,
            "server",
            python,
            cuda.server_recipe(prefix / "sglang/python", generation / "constraints/server.txt", build),
        )
        cuda_home = cuda.install_toolchain(prefix, build, read(generation / "cuda-toolchains.json"))
        cuda.prepare_headers(prefix, build, cuda_home, clean_env())
        state = {
            "schema": "recursive_memory_stack_v1",
            "prefix": str(prefix),
            "workspace": str(generation / "workspace"),
            "controller_root": str(controller_root),
            "target": str(target),
            "checkpoint": str(checkpoint),
            "server": config["server"],
            "source_sha256": canonical(manifest),
            "config": config,
            "cuda_build": build,
            "cuda_home": str(cuda_home),
            "cuda_driver_probe": driver,
            "document_sandbox": document_sandbox,
        }
        (prefix / "empty-corpus").mkdir(exist_ok=True)
        (prefix / "empty-corpus/wiki_corpus.jsonl").touch(exist_ok=True)
        env = environment(state)
        run(["bash", generation / "workspace/shared_runtime/setup_runtime.sh"], env=env)
        run(
            ["bash", generation / "workspace/automation_bench_eval/setup_runtime.sh"],
            env=env,
        )
        # Record resolved transitive packages, including currently unpinned
        # document extras. This release does not pretend to be a full wheel lock.
        for name in ("controller", "server", "documents"):
            freeze = subprocess.check_output(
                [str(prefix / name / "bin/python"), "-m", "pip", "freeze"],
                env=clean_env(),
                text=True,
            )
            atomic(
                prefix / "receipts" / f"{name}-packages.json",
                {"pip_freeze": freeze.splitlines()},
            )
        atomic(prefix / "installed.json", state)
        run(
            [controller, "-m", "recursive_memory.stack.runtime", "doctor"],
            env=env,
            cwd=state["workspace"],
        )
        print(
            f"Installed. No server or evaluation was started by the installation phase. Prefix: {prefix}"
        )


def load_state(prefix):
    state = read(Path(prefix).expanduser().resolve() / "installed.json")
    if state.get("schema") != "recursive_memory_stack_v1":
        raise ValueError("unsupported installed runtime")
    if Path(state["prefix"]).resolve() != Path(prefix).expanduser().resolve():
        raise ValueError(
            "installation moved; reinstall to regenerate absolute runtime contracts"
        )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    setup = subs.add_parser("install")
    setup.add_argument("--config", type=Path, required=True)
    setup.add_argument("--prefix", type=Path, required=True)
    setup.add_argument("--target-path", type=Path, help="reuse local target weights instead of downloading Qwen")
    setup.add_argument("--system-packages", action="store_true")
    setup.add_argument("--dry-run", action="store_true")
    setup.add_argument(
        "--smoke",
        action="store_true",
        help="after install, start/reuse SGLang and run synthetic smoke",
    )
    for name in ("doctor", "server", "smoke", "run", "monitor", "chat", "verify", "judge", "node", "gateway", "check-sandbox", "check-kernels"):
        command = subs.add_parser(name)
        command.add_argument("--prefix", type=Path, required=True)
        command.add_argument("--env-file", type=Path)
    args, rest = parser.parse_known_args()
    if args.command == "install":
        if rest:
            parser.error(f"unrecognized installation options: {rest}")
        install(
            args.config,
            args.prefix,
            system_packages=args.system_packages,
            dry_run=args.dry_run,
            target_path=args.target_path,
        )
        if args.smoke and not args.dry_run:
            state = load_state(args.prefix)
            run(
                [
                    Path(state["prefix"]) / "controller/bin/python",
                    "-m",
                    "recursive_memory.stack.runtime",
                    "smoke",
                    "--start-server",
                ],
                env=environment(state),
                cwd=state["workspace"],
            )
    else:
        if args.env_file:
            load_tool_env(args.env_file)
        state = load_state(args.prefix)
        argv = [
            str(Path(state["prefix"]) / "controller/bin/python"),
            "-m",
            ("recursive_memory.stack." + args.command) if args.command in {"node", "gateway"} else "recursive_memory.stack.runtime",
            *([] if args.command in {"node", "gateway"} else [args.command]),
            *rest,
        ]
        os.chdir(state["workspace"])
        os.execve(argv[0], argv, environment(state))


if __name__ == "__main__":
    main()
