"""Install the pinned SGLang runtime and its CUDA compiler components."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv

from toolchain import install_toolchain, prepare_headers

ROOT = Path(__file__).resolve().parents[1]
KERNEL130 = ("https://github.com/sgl-project/whl/releases/download/v0.4.7/"
    "sglang_kernel-0.4.7+cu130-cp310-abi3-manylinux2014_x86_64.whl"
    "#sha256=d312128f33e95dec2f7e9f6b7f4295d092e430cb9ef3e07a3a970e0741c7023e")


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def cuda_build(requested):
    lib = ctypes.CDLL("libcuda.so.1")
    def call(name, *args):
        if getattr(lib, name)(*args):
            raise RuntimeError(f"{name} failed; run on a machine with a visible NVIDIA GPU")
    call("cuInit", 0)
    count, driver = ctypes.c_int(), ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    call("cuDriverGetVersion", ctypes.byref(driver))
    if count.value == 0:
        raise RuntimeError("no NVIDIA GPUs are visible")
    build = "cu130" if requested == "auto" else requested
    if build != "cu130":
        raise RuntimeError("the shared Qwen/GLM SGLang runtime requires CUDA 13.0")
    if driver.value < 13000:
        raise RuntimeError(f"NVIDIA driver is too old for {build}")
    return build


def setup_module():
    spec = importlib.util.spec_from_file_location("sglang_setup", ROOT / "src/remory/backends/sglang_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(state, build):
    state.mkdir(parents=True, exist_ok=True)
    setup = setup_module()
    recipe = setup.recipe()
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()
        + (ROOT / "pyproject.toml").read_bytes() + (ROOT / "deploy/constraints.txt").read_bytes()
        + (ROOT / "deploy/cuda-toolchains.json").read_bytes()
        + build.encode()).hexdigest()
    marker = state / "installation.json"
    python = state / "venv/bin/python"
    source = state / "sglang"
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt["fingerprint"] != fingerprint:
            raise RuntimeError("deployment recipe changed; use a new --runtime-dir for this revision")
        setup.verify_installation(source / "python/sglang")
        return python, Path(receipt["cuda_home"])
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(state / "venv")
    if not source.exists():
        with tempfile.TemporaryDirectory(prefix=".sglang-", dir=state) as tmp:
            staging = Path(tmp) / "source"
            run(["git", "init", "-q", staging])
            run(["git", "-C", staging, "remote", "add", "origin", recipe["repository"]])
            run(["git", "-C", staging, "fetch", "--depth", "1", "origin", recipe["commit"]])
            run(["git", "-C", staging, "checkout", "--detach", "FETCH_HEAD"])
            staging.rename(source)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != recipe["commit"]:
        raise RuntimeError("unexpected SGLang source revision")
    setup.patch_installation(source / "python/sglang")
    env = {**os.environ, "PIP_CONFIG_FILE": os.devnull, "PIP_EXTRA_INDEX_URL": "",
           "SETUPTOOLS_SCM_PRETEND_VERSION": recipe["version"],
           "SGLANG_BUILD_RUST_EXTS": "none"}
    pip = [python, "-m", "pip", "install"]
    print(f"[Remory] Installing the pinned {build} runtime", flush=True)
    # Select CUDA wheels here; resolve their dependencies from PyPI below.
    run([*pip, "--no-deps", f"torch==2.13.0+{build}", f"torchvision==0.28.0+{build}",
         f"torchaudio==2.11.0+{build}", "--index-url", f"https://download.pytorch.org/whl/{build}"], env=env)
    run([*pip, "--index-url", "https://pypi.org/simple", "-c", ROOT / "deploy/constraints.txt",
         "-e", source / "python", KERNEL130,
         "-e", str(ROOT) + "[server,transformers,codex]", "dill==0.4.1"], env=env)
    cuda = install_toolchain(state, build)
    prepare_headers(python, cuda, build)
    marker.write_text(json.dumps({"fingerprint": fingerprint, "sglang": recipe["commit"],
                                 "cuda_home": str(cuda), "cuda_build": build}, indent=2) + "\n")
    return python, cuda


def prepare_runtime(runtime_dir, cuda):
    if sys.version_info < (3, 11) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("deployment requires Linux x86_64 and Python 3.11+")
    for binary in ("git", "c++"):
        if not shutil.which(binary):
            raise RuntimeError(f"install {binary} before deploying")
    state = Path(runtime_dir).expanduser().resolve()
    python, _ = install(state, cuda_build(cuda))
    return state, python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".remory")
    parser.add_argument("--cuda", choices=["auto", "cu130"], default="auto")
    args = parser.parse_args()
    state, python = prepare_runtime(args.runtime_dir, args.cuda)
    print(f"Runtime ready: {python}")
    print(f"source {shlex.quote(str(state / 'venv/bin/activate'))}")
    print("python -m remory.launch_server --help")


if __name__ == "__main__":
    main()
