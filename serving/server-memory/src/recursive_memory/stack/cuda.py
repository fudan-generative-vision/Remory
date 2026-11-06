"""Pinned CUDA wheels and a private JIT toolchain, selected by the CUDA driver."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

DRIVER_PROBE = r"""
import ctypes, json
lib = ctypes.CDLL("libcuda.so.1")
def call(name, *args):
    result = getattr(lib, name)(*args)
    if result:
        raise RuntimeError(f"{name}: CUDA error {result}")
call("cuInit", 0)
count, version = ctypes.c_int(), ctypes.c_int()
call("cuDeviceGetCount", ctypes.byref(count))
call("cuDriverGetVersion", ctypes.byref(version))
rows = []
for index in range(count.value):
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), index)
    call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), device)
    rows.append([major.value, minor.value])
print(json.dumps(dict(driver_cuda=version.value, capabilities=rows)))
"""
KERNEL_CU130 = (
    "https://github.com/sgl-project/whl/releases/download/v0.4.1/"
    "sglang_kernel-0.4.1+cu130-cp310-abi3-manylinux2014_x86_64.whl"
    "#sha256=9164b8fc2c1652a52156f21d1a960116670c29033f686946840ef5b13558c5c0"
)


def select_build(requested, python, *, env, probe=None):
    if requested not in {"auto", "cu128", "cu130"}:
        raise ValueError("CUDA build must be auto, cu128 or cu130")
    if probe is None:

        def probe():
            result = subprocess.run(
                [str(python), "-c", DRIVER_PROBE],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return json.loads(result.stdout)

    try:
        report = probe()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if requested == "auto":
            raise RuntimeError(
                "CUDA driver probe failed. Install on a GPU allocation, or set "
                "runtime.cuda to cu128/cu130 when preparing a GPU-less machine."
            ) from exc
        return requested, {"driver_probe": "unavailable", "selection": "explicit"}
    capabilities = report["capabilities"]
    if not capabilities and requested == "auto":
        raise RuntimeError(
            "No visible CUDA GPUs; set runtime.cuda explicitly for offline preparation"
        )
    needs_13 = any(tuple(capability) >= (10, 3) for capability in capabilities)
    build = ("cu130" if needs_13 else "cu128") if requested == "auto" else requested
    if needs_13 and build != "cu130":
        raise RuntimeError(
            "SM 10.3+ requires the cu130 SGLang kernel/toolchain in this release"
        )
    minimum = 13000 if build == "cu130" else 12080
    if capabilities and report["driver_cuda"] < minimum:
        raise RuntimeError(
            f"The NVIDIA driver cannot execute the selected {build} JIT toolchain"
        )
    return build, report


def server_recipe(engine, constraints, build):
    wheels = [
        f"torch==2.9.1+{build}",
        f"torchvision==0.24.1+{build}",
        f"torchaudio==2.9.1+{build}",
    ]
    kernel = KERNEL_CU130 if build == "cu130" else "sglang-kernel==0.4.1"
    return [
        [*wheels, "--index-url", f"https://download.pytorch.org/whl/{build}"],
        [
            "-c",
            str(constraints),
            "-e",
            str(engine),
            kernel,
            "kernels==0.10.5",
            "dill==0.3.9",
        ],
    ]


def download(url, expected, destination):
    from .deploy import sha

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != expected:
            raise RuntimeError(f"cached download checksum mismatch: {destination.name}")
        return destination
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as out, urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, out)
            if sha(temporary) != expected:
                raise RuntimeError(f"download checksum mismatch: {destination.name}")
            os.replace(temporary, destination)
            return destination
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def install_toolchain(prefix, build, manifest):
    from .deploy import atomic, read, sha

    recipe = manifest[build]
    destination = prefix / f"cuda-{recipe['release']}"
    marker = destination / "stack-toolchain.json"
    if marker.exists():
        receipt = read(marker)
        if receipt["recipe"] != recipe or any(
            sha(destination / name) != digest
            for name, digest in receipt["files"].items()
        ):
            raise RuntimeError(
                "private CUDA toolchain differs from the pinned installation"
            )
        return destination
    if destination.exists():
        raise RuntimeError(f"unmanaged CUDA toolchain: {destination}")
    with tempfile.TemporaryDirectory(prefix=".cuda-", dir=prefix) as name:
        stage = Path(name)
        merged = stage / "merged"
        merged.mkdir()
        for component, entry in recipe["components"].items():
            print(
                f"[stack] verified CUDA component: {component} {entry['version']}",
                flush=True,
            )
            archive = download(
                "https://developer.download.nvidia.com/compute/cuda/redist/"
                + entry["relative_path"],
                entry["sha256"],
                prefix / "downloads/cuda" / Path(entry["relative_path"]).name,
            )
            unpacked = stage / component
            with tarfile.open(archive) as stream:
                stream.extractall(unpacked, filter="data")
            roots = list(unpacked.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError("unexpected NVIDIA component layout")
            shutil.copytree(roots[0], merged, dirs_exist_ok=True, symlinks=True)
        if not (merged / "lib64").exists():
            (merged / "lib64").symlink_to("lib", target_is_directory=True)
        files = {
            str(p.relative_to(merged)): sha(p) for p in merged.rglob("*") if p.is_file()
        }
        atomic(merged / marker.name, {"recipe": recipe, "files": files})
        merged.rename(destination)
    return destination


HEADER_SETUP = r"""
import base64, hashlib, json, sys, sysconfig
from importlib.metadata import distribution
from pathlib import Path
home, build = Path(sys.argv[1]), sys.argv[2]
name, version, include, library = (
    ("nvidia-curand", "10.4.0.35", "nvidia/cu13/include/", "nvidia/cu13/lib/libnvrtc.so.13")
    if build == "cu130" else
    ("nvidia-curand-cu12", "10.3.9.90", "nvidia/curand/include/", "nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
)
package = distribution(name)
if package.version != version:
    raise RuntimeError("unexpected cuRAND version: " + package.version)
headers = [f for f in package.files if str(f).startswith(include) and str(f).endswith(".h")]
if not any(f.name == "curand.h" for f in headers):
    raise RuntimeError("cuRAND headers are missing")
receipt = {"package": name, "version": version, "headers": {}}
for entry in headers:
    source = Path(package.locate_file(entry)).resolve()
    digest = hashlib.sha256(source.read_bytes()).digest()
    if not entry.hash or entry.hash.mode != "sha256" or base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
        raise RuntimeError("cuRAND wheel RECORD mismatch")
    relative = Path(str(entry)).relative_to(include)
    target = home / "include" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError("conflicting cuRAND header: " + str(relative))
    else:
        target.symlink_to(source)
    receipt["headers"][str(relative)] = digest.hex()
site = Path(sysconfig.get_path("purelib"))
nvrtc = site / library
if not nvrtc.is_file():
    raise RuntimeError("NVRTC is missing from the serving environment")
(site / "recursive_memory_nvrtc.pth").write_text(
    "import ctypes; ctypes.CDLL(" + repr(str(nvrtc)) + ", mode=ctypes.RTLD_GLOBAL)\n"
)
print(json.dumps(receipt))
"""


def prepare_headers(prefix, build, home, env):
    from .deploy import atomic

    result = subprocess.run(
        [str(prefix / "server/bin/python"), "-c", HEADER_SETUP, str(home), build],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    atomic(prefix / "receipts/curand-headers.json", json.loads(result.stdout))


def runtime_environment(state, env):
    if state.get("cuda_home"):
        home = Path(state["cuda_home"])
        env.update(
            CUDA_HOME=str(home),
            CUDA_PATH=str(home),
            TRITON_PTXAS_PATH=str(home / "bin/ptxas"),
            FLASHINFER_WORKSPACE_BASE=str(
                Path(state["prefix"]) / "kernel-cache" / state["cuda_build"]
            ),
        )
        env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", os.defpath)
    return env
