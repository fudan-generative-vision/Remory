#!/usr/bin/env python3
"""Install only this pinned inference engine into a new prefix on the H20 host.

Downloads code/packages/toolchain directly. Never downloads model weights,
installs task runners, claims GPUs, or starts services. Use --dry-run first.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--python", default="python3.12")
    p.add_argument("--cuda", choices=("auto", "cu128", "cu130"), default="auto")
    p.add_argument("--cuda-home", type=Path, help="Use an existing compatible compiler instead of downloading the pinned toolkit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    from recursive_memory.stack import cuda, deploy
    manifest = deploy.verified_payload()
    if args.dry_run:
        print(json.dumps(dict(prefix=str(args.prefix.resolve()), sglang=manifest["sglang"],
            cuda=args.cuda, model_downloads=False, starts_services=False,
            packages="payload/constraints/server.txt + cuda.server_recipe", direct_downloads=True), indent=2))
        return
    # No transfer, including redirected dependency/toolchain downloads, uses VPN.
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            os.environ.pop(name)
    os.environ.update(NO_PROXY="*", no_proxy="*", HF_ENDPOINT="https://hf-mirror.com")
    python = shutil.which(args.python)
    if not python:
        p.error("Python 3.12 with venv and development headers is required")
    deploy.run([python, "-c", "import sys, sysconfig, pathlib; assert sys.version_info[:2] == (3, 12); assert pathlib.Path(sysconfig.get_path('include'), 'Python.h').is_file()"], env=deploy.clean_env())
    build, driver = cuda.select_build(args.cuda, python, env=deploy.clean_env())
    prefix = args.prefix.expanduser().resolve()
    owner = prefix / ".h20-inference-owner.json"
    recipe = dict(payload_sha256=deploy.canonical(manifest), cuda_build=build, python=python,
                  cuda_home=str(args.cuda_home.resolve()) if args.cuda_home else None)
    if prefix.exists() and (not owner.is_file() or deploy.read(owner) != recipe):
        p.error("choose a new prefix; this directory is unmanaged or has a different recipe")
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deploy.atomic(owner, recipe)
        deploy.install_engine(prefix, deploy.PAYLOAD, manifest)
        deploy.install_venv(prefix, "server", python,
            cuda.server_recipe(prefix / "sglang/python", deploy.PAYLOAD / "constraints/server.txt", build))
        cuda_home = args.cuda_home.resolve() if args.cuda_home else cuda.install_toolchain(
            prefix, build, deploy.read(deploy.PAYLOAD / "cuda-toolchains.json"))
        cuda.prepare_headers(prefix, build, cuda_home, deploy.clean_env())
        deploy.atomic(prefix / "h20-engine.json", dict(recipe, cuda_driver_probe=driver,
            cuda_home=str(cuda_home), starts_services=False, model_downloads=False))
    print(str(prefix / "h20-engine.json"))


if __name__ == "__main__":
    main()
