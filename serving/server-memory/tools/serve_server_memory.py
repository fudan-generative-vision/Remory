#!/usr/bin/env python3
"""Stage an isolated engine overlay and serve an existing, verified runtime.

Reuses installed interpreters and weights. Does not install dependencies,
download models, or modify the source engine / existing services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "src/recursive_memory/stack/payload"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True, help="Existing verified runtime.json")
    p.add_argument("--runtime-dir", type=Path, required=True, help="New directory for isolated engine and snapshots")
    p.add_argument("--devices", required=True, help="Allocated physical GPU IDs; TP equals their count")
    p.add_argument("--port", type=int, default=30300)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--token-pool", type=int, default=65536)
    p.add_argument("--max-running-requests", type=int, default=2)
    p.add_argument("--memory-fraction", type=float, default=0.75)
    p.add_argument("--cuda-home", type=Path)
    p.add_argument("--kernel-cache", type=Path)
    p.add_argument("--stage-only", action="store_true")
    args = p.parse_args()
    devices = args.devices.split(",")
    if (not all(d.isdigit() for d in devices) or len(set(devices)) != len(devices)
            or len(devices) not in (1, 2, 4, 8)):
        p.error("devices must be 1, 2, 4 or 8 distinct physical GPU IDs")
    if min(args.context, args.token_pool, args.max_running_requests) < 1 or not 0 < args.memory_fraction < 1:
        p.error("invalid server capacity")
    state = json.loads(args.state.read_text())
    original = Path(state["prefix"]) / "sglang/python"
    runtime = args.runtime_dir.resolve()
    engine = runtime / "sglang-python"
    # Never overlay the existing engine, even if passed an overlapping path.
    if engine == original.resolve() or engine.is_relative_to(original.resolve()) or original.resolve().is_relative_to(runtime):
        p.error("runtime directory must be separate from the source engine")
    manifest = (PAYLOAD / "manifest.json").read_bytes()
    fingerprint = hashlib.sha256(manifest).hexdigest()
    marker = runtime / "source.json"
    if engine.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != {"payload": fingerprint, "engine_source": str(original)}:
            p.error("runtime directory has different/incomplete sources; choose a new directory")
    else:
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copytree(original, engine, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        shutil.copytree(PAYLOAD / "sglang_overlay/python", engine, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        marker.write_text(json.dumps({"payload": fingerprint, "engine_source": str(original)}, indent=2) + "\n")
    if args.stage_only:
        print(engine)
        return
    sys.path[:0] = [str(ROOT / "src"), str(PAYLOAD / "workspace")]
    from recursive_memory.stack.runtime import server_command, occupied
    if occupied(args.port):
        p.error("port already occupied; no existing service will be replaced")
    state.update(workspace=str(PAYLOAD / "workspace"), controller_root=str(ROOT / "src"))
    state.setdefault("served_model_name", "qwen38-remory-stage2-14925")
    state["server"].update(devices=args.devices, tp=len(devices), port=args.port,
        context_length=args.context, max_total_tokens=args.token_pool,
        max_running_requests=args.max_running_requests, memory_fraction=args.memory_fraction, cuda_graph=False)
    os.environ.update(REMORY_SERVER_MEMORY="1", REMORY_MEMORY_DIR=str(runtime / "memory"))
    command, env = server_command(state)
    command.extend(["--max-mamba-cache-size", str(max(8, args.max_running_requests * 4))])
    env.update(PYTHONPATH=os.pathsep.join([str(engine), state["workspace"], state["controller_root"]]),
               HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1",
               SGLANG_MM_FORCE_CPU_TRANSPORT="1", SGLANG_USE_MESSAGE_QUEUE_BROADCASTER="false")
    if args.cuda_home:
        cuda = str(args.cuda_home.resolve())
        env.update(CUDA_HOME=cuda, CUDA_PATH=cuda, TRITON_PTXAS_PATH=cuda + "/bin/ptxas")
        env["PATH"] = cuda + "/bin:" + env["PATH"]
    if args.kernel_cache:
        env["FLASHINFER_WORKSPACE_BASE"] = str(args.kernel_cache.resolve())
    (runtime / "command.json").write_text(json.dumps({"argv": command, "state": state}, indent=2) + "\n")
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
