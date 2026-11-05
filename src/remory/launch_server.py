"""Launch SGLang with the Remory model hook and compaction routes."""
import argparse
import json
import os
from pathlib import Path
import sys


def configure_runtime(runtime_dir=None):
    installed = Path(sys.prefix).parent
    state = Path(runtime_dir) if runtime_dir else (
        installed if (installed / "installation.json").is_file() else Path.cwd() / ".remory")
    state = state.expanduser().resolve()
    marker = state / "installation.json"
    if marker.is_file():
        cuda = Path(json.loads(marker.read_text())["cuda_home"])
        if not (cuda / "bin/ptxas").is_file():
            raise RuntimeError("installed CUDA compiler is missing; run python deploy/install.py")
        os.environ.update(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
            TRITON_PTXAS_PATH=str(cuda / "bin/ptxas"),
            FLASHINFER_WORKSPACE_BASE=str(state / "kernel-cache"),
            PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", os.defpath))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--actor", dest="actor", metavar="MODEL_PATH",
        help="actor model ID or local snapshot (default: model recorded in the memory checkpoint)")
    parser.add_argument("--remory-checkpoint", "--checkpoint", dest="checkpoint",
        default="mocoV3/Qwen3.8-27B-REMORY-1.9B", help="memory checkpoint ID or local directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--tp", "--tp-size", "--tensor-parallel-size", dest="tp_size",
        type=int, default=1, help="number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--context-length", "--context-limit", dest="context_limit", type=int, metavar="TOKENS",
        help="context length (default: model maximum)")
    parser.add_argument("--mem-fraction-static", "--memory-fraction", dest="memory_fraction",
        type=float, default=0.8, help="SGLang static GPU memory fraction (default: 0.8)")
    parser.add_argument("--base-gpu-id", "--gpu", dest="gpu", type=int, default=0,
        help="GPU index within CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--store", help="session database (default: <runtime-dir>/memory.sqlite)")
    parser.add_argument("--online-training-dir", type=Path,
        help="Enable Qwen sample collection and encoder versions in this shared local directory")
    parser.add_argument("--runtime-dir", type=Path,
        help="runtime directory (default: detected from the installed environment)")
    args = parser.parse_args(argv)
    state = configure_runtime(args.runtime_dir)
    args.store = args.store or str(state / "memory.sqlite")
    # Configure the private CUDA compiler before importing Torch or SGLang.
    from .sglang_server import launch
    launch(args)


if __name__ == "__main__":
    main()
