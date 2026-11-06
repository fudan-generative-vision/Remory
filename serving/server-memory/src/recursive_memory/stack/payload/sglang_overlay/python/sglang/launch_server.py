"""Launch the inference server."""

import asyncio
import os
import sys
import uuid
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

_SPECFORGE_MEMORY_SERVER_EPOCH_ENV = "SGLANG_SPECFORGE_MEMORY_SERVER_EPOCH"


def _rotate_specforge_memory_server_epoch() -> str:
    """Mint a fresh epoch for every top-level server launch/restart."""

    epoch = uuid.uuid4().hex
    os.environ[_SPECFORGE_MEMORY_SERVER_EPOCH_ENV] = epoch
    return epoch


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    # Spawned HTTP/scheduler/model processes must share one restart-scoped
    # epoch before any TP-local GPU memory handle can be minted.  The feature
    # remains disabled unless an explicitly configured compressor hook
    # advertises its registry capability.
    _rotate_specforge_memory_server_epoch()
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
