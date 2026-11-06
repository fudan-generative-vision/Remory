"""Spawn-safe isolated server entrypoint for optional Stage-J Jacobi HSC.

The ordinary and sequential Stage-J DFLASH routing remains untouched.  This
entrypoint validates the fixed-round and content-addressed efficacy gate in the
HTTP parent, then installs the Jacobi worker independently in each spawned TP
scheduler process.
"""

from __future__ import annotations

import os
import sys


def run_dflow_dfly_full_jacobi_scheduler_process(*args, **kwargs):
    import sglang.srt.speculative.dflash_worker as dflash_worker_module
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.speculative.dflow_dfly_full_jacobi_worker import (
        DFlowDFlyFullJacobiWorker,
    )

    dflash_worker_module.DFlashWorker = DFlowDFlyFullJacobiWorker
    return run_scheduler_process(*args, **kwargs)


def main(argv=None) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.speculative.dflow_dfly_full_jacobi_worker import (
        _JACOBI_ROUNDS_ENV,
        parse_jacobi_rounds,
        validate_jacobi_gate_from_env,
    )
    from sglang.srt.utils import kill_process_tree

    argv = sys.argv[1:] if argv is None else list(argv)
    server_args = prepare_server_args(argv)
    if str(server_args.speculative_algorithm or "").upper() != "DFLASH":
        raise ValueError(
            "The full DFlow/DFly Jacobi entrypoint requires "
            "--speculative-algorithm DFLASH."
        )
    if not server_args.speculative_draft_model_path:
        raise ValueError(
            "The full DFlow/DFly Jacobi entrypoint requires "
            "--speculative-draft-model-path."
        )
    if int(server_args.tp_size) != 2:
        raise ValueError(
            "The full DFlow/DFly Jacobi entrypoint is restricted to TP=2, got "
            f"TP={int(server_args.tp_size)}."
        )
    parse_jacobi_rounds(os.environ.get(_JACOBI_ROUNDS_ENV, ""))
    validate_jacobi_gate_from_env(
        draft_model_path=server_args.speculative_draft_model_path
    )
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=(
                run_dflow_dfly_full_jacobi_scheduler_process
            ),
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()


__all__ = ["main", "run_dflow_dfly_full_jacobi_scheduler_process"]
