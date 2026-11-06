"""Isolated launch entrypoint for the full DFlow/DFly worker.

Run this module instead of ``sglang.launch_server``.  SGLang uses spawn for
scheduler processes, so patching a worker only in the HTTP parent is not
inherited.  The picklable scheduler target below installs the subclass inside
each spawned TP rank before the ordinary scheduler constructs its worker.
No global SGLang routing file is modified.
"""

from __future__ import annotations

import os
import sys


def run_dflow_dfly_full_scheduler_process(*args, **kwargs):
    """Spawn target that installs only this experiment's DFLASH worker."""

    import sglang.srt.speculative.dflash_worker as dflash_worker_module
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.speculative.dflow_dfly_full_worker import (
        DFlowDFlyFullWorker,
    )

    dflash_worker_module.DFlashWorker = DFlowDFlyFullWorker
    return run_scheduler_process(*args, **kwargs)


def main(argv=None) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    argv = sys.argv[1:] if argv is None else list(argv)
    server_args = prepare_server_args(argv)
    if str(server_args.speculative_algorithm or "").upper() != "DFLASH":
        raise ValueError(
            "The full DFlow/DFly entrypoint requires --speculative-algorithm DFLASH."
        )
    if not server_args.speculative_draft_model_path:
        raise ValueError(
            "The full DFlow/DFly entrypoint requires --speculative-draft-model-path."
        )
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_dflow_dfly_full_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()


__all__ = ["main", "run_dflow_dfly_full_scheduler_process"]
