"""Spawn-safe isolated server entrypoint for plain DFlow-Domino.

SGLang starts scheduler/TP workers with ``spawn``.  The scheduler target below
installs the dedicated worker inside every child process while leaving the
ordinary DFlash routing module and all Stage-I files unchanged on disk.
"""

from __future__ import annotations

import os
import sys


def run_dflow_domino_scheduler_process(*args, **kwargs):
    import sglang.srt.speculative.dflash_worker as dflash_worker_module
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.speculative.dflow_domino_worker import DFlowDominoWorker

    dflash_worker_module.DFlashWorker = DFlowDominoWorker
    return run_scheduler_process(*args, **kwargs)


def main(argv=None) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    argv = sys.argv[1:] if argv is None else list(argv)
    server_args = prepare_server_args(argv)
    if str(server_args.speculative_algorithm or "").upper() != "DFLASH":
        raise ValueError(
            "The plain DFlow-Domino entrypoint requires "
            "--speculative-algorithm DFLASH."
        )
    if not server_args.speculative_draft_model_path:
        raise ValueError(
            "The plain DFlow-Domino entrypoint requires "
            "--speculative-draft-model-path."
        )
    if (
        server_args.speculative_num_draft_tokens is not None
        and int(server_args.speculative_num_draft_tokens) != 16
    ):
        raise ValueError(
            "The plain DFlow-Domino entrypoint requires exactly 16 verify rows "
            "(root + 15 proposals)."
        )
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_dflow_domino_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()


__all__ = ["main", "run_dflow_domino_scheduler_process"]
