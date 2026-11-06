"""Spawn-safe, explicitly opted-in server for pure-DFlow OPD tracing."""

from __future__ import annotations

import os
import sys


def run_dflow_opd_trace_scheduler_process(*args, **kwargs):
    import sglang.srt.speculative.dflash_worker as dflash_worker_module
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.speculative.dflow_opd_trace_worker import DFlowOPDTraceWorker

    dflash_worker_module.DFlashWorker = DFlowOPDTraceWorker
    return run_scheduler_process(*args, **kwargs)


def _validate_server_args(server_args) -> None:
    if str(server_args.speculative_algorithm or "").upper() != "DFLASH":
        raise ValueError("The DFlow OPD trace server requires DFLASH.")
    if not server_args.speculative_draft_model_path:
        raise ValueError("The DFlow OPD trace server requires a draft checkpoint.")
    if (
        server_args.speculative_num_draft_tokens is not None
        and int(server_args.speculative_num_draft_tokens) != 16
    ):
        raise ValueError("The DFlow OPD trace server requires block size 16.")
    if int(server_args.dp_size) != 1:
        raise ValueError("The DFlow OPD trace server requires data parallel size 1.")
    if server_args.disable_overlap_schedule is not True:
        raise ValueError(
            "The DFlow OPD trace server requires --disable-overlap-schedule."
        )


def main(argv=None) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.speculative.dflow_opd_trace_worker import DFlowOPDTraceConfig
    from sglang.srt.utils import kill_process_tree

    argv = sys.argv[1:] if argv is None else list(argv)
    # Validate every explicit opt-in variable in the parent, but leave directory
    # creation to TP rank zero in the spawned scheduler process.
    DFlowOPDTraceConfig.from_environ()
    server_args = prepare_server_args(argv)
    _validate_server_args(server_args)
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_dflow_opd_trace_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()


__all__ = ["main", "run_dflow_opd_trace_scheduler_process"]
