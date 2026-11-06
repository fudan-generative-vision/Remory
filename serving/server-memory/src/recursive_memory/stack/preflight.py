"""Harbor subprocess entry point; checks the frozen portable pool contract."""

import argparse
import os
from pathlib import Path

from .deploy import read
from .runtime import endpoint, external_serving_state, tb_preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = read(os.environ["R2EACT_STACK_STATE"])
    if os.environ.get("R2EACT_SHARED_SGLANG_URL"):
        state = external_serving_state(state, os.environ["R2EACT_SHARED_SGLANG_URL"])
    if args.sglang_url.rstrip("/") != endpoint(state):
        parser.error("this worker must use its task-local attested endpoint")
    tb_preflight(state, args.config, args.output,
                 memory=os.environ.get("R2EACT_TASK_MEMORY", "residual"),
                 serving_concurrency=int(os.environ.get("R2EACT_TASK_CONCURRENCY", "1")))


if __name__ == "__main__":
    main()
