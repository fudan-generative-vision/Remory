#!/usr/bin/env bash
set -euo pipefail
training_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
training_python=${TRAIN_PYTHON:-python}
export PYTHONPATH="$training_dir${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
command=${1:-help}
[[ $# == 0 ]] || shift
case "$command" in
  stage1) module=context_compression.train_dflash2_pretrain_3b ;;
  stage2) module=context_compression.train_qwen38_dflash2_stage2 ;;
  prepare) exec "$training_python" "$training_dir/prepare.py" "$@" ;;
  summaries) exec "$training_python" -m context_compression.prepare_qwen38_codex_summaries "$@" ;;
  allturn-data) exec "$training_python" -m context_compression.qwen38_allturn_data "$@" ;;
  long-history) exec "$training_python" -m context_compression.qwen38_long_history_data "$@" ;;
  repeat-data) exec "$training_python" -m context_compression.qwen38_long_repeat_data "$@" ;;
  select-summaries) exec "$training_python" -m context_compression.prepare_qwen38_summary_selection "$@" ;;
  curriculum) exec "$training_python" -m context_compression.qwen38_residual_curriculum "$@" ;;
  test) export CUDA_VISIBLE_DEVICES=''; exec "$training_python" -m pytest "$training_dir/tests" "$@" ;;
  help|-h|--help)
    echo 'Usage: bash training/run.sh {stage1|stage2|prepare|summaries|allturn-data|long-history|repeat-data|select-summaries|curriculum|test} [arguments]'
    exit 0 ;;
  *) echo "Unknown training command: $command" >&2; exit 2 ;;
esac
# Help imports the entry point on CPU, without starting distributed workers.
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec "$training_python" -m "$module" "$@"
fi
exec "$training_python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module "$module" "$@"
