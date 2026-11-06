#!/usr/bin/env bash
set -Eeuo pipefail
# Do not load the serving engine's GPU libtorch into this CPU-only runtime.
unset LD_LIBRARY_PATH PYTHONPATH
jobbench_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
jobbench_mcp_venv=${R2EACT_DOCUMENT_VENV:-$jobbench_project/eval/document_venv}
mkdir -p "$(dirname "$jobbench_mcp_venv")"
exec 9>"$jobbench_mcp_venv.setup.lock"
flock 9
jobbench_constraints="$jobbench_project/../constraints/documents.txt"
jobbench_constraint_args=()
if [[ -f "$jobbench_constraints" ]]; then
  jobbench_constraint_args=(-c "$jobbench_constraints")
fi
if [[ ! -f "$jobbench_mcp_venv/.r2eact-runtime-v1" ]]; then
  python3 -m venv "$jobbench_mcp_venv"
  # The cluster's global pip constraint pins the serving GPU torch build.
  # Remove it ONLY for this isolated, CPU-only document-tools environment.
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$jobbench_mcp_venv/bin/pip" install \
    "${jobbench_constraint_args[@]}" \
    'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$jobbench_mcp_venv/bin/pip" install \
    "${jobbench_constraint_args[@]}" \
    'docling-mcp[local]==2.2.1' 'mineru-open-mcp==1.0.21' \
    'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' \
    pandas openpyxl python-pptx pyarrow pdfplumber mammoth python-docx matplotlib openai
  "$jobbench_mcp_venv/bin/pip" check
fi
if ! "$jobbench_mcp_venv/bin/python" -c 'import openai' >/dev/null 2>&1; then
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$jobbench_mcp_venv/bin/pip" install "${jobbench_constraint_args[@]}" openai
fi
if ! "$jobbench_mcp_venv/bin/python" -c 'import cv2, onnxruntime' >/dev/null 2>&1; then
  # Servers have no desktop/XCB stack; install the CPU/headless implementation
  # after Docling's transitive GUI OpenCV dependency.
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$jobbench_mcp_venv/bin/pip" install \
    "${jobbench_constraint_args[@]}" \
    'opencv-python-headless==5.0.0.93' onnxruntime
fi
"$jobbench_mcp_venv/bin/python" -c 'import torch; assert not torch.cuda.is_available(); import docling_mcp, mineru_open_mcp'
touch "$jobbench_mcp_venv/.r2eact-runtime-v1"
command -v jq >/dev/null || echo 'Note: install jq before running the optional official judge.' >&2
