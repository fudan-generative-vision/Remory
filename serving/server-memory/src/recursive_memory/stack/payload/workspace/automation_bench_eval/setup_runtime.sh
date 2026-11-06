#!/usr/bin/env bash
set -Eeuo pipefail
# Both interpreter and packages must be shared, not symlinked into node /root.
# Published generations are immutable. Never rewrite a runtime used by a worker.
automation_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
automation_root=${AUTOMATIONBENCH_RUNTIME_ROOT:-$automation_project/eval/automation_bench}
automation_revision=4a8e1061254004d9dac807054eed33fad7d1ff14
mkdir -p "$automation_root"
automation_root=$(cd -- "$automation_root" && pwd -P)
automation_source=$automation_root/source
automation_venv=$automation_root/runtimes/shared-cpython-3.13.7-v1/venv
automation_ready=$automation_root/runtimes/shared-cpython-3.13.7-v1/ready.json
automation_python_install=$automation_root/python-installations
exec 9>"$automation_root/setup.lock"
flock 9
# Do not run uv sync against a published venv: another worker may be using it.
if [[ -f "$automation_ready" ]]; then
  if env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u VIRTUAL_ENV \
    "$automation_venv/bin/python" -I "$automation_project/automation_bench_eval/runtime.py" \
    --runtime-root "$automation_root"; then
    exit 0
  fi
  echo "Published AutomationBench runtime cannot execute on this node; leaving it untouched." >&2
  echo "Check the shared mount and container compatibility. No task was started." >&2
  exit 1
fi
if [[ ! -x "$automation_root/bootstrap/bin/uv" ]]; then
  python3 -m venv "$automation_root/bootstrap"
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$automation_root/bootstrap/bin/pip" install 'uv==0.8.22'
fi
if [[ ! -d "$automation_source/.git" ]]; then
  git clone https://github.com/zapier/AutomationBench.git "$automation_source"
fi
if [[ -n "$(git -C "$automation_source" status --porcelain)" ]]; then
  echo "Refusing to alter a dirty AutomationBench source checkout: $automation_source" >&2
  exit 1
fi
if [[ "$(git -C "$automation_source" rev-parse HEAD)" != "$automation_revision" ]]; then
  git -C "$automation_source" checkout --detach "$automation_revision"
fi
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u VIRTUAL_ENV \
  UV_PYTHON_INSTALL_DIR="$automation_python_install" \
  "$automation_root/bootstrap/bin/uv" python install 3.13.7 --no-bin --no-config
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u PIP_CONSTRAINT -u UV_CONSTRAINT \
  UV_PYTHON_INSTALL_DIR="$automation_python_install" \
  UV_PROJECT_ENVIRONMENT="$automation_venv" \
  "$automation_root/bootstrap/bin/uv" sync --project "$automation_source" \
  --python 3.13.7 --managed-python --no-python-downloads --locked --no-dev --link-mode copy
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u VIRTUAL_ENV \
  "$automation_venv/bin/python" -I "$automation_project/automation_bench_eval/runtime.py" \
  --runtime-root "$automation_root" --publish
echo "AutomationBench official source: $automation_revision"
echo "Official runtime: $automation_venv/bin/python"
